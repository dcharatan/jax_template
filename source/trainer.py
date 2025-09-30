import logging
import shutil
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

import equinox as eqx
import jax
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from grain.python import (
    PyGrainCheckpointRestore,
    PyGrainCheckpointSave,
    PyGrainDatasetIterator,
)
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from optax import GradientTransformation

from .sharding import (
    filter_add_sharding_to_shape_dtype_struct,
    filter_device_put,
    get_distributed_sharding,
    get_replicated_sharding,
)
from .timer import Timer

OptState = PyTree


@dataclass(frozen=True)
class CheckpointingCfg:
    save_interval_steps: int
    max_to_keep: int
    load_dataset_state: bool


@dataclass(frozen=True)
class TrainerCfg:
    num_steps: int
    log_every: int
    checkpointing: CheckpointingCfg
    on_existing_workspace: Literal["restore", "overwrite", "throw"]


B = TypeVar("B")  # batch type


class Trainable(eqx.Module, Generic[B], ABC):
    @abstractmethod
    def train_step(self, batch: B, rng: PRNGKeyArray) -> Float[Array, ""]:
        pass

    @abstractmethod
    def configure_optimizer(self) -> GradientTransformation:
        pass


def compute_loss(
    trainable: Trainable[B],
    batch: B,
    rng: PRNGKeyArray,
) -> Float[Array, ""]:
    return trainable.train_step(batch, rng)


@eqx.filter_jit(donate="all")
def train_step(
    trainable: Trainable[B],
    opt_state: OptState,
    batch: B,
    rng: PRNGKeyArray,
    opt: optax.GradientTransformation,
) -> tuple[Trainable[B], OptState, Float[Array, ""]]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(trainable, batch, rng)
    updates, opt_state = opt.update(grads, opt_state)
    trainable = eqx.apply_updates(trainable, updates)
    return trainable, opt_state, loss


class Trainer(Generic[B]):
    def __init__(
        self,
        cfg: TrainerCfg,
        workspace: Path,
    ) -> None:
        self.cfg = cfg
        self.workspace = workspace

        # Set up the checkpoint manager.
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=cfg.checkpointing.save_interval_steps,
            max_to_keep=cfg.checkpointing.max_to_keep,
        )
        checkpoint_dir = workspace / "checkpoints"
        self.manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
            item_names=("parameters", "opt_state", "dataset"),
        )
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def train(
        self,
        rng: PRNGKeyArray,
        get_trainable: Callable[[], Trainable[B]],
        dataset: PyGrainDatasetIterator,
    ) -> None:
        logging.info("Entering training loop.")
        step, trainable, opt, opt_state = self.init_or_handle_existing_workspace(
            get_trainable,
            dataset,
        )

        # Set up a handler for preemption.
        preempted = False

        def preemption_handler(sig, frame):
            nonlocal preempted
            logging.info("Attempting to exit gracefully.")
            preempted = True

        def exit_if_preempted():
            if preempted:
                # Save a checkpoint at the current step, then exit gracefully.
                self.save(step, trainable, opt_state, dataset, force=True)
                self.manager.wait_until_finished()
                logging.info("Exiting gracefully.")
                sys.exit(0)

        signal.signal(signal.SIGINT, preemption_handler)
        signal.signal(signal.SIGTERM, preemption_handler)

        # Get the sharding that will be used for the trainable and opt_state, then use
        # it to replicate the trainable and opt_state across all devices.
        replicated = get_replicated_sharding((trainable, opt_state))
        trainable, opt_state = filter_device_put((trainable, opt_state), replicated)

        # Main training loop. Exit before long-running operations if preempted.
        writer = tf.summary.create_file_writer(str(self.workspace / "tensorboard"))
        timer = Timer()
        while step < self.cfg.num_steps:
            step_rng = jax.random.fold_in(rng, step)

            exit_if_preempted()
            self.save(step, trainable, opt_state, dataset)

            exit_if_preempted()
            batch = self.get_batch(dataset)

            exit_if_preempted()
            trainable, opt_state, loss = train_step(
                trainable,
                opt_state,
                batch,
                step_rng,
                opt,
            )

            # Log progress.
            if step % self.cfg.log_every == 0:
                rate = timer.get_rate(self.cfg.log_every)
                logging.info(f"Step {step}; loss {loss.item():.5f}; {rate}")

            # Write the loss, but only from one process.
            exit_if_preempted()
            if jax.process_index() == 0:
                with writer.as_default():
                    tf.summary.scalar("loss", loss.item(), step=step)

            step += 1

        # Ensure that the checkpoint at step == num_steps isn't dropped.
        self.save(step, trainable, opt_state, dataset)

        # Wait for checkpoints to finish saving before returning.
        self.manager.wait_until_finished()

    def init(
        self,
        get_trainable: Callable[[], Trainable[B]],
    ) -> tuple[Trainable[B], optax.GradientTransformation, OptState]:
        trainable = get_trainable()
        opt = trainable.configure_optimizer()
        opt_state = opt.init(eqx.filter(trainable, eqx.is_inexact_array))
        return trainable, opt, opt_state

    def save(
        self,
        step: int,
        trainable: Trainable[B],
        opt_state: OptState,
        dataset: PyGrainDatasetIterator,
        force: bool = False,
    ) -> None:
        # Avoid unnecessary work.
        if not (self.manager.should_save(step) or force):
            return
        logging.info(f"Saving checkpoint at step {step}.")

        # Only save the trainable's parameters (not configuration dataclasses).
        parameters, _ = eqx.partition(trainable, eqx.is_array)

        args = ocp.args.Composite(
            parameters=ocp.args.StandardSave(parameters),
            opt_state=ocp.args.StandardSave(opt_state),
            dataset=PyGrainCheckpointSave(dataset),
        )
        self.manager.save(step, args=args, force=force)

    def restore(
        self,
        step: int | None,
        get_trainable: Callable[[], Trainable[B]],
        dataset: PyGrainDatasetIterator,
    ) -> tuple[int, Trainable[B], optax.GradientTransformation, OptState]:
        if step is None:
            step = self.manager.latest_step()

        # Get the PyTree specifications for the parameters and optimizer states that
        # need to be loaded.
        trainable, _, opt_state = eqx.filter_eval_shape(self.init, get_trainable)
        parameters, static = eqx.partition(
            trainable, lambda x: isinstance(x, jax.ShapeDtypeStruct)
        )

        # Give the parameters and opt_state (which are ShapeDtypeStruct) the correct
        # sharding.
        parameters = filter_add_sharding_to_shape_dtype_struct(
            parameters, get_replicated_sharding(parameters)
        )
        opt_state = filter_add_sharding_to_shape_dtype_struct(
            opt_state, get_replicated_sharding(opt_state)
        )

        # Use Orbax to load the parameters and optimizer state. The dataset state can be
        # loaded too, but is optionally skipped because any particular dataset state is
        # only compatible with a specific number of training processes.
        args = dict(
            parameters=ocp.args.StandardRestore(parameters),
            opt_state=ocp.args.StandardRestore(opt_state),
        )
        if self.cfg.checkpointing.load_dataset_state:
            args["dataset"] = PyGrainCheckpointRestore(dataset)
        restored = self.manager.restore(step, args=ocp.args.Composite(**args))

        # Combine the loaded parameters with the static parts of the Trainable.
        trainable: Trainable[B] = eqx.combine(restored["parameters"], static)

        return step, trainable, trainable.configure_optimizer(), restored["opt_state"]

    def init_or_handle_existing_workspace(
        self,
        get_trainable: Callable[[], Trainable[B]],
        dataset: PyGrainDatasetIterator,
    ) -> tuple[int, Trainable[B], optax.GradientTransformation, OptState]:
        """Based on self.cfg.on_existing_workspace, handle initialization."""
        if self.cfg.on_existing_workspace == "throw" and self.workspace.exists():
            # In "throw" mode, throw an exception if the workspace already exists.
            raise Exception("Workspace already exists!")
        elif self.cfg.on_existing_workspace == "overwrite":
            # In "overwrite" mode, delete the workspace if it already exists.
            shutil.rmtree(self.workspace, True)
        elif self.cfg.on_existing_workspace == "restore":
            # In "restore" mode, attempt to load the workspace's latest checkpoint if it
            # already exists. Otherwise, initialize as usual.
            try:
                return self.restore(None, get_trainable, dataset)
            except FileNotFoundError:
                # This will fall through to regular initialization.
                pass

        # Avoid race conditions (e.g., creating the workspace in one process and then
        # throwing because it exists from another process).
        jax.experimental.multihost_utils.sync_global_devices("init")

        return 0, *self.init(get_trainable)

    def get_batch(self, dataset: PyGrainDatasetIterator) -> B:
        """Get the local (per-process) chunk of the batch from the iterator, then
        combine it with the other processes' chunks to form a sharded global array.
        """
        batch = next(dataset)
        return jax.tree.map(
            lambda x, sharding: jax.make_array_from_process_local_data(sharding, x),
            batch,
            get_distributed_sharding(batch),
        )
