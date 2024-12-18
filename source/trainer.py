import shutil
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

import equinox as eqx
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from grain.python import PyGrainDatasetIterator
from jaxtyping import Array, Float, PRNGKeyArray
from optax import GradientTransformation
from tqdm import trange

from .sharding import (
    filter_add_sharding_to_shape_dtype_struct,
    filter_device_put,
    get_distributed_sharding,
    get_replicated_sharding,
    per_process_all_gather_bytes,
)


@dataclass(frozen=True)
class CheckpointingCfg:
    save_interval_steps: int
    max_to_keep: int


@dataclass(frozen=True)
class TrainerCfg:
    num_steps: int
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
    opt_state: optax.OptState,
    batch: B,
    rng: PRNGKeyArray,
    opt: optax.GradientTransformation,
) -> tuple[Trainable[B], optax.OptState, Float[Array, ""]]:
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
        self.writer = tf.summary.create_file_writer(str(workspace / "tensorboard"))

        # Set up the checkpoint manager.
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=cfg.checkpointing.save_interval_steps,
            max_to_keep=cfg.checkpointing.max_to_keep,
        )
        self.manager = ocp.CheckpointManager(
            workspace / "checkpoints",
            options=options,
            item_names=("parameters", "opt_state", "dataset"),
        )

    def train(
        self,
        rng: PRNGKeyArray,
        get_trainable: Callable[[], Trainable[B]],
        dataset: PyGrainDatasetIterator,
    ) -> None:
        print("Entering training loop.")
        step, trainable, opt, opt_state = self.init_or_handle_existing_workspace(
            get_trainable,
            dataset,
        )

        # Set up a handler for preemption.
        preempted = False

        def preemption_handler(sig, frame):
            nonlocal preempted
            print("Attempting to exit gracefully.")
            preempted = True

        def exit_if_preempted():
            if preempted:
                # Save a checkpoint at the current step, then exit gracefully.
                self.save(step, trainable, opt_state, dataset, force=True)
                self.manager.wait_until_finished()
                print("Exiting gracefully.")
                sys.exit(0)

        signal.signal(signal.SIGINT, preemption_handler)
        signal.signal(signal.SIGTERM, preemption_handler)

        # Get the sharding that will be used for the trainable and opt_state, then use
        # it to replicate the trainable and opt_state across all devices.
        replicated = get_replicated_sharding((trainable, opt_state))
        trainable, opt_state = filter_device_put((trainable, opt_state), replicated)

        # Main training loop. Exit before long-running operations if preempted.
        for step in trange(
            step,
            self.cfg.num_steps,
            initial=step,
            total=self.cfg.num_steps,
            desc="Training",
        ):
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

            # Write the loss, but only from one process.
            exit_if_preempted()
            if jax.process_index() == 0:
                with self.writer.as_default():
                    tf.summary.scalar("loss", loss.item(), step=step)

        # Ensure that the checkpoint at step == num_steps isn't dropped.
        self.save(step + 1, trainable, opt_state, dataset)

        # Wait for checkpoints to finish saving before returning.
        self.manager.wait_until_finished()

    def init(
        self,
        get_trainable: Callable[[], Trainable[B]],
    ) -> tuple[Trainable[B], optax.GradientTransformation, optax.OptState]:
        trainable = get_trainable()
        opt = trainable.configure_optimizer()
        opt_state = opt.init(eqx.filter(trainable, eqx.is_inexact_array))
        return trainable, opt, opt_state

    def save(
        self,
        step: int,
        trainable: Trainable[B],
        opt_state: optax.OptState,
        dataset: PyGrainDatasetIterator,
        force: bool = False,
    ) -> None:
        # Avoid unnecessary work.
        if not (self.manager.should_save(step) or force):
            return
        print(f"Saving checkpoint at step {step}.")

        # Only save the trainable's parameters (not configuration dataclasses).
        parameters, _ = eqx.partition(trainable, eqx.is_array)

        # Checkpointing the state provided by PyGrainDatasetIterator.get_state is a bit
        # annoying, since it's provided as a per-process Python bytes object. We need to
        # all-gather the bytes across processes in order to add them to the checkpoint.
        dataset_state = per_process_all_gather_bytes(dataset.get_state())
        dataset_state = [np.frombuffer(x, dtype=np.uint8) for x in dataset_state]

        args = ocp.args.Composite(
            parameters=ocp.args.StandardSave(parameters),
            opt_state=ocp.args.StandardSave(opt_state),
            dataset=ocp.args.StandardSave(dataset_state),
        )
        self.manager.save(step, args=args, force=force)

    def restore(
        self,
        step: int | None,
        get_trainable: Callable[[], Trainable[B]],
        dataset: PyGrainDatasetIterator,
    ) -> tuple[int, Trainable[B], optax.GradientTransformation, optax.OptState]:
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

        # Use Orbax to load the parameters, optimizer state, and dataset state.
        args = ocp.args.Composite(
            parameters=ocp.args.StandardRestore(parameters),
            opt_state=ocp.args.StandardRestore(opt_state),
            dataset=ocp.args.StandardRestore(),
        )
        restored = self.manager.restore(step, args=args)

        # Combine the loaded parameters with the static parts of the Trainable.
        trainable: Trainable[B] = eqx.combine(restored["parameters"], static)

        # Restore the dataset iterator's state, making sure to grab the state from the
        # correct process.
        if len(restored["dataset"]) == jax.process_count():
            try:
                dataset.set_state(bytes(restored["dataset"][jax.process_index()]))
            except ValueError:
                print(
                    "Could not restore data loader state. This is probably because the "
                    "number of workers has changed."
                )
        else:
            print(
                "Cannot restore data loader state because the number of Jax processes "
                "used to create the checkpoint differs from the current number of Jax "
                "processes."
            )

        return step, trainable, trainable.configure_optimizer(), restored["opt_state"]

    def init_or_handle_existing_workspace(
        self,
        get_trainable: Callable[[], Trainable[B]],
        dataset: PyGrainDatasetIterator,
    ) -> tuple[int, Trainable[B], optax.GradientTransformation, optax.OptState]:
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
