from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import equinox as eqx
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from grain.python import PyGrainDatasetIterator
from jaxtyping import PRNGKeyArray
from optax import GradientTransformation
from tqdm import trange

from .dataset import get_typed_sharded_batch
from .sharding import replicate


@dataclass
class CheckpointingCfg:
    save_interval_steps: int
    max_to_keep: int


@dataclass
class TrainerCfg:
    num_steps: int
    checkpointing: CheckpointingCfg


D = TypeVar("D")  # data type


class Trainable(eqx.Module, Generic[D], ABC):
    @abstractmethod
    def train_step(self, batch: D, rng: PRNGKeyArray) -> None:
        pass

    @abstractmethod
    def configure_optimizer(self) -> GradientTransformation:
        pass


class Trainer(Generic[D]):
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
        self.manager = ocp.CheckpointManager(
            workspace / "checkpoints",
            options=options,
            item_names=("parameters", "opt_state", "dataset"),
        )

    def train(
        self,
        rng: PRNGKeyArray,
        get_trainable: Callable[[], Trainable[D]],
        dataset: PyGrainDatasetIterator,
    ) -> None:
        # Attempt to restore the state after preemption.
        try:
            step, trainable, opt, opt_state = self.restore(None, get_trainable, dataset)
        except FileNotFoundError:
            step = 0
            trainable, opt, opt_state = self.init(get_trainable)

        # Replicate the parameters and optimizer states on all devices.
        trainable, opt_state = replicate((trainable, opt_state), mode="put")

        # Main training loop.
        for step in trange(
            step,
            self.cfg.num_steps,
            initial=step,
            total=self.cfg.num_steps,
            desc="Training",
        ):
            # Save a checkpoint.
            self.save(step, trainable, opt_state, dataset)

            batch = get_typed_sharded_batch(dataset)
            a = 1

        # Ensure that the checkpoint at step == num_steps isn't dropped.
        self.save(step + 1, trainable, opt_state, dataset)

        # Wait for checkpoints to finish saving before returning.
        self.manager.wait_until_finished()

    def init(
        self,
        get_trainable: Callable[[], Trainable[D]],
    ) -> tuple[Trainable[D], optax.GradientTransformation, optax.OptState]:
        trainable = get_trainable()
        opt = trainable.configure_optimizer()
        opt_state = opt.init(eqx.filter(trainable, eqx.is_array))
        return trainable, opt, opt_state

    def save(
        self,
        step: int,
        trainable: Trainable[D],
        opt_state: optax.OptState,
        dataset: PyGrainDatasetIterator,
    ) -> None:
        # Avoid unnecessary work.
        if not self.manager.should_save(step):
            return
        print(f"Saving checkpoint at step {step}.")

        # Only save the trainable's parameters (not configuration dataclasses).
        parameters, _ = eqx.partition(trainable, eqx.is_array)

        # Convert the bytes object provided by PyGrainDatasetIterator.get_state to a
        # NumPy array for compatibility with Orbax.
        dataset_state = np.frombuffer(dataset.get_state(), dtype=np.uint8)

        args = ocp.args.Composite(
            parameters=ocp.args.StandardSave(parameters),
            opt_state=ocp.args.StandardSave(opt_state),
            dataset=ocp.args.ArraySave(dataset_state),
        )
        self.manager.save(step, args=args)

    def restore(
        self,
        step: int | None,
        get_trainable: Callable[[], Trainable[D]],
        dataset: PyGrainDatasetIterator,
    ) -> tuple[int, Trainable[D], optax.GradientTransformation, optax.OptState]:
        if step is None:
            step = self.manager.latest_step()

        # Get the PyTree specifications for the parameters and optimizer states that
        # need to be loaded.
        trainable, _, opt_state = eqx.filter_eval_shape(self.init, get_trainable)
        parameters, static = eqx.partition(
            trainable, lambda x: isinstance(x, jax.ShapeDtypeStruct)
        )

        # Use Orbax to load the parameters, optimizer state, and dataset state.
        args = ocp.args.Composite(
            parameters=ocp.args.StandardRestore(parameters),
            opt_state=ocp.args.StandardRestore(opt_state),
            dataset=ocp.args.ArrayRestore(),
        )
        restored = self.manager.restore(step, args=args)

        # Combine the loaded parameters with the static parts of the Trainable.
        trainable: Trainable[D] = eqx.combine(restored["parameters"], static)

        # Restore the dataset iterator's state.
        dataset.set_state(bytes(restored["dataset"]))

        return step, trainable, trainable.configure_optimizer(), restored["opt_state"]
