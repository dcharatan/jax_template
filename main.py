import os
from pathlib import Path

import jax

if __name__ == "__main__":
    # On Slurm, the environment variables shouldn't be necessary.
    print(f"PID: {os.getpid()}")

    def convert(x):
        return None if x is None else int(x)

    if os.environ.get("JAX_DIST_ENABLED", False):
        jax.distributed.initialize(
            coordinator_address=os.environ.get("JAX_DIST_COORDINATOR_ADDRESS", None),
            num_processes=convert(os.environ.get("JAX_DIST_NUM_PROCESSES", None)),
            process_id=convert(os.environ.get("JAX_DIST_PROCESS_ID", None)),
        )

    print(
        f"Process index {jax.process_index()} controls {jax.local_device_count()} "
        f"device(s) ({jax.local_devices()}). Total: {jax.process_count()} process(es) "
        f"and {jax.device_count()} device(s)"
    )

from jaxtyping import install_import_hook

with install_import_hook("source", "beartype.beartype"):
    from source.config import get_typed_config
    from source.dataset import get_dataset_iterator
    from source.trainable import get_trainable_type
    from source.trainer import Trainer


def main() -> None:
    # Read the configuration.
    cfg = get_typed_config()

    # Read the workspace directory.
    if os.environ.get("WORKSPACE", None) is None:
        raise ValueError("You must specify the WORKSPACE environment variable.")
    workspace = Path(os.environ["WORKSPACE"])

    trainer = Trainer(
        cfg.trainer,
        workspace,
    )
    rng = jax.random.key(cfg.seed)
    trainer_rng, init_rng = jax.random.split(rng)
    trainer.train(
        trainer_rng,
        lambda: get_trainable_type(cfg.trainable)(cfg.trainable, init_rng),
        get_dataset_iterator(cfg.dataset, cfg.data_loader),
    )


if __name__ == "__main__":
    main()
