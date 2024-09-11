import os
from pathlib import Path

import jax
from jaxtyping import install_import_hook

with install_import_hook("video_learning", "beartype.beartype"):
    from .config import get_typed_config
    from .dataset import get_dataset_iterator
    from .trainable import get_trainable_type
    from .trainer import Trainer


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
    trainer.train(
        jax.random.key(cfg.seed),
        lambda: get_trainable_type(cfg.trainable)(cfg.trainable),
        get_dataset_iterator(cfg.dataset, cfg.data_loader),
    )


if __name__ == "__main__":
    main()
