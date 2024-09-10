import jax
from jaxtyping import install_import_hook

with install_import_hook("video_learning", "beartype.beartype"):
    from .config import get_typed_config
    from .dataset import get_data_loader
    from .trainable import get_trainable
    from .trainer import train


def main() -> None:
    # Read the configuration.
    cfg = get_typed_config()

    # Set up the model and data loader.
    trainable = get_trainable(cfg.trainable)
    loader = get_data_loader(cfg.dataset, cfg.data_loader)

    train(
        jax.random.key(cfg.seed),
        trainable,
        loader,
    )


if __name__ == "__main__":
    main()
