# Jax Template

This is a template for training neural networks with Jax.

## Running the Code

See `.vscode/launch.json` for examples.

## Dependencies

- **Neural Networks:** [Equinox](https://docs.kidger.site/equinox/), because it's simpler than Flax.
- **Data Loading:** [Grain](https://github.com/google/grain/tree/main/docs), because it handles multi-process sharding 
- **Logging:** [Tensorboard](https://github.com/tensorflow/tensorboard), because W&B is expensive and slow to load.
- **Typing:** [Jaxtyping](https://docs.kidger.site/jaxtyping/), because everyone should use this everywhere.
- **Optimization:** [Optax](https://optax.readthedocs.io/en/latest/), because it's the de-facto standard for Jax.
- **Checkpointing:** [Orbax](https://orbax.readthedocs.io/en/latest/), because it's also the default.
- **Configuration:** [Hydra](https://hydra.cc/docs/intro/), because it's fairly simple and the command-line overrides are nice. Configurations are converted into typed dataclasses using [Dacite](https://github.com/konradhalas/dacite).

## Features

- **Multi-Device and Multi-Process Training:** This code supports multiple devices per process and multiple processes. On GPUs, NCCL seems to hang if there are multiple processes which each control more than one GPU, so one process per GPU is recommended for multi-node training.
- **SPMD via Sharding:** Instead of using `jax.pmap` for SPMD, this code uses sharding constraints.
- **Preemption Tolerance:** When preempted via SIGINT or SIGTERM, this code will save a checkpoint. When resumed/requeued, it will load this checkpoint and continue training.

## Code Structure

- The main entry point for training is `source/main.py`.
- The training loop is in `source/trainer.py`.
- The main configuration is in `config/main.yaml`.
- The model is in `source/trainable/trainable_mnist.py`. It is a `Trainable`, which is a model which specifies its loss function and optimizer (similar to a `LightningModule` in PyTorch Lightning).
- Both models (`Trainable`s) and datasets are designed to be easily swapped. See `source/trainable/__init__.py` and `source/dataset/__init__.py` for more details.
