from dataclasses import dataclass

import jax
import numpy as np
from grain.python import Batch as BatchOperation
from grain.python import (
    DataLoader,
    IndexSampler,
    PyGrainDatasetIterator,
    ShardOptions,
)

from ..sharding import get_distributed_sharding
from .dataset_mnist import DatasetMnist, DatasetMnistCfg
from .interface import Batch

DatasetCfg = DatasetMnistCfg


@dataclass
class DataLoaderCfg:
    per_device_batch_size: int
    worker_count: int
    worker_buffer_size: int
    num_epochs: int | None


def get_dataset_iterator(
    dataset_cfg: DatasetCfg,
    data_loader_cfg: DataLoaderCfg,
) -> PyGrainDatasetIterator:
    dataset = {"mnist": DatasetMnist}[dataset_cfg.name](dataset_cfg)
    data_source = dataset.get_data_source()

    # Combine the dataset's transformations with two batching operations: one for the
    # on-device batch dimension and one for the per-device batch dimension.
    operations = [
        *dataset.get_transformations(),
        BatchOperation(
            data_loader_cfg.per_device_batch_size * jax.local_device_count(),
            True,
        ),
    ]

    loader = DataLoader(
        data_source=data_source,
        worker_count=data_loader_cfg.worker_count,
        worker_buffer_size=data_loader_cfg.worker_buffer_size,
        operations=operations,
        sampler=IndexSampler(
            num_records=len(data_source),
            num_epochs=data_loader_cfg.num_epochs,
            shard_options=ShardOptions(
                shard_index=jax.process_index(),
                shard_count=jax.process_count(),
                drop_remainder=True,
            ),
            shuffle=False,
            seed=0,
        ),
    )
    return iter(loader)


def get_typed_sharded_batch(dataset: PyGrainDatasetIterator) -> Batch:
    """Get a typed batch from the data loader that has been sharded across processes."""
    per_process_batch = next(dataset)
    global_batch = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            get_distributed_sharding(x), x
        ),
        per_process_batch,
    )
    return Batch(**global_batch)
