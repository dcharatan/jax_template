from typing import Callable, Literal

import equinox as eqx
import jax
import numpy as np
from jaxtyping import PyTree

# Since we're only doing simple SPMD/data parallelism, we use this mesh everywhere.
mesh = jax.sharding.Mesh(jax.devices(), ["batch"])


def filter_shard_with_fn(
    x: PyTree,
    sharding_fn: Callable[[jax.Array], jax.sharding.Sharding],
    mode: Literal["put", "constraint"],
) -> PyTree:
    dynamic, static = eqx.partition(x, eqx.is_array)
    fn = jax.device_put if mode == "put" else jax.lax.with_sharding_constraint
    dynamic = jax.tree.map(lambda x: fn(x, sharding_fn(x)), dynamic)
    return eqx.combine(dynamic, static)


def get_distributed_sharding(
    array: jax.Array | np.ndarray,
) -> jax.sharding.NamedSharding:
    """Create a sharding that shards along the first axis and replicates along all
    other axes.
    """
    unsharded_axes = (None,) * (array.ndim - 1)
    partition_spec = jax.sharding.PartitionSpec(*("batch", *unsharded_axes))
    return jax.sharding.NamedSharding(mesh, partition_spec)


def distribute(
    x: PyTree,
    mode: Literal["put", "constraint"] = "constraint",
) -> PyTree:
    """Shard each PyTree node along its first axis and replicate it along all other
    axes.
    """
    return filter_shard_with_fn(x, get_distributed_sharding, mode)


def get_replicated_sharding(
    array: jax.Array | np.ndarray,
) -> jax.sharding.NamedSharding:
    """Create a sharding that replicates along all axes."""
    unsharded_axes = (None,) * array.ndim
    partition_spec = jax.sharding.PartitionSpec(*unsharded_axes)
    return jax.sharding.NamedSharding(mesh, partition_spec)


def replicate(
    x: PyTree,
    mode: Literal["put", "constraint"] = "constraint",
) -> PyTree:
    """Replicate each PyTree node along all of its axes."""
    return filter_shard_with_fn(x, get_replicated_sharding, mode)
