from typing import Any

import equinox as eqx
import jax
from jaxtyping import PyTree

# Since we're only doing simple SPMD/data parallelism, we use this mesh everywhere.
MESH = jax.sharding.Mesh(jax.devices(), ["batch"])

# Replicate on all axes.
SHARDING_REPLICATED = jax.sharding.NamedSharding(
    MESH,
    jax.sharding.PartitionSpec(),
)

# Shard along the first axis.
SHARDING_DISTRIBUTED = jax.sharding.NamedSharding(
    MESH,
    jax.sharding.PartitionSpec("batch"),
)


def filter_device_put(
    x: PyTree,
    sharding: PyTree[jax.sharding.Sharding | None],
) -> PyTree:
    dynamic, static = eqx.partition(x, eqx.is_array)
    dynamic = jax.device_put(dynamic, sharding)
    return eqx.combine(dynamic, static)


def filter_add_sharding_to_shape_dtype_struct(
    tree: PyTree[jax.ShapeDtypeStruct | Any],
    shardings: PyTree[jax.sharding.Sharding | None],
) -> PyTree:
    """Add shardings to the ShapeDtypeStruct leaves of a PyTree."""
    structs, others = eqx.partition(
        tree, lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct)
    )
    structs = jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=sharding, weak_type=leaf.weak_type
        ),
        structs,
        shardings,
    )
    return eqx.combine(structs, others)


def is_array_or_tracer(x: Any) -> bool:
    return eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)


def get_distributed_sharding(tree: PyTree) -> PyTree[jax.sharding.NamedSharding | None]:
    """Shard along the first axis and replicate along all other axes."""
    return jax.tree.map(
        lambda x: SHARDING_DISTRIBUTED if is_array_or_tracer(x) else None,
        tree,
    )


def get_replicated_sharding(tree: PyTree) -> PyTree[jax.sharding.NamedSharding | None]:
    """Replicate along all axes."""
    return jax.tree.map(
        lambda x: SHARDING_REPLICATED if is_array_or_tracer(x) else None,
        tree,
    )
