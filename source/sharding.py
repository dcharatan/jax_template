from typing import Any

import equinox as eqx
import jax
import numpy as np
from einops import repeat
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


def per_process_all_gather_bytes(x: bytes) -> list[bytes]:
    """A hacky way to gather bytes across processes using Jax."""
    x = np.frombuffer(x, dtype=np.uint8)

    # Find the maximum length of x across devices. Note that we repeat the length for
    # jax.local_device_count() times so that the total number of lengths (with
    # repetitions) is divisible by the total number of devices.
    length = repeat(np.array(len(x)), "-> b", b=jax.local_device_count())
    lengths = jax.make_array_from_process_local_data(
        get_distributed_sharding(length), length
    )
    max_length = lengths.max().item()

    # Pad the local value of x to the maximum length among all values of x.
    x_padded = np.zeros(max_length, dtype=np.uint8)
    x_padded[: len(x)] = x

    # All-gather x values across devices.
    x_padded = repeat(x_padded, "l -> b l", b=jax.local_device_count())
    xs_padded = jax.make_array_from_process_local_data(
        get_distributed_sharding(x_padded), x_padded
    )
    xs_padded = jax.device_put(xs_padded, get_replicated_sharding(xs_padded))

    # Transfer the padded x values and their lengths to host.
    lengths = jax.device_put(lengths, get_replicated_sharding(lengths))
    lengths = jax.device_get(lengths)
    xs_padded = jax.device_get(xs_padded)

    # Discard the per-device replicas.
    skip = jax.local_device_count()
    lengths = [lengths[i] for i in range(0, len(lengths), skip)]
    xs_padded = [xs_padded[i] for i in range(0, len(xs_padded), skip)]

    # Trim the bytes to the correct length.
    return [bytes(x_padded[:length]) for x_padded, length in zip(xs_padded, lengths)]
