import equinox as eqx
import jax
import numpy as np
from einops import repeat
from jaxtyping import ArrayLike, PyTree

# Since we're only doing simple SPMD/data parallelism, we use this mesh everywhere.
mesh = jax.sharding.Mesh(jax.devices(), ["batch"])


def filter_device_put(
    x: PyTree,
    sharding: PyTree[jax.sharding.Sharding],
) -> PyTree:
    dynamic, static = eqx.partition(x, eqx.is_array)
    dynamic = jax.device_put(dynamic, sharding)
    return eqx.combine(dynamic, static)


def get_distributed_sharding(tree: PyTree) -> PyTree[jax.sharding.NamedSharding | None]:
    """Create a sharding that shards along the first axis and replicates along all
    other axes.
    """

    def _get_distributed_sharding(array: ArrayLike) -> jax.sharding.NamedSharding:
        unsharded_axes = (None,) * (array.ndim - 1)
        partition_spec = jax.sharding.PartitionSpec(*("batch", *unsharded_axes))
        return jax.sharding.NamedSharding(mesh, partition_spec)

    return jax.tree.map(
        lambda x: _get_distributed_sharding(x) if eqx.is_array(x) else None, tree
    )


def get_replicated_sharding(tree: PyTree) -> PyTree[jax.sharding.NamedSharding | None]:
    """Create a sharding that replicates along all axes."""

    def _get_replicated_sharding(array: ArrayLike) -> jax.sharding.NamedSharding:
        unsharded_axes = (None,) * array.ndim
        partition_spec = jax.sharding.PartitionSpec(*unsharded_axes)
        return jax.sharding.NamedSharding(mesh, partition_spec)

    return jax.tree.map(
        lambda x: _get_replicated_sharding(x) if eqx.is_array(x) else None, tree
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
