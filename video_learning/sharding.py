import equinox as eqx
import jax
from jaxtyping import PyTree


def get_mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(jax.devices(), ["batch"])


def split(tree: PyTree, mesh: jax.sharding.Mesh) -> PyTree:
    """Shard the PyTree along the first (batch) axis."""

    def get_split_sharding(node):
        # Ignore non-array nodes.
        if not eqx.is_array(node):
            return None

        # Create a NamedSharding that shards the first axis and replicates all others.
        unsharded_axes = (None,) * (node.ndim - 1)
        partition_spec = jax.sharding.PartitionSpec(("batch", *unsharded_axes))
        return jax.sharding.NamedSharding(mesh, partition_spec)

    shardings = jax.tree.map(get_split_sharding, tree)
    return eqx.filter_shard(tree, shardings)


def replicate(tree: PyTree, mesh: jax.sharding.Mesh) -> PyTree:
    """Replicate the PyTree on all devices."""

    def get_replicated_sharding(node):
        # Ignore non-array nodes.
        if not eqx.is_array(node):
            return None

        # Create a sharding that replicates all axes.
        unsharded_axes = (None,) * node.ndim
        partition_spec = jax.sharding.PartitionSpec(unsharded_axes)
        return jax.sharding.NamedSharding(mesh, partition_spec)

    shardings = jax.tree.map(get_replicated_sharding, tree)
    return eqx.filter_shard(tree, shardings)
