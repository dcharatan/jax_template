from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import equinox as eqx
from grain.python import RandomAccessDataSource, Transformations
from jaxtyping import Array, Float

# The following types are a bit awkward. Ideally, we would want a struct-of-arrays
# annotation that links Example and Batch. See:
# https://github.com/patrick-kidger/jaxtyping/issues/242
# https://github.com/patrick-kidger/equinox/issues/692
# https://github.com/patrick-kidger/jaxtyping/issues/84


class Example(eqx.Module):
    rgb: Float[Array, "height width 3"]


class Batch(eqx.Module):
    rgb: Float[Array, "batch height width 3"]


C = TypeVar("C")  # config type


class Dataset(Generic[C], ABC):
    cfg: C

    def __init__(self, cfg: C) -> None:
        self.cfg = cfg

    @abstractmethod
    def get_data_source(self) -> RandomAccessDataSource[Any]:
        pass

    @abstractmethod
    def get_transformations(self) -> Transformations:
        pass
