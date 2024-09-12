from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

from grain.python import RandomAccessDataSource, Transformations
from jaxtyping import ArrayLike, Float, Int

# Ideally, we would want a struct-of-arrays annotation that constructs Batch from a
# scalar Example type. See:
# https://github.com/patrick-kidger/jaxtyping/issues/242
# https://github.com/patrick-kidger/equinox/issues/692
# https://github.com/patrick-kidger/jaxtyping/issues/84


class Batch(TypedDict):
    rgb: Float[ArrayLike, "batch 3 height width"]
    label: Int[ArrayLike, " batch"]


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
