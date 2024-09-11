from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

from grain.python import RandomAccessDataSource, Transformations
from jaxtyping import ArrayLike, Float


class Batch(TypedDict):
    rgb: Float[ArrayLike, "batch height width 3"]


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
