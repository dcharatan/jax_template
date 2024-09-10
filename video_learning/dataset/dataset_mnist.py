from dataclasses import dataclass
from typing import Any, Literal

import tensorflow_datasets as tfds
from grain.python import RandomAccessDataSource, Transformations

from .interface import Dataset


@dataclass
class DatasetMnistCfg:
    name: Literal["mnist"]


class DatasetMnist(Dataset[DatasetMnistCfg]):
    def get_data_source(self) -> RandomAccessDataSource[Any]:
        return tfds.data_source("mnist", split="train")

    def get_transformations(self) -> Transformations:
        return []
