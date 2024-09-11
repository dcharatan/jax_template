from dataclasses import dataclass
from typing import Any, Literal

import tensorflow_datasets as tfds
from einops import repeat
from grain.python import MapTransform, RandomAccessDataSource, Transformations

from .interface import Dataset


@dataclass
class DatasetMnistCfg:
    name: Literal["mnist"]


class DatasetMnist(Dataset[DatasetMnistCfg]):
    def get_data_source(self) -> RandomAccessDataSource[Any]:
        return tfds.data_source("mnist", split="train")

    def get_transformations(self) -> Transformations:
        class Convert(MapTransform):
            def map(self, element):
                return {"rgb": repeat(element["image"], "h w () -> h w c", c=3)}

        return [Convert()]
