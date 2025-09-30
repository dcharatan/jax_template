from dataclasses import dataclass
from typing import Any, Literal

import tensorflow_datasets as tfds
from einops import repeat
from grain.python import MapTransform, RandomAccessDataSource, Transformations

from .interface import Dataset


@dataclass(frozen=True)
class DatasetMnistCfg:
    name: Literal["mnist"]


class Convert(MapTransform):
    def map(self, element):
        return {
            "rgb": repeat(element["image"] / 255, "h w () -> c h w", c=3),
            "label": element["label"],
        }


class DatasetMnist(Dataset[DatasetMnistCfg]):
    def get_data_source(self) -> RandomAccessDataSource[Any]:
        return tfds.data_source("mnist", split="train")

    def get_transformations(self) -> Transformations:
        return [Convert()]
