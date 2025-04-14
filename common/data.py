from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Tuple

import polars as pl


class DataPath:
    REPO_ID = "klue/klue"
    LOCAL_DIR = Path("klue-sts-v1.1/hub")
    TRAIN_FILE = 'sts/train-00000-of-00001.parquet'
    TEST_FILE = 'sts/validation-00000-of-00001.parquet'

    @property
    def local_dir(self) -> str:
        return str(self.LOCAL_DIR)

    @property
    def files(self) -> list[str]:
        return [self.TRAIN_FILE, self.TEST_FILE]

    @property
    def local_train_file(self) -> Path:
        return self.LOCAL_DIR.joinpath(self.TRAIN_FILE)

    @property
    def local_test_file(self) -> Path:
        return self.LOCAL_DIR.joinpath(self.TEST_FILE)


@dataclass
class Row:
    guid: str = field()
    sentence1: str = field()
    sentence2: str = field()
    label: float = field()
    real_label: float = field()
    binary_label: float = field()

    @classmethod
    def from_dicts(cls, data: list[dict[str, Any]]) -> Iterable['Row']:
        for d in data:
            yield Row(
                guid=d["guid"],
                sentence1=d["sentence1"],
                sentence2=d["sentence2"],
                label=d['labels']["label"],
                real_label=d['labels']["real-label"],
                binary_label=d['labels']["binary-label"],
            )


class KlueDataLoader:

    def __init__(self, sample: bool = False):
        self.path = DataPath()
        self.sample = sample

    def train_test_dataframe(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        train = self.read_parquet(self.path.local_train_file)
        test = self.read_parquet(self.path.local_test_file)
        return train, test

    def train_test(self) -> Tuple[list[Row], list[Row]]:
        train = self.read_parquet(self.path.local_train_file)
        test = self.read_parquet(self.path.local_test_file)

        train = list(Row.from_dicts(train.to_dicts()))
        test = list(Row.from_dicts(test.to_dicts()))

        return train, test

    def read_parquet(self, file_path):
        df = pl.read_parquet(file_path)
        if self.sample:
            df = df.head(10)
        return df
