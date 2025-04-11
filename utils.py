import json

import polars as pl


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def read_parquet(file_path):
    df = pl.read_parquet(file_path)
    return df.to_dicts()
