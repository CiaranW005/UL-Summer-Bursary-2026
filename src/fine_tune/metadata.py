import pandas as pd
import sqlite3

from pathlib import Path
from dataclasses import dataclass

@dataclass
class DatasetMeta:
    meta: pd.DataFrame
    train_paths: list[str]
    test_paths: list[str]
    categories: list[str]
    types: list[str]

def load_dataset_meta(DB_PATH: Path) -> DatasetMeta:
    with sqlite3.connect(DB_PATH) as conn:
        meta = pd.read_sql_query(
            """
            SELECT path, split, category, type
            FROM meta
            """,
            conn
        )

    return DatasetMeta(
        meta=meta,
        train_paths=meta.loc[meta["split"] == "train", "path"].tolist(),
        test_paths=meta.loc[meta["split"] == "test", "path"].tolist(),
        categories=meta["category"].unique().tolist(),
        types=meta["type"].unique().tolist()
    )