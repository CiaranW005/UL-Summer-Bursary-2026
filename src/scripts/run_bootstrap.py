import os

import torch
import sqlite3
import pandas as pd

from ..algorithims.ellipsoid.factories import create_cover, create_evaluator
from ..algorithims.ellipsoid.bootstrap import BootstrapRunner

from ..config.paths import EMBEDS_DIR, EXPERIMENTS, RESULTS, DB_PATH

EMBED_PATH = EMBEDS_DIR / "dino/20260801_221150"
EMBED_NAME = EMBED_PATH.stem

EXPERIMENTS_DIR = EXPERIMENTS / EMBED_NAME / "ellipsoid_bootsrap"
RESULTS_DIR = RESULTS / EMBED_NAME / "ellipsoid_bootstrap"

def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cls_tokens = torch.load(EMBED_PATH/"cls.pt", weights_only=False)

    with sqlite3.connect(DB_PATH) as connection:
        meta = pd.read_sql_query(
            """
            SELECT category, type, label, split
            FROM meta
            """,
            connection,
        )

    categories = sorted(meta["category"].unique())
    
    train_mask = meta["split"] == "train"
    train_meta = meta[train_mask]
    test_meta = meta[~train_mask]

    train_emb = cls_tokens[train_mask]
    test_emb = cls_tokens[~train_mask]

    test_summaries : list[dict[str, object]] = []
    train_summaries: list[dict[str, object]] = []
    for category in categories:
        print("Running", category)

        output_dir = EXPERIMENTS_DIR / category 
        os.makedirs(output_dir, exist_ok=True)

        train_cat_mask = train_meta["category"] == category

        good_test_mask = (test_meta["category"] == category) & (test_meta["type"] == "good")
        defect_test_mask = (test_meta["category"] == category) & (test_meta["type"] != "good")

        cat_emb = train_emb[train_cat_mask]

        defect_test_emb = test_emb[defect_test_mask]
        good_test_emb = test_emb[good_test_mask]

        runner = BootstrapRunner(
            cover_factory=create_cover,
            eval_factory=create_evaluator,
            n_test_bootstraps=1000,
            n_train_bootstraps=100,
            seed=42,
            max_workers=1
        )

        test_bootstraps, train_bootstraps = runner.run(
            train_emb=cat_emb, 
            good_test_emb=good_test_emb, 
            defect_test_emb=defect_test_emb
            )

        test_bootstraps.insert(0,"category", category)
        train_bootstraps.insert(0, "category", category)

        test_bootstraps.to_csv(output_dir / "test_bootstraps.csv", index=False)

        train_bootstraps.to_csv(output_dir / "train_bootstraps.csv", index=False)

        test_summary = runner.summarise_bootstrap(test_bootstraps)
        train_summary = runner.summarise_bootstrap(train_bootstraps)

        test_summaries.append({"category": category, **test_summary})
        train_summaries.append({"category": category, **train_summary,})

        pd.DataFrame(test_summaries).to_csv(
            RESULTS_DIR / "test_summary.csv",
            index=False,
        )

        pd.DataFrame(train_summaries).to_csv(
            RESULTS_DIR / "train_summary.csv",
            index=False,
        )

    print("Bootstrap experiment complete", flush=True)

if __name__ == "__main__":
    main()