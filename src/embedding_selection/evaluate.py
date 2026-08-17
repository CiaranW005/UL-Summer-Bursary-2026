
import pandas as pd
import numpy as np

from tqdm.auto import tqdm 

from src.stats.mahalanobis_detector import MahalanobisDetector

def evaluate_embedding(embeds: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(columns=["category", "auroc"])

    for category in tqdm(meta["category"].unique(), desc="Evaluating Category", position=1):
        train_cls = embeds[(meta["category"] == category) & (meta["split"] == "train")]

        test_cls = embeds[(meta["category"] == category) & (meta["split"] == "test")]
        test_labels = meta[(meta["category"] == category) & (meta["split"] == "test")]["label"]

        good_embeds = test_cls[test_labels == 0]
        defect_emebds = test_cls[test_labels == 1]

        fitter = MahalanobisDetector(reg=1e-6)
        fitter.fit(train_cls)

        auroc = fitter.evaluate_detection(
            good_embeds=good_embeds,
            defect_embeds=defect_emebds
        )

        df.loc[len(df)] = {
            "category": category,
            "auroc": auroc,
        }

    return df

               