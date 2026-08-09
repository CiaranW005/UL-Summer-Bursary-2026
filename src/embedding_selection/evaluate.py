
import pandas as pd
import numpy as np

from tqdm.auto import tqdm 

from sklearn.metrics import roc_auc_score

def evaluate_embedding(embeds: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(columns=["category", "auroc"])

    for category in tqdm(meta["category"].unique(), desc="Evaluating Category", position=1):
        train_cls = embeds[(meta["category"] == category) & (meta["split"] == "train")]

        test_cls = embeds[(meta["category"] == category) & (meta["split"] == "test")]
        test_labels = meta[(meta["category"] == category) & (meta["split"] == "test")]["label"]

        centroid = train_cls.mean(axis=0)
        diff = test_cls - centroid

        cov = np.cov(train_cls.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.pinv(cov)

        mal_scores = np.sqrt(
            np.einsum(
                "ij, jk, ik->i",
                diff,
                cov_inv,
                diff
            )
        )

        auc = roc_auc_score(test_labels, mal_scores)
        df.loc[len(df)] = {
            "category": category,
            "auroc": auc
        }

    return df

               