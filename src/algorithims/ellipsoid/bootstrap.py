
import numpy as np
import pandas as pd 

from dataclasses import dataclass, asdict, field

from tqdm.auto import tqdm

from .algorithim import EllipsoidCover
from .evaluator import EllipsoidEvaluator
from ..types import EvaluationMetrics

from ...stats.mahalanobis_detector import MahalanobisDetector

@dataclass
class BootstrapResults:
    bootstrap: int

    alg_metrics: EvaluationMetrics
    mal_auroc: float

    delta: float = field(init=False)

    def __post_init__(self) -> None:
        self.delta = self.alg_metrics.auroc - self.mal_auroc

    def to_dict(self):
        return {
            "bootstrap": self.bootstrap,
            **self.alg_metrics.to_dict(),
            "mal_auroc": self.mal_auroc,
            "delta": self.delta,
        }

@dataclass
class TrainBootstrapResults(BootstrapResults):
    n_ellipsoids: int

    mean_n_points: float
    median_n_points: float

    fraction_supported: float

    pc95_mean: float # avg min component to explain 95% of the variance
    pc95_median: float

    pc1_ratio_mean: float
    rank_mean: float

    def to_dict(self):
        return {
            **super().to_dict(),
            "n_ellipsoids": self.n_ellipsoids,
            "mean_n_points": self.mean_n_points,
            "median_n_points": self.median_n_points,
            "fraction_supported": self.fraction_supported,
            "pc95_mean": self.pc95_mean,
            "pc95_median": self.pc95_median,
            "pc1_ratio_mean": self.pc1_ratio_mean,
            "rank_mean": self.rank_mean,
        }

class BootstrapRunner:
    def __init__(self, 
                cover: EllipsoidCover, 
                evaluator: EllipsoidEvaluator,
                mal_detector: MahalanobisDetector,
                n_test_bootstraps: int = 1000,
                n_train_bootstraps: int = 100, 
                seed: int = 42,
        ) -> None:
        self.cover = cover
        self.evaluator = evaluator
        self.mal_detector = mal_detector

        self.test_bootstraps = n_test_bootstraps
        self.train_bootstraps = n_train_bootstraps

        self.rng = np.random.default_rng(seed)

    def run(self,
            train_emb: np.ndarray,
            good_test_emb: np.ndarray,
            defect_test_emb: np.ndarray
        )-> tuple[pd.DataFrame, pd.DataFrame]:
        test_df = self.bootstrap_test_embeds(
            train_emb=train_emb,
            good_test_emb=good_test_emb,
            defect_test_emb=defect_test_emb
        )

        train_df = self.bootstrap_train_embeds(
            train_emb=train_emb,
            good_test_emb=good_test_emb,
            defect_test_emb=defect_test_emb
        )

        return test_df, train_df

    def bootstrap_test_embeds(self, 
            train_emb: np.ndarray, 
            good_test_emb: np.ndarray,
            defect_test_emb: np.ndarray
        )-> pd.DataFrame:
        results: list[BootstrapResults] = []

        collection = self.cover.run(embeds=train_emb)
        self.mal_detector.fit(train_emb)

        for idx in tqdm(
            range(self.test_bootstraps), 
            desc="Test Bootstraps",
            mininterval=1.0,
            miniters=50
            ):
            # Good and defect test ratios should remain the same
            good_idx = self.rng.choice(
                len(good_test_emb),
                len(good_test_emb),
                replace=True
            )

            defect_idx = self.rng.choice(
                len(defect_test_emb),
                len(defect_test_emb),
                replace=True
            )

            good_boot_emb = good_test_emb[good_idx]
            defect_boot_emb = defect_test_emb[defect_idx]

            evaluation = self.evaluator.evaluate_detection(
                good_test_emb=good_boot_emb,
                defect_test_emb=defect_boot_emb,
                collection=collection
            )

            mal_score = self.mal_detector.evaluate_detection(
                good_embeds=good_boot_emb,
                defect_embeds=defect_boot_emb
            )
    
            results.append(BootstrapResults(
                bootstrap=idx, 
                alg_metrics=evaluation.metrics,
                mal_auroc=mal_score,
            ))

        return pd.DataFrame(r.to_dict() for r in results)

    def bootstrap_train_embeds(self,
            train_emb: np.ndarray,
            good_test_emb: np.ndarray,
            defect_test_emb: np.ndarray
        )-> pd.DataFrame:

        results: list[TrainBootstrapResults] = []

        for idx in tqdm(range(self.train_bootstraps), desc="Train Bootstraps"):
            sample_idx = self.rng.choice(
                len(train_emb),
                len(train_emb),
                replace=True
            )

            train_bootstrap = train_emb[sample_idx]
            self.mal_detector.fit(train_bootstrap)
            collection = self.cover.run(embeds=train_bootstrap)

            evaluation = self.evaluator.evaluate_detection(
                good_test_emb=good_test_emb,
                defect_test_emb=defect_test_emb,
                collection=collection
            )

            mal_score = self.mal_detector.evaluate_detection(
                good_embeds=good_test_emb,
                defect_embeds=defect_test_emb
            )

            stats_df = collection.to_dataframe()

            results.append(TrainBootstrapResults(
                bootstrap=idx,
                alg_metrics=evaluation.metrics,
                mal_auroc=mal_score,
                n_ellipsoids=len(collection),

                mean_n_points=float(stats_df["n_points"].mean()),
                median_n_points=stats_df["n_points"].median(),

                fraction_supported=stats_df["support_id"].notna().mean(),

                pc95_mean=stats_df["pc95"].mean(),
                pc95_median=stats_df["pc95"].median(),

                pc1_ratio_mean=stats_df["pc1_ratio"].mean(),
                rank_mean=stats_df["rank"].mean()
            ))

        return (pd.DataFrame(r.to_dict() for r in results))

    @staticmethod
    def summarise_bootstrap(df: pd.DataFrame) -> pd.Series:
        summary: dict[str, float | int] = {}

        for col in df.columns:
            if col == "bootstrap" or col == "category":
                continue

            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_std"] = df[col].std()
            summary[f"{col}_ci_lower"] = df[col].quantile(0.025)
            summary[f"{col}_ci_upper"] = df[col].quantile(0.975)

        return pd.Series(summary)


