from dataclasses import dataclass, field, asdict
from typing import TypedDict
from collections.abc import Iterator

import pandas as pd
import numpy as np

@dataclass
class CategoryMasks:
    category: str
    
    train_mask: pd.Series
    test_mask: pd.Series
    train_category_mask: pd.Series

    good_test_mask: pd.Series
    defect_test_mask: pd.Series

@dataclass
class Ellipsoid:
    center: np.ndarray
    eigvecs: np.ndarray
    eigvals: np.ndarray
    threshold: float

    support_id: int | None = None
    weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    covered_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

@dataclass 
class EllipsoidStats:
    raw_eig_ratio: float
    reg_eig_ratio: float

    pc95: int # The min principal component needed to explain 95% of the variance   
    rank: int
    pc1_ratio: float # How much PC1 explains of the variance

    n_points: int = 1

@dataclass 
class EllipsoidFit:
    ellipsoid: Ellipsoid
    stats: EllipsoidStats

    def set_covered_idx(self, covered_idx: np.ndarray) -> None:
        self.ellipsoid.covered_idx = covered_idx
        self.stats.n_points = len(covered_idx)

@dataclass
class EllipsoidCollection:
    ellipsoids: list[Ellipsoid] = field(default_factory=list)
    stats: list[EllipsoidStats] = field(default_factory=list)

    def append(self, fit: EllipsoidFit) -> None:
        self.ellipsoids.append(fit.ellipsoid)
        self.stats.append(fit.stats)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []

        for idx, (ellipse, stat) in enumerate(zip(self.ellipsoids, self.stats, strict=True)):
            weights = ellipse.weights

            rows.append({
                "ellipsoid_id": idx,
                **asdict(stat),
                "threshold": ellipse.threshold,
                "support_id": ellipse.support_id,
                "weights_mean": float(weights.mean()) if len(weights) else np.nan,
                "weights_min": float(weights.min()) if len(weights) else np.nan,
                "n_reduced_weights": int(np.count_nonzero(weights < 1.0))
            })

        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.ellipsoids)

    def __iter__(self) -> Iterator[tuple[Ellipsoid, EllipsoidStats]]:
        return iter(zip(self.ellipsoids, self.stats, strict=True))

    def __getitem__(self, key: int) -> tuple[Ellipsoid, EllipsoidStats]:
        return self.ellipsoids[key], self.stats[key]
 
class EvaluationMetrics:
    def __init__(self, samples: pd.DataFrame, auroc: float, threshold: float):
        self.auroc = auroc
        self.best_threshold = threshold

        good = samples["y_true"] == 0
        defect = samples["y_true"] == 1

        self.accuracy=float(samples["correct"].mean())
        self.good_accuracy=float(samples.loc[good, "correct"].mean())
        self.defect_accuracy=float(samples.loc[defect, "correct"].mean())

        self.mean_good_score=float(samples.loc[good, "score"].mean())
        self.mean_defect_score=float(samples.loc[defect, "score"].mean())
        self.score_gap=self.mean_defect_score - self.mean_good_score

        self.false_pos=int((good & (samples["predicted_label"] == 1)).sum())
        self.false_neg=int((defect & (samples["predicted_label"] == 0)).sum())

        self.n_winning_ellipsoid=int(samples["winning_ellipsoid"].nunique())
        self.max_winning_fraction=float((samples["winning_ellipsoid"].value_counts(normalize=True)).max())

    def to_dict(self) -> dict[str, float | int]:
        return vars(self)

@dataclass
class EllipsoidEvaluation:
    samples: pd.DataFrame
    metrics: EvaluationMetrics
    
class EllipsoidOverlapResult(TypedDict):
    ellipsoid_i: int
    ellipsoid_j: int
    j_points_inside_i: int
    i_points_inside_j: int
    overlap: bool

@dataclass
class Hypersphere:
    center: np.ndarray
    radius: float
    covered_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

class HypersphereOverlapResult(TypedDict):
    sphere_i: int
    sphere_j: int
    centre_dist: float
    add_radii: float
    overlap: bool
    overlap_amount: float
    j_points_inside_i: int
    i_points_inside_j: int

class BucketAUROC(TypedDict):
    eigval_ratio_bucket: str
    auroc: float  | None
    count: int
    