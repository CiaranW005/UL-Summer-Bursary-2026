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
    
@dataclass
class Hypersphere:
    center: np.ndarray
    radius: float
    covered_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

class EllipsoidOverlapResult(TypedDict):
    ellipsoid_i: int
    ellipsoid_j: int
    j_points_inside_i: int
    i_points_inside_j: int
    overlap: bool

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
    