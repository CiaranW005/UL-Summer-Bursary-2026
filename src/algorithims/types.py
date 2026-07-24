from dataclasses import dataclass, field
from typing import TypedDict

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
    eig_ratio: float

    support_id: int | None = None
    weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    covered_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

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
    