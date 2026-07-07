from dataclasses import dataclass
import numpy as np

@dataclass
class Ellipsoid:
    center: np.ndarray
    eigvecs: np.ndarray
    eigvals: np.ndarray
    threshold: float
    eig_ratio: float
    support_id: int | None = None

