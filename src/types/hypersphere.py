from dataclasses import dataclass
import numpy as np

@dataclass
class Hypersphere:
    center: np.ndarray
    radius: float
    covered_idx: np.ndarray | None = None 