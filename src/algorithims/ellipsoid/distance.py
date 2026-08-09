import numpy as np

def distance_squared(
    diff: np.ndarray,
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    reg: float,
) -> np.ndarray:
    if eigvals.size:
        proj = diff @ eigvecs

        obs_d2 = np.sum(proj**2 / (eigvals + reg), axis=1)
        proj_norm2 = np.sum(proj**2, axis=1)
    else:
        obs_d2 = np.zeros(len(diff), dtype=float)
        proj_norm2 = np.zeros(len(diff), dtype=float)

    total_norm2 = np.sum(diff**2, axis=1)

    residual_norm2 = np.maximum(total_norm2 - proj_norm2, 0.0)

    return obs_d2 + residual_norm2 / reg