
import numpy as np

from sklearn.metrics import roc_auc_score

class MahalanobisDetector:
    def __init__(self, reg: float = 1e-6):
        self.reg = reg

        self._centroid: np.ndarray | None = None

        self._eigvals: np.ndarray | None = None
        self._eigvecs: np.ndarray | None = None

    def fit(self, embeds: np.ndarray) -> None:
        if len(embeds) < 2:
            raise ValueError("Mahlanobis fitting requires at least two embeddings")
        self._centroid = np.average(embeds, axis=0)

        diff = embeds - self._centroid

        _, S, Vt = np.linalg.svd(diff, full_matrices=False)

        eigvals = (S**2) / (len(diff) - 1)

        tolerance = np.finfo(eigvals.dtype).eps * max(diff.shape) * eigvals.max()
        keep = eigvals > tolerance

        self._eigvals = eigvals[keep]
        self._eigvecs = Vt.T[:, keep]


    def score(self, embeds: np.ndarray) -> np.ndarray:
        if self._centroid is None or self._eigvecs is None or self._eigvals is None:
            raise RuntimeError(
                "Mahlanobis detector must be fitted before scoring"
            )

        diff = embeds - self._centroid

        proj = diff @ self._eigvecs

        obs_d2 = np.sum((proj**2) / (self._eigvals + self.reg), axis=1)

        total2_norm = np.sum(diff**2, axis=1)
        proj2_norm = np.sum(proj**2, axis=1)

        residual2_norm = np.maximum(total2_norm - proj2_norm, 0.0)
        residual_d2 = residual2_norm / self.reg

        return np.sqrt(np.maximum(obs_d2 + residual_d2, 0.0))
        

    def evaluate_detection(
            self,
            good_embeds: np.ndarray,
            defect_embeds: np.ndarray
        ) -> float:

        good_scores = self.score(good_embeds)
        defect_scores = self.score(defect_embeds)

        labels = np.concatenate([
            np.zeros(len(good_scores), dtype=np.int8),
            np.ones(len(defect_scores), dtype=np.int8)
        ])

        scores = np.concatenate([
            good_scores,
            defect_scores
        ])

        return float(roc_auc_score(labels, scores))
        

        

