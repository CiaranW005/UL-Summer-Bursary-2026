import numpy as np

from collections.abc import Sequence
import logging

from .distance import distance_squared
from ..types import EllipsoidFit, Ellipsoid, EllipsoidStats

logger =  logging.getLogger(__name__)
class EllipsoidFitter:
    def __init__(self, 
            support_points: int = 5, 
            reg: float = 1e-4):
        self.support_points=support_points
        self.reg = reg

    def fit(self, X: np.ndarray, weights: np.ndarray) -> EllipsoidFit:
        """Fit a weighted ellipsoid to a collection of points."""

        center, diff, eigvals, eigvecs = self._weighted_covariance(X, weights)

        threshold = self._fit_threshold(
            diff=diff,
            eigvecs=eigvecs,
            eigvals=eigvals,
            weights=weights
        )

        stats = self._covariance_stats(eigvals)

        return EllipsoidFit(
            ellipsoid=Ellipsoid(
                center=center,
                eigvecs=eigvecs,
                eigvals=eigvals,
                threshold=threshold,
                weights=weights
            ),
            stats=stats
        )
    
    def fit_supported(
            self, 
            X: np.ndarray, 
            weights: np.ndarray, 
            previous_ellipsoids: Sequence[Ellipsoid]
        ) -> EllipsoidFit:
        """Fit an ellipsoid using nearby ellipsoids as covariance support candidates.

        Support is first restricted to the nearest ellipsoids in embedding space.

        For singleton candidates (rank 0), support is selected using:
        - Embedding cosine similarity between the candidate and support centres.
        - Alignment of the candidate direction with the support ellipsoid's principal axes.
        - Axis-length-weighted directional similarity.

        For higher-rank candidates, support is selected using:
        - Principal subspace similarity computed from the support and candidate eigenvectors.
        """

        if not previous_ellipsoids:
            logger.warning("Previous ellipsoids is empty")
            return self.fit(X, weights)
        
        center, diff, own_eigvals, own_eigvecs = self._weighted_covariance(X, weights)

        centers = np.stack([e.center for e in previous_ellipsoids])
        distances = np.linalg.norm(center - centers, axis=1) # Points fron support points to candidate
        order = np.argsort(distances)

        nearest_ids = order[:5]
        nearest_ellipsoids = [previous_ellipsoids[i] for i in nearest_ids]

        sims = self._ellipsoid_sims(
            X,
            eigvals=own_eigvals,
            eigvecs=own_eigvecs,
            nearest_ellipsoids=nearest_ellipsoids
        )

        support_id = int(np.argmax(sims))
        support = previous_ellipsoids[support_id]

        alpha = min(1.0, len(X) / self.support_points)
        eigvals, eigvecs = self._blend_covariance(
            own_eigvals=own_eigvals,
            own_eigvecs=own_eigvecs,
            sup_eigvals=support.eigvals,
            sup_eigvecs=support.eigvecs,
            alpha=alpha
        )
  
        threshold = self._fit_threshold(
            diff=diff,
            eigvals=eigvals,
            eigvecs=eigvecs,
            weights=weights
        )

        # singleton fallback scale
        if threshold == 0.0:
            threshold = support.threshold * (1.0 - alpha)

        stats = self._covariance_stats(eigvals)

        return EllipsoidFit(
            ellipsoid=Ellipsoid(
            center=center,
            eigvecs=eigvecs,
            eigvals=eigvals,
            threshold=threshold,
            support_id=support_id,
            weights=weights
            ),
            stats=stats
        )

    def _fit_threshold(
        self,
        diff: np.ndarray,
        eigvecs: np.ndarray,
        eigvals: np.ndarray,
        weights: np.ndarray
    )-> float:
        if len(diff) == 0:
            return 0.0

        d2 = distance_squared(
            diff, eigvecs, eigvals, self.reg
        )

        return float((d2 * weights).max())

    def _covariance_stats(self, eigvals: np.ndarray) -> EllipsoidStats:
        if len(eigvals) == 0:
            return EllipsoidStats(
                raw_eig_ratio=float("nan"),
                reg_eig_ratio=float("nan"),
                pc95=0,
                pc1_ratio=0,
                rank=0
            )
        
        total_var = eigvals.sum()
        explained = eigvals / total_var
        cumulative = np.cumsum(explained)

        pc95 = int(np.searchsorted(cumulative, 0.95) + 1)
        pc1_ratio = explained[0]
        

        raw_eig_ratio = (float(eigvals.max() / eigvals.min()) 
                            if len(eigvals) >=2 else float("nan"))

        reg_eigvals = eigvals + self.reg
        reg_eig_ratio = float(reg_eigvals.max() / reg_eigvals.min())

        return EllipsoidStats(
            raw_eig_ratio=raw_eig_ratio,
            reg_eig_ratio=reg_eig_ratio,
            pc95=pc95,
            pc1_ratio=pc1_ratio,
            rank=len(eigvals)
        )

    def inside(self, X: np.ndarray, ellipsoid: Ellipsoid) -> np.ndarray:
        """Return a mask indicating which points lie inside an ellipsoid."""
        diff = X - ellipsoid.center

        d2 = distance_squared(
            diff, ellipsoid.eigvecs, ellipsoid.eigvals, self.reg
        )

        return d2 <= ellipsoid.threshold 
    
    def grow(
        self,
        X: np.ndarray, 
        ellipsoid: Ellipsoid, 
        growth: float, 
        min_growth: float = 5e-3
    )-> np.ndarray:
        """Return points covered after variance-scaled ellipsoid growth."""
        diff = X - ellipsoid.center

        if ellipsoid.eigvals.size == 0:
            d2 = distance_squared(
                diff,
                ellipsoid.eigvecs,
                ellipsoid.eigvals,
                self.reg
            )
            return d2 <= ellipsoid.threshold
        
        var_ratio = ellipsoid.eigvals / ellipsoid.eigvals.max()
        axis_growth = 1.0 + (growth - 1) * np.clip(var_ratio, min_growth, 1.0)

        grown_eigvals = ellipsoid.eigvals * (axis_growth ** 2)

        d2 = distance_squared(
            diff, ellipsoid.eigvecs, grown_eigvals, self.reg
        )

        return d2 <= ellipsoid.threshold

    def _ellipsoid_sims(
        self,
        X: np.ndarray,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        nearest_ellipsoids: Sequence[Ellipsoid]
    ) -> list[float]:
        sims: list[float] = []

        if eigvals.size > 0:
            for ellipsoid in nearest_ellipsoids:
                if ellipsoid.eigvecs.shape[1] == 0:
                    sims.append(0.0)
                    continue

                pc_cosines = np.linalg.svd(
                    eigvecs.T @ ellipsoid.eigvecs,
                    compute_uv=False
                )

                sims.append(pc_cosines.mean())
        else:
            candidate = X.mean(axis=0)
            cand_norm = np.linalg.norm(candidate)

            for ellipsoid in nearest_ellipsoids:
                support_norm = np.linalg.norm(ellipsoid.center)
                denom = cand_norm * support_norm

                embed_sim = (candidate @ ellipsoid.center / denom) if denom > 0 else 0.0

                direc = candidate - ellipsoid.center
                dist = np.linalg.norm(direc)

                if dist == 0 or ellipsoid.eigvecs.shape[1] == 0:
                    shape_sim = 1.0
                else:
                    unit_direc = direc / dist

                    axis_cosines = np.abs(unit_direc @ ellipsoid.eigvecs)
                    axis_lengths = np.sqrt(ellipsoid.eigvals + self.reg)

                    shape_sim = (axis_cosines @ axis_lengths) / (axis_lengths.sum() + 1e-12)

                score = (embed_sim + shape_sim) / 2
                sims.append(score)

        return sims
    
    @staticmethod 
    def _weighted_covariance(
        X: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(X) == 0:
            raise ValueError("Cannot fit an ellispoids to an empty array")

        weight_sum = weights.sum()

        if weight_sum <= 0:
            raise ValueError("Weights must have a positive sum")

        w = weights/weight_sum

        center = np.average(X, axis=0, weights=w)
        diff = X - center

        if len(X) == 1:
            eigvals = np.empty(0, dtype=X.dtype)
            eigvecs = np.empty((X.shape[1], 0), dtype=X.dtype)
        else:
            weighted_diff = (diff * np.sqrt(w[:, None]))

            _, S, Vt = np.linalg.svd(weighted_diff, full_matrices=False)

            denom: float = 1.0 - np.sum(w**2)

            if denom <= 0:
                raise ValueError("Weighted covariance requires more than one effective sample")

            eigvals = (S**2) / denom

            tolerance = np.finfo(eigvals.dtype).eps * max(weighted_diff.shape) * max(eigvals)
            keep = eigvals > tolerance

            eigvals = eigvals[keep]
            eigvecs = Vt.T[:, keep]

        return center, diff, eigvals, eigvecs

    @staticmethod
    def _blend_covariance(
        own_eigvals: np.ndarray,
        own_eigvecs: np.ndarray,
        sup_eigvals: np.ndarray,
        sup_eigvecs: np.ndarray,
        alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        factors = []

        if len(own_eigvals):
            factors.append(own_eigvecs * np.sqrt(alpha * own_eigvals))

        if len(sup_eigvals):
            factors.append(sup_eigvecs * np.sqrt((1.0 - alpha) * sup_eigvals))

        if not factors:
            return (
                np.empty(0, dtype=own_eigvals.dtype),
                np.empty((own_eigvecs.shape[0], 0), dtype=own_eigvecs.dtype)
            )

        factor = np.concatenate(factors, axis=1)

        eigvecs, S, _ = np.linalg.svd(factor, full_matrices=False)

        eigvals = S ** 2
        tolerance = np.finfo(eigvals.dtype).eps * max(factor.shape) * max(eigvals)

        keep = eigvals > tolerance
        
        return eigvals[keep], eigvecs[:, keep]
