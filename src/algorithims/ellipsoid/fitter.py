import numpy as np

from collections.abc import Sequence
import logging

from ..types import Ellipsoid

logger =  logging.getLogger(__name__)
class EllipsoidFitter:
    def __init__(self, 
            support_points: int =5, 
            reg: float =1e-4):
        self.support_points=support_points
        self.reg = reg

    def fit(self, X: np.ndarray, weights: np.ndarray) -> Ellipsoid:
        """Fit a weighted ellipsoid to a collection of points."""

        # Normalize weights for statistical estimation of center and covariance.
        w = weights / weights.sum()

        center = np.average(X, axis=0, weights=w)
        diff = X - center

        if len(X) > 1:
            cov = (diff * w[:, None]).T @ diff 

            correction = 1.0 / (1.0 - np.sum(w ** 2))
            cov *= correction
        else:
            cov = np.eye(X.shape[1], X.shape[1])

        # Regularise because local regions may have very few points
        cov += np.eye(cov.shape[0]) * self.reg

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals = np.maximum(eigvals, self.reg)

        proj = diff @ eigvecs


        d2 = np.sum((proj ** 2) / eigvals, axis=1)
        # Use raw cleaning weights for the boundary so shrinking a point's weight
        # directly reduces its influence on the ellipsoid threshold.
        threshold = (d2 * weights).max()

        eig_ratio = eigvals.max() / eigvals.min() 

        return Ellipsoid(
            center=center,
            eigvecs=eigvecs,
            eigvals=eigvals,
            threshold=threshold,
            eig_ratio=eig_ratio,
            weights=weights
        )
    
    def fit_supported(self, X: np.ndarray, weights: np.ndarray, previous_ellipsoids: Sequence[Ellipsoid]):
        """Fit an ellipsoid using the nearest previous ellipsoid as support."""
        if not previous_ellipsoids:
            logger.warning("Previous ellipsoids is empty")
            return self.fit(X, weights)
        
        w = weights / weights.sum()
        
        center = np.average(X, axis=0, weights=w)
        diff = X - center

        if len(X) > 1:
            own_cov = (diff * w[:, None]).T @ diff 

            correction = 1.0 / (1.0 - np.sum(w ** 2))
            own_cov *= correction
        else:
            own_cov = np.eye(X.shape[1])

        distances = [
            np.linalg.norm(center - ellipsoid.center)
            for ellipsoid in previous_ellipsoids
        ]

        support_id = int(np.argmin(distances))
        support = previous_ellipsoids[support_id]

        support = previous_ellipsoids[support_id]

        sup_cov = support.eigvecs @ np.diag(support.eigvals) @ support.eigvecs.T

        alpha = min(1.0, len(X) / self.support_points)
        cov = alpha * own_cov + (1 - alpha) * sup_cov
        cov += np.eye(cov.shape[0]) * self.reg

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = np.maximum(eigvals[order], self.reg)
        eigvecs = eigvecs[:, order]

        diff = X - center
        proj = diff @ eigvecs
        d2 = np.sum((proj ** 2) / eigvals, axis=1)

        threshold = (d2 * weights).max() if len(d2) else 0.0

        # singleton fallback scale
        if threshold == 0.0:
            threshold = support.threshold * (1.0 - alpha)

        eig_ratio = eigvals.max() / eigvals.min()

        return Ellipsoid(
            center=center,
            eigvecs=eigvecs,
            eigvals=eigvals,
            threshold=threshold,
            eig_ratio=eig_ratio,
            support_id=support_id,
            weights=weights
        )

    @staticmethod
    def inside(X: np.ndarray, ellipsoid: Ellipsoid) -> np.ndarray:
        """Return a mask indicating which points lie inside an ellipsoid."""
        diff = X - ellipsoid.center
        proj = diff @ ellipsoid.eigvecs
        d2 = np.sum((proj ** 2) / ellipsoid.eigvals, axis=1)
        return d2 <= ellipsoid.threshold 
    
    @staticmethod
    def grow(
        X: np.ndarray, 
        ellipsoid: Ellipsoid, 
        growth: float, 
        min_growth: float = 5e-3
    )-> np.ndarray:
        """Return points covered after variance-scaled ellipsoid growth."""
        diff = X - ellipsoid.center
        proj = diff @ ellipsoid.eigvecs

        var_ratio = ellipsoid.eigvals / ellipsoid.eigvals.max()
        axis_growth = 1.0 + (growth - 1) * np.clip(var_ratio, min_growth, 1.0)

        grown_eigvals = ellipsoid.eigvals * (axis_growth ** 2)

        d2 = np.sum((proj ** 2) / grown_eigvals, axis=1)
        return d2 <= ellipsoid.threshold
    