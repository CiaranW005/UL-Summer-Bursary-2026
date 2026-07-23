import numpy as np

from collections.abc import Sequence

from .fitter import EllipsoidFitter

from ..types import Ellipsoid
class CandidateCleaner:
    def __init__(self, fitter: EllipsoidFitter, min_points: int =1):
        self.fitter = fitter

        self.min_points = min_points

    def clean_candidate(self, 
                    covered_idx: np.ndarray, 
                    embeds: np.ndarray, 
                    uncovered_mask: np.ndarray, 
                    weights: np.ndarray, 
                    ellipsoids: Sequence[Ellipsoid]
                ) -> Ellipsoid:
        """Shrink candidate-point weights until the encroachment is removed"""

        while len(covered_idx) > self.min_points:
            X = embeds[covered_idx]

            ellipsoid = self._fit_candidate(X, weights, ellipsoids)

            inside_all = self.fitter.inside(embeds, ellipsoid)
            shared_idx = np.where((inside_all) & (~uncovered_mask))[0]

            if len(shared_idx) == 0:
                ellipsoid.covered_idx = covered_idx
                return ellipsoid

            print("Encroachment Detected")

            shared_diff = embeds[shared_idx] - ellipsoid.center
            shared_proj = shared_diff @ ellipsoid.eigvecs
            shared_contrib = (shared_proj ** 2) / ellipsoid.eigvals

            bad_shared_local = shared_contrib.sum(axis=1).argmax()
            bad_axis = shared_contrib[bad_shared_local].argmax()

            cand_diff = embeds[covered_idx] - ellipsoid.center
            cand_proj = cand_diff @ ellipsoid.eigvecs
            cand_axis_contrib = (cand_proj[:, bad_axis] ** 2) / ellipsoid.eigvals[bad_axis]

            worst_local = cand_axis_contrib.argmax()
            weights = self.find_weight(X, embeds, weights, worst_local, uncovered_mask, ellipsoids)

            if np.isclose(weights[worst_local], 0.0, atol=1e-3):
                covered_idx = np.delete(covered_idx, worst_local)
                weights = np.delete(weights, worst_local)

        T = embeds[covered_idx]
        if len(covered_idx) < self.fitter.support_points and len(ellipsoids) > 0:
            ellipsoid = self.fitter.fit_supported(T, weights, ellipsoids)
        else:
            ellipsoid = self.fitter.fit(T, weights)
        
        ellipsoid.covered_idx = covered_idx
        return ellipsoid
    

    def find_weight(self, 
                X: np.ndarray, 
                embeds: np.ndarray, 
                weights: np.ndarray, 
                worst_local: int, 
                uncovered_mask: np.ndarray, 
                ellipsoids: Sequence[Ellipsoid],
                space: int = 10):
        """Find the largest non-encroaching weight using binary search"""
        lo = 0
        hi = weights[worst_local]

        best = lo
        test_weights = weights.copy()
        for _ in range(space):
            mid = (lo + hi) / 2
            
            test_weights[worst_local] = mid
            ellipsoid = self._fit_candidate(X, test_weights, ellipsoids)

            inside_all = self.fitter.inside(embeds, ellipsoid)
            shared_idx = np.where((inside_all) & (~uncovered_mask))[0]

            if len(shared_idx) == 0:
                best = mid
                lo = mid
            else:
                hi = mid
        
        weights[worst_local] = best
        return weights

    def _fit_candidate(self,
                X: np.ndarray,
                weights: np.ndarray,
                ellipsoids: Sequence[Ellipsoid]
            )-> Ellipsoid:
        """Fits a candidate using its own points or supported when too few points are available"""

        if (len(X) < self.fitter.support_points and len(ellipsoids) > 0):
            return self.fitter.fit_supported(X, weights, ellipsoids)

        return self.fitter.fit(X, weights)
