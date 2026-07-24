from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from collections.abc import Sequence

from ..types import Hypersphere, HypersphereOverlapResult

class HypersphereEvaluator:

    @staticmethod
    def overlaps(embeds: np.ndarray, spheres: Sequence[Hypersphere])-> pd.DataFrame:
        """Measure geometric and point-level overlao between sphere pairs
        
        For eahc pair, records the centre distance, geometric overlap amounts, and 
        how many pints assinged ot either sphere fall inside the other
        """
        rows: list[HypersphereOverlapResult] = []

        for i, j in combinations(range(len(spheres)), 2):
            s1 = spheres[i]
            s2 = spheres[j]

            c1, r1 = s1.center, s1.radius
            c2, r2 = s2.center, s2.radius

            centre_dist = float(np.linalg.norm(c1 - c2))
            overlap_amount = (r1 + r2) - centre_dist

            # points owned by sphere j/i that fall inside sphere i/j
            dist_to_c1 = np.linalg.norm(embeds[s2.covered_idx] - c1, axis=1) 
            dist_to_c2 = np.linalg.norm(embeds[s1.covered_idx] - c2, axis=1) 

            c1_overlap_count = int(np.sum(dist_to_c1 <= r1))
            c2_overlap_count = int(np.sum(dist_to_c2 <= r2))

            rows.append({
                "sphere_i": i,
                "sphere_j": j,
                "centre_dist": centre_dist,
                "add_radii": float(r1 + r2),
                "overlap": bool(overlap_amount > 0),
                "overlap_amount": max(0, overlap_amount),
                "j_points_inside_i": c1_overlap_count,
                "i_points_inside_j": c2_overlap_count
            })

        return pd.DataFrame(rows)
    
    @staticmethod
    def inside_any_count(X: np.ndarray, spheres: Sequence[Hypersphere]) -> tuple[np.ndarray, np.ndarray]:
        """Return whether and how many times each sample is covered.

        The first array indicates whether each sample lies inside at least one
        sphere. The second contains the number of spheres covering each sample.
        """
        inside_any = np.zeros(len(X), dtype=bool)
        inside_count = np.zeros(len(X), dtype=int)

        for s in spheres:
            d = np.linalg.norm(X - s.center, axis=1)
            inside = d <= s.radius

            inside_any |= inside
            inside_count += inside

        return inside_any, inside_count
    
    @staticmethod
    def scores(good_test_emb: np.ndarray, defect_test_emb: np.ndarray, spheres: Sequence[Hypersphere])-> float:
        """Evaluate anomaly detection using distance from the nearest boundary.

        Negative scores indicate samples inside a sphere, while positive scores
        indicate samples outside every sphere.
        """
        X = np.vstack([good_test_emb, defect_test_emb])

        # 0 = normal, 1 = defect
        y_true = np.concatenate([
            np.zeros(len(good_test_emb)),
            np.ones(len(defect_test_emb))
        ])

        scores = np.full(len(X), np.inf)

        for s in spheres:
            d = np.linalg.norm(X - s.center, axis=1)
            margin = d - s.radius     # <0 inside, >0 outside

            scores = np.minimum(scores, margin)

        auroc = float(roc_auc_score(y_true, scores))
        return auroc
    