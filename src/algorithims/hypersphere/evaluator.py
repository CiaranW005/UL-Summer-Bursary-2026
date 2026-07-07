from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

class HypersphereEvaluator:

    @staticmethod
    def overlaps(embeds, spheres):
        overlaps = []

        for i, j in combinations(range(len(spheres)), 2):
            s1 = spheres[1]
            s2 = spheres[j]

            c1, r1 = s1.center, s1.radius
            c2, r2 = s2.center, s1.radius

            centre_dist = np.linalg.norm(c1 - c2)
            overlap_amount = (r1+r2) - centre_dist
            
            dist_to_c1 = np.linalg.norm(embeds[s2.covered_idx] - c1, axis=1) 
            dist_to_c2 = np.linalg.norm(embeds[s1.covered_idx] - c2, axis=1) 

            c1_overlap_count = np.sum(dist_to_c1 <= r1)
            c2_overlap_count = np.sum(dist_to_c2 <= r2)

            overlaps.append({
                "sphere_i": i,
                "sphere_j": j,
                "centre_dist": centre_dist,
                "add_radii": r1 + r2,
                "overlap": overlap_amount > 0,
                "overlap_amount": max(0, overlap_amount),
                "j_points_inside_i": c1_overlap_count,
                "i_points_inside_j": c2_overlap_count
            })

        overlap_df = pd.DataFrame(overlaps)

        return overlap_df
    
    @staticmethod
    def inside_any_count(X, spheres):
        inside_any = np.zeros(len(X), dtype=bool)
        inside_count = np.zeros(len(X), dtype=int)

        for s in spheres:
            d = np.linalg.norm(X - s.center, axis=1)
            inside = d <= s.radius

            inside_any |= inside
            inside_count += inside

        return inside_any, inside_count
    
    @staticmethod
    def scores(good_test_emb, defect_test_emb, spheres):
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

        auroc = roc_auc_score(y_true, scores)
        return auroc