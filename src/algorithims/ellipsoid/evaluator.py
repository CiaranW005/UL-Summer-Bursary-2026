from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve

class EllipsoidEvaluator:
    @staticmethod
    def overlap(embeds, ellipsoids):
        overlaps = []

        for i, j in combinations(range(len(ellipsoids)), 2):
            e1 = ellipsoids[i]
            e2 = ellipsoids[j]

            # Points owned by j inside ellipsoid i
            diff = embeds[e2.covered_idx] - e1.center
            proj = diff @ e1.eigvecs
            d2 = np.sum((proj ** 2) / e1.eigvals, axis=1)
            j_inside_i = np.sum(d2 <= e1.threshold)

            # Points owned by i inside ellipsoid j
            diff = embeds[e1.covered_idx] - e2.center
            proj = diff @ e2.eigvecs
            d2 = np.sum((proj ** 2) / e2.eigvals, axis=1)
            i_inside_j = np.sum(d2 <= e2.threshold)

            overlaps.append({
                "ellipsoid_i": i,
                "ellipsoid_j": j,
                "j_points_inside_i": j_inside_i,
                "i_points_inside_j": i_inside_j,
                "overlap": (j_inside_i + i_inside_j) > 0
            })

        overlap_df = pd.DataFrame(overlaps)
        num_overlaps = overlap_df["overlap"].sum()

        return overlap_df, num_overlaps

    @staticmethod
    def inside_any_count(X, ellipsoids):
        inside_any = np.zeros(len(X), dtype=bool)
        inside_count = np.zeros(len(X), dtype=int)

        for e in ellipsoids:
            diff = X - e.center
            proj = diff @ e.eigvecs
            d2 = np.sum((proj ** 2) / e.eigvals, axis=1)

            inside = d2 <= e.threshold

            inside_any |= inside
            inside_count += inside

        return inside_any, inside_count
    
    @staticmethod
    def score_samples(X, ellipsoids):
        margins = np.full((len(X), len(ellipsoids)), np.inf)

        for i, e in enumerate(ellipsoids):
            diff = X - e.center
            proj = diff @ e.eigvecs
            d2 = np.sum((proj ** 2) / e.eigvals, axis=1)
            margins[:, i] = d2 - e.threshold

        best_ellipsoid = margins.argmin(axis=1)
        scores = margins.min(axis=1)

        return scores, best_ellipsoid
    
    def evaluate_detection(self, good_test_emb, defect_test_emb, ellipsoids):
        X = np.vstack([good_test_emb, defect_test_emb])

        y_true = np.concatenate([
            np.zeros(len(good_test_emb)),
            np.ones(len(defect_test_emb)),
        ])

        scores, best_ellipsoid = self.score_samples(X, ellipsoids)

        auroc = roc_auc_score(y_true, scores)

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        best_threshold = thresholds[(tpr - fpr).argmax()]

        predicted_label = (scores > best_threshold).astype(int)
        correct = predicted_label == y_true

        n_points = np.array([len(e.covered_idx) for e in ellipsoids])
        eig_ratios = np.array([e.eig_ratio for e in ellipsoids])

        results_df = pd.DataFrame({
            "y_true": y_true,
            "score": scores,
            "correct": correct,
            "winning_ellipsoid": best_ellipsoid,
            "winning_n_points": n_points[best_ellipsoid],
            "winning_eigval_ratio": eig_ratios[best_ellipsoid],
        })

        metrics = {
            "auroc": auroc,
            "best_threshold": best_threshold,
        }

        return results_df, metrics
    
    @staticmethod
    def bucket_diagnostics(results_df):
        results_df = results_df.copy()

        results_df["n_points_bucket"] = pd.cut(
            results_df["winning_n_points"],
            bins=[0, 3, 5, 10, 100],
        )

        results_df["eigval_ratio_bucket"] = pd.qcut(
            results_df["winning_eigval_ratio"],
            q=4,
            duplicates="drop",
        )

        return {
            "n_points": results_df.groupby("n_points_bucket")["correct"].agg(["mean", "count"]),
            "eig_ratio": results_df.groupby("eigval_ratio_bucket")["correct"].agg(["mean", "count"]),
            "n_points_by_class": results_df.groupby(["n_points_bucket", "y_true"])["correct"].agg(["mean", "count"]),
        }


