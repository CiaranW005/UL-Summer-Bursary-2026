import numpy as np
import faiss 
import pandas as pd 

class EllipsoidCover:
    def __init__(self, fitter, cleaner):
        self.fitter = fitter
        self.cleaner = cleaner

    def run(self, embeds, output_dir=None, k_frac=0.05, start_growth=1.2, min_growth=1.0):
        uncovered_mask = np.ones(len(embeds), dtype=bool)
        ellipsoids = []

        while uncovered_mask.any():
            uncovered_idx = np.where(uncovered_mask)[0]
            uncovered_emb = embeds[uncovered_idx].astype("float32")

            K = max(2, int(k_frac * len(uncovered_idx)))
            K = min(K, len(uncovered_idx) - 1)

            growth = max(min_growth, start_growth - 0.0025 * len(ellipsoids))
            
            if K < 1:
                covered_idx = uncovered_idx

                X = embeds[covered_idx]
                weights = np.ones(len(covered_idx))

                center, eigvecs, eigvals, threshold, eig_ratio, ellipse_id = self.fitter.fit_supported(X, ellipsoids, weights)

            else:
                index = faiss.IndexFlatL2(uncovered_emb.shape[1])
                index.add(uncovered_emb)

                D, I = index.search(uncovered_emb.astype("float32"), k=K+1) # +1 as nearest is itself 

                D = D[:, 1:]    # Remove self
                I = I[:, 1:]

                avg_knn = D.mean(axis=1)

                local_compact  = avg_knn.argmin()
                neighbours_local = I[local_compact]

                full_covered_idx = uncovered_idx[np.r_[local_compact, neighbours_local]]

                weights = np.ones(len(full_covered_idx))
                covered_idx, center, eigvecs, eigvals, threshold, eig_ratio, weights, ellipse_id = self.cleaner.clean_candidate(
                    full_covered_idx,
                    embeds,
                    uncovered_mask,
                    weights,
                    ellipsoids
                )

                while True:
                    old_covered_idx = covered_idx.copy()
                    
                    candidate_mask = self.fitter.grow(embeds, center, eigvecs, eigvals, threshold, growth)
                    full_new_covered_idx = np.where(candidate_mask & uncovered_mask)[0]

                    grow_weights = np.ones(len(full_new_covered_idx))
                    for i, idx in enumerate(full_new_covered_idx):
                        if idx in covered_idx:
                            old_pos = np.where(covered_idx == idx)[0][0]
                            grow_weights[i] = weights[old_pos]
                    weights = grow_weights

                    new_covered_idx, new_center, new_eigvecs, new_eigvals, new_threshold, new_eig_ratio, new_weights, new_ellipse_id = self.cleaner.clean_candidate(
                        full_new_covered_idx, 
                        embeds, 
                        uncovered_mask, 
                        weights,
                        ellipsoids,
                        min_points=len(old_covered_idx)
                        ) 

                    if len(new_covered_idx) > len(old_covered_idx):
                        covered_idx = new_covered_idx
                        center = new_center
                        eigvecs = new_eigvecs
                        eigvals = new_eigvals
                        threshold = new_threshold
                        eig_ratio = new_eig_ratio
                        weights = new_weights
                        ellipse_id = new_ellipse_id
                    else:
                        break

            ellipsoids.append({
                "center": center,
                "eigvecs": eigvecs,
                "eigvals": eigvals,
                "threshold": threshold,
                "eig_ratio": eig_ratio,
                "covered_idx": covered_idx,
                "weights": weights,
                "ellipse_id": ellipse_id
            })

            uncovered_mask[covered_idx] = False

        ellipsoids_df = pd.DataFrame([
            {
                "n_points": len(e["covered_idx"]),
                "threshold": e["threshold"],
                "eig_ratio": e["eig_ratio"],
                "weights": e["weights"]
            }
            for e in ellipsoids
        ])

        if output_dir is not None:
            ellipsoids_df.to_csv(output_dir / f"ellipsoids.csv", index=False)
        
        return ellipsoids, ellipsoids_df