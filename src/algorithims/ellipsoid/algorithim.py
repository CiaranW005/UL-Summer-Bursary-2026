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

                if len(ellipsoids) > 0:
                    ellipsoid = self.fitter.fit_supported(X, weights, ellipsoids)
                else:
                    ellipsoid = self.fitter.fit(X, weights)

                ellipsoid.covered_idx = covered_idx

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
                ellipsoid = self.cleaner.clean_candidate(
                    full_covered_idx,
                    embeds,
                    uncovered_mask,
                    weights,
                    ellipsoids
                )

                while True:
                    candidate_mask = self.fitter.grow(embeds, ellipsoid, growth)
                    full_new_covered_idx = np.where(candidate_mask & uncovered_mask)[0]

                    if len(full_new_covered_idx) == 0:
                        break

                    _, old_pos, new_pos = np.intersect1d(
                        ellipsoid.covered_idx,
                        full_new_covered_idx,
                        return_indices=True
                    )
                    grow_weights = np.ones(len(full_new_covered_idx))
                    grow_weights[new_pos] = ellipsoid.weights[old_pos]

                    weights = grow_weights

                    self.cleaner.min_points = len(ellipsoid.covered_idx)
                    new_ellipsoid = self.cleaner.clean_candidate(
                        full_new_covered_idx, 
                        embeds, 
                        uncovered_mask, 
                        weights,
                        ellipsoids
                        ) 

                    if len(new_ellipsoid.covered_idx) > len(ellipsoid.covered_idx):
                        ellipsoid = new_ellipsoid
                    else:
                        break

            ellipsoids.append(ellipsoid)
            uncovered_mask[ellipsoid.covered_idx] = False

        ellipsoids_df = pd.DataFrame([
            {
                "n_points": len(e.covered_idx),
                "threshold": e.threshold,
                "eig_ratio": e.eig_ratio,
                "weights": e.weights
            }
            for e in ellipsoids
        ])

        if output_dir is not None:
            ellipsoids_df.to_csv(output_dir / "ellipsoids.csv", index=False)
        
        return ellipsoids, ellipsoids_df