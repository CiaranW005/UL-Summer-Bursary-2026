import numpy as np
import faiss 
import pandas as pd 

from pathlib import Path

from ..types import Ellipsoid
from .fitter import EllipsoidFitter
from .cleaner import CandidateCleaner

class EllipsoidCover:
    def __init__(self, fitter: EllipsoidFitter, cleaner: CandidateCleaner):
        self.fitter = fitter
        self.cleaner = cleaner

    def run(self, 
            embeds: np.ndarray, 
            output_dir: Path | None = None, 
            k_frac: float = 0.05, 
            start_growth: float = 1.2, 
            min_growth: float = 1.0
        )-> tuple[list[Ellipsoid], pd.DataFrame]:
        """Construct ellipsoids until every embedding is covered.

        Each ellipsoid starts from the most locally compact uncovered
        neighbourhood, is cleaned to avoid encroaching on already covered
        points, and is then expanded while it gains additional points.
        """

        uncovered_mask = np.ones(len(embeds), dtype=bool)
        ellipsoids: list[Ellipsoid] = []

        while uncovered_mask.any():
            # Work only with currently uncovered embeddings
            uncovered_idx = np.where(uncovered_mask)[0]
            uncovered_emb = embeds[uncovered_idx].astype("float32")

            k = max(2, int(k_frac * len(uncovered_idx)))
            k = min(k, len(uncovered_idx) - 1)

            growth = max(min_growth, start_growth - 0.0025 * len(ellipsoids))
            
            if k < 1:
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

                dist, nbrs = index.search(uncovered_emb.astype("float32"), k=k+1) # +1 as nearest is itself 

                dist = dist[:, 1:]    # Remove self
                nbrs = nbrs[:, 1:]

                # Choose the most locally compact uncovered point 
                # TODO: Possible change to a better sampling so it tries to create an even amount of ellipsoids compared to a greedy approach
                avg_knn = dist.mean(axis=1)
                local_compact  = avg_knn.argmin()
                nbrs_local = nbrs[local_compact]

                # Map the selected uncovered points back to their original embedding indices.
                full_covered_idx = uncovered_idx[np.r_[local_compact, nbrs_local]]

                weights = np.ones(len(full_covered_idx))
                self.cleaner.min_points = 1
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

                    # Preserve previously reduced wieghts for points retained during growth
                    _, old_pos, new_pos = np.intersect1d(
                        ellipsoid.covered_idx,
                        full_new_covered_idx,
                        return_indices=True
                    )
                    grow_weights = np.ones(len(full_new_covered_idx))
                    grow_weights[new_pos] = ellipsoid.weights[old_pos]

                    weights = grow_weights

                    # Do not allow cleaning to shrink a grown candidate below its previous size.
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
    