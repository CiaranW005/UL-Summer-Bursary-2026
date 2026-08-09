import numpy as np
import faiss 
import pandas as pd 

from pathlib import Path

from dataclasses import asdict

from ..types import EllipsoidCollection
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
        )-> EllipsoidCollection:
        """Construct ellipsoids until every embedding is covered.

        Each ellipsoid starts from the most locally compact uncovered
        neighbourhood, is cleaned to avoid encroaching on already covered
        points, and is then expanded while it gains additional points.
        """
        uncovered_mask = np.ones(len(embeds), dtype=bool)
        collection = EllipsoidCollection()
        
        while uncovered_mask.any():
            # Work only with currently uncovered embeddings
            uncovered_idx = np.where(uncovered_mask)[0]
            uncovered_emb = embeds[uncovered_idx].astype("float32")

            k = max(2, int(k_frac * len(uncovered_idx)))
            k = min(k, len(uncovered_idx) - 1)

            growth = max(min_growth, start_growth - 0.0025 * len(collection))
            
            if k < 1:
                covered_idx = uncovered_idx

                X = embeds[covered_idx]
                weights = np.ones(len(covered_idx))

                if len(collection) > 0:
                    fit = self.fitter.fit_supported(X, weights, collection.ellipsoids)
                else:
                    fit = self.fitter.fit(X, weights)

                fit.set_covered_idx(covered_idx)

            else:
                index = faiss.IndexFlatL2(uncovered_emb.shape[1])
                index.add(uncovered_emb)

                dist, nbrs = index.search(uncovered_emb, k=k+1) # +1 as nearest is itself 

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
                fit = self.cleaner.clean_candidate(
                    full_covered_idx,
                    embeds,
                    uncovered_mask,
                    weights,
                    collection.ellipsoids
                )

                while True:
                    candidate_mask = self.fitter.grow(embeds, fit.ellipsoid, growth)
                    full_new_covered_idx = np.where(candidate_mask & uncovered_mask)[0]

                    if len(full_new_covered_idx) == 0:
                        break

                    # Preserve previously reduced wieghts for points retained during growth
                    _, old_pos, new_pos = np.intersect1d(
                        fit.ellipsoid.covered_idx,
                        full_new_covered_idx,
                        return_indices=True
                    )

                    grow_weights = np.ones(len(full_new_covered_idx))
                    grow_weights[new_pos] = fit.ellipsoid.weights[old_pos]

                    weights = grow_weights

                    # Do not allow cleaning to shrink a grown candidate below its previous size.
                    self.cleaner.min_points = fit.stats.n_points
                    new_fit = self.cleaner.clean_candidate(
                        full_new_covered_idx, 
                        embeds, 
                        uncovered_mask, 
                        weights,
                        collection.ellipsoids
                        ) 

                    if new_fit.stats.n_points > fit.stats.n_points:
                        fit = new_fit
                    else:
                        break

            collection.append(fit)
            uncovered_mask[fit.ellipsoid.covered_idx] = False

        if output_dir is not None:
            collection.to_dataframe().to_csv(output_dir / "ellipsoids.csv", index=False)
        
        return collection