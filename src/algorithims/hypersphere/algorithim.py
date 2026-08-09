import numpy as np
import pandas as pd
import faiss

from pathlib import Path 

from .cleaner import CandidateCleaner
from ..types import Hypersphere

class HypersphereCover:
    def __init__(self, cleaner: CandidateCleaner):
        self.cleaner = cleaner

    def run(self, 
            embeds: np.ndarray, 
            output_dir: Path | None = None, 
            k_frac: float = 0.05, 
            start_growth: float = 1.05, 
            min_growth: float = 1.0
            ) -> tuple[list[Hypersphere], pd.DataFrame]:
        """Cover all embeddings with locally fitted hyperspheres
        
        Each sphere begins from the most compact uncoverd neighbourhood.
        The candidate is cleaned to avoid covering previously assigned points then,
        if it has not been shrunk is then grown to add additional uncovered points
        """
        uncovered_mask = np.ones(len(embeds), dtype=bool)
        spheres: list[Hypersphere] = []

        while uncovered_mask.any():
            # Work with only uncovered embeddings
            uncovered_idx = np.where(uncovered_mask)[0]
            uncovered_emb = embeds[uncovered_idx].astype("float32")

            k = max(2, int(k_frac * len(uncovered_idx)))
            k = min(k, len(uncovered_idx) - 1)

            growth = max(min_growth, start_growth - 0.0025 * len(spheres))
            if k < 1:
                covered_idx = uncovered_idx
                hypersphere = Hypersphere(
                    center=embeds[covered_idx].mean(axis=0),
                    radius=0.0,
                    covered_idx=covered_idx
                    )
            
            else:
                index = faiss.IndexFlatL2(uncovered_emb.shape[1])
                index.add(uncovered_emb)

                dists, nbrs = index.search(uncovered_emb.astype("float32"), k=k+1) # +1 as nearest is itself 

                dists = dists[:, 1:]    # Remove self
                nbrs = nbrs[:, 1:]

                # Selects the embedding with most compact neighbourhood
                avg_knn = dists.mean(axis=1)

                local_compact  = avg_knn.argmin()
                nbrs_local = nbrs[local_compact]

                # Finds the original indices of the points
                full_covered_idx = uncovered_idx[np.r_[local_compact, nbrs_local]]

                hypersphere = self.cleaner.clean_candidate(
                    full_covered_idx,
                    embeds,
                    uncovered_mask
                )

                # Only grow candidates if it was not shrunk
                if len(full_covered_idx) == len(hypersphere.covered_idx):
                    while True:
                        search_radius = hypersphere.radius * growth
                        distances = np.linalg.norm(embeds - hypersphere.center, axis=1)

                        candidate_idx = np.where(distances <= search_radius)[0]
                        full_new_covered_idx = candidate_idx[uncovered_mask[candidate_idx]]
                        
                        new_hypersphere = self.cleaner.clean_candidate(full_new_covered_idx, embeds, uncovered_mask, min_points=len(hypersphere.covered_idx)) 

                        # Accept the new sphere if it covers additional points
                        if len(new_hypersphere.covered_idx) > len(hypersphere.covered_idx):
                            hypersphere = new_hypersphere
                        else:
                            break

            spheres.append(hypersphere)
            uncovered_mask[hypersphere.covered_idx] = False

        spheres_df = pd.DataFrame([
            {
                "n_points": len(s.covered_idx),
                "radius": s.radius
            }
            for s in spheres
        ])

        if output_dir is not None:
            spheres_df.to_csv(output_dir / "hyperspheres.csv", index=False)

        return spheres, spheres_df
