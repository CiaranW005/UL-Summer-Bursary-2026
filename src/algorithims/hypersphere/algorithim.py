import numpy as np
import pandas as pd
import faiss

from ...types.hypersphere import Hypersphere

class HypersphereCover:
    def __init__(self, cleaner):
        self.cleaner = cleaner

    def run(self, embeds, output_dir=None, k_frac=0.05, start_growth=1.05, min_growth=1.0):
        uncovered_mask = np.ones(len(embeds), dtype=bool)
        spheres = []

        while uncovered_mask.any():
            uncovered_idx = np.where(uncovered_mask)[0]
            uncovered_emb = embeds[uncovered_idx].astype("float32")

            K = max(2, int(k_frac * len(uncovered_idx)))
            K = min(K, len(uncovered_idx) - 1)

            growth = max(min_growth, start_growth - 0.0025 * len(spheres))
            if K < 1:
                covered_idx = uncovered_idx
                hypersphere = Hypersphere(
                    center=embeds[covered_idx].mean(axis=0),
                    radius=0.0,
                    covered_idx=covered_idx
                    )
            
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

                hypersphere = self.cleaner.clean_candidate(
                    full_covered_idx,
                    embeds,
                    uncovered_mask
                )

                if len(full_covered_idx) == len(hypersphere.covered_idx):
                    while True:
                        search_radius = hypersphere.radius * growth
                        distances = np.linalg.norm(embeds - hypersphere.center, axis=1)

                        candidate_idx = np.where(distances <= search_radius)[0]
                        full_new_covered_idx = candidate_idx[uncovered_mask[candidate_idx]]
                        
                        new_hypersphere = self.cleaner.clean_candidate(full_new_covered_idx, embeds, uncovered_mask, min_points=len(hypersphere.covered_idx)) 

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
            spheres_df.to_csv(output_dir / "hypersphere.csv", index=False)

        return spheres, spheres_df
