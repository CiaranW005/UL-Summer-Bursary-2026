import numpy as np

from .hypersphere import Hypersphere

class CandidateCleaner:
    @staticmethod
    def clean_candidate(covered_idx, cat_emb, uncovered_mask, min_points=1):
        """
        Iteratively emoved embeddings form a candidate hypersphere until it statisifes the
        exclusive embedding embedding assignment constraint

        Starting from a candiate set of embeddings, the function repeatedly computed the
        centroid and radius of the hyphersphere. If the resulting hyphersphere contains embeddings that 
        have been assinged to previous hyperspheres, the embedding contributing most to the current radius
        is removed and the hypersphere is recomputed. This process continues until no previously assinged 
        embedings lie within the hypersphere or onbly a single embedding remains

        Parameters
        ----------
        covered_idx : ndarray
            Indices of embeddings current assigned to the candidate hypersphere

        cat_emb : ndarray
            Embedding matrix of the current object category

        uncovered_mask : ndarray
            Boolean mask indicating which embeddings have not yet been assigned
            to a hypersphere

        min_points : int
            The minimum anoubt of points a sphere should keep. Used to ensure previously
            accepted candidates are not rejected.
            
        Returns
        -------
        covered_idx : ndarray
            Indices of the cleaned hypersphere

        centroid : ndarray
            Centroid of the final hypersphere
        
        radius : float
            Radius of the final hypersphere. (Singleton spheres have radius 0)
        """

        while len(covered_idx) > min_points:
            centroid = cat_emb[covered_idx].mean(axis=0)

            d_self = np.linalg.norm(cat_emb[covered_idx] - centroid, axis=1)
            radius = d_self.max()

            d_all = np.linalg.norm(cat_emb - centroid, axis=1)
            inside_all = np.where(d_all <= radius)[0]

            shared_count = np.sum(~uncovered_mask[inside_all])

            if shared_count == 0:
                return Hypersphere(
                    center=centroid,
                    radius=radius,
                    covered_idx=covered_idx
                )

            # remove point causing the current sphere to stretch the most
            worst_local = d_self.argmax()
            covered_idx = np.delete(covered_idx, worst_local)

        centroid = cat_emb[covered_idx].mean(axis=0)
        radius = 0.0 if len(covered_idx) == 1 else np.linalg.norm(cat_emb[covered_idx] - centroid, axis=1).max()

        return Hypersphere(
            center=centroid,
            radius=radius,
            covered_idx=covered_idx
        )