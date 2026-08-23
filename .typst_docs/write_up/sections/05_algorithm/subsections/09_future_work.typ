== Future Work

The first direction for future work is to address the limitations identified throughout this work. In particular, the covariance support estimation could be improved by selecting supporting ellipsoids based on both spatial proximity and covariance similarity rather than centre distance alone.
 
Similarly, adaptive growth strategies should be investigated to allow the growth factor to vary according to the embedding geometry of each category instead of using a single global parameter.

Candidate cleaning could also be extended by reducing the weights of multiple embeddings rather than a single embedding at each iteration. In principle, this would allow the algorithm to modify the covariance structure more smoothly by reducing the influence of all embeddings contributing to the offending principal axis. However, due to the sparsity of the current embedding space, candidate ellipsoids often contain very few embeddings, making such an approach difficult to estimate reliably. If future embedding spaces become more compact and contain larger local neighbourhoods, this strategy may become a more effective alternative.

A more significant direction for future work is the joint optimisation of the embedding space and the proposed statistical modelling algorithm. The current implementation operates on frozen DINOv2 embeddings, meaning that the embedding space has not been learned with the proposed ellipsoid fitting algorithm in mind. Most existing anomaly detection methods optimise representations by encouraging all normal samples to form a single compact cluster. However, this objective may not be optimal for local statistical modelling.

Instead, the representation learning objective could be designed to preserve the natural local covariance structure of the embedding space while encouraging normal embeddings to organise into compact, well-separated local neighbourhoods. Such an embedding space would be better suited to the proposed hypersphere and ellipsoid algorithms, producing more stable covariance estimates, reducing unnecessary overlap between neighbouring regions, and ultimately allowing the representation learning process and the statistical modelling algorithm to be optimised jointly rather than independently.

