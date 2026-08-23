== Unsupervised Hypersphere Fitting Algorithm

This notebook implements and adapts ideas from _A classification method based on a cloud of spheres_ @cloudspheres. The original work proposes constructing spherical regions to define decision boundaries between classes. In this project, the method is adapted for unsupervised anomaly detection using MVTec AD and DINOv2 embeddings.

The original paper differs from this setting in three important ways:

- Supervised boundary construction: 
The The original algorithm constructs spheres using both positive and negative training samples. In the unsupervised anomaly detection setting, only normal training embeddings are available, meaning sphere boundaries must be estimated without any knowledge of defective samples. To address this limitation, candidate regions are initialised from the densest local K-nearest neighbour (KNN) neighbourhood and subsequently expanded using a constrained growth procedure.

- Connected manifold assumption:
The original method assumes that the data lie on a connected manifold and therefore encourages neighbouring spheres to form connected regions. This assumption is less appropriate for DINOv2 embeddings, where normal samples from a single object category may occupy multiple disconnected regions within the 768-dimensional embedding space. As defects are not observed, there is no prior knowledge of where anomalous regions exist. Consequently, the adapted algorithm does not enforce connectivity between neighbouring hyperspheres, allowing disconnected regions of normality to be represented independently.

- Exclusive embedding assignment:
The adapted algorithm enforces that each normal training embedding belongs to exactly one hypersphere. During the growth stage, candidate regions are iteratively cleaned to ensure that previously assigned embeddings are not incorporated into newly generated hyperspheres. While geometric overlap between hyperspheres is permitted, overlap in embedding ownership is not. This results in a unique ownership of normal training embeddings, while allowing neighbouring hyperspheres to share unoccupied regions of the embedding space.

