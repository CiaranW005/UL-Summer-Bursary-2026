= Related Work

Two established anomaly-detection approaches evaluated on MVTec AD are PaDiM @padim and PatchCore @patchcore. Both use pretrained convolutional neural networks (CNNs) to extract representations from normal training images, but differ in how the resulting normal feature space is modelled.

Mahalanobis distance provides a simple distribution-based approach to anomaly detection by measuring the distance of a sample from a reference distribution while accounting for feature covariance @mahalanobis. It is particularly relevant to this work because it provides the baseline scoring method for the DINOv2 embeddings and is closely related to the covariance-aware geometry used by the proposed ellipsoidal regions. Whereas a global Mahalanobis model represents normality using a single covariance structure, the proposed approach instead models multiple local regions.

== Representation-based Anomaly Detection

PaDiM models normal features independently at each spatial position using multivariate Gaussian distributions. Features from multiple levels of a pretrained CNN are combined, and at inference the Mahalanobis distance between each test feature and its corresponding normal distribution is used to identify anomalous regions. To reduce computational cost, PaDiM also applies random feature selection, with the authors showing that strong anomaly-detection performance can be retained using a reduced representation. However, retaining similar AUROC does not establish which properties of the original representation have been preserved or discarded, which is relevant when interpretability of the embedding geometry is also of interest.

PatchCore instead represents normality using a memory bank of local features extracted from intermediate layers of a pretrained CNN. Test patches are compared with nearby normal representations in this memory, with greater distance indicating a more anomalous feature. Unlike PaDiM, PatchCore therefore retains representative normal examples rather than explicitly fitting a probability distribution.

PaDiM and PatchCore were not reimplemented here as this work focuses on embedding-space geometry rather than patch-level localisation; comparison is limited to Mahalanobis as the most directly related global baseline.

== Geometric Modelling with Cloud of Spheres

The original _Cloud of Spheres_ algorithm @cloudspheres represents complex binary class boundaries using multiple spherical regions rather than a single global decision region. It is formulated as a supervised optimisation problem in which positive and negative training samples constrain the construction of the cloud, while the number of required spheres is minimised. The resulting geometric representation allows nonlinear and non-convex class structures to be described using a collection of simpler local regions.

Several assumptions of the original formulation do not directly transfer to unsupervised anomaly detection. _Cloud of Spheres_ uses examples from both classes to constrain its decision boundary, whereas only normal training samples are available in the setting considered here. The location and structure of anomalous regions are therefore unknown during construction of the normal representation.

The original method also assumes that samples belonging to a class form a connected manifold and consequently encourages neighbouring spheres to form a connected cloud. In an unsupervised anomaly-detection setting, however, only normal samples are available, providing no information about the location or structure of anomalous regions between observed areas of normality. Enforcing connectivity may therefore incorrectly treat unsupported regions of the embedding space as normal simply because they lie between separate normal clusters. This is particularly restrictive for learned visual representations, where normal samples may occupy multiple distinct regions of a high-dimensional embedding space. In addition, spherical regions assume approximately isotropic local structure and cannot adapt their orientation to directional variation within the representation.

These limitations motivate the extension investigated in this work: an unsupervised multi-region model that does not require connectivity between normal regions and replaces spherical boundaries with covariance-adaptive ellipsoids.