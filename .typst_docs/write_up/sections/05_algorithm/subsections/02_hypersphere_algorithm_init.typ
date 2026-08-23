=== How does Mahlanobis fit to the data

This section visualises how a global Mahalanobis ellipsoid fits the normal training embeddings and uses it as a baseline for comparison with the proposed hypersphere fitting algorithm.

The objective is not to evaluate the performance of Mahalanobis, but rather to understand the assumptions it makes about the embedding space. In particular, Mahalanobis models the normal distribution as a single ellipsoidal region defined by the global centroid and covariance matrix. The following visualisations illustrate how well this assumption represents the underlying embedding distribution before introducing the proposed multi-region fitting approach.

#figure(
  image("../../../images/ellipsoid/base_embeds/mahlanobis_fit/screw.svg", width: 105%),
  caption: [Mahalanobis fit of the screw category in PCA]
)

For the screw category, it can be seen that the Mahalanobis ellipsoid captures the dominant variance of the normal training embeddings along the first two principal components. The orientation of the ellipsoid reflects the covariance structure of the data, demonstrating how Mahalanobis adapts to anisotropic variance rather than assuming a spherical distribution. However, the figure also shows considerable overlap between the projected normal and defective embeddings, illustrating why this category remains challenging for a single global statistical model and contributing to the observed AUROC of 0.802.

=== Choosing a Value for K

The first step of the proposed algorithm is identifying the densest local region of the normal embedding space. This is achieved using a K-nearest neighbours (KNN) search, where the embedding with the smallest average distance to its neighbours is selected as the initial sphere centre.

Selecting an appropriate value for K is important. If K is too small, the initial neighbourhood becomes overly sensitive to local noise and may fail to capture a representative region. Conversely, if K is too large, the neighbourhood becomes overly broad, increasing the likelihood of creating a hypersphere that encompasses a significant proportion of the embedding space.

As each MVTec AD category contains a different number of normal training samples, selecting a fixed value of K would not scale consistently across categories. Instead, K is defined as a percentage of the available training embeddings. Three candidate values (2.5%, 5%, and 10%) are visually compared to determine a suitable compromise between capturing a dense local neighbourhood while avoiding an overly large initial region.

#figure(
  image("../../../images/ellipsoid/base_embeds/k_rate/screw.svg", width: 110%),
  caption: [K values for each percentage of the screw category]
)

The value in the end selected was 5% of the available training embeddings. Visually, this provided the best balance between capturing a representative local neighbourhood while avoiding an initial hypersphere that encompassed an excessively large region of the embedding space.

Although this percentage is unlikely to be optimal for every category, as the structure of the embedding space varies between object classes, it provides a consistent and scalable initialisation strategy across the MVTec AD dataset. Furthermore, the subsequent growth stage allows the hypersphere to adapt to the local embedding distribution, reducing the sensitivity of the overall algorithm to the initial choice of K.

=== Growth Rate for the Hyperspheres

Although the initial KNN neighbourhood provides a suitable starting point, the choice of K remains arbitrary and should not solely determine the size of a hypersphere. To reduce the dependence on this initial neighbourhood, a growth stage is introduced.

Starting from the initial KNN region, the hypersphere is iteratively expanded by a small growth factor. After each expansion, any newly enclosed embeddings are incorporated into the region and the hypersphere is recalculated using the updated centroid and radius. This process continues until no additional embeddings can be added.

The purpose of the growth stage is to allow each hypersphere to adapt naturally to the local structure of the embedding space rather than being constrained by the initial KNN selection. In particular, it reduces the likelihood of producing a large number of singleton or very small hyperspheres, allowing neighbouring embeddings that belong to the same local region to be grouped together.

#figure(
  image("../../../images/ellipsoid/base_embeds/growth_rate/screw.svg", width: 80%),
  caption: [Various growth rates for the screw category]
)

A growth rate of 5% was selected as the most suitable value. This provided enough expansion to include additional nearby embeddings from the same local neighbourhood, while avoiding excessive growth. A smaller growth rate of 2.5% produced little change from the initial KNN region, whereas a larger growth rate of 10% caused the hypersphere to expand too broadly, creating a catch-all region. This behaviour would reduce the benefit of the proposed method, as the resulting hypersphere would begin to resemble a global centroid-based model rather than a local region-based representation.

