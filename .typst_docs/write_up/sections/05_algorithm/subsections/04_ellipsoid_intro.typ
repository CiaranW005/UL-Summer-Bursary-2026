== Unsupervised Ellipsoid Fitting Algorithm

This notebook extends the hypersphere-based algorithm by replacing each spherical region with an ellipsoid. The motivation is that normal embeddings in DINOv2 feature space are unlikely to form locally isotropic clusters. Instead, neighbourhoods may stretch more strongly along some directions than others. Ellipsoids are therefore able to model local variance more naturally than hyperspheres.

The ellipsoid formulation has several advantages:

- Aligns each region with the natural variance structure of the local KNN neighbourhood.
- Reduces unused empty space compared with hyperspheres, since the boundary can contract along low-variance directions.
- It can reduce unnecessary overlap between neighbouring regions by following the dominant principal axes of the local embedding distribution.
- It is better suited to high-variance categories, where the normal embedding space may contain elongated or anisotropic regions.
- Provides additional interpretability through eigenvalues, eigenvectors, axis ratios, and local region structure.

Several changes were introduced compared with the hypersphere version:

- Growth is variance-scaled rather than uniform. Expansion along each axis is controlled by the relative eigenvalue contribution, so high-variance directions can grow more than low-variance directions.
- Candidate cleaning is weight-based. Instead of immediately removing a point when a candidate ellipsoid overlaps a previous region, the algorithm first reduces that point's contribution to the ellipsoid  fit. If its weight reaches zero and overlap remains, the point is removed.
- Sparse ellipsoids require additional support. Unlike hyperspheres, ellipsoids fitted from very few points can become geometrically unstable. To address this, the covariance of a small candidate region is blended with covariance information from a previous ellipsoid.
