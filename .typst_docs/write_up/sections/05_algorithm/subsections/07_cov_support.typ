=== Covaraince Support Estimation

Unlike hyperspheres, ellipsoids require a reliable covariance estimate to define their shape and orientation. When only a small number of embeddings are available, the covariance estimate becomes unstable, often producing highly elongated or degenerate ellipsoids. This is particularly problematic during the later stages of the covering algorithm, where only a few uncovered embeddings may remain.

To address this, the proposed algorithm introduces a covariance support mechanism. If the number of embeddings used to fit an ellipsoid falls below a predefined threshold, the covariance estimate is blended with that of a neighbouring ellipsoid. This provides a more stable estimate while preserving the local centroid of the candidate region. 

The current implementation selects the supporting ellipsoid based solely on Euclidean distance between ellipsoid centres. This assumes that nearby ellipsoids exhibit similar covariance structure, however this is not necessarily true. A more principled approach would identify the neighbouring ellipsoid that is most similar in shape, for example using eigenvalue ratios or principal axis alignment, rather than spatial proximity alone.

A further limitation is that repeated covariance borrowing can produce multiple ellipsoids with very similar covariance structure. In the extreme case, this may result in neighbouring ellipsoids becoming almost identical in shape. While this would be undesirable for applications requiring accurate local density estimation, it is less problematic for anomaly detection. The objective of the proposed algorithm is to partition the normal embedding space rather than recover its exact local geometry, and normal training embeddings are expected to exhibit broadly consistent covariance structure.

Let the covariance estimated from the current candidate neighbourhood be

$
C_("local")
=
frac(
sum_i w_i (x_i - mu)(x_i - mu)^T,
1 - sum_i w_i^2
).
$

The supporting ellipsoid is selected as the previously fitted ellipsoid whose centre is closest to the current candidate,

$
k^*
=
arg min_k
||mu - mu_k||_2.
$

Its covariance matrix is reconstructed from its eigendecomposition,

$
C_("support")
=
V_k Lambda_k V_k^T.
$

The final covariance estimate is obtained by blending the local and supporting covariance matrices,

$
C
=
alpha C_("local")
+
(1-alpha)C_("support"),
$

where

$
alpha
=
min(
1,
frac(n, n_("support"))
),
$

$n$ is the number of embeddings in the current candidate neighbourhood and $n_("support")$ is the minimum number of embeddings required to estimate a stable covariance matrix.