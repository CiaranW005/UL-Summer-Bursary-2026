=== Candidate cleaning

In the hypersphere algorithm, candidate cleaning was performed by iteratively removing the point that determined the current sphere radius. Since the radius is defined by the furthest point from the centroid, removing this point always reduces the size of the sphere and therefore the likelihood of overlap with previously assigned regions.

This strategy does not naturally extend to ellipsoids. The size and orientation of an ellipsoid are jointly determined by its covariance matrix rather than by a single boundary point. Consequently, removing the embedding with the largest Mahalanobis distance does not necessarily reduce overlap with previously assigned ellipsoids, as that embedding may contribute little to the principal direction responsible for the overlap. Repeated deletion can therefore remove many embeddings before a satisfactory solution is obtained.

To address this, candidate cleaning is rebuilt as a weighted covariance estimation problem. Rather than immediately removing an embedding, the algorithm first identifies the principal axis contributing most to the overlap and progressively reduces the contribution of the corresponding embedding to the covariance matrix. The embedding weight is updated using a binary search until either a valid ellipsoid is obtained or the weight reaches zero, at which point the embedding is removed from the candidate and the covariance matrix is recomputed.

This approach attempts to preserve as much local neighbourhood information as possible before permanently discarding an embedding. However, the current implementation has an important limitation. An embedding whose weight has been reduced to zero may still lie geometrically inside the resulting ellipsoid despite no longer contributing to its covariance estimate. Although this embedding is no longer considered part of the fitted neighbourhood, it may still be enclosed by the final ellipsoid, suggesting that future work should jointly optimise covariance estimation and geometric membership.

Let the candidate neighbourhood be

$ cal(X) = {(x_i, w_i)}_(i=1)^n $

where $w_i in [0,1]$ is the weight associated with embedding $x_i$.

The weighted centroid is given by

$ mu = sum^n_i w_i x_i $

The weighted covariance matrix is then estimated as

$ C = frac(sum_i w_i (x_i - mu)(x_i - mu)^T, 1-sum_i w_i^2). $

After eigendecomposition,

$ C = V Lambda V^T, $

the ellipsoid is constructed from the eigenvectors $V$ and eigenvalues $Lambda$.

If a previously assigned embedding lies within the candidate ellipsoid, an encroachment has occurred. For an encroaching embedding $x_s$, the contribution of each principal axis is computed as

$
c_i(x_s) =
frac((v_i^T (x_s - mu))^2,
lambda_i).
$

The principal axis responsible for the greatest contribution is

$
i^* = arg max_i c_i(x_s).
$

The candidate embedding contributing most strongly along this axis is then identified by

$
j^* =
arg max_j
frac((v_(i^*)^T (x_j - mu))^2,
lambda_(i^*)).
$

Rather than immediately removing this embedding, its weight is reduced using a binary search,

$
w_(j^*) in [0, w_(j^*)].
$

The covariance matrix is recomputed after each update until either no encroachment remains or the weight reaches zero. If the weight becomes zero and overlap still exists, the embedding is permanently removed from the candidate neighbourhood and the ellipsoid is refitted.


