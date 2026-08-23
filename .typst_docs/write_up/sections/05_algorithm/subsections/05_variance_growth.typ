=== Varaince-Scaled Growth

In the hypersphere algorithm, uniform growth was appropriate because the radius is identical in every dimension. Expanding the hypersphere therefore preserved its geometry regardless of direction. This assumption no longer holds for ellipsoids, where each principal axis represents a different amount of local variance. Local neighbourhoods in the DINOv2 embedding space are often anisotropic, meaning that the variance differs substantially between principal axes. Uniformly scaling every axis would therefore assume that each principal component contributes equally to the local structure of the embedding space. In practice, this can cause low-variance directions to expand excessively, increasing the amount of empty embedding space enclosed by the ellipsoid and the likelihood of overlap with neighbouring regions.

To address this, the proposed algorithm scales growth according to the variance explained by each principal axis. Consequently, axes with greater variance receive proportionally more growth, while lower-variance axes expand more conservatively.

Let the eigenvalues of the covariance matrix be $lambda_i$, where $lambda_max$ is the largest eigenvalue. The relative variance contribution of axis $i$ is defined as

$ r_i = max(frac(lambda_i, lambda_max), r_min) $

A minimum variance ratio r
min is introduced to ensure that every principal axis receives a small amount of growth, preventing axes associated with very small eigenvalues from becoming numerically unstable.

The growth factor for each axis is then:

$ a_i = 1 + g r_i $

where $g$ is the global growth parameter

The grown eigenvalue is:

$ lambda_i' = lambda_i a_i^2 $


A point $x$ is inside the grown ellipsoid if:

$ sum_i ((v_i^T (x - mu))^2) / lambda_i' <= tau $

where: 
- $mu$ = ellipsoid centre   
- $v_i$ = eigenvector for axis $i$  #h(2em) $r_min$ = miniumum vaiance ratio   
- $lambda_i$ = eigenvalue for axis $i$ #h(2em)$lambda_i'$ = grown eigenvalue  
- $tau$ = ellipsoid threshold  
- $g$ = growth factor  

The result of this allowed the algorihtm to grow in direction of where points already are and potentially capturing anymore points into the ellipsoid around its principal axes that KNN was unable to capture.

