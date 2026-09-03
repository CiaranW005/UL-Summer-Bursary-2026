#import "@preview/wrap-it:0.1.1": wrap-content

= Method
DINOv2 ViT-S/14 is used to extract the 384-dimensional [CLS] embeddings from the normal training images, which form the representation space modelled by the algorithm. Images are resized to $224 times 224$ pixels and normalised using the ImageNet@imagenet channel statistics employed by the DINOv2@dinov2 preprocessing pipeline, with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225] for the red, green, and blue channels respectively. Ellipsoids are fitted using the standard MVTec AD@mvtec2019 training split, which contains only normal samples, and evaluated on the corresponding test split containing both normal and anomalous samples. Ellipsoids are constructed sequentially from the remaining unassigned embeddings. After a candidate region is fitted and accepted, its assigned embeddings are consumed, and the process repeats until the training set has been represented.

== Baseline 

A global Mahalanobis model is fitted independently for each MVTec AD category using its normal training embeddings. The centroid $mu$ is calculated from all available training embeddings, after which SVD@svd is applied to the centred embedding matrix. The resulting singular values are converted to covariance eigenvalues, with numerically negligible directions removed using a machine-precision tolerance. This produces a single covariance-aware representation of the normal embedding space for each category.

At inference, a test embedding is projected onto the retained principal directions and its Mahalanobis distance from the global centroid is calculated. A regularisation term of $1 times 10^(-6)$ is added to the retained eigenvalues for numerical stability. Any component of the embedding lying outside the retained subspace is additionally penalised using the same regularisation term. The resulting distance is used directly as the anomaly score, with larger values indicating greater deviation from the fitted normal distribution. The score is defined as

$
d(x) = sqrt(
  sum_i (v_i^T (x - mu))^2 / (lambda_i + epsilon) + norm(r(x))_2^2 / epsilon
)
$

where $v_i$ and $lambda_i$ denote the retained principal directions and corresponding covariance eigenvalues, respectively, and $r(x)$ is the residual component of $x - mu$ lying outside the retained subspace.

== The Starting Position

The algorithm begins by identifying the densest local region of the remaining normal embedding space using K-nearest neighbours (KNN)@knn. The embedding with the smallest mean distance to its K neighbours is selected as the initial centre. \
Because MVTec AD categories contain different numbers of training samples, K is defined as a proportion of the available embeddings rather than a fixed count. Candidate values of 2.5%, 5%, and 10% were evaluated, balancing sensitivity to local noise at small K against overly broad initial regions at large K. A value of 5% was selected as a practical compromise across categories, although the most suitable neighbourhood size varied between categories.

== Ellipsoid Fitting

Given a set of normal embeddings assigned to a candidate region, an ellipsoid is fitted to describe its local geometry. The centre $mu$ is calculated as the weighted mean of the assigned embeddings. The samples are then centred around $mu$ and scaled according to their weights.

Let $X_w$ denote the resulting weighted centred data matrix. Singular Value
Decomposition (SVD)@svd is applied as $X_w = U Sigma V^T$.\
The columns of $V$ define the principal directions of the region, while the singular values in $Sigma$ are used to derive the variance $lambda_i$ associated with each supported axis. Directions with negligible variance are removed, allowing the ellipsoid to retain only the dimensional structure supported by the local samples.

#grid(
  columns: (3fr, 1fr),
  column-gutter: 2em,
  align: (left + horizon, center + horizon),

  [
    The boundary of the ellipsoid is determined from the distances of its assigned embeddings. The boundary threshold τ is fitted as the maximum weighted squared ellipsoidal distance among the samples supporting the region. Fully weighted samples therefore determine the fitted extent of the ellipsoid, while downweighted samples exert progressively less influence on its boundary. Membership of a point $x$ is then determined using its squared ellipsoidal distance.
  ],

  [
    $ 
    sum_i (v_i^T (x - mu))^2 / (lambda_i + epsilon) <= tau 
    $ 
  ],
)
Here, $v_i$ and $lambda_i$ denote the principal direction and variance associated with axis $i$, while $epsilon$ is a small regularisation term used for numerical stability set as $1 times 10^(-4)$. \
This covariance-aware representation allows each region to adapt to the locally observed shape of the embedding distribution rather than assuming equal variation in every direction.

== Low-Support Ellipsoid Fitting <section:low_support_ell>

As normal embeddings are progressively assigned to ellipsoids, later candidate regions are constructed from fewer remaining samples. With very small sample sizes, the estimated covariance becomes increasingly unstable, which can produce poorly defined principal directions and unreliable growth. Covariance estimators such as Ledoit-Wolf@Ledoit_2012 can stabilise small-sample estimates through shrinkage towards a more isotropic covariance structure. However, this is undesirable here because it can introduce variance in directions that are weakly supported by the local samples, potentially encouraging subsequent ellipsoid growth into unsupported regions of the embedding space.

Based on validation analysis of ellipsoid size and covariance structure, including the ratio between the largest and smallest eigenvalues and the proportion of variance explained by the dominant principal component, candidate regions containing fewer than five samples were treated as low-support regions. Rather than relying entirely on their own covariance estimate, these regions borrow covariance structure from a previously fitted ellipsoid while progressively retaining more of their own geometry as additional local samples become available.

Support selection is performed in two stages. First, the candidate set is restricted to the five previously fitted ellipsoids whose centres are nearest to the candidate centre under Euclidean distance. This preserves locality in the original DINOv2 embedding space before geometric similarity is considered.

#grid(
  columns: (3fr, 1fr),
  column-gutter: 3em,
  row-gutter: 0.75em,
  align: (left + horizon, center + horizon),

  [
    For a singleton candidate, no local covariance orientation can be estimated. The support ellipsoid is therefore selected using both embedding similarity and the alignment between the candidate direction and the existing support geometry. Let $c$ denote the candidate embedding and $mu_s$ the centre of a support ellipsoid. The unit direction from the support towards the candidate is


  ],

  [
    $
    d = (c - mu_s) / norm(c - mu_s).
    $
  ],

  [
    Embedding similarity is measured using cosine similarity,
  ],

  [
    $
    S_"embed" = (c^T mu_s) / (norm(c) norm(mu_s)).
    $
  ],
  [
    Geometric similarity measures how closely $d$ aligns with the principal axes $v_(s,i)$ of the support ellipsoid. Each alignment is weighted by the corresponding axis length,
  ],
  [
    $
    S_"shape" =
    frac(
      sum_i abs(d^T v_(s,i)) sqrt(lambda_(s,i) + epsilon),
      sum_i sqrt(lambda_(s,i) + epsilon)
    ).
    $
  ],
  [
    The final singleton support score is the mean of the two components,
  ],
  [
    $
    S_"support" = (S_"embed" + S_"shape") / 2.
    $
  ]
)
For low-support candidates containing more than one sample, some local covariance structure can already be estimated. Support similarity is therefore defined as the mean of the singular values of $V_c^T V_s$, which measures alignment between the candidate and support principal subspaces.

After selecting the highest-scoring local support ellipsoid, its covariance structure is blended with that of the candidate according to $C_b=alpha C_c+(1-alpha)C_s$, where $alpha = min(1, n / 5)$ Consequently, the contribution of the borrowed covariance decreases as additional local evidence becomes available, with the candidate covariance used independently once $n>=5$.

== Ellipsoid Growth

The initial KNN neighbourhood provides a dense starting region, but its size should not directly determine the final ellipsoid. A growth stage is therefore used to expand each candidate region and incorporate nearby normal embeddings. When new points are enclosed, the centroid and covariance of the region are recalculated before growth continues. This process repeats until no additional embeddings can be incorporated.

#grid(
  columns: (3fr, 1fr),
  column-gutter: 2em,
  align: (left + horizon, center + horizon),

  [
  Growth is _non-uniform_ across the ellipsoid axes. Rather than applying the same scale to every direction, growth is proportional to the variance already observed along each principal axis. Let $lambda_i$ denote the variance associated with axis $i$ and $lambda_max$ the largest eigenvalue. The relative axis variance is
  ],

  [
    $
    r_i = max(lambda_i / lambda_max, r_min).
    $
  ],
)


Given a global growth factor $g$, the scale applied to axis $i$ is $a_i = 1 + (g - 1) r_i, quad lambda_i' = lambda_i a_i^2.$\
Here, $r_min$ prevents axes with very small but non-zero variance from receiving no growth. Consequently, an axis with $r_i = 1$ receives the full growth factor $g$, while lower-variance directions expand more conservatively.

== Candidate Cleaning

As ellipsoids are constructed sequentially, embeddings assigned to an earlier region are removed from consideration when fitting subsequent regions. However, a newly fitted candidate may still geometrically expand into a previously assigned embedding. Candidate cleaning is therefore introduced to constrain this encroachment while preserving as much of the candidate's local covariance structure as possible.

Rather than immediately removing an embedding from the candidate neighbourhood, each candidate embedding is assigned a weight $w_i in [0,1]$. These weights are normalised before the weighted centroid and covariance structure are estimated using the fitting procedure described above. A weight of one represents full contribution to the fitted geometry, while a weight of zero removes that embedding from the covariance estimate.

#grid(
  columns: (3fr, 1fr),
  column-gutter: 2em,
  align: (left + horizon, center + horizon),

  [
  When a previously assigned embedding $x_s$ lies inside the candidate ellipsoid, its contribution along each retained principal axis is evaluated as
  ],

  [
    $
    c_i(x_s) =
    (v_i^T (x_s - mu))^2 / lambda_i.
    $
  ],
)

The axis with the greatest contribution is #h(20pt)$i^* = arg max_i c_i(x_s).$

#grid(
  columns: (3fr, 1fr),
  column-gutter: 2em,
  align: (left + horizon, center + horizon),

  [
  Reducing the candidate variance along this direction provides the most direct means of moving the conflicting point towards the ellipsoid boundary. The candidate embedding contributing most strongly to this axis is therefore selected as
  ],

  [
    $
    j^* =
    arg max_j
    (v_(i^*)^T (x_j - mu))^2 / lambda_(i^*).
    $
  ],
)

Rather than immediately discarding $x_(j^*)$, its weight is reduced using binary search. After each update, the ellipsoid is refitted and the conflicting embedding is tested again. The largest weight that removes the encroachment is retained, preserving as much of the original neighbourhood as possible. If no positive weight resolves the conflict, the embedding weight is reduced to zero and the candidate is refitted without its contribution.

== Scoring

At inference, each test embedding is evaluated against every ellipsoid in the learned cloud. From ellipsoid j, the boundary margin is defined as the squared ellipsoidal distance from the sample minus the fitted threshold, $m_j(x) = d^2_j(x) - tau_j$, the anomaly score is the minimum margin across all ellipsoids 
$s(x) = min_(j) m_j(x)$ with $j^* = arg min_j m_j(x)$ identifying the corresponding normal region. Negative scores indicate that the sample lies within at least one ellipsoid, while positive scores indicate that it lies outside every modelled region of normality. Increasing positive margin therefore represents increasing deviation from the learned normal space.

#align(center)[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    table(
      columns: (auto, auto),
      [*Hyperparameter*], [*Value*],
      [K (init. neighbourhood)], [5%],
      [Low-support threshold], [5],
      [Local support candidates], [5],
      [Growth factor $g$], [1.1],
    ),
    table(
      columns: (auto, auto),
      [*Hyperparameter*], [*Value*],
      [Ellipsoidal Reg. $epsilon$], [$1 times 10^(-4)$],
      [Mahalanobis Reg. $epsilon$], [$1 times 10^(-6)$],
      [$r_"min"$], [$1 times 10^(-4)$],
      [Blending weight $alpha$], [$min(1, n/5)$],
    )
  )
]