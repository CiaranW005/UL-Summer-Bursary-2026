#align(center)[
  *Abstract*
]

#block(
  inset: (x: 1.5cm),
)[
  This paper investigates how normal image representations can be modelled in an unsupervised embedding space using multiple local multivariate regions rather than a single global distribution. DINOv2 embeddings are extracted from the 15 object and texture categories of MVTec AD, and a Cloud of Ellipsoids method is proposed to represent normal variation using locally fitted covariance-aware regions. Each test sample is scored according to its relationship with the learned ellipsoids, providing both an anomaly score and a local geometric interpretation of how it relates to the normal embedding space. The method is compared with a global Mahalanobis-distance baseline. Across the dataset, the proposed approach achieves a mean AUROC of 0.934 compared with 0.958 for Mahalanobis, indicating that the additional flexibility of multiple local regions does not improve overall anomaly-detection performance. Bootstrap evaluation further shows greater sensitivity of the ellipsoidal method to changes in the available training samples. These results suggest that local geometric modelling may be more effective when used to refine a stable global representation rather than constructing the normal embedding space entirely from independently fitted local regions.
]