#import "@preview/wrap-it:0.1.1": wrap-content

#let ell_results = csv("../../../../data/results/pretrained/ellip_auroc.csv")
#let aurocs = csv("../../../../data/results/pretrained/aurocs.csv")

// discuss results, performance and boostraps
= Results

#let results = figure(
  box(
  stroke: (
    left: 0.8pt,
    right: 0.8pt,
    top: none,
    bottom: none,
  ),
  inset: (x: 0.5em, y: 1em),
  )[
  #align(center)[
    *Results Table (AUROC)*
  ]
  
  #table(
  columns: (auto, auto, auto),
  rows: 15,
  stroke: (x: none, y: 0.5pt),
  
  table.hline(stroke: 0.8pt),

  table.header(
    [*Category*],
    [*Ellip.*],
    [*Mahal.*],
  ),

  ..ell_results.slice(1).zip(aurocs.slice(1))
  .map(pair => (
    [#pair.at(0).at(0)],
    [#pair.at(0).at(1)],
    [#pair.at(1).at(4)]
  )).flatten()
  )
  ]
)


#wrap-content(
  [
    #results
  ],
  [

Results are reported using image-level AUROC @auroc. AUROC measures the probability that a randomly selected anomalous sample receives a higher anomaly score than a randomly selected normal sample, providing a threshold-independent measure of how well the two groups are ranked.

Overall, the proposed ellipsoidal algorithm performs worse than the Mahalanobis baseline, achieving a mean AUROC of 0.934 compared with 0.958 for Mahalanobis. This corresponds to a reduction of 0.024 in mean AUROC, indicating that the additional flexibility introduced by modelling multiple ellipsoidal regions does not generally improve anomaly separation across the dataset. Performance varies between categories, however, and screw is a notable exception, where the ellipsoidal approach improves upon the baseline. This initially suggests a possible category-specific benefit, although the bootstrap analysis below examines whether this improvement is robust.

However, these benchmark results are based on a single train/test configuration and therefore do not indicate how stable either method is to variation in the available samples. To assess robustness, training and test variation were evaluated separately using 1,000 bootstrap resamples. For the training bootstrap, normal training embeddings were resampled with replacement and both models were refitted before evaluation on the original test set. For the test bootstrap, normal and anomalous test embeddings were resampled with replacement while the fitted models were held fixed. Within each bootstrap iteration, both methods were evaluated using the same resampled data, allowing paired differences in AUROC to be calculated. Reported 95% intervals correspond to the 2.5th and 97.5th percentiles of the bootstrap distributions.

Across the training bootstraps, the difference between the two methods becomes more pronounced. The ellipsoidal algorithm decreases to a mean AUROC of 0.924 (95% CI [0.903, 0.944]), while the Mahalanobis baseline remains comparatively stable at 0.957 (95% CI [0.944, 0.967]), producing a mean difference of -0.033 (95% CI [-0.054, -0.011]). 
// This suggests that the ellipsoidal approach is more sensitive to variation in the training data, whereas the Mahalanobis baseline maintains more consistent performance across resampled training sets.

In contrast, performance of the ellipsoidal approach remains more stable across the test bootstraps, maintaining a mean AUROC of approximately 0.934 (95% CI [0.887, 0.971]). The Mahalanobis baseline increases to approximately 0.964 (95% CI [0.927, 0.986]), producing a mean paired difference of -0.030 (95% CI [-0.059, -0.001]) and further increasing the performance gap between the two methods. 
// This suggests that the reduced performance of the ellipsoidal approach is not primarily caused by a particular composition of the original test set. Instead, the greater degradation observed during training resampling indicates that the construction of the ellipsoidal representation is more sensitive to changes in the available normal training samples.

The improvement previously observed for screw does not remain stable under bootstrap resampling. While the ellipsoidal approach slightly outperformed the Mahalanobis baseline for screw in the original benchmark, this advantage is not maintained across either the training or test bootstraps, where the ellipsoidal method instead performs worse than the baseline on average. 
// This indicates that the apparent benchmark improvement is sensitive to the particular sample composition used in that evaluation. Although the embedding analysis suggests that screw has an unusual representation structure that may benefit from a multi-region model, the current implementation does not exploit this structure consistently enough to provide a reliable performance advantage.

In terms of computational cost, Mahalanobis is expected to be faster to fit, as it models a single global distribution, whereas the proposed method constructs multiple ellipsoidal regions. Across the bootstrap experiments, the ellipsoidal method required an average fitting time of approximately 115 ms per category, compared with approximately 10 ms for the Mahalanobis baseline, corresponding to an approximately 11-fold increase in fitting time. Interestingly, this increase is substantially smaller than the number of regions being fitted: the proposed method typically constructs around 30-40 ellipsoids per category, yet fitting time increases by only around one order of magnitude rather than proportionally with the number of ellipsoids. \
A similar but smaller difference is observed during evaluation, where the ellipsoidal method requires approximately 10.3 ms compared with 1.6 ms for Mahalanobis, corresponding to an approximately 6.5-fold increase. Despite these relative increases, the absolute runtime of both fitting and evaluation remains low.

Although the ellipsoidal approach does not outperform the Mahalanobis baseline overall, it provides additional information about how each test sample relates to the learned normal representation. Rather than producing a score relative to a single global distribution, each sample is evaluated against multiple local normal regions. This makes it possible to identify the region to which the sample is most closely related and examine its distance from the surrounding ellipsoidal boundaries. As a result, anomalous samples can be interpreted in terms of their relationship to specific local structures within the normal embedding space, information that is not available from a single global scoring model.
  ],
  align: right 
)
