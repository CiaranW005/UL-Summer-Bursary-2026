= Analyse the distribution of anomaly scores for normal and defective samples


A reusable plotting function is used to compare the anomaly score distributions of normal and defective test samples for each category. The score distributions are plotted separately for each anomaly scoring method, allowing the degree of separation between normal and defective samples to be visually inspected.


AUROC is also computed for each category to quantify how well the selected scoring method ranks defective samples above normal samples.

#let results = csv("../../../data/results/pretrained/auroc_stats.csv")

#let labels = (
  [Centroid],
  [KNN],
  [Avg-KNN],
  [Mahalanobis]
)

#align(center)[
  #figure(
    table(
      columns: 8,
      align: center,
      stroke: (x: none, y: 0.5pt),
      inset: (x: 2.5mm),

      table.hline(stroke: 0.8pt),
      table.vline(x: 4, stroke: 0.5pt),
      table.vline(x: 6, stroke: 0.5pt),

      [*Method*],
      [*Mean*], [*Median*], [*Std*],
      [*Min Category*], [*Value*],
      [*Max Category*], [*Value*],

      table.hline(stroke: 0.8pt),

      ..results.slice(1).enumerate().map(((i, row)) => (
        labels.at(i),
        ..row.slice(1)
      )).flatten(),
    ),
    caption: [Aurocs for each scoring method]
  )
]


The centroid-based method has reasonable performance across several categories; however, there is a noticeable overlap between the score distributions of normal and defective samples. This indicates that the distance from the centroid alone is not enough to fully describe the embedding space. While the centroid gets the average location of normal samples, it does not take into account the spread or covariance of the distribution, meaning some anomalous samples receive scores similar to normal samples.


The K-Nearest Neighbour approach improves upon the centroid-based method by producing clearer separation between the score distributions of normal and defective samples. The leather category has perfect separation, achieving an AUROC of 1.0. This suggests that anomalous samples are characterised more by their distance to nearby normal samples than by their distance to a single global centroid. The results indicate that normal and defective samples occupy distinct regions of the embedding space, allowing KNN to identify anomalies through local neighbourhood structure rather than relying solely on the overall centre of the distribution.

Average KNN does not improve upon the performance of standard KNN and achieves a lower average AUROC across the dataset. This suggests that the nearest normal neighbour contains the most informative signal for anomaly detection. By averaging across multiple neighbours, the anomaly score becomes influenced by additional normal samples that may lie further away in the embedding space, reducing the contrast between normal and defective samples. As a result, the score distributions exhibit greater overlap, leading to weaker separation and reduced anomaly detection performance.

The Mahalanobis score performed best among all evaluated methods, achieving perfect separation between normal and defective samples in several categories, including bottle, leather, and tile. Mahalanobis achieved the highest mean AUROC overall. KNN performed marginally better on screw, achieving 0.804 compared with 0.802 for Mahalanobis. Overall, Mahalanobis was therefore selected as the preferred anomaly scoring method for this study.. Furthermore, the categories exhibiting perfect separation could be used as control groups in future experiments to evaluate whether proposed modifications improve anomaly detection performance or instead degrade the structure and separability of the embedding space.

// TODO: Refer to the previous section with the small defect count
Despite strong overall performance, categories such as screw achieved lower AUROC values across multiple scoring methods. This suggests that normal and defective samples occupy more similar regions of the embedding space,  making anomaly detection more challenging. These categories provide useful benchmarks for evaluating future improvements, as successful methods should increase separability in these difficult cases.
