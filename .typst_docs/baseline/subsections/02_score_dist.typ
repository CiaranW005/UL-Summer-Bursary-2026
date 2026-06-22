= Analyse the distribution of anomaly scores for normal and defective samples

A reusable plotting function is used to compare the anomaly score distributions of normal and defective test samples for each category. The score distributions are plotted separately for each anomaly scoring method, allowing the degree of separation between normal and defective samples to be visually inspected.

AUROC is also computed for each category to quantify how well the selected scoring method ranks defective samples above normal samples.

#let results = csv("../../../data/results/base_embeds/auroc_stats.csv")


#grid(
  columns: 4,
  column-gutter: 0.39cm,
  align: bottom + center,

  table(
    columns: 1,
    rows: 4,
    align: center,

    [Centroid], [KNN],
    [Avg-KNN], [Mahalanobis]
  ),

  table(
    columns: 3,
    rows: 5,
    align: center,

    [*#results.at(0).at(1)*],
    [*#results.at(0).at(2)*],
    [*#results.at(0).at(3)*],

    ..results.slice(1).map(row => (
      [#row.at(1)],
      [#row.at(2)],
      [#row.at(3)]
    )).flatten()
  ),

  table(
    columns: 2,
    rows: 5,
    align: center,

    [*Min Category*], [*Value*],

    ..results.slice(1).map(row => (
      [#row.at(4)],
      [#row.at(5)]
    )).flatten()
  ),

  table(
    columns: 2,
    rows: 5,
    align: center,

    [*Max Category*], [*Value*],

    ..results.slice(1).map(row => (
      [#row.at(6)],
      [#row.at(7)]
    )).flatten()
  ),
)

#align(center)[
  *AUROCS for each scoring method*
]

The centroid based method has reasonable perfomance across serveral categories, however there is a noticebale overlapo between the score distributions of normla and defevtive samples. This indicates that the diostance form the centroid alone is not enough to fully describe the embedding space. While the cnetoird get the average location of normal samples, it does not take into account for the spread or covariance of the distribution, meannig some anomalous samples to recieve scores similiar to normal samples. 

The K-Nearest Neighbour approach improves upon the centroid-based method by producing clearer separation between the score distributions of normal and defective samples. With the leather category having a perfect speration achieving an AUROC of 1.0. This suggests that anomalous samples are characterised by their distance to nearby normal samples than by their distance to a single global centroid. The results indicate that normal and defective samples occupy distinct regions of the embedding space, allowing KNN to identify anomalies through local neighbourhood structure rather than relying solely on the overall centre of the distribution.

Average KNN does not improve upon the performance of standard KNN and achieves a lower average AUROC across the dataset. This suggests that the nearest normal neighbour contains the most informative signal for anomaly detection. By averaging across multiple neighbours, the anomaly score becomes influenced by additional normal samples that may lie further away in the embedding space, reducing the contrast between normal and defective samples. As a result, the score distributions exhibit greater overlap, leading to weaker separation and reduced anomaly detection performance.

The Mahalanobis score performed best among all evaluated methods, achieving perfect separation between normal and defective samples in several categories, including bottle, leather, and tile. It also achieved the highest average AUROC across all categories except screw, making it the preferred anomaly scoring method for this study. Furthermore, the categories exhibiting perfect separation could be used as control groups in future experiments to evaluate whether proposed modifications improve anomaly detection performance or instead degrade the structure and separability of the embedding space.

Despite strong overall performance, categories such as screw achieved lower AUROC values across multiple scoring methods. This suggests that normal and defective samples occupy more similar regions of the embedding space, making anomaly detection more challenging. These categories provide useful benchmarks for evaluating future improvements, as successful methods should increase separability in these difficult cases.