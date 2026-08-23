#import "@preview/wrap-it:0.1.1": wrap-content

== Analyse the distribution of Anomaly Scores

The anomaly score distributions of normal and defective test samples are compared separately for each category and scoring method to examine how effectively the pretrained CLS embeddings distinguish anomalous samples from the learned normal representation. AUROC is computed for each category to quantify how consistently defective samples receive higher anomaly scores than normal samples. The methods are first compared using their aggregate performance across all categories, after which the category-level results of the selected Mahalanobis baseline are examined in greater detail. These results provide a reference for the subsequent embedding-space analysis, where differences in representation structure are considered in relation to anomaly-detection performance.

#let results = csv("../../../../../data/results/pretrained/auroc_stats.csv")

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
    caption: [AUROCs for each scoring method]
  ) <table:baseline>
]


The centroid-based method has reasonable performance across several categories; however, there is a noticeable overlap between the score distributions of normal and defective samples. This indicates that the distance from the centroid alone is not enough to fully describe the embedding space. While the centroid gets the average location of normal samples, it does not take into account the spread or covariance of the distribution, meaning some anomalous samples receive scores similar to normal samples.

The K-Nearest Neighbour approach improves upon the centroid-based method by producing clearer separation between the score distributions of normal and defective samples. The leather category has perfect separation, achieving an AUROC of 1.0. This suggests that anomalous samples are characterised more by their distance to nearby normal samples than by their distance to a single global centroid. The results indicate that normal and defective samples occupy distinct regions of the embedding space, allowing KNN to identify anomalies through local neighbourhood structure rather than relying solely on the overall centre of the distribution.

Average KNN does not improve upon the performance of standard KNN and achieves a lower average AUROC across the dataset. This suggests that the nearest normal neighbour contains the most informative signal for anomaly detection. By averaging across multiple neighbours, the anomaly score becomes influenced by additional normal samples that may lie further away in the embedding space, reducing the contrast between normal and defective samples. As a result, the score distributions exhibit greater overlap, leading to weaker separation and reduced anomaly detection performance.

The Mahalanobis score performed best among all evaluated methods, achieving perfect separation between normal and defective samples in several categories, including bottle, leather, and tile. Mahalanobis achieved the highest mean AUROC overall. KNN performed marginally better on screw, achieving 0.804 compared with 0.802 for Mahalanobis. Overall, Mahalanobis was therefore selected as the preferred anomaly scoring method for this study. Furthermore, the categories exhibiting perfect separation could be used as control groups in future experiments to evaluate whether proposed modifications improve anomaly detection performance or instead degrade the structure and separability of the embedding space.

Despite strong overall performance, categories such as screw achieved lower AUROC values across multiple scoring methods. This suggests that normal and defective samples occupy more similar regions of the embedding space,  making anomaly detection more challenging. These categories provide useful benchmarks for evaluating future improvements, as successful methods should increase separability in these difficult cases.

=== Anomaly scores across categories

#let aurocs = csv("../../../../../data/results/pretrained/aurocs.csv")

#let mal_table = figure(
  box(
    stroke: (
      left: 0.8pt,
      right: 0.8pt,
      top: none,
      bottom: none
    ),
    inset: (x: 0.5em, y: 1em)
  )[
    #align(center)[
      *Mahalanobis Across Categories*
    ]

    #table(
      columns: (auto, auto),
      stroke: (x: none, y: 0.5pt),

      table.hline(stroke: 0.8pt),

      table.header(
        [*Category*], [*Mahalanobis*]
      ),

      ..aurocs.slice(1).map(row => (
        [#row.at(0)],
        [#row.at(4)]
      )).flatten()
    )
  ],
  caption: figure.caption(
    separator: [],
    []
  )
)

#wrap-content(
  [
    #mal_table <table:mal_aurocs>
  ],
  [
  The category-level Mahalanobis results provide additional context to the aggregate results, where tile was identified as the best-performing category with an AUROC of 1.0. The full category breakdown shows that this performance is not unique to tile, with bottle and leather also achieving complete normal-defective separation. These three categories therefore provide useful reference cases for subsequent experimentation, as their embedding characteristics can be compared with more challenging categories to investigate which properties are associated with strong anomaly separation.
  
    Another potentially interesting comparison can be made between categories with relatively uniform, full-frame appearances, such as tile, wood, leather, and carpet. Although tile and leather achieve perfect Mahalanobis separation, wood and carpet do not. This raises the question of which characteristics of visually uniform categories contribute to stronger anomaly separation. Defect coverage is one possible factor; however, the values in @fig:defect_coverage do not suggest a simple relationship. The perfectly separated categories, leather and tile, have average defect coverages of approximately 0.9% and 9.8%, respectively, while carpet and wood have intermediate coverages of approximately 2.1% and 5.1%. This suggests that defect extent alone is unlikely to explain the difference in AUROC. Other factors may include how visually distinctive a defect is from the surrounding texture, whether the anomaly produces a strong local contrast, or how the pretrained representation responds to different spatial regions of the image. Differences in the visual characteristics represented during pretraining could also contribute, although this cannot be directly verified from the current analysis.
  ],
  align: top + right,
  column-gutter: 1em
)
