== CLS Embedding Statistics

#let cls_results = csv("../../../data/results/pretrained/cls_embed_summary.csv")

#let labels = (
  [Sil.],
  [DB],
  [CH],

  [Normal],
  [Defect],
  [Sep. Ratio],
)

#figure(
  align(center)[
    #table(
    columns: 8,
    align: center,
    stroke: (x: none, y: 0.5pt),

    table.vline(x: 6, stroke: 0.6pt),
    table.hline(stroke: 0.8pt),

    [*Metric*], [*Mean*], [*Median*], [*Std*],
    [*Min*], [*Max*], [*Worst*], [*Best*],

    table.hline(stroke: 0.8pt),

    table.cell(align: left, colspan: 6, inset: (top: 8pt, bottom: 4pt))[*Cluster Quality*],
    table.cell(colspan: 2)[],
    table.hline(stroke: 0.8pt),

    ..cls_results.slice(1, 2).enumerate().map(((i, row)) => (
      labels.at(i),
      ..row.slice(1)
    )).flatten(),

    ..cls_results.slice(7, 9).enumerate(start: 1).map(((i, row)) => (
      labels.at(i),
      ..row.slice(1)
    )).flatten(),

    table.cell(align: left, colspan: 6, inset: (top: 8pt, bottom: 4pt))[*Inter-Cluster Normal-Reference Distances*],
    table.cell(colspan: 2)[],
    table.hline(stroke: 0.8pt),

    ..cls_results.slice(2, 5).enumerate(start: 3).map(((i, row)) => (
      labels.at(i),
      ..row.slice(1)
    )).flatten(),

    table.cell(align: left, colspan: 6, inset: (top: 8pt, bottom: 4pt))[*Intra-Cluster Distance*],
    table.cell(colspan: 2)[],
    table.hline(stroke: 0.8pt),

    ..cls_results.slice(5, 7).enumerate(start: 3).map(((i, row)) => (
      labels.at(i),
      ..row.slice(1)
    )).flatten(),
  ) 
  ],
  caption: [CLS Embedding Statistics]
) <table:cls_stats>

The mean Silhouette Score across categories is 0.04, indicating weak separation between normal and defective CLS embeddings. Since a score close to zero indicates substantial overlap between clusters, the result suggests that normal and defective samples often occupy similar regions of the CLS embedding space. This can be seen for the best-performing (leather: 0.1615) and worst-performing (capsule: -0.0722) categories in @fig:nvd_cls. For leather, the normal and defective embeddings are comparatively well separated in the projection, forming two more distinct regions. In contrast, capsule shows substantial overlap between normal and defective samples, with both occupying similar regions of the projected embedding space. However, this difference is not explained simply by the spatial extent of the anomalies. As shown in "eda_table", defects occupy approximately 0.9% of the image area on average for leather and 1.1% for capsule, despite their substantially different Silhouette Scores. The defect categories shown in "eda_figure" indicate that the leather class contains several distinct anomaly types, suggesting that characteristics other than spatial extent may influence how defects are represented. Further analysis of the individual defect types would be required to determine whether these differences contribute to the relatively strong CLS separability observed for leather.

The mean Davies-Bouldin Index across categories is approximately 3.28, indicating that the normal and defective CLS embeddings generally exhibit substantial within-cluster dispersion relative to the distance separating their clusters. This is most apparent for the worst-performing category, screw (6.6545), where @fig:nvd_cls shows that neither the normal nor defective samples form clearly distinct regions, with both distributed broadly across the projected embedding space. In contrast, the best-performing category, leather (1.5058), forms more clearly separated normal and defective regions in @fig:nvd_cls. However, both regions remain relatively dispersed rather than forming highly compact clusters, which is consistent with a DB value that, although substantially lower than screw and the overall mean, is still above zero. As these observations are based on a low-dimensional projection of the original embedding space, the projection alone cannot fully explain the DB values and should be treated as supporting visual evidence rather than a direct representation of the full embedding geometry.

The mean Calinski-Harabasz Index across categories is 8.8111, substantially lower than the best-performing category, leather (27.5506). The comparatively high CH value for leather indicates that the separation between its normal and defective CLS embeddings is large relative to the variation within those groups. This is consistent with the preceding Silhouette analysis, where leather also exhibited the strongest normal-defective separation and where its relatively strong separability was discussed in relation to the defect characteristics shown in "eda_table" and "eda_figure". In contrast, screw achieves the lowest CH value (2.6555), indicating that the separation between its normal and defective embeddings is small relative to their within-group variation. This may relate to the defect coverage reported in "eda_table", where anomalous regions occupy only approximately 0.3% of the image on average, potentially limiting their influence on the global CLS representation.

Both the Normal and Defect Inter-Cluster Distances are measured relative to the centroid of the normal training embeddings for each category, rather than relative to one another. Across categories, normal samples have a mean distance of 13.53 from this reference, compared with 20.54 for defective samples, producing a mean Separation Ratio of 1.60. This indicates that defective CLS embeddings are generally positioned farther from the learned normal representation. However, the strength of this distinction varies considerably between categories. Bottle achieves the highest Separation Ratio of 2.4991, indicating that its defective embeddings lie approximately 2.5 times farther from the normal training centroid than its normal test embeddings. In contrast, screw achieves the lowest ratio of 0.9871, meaning that its defective samples are, on average, marginally closer to the normal training centroid than its normal test samples. This is particularly problematic for distance-based anomaly scoring, since distance from the learned normal representation provides little discriminatory signal for this category. This is consistent with the high normal-defective overlap previously observed for screw in @fig:nvd_cls. The variation also motivates investigating whether category-level Separation Ratio is associated with anomaly-detection performance by comparing it with AUROC.

Normal Intra-Cluster Distance is generally lower than Defect Intra-Cluster Distance, indicating that normal samples form more compact clusters while defective samples exhibit greater variability. This may partly reflect the diversity of defect types within each category, since different anomalies can alter the visual representation in different ways. Consequently, defective CLS embeddings may occupy multiple distinct regions or neighbourhoods rather than forming a single coherent cluster, increasing their average distance from the defective centroid. The earlier "eda_table" also showed substantial variation in defect coverage within some categories, suggesting that defective samples can differ considerably in both appearance and spatial extent. This additional variation may further contribute to the greater spread of defective embeddings around their centroid.