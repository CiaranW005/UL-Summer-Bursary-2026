== CLS Embedding Statistics

#let cls_results = csv("../../../data/results/pretrained/cls_embed_summary.csv")

#table(
  columns: 8,
  ..cls_results.flatten()
)

Silhouette Score has a very low mean, indicating that when all categories are considered simultaneously, the CLS embeddings do not form highly compact and well-separated clusters. This is expected given that many categories contain visually similar objects and that silhouette scores become more difficult to interpret when a large number of classes are present.

Defect Inter-Cluster Distance is consistently larger than Good Inter-Cluster Distance, suggesting that defective samples tend to lie further from the category centroid than normal samples. This supports the assumption that anomalies occupy more distant regions of the embedding space.

The mean Separation Ratio of approximately 1.60 indicates that defective samples are, on average, around 60% further from the category centroid than normal samples. This suggests that the pretrained CLS embeddings already contain a useful anomaly signal despite no task-specific training.

Good Intra-Cluster Distance is generally lower than Defect Intra-Cluster Distance, indicating that normal samples form more compact clusters while defective samples exhibit greater variability. This is consistent with anomalies introducing additional visual variation within a category.

Several metrics identify bottle as the best-performing category, while screw, capsule, frequently appear as the most challenging categories. This aligns with the earlier AUROC analysis, where screw consistently demonstrated poorer anomaly detection performance.