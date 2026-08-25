= Embedding Space Visualisation <section:embed_vis>

This section investigates the structure of the pretrained embedding space to determine what information is represented before task-specific training and how this structure relates to anomaly detection performance. Analysis is performed at both the global image level, using CLS embeddings, and the local level, using patch embeddings. This distinction allows the representation of complete images to be compared with the local structure associated with individual anomalous regions, providing insight into whether anomaly information that is distinct at the patch level is retained within the final global representation.

Embedding structure is examined qualitatively using PCA, t-SNE, and UMAP, and quantitatively using measures of cluster separation, compactness, and distance from the learned normal representation. These analyses are used to investigate differences between normal and defective samples, variation between dataset categories, and whether characteristics identified during the earlier EDA, such as defect coverage and visual structure, help explain the observed behaviour. The resulting embedding characteristics are also considered in relation to the baseline AUROC results to identify properties that may contribute to strong or weak anomaly separation.

Scoring metrics will be used to analyse cluster quality and separation, including:

- *Silhouette Score (Sil.)* \
  Measures how similar samples are to their own cluster relative to the opposing cluster. Higher values indicate stronger separation, with a maximum value of 1. This metric is used to assess how distinctly the pretrained embedding space separates normal and defective representations.

- *Davies-Bouldin Index (DB)* \
  Measures within-cluster dispersion relative to the separation between clusters. Lower values indicate more compact and well-separated clusters. This provides a complementary measure to the Silhouette Score by considering how tightly normal and defective embeddings are grouped relative to the distance between them.

- *Calinski-Harabasz Index (CH)* \
  Measures between-cluster separation relative to within-cluster variation, with higher values indicating stronger cluster structure. This is used to determine whether differences between normal and defective representations are large relative to the variability present within each group.

- *Intra-Cluster Distance* \
  Measures how far embeddings within a group lie from their own centroid, where a value of 0 would indicate that all embeddings occupy the same point. This is used to examine the compactness and variability of normal and defective representations, and at the patch level also allows the stability of normal representations between the training and test distributions to be compared.

- *Inter-Cluster Distance* \
  Measures the distance of embeddings from a shared reference centroid. In this analysis, the normal training representation is generally used as the reference, allowing the distances of normal and defective test embeddings to be compared. This is used to determine whether defective representations move farther from the learned normal distribution and therefore contain a useful distance-based anomaly signal.

Together, these metrics provide complementary views of the embedding space: Silhouette, DB, and CH evaluate normal-defective cluster structure, while the intra- and inter-cluster distances examine the compactness of individual representations and their displacement from the learned normal reference.