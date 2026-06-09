= Embedding Space Analysis and Visualisation

This stage of the project will investigate how normal and anomalous samples are distributed within the learned embedding space. The objective is to understand how model representations separate different types of data and to determine whether anomalies form distinct regions or deviations from normal samples.

Analysis will be performed at both the image level, using CLS embeddings, and the local level, using patch embeddings. This will allow investigation of both how entire images relate to one another within the embedding space and how individual patches relate to other patches within the same image and across different images.

Scoring metrics will be used to analyse cluster quality and separation, including:

- Silhouette Score
- Intra-Cluster Distance
- Inter-Cluster Distance

To help interpret these results, visualisation techniques to understand the embedding space include:

- PCA
- UMAP
- t-SNE

These visualisations will be used to investigate cluster compactness, the separation between normal and anomalous samples, and the distribution of different defect types within the embedding space.
