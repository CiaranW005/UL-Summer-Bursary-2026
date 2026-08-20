= Embedding Space Visualisation

This section investigates how embeddings extracted from the model are organised within the embedding space.

The embedding space is analysed using three dimensionality reduction techniques, including:

- PCA
- t-SNE
- UMAP

Analysis is performed at both the image level, using CLS embeddings, and the local level, using patch embeddings. This allows investigation of how entire images relate to one another within the embedding space, as well as how individual patches relate to other patches within the same image.

Scoring metrics will be used to analyse cluster quality and separation, including:

- Silhouette Score (Sil.)
- Davies-Bouldin Index (DB)
- Calinski-Harabaz Index (CH)
- Intra-Cluster Distance
- Inter-Cluster Distance

(Should probably explain metrics just so its giving a why these ones are chosen, same with what does each tell and why, this is a dissertation so it can go more repetitive)

The objective of this notebook is to understand the structure of the embedding space and identify characteristics that can help with anomaly detection. These findings will be used to explain the performance of the baseline model and provide a reference for future experiments involving fine-tuning and alternative embedding representations.