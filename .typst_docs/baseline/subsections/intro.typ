= Baseline Model

This notebook explores the pretrained embeddings produced by the DINOv2 ViT-S/14 model (approximately 22 million parameters). DINOv2 is a self-supervised Vision Transformer that has demonstrated strong performance across a wide range of computer vision tasks, including anomaly detection.

The model divides an image into 14x14 pixel patches and produces a global image representation through the CLS token. In addition, patch-level embeddings can be extracted to provide local representations of different image regions.

The purpose of this notebook is to:

+ Evaluate several anomaly scoring methods using the extracted CLS embeddings:
  - Centroid Distance
  - K-Nearest Neighbours (KNN)
  - Mahalanobis Distance

+ Analyse the distribution of anomaly scores for normal and defective samples to determine how well each method separates the two classes.
+ Compare methods using AUROC to identify the most effective baseline anomaly scoring approach.

+ Identify categories that are already well separated within the embedding space, as well as categories that remain challenging and may benefit from additional experimentation.

The results obtained in this notebook will serve as a baseline against which future approaches, including patch-level embeddings, hybrid representations, and semi-supervised methods, can be compared.