= Baseline Model

This section explores the pretrained embeddings produced by the DINOv2 ViT-S/14 model (approximately 22 million parameters). DINOv2 was selected because its self-supervised training produces general-purpose visual representations without requiring task-specific labels, making it well suited to the unsupervised setting considered in this work. The ViT-S/14 (approximately 22 million parameters) variant was chosen as a relatively compact model that provides both global CLS and local patch representations, allowing the same backbone to support the image-level experiments in this study and subsequent investigation of local anomaly representations.

The purpose of this section is to:

+ Evaluate several anomaly scoring methods using the extracted CLS embeddings:
  - Centroid Distance
  - K-Nearest Neighbours (KNN)
  - Mahalanobis Distance

+ Analyse the distribution of anomaly scores for normal and defective samples to determine how well each method separates the two classes.
+ Compare methods using AUROC to identify the most effective baseline anomaly scoring approach. AUROC is used because it evaluates ranking performance across all possible decision thresholds, avoiding the need to select a fixed anomaly threshold for baseline comparison

+ Identify categories that are already well separated within the embedding space, as well as categories that remain challenging and may benefit from additional experimentation.

The results obtained in this section serve as a baseline against which subsequent approaches, including fine-tuned and semi-supervised representations, are compared.