= Baseline Model

This section explores the pretrained embeddings produced by the DINOv2 ViT-S/14 model@dinov2. DINOv2 was selected because its self-supervised training produces general-purpose visual representations without requiring task-specific labels, making it well suited to the unsupervised setting considered in this work. The ViT-S/14 (approximately 21 million parameters) variant was chosen as a relatively compact model that provides both global CLS and local patch representations, allowing the same backbone to support the image-level experiments in this study and subsequent investigation of local anomaly representations.

The purpose of this section is to:

+ Evaluate several anomaly scoring methods using the extracted CLS embeddings:
  - Centroid Distance
  - K-Nearest Neighbours (KNN)
  - Mahalanobis Distance
  and compare their ability to distinguish normal and defective samples using AUROC. AUROC is used because it evaluates ranking performance across all possible decision thresholds, avoiding the need to select a fixed anomaly threshold for baseline comparison.

+ Identify categories that are already well separated within the embedding space, as well as categories that remain challenging and may benefit from additional experimentation.

+ Examine anomaly-detection performance across the twelve DINOv2 transformer layers to determine how the usefulness of the representation develops through the network and whether earlier layers provide comparable performance at reduced computational cost.

+ Compare the DINOv2-based baseline with established anomaly-detection approaches, examining differences in their backbone architectures, feature representations, and methods for modelling normal data and producing anomaly scores.

The results obtained in this section serve as a baseline against which subsequent approaches, including fine-tuned and semi-supervised representations, are compared.