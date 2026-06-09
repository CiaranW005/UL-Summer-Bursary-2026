= Baseline Model

The first stage of experimentation will focus on establishing baseline anomaly detection performance using the CLS token. The objective of this stage is to understand how effectively anomalies can be detected using the learned representations before introducing additional techniques such as fine-tuning, hybrid embeddings, or statistical modelling.

Initially, experiments will be performed using the extracted embeddings, with minimal hyperparameter tuning, focusing on ensuring stable convergence and reproducible results. This will provide a reference point against which later approaches can be compared.

Several distance- and similarity-based methods will be investigated for anomaly scoring, including:

- K-Nearest Neighbours
- Mean Embedding Distance
- Centroid-Based Methods
- Mahalanobis Distance @kamoi2020mahalanobisdistanceeffectiveanomaly

Hybrid approaches will also be investigated. One potential approach is a hierarchical pipeline in which global CLS embeddings are first used to identify potentially anomalous samples, with patch-level analysis applied only to images that exceed a predefined anomaly threshold. This may improve computational efficiency while retaining the ability to localise anomalies when required.

The performance of each approach will be evaluated using standard anomaly detection metrics, including:

- AUROC & AUPRC

The results obtained during this stage will serve as a baseline for later experiments involving patch-level embeddings, hybrid representations, and embedding-space analysis.
