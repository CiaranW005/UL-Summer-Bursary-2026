# Objective
The main objective of this project is to investigate unsupervised out-of-distribution detection in computer vision using Vision Transformer-based models. This includes comparing global, patch-level, and hybrid embeddings to evaluate their impact on anomaly detection performance, and exploring statistical methods to analyse how deviations form within embedding space, improving both separability and interpretability. The project will also analyse the structure and stability of the embedding space under different data conditions. The model will be benchmarked on the MVTec dataset and evaluated using standard anomaly detection metrics.

# Description of Project
Out-of-distribution detection evaluates how a sample fits within the range of known data and whether it is similar or dissimilar to previously observed samples.  Using Vision Transformers provides different representation outputs within the model itself, allowing analysis of how data is characterised at various levels (global/patch). The idea is to use these representations and compare how they can be used to efficiently identify anomalies. Not only is performance considered, but also computational efficiency, particularly in terms of processing speed and data handling.  The method will involve building an embedding space using only in-distribution data and detecting anomalies as deviations from these learned representations. Various approaches will be compared to understand how different embedding layers affect the separation of normal and anomalous samples, as well as how techniques such as fine-tuning and statistical methods can further improve this separation.  Interpretability will be explored through the structure of the embedding space. Clusters of normal data will be modelled as ellipsoidal regions to help define boundaries for what is considered normal. Principal Component Analysis will be used to visualise how the data is distributed and to measure cluster compactness, directional spread, and separation, to assess how anomalies differ under conditions such as rotation, lighting, and surface reflection. Experiments will be conducted using the MVTec dataset, which is a common benchmark for anomaly detection in industrial inspection tasks, containing images of normal and defective objects across multiple categories. Model performance will be evaluated using anomaly detection metrics, including AUROC, precision-recall and confusion matrices to assess detection accuracy. These metrics will be complemented by embedding-based structure metrics, such as silhouette score and intra- and inter-cluster distances, to better understand how anomalies are separated from normal data.

# Each Section to Explore
1. Dataset analysis 
2. Embedding extraction pipeline 
3. Baseline anomaly detector
4. Emedding space visualisation
5. Statistical modelling
6. Fine-tuning the model.
7. Evaluation
8. Extended Work
