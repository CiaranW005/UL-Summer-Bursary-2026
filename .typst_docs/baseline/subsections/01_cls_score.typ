= Evaluate several anomaly scoring methods using the extracted CLS embeddings

Metrics chosen include:

- *Centroid Distance:* Computes the Euclidean distance between each test embedding and the centroid (mean embedding) of the corresponding training category.

- *KNN:* Computes the distance between each test embedding and its K nearest training embeddings (K=5). The anomaly score is taken as the distance to the closest neighbour.

- *Average KNN:* Computes the average distance between each test embedding and its K nearest training embeddings.

- *Mahalanobis Distance* - Computes the distance between each test embedding and the training distribution while accounting for the covariance structure of the embedding space.

