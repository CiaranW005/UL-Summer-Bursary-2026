== Evaluate several anomaly scoring methods using the extracted CLS embeddings

Metrics chosen include:

- *Centroid Distance:* Computes the Euclidean distance between each test embedding and the centroid (mean embedding) of the corresponding training category.

- *KNN:*@knn The training embeddings are used as the reference set for KNN. Each test embedding is then compared against them to find its K nearest training embeddings and their distances. The anomaly score is taken as the distance to the closest neighbour.

- *Average KNN:* Computes the average distance between each test embedding and its K (K=5) nearest training embeddings.

- *Mahalanobis Distance:*@mahalanobis Computes the distance between each test embedding and the training distribution while accounting for the covariance structure of the embedding space.

