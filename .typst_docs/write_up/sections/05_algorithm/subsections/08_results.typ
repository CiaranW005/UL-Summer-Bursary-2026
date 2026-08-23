== Results

This section presents the experimental evaluation of the proposed hypersphere and ellipsoid algorithms on the MVTec AD dataset. The methods are compared against several statistical baselines using image-level AUROC. The results are analysed both in terms of overall performance across all categories and individual category performance to identify the strengths and limitations of the proposed approaches.

#include "08_result_tables/auroc_stats.typ"

The proposed hypersphere algorithm performs similarly to the Centroid, KNN and Avg-KNN scoring methods, which is expected as these statistics are incorporated during the construction of the hyperspheres. However, the final mean AUROC is lower than the KNN-based baselines. Inspection of the per-category results suggests that, for several categories, the fitted hyperspheres become 
sufficiently large that their behaviour approaches that of the Centroid method.

As the Centroid baseline achieves the lowest overall performance, this reduces the mean AUROC of the hypersphere algorithm. One possible explanation is that the global growth factor is not equally suitable for every category. Categories with more dispersed/compact embedding distributions may require larger/smaller growth, suggesting that category-specific or adaptive growth strategies should be investigated in future work.

The proposed ellipsoid algorithm achieved the second-best median AUROC, indicating that it performed competitively across most categories. However, a small number of outlier categories reduced the overall mean performance, with the transistor category performing substantially worse than the other methods despite similar performance on categories that were generally considered easier.

A possible explanation for the poor performance on transistor is the covariance support estimation. Unlike most categories, transistor contained relatively few large ellipsoids, with the largest consisting of only 11 embeddings. Covariance support was enabled for ellipsoids containing fewer than five embeddings, however only 18 of the 43 fitted ellipsoids contained enough embeddings to estimate their covariance independently. Consequently, over half of the ellipsoids relied on borrowed covariance from neighbouring regions.

Examining the eigenvalue ratios ($lambda_max / lambda_min$) of these ellipsoids shows that many possess extremely anisotropic covariance structures, where a single principal direction dominates the local variance. Repeatedly borrowing these highly anisotropic covariance estimates may therefore cause this dominant direction to propagate throughout the embedding space, stretching neighbouring ellipsoids in directions that are not supported by their own local embeddings. This can cause the normal region to expand excessively, resulting in more defective embeddings being classified as normal and ultimately reducing the AUROC.

#pagebreak()

#include "08_result_tables/cat_aurocs.typ"

Interestingly, at the category level the ellipsoidal algorithm generally performed similarly to, or slightly worse than, the global Mahalanobis approach until the screw category, where it outperformed all other methods. A possible explanation can be found in the earlier embedding analysis, where the screw category exhibited a distinctive UMAP structure consisting of three well-separated regions.

Although UMAP is only a low-dimensional projection of the embedding space, this separation suggests that the normal screw embeddings occupy multiple disconnected local neighbourhoods rather than a single compact distribution. A global Mahalanobis model attempts to capture these regions using a single covariance matrix, whereas the proposed ellipsoidal algorithm models each local neighbourhood independently. By fitting multiple ellipsoids to the local covariance structure, the algorithm is able to represent these disconnected regions more accurately, resulting in a higher AUROC for the screw category.

Overall, although the proposed ellipsoid algorithm did not consistently outperform the global Mahalanobis baseline in terms of mean AUROC, it remained competitive across the majority of MVTec AD categories. More importantly, unlike global statistical methods, the proposed approach constructs an explicit partitioning of the normal embedding space through multiple local statistical regions. This provides a more interpretable representation of the decision boundary, allowing individual ellipsoids to be inspected, analysed, and related directly to the local structure of the embedding space. The competitive performance obtained in conjunction with this increased explainability suggests that the proposed algorithm has room to be improved.

