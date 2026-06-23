== CLS Embedding Projections

#figure(
  image("../../../images/baseline/base_embeds/cls_embedding_comparison.svg",width: 120%),
  caption: ("Embedding Projections of CLS tokens")
)

Within the PCA projection, we can observe how the embedding space is organised along the two principal components that capture the greatest variance in the data. Several categories form distinct clusters that reflect similarities in their visual appearance. For example, relatively flat and homogeneous objects such as tile, leather, and wood are positioned closer together, while categories such as screw occupy a more distant region of the embedding space due to their more complex geometric structure and visual texture.

In the t-SNE projection, we can observe how compact or diffuse each category is within the embedding space. Most categories form relatively tight clusters, indicating that samples from the same category are represented consistently by the model. Notable exceptions include grid and screw, which exhibit more diffuse and fragmented structures. The screw category is particularly interesting, as it also achieved some of the lowest anomaly detection performance in the previous analysis. Rather than forming a single compact cluster, the embeddings are distributed in a figure-eight like structure, suggesting that normal and defective samples may occupy multiple overlapping regions of the embedding space. This reduced compactness may contribute to the lower AUROC scores observed for this category, as anomalous samples become more difficult to distinguish from normal samples using anomaly scoring methods.

Finally, the UMAP projection highlights how distinct the local neighbourhoods are within the embedding space and how compact these neighbourhoods remain. The observations regarding cluster compactness are similar to those seen in the t-SNE projection, with grid and screw again emerging as the most diffuse categories. Rather than forming a single coherent cluster, samples from these categories occupy multiple regions of the embedding space, with the screw category appearing to split into three distinct subregions. This provides further insight into the lower anomaly detection performance observed for screw in the previous analysis. Since normal samples are distributed across several disconnected regions rather than a single compact cluster, anomaly scoring methods may struggle to distinguish normal and defective samples consistently, resulting in reduced AUROC performance.

#figure(
  image("../../../images/baseline/base_embeds/def_normal.svg", width: 120%)
)

For the PCA projection, there is unsurprisingly little separation between normal and defective samples. Since PCA preserves the directions of greatest global variance, the principal components are dominated by category-level differences rather than the subtle differences introduced by anomalies. As a result, normal and defective samples largely occupy the same regions of the embedding space.

Within the t-SNE projection, we can observe how defective samples are positioned relative to local neighbourhoods of normal samples. For many categories, particularly those containing flat and visually consistent objects, normal and defective samples form relatively distinct regions. However, the screw category remains an exception. Here, normal and defective samples occupy almost identical regions of the embedding space, with little evidence of a distinct anomaly cluster. This provides a possible explanation for the lower anomaly detection performance observed previously, as the embedding representation does not naturally separate normal and defective screw samples.


The UMAP projection exhibits similar behaviour to t-SNE. For many categories, defective samples tend to occupy regions adjacent to, but partially separated from, normal samples. However, in more challenging categories such as screw, normal and defective samples continue to share the same embedding regions. This suggests that the model struggles to learn discriminative representations for these anomalies, making them difficult to separate using distance-based anomaly scoring methods.

