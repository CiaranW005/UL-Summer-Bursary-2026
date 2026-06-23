== Patch Embedding Projections

#figure(
  image("../../../images/baseline/base_embeds/patch_plots/screw_plot.png", width: 112%)
)

The screw category performs noticeably better at the patch level than at the CLS embedding level. With stats being:
- Silhoutte ≈ 0.125
- Separation ≈ 1.2
- Intra-ratio ≈ 0.97

One possible explanation is that the anomalous regions within screw images typically occupy only a small portion of the image. As a result, the global CLS embedding is dominated by information from the surrounding normal screw structure, causing normal and defective images to appear highly similar in the image-level embedding space. In contrast, patch embeddings focus on local image regions, allowing patches containing anomalies to be represented separately from normal patches. This suggests that local representations are better able to capture the subtle visual differences associated with screw defects.