== Patch Embedding Projections

(include more patches such as nut and wood for later questions to connect back to)

#figure(
  image("../images/screw_patch_plot.png", width: 112%)
)

The screw category performs noticeably better at the patch level than at the CLS embedding level. With stats being: (This is incorrect to say they are not overly comparable can mention this just word it better)
- Silhoutte ≈ 0.125
- Separation ≈ 1.2
- Intra-ratio ≈ 0.97

One possible explanation is that the anomalous regions within screw images typically occupy only a small portion of the image. As a result, the global CLS embedding is dominated by information from the surrounding normal screw structure, causing normal and defective images to appear highly similar in the image-level embedding space. In contrast, patch embeddings focus on local image regions, allowing patches containing anomalies to be represented separately from normal patches. This suggests that local representations are better able to capture the subtle visual differences associated with screw defects.

#figure(
  image("../images/metal_nut_patch_plot.png", width: 100%)
)

#figure(
  image("../images/wood_patch_plot.png", width: 112%)
)