#import "@preview/wrap-it:0.1.1": wrap-content

== Patch Embeddings 
Patch embeddings retain local representations for individual image regions rather than collapsing the image into a single global embedding. This allows anomalous regions to be examined independently of the surrounding normal content and provides a way to investigate whether local anomaly information is more distinct than it appears in the CLS representation. The following subsections quantify patch-level structure and then examine selected defect types visually.

=== Embedding Statistics

#let patch_subtable(body, letter) = figure(
  body,
  kind: "patch_subtable",
  caption: figure.caption(
    separator: [],
    []
  ),
  supplement: [Table],
  numbering: _ => context {
    let parent = counter(figure.where(kind: table)).get().last() + 1
    [#parent#letter]
  },
)

#let patch_results = csv("../../../../../data/results/pretrained/patch_embed_summary.csv")

#set text(size: 10pt)

#let cluster_table = patch_subtable(
  box(
  stroke: (
    left: 0.8pt,
    right: 0.8pt,
    top: none,
    bottom: none,
  ),
  inset: (x: 0.5em, y: 1em),
)[
  #align(center)[
    *Cluster Quality Metrics*
  ]

  #table(
  columns: (auto, auto, auto, auto),
  stroke: (x: none, y: 0.5pt),
  
  table.hline(stroke: 0.8pt),

  table.header(
    [*Category*],
    [*Sil.*],
    [*DB*],
    [*CH*]
  ),

  ..patch_results.slice(1).map(row => (
    [#row.at(0)],
    [#row.at(1)],
    [#row.at(2)],
    [#row.at(3)]
  )).flatten()
  )],
  "a"
)

#wrap-content(
  [
    #cluster_table <table:patch_cluster_metrics>
  ],
  [
  The patch-level Silhouette Scores are consistently positive and generally higher than the mean CLS score of approximately 0.04 reported in @table:cls_stats, suggesting that local patch representations contain clearer normal-defective structure than the global image representation. However, the two values should not be interpreted as directly equivalent. Whereas the CLS score evaluates normal and defective images collectively within each category, the patch-level score is calculated separately within each defective image by comparing its normal and anomalous patches, before being averaged across the category. This means that local anomalous regions are evaluated within the context of their own image, rather than requiring visually different defect types to occupy a common region of the category-level embedding space. The stronger patch-level separation therefore provides additional support for the earlier CLS observation that anomaly characteristics, rather than spatial extent alone, may influence how clearly defects are represented. \
  This interpretation is also supported by the EDA. As shown in @fig:defect_type_example and @fig:defect_coverage, categories such as leather and grid achieve some of the highest patch-level Silhouette Scores despite containing relatively small anomalous regions, while metal nut exhibits substantially greater average defect coverage but a comparatively low Silhouette Score. This suggests that patch-level separability depends more strongly on the visual characteristics and consistency of the anomaly than on its spatial extent alone. Because anomalous regions usually occupy only a small part of each image, the patch-level Silhouette Score may be influenced by the much larger number of normal patches. However, differences in Silhouette Score do not appear to follow category sample count directly, as shown in @fig:defect_coverage. This suggests that the variation between categories cannot be explained simply by differences in the number of available samples.

  The mean patch-level Davies-Bouldin Index across categories is approximately 1.63, considerably lower than the mean CLS value of 3.28 reported in @table:cls_stats. This suggests that, overall, the normal and anomalous patch embeddings exhibit lower within-cluster dispersion relative to the separation between them than the corresponding CLS representations. The relatively low DB values observed for categories such as leather, carpet, and grid support the stronger Silhouette Scores, indicating that their normal and defective patch embeddings form comparatively compact and separated local structures. This is particularly notable for grid, which appeared diffuse in the earlier CLS projections @fig:embed_proj_cls, suggesting that local patch representations can retain coherent anomaly-related structure even where the global image representation is more fragmented. Compared with @table:cls_stats, leather remains the best-performing category, with its DB value decreasing from 1.5058 at the CLS level to 0.946 at the patch level. However, the behaviour of screw changes substantially: despite having the worst CLS DB value of 6.6545, its patch-level DB falls to 1.2913, below the patch-level mean. This suggests that the patch representation captures more coherent local structure for screw than is apparent in its global CLS representation. Whether these changes in embedding structure are associated with anomaly-detection performance is can be for future experimentation.
  
  The mean patch-level Calinski-Harabasz Index is approximately 21.13, compared with 8.81 for the CLS embeddings reported in @table:cls_stats, indicating stronger between-group separation relative to within-group variation at the patch level. However, this structure varies considerably between categories. Screw remains the worst-performing category, with a patch-level CH value of 5.5748 compared with 2.6555 at the CLS level. As shown in @fig:defect_coverage, screw defects occupy only approximately 0.3% of the image on average, meaning that relatively few patches contain anomalous information. This suggests that defect coverage may influence patch-level cluster structure by limiting the amount of anomalous information available for separation. In contrast, tile, with an average defect coverage of approximately 9.8%, achieves the highest patch-level CH value of 56.9894. However, defect coverage does not fully explain the variation between categories, indicating that characteristics of the anomalies themselves also influence the resulting patch embedding structure.
  ],
  align: top + right,
  column-gutter: 1em
) 

#let inter_table = patch_subtable(
  box(
  stroke: (
    left: 0.8pt,
    right: 0.8pt,
    top: none,
    bottom: none,
  ),
  inset: (x: 0.5em, y: 1em),
)[
  #align(center)[
    *Inter-Cluster Distances*
  ]

  #table(
    columns: (auto, auto, auto, auto),
    stroke: (x: none, y: 0.5pt),

  table.header(
    [*Category*],
    [*Normal*],
    [*Defect*],
    [*Sep. Ratio*]
  ),

  ..patch_results.slice(1).map(row => (
    [#row.at(0)],
    [#row.at(4)],
    [#row.at(5)],
    [#row.at(6)]
  )).flatten()
  )],
  "b"
)

#v(1cm)

#wrap-content(
  [
    #inter_table <table:inter_distances>
  ],
  [
  For each category, a normal reference centroid is constructed from the patch embeddings of all normal training images. Both Normal and Defect Inter-Cluster Distances are measured relative to this shared training centroid for each test image, before being averaged across the category. Across categories, the mean Normal Inter-Cluster Distance is approximately 34.46. Several categories exhibit comparatively low normal-reference distances, including carpet (28.0213), grid (29.8463), wood (30.4740), and leather (30.8956), indicating that their normal patches remain relatively close to the learned normal reference. For categories such as carpet, leather, and wood, this may reflect comparatively consistent local appearance. Grid is particularly interesting because it also achieves one of the strongest patch-level Silhouette Scores. One possible explanation is its spatial structure,much of each image consists of background between relatively thin grid elements, meaning that many patches may contain highly similar normal content, while patches intersecting a defect can contain more distinctive local changes. This could contribute both to the relatively small normal-reference distance and the stronger separation observed for anomalous patches. However, further patch-level analysis would be required to determine whether this spatial structure is responsible for the observed behaviour.

  Defect Inter-Cluster Distance averages approximately 43.00 across categories. A notable contrast with the CLS results in @table:cls_stats is observed for capsule. At the CLS level, capsule has the lowest distance (13.0549), whereas its patch-level distance of 45.1315 is above the patch-level mean. As shown in "eda_table", capsule defects occupy only approximately 1.1% of the image on average. This suggests that anomalous regions can be locally distinct within the patch embedding space while contributing relatively little to the global CLS representation because they occupy only a small proportion of the image. This provides further evidence that local anomaly information may be weakened when aggregated into a global image representation.

  Most categories achieve Separation Ratios above 1, with a mean of approximately 1.263, indicating that defective patches generally lie farther from the normal reference than normal patches. Leather achieves the strongest separation (1.6115), while metal nut has a ratio of approximately 1.00 (0.9997), indicating almost no difference between the average normal and defective patch distances. This result is particularly interesting in relation to @fig:defect_coverage, where metal nut has the largest average defect coverage, and the earlier EDA identified the flip defect type as contributing to this unusually large anomalous area. A possible explanation is that a spatially large defect does not necessarily produce locally distinctive patch content. In the case of a structural anomaly such as a flipped object, many patches may still contain visual features similar to those observed in normal samples despite their altered global arrangement. This could reduce the distinction between normal and defective patch distances and provides further evidence that defect coverage alone does not determine the quality of the learned anomaly representation. 
  
  Another notable contrast is bottle, which achieves a relatively low patch-level Separation Ratio of 1.0888 despite having the strongest CLS-level Separation Ratio of 2.4991 in @table:cls_stats. Conversely, grid exhibits an above-average Defect Inter-Cluster Distance and one of the highest patch-level Separation Ratios (1.5073), despite showing substantially weaker structure at the CLS level. These contrasting behaviours indicate that strong local anomaly separation does not necessarily translate directly into strong global separation, and vice versa. This suggests that the relationship between the number and spatial distribution of anomalous patches, their local distinctiveness, and the extent to which this information contributes to the global CLS representation warrants further investigation.
  ],
  align: top + left,
  column-gutter: 1em
)

#let intra_table =  patch_subtable(
  box(
  stroke: (
    left: 0.8pt,
    right: 0.8pt,
    top: none,
    bottom: none,
  ),
  inset: (x: 0.5em, y: 1em),
)[
  #align(center)[
    *Intra-Cluster Distances*
  ]
 
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: (x: none, y: 0.5pt),
    align: center,

  table.vline(x: 4, stroke: 0.8pt),

  table.header(
    [*Category*],
    [*Normal*],
    [*Train*],
    [*Ratio*],
    [*Defect*],
  ),

  ..patch_results.slice(1).map(row => (
    [#row.at(0)],
    [#row.at(7)],
    [#row.at(9)],
    [#row.at(10)],
    [#row.at(8)],
  )).flatten()
)],
  "c"
)

#v(1cm)

#wrap-content(
  [
    #intra_table <table:intra_distances>
  ],
  [
  Similarly to the inter-cluster analysis, the normal patch intra-cluster distances show substantial variation between categories, with a mean of approximately 31.55. Some of the most compact normal patch distributions are observed for carpet (24.8757), grid (24.6649), wood (23.5885), and leather (26.4941). This may partly reflect the consistent local appearance of these categories, as discussed in the preceding inter-cluster analysis. For grid, the large regions of visually similar background between the thin grid structures may also contribute to the comparatively compact normal patch representation. Normal test-patch compactness is additionally compared with that of the normal training patches to determine whether the spread of normal representations remains consistent when they occur within defective test images. On average, the normal test patches are slightly more compact than the training patches, with mean intra-cluster distances of approximately 31.55 and 32.85, respectively. In defective images, some categories contain fewer normal patches because a larger proportion of the image is occupied by anomalous regions. This can make the estimated test intra-cluster distance less stable for those images, although it does not inherently cause the distance itself to be lower.
  
  Most categories have a Normal Test/Train ratio close to 1, with a mean of approximately 0.961, suggesting that the compactness of normal patches is generally preserved between the training and test distributions. However, some categories deviate more substantially. Wood achieves the lowest ratio of 0.8228, indicating the largest reduction in normal patch dispersion relative to its training distribution. This behaviour cannot be explained straightforwardly by defect coverage alone. As shown in @fig:defect_coverage, wood has an average defect coverage of approximately 5.1%, while bottle has a larger average coverage of 7.6% but retains a ratio much closer to 1 (0.9763). Interestingly, wood also has the highest Normal Intra-Cluster Distance at the CLS level (18.9459) in @table:cls_stats, indicating comparatively high dispersion in its global normal representation. The combination of a large change in patch-level compactness and high CLS-level dispersion suggests that the relationship between local patch structure and the resulting global representation warrants further investigation. The following section examines these embedding characteristics further, while future work could investigate how local patch information is aggregated into the CLS token. <section:intra_clust_dist>

  Defect Intra-Cluster Distance is particularly interesting, as defective patch embeddings are substantially more compact than either the normal test or normal training patch distributions, with a mean distance of approximately 21.46. This suggests that anomalous patches within individual defective images often form relatively coherent local structures in the embedding space. One possible explanation is that patches containing the same anomaly share strongly related visual information, causing their representations to occupy similar neighbourhoods. However, the current distance-based analysis cannot determine how this structure arises within the transformer. Future investigation could examine the attention between anomalous patch tokens, their interaction with surrounding normal patches, and the extent to which this local information is subsequently propagated into the CLS representation. This could help determine whether the compactness observed at the patch level is related to how anomalous information is aggregated by the model.
  ],
  align: top + right,
  column-gutter: 1em
) 
