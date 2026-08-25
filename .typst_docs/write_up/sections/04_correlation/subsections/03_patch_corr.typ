
== Patch Correlation

This section examines how patch-level embedding statistics relate to the final image-level AUROC and whether particular characteristics of the local embedding space are associated with anomaly-detection performance. Unlike the preceding CLS analysis, these statistics describe local patch behaviour rather than the final global image representation. The correlations are therefore used to investigate whether properties of the local representation are reflected in the performance achieved using the CLS embeddings.

#align(center)[
  #box(width: 120%)[
    #grid(
      columns: (1fr, 1fr),
      column-gutter: auto,

      [
        #figure(
        image("../images/patch/patch_corr.svg", width: 84%),
        caption: [Patch metric correlations across all categories]
      ) <fig:patch_corr>
      ],
      [
      #figure(
        image("../images/patch/patch_no_screw_corr.svg", width: 100%),
        caption: [Patch metric correlations with screw excluded]
      ) <fig:patch_corr_no_screw>
      ],
    )
    ]
  ]

At the patch level, the Davies-Bouldin Index exhibits substantially different behaviour from the corresponding CLS metric. When all categories are considered in @fig:patch_corr, patch DB has almost no correlation with AUROC ($r=-0.01$). However, when screw is excluded in @fig:patch_corr_no_screw, the correlation increases in magnitude to approximately $r=-0.40$. This remains considerably weaker than the relationship observed at the CLS level, although the two metrics describe different structures and are therefore not directly comparable. The patch DB statistic is calculated from normal and anomalous patches within individual defective images before being aggregated across the category. Categories with very small defect regions, such as screw with an average coverage of approximately 0.3% in @fig:defect_coverage, contain relatively few anomalous patches within each image. This may make category-level patch statistics more sensitive to the behaviour of a small number of anomalous regions and contribute to greater variability in their relationship with final image-level AUROC.

A notable relationship is observed between Normal Train Intra-Cluster Distance and AUROC, with a moderate negative correlation of approximately $r=-0.44$. When screw is excluded, this strengthens to approximately $r=-0.50$. Although screw has a relatively large training intra-cluster distance and therefore lies in the expected direction of the negative relationship, its AUROC is substantially lower than would be expected from the broader linear trend, making it an influential category in the analysis. Its removal therefore produces a more consistent relationship across the remaining categories. The strengthened relationship among the remaining categories provides some evidence that compactness of the normal training patch representation may be a useful diagnostic during future fine-tuning experiments. However, further experiments would be required to determine whether deliberately increasing compactness leads to improved AUROC.

Another interesting patch-level relationship is observed between Defect Intra-Cluster Distance and AUROC. When screw is excluded in @fig:patch_corr_no_screw, the metric exhibits a negative correlation of $r = -0.43$, corresponding to $R^2 = 0.18$. This indicates a weak but notable association in which categories with more compact anomalous patch representations tend to achieve stronger image-level anomaly-detection performance. As discussed previously in #ref(<section:intra_clust_dist>, form: "page"), this also raises the question of how local patch structure is aggregated into the final CLS representation. If anomalous patches form coherent regions within the embedding space, this structure may contribute useful information to the global representation. However, Defect Intra-Cluster Distance measures only the internal compactness of the anomalous patches and does not describe how those patches are positioned relative to the normal representation.

Importantly, Defect Intra-Cluster Distance alone does not determine AUROC. For example, from @table:intra_distances, wood has a lower Defect Intra-Cluster Distance of 20.31 than metal nut at 25.82, indicating a more compact anomalous patch representation, yet achieves a lower AUROC of 0.944 compared with 0.983 for metal nut in @table:mal_aurocs. This demonstrates that compactness of anomalous patches is only one component of useful anomaly structure. A compact defect cluster may still occupy a region close to, or enclosed by, the normal patch distribution, while a more dispersed defect representation may remain sufficiently distinct to support strong detection. Future work could therefore examine Defect Intra-Cluster Distance together with measures of normal-defect separation and investigate how coherent but discriminative anomalous patch structure is aggregated into the CLS representation. In a semi-supervised setting, this could additionally motivate experiments examining whether explicitly encouraging such structure improves the resulting CLS representation and final AUROC.

#align(center)[
  #figure(
    image("../images/patch/patch_auroc_def_intra.svg", width: 120%),
    caption: [AUROC vs Defect Intra Distance]
  ) <fig:auroc_defect_intra>
]