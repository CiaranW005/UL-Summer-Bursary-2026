
== Defects Size vs AUROC

As proposed in @section:embed_vis, one objective of the embedding analysis was to investigate whether the spatial extent of a defect is associated with anomaly-detection performance. Several observations throughout the preceding analysis suggested that defect coverage alone had limited explanatory power; however, the relationship can be examined directly by comparing average category-level defect coverage with AUROC.

As shown in @fig:auroc_defect, the observed relationship is substantially weaker than initially expected. It was hypothesised that larger anomalous regions might be easier for the pretrained representation to distinguish, since they alter a greater proportion of the image and therefore have greater potential to influence the resulting global representation. However, little evidence of such a relationship is observed across the categories. Screw represents an influential case, combining both the smallest average defect coverage and the lowest AUROC. When screw is excluded, the already weak relationship is reduced further, with defect coverage exhibiting effectively no linear association with AUROC.

This suggests that the spatial extent of an anomaly alone is not a strong determinant of category-level anomaly-detection performance. Instead, characteristics such as the visual distinctiveness of the defect and the resulting embedding-space geometry are likely to be more informative. Defect coverage may still influence individual categories, particularly in extreme cases such as screw, but it does not appear to provide a general explanation for differences in AUROC across MVTec AD.

#align(center)[
  #figure(
    image("../images/defect_coverage_auroc.svg", width: 110%),
    caption: [AUROC vs Defect Coverage with fitted trendlines]
  ) <fig:auroc_defect>
]