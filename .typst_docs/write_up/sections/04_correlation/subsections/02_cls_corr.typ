== CLS Correlation

This section examines how CLS-level embedding statistics relate to the final AUROC and whether particular characteristics of the global embedding space are more strongly associated with anomaly-detection performance. \
As the analysis is performed across the 15 MVTec AD categories, the correlations are treated as exploratory associations rather than evidence of causal relationships. Sensitivity analyses are additionally performed where individual categories exhibit extreme metric values to determine how strongly these observations influence the resulting correlations.

#align(center)[
  #box(width: 115%)[
    #grid(
      columns: (1fr, 1fr),
      column-gutter: auto,

      [
        #figure(
        image("../images/cls/cls_corr.svg", width: 84%),
        caption: [CLS metric correlations across all categories]
      ) <fig:cls_corr>
      ],
      [
      #figure(
        image("../images/cls/cls_no_screw_corr.svg", width: 100%),
        caption: [CLS metric correlations with screw excluded]
      ) <fig:cls_corr_no_screw>
      ],
    )
    ]
  ]

At the CLS level, the Davies-Bouldin Index exhibits the strongest relationship with AUROC, with a strong negative correlation of approximately $r=-0.85$. As shown in @fig:auroc_vs_db, the corresponding $R^2$ of approximately 0.716 indicates that 71.6% of the observed category-level variation in AUROC is associated with the linear relationship with DB. This suggests that the balance between normal-defective separation and within-group dispersion captured by DB is strongly associated with anomaly-detection performance. @table:cls_stats reports a mean DB of 3.28 across categories, with a standard deviation of 1.4754 and values ranging from 1.5058 for leather to 6.6545 for screw. This substantial variation between categories appears particularly informative when considered alongside their anomaly-detection performance.

The strength of this relationship is partly influenced by the screw category. Screw exhibits both the highest DB value and the lowest AUROC among the evaluated categories, placing it at an extreme of the observed relationship. When screw is excluded from the analysis in @fig:cls_corr_no_screw, the correlation between DB and AUROC weakens from $r=-0.85$ to $r=-0.79$, with the corresponding $R^2$ decreasing from approximately 0.716 to 0.624. Therefore, although screw strengthens the observed relationship, DB remains strongly associated with AUROC across the remaining categories, accounting for approximately 62.4% of the observed variation in AUROC under the linear model. Interestingly, several other CLS statistics exhibit stronger correlations with AUROC after screw is removed. This suggests that screw is not simply a noisy observation, but an unusual category whose embedding structure is captured particularly strongly by DB while differing from broader relationships observed across the remaining categories.

#align(center)[
  #figure(
    image("../images/cls/cls_auroc_db_plot.svg", width: 120%),
    caption: [Davies-Bouldin Index vs AUROC]
  ) <fig:auroc_vs_db>
]

The Calinski-Harabasz Index provides a complementary view of this relationship by considering normal-defective separation relative to the variation within the two groups. Unlike DB, higher CH values indicate stronger cluster structure [REF]. When screw is excluded, as shown in @fig:cls_corr_no_screw, CH exhibits a moderately strong positive correlation with AUROC ($r=0.70$), corresponding to an $R^2$ of 0.49. This indicates that approximately 49% of the observed variation in category-level AUROC is associated with the linear relationship with CH. The result therefore provides additional evidence that anomaly-detection performance is associated not simply with the absolute displacement of defective embeddings, but with how strongly the normal and defective groups are separated relative to their internal variability.

@table:cls_stats reports a mean CH of 8.8111 with a standard deviation of 6.7326, ranging from 2.6555 for screw to 27.5506 for leather. This large variation is partly influenced by the extreme values of screw and leather at opposite ends of the distribution. As shown in @fig:auroc_vs_ch, categories with higher CH values generally tend to achieve higher AUROC, supporting the observed positive correlation. However, the relationship is not deterministic. Leather combines the highest CH with an AUROC of 1.0, while bottle and tile also achieve an AUROC of 1.0 despite substantially lower CH values. Similarly, metal nut and grid achieve AUROC values close to 1.0 with CH values considerably below that of leather and closer to the lower end of the observed range. This suggests that strong normal-defective cluster structure is associated with improved anomaly-detection performance, but represents only one of several characteristics contributing to category-level AUROC.

When both extreme categories, screw and leather, are excluded, the relationship strengthens slightly further in @fig:auroc_vs_ch to $r=0.716$, with $R^2=0.513$. Therefore, approximately 51.3% of the observed AUROC variation among the remaining categories is associated with the linear relationship with CH. Importantly, the correlation does not weaken when the two extreme CH observations are removed, suggesting that the relationship is not dependent on these categories alone. CH may therefore provide a useful diagnostic metric when evaluating subsequent fine-tuning or modifications to the embedding space, particularly for assessing whether changes to normal-defective cluster structure are associated with changes in anomaly-detection performance.

#align(center)[
  #figure(
    image("../images/cls/cls_auroc_ch_plot.svg", width: 120%),
    caption: [Calinksi-Harabasz Index vs AUROC]
  ) <fig:auroc_vs_ch>
]

The last notable, although less surprising, relationship is between Separation Ratio and AUROC. This relationship becomes particularly clear in @fig:cls_corr_no_screw when screw is excluded. As observed previously in @fig:nvd_cls, screw contains substantial overlap between normal and defective CLS embeddings and represents one of the most diffuse categories. More generally, categories with a larger Separation Ratio tend to achieve higher AUROC, which is consistent with the intended behaviour of the anomaly representation: as defective embeddings move farther from the learned normal reference relative to normal test embeddings, they become easier to distinguish using distance-based anomaly scoring. Unlike the relationship observed between DB and AUROC, the positive association between Separation Ratio and AUROC is largely expected from the definition of the metric and therefore serves primarily as confirmation that the observed embedding geometry is reflected in anomaly-detection performance.

