
== CLS #sym.arrow.l.r Patch Correlation

This section examines relationships between CLS- and patch-level embedding statistics to determine whether properties of the local patch representation are reflected in the final global representation. These correlations do not directly describe how patch information is aggregated into the CLS token, but may identify local embedding characteristics associated with desirable CLS-level properties. In particular, this analysis considers whether patch-level characteristics could provide useful objectives or diagnostics during future fine-tuning, with the aim of improving CLS metrics such as the Davies-Bouldin Index and Calinski-Harabasz Index. 
#align(center)[
  #figure(
    image("../images/cls_patch/cls_patch_corr.svg"),
    caption: [Correlation matrix between CLS and Patch metrics]
  ) <fig:cls_patch_corr>
]

The patch-level metric most strongly associated with the CLS metrics previously linked to anomaly-detection performance, the Davies-Bouldin Index and Calinski-Harabasz Index, is Patch Train Intra-Cluster Distance. As shown in @fig:cls_patch_corr, Train Intra Distance has correlations of $r_(C H) = -0.70$ with CLS Calinski-Harabasz and $r_(D B) = 0.61$ with CLS Davies-Bouldin. Both relationships are consistent in direction: more compact normal training patch representations, represented by a lower Train Intra Distance, tend to be associated with higher CLS Calinski-Harabasz values and lower CLS Davies-Bouldin values. This suggests that the compactness of the normal patch representation may be related to the quality of the resulting global normal-defective structure. 

The relationship with CLS Davies-Bouldin is shown in @fig:cls_db_patch_train_intra. Although screw lies somewhat outside the main concentration of categories, excluding it has almost no effect on the strength of the linear relationship, with $R^2_(D B)$ changing only from 0.366 to 0.369. This indicates that the relationship is not dependent on the unusual behaviour of screw and instead reflects a moderate trend across the remaining categories. 

A similar relationship is observed for CLS Calinski-Harabasz in @fig:cls_ch_patch_train_intra. Leather represents a more extreme observation in this relationship, and its removal reduces $R^2_(C H)$ from 0.493 to 0.377. Leather therefore contributes more strongly to the observed relationship than screw does in the Davies-Bouldin comparison. However, the relationship remains present after its removal, suggesting that it is not solely produced by this category. Taken together, these results indicate that categories with more compact normal training patch representations tend to exhibit more favourable CLS cluster structure. This is particularly relevant for future fine-tuning because Patch Train Intra-Cluster Distance can be measured using only the normal training representation, making it a potentially useful unsupervised diagnostic for investigating whether changes in local patch compactness are accompanied by improvements in CLS Davies-Bouldin and Calinski-Harabasz.


#align(center)[
  #figure(
    image("../images/cls_patch/cls_db_patch_train_intra.svg", width: 120%),
    caption: [CLS Davies-Bouldin vs Patch Train Intra Dist.]
  ) <fig:cls_db_patch_train_intra>
]

#align(center)[
  #figure(
    image("../images/cls_patch/cls_ch_patch_train_intra.svg", width: 120%),
    caption: [CLS Calinski-Harabasz vs Patch Train Intra Dist.]
  ) <fig:cls_ch_patch_train_intra>
]

The strongest observed relationship between the CLS- and patch-level metrics is between CLS Defect Inter-Cluster Distance and Patch Silhouette, with $r = 0.77$. As shown in @fig:cls_def_inter_patch_sil, the corresponding linear fit has an $R^2$ of 0.587, indicating that approximately 58.7% of the observed category-level variation in CLS Defect Inter-Cluster Distance is associated with its linear relationship with Patch Silhouette. This suggests that categories in which anomalous patches are more clearly separated from normal patches also tend to place their global defective representations further from the learned normal reference. \
The distribution of categories along this relationship is also notable. Several visually more uniform or structurally simpler categories identified during the earlier EDA occupy the upper region of the plot, while the more heterogeneous object categories are concentrated at lower Patch Silhouette values. This initially suggests that the overall correlation may partly reflect systematic differences between these category groups rather than a direct relationship between patch separation and CLS aggregation alone. However, the separate trend lines in @fig:cls_def_inter_patch_sil show that the relationship remains within both groups. The object categories exhibit the stronger within-group relationship at $r = 0.540$ and $R^2 = 0.292$, compared with $r = 0.440$ and $R^2 = 0.194$ for the more uniform categories. The overall relationship is therefore strengthened by the separation between the two groups, but is not solely produced by it. In particular, the continued relationship across the more visually varied object categories provides evidence that local normal-defect separation and global defective displacement are related properties of the pretrained representation rather than simply consequences of category uniformity. \
This relationship is therefore particularly relevant for future fine-tuning experiments. If improvements in patch-level normal-defect separation are consistently accompanied by greater CLS-level defective displacement, Patch Silhouette or related local separation metrics could potentially be monitored or incorporated into a patch-level objective. The object-level trend is especially relevant here, as it suggests that this relationship persists even across categories with substantially greater visual and structural variation. However, the observed correlations do not establish that improving patch separation will directly improve CLS geometry. Determining whether such a relationship is causal would require future experiments that track both patch- and CLS-level metrics throughout fine-tuning and examine whether deliberate changes to the local representation produce corresponding changes in the global representation.


#align(center)[
  #figure(
    image("../images/cls_patch/cls_def_inter_patch_sil.svg", width: 120%),
    caption: [CLS Defect Inter Dist. vs Patch Silhouette]
  ) <fig:cls_def_inter_patch_sil>
]

Another notable relationship is observed between CLS Separation Ratio and Patch Normal Test/Train Intra Ratio, with a Pearson correlation of $r = 0.55$. A weaker relationship is also observed between CLS Separation Ratio and Patch Train Intra-Cluster Distance at $r = -0.44$, while the corresponding relationship with Patch Normal Test Intra-Cluster Distance is weaker again at $r = -0.26$. Together, these results suggest that the relationship between the normal training and test patch distributions may contain more information about global separation than either of the individual intra-cluster distances alone. 

However, the distribution of categories does not follow a particularly clear linear structure, with most categories concentrated around Test/Train ratios close to 1. A Spearman rank correlation was therefore additionally calculated to determine whether the relationship was better described as monotonic rather than linear. This produces a substantially stronger correlation of $rho = 0.75$, compared with the Pearson correlation of $r = 0.55$. When wood and leather, which occupy the extremes of the Patch Intra Ratio, are excluded, the Pearson correlation decreases to $r = 0.48$, while the Spearman correlation remains similar at $rho = 0.72$. This suggests that the association is not solely produced by these extreme categories, but that higher Patch Intra Ratios generally correspond to higher CLS Separation Ratios in a relationship that is not well represented by a single straight line. 

Because a Test/Train Intra Ratio close to 1 represents similar normal-patch dispersion between the training and test distributions, the absolute deviation of the ratio from 1 was also examined. This produces only a weak linear relationship with CLS Separation Ratio at $r = -0.252$, but a stronger negative Spearman correlation of $rho = -0.504$. The direction of this relationship is consistent with categories whose normal patch dispersion remains closer to its training behaviour tending to exhibit greater CLS normal-defect separation. However, the weaker relationship compared with the raw Intra Ratio shows that preservation of normal-patch dispersion alone does not explain the stronger monotonic association observed previously. 

The category-level results further demonstrate this limitation. As shown in @table:intra_distances, wood has the lowest Patch Intra Ratio at 0.8228, indicating substantially more compact normal test patches relative to its normal training representation, and does not exhibit correspondingly strong CLS separation. Leather instead has a ratio of 1.0831, indicating greater normal test-patch dispersion than in its training representation, while bottle has a ratio close to 1 at 0.9763 yet achieves the strongest CLS Separation Ratio. These differences indicate that stability of normal patch dispersion may be one contributing property of the representation, but cannot independently determine global separation. Category-specific visual structure, the characteristics of the anomalous regions, and the way local anomalous information is aggregated into the CLS representation are also likely to contribute.

These observations are particularly relevant when considering future fine-tuning objectives. The relationships identified here suggest that useful patch representations should not necessarily be reduced to a simple objective of maximising normal-defect separation or compressing each group into a single compact region. When a defective image is processed, preserving meaningful local variation across its patch representations may provide the CLS token with a richer representation of both the underlying normal structure and the anomalous regions. The Test/Train Intra Ratio results suggest that substantial contraction of the normal patches within defective images is associated with weaker CLS separation, while the Defect Intra-Cluster Distance results also demonstrate that strong anomaly-detection performance does not require defective patches to collapse into a single compact region. Together, these observations suggest that the geometry retained across the patch representation may be more important than simple class compactness alone. Future fine-tuning experiments should therefore track how patch-level geometry changes alongside CLS Davies-Bouldin, Calinski-Harabasz, Separation Ratio, and AUROC, rather than assuming that a generic contrastive objective such as InfoNCE will improve the representation simply by increasing separation.

#align(center)[
  #figure(
    image("../images/cls_patch/cls_sep_patch_intra_ratio.svg", width: 120%),
    caption: [CLS Separation Ratio vs Patch Intra Ratio]
  ) <fig:cls_sep_patch_intra_ratio>
]
