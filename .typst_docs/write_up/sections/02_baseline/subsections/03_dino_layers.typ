== DINOv2 Layers

This section examines anomaly-detection performance across all twelve transformer layers of DINOv2 [REF]. The purpose is to determine how the quality of the representation develops through the network and whether the final layers provide sufficient additional performance to justify their computational cost. This is relevant both during inference, where an earlier output could potentially allow the model to terminate before the final transformer blocks, and during future fine-tuning, where an MLP or alternative prediction head could potentially operate on an earlier representation.

#align(center)[
  #figure(
    image("../images/mean_auroc_layer.svg", width: 80%),
    caption: [AUROC across DINO layers]
  ) <fig:auroc_layers>
]

In @fig:auroc_layers, mean AUROC improves steadily across the earlier layers while the variation in performance between categories decreases. From approximately layer 7 onward, the rate of improvement begins to slow, although performance continues to increase through the remaining layers. The difference between layers 11 and 12 is comparatively small, suggesting that most of the useful anomaly-related representation has already developed by this stage.

As shown in @table:auroc_change, the mean AUROC increases from 0.951 at layer 11 to 0.958 at layer 12. Although several individual categories perform marginally better at layer 11, these improvements are small and do not outweigh the higher aggregate performance of the final layer. Layer 12 is therefore retained as the preferred representation for the subsequent experiments. Layer 11 nevertheless provides similar performance and may represent a potential early-exit point where a small reduction in accuracy is acceptable, although omitting only the final transformer block would provide a relatively limited reduction in computation.

#let auroc_change = csv("../../../../../data/results/layers/layer_change.csv")

#align(center)[
  #figure(
    table(
      columns: 4,
      inset: (x: 1em, y: 0.4em),
      stroke: (x: none, y: 0.5pt),

      [*Category*],
      [*Layer 11*], [*Layer 12*],
      [*Change: 12 vs 11*],

      ..auroc_change.slice(1).map(row => (row)).flatten()
    ),
    caption: [AUROC layer 12 vs 11]
  ) <table:auroc_change>
]

