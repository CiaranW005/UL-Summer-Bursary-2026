#import "@preview/wrap-it:0.1.1": wrap-content

== Patch Embedding Statistics

#let patch_results = csv("../../../data/results/pretrained/patch_embed_summary.csv")

#set text(size: 10pt)

#let cluster_table = table(
  columns: (auto, auto, auto, auto),

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
)

#wrap-content(
  cluster_table,
  [so does typing here lead to this being moved over enough such that itll overflow naturally yes it does very cool and how does this work with overflow,  how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow how does this work with overflow ],
  align: top + right,
  column-gutter: 1em
) 

#let inter_table = table(
    columns: (auto, auto, auto, auto),

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
)

#wrap-content(
  inter_table,
  [so does typing here lead to this being moved over enough such that itll overflow naturally yes it does very cool ],
  align: top + left,
  column-gutter: 1em
) 

#let intra_table = table(
    columns: (auto, auto, auto, auto),

  table.header(
    [*Category*],
    [*Normal*],
    [*Defect*],
    [*Ratio*]
  ),

  ..patch_results.slice(1).map(row => (
    [#row.at(0)],
    [#row.at(7)],
    [#row.at(8)],
    [#row.at(9)]
  )).flatten()
)

#wrap-content(
  intra_table,
  [so does typing here lead to this being moved over enough such that itll overflow naturally yes it does very cool ],
  align: top + right,
  column-gutter: 1em
) 

#scale(
  x: 125%,
  grid(
  columns: 4,
  column-gutter: 0.2cm,
  align: bottom + center,

  table(
    columns: 1,

    ..patch_results.slice(1).map(row => [#row.at(0)])
  ),

  table(
    columns: 1,
    
    [*#patch_results.at(0).at(1)*],

    ..patch_results.slice(1).map(row => [#row.at(1)])
    ),

  table(
    columns: 3,

    [*#patch_results.at(0).at(2)*],
    [*#patch_results.at(0).at(3)*],
    [*#patch_results.at(0).at(4)*],

    ..patch_results.slice(1).map(row => (
      [#row.at(2)],
      [#row.at(3)],
      [#row.at(4)]
    )).flatten()
  ),

  table(
    columns: 3,

    [*#patch_results.at(0).at(5)*],
    [*#patch_results.at(0).at(6)*],
    [*#patch_results.at(0).at(7)*],

    ..patch_results.slice(1).map(row => (
      [#row.at(5)],
      [#row.at(6)],
      [#row.at(7)]
    )).flatten()
  )
)
)

The silhouette scores are generally much higher than those seen for the CLS embeddings, with most categories falling between 0.1 and 0.3. This suggests that patch embeddings form more coherent local structures than global image-level embeddings.

Most categories achieve separation ratios above 1, suggesting that defects are generally positioned further from the normal embedding distribution. Leather (1.61) demonstrates the strongest separation, while metal_nut (≈1.00) exhibits almost no difference between normal and defective patch distances. This indicates that defective metal nut patches occupy nearly the same embedding regions as normal patches.

Most categories have an intra-cluster distance value close to 1, suggesting that defective patches are similarly dispersed to normal patches. However, categories such as leather (1.08) show defective patches becoming more diffuse, while wood (0.82) exhibits more compact defective clusters. This suggests that anomalies affect categories differently, with some defects introducing greater variation and others producing consistent visual patterns.