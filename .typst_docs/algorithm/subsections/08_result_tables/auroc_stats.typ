#let org_results = csv("../../../../data/results/pretrained/auroc_stats.csv")
#let hyp_results = csv("../../../../data/results/pretrained/hypersphere_auroc_stats.csv")
#let ell_results = csv("../../../../data/results/pretrained/ellipsoid_auroc_stats.csv")

#grid(
  columns: 4,
  column-gutter: 0.38cm,
  align: bottom + center,

  table(
    columns: 1,
    rows: 6,
    align: center,

    [Centroid], [KNN],
    [Avg-KNN], [Mahalanobis],
    [Hypersphere], [Ellipsoid]
  ),

  table(
    columns: 3,
    rows: 7,
    align: center,

    [*#org_results.at(0).at(1)*],
    [*#org_results.at(0).at(2)*],
    [*#org_results.at(0).at(3)*],

    ..org_results.slice(1).map(row => (
      [#row.at(1)],
      [#row.at(2)],
      [#row.at(3)]
    )).flatten(),

    ..hyp_results.slice(1).map(row => (
      [#row.at(0)], 
      [#row.at(1)], 
      [#row.at(2)]
    )).flatten(),
    
    ..ell_results.slice(1).map(row => (
      [#row.at(0)], 
      [#row.at(1)], 
      [#row.at(2)]
    )).flatten()
  ),

  table(
    columns: 2,
    rows: 7,
    align: center,

    [*Min Category*], [*Value*],

    ..org_results.slice(1).map(row => (
      [#row.at(4)],
      [#row.at(5)]
    )).flatten(),

    ..hyp_results.slice(1).map(row => (
      [#row.at(3)],
      [#row.at(4)]
    )).flatten(),

    ..ell_results.slice(1).map(row => (
      [#row.at(3)],
      [#row.at(4)]
    )).flatten()
  ),

  table(
    columns: 2,
    rows: 7,
    align: center,

    [*Max Category*], [*Value*],

    ..org_results.slice(1).map(row => (
      [#row.at(6)],
      [#row.at(7)]
    )).flatten(),

    ..hyp_results.slice(1).map(row => (
      [#row.at(5)],
      [#row.at(6)]
    )).flatten(),

    ..ell_results.slice(1).map(row => (
      [#row.at(5)],
      [#row.at(6)]
    )).flatten()
  ),
)

#align(center)[
  *AUROCS for each scoring method*
]