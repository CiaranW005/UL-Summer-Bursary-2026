#let org_reuslts = csv("../../../../../../data/results/pretrained/aurocs.csv")
#let hyp_reuslts = csv("../../../../../../data/results/pretrained/hypersphere_aurocs.csv")
#let ell_results = csv("../../../../../../data/results/pretrained/ellipsoid_aurocs.csv")

#grid(
  columns : 4,
  rows : 1,
  column-gutter: 0.3cm,
  align: bottom + center,

  table(
    rows : 15,
    columns : 1,

    ..org_reuslts.slice(1).map(row => (
      [#row.at(0)]
    )),
  ),

  table(
    rows : 16,
    columns : 4,

    [*Centroid*], [*KNN*], [*Avg-KNN*], [*Mahlanobis*],

    ..org_reuslts.slice(1).map(row => (
      [#row.at(1)],
      [#row.at(2)],
      [#row.at(3)],
      [#row.at(4)]
    )).flatten()
  ),

  table(
    rows : 16,
    columns: 2,

    [*Hypersphere*], [*Ellipsoid*],

    ..hyp_reuslts.zip(ell_results).slice(1).map(((h, e)) => (
      [#h.at(1)],
      [#e.at(1)]
    )).flatten(),
  )
)
