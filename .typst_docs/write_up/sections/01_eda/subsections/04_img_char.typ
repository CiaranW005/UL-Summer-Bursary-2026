== Image Characteristics 

This section investigates several image characteristics that may influence feature extraction and anomaly detection performance. Resolution, brightness, sharpness, and contrast are analysed to identify potential inconsistencies or preprocessing requirements.

Image-level statistics are computed for every image in the dataset. Brightness is measured as the mean pixel intensity, contrast as the standard deviation of pixel intensities, and sharpness using the variance of the Laplacian.

#let group-header(title) = pad(
  left: 5pt,
  right: 5pt,
  block(
    width: 100%,
    inset: (bottom: 3pt),
    stroke: (bottom: 0.5pt),
    align(center)[#title],
  )
)

#let results = csv("../../../../../data/results/data_analysis/dataset_stats.csv")

#figure(
  table(
    columns: (
      1.4fr,
      1fr, 1fr,
      1fr, 1fr,
      1fr, 1fr,
    ),
    align: center,
    inset: (x: 6pt, y: 3pt),

    stroke: (x, y) => (
      x: none,
      y: if y >= 2 { 0.25pt } else { none },
    ),

    table.cell(rowspan: 2)[*Category*],

    table.cell(colspan: 2)[
      #group-header([*Brightness*])
    ],
    table.cell(colspan: 2)[
      #group-header([*Sharpness*])
    ],
    table.cell(colspan: 2)[
      #group-header([*Contrast*])
    ],

    [*mean*], [*std*],
    [*mean*], [*std*],
    [*mean*], [*std*],

    ..results.slice(1).map(row => (
      [#row.at(0)],

      [#row.at(1)],
      [#row.at(2)],

      [#row.at(3)],
      [#row.at(4)],

      [#row.at(5)],
      [#row.at(6)],
    )).flatten()
  ),
  caption: [Image-level statistics by MVTec AD category.]
)

=== Observations

Several notable observations can be made from the image characteristic analysis:

- Image resolution is consistent within each category but varies between categories, ranging from 700x700 to 1024x1024 pixels. Higher-resolution images may   preserve finer defect details, while resizing may reduce the visibility of very small anomalies, particularly when images are resized to the 224x224 resolution used in this work.

- Brightness varies significantly between categories due to differences in object appearance and material properties, but remains highly consistent within individual categories. Categories with greater variation in brightness may be more difficult to model consistently, as changes in illumination could resemble or obscure anomalous regions.

- Sharpness and contrast exhibit greater variation between categories, reflecting differences in texture complexity and surface structure. Higher contrast may make local defects easier to distinguish where the anomaly differs strongly from the surrounding surface. Differences in sharpness may also influence how clearly small structural defects are represented, particularly where anomalies depend on fine edges or surface details.

- Categories such as carpet, wood, and toothbrush contain substantially more texture detail than smoother categories such as capsule and screw. Categories with greater texture complexity may be more challenging because normal appearance already contains substantial local variation, while visually smoother and more consistent categories may allow anomalous regions to stand out more clearly against the normal appearance.

These observations provide several hypotheses regarding how image characteristics may influence anomaly-detection difficulty, which can be investigated in subsequent analysis and future work.