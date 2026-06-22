= Analyse image characteristics such as resolution, brightness, sharpness, and contrast.

This section investigates several image characteristics that may influence feature extraction and anomaly detection performance. Resolution, brightness, sharpness, and contrast are analysed to identify potential inconsistencies or preprocessing requirements.

Image-level statistics are computed for every image in the dataset. Brightness is measured as the mean pixel intensity, contrast as the standard deviation of pixel intensities, and sharpness using the variance of the Laplacian.

#let results = csv("../../../data/results/data_analysis/dataset_stats.csv")

#grid(
  columns: 4,
  rows: 2,
  row-gutter: 0.2cm,
  column-gutter: 2cm,
  align: bottom + center,

  [*Category*],
  [*Brightness*],
  [*Sharpness*],
  [*Contrast*],

  table(
    columns: 1,
    align: center,

    ..results.slice(1).map(row => [#row.at(0)])
  ),

  table(
    columns: 2,
    align: center,

    [*mean*], [*std*],

    ..results.slice(1).map(row => (
          [#row.at(1)], 
          [#row.at(2)]
          ))
          .flatten()
  ),

  table(
    columns: 2,
    align: center,

    [*mean*], [*std*],

    ..results.slice(1).map(row => (
          [#row.at(3)], 
          [#row.at(4)]
          ))
          .flatten()
  ),

  table(
    columns: 2,
    align: center,

    [*mean*], [*std*],

    ..results.slice(1).map(row => (
          [#row.at(5)], 
          [#row.at(6)]
          ))
          .flatten()
  )
)

= Observations

Several notable observations can be made from the image characteristic analysis:

- Image resolution is consistent within each category but varies between categories, ranging from 700×700 to 1024×1024 pixels.
- Brightness varies significantly between categories due to differences in object appearance and material properties, but remains highly consistent within individual categories.
- Sharpness and contrast exhibit greater variation between categories, reflecting differences in texture complexity and surface structure.
- Categories such as carpet, wood, and toothbrush contain substantially more texture detail than smoother categories such as capsule and screw.

These findings suggest that resizing and normalization will be necessary during preprocessing, while category-specific texture complexity may influence anomaly detection performance.