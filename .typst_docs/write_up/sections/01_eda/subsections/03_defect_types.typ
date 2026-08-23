== Defect Types by Category

The category-level distributions in Figure 3 show that the amount of defective data varies across MVTec AD. However, the number of defective samples alone does not describe the diversity of anomalies represented within each category. The individual defect types are therefore examined to determine how anomaly frequency and variation differ between categories.

#align(center)[
    #figure(
        image("../images/distribution_defect_sample.png", width: 130%),
        caption: ("A sample of the distributions of defects in the data")
    ) <fig:defect_type_example>
]

@fig:defect_type_example shows examples of the distribution of defect types within the test sets. Considerable variation is present both in the number of defect types associated with each category and in their relative frequency. Some categories contain a small number of relatively common defect types, while others contain a wider range of anomalies with more uneven distributions. Consequently, two categories containing similar numbers of defective images may still represent substantially different anomaly-detection problems. For example, categories with a similar overall number of defective samples can still differ substantially in the composition of those defects. Leather contains five defect types, compared with four for metal nut, and the frequency of each defect type is not necessarily uniform. As a result, category-level statistics may be influenced more strongly by defect types that occur frequently, particularly if some anomalies are consistently easier or harder to distinguish than others. The number of normal test samples also varies between categories. While this does not inherently bias AUROC towards categories with more normal samples, smaller sample counts may produce less stable estimates of category-level performance. Consequently, both the diversity and distribution of defect types should be considered when interpreting differences between categories.

#align(center)[
    #figure(
        image("../images/bottle_defect_example.png", width: 100%),
        caption: ("Example of defects in the bottle category")
    ) <fig:bottle_example>
]

Example normal and defective images are also examined to provide visual context for these distributions. These examples illustrate that defect types differ not only in frequency but also in their appearance, spatial extent, and relationship to the underlying object or texture. This variation can be seen in @fig:bottle_example, where the defective samples do not all exhibit visually similar anomalies. Although some defects represent a similar underlying failure, such as broken_small and broken_large, they differ substantially in their spatial extent. Other defect types, such as contamination, can instead appear primarily as a local change in colour or surface appearance. Consequently, defect categories with similar sample counts may still contain anomalies that differ considerably in both scale and visual characteristics. These properties are examined further in the subsequent defect-coverage analysis and later embedding-space analysis, where their relationship with the learned representation is investigated.

=== Defect Size and Image Coverage

Defect severity is not determined solely by defect type, the spatial coverage of an anomaly may also influence detection difficulty. Using the provided segmentation masks, the proportion of each defective image occupied by anomalous regions was calculated to examine how defect size varies both within and between categories.

#let results = csv("../../../../../data/results/data_analysis/defect_coverage_stats.csv")

#align(center)[
  #figure(
    grid(
      columns: (3fr, 1fr),
      gutter: 1em,

            box(
        width: 100%,
        height: 11cm,
      )[
        #image(
          "../images/defect_coverage_boxplots.svg",
          width: 100%,
          height: 100%,
          fit: "contain",
        )
      ],

     box(
        width: 100%,
        height: 11cm,
      )[
        #table(
          columns: 2,
          [*Category*], [*Coverage*],
          
        ..results.slice(1).map(row => (
            [#row.at(0)],
            [#row.at(1)],
            )).flatten()
        )
      ],
    ),
    caption:  [Defect coverage by category accompanied by the average coverage by category.]
  ) #label("fig:defect_coverage")
]

As shown in @fig:defect_coverage, defect coverage varies 
substantially both between and within categories. Metal nut is 
particularly notable: while many defective samples occupy 
approximately 10% of the image, several contain anomalous regions 
approaching 50%. As shown in @fig:defect_type_example, one of the 
metal nut anomaly types consists of a flipped object, which may 
contribute to the substantially larger spatial coverage observed for 
some samples. Further inspection shows that these high-coverage 
samples correspond to the flip defect type, in which a substantial 
portion of the object is spatially altered. This explains the 
unusually large anomalous regions observed for the category and 
highlights how defect type can strongly influence spatial coverage. 
In contrast, screw contains particularly small anomalous regions, 
with defective samples covering only 0.3% of the image on average. 
Such limited spatial coverage may make this category more difficult 
to detect, particularly when using global image representations, 
where the anomalous region contributes only a small proportion of the 
overall representation.
