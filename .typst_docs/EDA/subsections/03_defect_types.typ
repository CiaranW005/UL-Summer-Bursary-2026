= Defect Types by Category

Understanding the variety of defect types present within each category is important for assessing dataset complexity. Categories containing a larger number of defect types may present a more challenging anomaly detection task.

The following visualisations show how defect types are distributed within each category's test set. This provides insight into the relative frequency of different anomaly types.

#align(center)[
    #figure(
        image("../images/distribution_defect_sample.png", width: 130%),
        caption: ("A sample of the distributions of defects in the data")
    ) <fig:defect_type_example>
]

The number and distribution of defect types varies substantially between categories. Some categories contain only a small number of anomaly classes, while others contain a wider range of defect variations.

Examples of normal and defective samples are shown below to illustrate the visual variation between anomaly types. A larger number of certain defect types may indicate increased anomaly diversity and greater detection complexity.

#align(center)[
    #figure(
        image("../images/bottle_defect_example.png", width: 100%),
        caption: ("Example of defects in the bottle category")
    )
]

#pagebreak()

== Defect Size and Image Coverage

Defect severity is not determined solely by defect type, the spatial coverage of an anomaly may also influence detection difficulty. Using the provided segmentation masks, the proportion of each defective image occupied by anomalous regions was calculated to examine how defect size varies both within and between categories.

#let results = csv("../../../data/results/data_analysis/defect_coverage_stats.csv")

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
  ) <fig:defect-coverage>
]

As shown in @fig:defect-coverage, defect coverage varies 
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
