= Investigate the defect types available within each category.

Understanding the variety of defect types present within each category is for assessing dataset complexity. Categories containing a larger number of defect types may present a more challenging anomaly detection task.

The following visualisations show how defect types are distributed within each category's test set. This provides insight into the relative frequency of different anomaly types.

#align(center)[
    #figure(
        image("../images/distribution_defect_sample.png", width: 130%),
        caption: ("A sample of the distributions of defects in the data")
    )
]

The number and distribution of defect types varies substantially between categories. Some categories contain only a small number of anomaly classes, while others contain a wider range of defect variations.

This analysis examines the number of unique defect types present within each category. A larger number of defect types may indicate increased anomaly diversity and greater detection complexity.

#align(center)[
    #figure(
        image("../images/bottle_defect_example.png", width: 100%),
        caption: ("Example of defects in the bottle category")
    )
]