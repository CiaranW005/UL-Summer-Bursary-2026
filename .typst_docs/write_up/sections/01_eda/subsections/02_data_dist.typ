== Normal and Anomalous Sample Distribution

This section examines the overall composition of the dataset in terms of its train/test split and the proportion of normal and defective samples. These distributions provide context for how MVTec AD is structured for anomaly detection, with training data consisting of normal samples and anomalous examples appearing within the test set. Examining these proportions is important for interpreting later evaluation results, particularly because the available numbers of normal and defective test samples influence how category-level performance metrics are estimated.

#v(1cm)
#align(center)[
    #figure(
        image("../images/class_distribution_pie.svg", width: 95%),
        caption: ("Overall normal/defective and train/test distribution of MVTec AD")
    ) <fig:balance_pie>
]

=== Category Distribution

The distribution of images across the 15 MVTec AD categories is first examined to identify whether category-level imbalance could disproportionately influence the subsequent evaluation. As shown in @fig:cat_dist, the dataset is reasonably balanced across categories, although some variation in sample count is present. No individual category dominates the dataset sufficiently to suggest that the overall evaluation would be driven primarily by a single class. 

#align(center)[
    #figure(
        move(dx: -8mm)[
            #image("../images/category_distribution_bar.svg", width:70%)
        ],
        caption: ("Distribution of images across MVTec AD categories")
    ) <fig:cat_dist>
]

However, total category size does not describe how the available samples are distributed between normal and defective images. @fig:anom_prop therefore separates the category distributions according to sample type. The proportion of defective samples varies between categories, meaning that the number of anomalous examples available for evaluation is not uniform across the dataset. This variation should be considered alongside model results, particularly where differences in cluster statistics or anomaly-detection performance are observed.

#align(center)[
    #figure(
        image("../images/defect_distribution_category_chart.svg", width: 70%),
        caption: ("Proportion of normal and defective samples by category")
    ) <fig:anom_prop>
]

Differences in the proportion of defective samples may also be related to the number and frequency of defect types present within each category. This is examined in the following section.
