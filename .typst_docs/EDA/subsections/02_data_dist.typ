= Normal and Anomalous Sample Distribution

This section examines how samples are distributed across the dataset. Understanding the balance between normal and anomalous images helps identify class imbalances that may influence anomaly detection performance and evaluation metrics.

#v(1cm)
#align(center)[
    #figure(
        image("../../../images/EDA/class_distribution_pie.svg", width: 105%),
        caption: ("Overall normal/defective and train/test distribution of MVTec AD")
    )
]

== Category Distribution

This investigates how images are distributed across the 15 MVTec categories. Identifying category-level imbalances helps determine whether certain categories may be overrepresented during evaluation.

#align(center)[
    #figure(
        move(dx: -8mm)[
            #image("../../../images/EDA/category_distribution_bar.svg", width:75%)
        ],
        caption: ("Distribution of images across MVTec AD categories")
    )
]

The dataset is reasonably balanced across categories, although some categories contain more images than others. No category dominates the dataset to a degree that would significantly bias overall evaluation.


This figure shows the proportion of normal and defective samples within each category. As MVTec is designed for anomaly detection, categories may contain different ratios of normal and anomalous images depending on the number of defect types available.

#align(center)[
    #figure(
        image("../../../images/EDA/defect_distribution_category_chart.svg", width: 75%),
        caption: ("Proportion of normal and defective samples by category")
    )
]

The proportion of defective samples varies between categories. Categories containing a larger number of defect types generally show a higher proportion of anomalous test samples.

