= Dataset Analysis
The first part of this project will focus on understanding the dataset used throughout the study. The MVETC dataset @mvtec2019 is a widely used benchmark for anomaly detection in industrial inspection tasks and contains multiple object categories, including both normal and defective samples.

The main goals of this stage are:

+ Identify the different object categories present within the dataset (e.g., bottles, cables, capsules, hazelnuts, etc.).
+ Analyse the distribution of normal and anomalous samples within each category and identify any class imbalances.
+ Examine the different defect types present in the dataset (e.g., scratches, cracks, contamination, structural defects) and understand how they are represented visually.
+ Investigate the characteristics of each category, including image resolution, sample count, and defect variability.
+ Identify potential challenges and limitations within the dataset that may influence anomaly detection performance.
+ Choose which categories and defect types to focus on for the subsequent stages of the project based on the analysis.

This stage will provide an understanding of the data and help guide the design of the anomaly detection pipeline used in later stages of the project.

An additional thing that can be looked at later is looking at other anomaly detection datasets to assess how the generalisation of the developed methods and determine whether changes made on MVTec transfer to other domains.

#v(1cm)

#align(center)[
    #figure(
        image("../images/mvtec-example.png",width: 80%),
        caption: ("Example images from the MVTec dataset")
    )
]
