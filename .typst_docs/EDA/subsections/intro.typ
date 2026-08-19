= Exploratory Data Analysis

This notebook presents an Exploratory Data Analysis (EDA) of the MVTec Anomaly Detection@mvtec2019 dataset. MVTec AD is a benchmark dataset for unsupervised anomaly detection in industrial inspection and contains over 5,000 images across 15 object categories. Each category contains defect-free training images and a test set consisting of both normal and anomalous samples.

The purpose of this analysis is to:

+ Identify the object and texture categories present within the dataset.
+ Examine the distribution of normal and anomalous samples.
+ Investigate the defect types available within each category.
+ Quantify the spatial extent of anomalous regions using the provided segmentation masks
+ Analyse image characteristics such as resolution, brightness, sharpness, and contrast.
+ Identify potential challenges and limitations that may influence anomaly detection performance.
+ Inform preprocessing and model design decisions for the subsequent stages of the project.