= Embedding Space Optimisation

This stage of the project will investigate whether the structure of the embedding space can be improved through fine-tuning and alternative training objectives. The goal is to determine whether more distinct boundaries can be established between normal and anomalous samples, thereby improving anomaly detection performance and interpretability.

Experiments will investigate techniques such as sample mining, alternative objective functions, and embedding-space regularisation. These approaches will be explored to determine whether the learned representations can be made more compact, better separated, and more robust to subtle variations within the data.

Particular attention will be given to the formation of neighbourhoods and clusters within the embedding space, and how normal samples relate to one another compared to anomalous samples. Statistical measures developed in earlier stages of the project will be used to analyse whether fine-tuning leads to improved cluster formation and anomaly separability.

Where anomaly detection performance is limited, experiments will investigate whether fine-tuning can make subtle anomalies more distinguishable within the embedding space. The objective is to increase the separation between normal and anomalous samples while preserving the structure of the normal data.

The project will also explore the use of adaptive or dynamic objective functions. Rather than using a fixed objective throughout optimisation, different training objectives may be emphasised at different stages of training based on the observed structure of the embedding space. This will be investigated as a potential mechanism to improve cluster quality, reduce undesirable embedding-space behaviour, and create more representative statistical boundaries.
