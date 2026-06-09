= Extended Work

If time permits towards the end of the bursary, additional work will focus on improving the interpretability of anomaly detection decisions. This will build on the outputs of previous stages of the project and investigate ways to make anomaly detection decisions easier to understand and explain.

Potential areas include:

- Identifying and visualising the patch regions that contribute most strongly to an anomaly decision.

- Highlighting the likely location of defects within an image using patch-level anomaly scores.

- Retrieving similar normal and anomalous samples from the embedding database to provide additional context for anomaly decisions.

A further extension would investigate using a small local language model, such as Mistral, to generate natural-language explanations for anomaly detection. The model could be adapted using LoRA fine-tuning and provided with statistical outputs from the anomaly detection pipeline, including anomaly scores, embedding-space distances, nearest-neighbour results, and patch-level information.

The objective would be to generate human-readable explanations describing why an image was classified as anomalous and which characteristics contributed most strongly to the decision. This could provide an additional layer of interpretability beyond the anomaly scores and visualisations provided.

