= Embedding Pipeline

This stage of the project will focus on developing a reusable embedding extraction pipeline. The purpose of this pipeline is to avoid repeatedly processing the same images through the Vision Transformer model during experimentation.

*Vision Transformer Selection*

_A pre-trained Vision Transformer will be used as the primary feature-extraction model. The first implementation will focus on DINO @oquab2024dinov2learningrobustvisual family of models due to its strong performance as a general-purpose visual representation model._

_Alternative architectures may also be looked at to determine how model choice influences embedding quality and anomaly detection performance._

The dataset will first be organised into a structured metadata table containing information such as image categories, defect types, labels, and train/test splits. This metadata will allow normal and anomalous samples to be queried and compared during experiments.

Each image will then be passed through the model to extract both global and local representations. The global representation will be obtained from the CLS token, while local representations will be obtained from patch-level embeddings. These embeddings will be stored separately so that global, patch-level, and hybrid approaches can be compared later in the project.

The planned storage approach is to use SQLite for metadata and FAISS for embeddings. SQLite will provide a way to query image-level information, while FAISS will support similarity search and nearest-neighbour lookup in embedding space. This structure will make it easier to run experiments repeatedly without recomputing embeddings each time.

#v(1cm)
#align(center)[
    #figure(
        image("../images/bursary-architecture.png", width: 40%),
        caption: ("Diagram of the embedding extraction pipeline")
    )
]
