== Adaptive Multi-Hypersphere Fitting

This cell implements the full unsupervised hypersphere fitting algorithm. The aim is to cover all normal training embeddings using a set of local hyperspheres, while ensuring that each training embedding is assigned to only one region.

The algorithm proceeds as follows:

+ *Initialise uncovered embeddings*  
   All normal embeddings begin as uncovered. A boolean mask is used to track which embeddings have not yet been assigned to a hypersphere.

+ *Find the densest uncovered region*  
   For the current set of uncovered embeddings, a FAISS KNN search is performed. The embedding with the smallest average distance to its neighbours is selected as the centre of the densest local  region.

+ *Create an initial candidate hypersphere*  
   The densest embedding and its nearest neighbours form the first candidate region. This candidate is cleaned to ensure it does not contain embeddings already assigned to previous hyperspheres.

+ *Grow the candidate hypersphere*  
   If the initial candidate is valid, the hypersphere is expanded using a small growth factor. Newly included uncovered embeddings are added to the candidate region.

+ *Clean after growth*  
   After each growth step, the candidate is cleaned again. If the cleaned candidate contains more embeddings than before, the growth is accepted. Otherwise, growth stops.

+ *Store the final hypersphere*  
   Once no further valid growth is possible, the final centroid, radius, and assigned embedding indices are stored. These embeddings are then marked as covered.

+ *Repeat until full coverage*  
   The process repeats until every normal training embedding has been assigned to a hypersphere.

#figure(
  image("../../../images/ellipsoid/base_embeds/sphere_plot/screw.svg", width: 800pt, height: 300pt),
  caption: [Hypersphere PCA plot for screw category]
)