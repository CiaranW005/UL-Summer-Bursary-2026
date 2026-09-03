#import "config.typ": easrp

#set heading(numbering: "1.")
#set page(numbering: "1")

#show: easrp.with(
  title: "Cloud of Ellipsoids: Local Geometric Modelling of DINOv2 Embeddings for Unsupervised Anomaly Detection",
  accepted: false
)

#include "sections/00_abstract/main.typ"

#include "sections/01_intro/main.typ"

#include "sections/02_related_work/main.typ"

#include "sections/03_method/main.typ"

#include "sections/04_results/main.typ"

#include "sections/05_discussion/main.typ"

#include "sections/06_conclusion/main.typ"

#bibliography((
  "references/intro.bib",
  "references/related_work.bib",
  "references/method.bib",
  "references/results.bib",
  "references/discussion.bib"
))