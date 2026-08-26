#set page(
  margin: (
    x: 2.5cm,
    y: 2.1cm,
  ),
)
#set page(
  numbering: "i",
)

#outline()
#pagebreak()

#counter(page).update(1)

#set page(
  numbering: "1",
)

#include "sections/01_eda/main.typ"

#pagebreak()
#include "sections/02_baseline/main.typ"

#pagebreak()
#include "sections/03_embed_visuals/main.typ"

#pagebreak()
#include "sections/04_correlation/main.typ"

#pagebreak()
#include "sections/05_algorithm/main.typ"

#pagebreak()
#include "sections/09x_future_work/main.typ"

#pagebreak()
#bibliography((
  "references/eda.bib",
  "references/baseline.bib",
  "references/embed_visuals.bib",
  "references/algorithm.bib",
  "references/fine_tuning.bib"
))
