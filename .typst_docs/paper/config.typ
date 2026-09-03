#let easrp(
  title: "",
  accepted: false,
  authors: none,
  affiliation: none,
  body,
) = {
  set page(
    paper: "a4",
    margin: 1in,
    numbering: "1",
  )

  // The style file itself does not force a font.
  // Set 11pt here only if required elsewhere by the submission rules.
  set text(
    size: 11pt,
    lang: "en",
  )

  // Approximate LaTeX onehalfspacing.
  set par(
    leading: 0.5em,
    justify: true,
  )

  // set link(fill: black)

  align(center)[
    #text(size: 17pt, weight: "bold")[#title] \
    #if accepted {
      if authors != none {
        text(size: 12pt)[#authors]
      }

      if affiliation != none {
        v(0.2em)
        text(size: 11pt, style: "italic")[#affiliation]
      }
    } else {
      text(size: 12pt)[Anonymous submission]
    }
  ]

  v(0.5em)

  body
}