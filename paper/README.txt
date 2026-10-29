UNCOMPUTABLE FINITE-SIZE GAP EXPONENTS
Seth Douglas and Nidhal Mghirbi. Version 1.0.0, October 2026. A preprint; it has not been peer reviewed.

main.tex is the main document. references.bib is its bibliography; main.bbl
is supplied for portability. main.pdf is the built manuscript. All TeX inputs
are self-contained. Standard packages for pdfLaTeX are required.

Build in a separate directory:
  python build.py --output-dir ../manuscript-build
The builder runs pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX, checks the final log for
warnings, unresolved references and box diagnostics, and records a PDF hash.

Alternatively copy main.tex and references.bib into an empty directory and run:
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex

anc/ contains finite checks, a runner, recorded outputs and SHA256SUMS.
Read anc/README.txt for their precise coverage and numerical limitations.
arxiv/ is a standalone source distribution with the same source and ancillary
files. It has no dependency on its parent directory.
