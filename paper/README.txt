Uncomputable critical exponents
Seth Douglas and Nidhal Mghirbi
Version 2.0.0, October 2026. A preprint; it has not been peer reviewed.

main.pdf is the manuscript. main.tex inputs the named section files.
references.bib and main.bbl provide the bibliography.
Build in an external directory: python build.py --output-dir BUILD_DIRECTORY
The builder runs pdflatex, BibTeX and two further pdflatex passes, then rejects
warnings, undefined references and overfull or underfull boxes.
arxiv/ is a self-contained source package. anc/ contains all finite checks.
The thermodynamic gap conventions are specified in the manuscript: factorial
tori for both bulk rulers and factorial open cubes for the wetting ruler.
The bulk broadcast theorem uses the gap above the entire ground space;
it does not identify a locally nondegenerate or KMS-extremized gap.
