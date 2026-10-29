# Uncomputable finite-size gap exponents

Seth Douglas and Nidhal Mghirbi. Version 1.0.0, October 2026. DOI: [10.5281/zenodo.23753840](https://doi.org/10.5281/zenodo.23753840). Concept DOI (all versions): [10.5281/zenodo.23753839](https://doi.org/10.5281/zenodo.23753839). A preprint; it has not been peer reviewed.

We classify the nonnegative finite-size spectral-gap exponents of translation-invariant nearest-neighbour Hamiltonians on the square lattice, with finite local dimension, integer interaction matrices and open boundaries. The gap is measured above the entire ground space on an $L\times L$ box. The existing exponents in $\Delta(L)=L^{-z+o(1)}$ are exactly the nonnegative limit-computable ($\Delta^0_2$) reals. The finite nonnegative lower/upper exponent pairs are exactly the ordered pairs consisting of a left-$\Sigma^0_2$ real and a left-$\Pi^0_2$ real. In particular, one fixed integer-valued model has exponent equal to the halting probability of any chosen prefix-universal machine. A classical boundary preference fixes a size counter and a computation tableau; a reversible mixed-radix counter supplies the small gap. We bound every sector of the full Hilbert space. The closing modes live on a boundary segment of length $O(\log L)$, and the local dimension may depend on the prescribed pair. These are finite-box exponents, with no assertion about bulk dispersion or stability of thermodynamic critical exponents.

## Contents
- `paper/main.pdf` (11 pages), with its LaTeX source (`paper/main.tex`, `paper/references.bib`, `paper/main.bbl`) and a builder
  (`paper/build.py`).
- `paper/arxiv/`: a standalone source folder with the same source and checks.
- `paper/anc/`: the finite checks (exact diagonalization of the counter Hamiltonian, exhaustive minimization of the classical layer, the
  penalty-localization lemma, independent checks), with a runner, recorded outputs and SHA-256 hashes. See `paper/anc/README.txt` for
  their coverage and limitations. The theorems rest on the written proofs; the computations support them and do not replace them.

Related: "The uncomputable price of precision" (Seth Douglas and Nidhal Mghirbi,
[10.5281/zenodo.23753741](https://doi.org/10.5281/zenodo.23753741)).

## License
The paper and documentation are licensed CC BY 4.0, and the scripts MIT or Apache 2.0, at your option. See RIGHTS.md.
