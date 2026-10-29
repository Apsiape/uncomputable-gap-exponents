FINITE CHECKS

Run from any directory with Python 3, NumPy and SciPy installed:
  python -u run_checks.py
The runner resolves its own directory, runs scripts sequentially, sets numerical
library thread counts to one, checks the JSON results and records console output.
The supplied outputs were produced with Python 3.13.12, NumPy 2.4.3 and SciPy
1.17.1. Other versions may change last digits and elapsed times.

Coverage:
  check_dynamic_counter.py constructs full register product spaces for twelve
  small radix patterns, up to dimension 262144, and diagonalizes each transition
  component. It includes all head counts and illegal digits. Its last-cell
  operation is the register restriction of the guarded bond term in the paper;
  it is not a separate full two-dimensional simulation.

  check_classical_layer.py minimizes over every configuration of the binary
  counter and three boundary flags on boxes of side 2, 3 and 4 by a transfer
  profile dynamic program. It does not encode the arbitrary machine tableau.
  Energies and the minimizing multiplicity are checked against the forced
  tableau. The analytic four-flag identity controls all box sizes.

  check_penalty_lemma.py tests 2000 seeded path, cycle, tree, bounded-graph and
  star instances, with 3 through 89 vertices, against the penalty inequality.

  independent_checks.py uses a direct state-graph implementation. It checks all
  radix words of length at most three over {1,2,4}, a four-cell binary example,
  and six frozen-exterior examples. It retains parallel edge weights. It also
  enumerates connected simple graphs through five vertices and all nonempty
  proper penalty sets; enumerates the complete potentially minimizing sector
  of a four-flag 2 by 2 counter layer (all omitted flags already cost at least
  two); checks carry/region identities; and verifies exact rational prefix
  identities for mechanical words. These checks have no imports from the other
  three scripts.

The JSON and _output.txt files record results. summary.json collects coverage.
The linear algebra is floating point, not interval-certified. Numerical
diagonalization does not replace the proofs, verify a noncomputable limit,
or certify arbitrary full two-dimensional machine tableaux. The algorithms
operate sequentially; the largest classical profile uses less than 1.5 GB.

SHA256SUMS covers every other file in this directory. To verify it:
  python verify_hashes.py
The runner can overwrite records with fresh timings; after rerunning it, the
original checksum file is expected to differ until intentionally regenerated.
