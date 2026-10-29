Finite consistency checks
Version 2.1.0, October 2026. A preprint; it has not been peer reviewed.

Run from this directory with Python, NumPy, SciPy and psutil installed:
    python -u run_checks.py
    python verify_hashes.py
The runner invokes programs sequentially, fixes numerical-library threads to
one, disables bytecode writes and monitors the process tree below 1 GB.
All scripts write only their local recorded output and JSON result files.

Coverage:
check_penalty_lemma.py: random bounded graph penalty estimates.
check_dynamic_counter.py: complete finite register product spaces, including
illegal digits, head sectors, repeated edges and the main-orbit upper witness.
check_classical_layer.py: exhaustive small classical profile minimization.
independent_checks.py: separate finite-size engine, four flags, frozen sites,
parallel edges, carry recursion and exact rational word interpolation.
check_wall_layer.py: exhaustive floating-point min-plus transfers for a
six-symbol enlarged wall alphabet. Both state-space search and weights are
recorded; the arithmetic is double precision, not exact rational arithmetic.
check_wall_layer_noO.py: the actual five-symbol wall alphabet.
check_counter_perturbation.py: full finite product spaces with digit mixing;
36 samples include the unperturbed case and two positive strengths.
check_film_counter.py: a second product-space counter implementation. Its
main-cycle equality test holds on its samples, not for every radix vector.
check_film_classical_layer.py: six face preferences and unique carry/output
transport solutions in randomized and reversed assignment orders. No arbitrary
program's machine tableau is simulated by this finite constraint search.
check_nested_limits.py: independent counter components, the seeded legal-path
counterexample, flag identity, thickness counting, binary delimiters, uniform
name obstruction, rational words and finite persistent-write schedules.
check_critical_ruler.py: exact enumeration of wall configurations and charging
deficits, exact wetting-track minima, height-tie residues, phase excitations,
legal counter components and rational pitch selection.

check_bulk_broadcast.py: literal three-field copying graphs including deliberately
noncommensurate cases, local/global prefix masks, exact ruler minimization and
charging on small tori, tie alignment, inert-ring upper witnesses, tensor sums,
retained-stage obstruction and rational Lipschitz interpolation.
check_bulk_independent.py: separate GF(2) elimination for translated broadcast
graphs and all finite source fillings, forced contradictions from wrong bits,
defective rulers, full native counter products, the long legal-path example,
degenerate-kernel tensor sums, pitch ties, guard positions and reader counts.
These two programs use only the Python standard library; they import none of
the other checks. They do not enumerate a general machine's full alphabet or
the complete three-dimensional Hamiltonian, prove a convergence modulus, or
check every possible ground-state phase. Noncommensurate collapse and the
failure of universal main-cycle equality are expected, reproduced outcomes.

The independent component checks explicitly exhibit legal paths whose
positive gap is smaller than the main cycle's. PASS certifies only the actual
bracket, local tests and finite predictions, including these counterexamples.
The wetting tie examples explain why the manuscript uses factorial open cubes.
summary.json records the finite-size suite; all_checks_summary.json records
all groups and sampled process-tree RSS. Outputs ending in _output.txt and
their JSON companions are retained. SHA256SUMS covers every ancillary file
except the manifest itself. A rerun changes timing fields; regenerate the
manifest only when intentionally accepting new recorded outputs.

Limitations:
Floating-point eigenvalues are finite consistency checks, not interval proofs.
No full two- or three-dimensional many-body matrix is diagonalized, no general
target program or uncomputable limiting real is numerically certified, and no
arbitrary-level Hamiltonian is formally verified. The manuscript's analytic
proofs control those statements. Thermal ceilings are not attainment results.

Additional groups:
registers/: shared-coordinate thickness masks, translated final-row sources,
literal broadcast fillings, ruler defects, tensor minima and chain spectra.
states/: finite stationary and Ising calculations, independent cone majority,
causal histories, full DLR conditionals and variable-normalizer counterchecks.
restricted/: cycle bounds, sparse conditional variance, actual MPS contractions,
rational dual-grid envelopes and feedback tracking.
ruler/: transverse cloning, static slab competitors, calibrated finite-pitch
preemption, schedule counts, logarithmic meshes and blank-field bounds.
inquiry/: two independent finite implementations verify adaptive inverse-CDF
coupling, random-limit versus fixed-target quantifiers, the factor-two
logarithmic estimate, factorial rounding and exact ruler ties. These are finite
consistency checks; they do not prove randomized reach, noncomputability or the
absence of a computable settling bound. Every intended pass condition raises
on failure; the primary inquiry checker also rejects three corruptions.
robust/: majority bounds, finite mechanical blocks, counting exponents, toy
late revisions, positive-noise countdown histories, independent all-prefix and
rational-power checks, protected output invariants, drift and disjoint cones.
These are finite illustrations for Sections 20.6-20.9, especially Theorems
20.9 and 20.12 and Proposition 20.11. The countdown is capped and its plane
majorities have overlapping cones; its table shows a finite transient, not
noncommuting infinite limits. The independent simulation uses disjoint cones.
The finite noise rates are not certified members of an all-time reliability
region; toy programs have computable settling and establish no noncomputability.

The sequential psutil runner monitors memory across the child process tree.
Individual checkers omit platform-specific peak-memory fields when those APIs
are unavailable. The numerical checks do not depend on Windows memory APIs.
The runner labels its measurements sampled_peak_rss_bytes: sampled process-tree
RSS, not a process's historical peak. The exhaustive classical profile checker
uses exact small-integer energies to reduce memory while retaining every case.
