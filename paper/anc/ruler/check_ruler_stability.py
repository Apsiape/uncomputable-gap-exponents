"""Finite checks for the manuscript. Standard library, sequential, writes beside this script.

These checks do not certify an infinite lattice construction or noncomputability.
Run: python -u check_ruler_stability.py. The related result is Theorem 20.7.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import math
import time
import tracemalloc

OUT = Path(__file__).resolve().parent


def check_cloning():
    cases = 0
    width, steps = 3, 2
    for truth in product((0, 1), repeat=4):
        def rule(a, b):
            return truth[2*a+b]
        for bits in product((0, 1), repeat=width*(steps+1)):
            hist = [bits[t*width:(t+1)*width] for t in range(steps+1)]
            base_d = sum(hist[t+1][i] != rule(hist[t][i], hist[t][(i+1) % width])
                         for t in range(steps) for i in range(width))
            base_m = sum(sum(row) for row in hist[1:])
            for transverse in (2, 3):
                clone = [[[[hist[t][i] for i in range(width)]
                           for _b in range(transverse)]
                          for _a in range(transverse)]
                         for t in range(steps+1)]
                clone_d = 0
                clone_m = 0
                for t in range(steps):
                    for a in range(transverse):
                        for b in range(transverse):
                            for i in range(width):
                                def corrected(j):
                                    old = clone[t]
                                    return int(old[a][b][j] +
                                               old[(a+1) % transverse][b][j] +
                                               old[a][(b+1) % transverse][j] >= 2)
                                expected = rule(corrected(i), corrected((i+1) % width))
                                clone_d += clone[t+1][a][b][i] != expected
                                clone_m += clone[t+1][a][b][i]
                assert clone_d == transverse**2 * base_d
                assert clone_m == transverse**2 * base_m
                cases += 1
    return {"cases": cases, "status": "PASS", "rules": 16,
            "scope": "all binary two-input rules and all width-3 length-3 histories"}


def check_toom_slab():
    rows = []
    for width in (1, 2, 3, 5):
        for steps in (1, 2, 4, 8, 32):
            hist = [[[int(t == 0) for _b in range(width)]
                     for _a in range(width)] for t in range(steps+1)]
            errors = 0
            for t in range(steps):
                for a in range(width):
                    for b in range(width):
                        expected = int(hist[t][a][b] +
                                       hist[t][(a+1) % width][b] +
                                       hist[t][a][(b+1) % width] >= 2)
                        errors += hist[t+1][a][b] != expected
            avoided = width**2 * steps
            assert errors == width**2
            assert sum(sum(row) for plane in hist[1:] for row in plane) == 0
            rows.append({"width": width, "steps": steps,
                         "errors": errors, "avoided_marks": avoided,
                         "ratio": str(F(errors, avoided))})
    return {"status": "PASS", "cases": len(rows), "rows": rows,
            "scope": "fixed initial plane; not a claimed enclosed ruler-cell compiler"}


def check_residual_preemption():
    rows = []
    for nstar in (6, 12, 24, 60, 120):
        eps = F(3, 2*nstar)
        def q(n):
            return (F(n)-eps*n*n) / n**4
        expected_gc = -F(1, 2*nstar**3)
        assert q(nstar) == expected_gc
        for n in range(2, 3*nstar+1):
            assert q(n) >= expected_gc
            if n != nstar:
                assert q(n) > expected_gc
            assert n**8 - n*n - n >= 0
            assert (n**8 - n*n - n) + n*n + n == n**8
        # Analytically q'(x) has the sign of x-nstar; this scan is not the tail proof.
        onset_window = min(
            F(nstar**4) * (q(n)-expected_gc) / (nstar**4-n**4)
            for n in range(2, nstar))
        t = onset_window / 3
        g = expected_gc + t
        best_energy = -t / nstar**4
        for n in range(2, 3*nstar+1):
            assert (q(n)-g) / n**4 >= best_energy
        assert best_energy < 0
        rows.append({"epsilon": str(eps), "minimizing_n": nstar,
                     "critical_coupling": str(expected_gc),
                     "pitch": nstar**4, "density_jump": str(F(1, nstar**4)),
                     "tested_t": str(t)})
    return {"status": "PASS", "cases": len(rows), "rows": rows,
            "scope": "exact rational algebra; analytic monotonicity proves the infinite tail"}


def check_residual_directions():
    # Large l1 ratios can rotate among coordinates/signs. A finite signed basis
    # nevertheless contains a single perturbation defeating arbitrarily large terms.
    dims = 5
    winners = {(i, sign): [] for i in range(dims) for sign in (-1, 1)}
    for k in range(1, 501):
        i = k % dims
        sign = -1 if (k // dims) % 2 else 1
        residual = [0]*dims
        residual[i] = sign*k*k
        cost = k
        winners[(i, sign)].append(abs(residual[i]) / cost)
    assert all(len(values) == 50 for values in winners.values())
    assert all(max(values) > 450 for values in winners.values())
    return {"status": "PASS", "cases": 500,
            "scope": "rotating signed-coordinate example; finiteness argument is analytic"}


def check_branching():
    cases = 0
    rows = []
    for theta in (F(1, 7), F(3, 7), F(6, 7)):
        selected = 0
        samples = []
        for k in range(1, 100001):
            sign = 1 if k.bit_length() % 2 else -1
            approx = theta + F(sign, k.bit_length()+8)
            approx = min(F(1), max(F(0), approx))
            selected += selected < k*approx
            cases += 1
            if k in (100, 1000, 10000, 100000):
                samples.append({"k": k, "selected": selected,
                                "logarithmic_count_exponent": float(F(selected, k))})
        assert abs(F(selected, 100000)-theta) < F(1, 20)
        rows.append({"test_limit": str(theta), "samples": samples})
    return {"status": "PASS", "cases": cases, "rows": rows,
            "scope": "computable test limits only; no finite check proves noncomputability"}


def check_sparse_mesh():
    # Exact rational pitches n^4, cost n (a=1/4): crossing slopes are ordered.
    pitches = [n**4 for n in range(2, 81)]
    costs = list(range(2, 81))
    crossings = [(F(costs[i], pitches[i]**2)-F(costs[i+1], pitches[i+1]**2)) /
                 (F(1, pitches[i])-F(1, pitches[i+1]))
                 for i in range(len(pitches)-1)]
    assert all(a > b > 0 for a, b in zip(crossings, crossings[1:]))
    for i in range(1, len(pitches)-1):
        t = (crossings[i-1]+crossings[i])/2
        e = -t/pitches[i] + F(costs[i], pitches[i]**2)
        assert e < 0
        assert all(-t/p + F(c, p*p) >= e for p, c in zip(pitches, costs))
    # S17's separated example, with large powers handled in logarithmic form.
    endpoints = []
    for k in range(3, 10):
        logp = (2**k)*math.log(2)
        logt = -logp/2 + math.log1p(-math.exp(-1.5*logp)) - math.log1p(-math.exp(-logp))
        left = (-2*logp)/(logt + math.log1p(-1/(k+10)))
        right = (-logp)/(logt + math.log1p(1/(k+10)))
        endpoints.append({"k": k, "upper_endpoint_sample": left,
                          "lower_endpoint_sample": right})
    assert abs(endpoints[-1]["upper_endpoint_sample"]-4) < 0.01
    assert abs(endpoints[-1]["lower_endpoint_sample"]-2) < 0.01
    return {"status": "PASS", "ordered_crossings": len(crossings),
            "selection_cases": len(pitches)-2, "endpoint_samples": endpoints}


def check_blank_field():
    cases = 0
    for volume in (1, 7, 31):
        for nonblank in range(volume+1):
            for errors in range(volume+1):
                for field in (F(-2, 7), F(0), F(1, 5)):
                    energy = F(3, 2)*errors + field*nonblank
                    lower = volume*min(F(0), field)
                    assert energy >= lower
                    cases += 1
    return {"status": "PASS", "cases": cases,
            "scope": "universal density bound; endpoint configurations imported from primary paper"}


def main():
    started = time.perf_counter()
    tracemalloc.start()
    checks = {}
    for name, fn in (("clone_identity", check_cloning),
                     ("toom_slab", check_toom_slab),
                     ("residual_preemption", check_residual_preemption),
                     ("residual_directions", check_residual_directions),
                     ("branching_schedule", check_branching),
                     ("scale_mesh", check_sparse_mesh),
                     ("blank_field", check_blank_field)):
        checks[name] = fn()
        print(name + ": PASS", flush=True)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {"status": "PASS", "seconds": time.perf_counter()-started,
              "peak_traced_python_bytes": peak,
              "resource_note": "sequential standard-library Python; no worker processes or numerical libraries",
              "checks": checks}
    (OUT / "checks.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print("ALL PASS; %.3f seconds; peak traced allocations %d bytes" %
          (result["seconds"], peak), flush=True)


if __name__ == "__main__":
    main()
