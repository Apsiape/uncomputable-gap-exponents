"""Independent finite consistency checks. Standard library; sequential; no input imports.

Run: python -u independent_checks.py. All generated files remain beside this file.
Finite checks exercise identities and counterexample attempts, not infinite limits.
"""
import os
for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[thread_var] = "1"
import ctypes
from ctypes import wintypes
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import random
import time

OUT = Path(__file__).resolve().parent


def open_path_bound():
    """Ternary range-one graph: cycle bound versus independent DP on open paths."""
    rng = random.Random(8675309)
    count = 0
    for case in range(25):
        weights = {(u, v): rng.randrange(-13, 14) for u in range(3) for v in range(3)}
        # All simple cycles in a complete three-vertex digraph, with duplicates harmless.
        means = [F(weights[u, u]) for u in range(3)]
        means += [F(weights[u, v] + weights[v, u], 2)
                  for u in range(3) for v in range(u + 1, 3)]
        means += [F(weights[0, u] + weights[u, v] + weights[v, 0], 3)
                  for u, v in ((1, 2), (2, 1))]
        minimum = min(means)
        W = max(abs(w) for w in weights.values())
        distances = [0] * 3
        for length in range(1, 121):
            distances = [min(distances[u] + weights[u, v] for u in range(3))
                         for v in range(3)]
            # Residual simple path has at most V-1 = 2 edges.
            assert min(distances) >= length * minimum - 2 * W * 2
            count += 1
    return {"status": "PASS", "exact_open_path_bounds": count,
            "scope": "25 ternary potentials, lengths 1..120; DP distinct from cycle enumeration"}


def thermal_conditional_factorization():
    """3-state 3x3 open systems, signed bonds and plaquette interactions."""
    sites = list(product(range(3), repeat=2))
    index = {site: j for j, site in enumerate(sites)}
    selected = [index[x] for x in ((0, 0), (0, 2), (2, 0), (2, 2))]
    outside = [j for j in range(9) if j not in selected]
    spin, charge = (-1, 0, 1), (0, 1, 3)
    results = []
    total_contexts = 0
    for jx, jy, plaquette, field in ((.23, -.31, .17, .12), (-.18, -.27, -.14, -.2)):
        terms = []
        for a, b in sites:
            if a < 2:
                terms.append((jx, (index[a, b], index[a + 1, b])))
            if b < 2:
                terms.append((jy, (index[a, b], index[a, b + 1])))
            if a < 2 and b < 2:
                terms.append((plaquette, tuple(index[x] for x in
                              ((a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1)))))
        assert all(len(set(support).intersection(selected)) <= 1 for _, support in terms)
        B = max(3 * abs(field) + sum(2 * abs(c) for c, support in terms if j in support)
                for j in range(9))
        beta = .7
        pmin = math.exp(-beta * B) / 3
        z = zn = zn2 = conditional_var_sum = mean_square_sum = 0.
        worst_factorization_error = 0.
        smallest_probability = 1.
        for outer in product(range(3), repeat=len(outside)):
            context = [0] * 9
            for j, a in zip(outside, outer):
                context[j] = a
            weighted = []
            for inner in product(range(3), repeat=4):
                for j, a in zip(selected, inner):
                    context[j] = a
                energy = -field * sum(charge[a] for a in context)
                for c, support in terms:
                    value = c
                    for j in support:
                        value *= spin[context[j]]
                    energy += value
                weighted.append((inner, math.exp(-beta * energy), sum(charge[a] for a in context)))
            cz = sum(w for _, w, _ in weighted)
            means = [[sum(w for inner, w, _ in weighted if inner[j] == a) / cz
                      for a in range(3)] for j in range(4)]
            smallest_probability = min(smallest_probability, *(p for row in means for p in row))
            cn = sum(w * n for _, w, n in weighted) / cz
            cn2 = sum(w * n * n for _, w, n in weighted) / cz
            cv = cn2 - cn * cn
            assert cv >= 4 * pmin * pmin * 9 - 1e-10
            for inner, w, _ in weighted:
                predicted = math.prod(means[j][a] for j, a in enumerate(inner))
                worst_factorization_error = max(worst_factorization_error, abs(w / cz - predicted))
            z += cz
            zn += cz * cn
            zn2 += cz * cn2
            conditional_var_sum += cz * cv
            mean_square_sum += cz * cn * cn
            total_contexts += 1
        assert smallest_probability >= pmin - 1e-12
        variance = zn2 / z - (zn / z) ** 2
        explained = mean_square_sum / z - (zn / z) ** 2
        assert abs(variance - conditional_var_sum / z - explained) < 1e-10
        assert explained >= -1e-10
        assert worst_factorization_error < 1e-12
        curvature = beta * variance / 9
        lower = beta * F(4, 9) * pmin * pmin * 9
        assert curvature >= lower
        results.append({"B": B, "curvature": curvature, "finite_volume_bound": float(lower),
                        "minimum_conditional_probability": smallest_probability,
                        "probability_lower_bound": pmin,
                        "max_factorization_error": worst_factorization_error})
    return {"status": "PASS", "outside_contexts": total_contexts,
            "full_configurations": 2 * 3**9, "cases": results}


def jump_and_tangent():
    """The same linear onset has different vacuum-relative and right-tangent densities."""
    for p in (2, 16, 81):
        for t in (F(1, 11), F(1, 101), F(1, 1001)):
            e = -t / p
            right_tangent = -F(1, p)
            assert right_tangent * t - e == 0
            assert -right_tangent == F(1, p)
    # Thermal-style kink plus smooth background: jump 1, right-excess c*t.
    c = F(3, 7)
    for t in (F(1, 11), F(1, 101), F(1, 1001)):
        pressure = t + c * t * t / 2
        assert pressure - t == c * t * t / 2
        assert (1 + c * t) - 1 == c * t
    return {"status": "PASS", "exact_identities": 15,
            "finding": "BUILD jump exponent 0 is vacuum-relative; S17 right-tangent excess is identically zero"}


def preemption_and_mesh():
    rows = []
    for N in (3, 4, 7, 13, 29):
        eps = F(3, 2 * N)
        gc = -F(1, 2 * N**3)
        threshold = lambda n: F(1, n**3) - eps / n**2
        # Exact difference factorization proves every real n>0, hence the infinite tail:
        # h(n)-h(N) = (n-N)^2 * (n+2N) / (2*N^3*n^3).
        for n in range(2, 8 * N + 1):
            gap = F((n - N)**2 * (n + 2 * N), 2 * N**3 * n**3)
            assert threshold(n) - gc == gap
        # Only smaller n can overtake the largest density branch on the right.
        window = min((threshold(n) - gc) / (1 - F(n**4, N**4)) for n in range(2, N))
        t = min(window / 2, -gc / 2)
        for n in range(2, 8 * N + 1):
            assert (threshold(n) - gc - t) / n**4 >= -t / N**4
        rows.append({"N": N, "epsilon": str(eps), "gc": str(gc),
                     "rho_jump": str(F(1, N**4)), "tested_t": str(t)})
    # A deliberately tied minimum; finite residual has zero tail.
    for t in (F(1, 20), F(1, 100), F(1, 10000)):
        energies = [-t / 16, -t / 81]
        assert min(energies) == -t / 16
    # Logarithmically dense but unbounded-ratio mesh: p_k=2^(2*k^2), C_k=2^(k^2).
    # Use logarithms for scale safety. Consecutive crossings determine all selection intervals.
    endpoints = []
    for k in (10, 30, 100):
        logp, logpnext = 2 * k*k * math.log(2), 2 * (k+1)**2 * math.log(2)
        logtcross = -.5 * logp + math.log1p(-math.exp(-1.5 * (logpnext - logp))) - math.log1p(-math.exp(-(logpnext - logp)))
        endpoints.append({"k": k, "density_order_before": -logpnext / logtcross,
                          "density_order_after": -logp / logtcross})
    assert abs(endpoints[-1]["density_order_before"] - 2) < .041
    return {"status": "PASS", "exact_preemption": rows,
            "tie_cases": 3, "log_dense_mesh_samples": endpoints}


def multiply(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2))
                 for i in range(2))


def pauli_finite_ring():
    """Direct 4^6 amplitudes, independent of transfer eigenvector checks."""
    ident = ((1, 0), (0, 1))
    paulis = (((0, 1), (1, 0)), ((0, -1j), (1j, 0)), ((1, 0), (0, -1)))
    length = 6
    rows = []
    for t in (.25, .5):
        tensors = (ident,) + tuple(tuple(tuple(t * z for z in row) for row in p) for p in paulis)
        amplitudes = {}
        for word in product(range(4), repeat=length):
            m = ident
            for digit in word:
                m = multiply(m, tensors[digit])
            amplitudes[word] = m[0][0] + m[1][1]
        norm = sum(abs(a)**2 for a in amplitudes.values())
        l0, l1 = 1 + 3 * t*t, 1 - t*t
        assert abs(norm - (l0**length + 3*l1**length)) < 1e-10
        for separation in (1, 2, 3):
            numerator = 0j
            onepoint = 0j
            for word, amplitude in amplitudes.items():
                if word[0] in (0, 1):
                    changed = list(word)
                    changed[0] = 1 - changed[0]
                    onepoint += amplitude.conjugate() * amplitudes[tuple(changed)]
                    if word[separation] in (0, 1):
                        changed[separation] = 1 - changed[separation]
                        numerator += amplitude.conjugate() * amplitudes[tuple(changed)]
            assert abs(onepoint) < 1e-12
            expected = 4*t*t * (l1**(separation-1)*l0**(length-separation-1)
                                + l0**(separation-1)*l1**(length-separation-1)) / norm
            assert abs(numerator / norm - expected) < 1e-12
            rows.append({"t": t, "separation": separation, "correlation": expected})
    return {"status": "PASS", "wavefunctions_enumerated": 2, "amplitudes": 8192,
            "correlations": rows}


def polyhedral_dual():
    """New exact profile rho=u^2; c(r)=(2/3)r^(3/2), rational at square nodes."""
    count = 0
    # For arbitrary r_j use certified rational square-root enclosures via integer sqrt.
    for N in (3, 7, 13, 31):
        grid = []
        precision = 16 * N*N
        for j in range(N + 1):
            r = F(j, N)
            root_floor = math.isqrt(j * precision**2 // N)
            low = F(root_floor, precision)
            high = low + F(1, precision)
            assert low*low <= r <= high*high
            c_low = F(2, 3) * r * low
            k = (N * c_low).numerator // (N * c_low).denominator
            a = F(k, N)
            # Approximation is within 1/N + small enclosure error; nearest rounding fixes it.
            mid = F(2, 3) * r * (low + high) / 2
            k = (N * mid + F(1, 2)).numerator // (N * mid + F(1, 2)).denominator
            a = F(k, N)
            assert a - F(2, 3)*r*high >= -F(1, N)
            assert a - c_low <= F(1, N)
            grid.append((a, r))
        for A in (F(1, 3), F(1), F(7, 3)):
            for u in (F(-2), F(0), F(1, 11), F(2, 3), F(1), F(3)):
                h = A * u
                s = F(0) if u <= 0 else u**3 / 3 if u < 1 else u - F(2, 3)
                exact = -A * s
                rounded = min(A*a - h*r for a, r in grid)
                assert exact - A/N <= rounded <= exact + (2*A + abs(h))/N
                count += 1
    return {"status": "PASS", "exact_parameter_cases": count,
            "profile": "rho(u)=u^2, Legendre dual certified by rational sqrt intervals"}


def schedule_tracking():
    """Adversarial changing targets: verify finite tracking invariants, not just endpoints."""
    total = 0
    for lower, upper in ((F(1, 9), F(8, 9)), (F(3, 7), F(4, 7)), (F(1, 3), F(2, 3))):
        s = 0
        for k in range(1, 15001):
            target = lower if (k // 37) % 2 else upper
            s += int(s < k * target)
            assert lower*k - 1 <= s <= upper*k + 1
            total += 1
    return {"status": "PASS", "exact_tracking_steps": total,
            "scope": "invariants tolerate target jumps; convergence itself proved analytically"}


def peak_memory():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    assert c.PeakWorkingSetSize < 10**9
    return c.PeakWorkingSetSize


if __name__ == "__main__":
    started = time.perf_counter()
    result = {}
    for test in (open_path_bound, thermal_conditional_factorization, jump_and_tangent,
                 preemption_and_mesh, pauli_finite_ring, polyhedral_dual, schedule_tracking):
        result[test.__name__] = test()
        print(test.__name__ + ": PASS", flush=True)
    result["peak_working_set_bytes"] = peak_memory()
    result["seconds"] = time.perf_counter() - started
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "independent-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ALL PASS", "seconds": result["seconds"],
                      "peak_working_set_bytes": result["peak_working_set_bytes"]}), flush=True)
