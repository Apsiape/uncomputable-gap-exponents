"""Finite sanity checks for the manuscript; no numerical test proves computability.

Run: python -u check_restricted.py. Standard library only; sequential.
"""
import ctypes
from ctypes import wintypes
from fractions import Fraction as F
from itertools import product
import json
import math
import os
import random
import sys


def cycle_means(weights, radius):
    vertices = list(product(range(2), repeat=radius))
    cycles = set()

    def visit(start, path):
        last = path[-1]
        for digit in (0, 1):
            nxt = last[1:] + (digit,)
            if nxt == start:
                rotations = [tuple(path[i:] + path[:i]) for i in range(len(path))]
                cycles.add(min(rotations))
            elif nxt not in path:
                visit(start, path + [nxt])

    for v in vertices:
        visit(v, [v])
    means = []
    for cycle in cycles:
        total = sum(weights[cycle[i] + (cycle[(i + 1) % len(cycle)][-1],)]
                    for i in range(len(cycle)))
        means.append(F(total, len(cycle)))
    return means


def check_cycles():
    rng = random.Random(1729)
    checked = 0
    for radius in (1, 2, 3):
        words = list(product(range(2), repeat=radius + 1))
        for _ in range(12):
            weights = {w: rng.randrange(-7, 8) for w in words}
            best = min(cycle_means(weights, radius))
            observed = []
            for length in range(1, 10):
                torus_best = min(
                    F(sum(weights[tuple(spins[(i + j) % length]
                                             for j in range(radius + 1))]
                          for i in range(length)), length)
                    for spins in product(range(2), repeat=length))
                assert torus_best >= best
                observed.append(torus_best)
                checked += 1
            # A minimum simple cycle has length at most 2**radius.
            assert min(observed[:2 ** radius]) == best
    return {"torus_minima_checked_exactly": checked, "status": "PASS"}


def check_polyhedral_grid():
    # Computable member rho(u)=u on [0,1]. Its dual c(r)=r*r/2.
    # The theorem works with an arbitrary computable dual via its compact supremum.
    checked = 0
    worst_normalized_error = 0.0
    for n in (4, 8, 16, 32, 64):
        grid = [(F((j * j) // (2 * n), n), F(j, n)) for j in range(n + 1)]
        for a in (F(1, 2), F(1), F(2)):
            for h in (F(-1), F(0), F(1, 8), F(1, 2), F(1), F(2), F(3)):
                u = h / a
                s = F(0) if u <= 0 else u * u / 2 if u <= 1 else u - F(1, 2)
                exact = -a * s
                approx = min(a * c - h * r for c, r in grid)
                bound = (2 * a + abs(h)) / n
                assert abs(approx - exact) <= bound
                assert all((n * c).denominator == 1 and (n * r).denominator == 1
                           for c, r in grid)
                worst_normalized_error = max(worst_normalized_error,
                                             float(abs(approx - exact) / bound))
                checked += 1
    return {"rational_parameter_cases": checked,
            "max_error_divided_by_proved_bound": worst_normalized_error,
            "status": "PASS"}


def check_thermal_variance():
    cases = []
    for length, beta, j1, j2, field in (
        (9, 0.6, -0.7, 0.4, 0.2),
        (12, 0.8, 0.5, -0.2, -0.3),
        (12, 0.4, -0.9, -0.3, 0.1),
    ):
        z = zn = zn2 = 0.0
        for spins in product((-1, 1), repeat=length):
            charge = sum(x == 1 for x in spins)
            energy = sum(j1 * spins[i] * spins[(i + 1) % length]
                         + j2 * spins[i] * spins[(i + 2) % length]
                         for i in range(length)) - field * charge
            weight = math.exp(-beta * energy)
            z += weight
            zn += weight * charge
            zn2 += weight * charge * charge
        curvature = beta * (zn2 / z - (zn / z) ** 2) / length
        oscillation_bound = 4 * (abs(j1) + abs(j2)) + abs(field)
        pmin = math.exp(-beta * oscillation_bound) / 2
        lower = beta * pmin * pmin / 3
        assert curvature >= lower * (1 - 1e-10)
        cases.append({"L": length, "pressure_curvature": curvature,
                      "conditional_variance_lower_bound": lower})
    return {"cases": cases, "status": "PASS"}


def check_transfer_gap():
    # The two-state stochastic matrix has eigenvalues 1 and 1-2*t**k.
    # Its correlation length exponent is exactly k, by -log(1-x)~x.
    results = []
    for k in (1, 2, 3):
        vals = []
        for n in (8, 16, 32):
            t = 2.0 ** (-n)
            # log1p avoids subtracting two numerically equal eigenvalues.
            xi = -1.0 / math.log1p(-2 * t ** k)
            slope = math.log(xi) / (-math.log(t))
            assert abs(slope - k) <= 1 / n + 1e-3
            vals.append(slope)
        results.append({"exact_exponent": k, "finite_scale_log_ratios": vals})
    return {"cases": results, "status": "PASS"}


def check_pauli_mps():
    ident = ((1, 0), (0, 1))
    paulis = (((0, 1), (1, 0)), ((0, -1j), (1j, 0)), ((1, 0), (0, -1)))

    def multiply(a, b):
        return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(2))
                           for j in range(2)) for i in range(2))

    checked = 0
    for t in (0.5, 0.25, 0.125):
        for basis in (ident,) + paulis:
            channel = [list(row) for row in basis]
            for p in paulis:
                transformed = multiply(multiply(p, basis), p)
                for i in range(2):
                    for j in range(2):
                        channel[i][j] += t * t * transformed[i][j]
            eigenvalue = 1 + 3 * t * t if basis == ident else 1 - t * t
            for i in range(2):
                for j in range(2):
                    assert abs(channel[i][j] - eigenvalue * basis[i][j]) < 1e-14
            checked += 1
    slopes = []
    for n in (8, 16, 32):
        t = F(1, 2 ** n)
        delta = 4 * t * t / (1 + 3 * t * t)
        xi = -1 / math.log1p(-float(delta))
        slope = math.log(xi) / (n * math.log(2))
        assert abs(slope - 2) <= 2 / n + 1e-3
        slopes.append(slope)
    return {"independent_channel_eigenvector_checks": checked,
            "exact_exponent": 2, "finite_scale_log_ratios": slopes,
            "status": "PASS"}


def check_dyadic_dual():
    # Exact rational evaluation of the dual-grid profile in Section 20.3 when zeta_j is integral
    # and eventually constant. This checks the displayed finite dual construction.
    profiles = [([], 2), ([1, 1, 2, 2, 1, 2, 1, 1], 2)]
    checked = 0
    for prefix, tail in profiles:
        def zeta(j):
            return prefix[j - 1] if j <= len(prefix) else tail

        def node(n):
            return F(1, 2 ** n), F(1, 2 ** sum(zeta(j) for j in range(1, n + 1)))

        def integral_node(n):
            cutoff = max(n, len(prefix))
            t_cut, r_cut = node(cutoff)
            ratio = F(1, 2 ** tail)
            total = t_cut * r_cut * (1 + ratio) / (4 * (1 - ratio / 2))
            for j in range(n, cutoff):
                t_hi, r_hi = node(j)
                t_lo, r_lo = node(j + 1)
                total += (r_hi + r_lo) * (t_hi - t_lo) / 2
            return total

        def primitive(t):
            if t <= 0:
                return F(0)
            if t >= 1:
                return integral_node(0) + t - 1
            n = 0
            while t < node(n + 1)[0]:
                n += 1
            t_hi, r_hi = node(n)
            t_lo, r_lo = node(n + 1)
            slope = (r_hi - r_lo) / (t_hi - t_lo)
            delta = t - t_lo
            return integral_node(n + 1) + r_lo * delta + slope * delta * delta / 2

        def dual(r):
            if r == 0:
                return F(0)
            n = 0
            while r < node(n + 1)[1]:
                n += 1
            t_hi, r_hi = node(n)
            t_lo, r_lo = node(n + 1)
            u = t_lo + (r - r_lo) * (t_hi - t_lo) / (r_hi - r_lo)
            return r * u - primitive(u)

        for n in (4, 8, 16, 32):
            grid = []
            for j in range(n + 1):
                r = F(j, n)
                c = dual(r)
                a = F((n * c).numerator // (n * c).denominator, n)
                assert 0 <= c - a < F(1, n)
                grid.append((a, r))
            for a in (F(1, 2), F(1), F(2)):
                for h in (F(-1), F(1, 8), F(1, 2), F(1), F(2), F(3)):
                    exact = -a * primitive(h / a)
                    approx = min(a * c - h * r for c, r in grid)
                    assert exact - a / n <= approx <= exact + (2 * a + abs(h)) / n
                    checked += 1
    return {"exact_nonstationary_dual_cases": checked, "status": "PASS"}


def peak_memory_bytes():
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                         wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                    counters.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


if __name__ == "__main__":
    output = {"python": sys.version.split()[0], "random_seed": 1729,
              "cycle_means": check_cycles(),
              "polyhedral_grid": check_polyhedral_grid(),
              "thermal_variance": check_thermal_variance(),
              "transfer_gap": check_transfer_gap(),
              "pauli_mps": check_pauli_mps(),
              "dyadic_dual": check_dyadic_dual()}
    output["peak_working_set_bytes"] = peak_memory_bytes()
    assert output["peak_working_set_bytes"] is None or output["peak_working_set_bytes"] < 10 ** 9
    print(json.dumps(output, indent=2))
