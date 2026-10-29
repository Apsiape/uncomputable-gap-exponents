"""Finite bulk broadcast checks; standard library, sequential."""
import ctypes
import cmath
import itertools
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
import random
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parent
results = {}


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def broadcast_check(L, p, ny):
    # y may be shorter: copying in y has no dependence on its length.
    N = L * L * ny
    dsu = DSU(3 * N)

    def ix(field, x, y, z):
        return field * N + ((z % L) * ny + (y % ny)) * L + (x % L)

    for z in range(L):
        for y in range(ny):
            for x in range(L):
                b = ix(0, x, y, z)
                dsu.join(b, ix(0, x, y + 1, z))
                dsu.join(b, ix(0, x, y, z + 1))
                dsu.join(ix(2, x, y, z), ix(1, x - 1, y, z))
                dsu.join(ix(1, x, y, z + 1), ix(2, x, y, z))
                if z % p == 0:
                    dsu.join(ix(1, x, y, z), b)
    roots = {dsu.find(i) for i in range(3 * N)}
    assert len(roots) == math.gcd(L, p), (L, p, ny, len(roots))
    if L % p:
        return len(roots)
    for x in range(L):
        assert dsu.find(ix(0, x, 0, 0)) == dsu.find(ix(0, x % p, 0, 0))
    ell = (p - 2).bit_length() if p >= 3 else 0
    assignments = {}
    word = [0] * p
    for j in range(ell):
        word[1 + j] = ((j * j + 3 * j + p) // 2) % 2
    for x in range(L):
        # Freeze non-prefix positions; source fixes only one copy of the prefix.
        if not (1 <= x % p <= ell):
            assignments[dsu.find(ix(0, x, 0, 0))] = 0
    for j in range(ell):
        r = dsu.find(ix(0, 1 + j, ny - 1, 1))
        v = word[1 + j]
        assert r not in assignments or assignments[r] == v
        assignments[r] = v
    assert set(assignments) == roots, (L, p, ell)
    for z in range(L):
        for y in range(ny):
            for x in range(L):
                assert assignments[dsu.find(ix(0, x, y, z))] == word[x % p]
                assert assignments[dsu.find(ix(1, x, y, z))] == word[(x - z) % p]
                assert assignments[dsu.find(ix(2, x, y, z))] == word[(x - z - 1) % p]
    return len(roots)


broadcast_cases = []
for p in range(3, 17):
    for multiple in (1, 2, 3):
        L = p * multiple
        ny = min(L, 4)
        broadcast_cases.append([L, p, ny, broadcast_check(L, p, ny)])
for p, L in ((3, 10), (4, 10), (5, 12), (6, 15), (7, 16)):
    broadcast_cases.append([L, p, 3, broadcast_check(L, p, 3)])
results['broadcast'] = {'cases': len(broadcast_cases), 'records': broadcast_cases,
                        'noncommensurate_period_collapse_detected': True}


def thickness_row(width, height):
    bits, past = [0] * width, [0] * width
    for _ in range(height - 1):
        carry = [1] + [0] * (width - 1)
        for x in range(1, width):
            carry[x] = bits[x - 1] & carry[x - 1]
        new_bits = [b ^ c for b, c in zip(bits, carry)]
        past = [f | b for f, b in zip(past, bits)]
        bits = new_bits
    return [b | f for b, f in zip(bits, past)]


prefix_cells = 0
for p in range(6, 257):
    h, L = p - 1, 2 * p
    ell = (p - 2).bit_length()
    local = thickness_row(h, h)
    global_row = thickness_row(L - 1, h)
    assert local == [int(j < ell) for j in range(h)]
    assert global_row == [int(j < ell) for j in range(L - 1)]
    assert ell < h
    # First cell's global and local source indices coincide.
    assert global_row[:h] == local
    prefix_cells += (h + L - 1) * h
results['prefix'] = {'pitches': 251, 'recurrence_cells': prefix_cells,
                     'first_cell_alignment_and_physical_last_guard': True}


r = F(1, 16)
def gp(p):
    return r * F(2 * p + 1, p * (p + 1))

for p in range(6, 129):
    g = (gp(p) + gp(p - 1)) / 2
    energies = {q: -g / q + r / (q * q) for q in range(2, 4 * p + 1)}
    assert energies[p] == min(energies.values())
    assert sum(e == energies[p] for e in energies.values()) == 1
    assert abs(F(p) - 2 * r / g) < 1
    g = gp(p)
    assert -g / p + r / (p * p) == -g / (p + 1) + r / ((p + 1) ** 2)
results['ruler_selection'] = {'off_tie_pitches': 123, 'ties': 123, 'exact': True}


H_PAIRS = set(('WW', 'AA', 'SA', 'DS', 'DD', 'AV', 'SV', 'VS', 'VD'))
V_PAIRS = set(('WS', 'WA', 'WV', 'SW', 'DW', 'VW', 'AA', 'AS', 'SD', 'DD', 'VV'))
symbols = 'WVASD'


def charge_check(config, nx, nz):
    def at(x, z):
        return config[(z % nz) * nx + (x % nx)]
    violations = sum(at(x, z) + at(x + 1, z) not in H_PAIRS
                     for x in range(nx) for z in range(nz))
    violations += sum(at(x, z) + at(x, z + 1) not in V_PAIRS
                      for x in range(nx) for z in range(nz))
    nW = config.count('W')
    # sum 1/p_s <= nW; this is already enough in most configurations.
    if 6 * violations >= nW:
        return
    nc = sum(at(x, z) == 'V' and at(x, z + 1) == 'W'
             for x in range(nx) for z in range(nz))
    deficit = F(nc)
    for x in range(nx):
        walls = [z for z in range(nz) if at(x, z) == 'W']
        for k, z in enumerate(walls):
            p = (walls[(k + 1) % len(walls)] - z) % nz or nz
            deficit -= F(1, p)
    assert deficit >= -6 * violations, (config, nx, nz, deficit, violations)


wall_cases = 0
for nx, nz in ((2, 2), (2, 3), (3, 2), (3, 3)):
    for c in itertools.product(symbols, repeat=nx * nz):
        charge_check(c, nx, nz)
        wall_cases += 1
results['charging'] = {'exhaustive_tori': [[2, 2], [2, 3], [3, 2], [3, 3]],
                        'configurations': wall_cases, 'exact': True}


def gap_function(stage, g, approximating_bits):
    if g == 0:
        return F(0)
    bits = [approximating_bits(j, stage) for j in range(stage)]
    values, knots, k = [F(1)], [F(1)], 0
    for n, a in enumerate(bits, 1):
        k += a
        knots.append(F(1, 2 ** n))
        values.append(F(1, 2 ** (n + k)))
    for n in range(stage):
        if knots[n + 1] <= g <= knots[n]:
            return values[n + 1] + (g - knots[n + 1]) * (values[n] - values[n + 1]) / (knots[n] - knots[n + 1])
    return g * F(1, 2 ** k)


def toy_guess(j, stage):
    final = int(j % 3 != 0)
    return final if stage >= j * j + 3 else final ^ (stage % 2)


modulus_checks = 0
for stage in range(1, 33):
    points = sorted({F(0), F(1), *[F(1, 2 ** n) for n in range(stage + 1)]})
    for a, b in zip(points, points[1:]):
        fa, fb = gap_function(stage, a, toy_guess), gap_function(stage, b, toy_guess)
        assert 0 <= fb - fa <= F(3, 2) * (b - a)
        modulus_checks += 1
for n in range(1, 12):
    settled = max(j * j + 3 for j in range(n))
    k = sum(int(j % 3 != 0) for j in range(n))
    for s in (max(n, settled), max(n, settled) + 1, max(n, settled) + 13):
        assert gap_function(s, F(1, 2 ** n), toy_guess) == F(1, 2 ** (n + k))
results['modulus_counterexample'] = {'exact_interval_checks': modulus_checks,
                                    'toy_prefix_stabilization': True}


# A retained old guess makes the independent-block gap an infimum, not a final output.
history = [F(1, 8), F(1, 2), F(1, 2), F(1, 2)]
assert min(history) == F(1, 8) != history[-1]
results['obsolete_guess_attack'] = {'retained_gap': str(min(history)),
                                   'final_gap': str(history[-1])}

# Product of two-level counters: exact spectrum and extensive flat-band multiplicity.
for nc in range(1, 9):
    delta = F(1, 97)
    spectrum = [delta * sum(bits) for bits in itertools.product((0, 1), repeat=nc)]
    assert min(e for e in spectrum if e > 0) == delta
    assert spectrum.count(delta) == nc
results['tensor_gap'] = {'counter_counts': 8, 'exact_flat_band_multiplicity': True}

# The inert-phase qubit layer has a positive one-magnon ring eigenvalue tending to zero.
max_ring_residual = 0.0
for L in range(2, 34):
    vector = [cmath.exp(2j * math.pi * x / L) / math.sqrt(L) for x in range(L)]
    eigenvalue = 2 - 2 * math.cos(2 * math.pi / L)
    residual = max(abs(2 * vector[x] - vector[(x - 1) % L] - vector[(x + 1) % L]
                       - eigenvalue * vector[x]) for x in range(L))
    max_ring_residual = max(max_ring_residual, residual)
    assert residual < 1e-12
results['inert_phase_ring'] = {'sizes': 32, 'max_eigenvector_residual': max_ring_residual,
                               'used_only_as_upper_witness': True}

# A vertically constant V-marker forbids a mixture of distinct stripe pitches.
for p in range(2, 129):
    L = p * (p + 1)
    row_p = [int(x % p == 0) for x in range(L)]
    row_q = [int(x % (p + 1) == 0) for x in range(L)]
    assert sum(row_p) != sum(row_q)
    assert row_p != row_q
results['tie_alignment'] = {'distinct_pitches_rejected': 127,
                            'reason': 'different number of V columns in a common row'}


peak = None
if os.name == 'nt':
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    peak = pmc.PeakWorkingSetSize
    assert peak < 1_000_000_000
results['resources'] = {'elapsed_seconds': time.perf_counter() - START,
                        'peak_working_set_bytes': peak, 'stdlib_only': True,
                        'processes': 1, 'threads_requested': 1}
results['status'] = 'PASS'
(ROOT / 'check_bulk_broadcast.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('PASS: broadcast, source-prefix alignment, ruler selection, charging, modulus, and tensor checks', flush=True)
print(json.dumps(results['resources']), flush=True)
print('Finite checks do not verify arbitrary Turing computations or a full many-body matrix.', flush=True)
