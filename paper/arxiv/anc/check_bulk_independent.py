"""Finite bulk broadcast checks; standard library, sequential."""
import ctypes
from ctypes import wintypes
from fractions import Fraction as Q
from itertools import product, combinations
from collections import Counter
from pathlib import Path
import math
import os
import random
import time
import json

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
RESULT = {}
assert all(os.environ.get(v) == '1' for v in
           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))


class Linear:
    """Sparse row echelon elimination over GF(2), with integer bit rows."""
    def __init__(self, other=None):
        self.rows = {} if other is None else other.rows.copy()

    def add(self, row, rhs=0):
        while row:
            pivot = row.bit_length() - 1
            if pivot not in self.rows:
                self.rows[pivot] = row, rhs
                return True
            old, bit = self.rows[pivot]
            row ^= old
            rhs ^= bit
        return rhs == 0


def equations(L, p, z0):
    n = L * L
    def ix(f, x, z):
        return f * n + (z % L) * L + x % L
    pairs = []
    for z in range(L):
        for x in range(L):
            pairs.extend(((ix(0, x, z), ix(0, x, z + 1)),
                          (ix(2, x, z), ix(1, x - 1, z)),
                          (ix(1, x, z + 1), ix(2, x, z))))
            if (z - z0) % L % p == 0:
                pairs.append((ix(1, x, z), ix(0, x, z)))
    system = Linear()
    for a, b in pairs:
        assert system.add((1 << a) ^ (1 << b))
    return ix, pairs, system


geometries = fillings = contradictions = 0
for p in range(6, 11):
    ell = (p - 2).bit_length()
    for L in (p, 2 * p):
        for z0 in (0, 1, p - 1):
            ix, pairs, base = equations(L, p, z0)
            assert 3 * L * L - len(base.rows) == p
            for frame in (0, L - 1, p // 2):
                geometries += 1
                for prefix in product((0, 1), repeat=ell):
                    word = [0] + list(prefix) + [0] * (p - ell - 1)
                    b = [word[(x - frame) % p] for x in range(L)]
                    expected = []
                    for field in range(3):
                        for z in range(L):
                            shift = 0 if field == 0 else (z - z0) % L + (field == 2)
                            expected.extend(b[(x - shift) % L] for x in range(L))
                    assert all(expected[a] == expected[b_] for a, b_ in pairs)
                    fixed = Linear(base)
                    for x in range(L):
                        if not 1 <= (x - frame) % p <= ell:
                            assert fixed.add(1 << ix(0, x, 0), 0)
                    for j, bit in enumerate(prefix):
                        assert fixed.add(1 << ix(0, frame + 1 + j, z0 + 1), bit)
                    assert len(fixed.rows) == 3 * L * L
                    for j, bit in enumerate(prefix):
                        bad = Linear(fixed)
                        assert not bad.add(1 << ix(0, frame + 1 + j + p, z0), 1 - bit)
                        contradictions += 1
                    fillings += 1
for L, p in ((7, 3), (10, 4), (13, 5), (15, 6)):
    _, _, base = equations(L, p, 0)
    assert 3 * L * L - len(base.rows) == math.gcd(L, p)
RESULT['broadcast_linear_algebra'] = dict(translated_geometries=geometries,
    all_prefix_fillings=fillings, wrong_repeated_bit_contradictions=contradictions,
    noncommensurate_tests=4, y_copy='eliminated exactly before GF(2) calculation')
print('PASS independent GF(2) broadcast and translated source', flush=True)

H = set('WW AA SA DS DD AV SV VS VD'.split())
V = set('WS WA WV SW DW VW AA AS SD DD VV'.split())
SYMS = 'WVASD'
r = Q(1, 16)
def clean(L, p, x0=0, z0=0):
    ans = []
    for z in range(L):
        t = (z - z0) % p
        for x in range(L):
            col = (x - x0) % p
            if t == 0:
                s = 'W'
            elif col == 0:
                s = 'V'
            else:
                off = col - 1
                s = 'A' if t - 1 < off else 'S' if t - 1 == off else 'D'
            ans.append(s)
    return ans


def wall_stats(a, L):
    errors = corners = 0
    for z in range(L):
        for x in range(L):
            s = a[z * L + x]
            hpair = s + a[z * L + (x + 1) % L]
            vpair = s + a[((z + 1) % L) * L + x]
            errors += (hpair not in H) + (vpair not in V)
            corners += vpair == 'VW'
    inv_lengths = Q(0)
    for x in range(L):
        walls = [z for z in range(L) if a[z * L + x] == 'W']
        for j, z in enumerate(walls):
            inv_lengths += Q(1, (walls[(j + 1) % len(walls)] - z) % L or L)
    assert Q(corners) - inv_lengths >= -6 * errors
    return errors, corners, a.count('W')


wall_count = 0
L = p = 6
g = (r * Q(13, 42) + r * Q(11, 30)) / 2
fstar = -g / p + r / p ** 2
baseline = clean(L, p)
assert wall_stats(baseline, L) == (0, 1, L)
for support in ([(i,) for i in range(L * L)], combinations(range(L * L), 2)):
    for sites in support:
        for replacements in product(*[[s for s in SYMS if s != baseline[i]] for i in sites]):
            a = baseline.copy()
            for i, s in zip(sites, replacements):
                a[i] = s
            err, cor, nw = wall_stats(a, L)
            energy = err + r * cor - g * nw
            assert energy - L * L * fstar >= Q(5, 8) * err
            wall_count += 1
rng = random.Random(29102026)
for L, p in ((12, 6), (14, 7), (16, 8)):
    for _ in range(1200):
        a = clean(L, p, rng.randrange(p), rng.randrange(p))
        for _ in range(rng.randint(1, 12)):
            a[rng.randrange(L * L)] = rng.choice(SYMS)
        wall_stats(a, L)
        wall_count += 1
RESULT['ruler_defects'] = dict(configurations=wall_count,
    all_one_and_two_site_changes_at_pitch_6=10224,
    larger_random_defects=3600, arithmetic='exact fractions')
print('PASS independent ruler defects', flush=True)

# Native local transitions, with multiplicity. s=0,1,2,3 are NL,NR,INC,RET.
def neighbours(state, radices):
    ell = len(state)
    ans = []
    def change(edits):
        a = list(state)
        for j, v in edits:
            a[j] = v
        ans.append(tuple(a))
    for j, (d, s) in enumerate(state):
        if s == 2 and d + 1 < radices[j]:
            change(((j, (d + 1, 3)),))
        if s == 3 and 0 < d < radices[j]:
            change(((j, (d - 1, 2)),))
    d, s = state[-1]
    if (d, s) == (radices[-1] - 1, 2):
        change(((ell - 1, (0, 3)),))
    if (d, s) == (0, 3):
        change(((ell - 1, (radices[-1] - 1, 2)),))
    d, s = state[0]
    if s in (2, 3):
        change(((0, (d, 5 - s)),))
    for j in range(ell - 1):
        d, s = state[j]
        e, t = state[j + 1]
        if (d, s, t) == (radices[j] - 1, 2, 1):
            change(((j, (0, 0)), (j + 1, (e, 2))))
        if (d, s, t) == (0, 0, 2):
            change(((j, (radices[j] - 1, 2)), (j + 1, (e, 1))))
        if (d, s, t) == (0, 0, 3):
            change(((j, (0, 3)), (j + 1, (e, 1))))
        if (d, s, t) == (0, 3, 1):
            change(((j, (0, 0)), (j + 1, (e, 3))))
    return ans


def potential(state, radices):
    val = sum(d >= rj for (d, s), rj in zip(state, radices))
    for (_, s), (_, t) in zip(state, state[1:]):
        val += (s == 1 and t == 0) or (s >= 2 and t == 0) or (s == 1 and t >= 2) or (s >= 2 and t >= 2)
    return val


def legal_graph(radices):
    ell = len(radices)
    vertices = set()
    for digits in product(*(range(v) for v in radices)):
        for j in range(ell):
            for head in (2, 3):
                vertices.add(tuple((d, 0 if i < j else 1 if i > j else head)
                                   for i, d in enumerate(digits)))
    graph = {s: neighbours(s, radices) for s in vertices}
    assert all(len(ns) <= 2 and all(t in vertices for t in ns) for ns in graph.values())
    seen = set()
    records = []
    gaps = []
    for s in vertices:
        if s in seen:
            continue
        todo = [s]
        component = []
        seen.add(s)
        while todo:
            v = todo.pop()
            component.append(v)
            for w in graph[v]:
                if w not in seen:
                    seen.add(w)
                    todo.append(w)
        n = len(component)
        cycle = all(len(graph[v]) == 2 for v in component)
        records.append(('cycle' if cycle else 'path', n))
        if n > 1:
            # Stable evaluation of 2-2cos(theta).
            angle = (math.pi if cycle else math.pi / 2) / n
            gaps.append(4 * math.sin(angle) ** 2)
    return min(gaps), sorted(records)


counter_records = []
full_vertices = 0
vectors = [(D, rads) for D, maxell in ((2, 4), (4, 3))
           for ell in range(1, maxell + 1)
           for rads in product((1, D), repeat=ell)]
for D, radices in vectors:
    ell = len(radices)
    free_one = 0
    for state in product(tuple(product(range(D), range(4))), repeat=ell):
        pot = potential(state, radices)
        heads = sum(s >= 2 for d, s in state)
        ns = neighbours(state, radices)
        if heads >= 2:
            assert pot >= 1
        if heads == 1:
            assert len(ns) <= 2
        assert all(sum(s >= 2 for d, s in v) == heads for v in ns)
        if pot == 0:
            assert all(potential(v, radices) == 0 for v in ns)
            free_one += heads == 1
        full_vertices += 1
    B = math.prod(radices)
    assert free_one == 2 * ell * B
    free_gap, components = legal_graph(radices)
    assert min(1, free_gap) >= 1 / (64 * ell * ell * B * B) - 1e-14
    assert free_gap <= 4 * math.pi ** 2 / (B * B) + 1e-12
    counter_records.append(dict(D=D, radices=radices, free_gap=free_gap, components=components))
long_radices = (2, 2, 2) + (1,) * 15
long_gap, long_components = legal_graph(long_radices)
N = 2 * sum(math.prod(long_radices[j:]) for j in range(len(long_radices)))
main_gap = 4 * math.sin(math.pi / N) ** 2
assert long_gap < main_gap
assert ('path', 34) in long_components and ('cycle', 58) in long_components
RESULT['full_counter_sector_audit'] = dict(radix_vectors=len(vectors),
    full_product_vertices=full_vertices, max_full_dimension=max((4 * D) ** len(v) for D, v in vectors),
    invariance_and_head_sector_checks=True, records=counter_records,
    long_example=dict(radices=long_radices, components=long_components,
                      full_gap=long_gap, main_cycle_gap=main_gap))
print('PASS independent full counter sectors; main-cycle equality refuted again', flush=True)

# Degenerate kernels: spectrum of tensor sums of diag(0,0,delta).
delta = Q(1, 97)
for nc in range(1, 9):
    dist = Counter(sum(levels) for levels in product((Q(0), Q(0), delta), repeat=nc))
    assert min(x for x in dist if x > 0) == delta
    assert dist[delta] == nc * 2 ** (nc - 1)
RESULT['degenerate_tensor_sum'] = dict(cases=8, smallest_positive='1/97',
                                      multiplicity='N*2**(N-1)')

boundary_checks = 0
for p in range(6, 2049):
    ell = (p - 2).bit_length()
    assert ell < p - 1
    assert 2 ** (ell - 1) <= p - 2 < 2 ** ell
    g_tie = r * Q(2 * p + 1, p * (p + 1))
    fp = -g_tie / p + r / p ** 2
    fq = -g_tie / (p + 1) + r / (p + 1) ** 2
    assert fp == fq
    assert (p * (p + 1)) % p == (p * (p + 1)) % (p + 1) == 0
    # Uniform vertical marker cannot have both cardinalities.
    assert p * (p + 1) // p != p * (p + 1) // (p + 1)
    # Corner density and reader volume count, with L=p(p+1).
    side = p * (p + 1)
    assert (side // p) ** 2 * side == side ** 3 // p ** 2
    boundary_checks += 1
RESULT['ties_and_bulk_count'] = dict(pitches=boundary_checks, exact=True)

if os.name == 'nt':
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD)
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    peak = pmc.PeakWorkingSetSize
    assert peak < 500_000_000
else:
    peak = None
RESULT['resources'] = dict(elapsed_seconds=time.perf_counter() - START,
                          peak_working_set_bytes=peak, standard_library_only=True,
                          processes=1, numerical_threads_requested=1)
RESULT['status'] = 'PASS'
(OUT / 'check_bulk_independent.json').write_text(json.dumps(RESULT, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in RESULT.items() if k != 'full_counter_sector_audit'}, indent=2), flush=True)
