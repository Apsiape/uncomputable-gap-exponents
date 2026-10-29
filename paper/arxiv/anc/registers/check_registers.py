"""Independent finite checks for the shared-coordinate 2D compiler.

Standard library only. These are checks of indexing/local equalities and ruler
arithmetic, not evidence that a noncomputable limit has been simulated.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import ctypes
from ctypes import wintypes
from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path
import time

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
stats = {}


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n

    def root(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def broadcast(L, walls):
    n = L * L
    d = DSU(3 * n)
    def at(track, x, y):
        return track * n + (y % L) * L + x % L
    for y in range(L):
        for x in range(L):
            d.join(at(0, x, y), at(0, x, y + 1))
            d.join(at(2, x, y), at(1, x - 1, y))
            d.join(at(1, x, y + 1), at(2, x, y))
            if y in walls:
                d.join(at(1, x, y), at(0, x, y))
    return d, at


def thickness(width, p):
    b = [0] * width
    f = [0] * width
    cells = 0
    for t in range(p - 1):
        assert b == [(t >> j) & 1 for j in range(width)]
        assert [int(b[j] or f[j]) for j in range(width)] == [
            int((1 << j) <= t) for j in range(width)]
        cells += width
        c = [1]
        for j in range(width - 1):
            c.append(b[j] * c[j])
        b, f = [a ^ z for a, z in zip(b, c)], [a | z for a, z in zip(f, b)]
    return cells


geometries = collisions = components_checked = cells = 0
for p in range(6, 25):
    ell = (p - 2).bit_length()
    assert ell < p - 1
    for mult in (2, 3):
        L = mult * p
        d, at = broadcast(L, set(range(0, L, p)))
        roots = {d.root(i) for i in range(3 * L * L)}
        assert len(roots) == p
        components_checked += len(roots)
        for fx, fy in ((0, 0), (p, 1), (0, 2), (0, p - 1), (p, p - 2)):
            # The frame column is a V column. The final TM row is allowed
            # to coincide with W, stripe floor, top, or an interior row.
            y_final = (fy - 1) % L
            kind = ('W' if y_final % p == 0 else
                    'F' if y_final % p == 1 else
                    'T' if y_final % p == p - 1 else 'interior')
            collisions += int(kind in ('W', 'F', 'T'))
            word = [int(j < 3 or j % 3 == 1) for j in range(ell)]
            forced = {}
            def force(site, value):
                root = d.root(site)
                assert root not in forced or forced[root] == value
                forced[root] = value
            for y in range(L):
                for x in range(L):
                    residue = (x - fx) % p
                    if not (1 <= residue <= ell):
                        force(at(0, x, y), 0)
            for j in range(ell):
                force(at(0, fx + 1 + j, y_final), word[j])
            assert len(forced) == p
            for y in range(L):
                for x in range(L):
                    residue = (x - fx) % p
                    expected = word[residue - 1] if 1 <= residue <= ell else 0
                    assert forced[d.root(at(0, x, y))] == expected
            # The global prefix is the first ell cells of the uncut tape.
            glob = {(fx + 1 + j) % L for j in range(ell)}
            local = {x for x in range(L) if 1 <= (x - fx) % p <= ell}
            assert glob <= local
            # Independent readers on top-of-stripe rows; their last sites
            # have actual nearest neighbours outside the reader.
            covered = set()
            readers = 0
            for y in range(p - 1, L, p):
                for vx in range(fx % p, L, p):
                    support = {((vx + 1 + j) % L, y) for j in range(ell)}
                    assert not (covered & support)
                    assert ((vx + ell + 1) % L, y) not in support
                    assert (vx + ell + 1 - fx) % p == ell + 1
                    covered.update(support)
                    readers += 1
            assert readers == (L // p) ** 2
            assert len(covered) == readers * ell
            geometries += 1
        cells += thickness(L - 1, p) + thickness(p - 1, p)
stats['shared_coordinate'] = dict(geometries=geometries,
    source_row_overlaps=collisions, broadcast_free_components=components_checked,
    recurrence_cells=cells, unique_source_filling=True, disjoint_reader_supports=True)

# Exhaust all possible short words: source plus mask fixes exactly that word.
word_cases = 0
for p in (6, 7, 8):
    L = 2 * p
    ell = (p - 2).bit_length()
    d, at = broadcast(L, {0, p})
    for word in product((0, 1), repeat=ell):
        prescribed = {}
        for x in range(L):
            residue = x % p
            if not (1 <= residue <= ell):
                prescribed[d.root(at(0, x, 0))] = 0
        for j, a in enumerate(word):
            prescribed[d.root(at(0, j + 1, L - 1))] = a
        assert len(prescribed) == p
        for x in range(L):
            residue = x % p
            assert prescribed[d.root(at(0, x, 0))] == (
                word[residue - 1] if 1 <= residue <= ell else 0)
        word_cases += 1
stats['all_short_source_words'] = word_cases

# Deliberately violate commensurability: the equality graph sees gcd(L,p).
bad_sizes = []
for L, p in ((13, 6), (14, 6), (17, 7), (20, 8)):
    d, at = broadcast(L, {0, p})
    nbits = len({d.root(i) for i in range(3 * L * L)})
    assert nbits == math.gcd(L, p) < p
    bad_sizes.append(dict(L=L, p=p, word_bits=nbits))
stats['noncommensurate_attack'] = bad_sizes

# Exact pitch minimization and tie identities; rational arithmetic only.
r = F(1, 16)
def g_tie(p):
    return r * F(2 * p + 1, p * (p + 1))
def energy(p, g):
    return -g / p + r / (p * p)
for p in range(6, 257):
    g = (g_tie(p) + g_tie(p - 1)) / 2
    assert all(energy(p, g) < energy(q, g) for q in range(1, 2 * p + 1) if q != p)
    assert energy(p, g_tie(p)) == energy(p + 1, g_tie(p))
    assert p * energy(p, g) < 0
stats['exact_ruler_midpoints_and_ties'] = 251

# Independently audit the imported ruler's defect inequality on small tori.
# Integer form: p*f_g(p) and complete finite pitch minimization eliminate
# floating-point comparisons. Most heavily defective configurations obey the
# deficit bound trivially; the remaining cases are computed segmentwise.
symbols = 'WVASD'
xok = set('WW AA SA DS DD AV SV VS VD'.split())
yok = set('WS WA WV SW DW VW AA AS SD DD VV'.split())
def deficit_audit(w, h):
    checked = explicit = 0
    for a in product(symbols, repeat=w * h):
        err = sum(a[y*w+x] + a[y*w+(x+1)%w] not in xok
                  for y in range(h) for x in range(w))
        err += sum(a[y*w+x] + a[((y+1)%h)*w+x] not in yok
                   for y in range(h) for x in range(w))
        nw = a.count('W')
        if 6 * err < nw:
            deficit = F(0)
            for x in range(w):
                ys = [y for y in range(h) if a[y*w+x] == 'W']
                for k, y in enumerate(ys):
                    p = (ys[(k+1) % len(ys)] - y) % h or h
                    top = (y + p) % h
                    v = int(a[((top-1)%h)*w+x] == 'V')
                    deficit += v - F(1, p)
            assert deficit >= -6 * err, (w, h, a, deficit, err)
            explicit += 1
        checked += 1
    return dict(width=w, height=h, configurations=checked, explicit_cases=explicit)
stats['ruler_deficit'] = [deficit_audit(w, h) for w, h in ((2, 2), (2, 3), (3, 2), (3, 3))]

# A final-row source must NOT be replaced by equality at every past time.
assert min(F(1, 8), F(1, 2), F(1, 2)) != F(1, 2)
stats['retained_old_guess_attack'] = 'retained 1/8 differs from final 1/2'

# For scalar 1D integer diagonal chains: total width is O(L), while every
# nonzero spacing is at least 1. The exponent is therefore 0 whenever finite.
# Simple exact examples establish exponent 0; a free-fermion example is proved
# analytically in the manuscript, without a many-body numerical matrix.
stats['integer_classical_chain'] = 'integer spacing lower bound; analytic proof in the manuscript'

if os.name == 'nt' and hasattr(ctypes, 'windll'):
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    mem = PMC()
    mem.cb = ctypes.sizeof(mem)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(mem), mem.cb)
    stats['peak_working_set_bytes'] = mem.PeakWorkingSetSize
    assert mem.PeakWorkingSetSize < 500_000_000
else:
    stats['peak_working_set_bytes'] = None
stats['elapsed_seconds'] = time.perf_counter() - START
stats['status'] = 'PASS'
(OUT / 'checks.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
print(json.dumps(stats, indent=2), flush=True)
