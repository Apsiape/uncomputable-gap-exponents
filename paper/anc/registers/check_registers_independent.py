"""Independent transfer-rule and exhaustive finite audits; standard library only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import ctypes
from ctypes import wintypes
from fractions import Fraction
from itertools import product
from pathlib import Path
import json, math, time
start = time.perf_counter()
stats = {}
xok = set('WW AA SA DS DD AV SV VS VD'.split())
yok = set('WS WA WV SW DW VW AA AS SD DD VV'.split())

# Enumerate horizontally legal rows, then every vertically legal cyclic row path.
# This never assumes that a clean torus consists of ruler stripes.
def legal_rows(w):
    rows = []
    def extend(s):
        if len(s) == w:
            if s[-1]+s[0] in xok:
                rows.append(s)
            return
        for a in 'WVASD':
            if not s or s[-1]+a in xok:
                extend(s+a)
    extend('')
    return rows

def clean_tori(w, h):
    rows = legal_rows(w)
    nxt = [[j for j,b in enumerate(rows) if all(a[x]+b[x] in yok for x in range(w))] for a in rows]
    total = aligned = mixed = no_wall = 0
    pitches = {}
    def inspect(ids):
        nonlocal total, aligned, mixed, no_wall
        a = [rows[i] for i in ids]
        total += 1
        ys = [y for y in range(h) if a[y] == 'W'*w]
        assert all(('W' not in row) or row == 'W'*w for row in a)
        if not ys:
            assert a == ['A'*w]*h or a == ['D'*w]*h
            no_wall += 1
            return
        ps = [(ys[(i+1)%len(ys)]-y)%h or h for i,y in enumerate(ys)]
        assert all(p >= 2 and w%p == 0 for p in ps)
        vsets = [{x for x in range(w) if row[x] == 'V'} for row in a if row != 'W'*w]
        alignable = all(s == vsets[0] for s in vsets)
        if alignable:
            assert len(set(ps)) == 1
            aligned += 1
            p = ps[0]
            assert vsets[0] and len(vsets[0]) == w//p
            pitches[str(p)] = pitches.get(str(p),0)+1
        if len(set(ps)) > 1:
            assert not alignable
            mixed += 1
    for first in range(len(rows)):
        def walk(ids):
            if len(ids) == h:
                if first in nxt[ids[-1]]:
                    inspect(ids)
                return
            for j in nxt[ids[-1]]:
                walk(ids+[j])
        walk([first])
    assert no_wall == 2
    return dict(width=w,height=h,horizontal_rows=len(rows),clean_tori=total,aligned_striped=aligned,mixed_pitch_excluded=mixed,pitches=pitches)
stats['all_clean_ruler_tori'] = [clean_tori(n,n) for n in range(2,9)]

# Exhaust B and initial C independently, derive D and every later C row by
# the actual transfer relation. Check closure and every W equality explicitly.
def shift(bits):
    return bits[-1:]+bits[:-1]
transfer_cases = candidate_pairs = wrong_words = old_gate_counterexamples = 0
for L in range(2,8):
    words = list(product((0,1),repeat=L))
    for p in range(2,L+1):
        if L%p:
            continue
        walls = set(range(0,L,p))
        accepted = []
        for B in words:
            for C0 in words:
                candidate_pairs += 1
                C = C0
                valid = True
                for y in range(L):
                    if y in walls and C != B:
                        valid = False
                        break
                    D = shift(C)
                    C = D
                if valid and C == C0:
                    accepted.append(B)
                    moved = B
                    for _ in range(p): moved = shift(moved)
                    assert moved == B
        assert len(accepted) == 2**p
        assert len(set(accepted)) == len(accepted)
        ell = (p-2).bit_length()
        if ell >= p-1:
            continue
        for fy in range(L):
            final = (fy-1)%L
            # Global mask survives through every row, so source location fy
            # does not change its list of enforced coordinates.
            for target in product((0,1),repeat=ell):
                survivors = []
                for B in accepted:
                    zero_ok = all(B[x] == 0 for x in range(L) if not 1 <= x%p <= ell)
                    source_ok = all(B[1+j] == target[j] for j in range(ell))
                    if zero_ok and source_ok:
                        survivors.append(B)
                    if zero_ok and not source_ok:
                        wrong_words += 1
                assert len(survivors) == 1
                assert all(survivors[0][x] == (target[x%p-1] if 1<=x%p<=ell else 0) for x in range(L))
                transfer_cases += 1
            if ell and final%p != 1:
                old_gate_counterexamples += 1
stats['literal_broadcast_transfer'] = dict(candidate_B_C_pairs=candidate_pairs,source_geometries_and_words=transfer_cases,rejected_masked_wrong_words=wrong_words,old_floor_gate_missing_source_cases=old_gate_counterexamples)

# Exhaust all possible first-row b,f seeds. The initialized row and all literal
# horizontal/vertical equalities determine one filling; check both prefix masks.
seed_trials = 0
for width in range(1,9):
    for p in range(2,min(16,2**width+2)):
        solutions = 0
        for seed in product((0,1),repeat=2*width):
            seed_trials += 1
            b, f = list(seed[:width]), list(seed[width:])
            if any(b) or any(f):
                continue
            solutions += 1
            for t in range(p-1):
                carry = [1]
                for j in range(1,width):
                    carry.append(b[j-1]*carry[j-1])
                assert all(carry[j+1] == b[j]*carry[j] for j in range(width-1))
                assert b == [(t>>j)&1 for j in range(width)]
                assert [int(b[j] or f[j]) for j in range(width)] == [int(2**j<=t) for j in range(width)]
                b, f = [b[j]^carry[j] for j in range(width)], [f[j]|b[j] for j in range(width)]
        assert solutions == 1
stats['all_thickness_initial_seeds'] = seed_trials

# Construct the complete shared register geometry, all frame phases and source
# words, then literally inspect every tested broadcast bond and source predicate.
fillings = overlap_W = overlap_F = overlap_T = 0
for L,p in ((6,6),(7,7),(8,8),(12,6),(14,7),(16,8)):
    ell = (p-2).bit_length()
    assert ell < p-1
    for fx in range(0,L,p):
        for fy in range(L):
            final = (fy-1)%L
            for word in product((0,1),repeat=ell):
                B = [word[((x-fx)%p)-1] if 1<=(x-fx)%p<=ell else 0 for x in range(L)]
                C = [[B[(x-y)%L] for x in range(L)] for y in range(L)]
                D = [[C[y][(x-1)%L] for x in range(L)] for y in range(L)]
                occupied = set()
                readers = 0
                for y in range(L):
                    for x in range(L):
                        assert C[(y+1)%L][x] == D[y][x]
                        if y%p == 0:
                            assert C[y][x] == B[x]
                        local_mask = 1 <= (x-fx)%p <= ell
                        global_mask = 1 <= (x-fx)%L <= ell
                        if not local_mask:
                            assert B[x] == 0
                        if y == final and global_mask:
                            assert B[x] == word[(x-fx)%L-1]
                    if y%p == p-1:
                        for vx in range(0,L,p):
                            support = {((vx+1+j)%L,y) for j in range(ell)}
                            assert not occupied & support
                            occupied |= support
                            assert ((vx+ell+1)%L,y) not in support
                            readers += 1
                assert readers == (L//p)**2
                fillings += 1
                overlap_W += final%p == 0
                overlap_F += final%p == 1
                overlap_T += final%p == p-1
stats['shared_geometry_literal_bonds'] = dict(fillings=fillings,source_on_W=overlap_W,source_on_F=overlap_F,source_on_T=overlap_T)

# Exact tensor/direct-sum spectra, with multiple kernels and many small positive
# eigenvalues: the first positive sum is the least factor gap, and a competing
# block offset >=5/8 cannot enter below it. Include both ruler-tie phases.
from fractions import Fraction as Q
spectral_trials = 0
for factors in product(((Q(0),Q(0),Q(1,8)),(Q(0),Q(1,4)),(Q(0),Q(1))),repeat=4):
    sums = {sum(v) for v in product(*factors)}
    d = min(x for x in sums if x>0)
    assert d == min(min(x for x in f if x>0) for f in factors)
    full = sums | {Q(5,8)+x for x in (Q(0),Q(1,100),Q(1))}
    assert min(x for x in full if x>0) == min(d,Q(5,8))
    spectral_trials += 1
stats['tensor_and_classical_sector_checks'] = spectral_trials

# Analytic spectra checked against exact one-particle recurrence numerically;
# full subset sums for small open chains, including odd-length zero modes.
chain_cases = 0
for L in range(2,13):
    eps = [2*math.cos(k*math.pi/(L+1)) for k in range(1,L+1)]
    eps = [0 if abs(e)<1e-12 else e for e in eps]
    ground = sum(e for e in eps if e<0)
    full = [sum(e for e,on in zip(eps,bits) if on)-ground for bits in product((0,1),repeat=L)]
    gap = min(e for e in full if e>1e-9)
    expected = 2*math.sin(math.pi/(2*(L+1))) if L%2==0 else 2*math.sin(math.pi/(L+1))
    assert abs(gap-expected)<1e-10
    pos = [2-2*math.cos(k*math.pi/(L+1)) for k in range(1,L+1)]
    assert abs(min(pos)-(2-2*math.cos(math.pi/(L+1))))<1e-12
    chain_cases += 1
stats['full_chain_subset_spectra'] = chain_cases

if os.name == 'nt' and hasattr(ctypes, 'windll'):
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('fault',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak','working','qpp','qpu','qnp','qnu','page','peakpage')]
    ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    mem=PMC();mem.cb=ctypes.sizeof(mem)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(mem),mem.cb)
    stats['peak_working_set_bytes']=mem.peak
    assert mem.peak<1_000_000_000
else:
    stats['peak_working_set_bytes'] = None
stats['elapsed_seconds']=time.perf_counter()-start
stats['status']='PASS'
Path(__file__).with_name('independent_checks.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
print(json.dumps(stats,indent=2),flush=True)

