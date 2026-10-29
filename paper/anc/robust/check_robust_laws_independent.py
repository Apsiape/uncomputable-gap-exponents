"""Independent finite checks for Uncomputable critical exponents, version 2.1.0. All writes stay beside this file; one numeric thread.

The Monte Carlo uses fresh, exactly disjoint backward cones, rather than a torus
or a majority over correlated sites. Its finite register program revises each bit
three times. It illustrates the imported reliability mechanism, not universality.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import math
import time
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
START = time.perf_counter()
OUT = {}

def majority_tail(m, p):
    return sum(math.comb(m, k)*p**k*(1-p)**(m-k) for k in range(m//2+1, m+1))

# All-prefix density checks, with exact arithmetic and interior block probes.
density = []
for beta in (Fraction(0), Fraction(3, 10), Fraction(997, 1000), Fraction(1)):
    N, count = 1, 0
    rows = []
    for s in range(35):
        u = min(Fraction(1), max(Fraction(0), beta + Fraction((-1)**s, s+3)))
        end = (s+2)*N
        probes = sorted(set((0, 1, (end-N)//100, (end-N)//2, end-N)))
        deviations = []
        for t in probes:
            if 0 <= t <= end-N:
                ct = count + (t*u.numerator)//u.denominator
                err = abs(Fraction(ct, N+t)-beta)
                bound = (N*abs(Fraction(count, N)-beta)+t*abs(u-beta)+1)/(N+t)
                assert err <= bound
                deviations.append(float(err))
        count += ((end-N)*u.numerator)//u.denominator
        N = end
        rows.append({'s': s, 'N': N, 'endpoint_error': float(abs(Fraction(count, N)-beta)),
                     'largest_probe_error': max(deviations)})
    density.append({'beta': str(beta), 'tail': rows[-3:]})
OUT['density_all_prefix_bound'] = {'pass': True, 'runs': density}
print('All-prefix mechanical density bound: PASS', flush=True)

# Exact exponent block lengths for rational u: floor(2^(k*a/b)) is the
# largest integer m with m^b <= 2^(k*a), so no equality-of-real-names problem.
def exact_block(k, u):
    a, b = u.numerator, u.denominator
    target = 1 << (k*a)
    lo, hi = 0, (1 << k)+1
    while hi-lo > 1:
        mid = (lo+hi)//2
        if mid**b <= target:
            lo = mid
        else:
            hi = mid
    return lo

exponents = []
for gamma in (Fraction(0), Fraction(3, 5), Fraction(1)):
    cumulative = 0
    rows = []
    for k in range(1, 161):
        u = min(Fraction(1), max(Fraction(0), gamma+Fraction((-1)**k, k+3)))
        M = exact_block(k, u)
        for N in ((1 << k), (1 << k)+M//2, (1 << (k+1))-1):
            count = cumulative + max(0, min(M, N-(1 << k)))
            rows.append({'k': k, 'N': N, 'exponent': math.log2(1+count)/math.log2(N)})
        cumulative += M
    assert all(abs(row['exponent']-float(gamma)) < 0.06 for row in rows[-3:])
    exponents.append({'gamma': str(gamma), 'tail': rows[-3:]})
OUT['exponent_exact_floor'] = {'pass': True, 'runs': exponents}
print('Exact rational-power block lengths and interior exponents: PASS', flush=True)

# Independent one-sided/blocking/drift identities, including negative registers.
rng = np.random.default_rng(2910202601)
drift_rows = []
for r in (1, 2, 3):
    B, L = 2*r, 192
    table = rng.integers(0, 2, size=1 << (2*r+1), dtype=np.uint8)
    x0 = rng.integers(0, 2, size=L, dtype=np.uint8)
    original = x0.copy()
    blocked = x0.reshape(-1, B).copy()
    def fstep(x):
        ix = np.zeros(L, dtype=np.int64)
        for shift in range(-r, r+1):
            ix = 2*ix + np.roll(x, -shift)
        return table[ix]
    for t in range(1, 13):
        original = fstep(original)
        # The two-block local table is evaluated directly at each block.
        nxt = np.empty_like(blocked)
        for j in range(len(blocked)):
            pair = np.concatenate((blocked[j], blocked[(j+1) % len(blocked)]))
            for off in range(B):
                ix = 0
                for v in pair[off:off+2*r+1]:
                    ix = 2*ix+int(v)
                nxt[j, off] = table[ix]
        blocked = nxt
        assert np.array_equal(blocked.ravel(), np.roll(original, -r*t))
        if t % 2 == 0:
            for n in range(12):
                i = -2*r*(n+1)
                j = -(n+1)-t//2
                assert blocked[j % len(blocked), 0] == original[i % L]
    drift_rows.append({'r': r, 'block_size': B, 'steps': 12})
OUT['blocked_frame_drift'] = {'pass': True, 'runs': drift_rows}
print('One-sided blocking, even-time fixed projection, negative positions: PASS', flush=True)

# Protected output track: scratch writes and head crossings cannot touch bits.
bits, scratch, heads, changes = {}, {}, 0, []
pos = 1
def move_to(target):
    global pos, heads
    while pos != target:
        pos += 1 if pos < target else -1
        before = dict(bits)
        scratch[pos] = (scratch.get(pos, 0)+1) % 7
        heads += 1
        assert bits == before
for s in range(15):
    for n in range(s+1):
        target = -2*(n+1)
        move_to(target)
        guess = (s >= n+1) ^ (s >= n+3) ^ ((n % 2 == 1) and (s >= n+5))
        old = bits.get(target, 0)
        bits[target] = int(guess)
        if old != int(guess):
            changes.append((n, s, int(guess)))
    move_to(1)
OUT['protected_register_track'] = {'pass': True, 'head_steps': heads, 'changes': changes,
                                   'note': 'finite interpreter sanity check; the universal rule is specified in the proof of Theorem 20.9, not certified by this check'}
print('Protected output track under workspace/head operations: PASS', flush=True)

# A counterexample delimiting the restricted forgetting comparison in Section 20.7.
# K is a radius-zero binary chain with kappa=lambda=1/2 and stationary pi=1/2.
forgetting = []
for t in range(1, 6):
    p = (1+2.0**(-t))/2
    m = 16**t+1
    bound_wrong = math.exp(-m*2.0**(-2*t-1))
    exact = majority_tail(m, p) if m <= 257 else None
    forgetting.append({'t': t, 'm': m, 'p_t': p, 'P_majority_1_exact': exact,
                       'P_majority_1_lower_bound': 1-bound_wrong})
OUT['forgetting_fast_majority_counterexample'] = forgetting
OUT['forgetting_critical_majority_counterexample'] = {
    'limiting_value': 0.5*(1+math.erf(1/math.sqrt(2))),
    'formula': 'lambda=1/2, p_t=(1+2^-t)/2, m_t=4^t+1; limit=normal_CDF(1)',
    'finite_rows': [{'t': t, 'm': 4**t+1, 'majority_probability': majority_tail(4**t+1, (1+2.0**(-t))/2)}
                    for t in range(1, 5)]}
print('Fast-majority counterexample: canonical growth condition is necessary', flush=True)

# Narrowing-interval issue in Theorem 20.12: a symmetric interval at a dyadic
# point has no common first binary digit, however small it becomes.
OUT['dyadic_prefix_obstruction'] = {'p': '1/2', 'intervals': ['[1/2-2^-n,1/2+2^-n]'],
                                   'common_binary_prefix_length': 0,
                                   'repair': 'encode in an irrational entry, as the proof of Theorem 20.12 stipulates'}

# Exact disjoint-cone Monte Carlo for a Q=40 Toom-stacked register program.
# Symbol = ((program*10+clock)*2+bit), clock saturates at 9.
# Program 0 flips at stages 1,4,7; program 1 at 2,5,8. Each makes three revisions.
Q = 40
NEXT = np.empty(Q, dtype=np.int16)
for sym in range(Q):
    bit = sym % 2
    clock = (sym//2) % 10
    prog = sym//20
    nc = min(clock+1, 9)
    flip = clock < 9 and nc in ((1, 4, 7) if prog == 0 else (2, 5, 8))
    NEXT[sym] = ((prog*10+nc)*2)+(bit ^ flip)

def intended(prog, T):
    sym = 20*prog
    for _ in range(T):
        sym = int(NEXT[sym])
    return sym

MC = []
TRIALS, BATCH = 4096, 128
rng = np.random.default_rng(2910202602)
for p in (0.02, 0.06, 0.12):
    for T in (1, 2, 4, 6, 8, 12):
        m = 2*math.ceil(math.log2(max(T, 2)))**2+1
        symbol_errors = bit_errors = total = 0
        wrong = {k: 0 for k in sorted(set((3, min(7, m), m)))}
        for start in range(0, TRIALS, BATCH):
            num = min(BATCH, TRIALS-start)
            programs = (np.arange(start, start+num) % 2).astype(np.int16)
            x = np.broadcast_to((20*programs)[:, None, None, None], (num, m, T+1, T+1)).copy()
            for step in range(T):
                c, n, e = x[:, :, :-1, :-1], x[:, :, 1:, :-1], x[:, :, :-1, 1:]
                # Retain center unless north and east agree; all-distinct tie=center.
                maj = np.where(n == e, n, c)
                x = NEXT[maj]
                hit = rng.random(x.shape) < p
                count = int(hit.sum())
                if count:
                    x[hit] = (x[hit]+rng.integers(1, Q, count, dtype=np.int16)) % Q
            endpoints = x[:, :, 0, 0]
            goal = np.array([intended(int(k), T) for k in programs], dtype=np.int16)
            err = (endpoints % 2) != (goal[:, None] % 2)
            symbol_errors += int((endpoints != goal[:, None]).sum())
            bit_errors += int(err.sum())
            total += endpoints.size
            for k in wrong:
                wrong[k] += int((err[:, :k].sum(axis=1) > k//2).sum())
        row = {'fault_probability': p, 'T': T, 'canonical_m': m, 'trials': TRIALS,
               'program0_logical_bit': intended(0, T) % 2, 'program1_logical_bit': intended(1, T) % 2,
               'single_symbol_error': symbol_errors/total, 'single_bit_error': bit_errors/total,
               'majorities': [{'m': k, 'errors': wrong[k], 'observed_error': wrong[k]/TRIALS,
                                'claimed_exp_minus_m_over_8': math.exp(-k/8),
                                'binomial_tail_at_one_quarter': majority_tail(k, 0.25)} for k in wrong]}
        assert row['single_symbol_error'] < 0.25
        assert all(q['observed_error'] <= q['claimed_exp_minus_m_over_8'] for q in row['majorities'])
        MC.append(row)
        print('MC p=%.2f T=%2d symbol_err=%.5f bit_err=%.5f majority_errors=%s' %
              (p, T, row['single_symbol_error'], row['single_bit_error'], wrong), flush=True)
OUT['independent_disjoint_cone_monte_carlo'] = {
    'alphabet': Q, 'seed': 2910202602, 'trials_per_kernel_time': TRIALS,
    'construction': 'exact finite backward squares; independent copies have independent noise; no periodic boundary',
    'certification_limit': 'chosen p values are not certified to belong to imported U; observed inequalities are finite checks only',
    'runs': MC}
OUT['runtime_seconds'] = time.perf_counter()-START
OUT['peak_working_set_bytes'] = None
try:
    import ctypes
    from ctypes import wintypes
    if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
        raise OSError("Windows peak API unavailable")
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    kernel = ctypes.WinDLL('kernel32')
    psapi = ctypes.WinDLL('psapi')
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    OUT['peak_working_set_bytes'] = int(pmc.PeakWorkingSetSize)
    assert OUT['peak_working_set_bytes'] < 1_000_000_000
except OSError as exc:
    OUT['memory_probe_error'] = str(exc)
OUT['all_assertions_pass'] = True
(ROOT/'independent_checks.json').write_text(json.dumps(OUT, indent=2), encoding='utf-8')
print('PASS | runtime %.3f s | peak working set %s bytes' %
      (OUT['runtime_seconds'], OUT.get('peak_working_set_bytes')), flush=True)
