"""Independent finite checks. One process, one numerical thread, bounded batches.

All writes go beside this file. Monte Carlo is not an infinite reliability proof.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import ctypes
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np

OUT = Path(__file__).resolve().parent

def peak_bytes():
    if os.name != "nt" or not hasattr(ctypes, "windll"):
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak_ws', 'ws', 'peak_paged', 'paged',
                'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    get = ctypes.windll.kernel32.GetCurrentProcess
    get.restype = ctypes.c_void_p
    mem = ctypes.windll.psapi.GetProcessMemoryInfo
    mem.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    mem.restype = ctypes.c_int
    assert mem(get(), ctypes.byref(c), c.cb)
    return int(c.peak_ws)

def blocking_check():
    rng = np.random.default_rng(10029)
    cases = 0
    for radius in (1, 2, 3):
        block = 2*radius
        size = 12*block
        table = rng.integers(0, 2, size=2**(2*radius+1), dtype=np.uint8)
        x = rng.integers(0, 2, size=size, dtype=np.uint8)
        original = x.copy()
        blocked = x.copy()
        for t in range(1, 9):
            code = np.zeros(size, dtype=np.int32)
            for d in range(-radius, radius+1):
                code = 2*code + np.roll(original, -d)
            original = table[code]
            new = np.zeros(size, dtype=np.uint8)
            for start in range(0, size, block):
                # A block of H(x)_k = F(x)_(k+r) uses these two blocks only.
                two = [int(blocked[(start+d) % size]) for d in range(2*block)]
                for j in range(block):
                    code = 0
                    for d in range(2*radius+1):
                        code = 2*code + two[j+d]
                    new[start+j] = table[code]
            blocked = new
            assert np.array_equal(blocked, np.roll(original, -radius*t))
            cases += size
    return {'coordinate_time_checks': cases, 'radii': [1, 2, 3], 'times': 8}

def majority3(a, b, c):
    # Ties choose the first symbol. Equal-pair majority on arbitrary alphabet.
    return np.where(b == c, b, a)

def full_symbol_majority_check():
    count = 0
    q = 3
    for target in range(q):
        for triple in itertools.product(range(q), repeat=3):
            got = triple[1] if triple[1] == triple[2] else triple[0]
            assert got == target or sum(x != target for x in triple) >= 2
            count += 1
    return {'triples_with_targets': count}

def finite_dlr_check():
    width, depth, beta = 3, 3, 1.7
    bottom = (1, 0, 1)
    region = {(1, 1), (2, 2)}
    sites = [(t, i) for t in range(1, depth+1) for i in range(width)]
    complement = [s for s in sites if s not in region]
    grouped = {}
    z = 1 + math.exp(-beta)
    max_factor_error = 0.0
    partition = 0.0
    for bits in itertools.product((0, 1), repeat=len(sites)):
        values = dict(zip(sites, bits))
        values.update({(0, i): v for i, v in enumerate(bottom)})
        total = incident = omitted = 0
        for t, i in sites:
            parents = ((t-1, i), (t-1, (i+1) % width))
            prediction = values[parents[0]] ^ values[parents[1]]
            fault = values[t, i] != prediction
            total += fault
            if region.intersection(((t, i),)+parents):
                incident += fault
            if (t, i) in region:
                omitted += fault
        weight = math.exp(-beta*total)
        partition += weight
        prob = math.prod((math.exp(-beta) if values[t,i] != (values[t-1,i]^values[t-1,(i+1)%width]) else 1)/z
                         for t, i in sites)
        max_factor_error = max(max_factor_error, abs(prob-weight/z**len(sites)))
        key = tuple(values[s] for s in complement)
        grouped.setdefault(key, []).append((weight, math.exp(-beta*incident), math.exp(-beta*omitted)))
    residual = wrong_residual = 0.0
    for rows in grouped.values():
        norms = [sum(row[k] for row in rows) for k in range(3)]
        for row in rows:
            actual, local, wrong = [row[k]/norms[k] for k in range(3)]
            residual = max(residual, abs(actual-local))
            wrong_residual = max(wrong_residual, abs(actual-wrong))
    assert residual < 1e-14 and max_factor_error < 1e-14
    assert wrong_residual > .1
    assert abs(partition-z**9) < 1e-10
    return {'histories': 512, 'conditional_exterior_classes': len(grouped),
            'dlr_conditional_max_error': residual,
            'error_when_future_child_factors_omitted': wrong_residual,
            'factorization_max_error': max_factor_error,
            'partition_error': abs(partition-z**9)}

def normalizer_countercheck():
    # Two time steps, fixed bottom=0. Energy E(a|r)=J(r)*1[a!=r].
    # At beta=2, row sums depend on r; raw energies do not give the causal law.
    beta = 2.0
    joint_e = {}
    joint_p = {}
    for a, b in itertools.product((0, 1), repeat=2):
        e1 = int(a != 0)
        e2 = (1+2*a)*int(b != a)
        joint_e[a,b] = math.exp(-beta*(e1+e2))
        joint_p[a,b] = math.exp(-beta*(e1+e2))/((1+math.exp(-beta))*(1+math.exp(-beta*(1+2*a))))
    z = sum(joint_e.values())
    residual = max(abs(joint_e[k]/z-joint_p[k]) for k in joint_e)
    assert residual > .01
    return {'nonconstant_row_normalizer_joint_discrepancy': residual}

def revision_step(left, right):
    # Alphabet {0,...,25} encodes (phase in 0..12, bit in 0..1).
    # This is a finite, internal stand-in for a revising program, not a universal CA.
    c = left // 2
    next_phase = np.minimum(c+1, 12)
    toggle = (c == 2) | (c == 6) | (c == 10)
    return (2*next_phase + ((right % 2) ^ toggle)).astype(np.uint8)

def desired(t):
    phase = min(t, 12)
    bit = int(t >= 3) ^ int(t >= 7) ^ int(t >= 11)
    return 2*phase+bit

def cone_outputs(t, p, n, seed):
    """n independent endpoints on exact finite lattice cones, no periodic wrap.

    Shrinking cubes contain each true triangular cone, with unused auxiliary
    variables. Replicas correspond to placements separated by 2*t+1 in a.
    """
    rng = np.random.default_rng(seed)
    outputs = np.empty(n, dtype=np.uint8)
    for start in range(0, n, 48):
        size = min(48, n-start)
        state = np.zeros((size, t+1, t+1, t+1), dtype=np.uint8)
        for step in range(t):
            correction = majority3(state[:, :-1, :-1, :], state[:, 1:, :-1, :], state[:, :-1, 1:, :])
            nxt = revision_step(correction[..., :-1], correction[..., 1:])
            faults = rng.random(nxt.shape) < p
            # Always choose one of the 25 wrong symbols when a fault occurs.
            offset = rng.integers(1, 26, size=nxt.shape, dtype=np.uint8)
            state = np.where(faults, (nxt+offset) % 26, nxt).astype(np.uint8)
        outputs[start:start+size] = state[:, 0, 0, 0]
    return outputs

def binomial_tail(p, m):
    return sum(math.comb(m,k)*p**k*(1-p)**(m-k) for k in range(m//2+1,m+1))

def monte_carlo():
    cases = []
    # Rigorous *finite-horizon* marginal certificate from the no-fault event.
    # V is exact number of noisy vertices in the six-parent backward cone.
    # A nontrivial imported infinite-time threshold is not estimated here.
    configs = [(2,.03), (3,.01), (4,.003), (8,.0001)]
    trials, largest = 12000, 17
    for idx, (t,p) in enumerate(configs):
        outputs = cone_outputs(t, p, trials*largest, 291029+idx)
        full = outputs != desired(t)
        errors = ((outputs % 2) != (desired(t) % 2)).reshape(trials, largest)
        vertices = sum((k+1)**2*(k+2)//2 for k in range(t))
        delta = -math.expm1(vertices*math.log1p(-p))
        assert delta < .5
        rows = []
        for m in (1,3,5,9,17):
            wrong = np.sum(errors[:, :m], axis=1) > m//2
            bound = math.exp(-2*m*(.5-delta)**2)
            exact_upper = binomial_tail(delta,m)
            assert exact_upper <= bound+1e-14
            rows.append({'m':m, 'errors':int(np.sum(wrong)), 'trials':trials,
                         'observed_rate':float(np.mean(wrong)),
                         'certified_binomial_upper':exact_upper, 'claimed_exponential_upper':bound})
        cases.append({'t':t, 'p':p, 'desired_full_symbol':desired(t),
                      'true_cone_noisy_vertices':vertices,
                      'certified_finite_horizon_delta':delta,
                      'observed_full_symbol_error':float(np.mean(full)),
                      'observed_bit_error':float(np.mean(errors)), 'readouts':rows})
        print(json.dumps({'monte_carlo_case':cases[-1]}), flush=True)
    # Following all three revisions in a genuinely noisy finite computation.
    revisions=[]
    for t in (3,7,11,12):
        out=cone_outputs(t,.03,6000,8190+t)
        revisions.append({'t':t,'target_bit':desired(t)%2,
                          'full_symbol_error':float(np.mean(out!=desired(t))),
                          'bit_error':float(np.mean(out%2 != desired(t)%2))})
    return {'alphabet_size':26, 'rule':'Toom NEC then revision_step(left,right)',
            'noise':'independent uniform choice among 25 wrong symbols',
            'initial_row':'constant phase=0,bit=0, cloned in a,b',
            'certified_cases':cases, 'revision_snapshots_p_003':revisions}

def random_limit_counterexample():
    # Q_n=sum_{k=1}^n X_k/2^k has deterministic tail <=2^-n under fair product law.
    rows = [{'n':n, 'expectation':(1-2.0**(-n))/2,
             'uniform_sample_tail_upper':2.0**(-n)} for n in (1,4,8,16)]
    return {'expectation_limit':.5, 'sample_limit_distribution':'uniform[0,1]',
            'rows':rows}

def main():
    started=time.perf_counter()
    out={'blocking':blocking_check(), 'majority_symbols':full_symbol_majority_check(),
         'dlr':finite_dlr_check(), 'normalizer':normalizer_countercheck(),
         'random_limit':random_limit_counterexample()}
    out['monte_carlo']=monte_carlo()
    out['seconds']=time.perf_counter()-started
    out['peak_working_set_bytes']=peak_bytes()
    out['numpy_version']=np.__version__
    out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert out['peak_working_set_bytes'] is None or out['peak_working_set_bytes']<1_000_000_000
    (OUT/'independent-checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    main()
