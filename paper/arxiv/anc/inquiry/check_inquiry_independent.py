"""Independent finite checks. No supplied code is imported.

Numerical tests corroborate finite statements; they do not test noncomputability.
Run with python -u. All artifacts stay next to this file.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path
import time

started = time.perf_counter()
results = {}

def record(name, **data):
    results[name] = data
    print(name + ': ' + json.dumps(data, sort_keys=True), flush=True)

# Exact adaptive inverse-CDF coupling. Each next setting depends on the history.
leaves = [('', F(0), F(1), F(1))]
for depth in range(8):
    new = []
    for history, left, right, probability in leaves:
        p1 = F(1 + history.count('1'), depth + 2)
        split = left + (right-left) * (1-p1)
        new.append((history+'0', left, split, probability*(1-p1)))
        new.append((history+'1', split, right, probability*p1))
    leaves = new
assert sum(x[3] for x in leaves) == 1
assert leaves[0][1] == 0 and leaves[-1][2] == 1
for i, (history, left, right, probability) in enumerate(leaves):
    assert right-left == probability
    if i:
        assert leaves[i-1][2] == left
    # Decode an interior point by the actual adaptive comparisons.
    point = (left+right)/2
    lo, hi, decoded = F(0), F(1), ''
    for stage in range(8):
        p1 = F(1+decoded.count('1'), stage+2)
        split = lo+(hi-lo)*(1-p1)
        if point < split:
            hi, decoded = split, decoded+'0'
        else:
            lo, decoded = split, decoded+'1'
    assert decoded == history
record('adaptive_coupling', leaves=len(leaves), exact_total_probability='1',
       exact_interval_product_identities=True, midpoint_decodings=256)

# A coin-prefix protocol converges uniformly as a sequence of random variables.
# Every fixed target has probability <= 2*2**(-k), hence probability zero.
for k in range(1, 80):
    remainder = sum((F(1, 2**i) for i in range(k+1, k+101)), F(0))
    assert remainder < F(1, 2**k)
record('random_limit_quantifiers', exact_tail_checks=79,
       note='Uniform random limit exists; fixed-target atoms vanish. No test of randomness degrees.')

# Exact HS logarithmic squeeze at dyadic couplings, with computable example
# D(g)=g**3 and finite-volume ratio in [1/2,2].
hs_checks = 0
for j in range(1,100):
    g = 2.0**(-j)
    for multiplier in (0.5,0.75,1.0,1.5,2.0):
        slope = -math.log(multiplier*g**3)/math.log(1/g)
        assert abs(slope-3) <= 1/j+1e-12
        hs_checks += 1
# Factorial rounding: a valid raw tail threshold need not itself be in S.
def factorial_ceiling(threshold):
    n, value = 1, 1
    while value < threshold:
        n += 1
        value *= n
    return value
assert factorial_ceiling(25) == 120 > 25
for threshold in range(1,5000):
    assert factorial_ceiling(threshold) >= threshold
record('hidden_scale_and_rounding', logarithmic_squeeze_checks=hs_checks,
       factorial_rounding_checks=4999, threshold_25_rounds_to=120)

# Ruler ties: purely geometric minimizers are computable, including both ties.
r = F(1,16)
tie_checks = 0
for p in range(6,206):
    g = r*(F(1,p)+F(1,p+1))
    def energy(q):
        return -g/q+r/F(q*q)
    assert energy(p) == energy(p+1)
    assert energy(p) < energy(p-1) and energy(p) < energy(p+2)
    assert abs(F(2)*r/g-p) < 1
    tie_checks += 1
record('ruler_ties', exact_checks=tie_checks, r='1/16',
       note='Checks geometric pitches, not the unknown gap-carrying choice.')

record('resources', elapsed_seconds=time.perf_counter()-started,
       peak_working_set_bytes=None, numerical_worker_threads=1,
       memory_method='Per-process peak unavailable; use the process-tree runner.')
Path(__file__).with_name('independent_checks.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print('ALL INDEPENDENT CHECKS PASS', flush=True)
