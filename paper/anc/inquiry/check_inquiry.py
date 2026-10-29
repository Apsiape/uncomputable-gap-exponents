"""Exact adaptive inquiry checks and finite hidden-scale consistency checks.

These assertions do not prove noncomputability or an infinite-limit theorem.
The separate independent implementation uses binary coupling and its own code.
"""
from fractions import Fraction as F
from pathlib import Path
import json,math,time
START=time.perf_counter()

def check_partition(intervals):
    assert intervals[0][0]==0 and intervals[-1][1]==1
    previous=F(0)
    for left,right,probability in intervals:
        assert left==previous and right>left and right-left==probability
        previous=right
    assert sum(row[2] for row in intervals)==1

leaves=[((),F(0),F(1),F(1))]
for depth in range(6):
    next_leaves=[]
    for history,left,right,probability in leaves:
        probabilities=[F(1+history.count(a),depth+3) for a in range(3)]
        assert sum(probabilities)==1
        endpoint=left
        for a,p in enumerate(probabilities):
            end=endpoint+(right-left)*p
            next_leaves.append((history+(a,),endpoint,end,probability*p))
            endpoint=end
        assert endpoint==right
    leaves=next_leaves
check_partition([(a,b,p) for h,a,b,p in leaves])
for history,left,right,probability in leaves:
    point=(left+right)/2;lo,hi=F(0),F(1);decoded=()
    for depth in range(6):
        previous=lo
        for a in range(3):
            end=previous+(hi-lo)*F(1+decoded.count(a),depth+3)
            if previous<=point<end:
                lo,hi=previous,end;decoded+=(a,);break
            previous=end
        else:raise AssertionError('Interior point failed inverse-CDF decoding')
    assert decoded==history

# Prefixes converge to a random limit; at most two binary cells cover a fixed
# target, so these finite bounds tend to zero for each target separately.
atom_checks=0
for k in range(1,13):
    assert sum((F(1,2**j) for j in range(k+1,k+101)),F(0))<F(1,2**k)
    for target in [F(0),F(1,3),F(1,2),F(7,8),F(1)]:
        count=sum(F(j,2**k)<=target<=F(j+1,2**k) for j in range(2**k))
        assert F(count,2**k)<=F(2,2**k)
        atom_checks+=1

def check_log_ratio(j,multiplier):
    assert j>=1 and F(1,2)<=multiplier<=2
    sampled=-math.log(float(multiplier)*2.0**(-3*j))/(j*math.log(2))
    assert abs(sampled-3)<=1/j+1e-12
for j in range(1,201):
    for multiplier in [F(1,2),F(3,4),F(1),F(3,2),F(2)]:check_log_ratio(j,multiplier)

def factorial_ceiling(threshold):
    value=1;k=1
    while value<threshold:k+=1;value*=k
    return value

def check_rounding(threshold,rounded):
    value=1;k=1;previous=0
    while value<rounded:previous=value;k+=1;value*=k
    assert value==rounded and rounded>=threshold
    assert previous<threshold
for threshold in range(1,5001):check_rounding(threshold,factorial_ceiling(threshold))
assert factorial_ceiling(25)==120

ties=0;r=F(1,16)
for p in range(6,126):
    g=r*(F(1,p)+F(1,p+1))
    energies={q:-g/q+r/F(q*q) for q in range(2,p+12)}
    minimizers=[q for q,e in energies.items() if e==min(energies.values())]
    assert minimizers==[p,p+1]
    assert abs(2*r/g-p)<1
    ties+=1

def rejects(function,*args):
    try:function(*args)
    except AssertionError:return
    raise AssertionError('Corruption was not rejected')
rejects(check_partition,[(F(0),F(3,4),F(3,4)),(F(1,2),F(1),F(1,2))])
rejects(check_log_ratio,4,F(4))
rejects(check_rounding,25,25)
result={'status':'PASS','adaptive_ternary_histories':len(leaves),
 'random_prefix_fixed_target_bounds':atom_checks,'factor_two_log_checks':1000,
 'factorial_rounding_checks':5000,'exact_pitch_ties':ties,'negative_controls_rejected':3,
 'peak_working_set_bytes':None,'seconds':time.perf_counter()-START,
 'scope':'Finite consistency and rejection tests; no test proves the reach theorem or a noncomputable settling bound.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
