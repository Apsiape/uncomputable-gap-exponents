"""Independent finite checks; one process, one numerical thread, exact wall arithmetic.

No imports from supplied check scripts. Arrays are chunked below 30 MB.
The legal-counter enumeration uses only the invariant one-head space.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import itertools, math, json, time
from fractions import Fraction as Q
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
HPAIRS = set('WW AA SA DS DD AV SV VS VD'.split())
VPAIRS = set('WS WA WV SW DW VW AA AS SD DD VV'.split())
ALPH = 'WVASD'
HOK = np.array([[a+b in HPAIRS for b in ALPH] for a in ALPH])
VOK = np.array([[a+b in VPAIRS for b in ALPH] for a in ALPH])

def forcing(a):
    ht, wd = a.shape
    walls = [y for y in range(ht) if 'W' in a[y]]
    if not walls:
        return np.all(a == 'A') or np.all(a == 'D')
    if any(not np.all(a[y] == 'W') for y in walls): return False
    for i, top in enumerate(walls):
        p = (walls[(i+1) % len(walls)]-top) % ht or ht
        if p < 2 or wd % p: return False
        stripe = a[[(top+j) % ht for j in range(1,p)], :]
        types = []
        for col in stripe.T:
            word = ''.join(col)
            if word == 'V'*(p-1): types.append(0)
            elif word.count('S') == 1:
                off = word.index('S')
                if word != 'A'*off+'S'+'D'*(p-2-off): return False
                types.append(off+1)
            else: return False
        if any(types[(x+1)%wd] != (types[x]+1)%p for x in range(wd)): return False
    return True

def exhaustive_wall(wd, ht):
    den = math.lcm(*range(1, ht+1))
    demand = np.zeros(5**ht, dtype=np.int16)
    for code in range(5**ht):
        col = [(code//5**y)%5 for y in range(ht)]
        ww = [y for y in range(ht) if col[y] == 0]
        for i, y in enumerate(ww):
            p = (ww[(i+1)%len(ww)]-y)%ht or ht
            demand[code] += den//p
    n = 5**(wd*ht)
    vf = 0; maxratio = Q(0); maximizer = None
    powers = 5**np.arange(wd*ht, dtype=np.int64)
    cpowers = 5**np.arange(ht, dtype=np.int64)
    for start in range(0,n,20000):
        codes = np.arange(start,min(n,start+20000),dtype=np.int64)
        a = ((codes[:,None]//powers)%5).reshape(-1,ht,wd)
        vh = (~HOK[a,np.roll(a,-1,axis=2)]).sum((1,2))
        vv = (~VOK[a,np.roll(a,-1,axis=1)]).sum((1,2))
        viol = vh+vv
        corners = ((a==1)&(np.roll(a,-1,axis=1)==0)).sum((1,2))
        dcodes = (a*cpowers[None,:,None]).sum(1)
        deficit = demand[dcodes].sum(1)-den*corners
        assert np.all(deficit <= 6*den*viol), (wd,ht,'charging')
        zero = np.flatnonzero(viol==0)
        for j in zero:
            assert forcing(np.array(list(ALPH))[a[j]]), (wd,ht,'forcing',a[j].tolist())
        vf += len(zero)
        nz = np.flatnonzero(viol)
        if len(nz):
            j = nz[np.argmax(deficit[nz]/viol[nz])]
            ratio = Q(int(deficit[j]),den*int(viol[j]))
            if ratio > maxratio:
                maxratio = ratio; maximizer = a[j].tolist()
    rec = dict(width=wd,height=ht,configurations=n,violation_free=vf,
               largest_corner_deficit_per_violation=str(maxratio),witness=maximizer)
    print('wall',rec,flush=True)
    return rec

def legal_counter(rad):
    ell = len(rad)
    digits = list(itertools.product(*(range(r) for r in rad)))
    nodes = [(v,i,s) for v in digits for i in range(ell) for s in (0,1)]
    index = {x:j for j,x in enumerate(nodes)}
    edges = {}
    for v,i,s in nodes:
        if s == 0:  # INC
            nv = list(v)
            if v[i]+1 < rad[i]:
                nv[i] += 1; target=(tuple(nv),i,1)
            elif i+1 < ell:
                nv[i]=0; target=(tuple(nv),i+1,0)
            else:
                nv[i]=0; target=(tuple(nv),i,1)
        elif i == 0:
            target=(v,i,0)
        elif v[i-1] == 0:
            target=(v,i-1,1)
        else: continue
        e=tuple(sorted((index[(v,i,s)],index[target])))
        edges[e]=edges.get(e,0)+1
    adj=[[] for _ in nodes]
    for a,b in edges: adj[a].append(b); adj[b].append(a)
    seen=set(); comps=[]; gap=math.inf
    for root in range(len(nodes)):
        if root in seen: continue
        ids=[]; todo=[root]; seen.add(root)
        while todo:
            a=todo.pop(); ids.append(a)
            for b in adj[a]:
                if b not in seen: seen.add(b); todo.append(b)
        if len(ids)<2: continue
        ns=len(ids); cyc=all(len(adj[a])==2 for a in ids)
        if ns==2:
            ev=2*edges[tuple(sorted(ids))]
        else:
            ev=2-2*math.cos((2 if cyc else 1)*math.pi/ns)
        gap=min(gap,ev); comps.append(dict(vertices=ns,cycle=cyc,gap=ev))
    return dict(radices=rad,gap=min(1.0,gap),components=comps)

def film_checks():
    r=Q(1,16); h=7; g=r/Q((h+1)*(h+2))
    records=[]
    for L in (728,729,1448,1449):
        energies={hh: Q(L)*g*hh+r*(L//(hh+1)) for hh in range(1,L-1)}
        emin=min(energies.values()); opt=[hh for hh,e in energies.items() if e==emin]
        assert opt == ([8] if L%72==8 else [7])
        records.append(dict(L=L,g=str(g),optimal_heights=opt,
                            row_energy_h7=str(energies[7]),row_energy_h8=str(energies[8])))
    # At h=7, L=72m+1, shifting one y-plane's signal phase adds one corner.
    L=721; p=8; counts=[]; tracks=[]
    cycle=['V'*h]+['A'*o+'S'+'D'*(h-1-o) for o in range(h)]
    for phase in range(p):
        words=[cycle[(x+phase)%p] for x in range(L)]
        assert all(words[x][y]+words[x+1][y] in HPAIRS for x in range(L-1) for y in range(h))
        counts.append(sum(w[-1]=='V' for w in words))
        tracks.append(words)
    assert max(counts)-min(counts)==1
    out=dict(tie_g=str(g),tie_residue_tests=records,
             phase_defect=dict(L=L,height=h,corner_counts=counts,energy_difference=str(r)),
             counter_h7=legal_counter((2,2,2)),counter_h8=legal_counter((2,2,2,1)),
             long_counter_17=legal_counter((2,2,2)+(1,)*14),
             long_counter_18=legal_counter((2,2,2)+(1,)*15),
             DC2_counterexample=legal_counter((2,1,1,1)))
    print('film',out,flush=True)
    return out

def scalar_checks():
    r=Q(1,16); out=[]
    for p in range(2,100):
        gp=r*Q(2*p+1,p*(p+1))
        for g in (gp, (gp+r*Q(2*p-1,p*(p-1)))/2):
            vals={j:-g/Q(j)+r/Q(j*j) for j in range(2,10000)}
            m=min(vals.values()); opt=[j for j,v in vals.items() if v==m]
            assert opt==([p,p+1] if g==gp else [p])
        out.append(p)
    return dict(wall_pitches_checked=len(out),all_passed=True)

def wetting_transfer():
    """Exhaust all open x,z tracks, including wetting-rule defects and false floors.

    Physical flags are fixed correctly. The F iff Z0 rule and the Z1 rule
    are soft unit penalties, so track configurations violating them are included.
    All arithmetic is integer after a common denominator is chosen.
    """
    alph='FASDVWO'; ia={a:i for i,a in enumerate(alph)}
    hp=set('FF WW OO AA SA DS DD AV SV VS VD'.split())
    vp=set('WS WA WV SF DF VF AA AS SD DD VV OW OO'.split())
    interior=set('ASDV'); r=Q(1,16); records=[]
    for Z, widths in ((3,range(1,13)),(4,range(1,7))):
        cols=np.array(list(itertools.product(range(7),repeat=Z)),dtype=np.int16)
        n=len(cols)
        xp=np.zeros((n,n),dtype=np.int16)
        for z in range(Z):
            ok=np.array([[a+b in hp for b in alph] for a in alph])
            xp += (~ok[cols[:,z,None],cols[None,:,z]])
        ip=np.isin(cols,[ia[a] for a in interior]).sum(1)
        penalty=(cols[:,0]!=ia['F']).astype(np.int16)
        penalty+=(cols[:,1:]==ia['F']).sum(1)
        penalty+=~np.isin(cols[:,-1],[ia['O'],ia['W']])
        corner=np.zeros(n,dtype=np.int16)
        for z in range(Z-1):
            ok=np.array([[a+b in vp for b in alph] for a in alph])
            penalty+=~ok[cols[:,z+1],cols[:,z]]
            corner+=(cols[:,z+1]==ia['V'])&(cols[:,z]==ia['F'])
        for g in (Q(-1,32),Q(0),r/Q(63),Q(1,128),Q(1,16),Q(1,8)):
            den=math.lcm(g.denominator,16)
            unary=den*penalty+int(den*g)*ip+(den//16)*corner
            dp=unary.astype(np.int64)
            trans=xp.astype(np.int64)*den
            for X in widths:
                if X>1: dp=unary+np.min(dp[:,None]+trans,axis=0)
                actual=int(dp.min())
                pred=min(Q(X)*g*h+r*(X//(h+1)) for h in range(1,Z-1))
                assert Q(actual,den)==pred,(X,Z,str(g),str(Q(actual,den)),str(pred))
                records.append(dict(X=X,Z=Z,g=str(g),minimum=str(pred),pass_=True))
    print('wetting transfer',len(records),'exact minima PASS',flush=True)
    return records

if __name__=='__main__':
    t=time.time()
    results={'wall': [exhaustive_wall(w,h) for w,h in ((2,2),(2,3),(3,2),(2,4),(4,2),(3,3),(2,5),(5,2))],
             'film':film_checks(),'scalar':scalar_checks(),'wetting_transfer':wetting_transfer()}
    results['passed']=True; results['seconds']=time.time()-t
    (ROOT/'check_critical_ruler.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('PASS',results['seconds'],flush=True)
