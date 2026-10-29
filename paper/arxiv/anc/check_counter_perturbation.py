"""Third-engine check of the reversible mixed-radix counter (counter construction Lemma DC) and of the
instability theorem instability: adding eps * (one-site digit-flip Laplacian among legal digits) sends the
cell gap from ~4^-k to >= (2/5) eps 2^-beta / l^2 (polylogarithmic in the cell size).

Written from the rule list T1-T5 in counter construction 3.2; imports none of the input scripts.
Region-only engine (no exterior sites); weights accumulate, so the l=1, r=1 double edge is kept.
"""
import itertools, json, sys, time, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

NL, NR, INC, RET = range(4)
HEAD = (INC, RET)

def build(radices, eps):
    l = len(radices); dmax = max(radices)
    cells = [(d, s) for d in range(dmax) for s in range(4)]
    idx_cell = {c: i for i, c in enumerate(cells)}
    nc = len(cells)
    confs = list(itertools.product(range(nc), repeat=l))
    index = {c: i for i, c in enumerate(confs)}
    n = len(confs)
    pen = np.zeros(n)
    edges = {}
    def add(a, b, w=1.0):
        if a == b: return
        key = (min(a, b), max(a, b))
        edges[key] = edges.get(key, 0.0) + w
    for i, cf in enumerate(confs):
        cl = [cells[k] for k in cf]
        p = 0
        for j, (d, s) in enumerate(cl):
            if d >= radices[j]: p += 1
        for j in range(l - 1):
            a, b = cl[j][1], cl[j + 1][1]
            if (a == NR and b == NL) or (a in HEAD and b == NL) or (a == NR and b in HEAD) or (a in HEAD and b in HEAD):
                p += 1
        pen[i] = p
        # forward transitions only (edges are symmetric)
        for j, (d, s) in enumerate(cl):
            rj = radices[j]
            if s == INC and d + 1 < rj:                                   # T1
                new = list(cl); new[j] = (d + 1, RET); add(i, index[tuple(idx_cell[c] for c in new)])
            if s == INC and d == rj - 1 and j < l - 1 and cl[j + 1][1] == NR:   # T2
                new = list(cl); new[j] = (0, NL); new[j + 1] = (cl[j + 1][0], INC)
                add(i, index[tuple(idx_cell[c] for c in new)])
            if s == INC and d == rj - 1 and j == l - 1:                    # T3
                new = list(cl); new[j] = (0, RET); add(i, index[tuple(idx_cell[c] for c in new)])
            if s == RET and j >= 1 and cl[j - 1] == (0, NL):               # T4 (pair (j-1, j))
                new = list(cl); new[j - 1] = (0, RET); new[j] = (d, NR)
                add(i, index[tuple(idx_cell[c] for c in new)])
            if s == RET and j == 0:                                        # T5
                new = list(cl); new[j] = (d, INC); add(i, index[tuple(idx_cell[c] for c in new)])
            if eps > 0 and d < rj:                                         # perturbation: legal digit flips
                for d2 in range(d + 1, rj):
                    new = list(cl); new[j] = (d2, s); add(i, index[tuple(idx_cell[c] for c in new)], eps)
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for (a, b), w in edges.items():
        rows += [a, b]; cols += [b, a]; vals += [-w, -w]; deg[a] += w; deg[b] += w
    H = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(deg + pen)
    G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return H, G, pen, confs, cells

def analyse(radices, eps):
    H, G, pen, confs, cells = build(radices, eps)
    ncomp, lab = connected_components(G, directed=False)
    gap = math.inf; pure = True; maxsize = 0
    for c in range(ncomp):
        ids = np.where(lab == c)[0]
        maxsize = max(maxsize, len(ids))
        pc = pen[ids]
        if (pc > 0).any() and (pc == 0).any():
            pure = False
        if (pc > 0).all():
            continue  # fully penalized component: >= 1 by PL(a); not needed for the low gap
        if len(ids) == 1:
            continue
        sub = H[ids][:, ids].toarray()
        ev = np.linalg.eigvalsh(sub)
        pos = ev[ev > 1e-10]
        if len(pos): gap = min(gap, pos[0])
    l = len(radices); B = int(np.prod(radices)); k = int(round(math.log2(B)))
    N = 2 * sum(B // int(np.prod(radices[:j])) if j else B for j in range(l))
    cyc = 2 - 2 * math.cos(2 * math.pi / N) if N > 2 else 4.0
    beta = int(round(math.log2(max(radices)))) if max(radices) > 1 else 0
    return dict(radices=list(radices), eps=eps, k=k, l=l, N=N, gap=float(min(gap, 1.0)), main_cycle=cyc,
                lower_DC=4.0**(-k) / (64 * l * l), upper_DC=4 * math.pi**2 * 4.0**(-k),
                lower_instability=0.4 * eps * 2.0**(-beta) / l**2, pure=pure, maxcomp=int(maxsize))

def main():
    cases = [(1,), (2,), (4,), (2, 2), (4, 4), (1, 2), (2, 2, 2), (4, 4, 4), (1, 4, 2), (2, 2, 2, 2), (4, 4, 4, 4), (4, 1, 4, 2)]
    out = []; ok = True; t0 = time.time()
    for rad in cases:
        for eps in [0.0, 0.01, 0.1]:
            rec = analyse(rad, eps)
            if eps == 0:
                rec['pass'] = bool(rec['pure'] and rec['lower_DC'] - 1e-12 <= rec['gap'] <= rec['upper_DC'] + 1e-12)
            else:
                rec['pass'] = bool(rec['pure'] and rec['gap'] >= rec['lower_instability'] - 1e-12)
            ok &= rec['pass']
            out.append(rec)
            print(f"{str(rad):14s} eps={eps:<5} k={rec['k']:2d} gap={rec['gap']:.3e} cycle={rec['main_cycle']:.3e} "
                  f"instability_bound={rec['lower_instability']:.2e} pure={rec['pure']} pass={rec['pass']} ({time.time()-t0:.0f}s)", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    json.dump(dict(records=out, passed=ok), open(sys.argv[0].replace('.py', '.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
