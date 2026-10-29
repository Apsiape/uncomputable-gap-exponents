"""
Check of the dynamic layer (reversible head-driven mixed-radix counter) used in the 2D realization theorem.

The region is a row segment of l cells. Cell j carries a classical label (first?, last?, radix r_j) supplied by the
classical layer; the dynamic layer of every cell has the SAME local space (digit d in [0, RMAX), head in
{none-left, none-right, INC, RET}), so the Hamiltonian below is literally a sum of identical 1-site and 2-site
operators conditioned on the classical labels (projectors), as in the theorem.  Digits d >= r_j are penalized
(static one-site penalty).

Local terms:
  penalties (diagonal, weight 1):
    penalty_0  1-site: digit d >= r_j
    penalty_1  2-site: (none-right, none-left)        [a right-side cell followed by a left-side cell]
    penalty_2  2-site: (head, none-left)               [cells right of the head must be right-side]
    penalty_3  2-site: (none-right, head)              [cells left of the head must be left-side]
    penalty_4  2-site: (head, head)
  transitions (each an edge Laplacian |x><x|+|y><y|-|x><y|-|y><x| on the local pattern):
    T1  1-site:  (d, INC) -> (d+1, RET)                       if d+1 < r_j
    T2  2-site:  (r_j - 1, INC),(e, none-right) -> (0, none-left),(e, INC)   (cell j not last)
    T3  1-site:  (r_j - 1, INC) -> (0, RET)                   if cell j is last (overflow, wraps)
    T4  2-site:  (0, none-left),(e, RET) -> (0, RET),(e, none-right)
    T5  1-site:  (d, RET) -> (d, INC)                         if cell j is first

The script builds the finite product-space Hamiltonian, checks that one-head configurations have weighted
transition degree at most two, verifies that every multi-head configuration is penalized, and numerically
diagonalizes its connected blocks in double precision. It reports consistency checks for the rigorous bracket
and the main-orbit upper witness; these eigenvalues are not exact or interval-certified spectra.
Run: python -u check_dynamic_counter.py
"""
import itertools, json, math, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

NL, NR, INC, RET = 0, 1, 2, 3   # head field: none-left, none-right, INC head, RET head


def local_states(RMAX):
    return [(d, h) for d in range(RMAX) for h in (NL, NR, INC, RET)]


def build(radices, RMAX):
    l = len(radices)
    S = local_states(RMAX)
    idx = {s: i for i, s in enumerate(S)}
    q = len(S)
    dim = q ** l
    I1 = sp.identity(q, format='csr')

    def embed1(op, j):
        mats = [I1] * l
        mats[j] = op
        out = mats[0]
        for m in mats[1:]:
            out = sp.kron(out, m, format='csr')
        return out

    def embed2(op, j):  # op acts on cells j, j+1 (q^2 x q^2)
        left = sp.identity(q ** j, format='csr')
        right = sp.identity(q ** (l - j - 2), format='csr')
        return sp.kron(sp.kron(left, op, format='csr'), right, format='csr')

    def edge1(pairs):
        m = sp.lil_matrix((q, q))
        for x, y in pairs:
            a, b = idx[x], idx[y]
            m[a, a] += 1; m[b, b] += 1; m[a, b] -= 1; m[b, a] -= 1
        return m.tocsr()

    def edge2(pairs):
        m = sp.lil_matrix((q * q, q * q))
        for (x1, x2), (y1, y2) in pairs:
            a = idx[x1] * q + idx[x2]; b = idx[y1] * q + idx[y2]
            m[a, a] += 1; m[b, b] += 1; m[a, b] -= 1; m[b, a] -= 1
        return m.tocsr()

    H = sp.csr_matrix((dim, dim))
    # penalties
    for j, r in enumerate(radices):
        penalty_0 = sp.diags([1.0 if s[0] >= r else 0.0 for s in S])
        H = H + embed1(penalty_0, j)
    pen2 = np.zeros(q * q)
    for a, x in enumerate(S):
        for b, y in enumerate(S):
            hx, hy = x[1], y[1]
            headx, heady = hx in (INC, RET), hy in (INC, RET)
            bad = (hx == NR and hy == NL) or (headx and hy == NL) or (hx == NR and heady) or (headx and heady)
            if bad:
                pen2[a * q + b] = 1.0
    penalty_2 = sp.diags(pen2)
    for j in range(l - 1):
        H = H + embed2(penalty_2, j)
    # transitions
    for j, r in enumerate(radices):
        first, last = (j == 0), (j == l - 1)
        t1 = [((d, INC), (d + 1, RET)) for d in range(r - 1)]
        if t1:
            H = H + embed1(edge1(t1), j)
        if last:
            H = H + embed1(edge1([((r - 1, INC), (0, RET))]), j)
        if first:
            H = H + embed1(edge1([((d, RET), (d, INC)) for d in range(RMAX)]), j)
        if not last:
            rn = radices[j + 1]
            t2 = [(((r - 1, INC), (e, NR)), ((0, NL), (e, INC))) for e in range(RMAX)]
            H = H + embed2(edge2(t2), j)
            t4 = [(((0, NL), (e, RET)), ((0, RET), (e, NR))) for e in range(RMAX)]
            H = H + embed2(edge2(t4), j)
    return H.tocsr(), S


def analyse(radices, RMAX):
    t0 = time.time()
    H, S = build(radices, RMAX)
    dim = H.shape[0]
    off = H.copy(); off.setdiag(0); off.eliminate_zeros()
    deg = np.diff(off.indptr)
    maxdeg = int(deg.max())
    ncomp, lab = connected_components(off, directed=False)
    diag = H.diagonal()
    order = np.argsort(lab, kind='stable')
    bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    gap = math.inf
    kernel = 0
    longest_free = 0
    comp_sizes = []
    # main cycle: component of the all-zero configuration with INC head at cell 0
    l = len(radices)
    q = len(S)
    sidx = {s: i for i, s in enumerate(S)}
    start = [sidx[(0, INC)]] + [sidx[(0, NR)]] * (l - 1)
    g0 = 0
    for c in start:
        g0 = g0 * q + c
    main_label = lab[g0]
    main_len = None
    for c in range(ncomp):
        members = order[bounds[c]:bounds[c + 1]]
        n = len(members)
        if n == 1:
            if diag[members[0]] < 1e-12:
                kernel += 1
            else:
                gap = min(gap, diag[members[0]])
            continue
        sub = H[members][:, members].toarray()
        ev = np.linalg.eigvalsh(sub)
        kernel += int(np.sum(ev < 1e-9))
        pos = ev[ev > 1e-9]
        if len(pos):
            gap = min(gap, pos[0])
        comp_sizes.append(n)
        if c == main_label:
            main_len = n
    # penalty-free configurations: diagonal equals transition degree exactly (no penalty contribution)
    free = np.abs(diag - deg) < 1e-9
    g = int(free.sum())
    maxdeg_free = int(deg[free].max())
    # largest connected cluster of penalty-free configurations (induced subgraph)
    fidx = np.where(free)[0]
    sub = off[fidx][:, fidx]
    nf, flab = connected_components(sub, directed=False)
    gclus = int(np.bincount(flab).max())
    # head count of every configuration, from the mixed-radix product index (digit-major local index s = 4 d + h)
    heads = np.zeros(dim, dtype=np.int64)
    rem = np.arange(dim)
    for _ in range(l):
        loc = rem % q
        rem //= q
        heads += np.isin(loc % 4, [INC, RET])
    one = heads == 1
    maxdeg_one_head = int(deg[one].max())
    multi_all_penalized = bool(np.all(~free[heads >= 2]))
    k = sum(math.log2(r) for r in radices)
    pred = 2 - 2 * math.cos(2 * math.pi / main_len)
    lower = min(1.0 / (8 * maxdeg * gclus ** 2), 1.0 / gclus ** 2)
    return dict(radices=radices, RMAX=RMAX, dim=dim, maxdeg=maxdeg, maxdeg_penalty_free=maxdeg_free, maxdeg_one_head=maxdeg_one_head,
                multi_head_all_penalized=multi_all_penalized, largest_free_cluster=gclus, ncomp=int(ncomp), kernel=kernel,
                k_bits=k, main_cycle_len=main_len, gap=float(gap), pred_main_cycle_gap=pred,
                rigorous_lower=lower, penalty_free_configs=g, largest_component=max(comp_sizes) if comp_sizes else 1,
                log2_inv_gap=float(-math.log2(gap)), seconds=round(time.time() - t0, 2))


if __name__ == '__main__':
    cases = [([2], 2), ([2, 2], 2), ([2, 2, 2], 2), ([2, 2, 2, 2], 2), ([2, 2, 2, 2, 2], 2),
             ([2, 1, 2], 2), ([1, 2, 1, 2], 2), ([4, 2], 4), ([4, 4], 4), ([2, 4, 1], 4), ([4, 4, 4], 4),
             ([2, 2, 2, 2, 2, 2], 2)]
    out = []
    for radices, RMAX in cases:
        res = analyse(radices, RMAX)
        out.append(res)
        print(json.dumps(res), flush=True)
    ok = all(r['maxdeg_penalty_free'] <= 2 and r['maxdeg_one_head'] <= 2 and r['multi_head_all_penalized'] for r in out)
    ok = ok and all(abs(r['gap'] - r['pred_main_cycle_gap']) < 1e-9 or r['gap'] <= r['pred_main_cycle_gap'] + 1e-12 for r in out)
    ok = ok and all(r['gap'] >= r['rigorous_lower'] for r in out)
    print('one-head configs have degree<=2, multi-head configs all penalized, gap <= main-cycle gap, gap >= rigorous lower bound:', 'PASS' if ok else 'FAIL', flush=True)
    with open('check_dynamic_counter.json', 'w') as f:
        json.dump(out, f, indent=1)
