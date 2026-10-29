"""Second engine for the reversible mixed-radix counter (counter construction Lemma DC), written independently.

The counter region is a line of l sites. Local state per site: digit d in {0..D-1} and a symbol in
{NL, NR, INC, RET}. The Hamiltonian is built LITERALLY from local terms on the full product space
(4D)^l: diagonal penalties (weight 1) and edge Laplacians |x><x|+|y><y|-|x><y|-|y><x| on local patterns.
We then split H into connected blocks of its off-diagonal support and diagonalize each block densely.

Checks, per radix vector r = (r_0..r_{l-1}) (each r_j a power of 2, product 2^k):
 (a) every configuration with exactly one head has at most 2 transition neighbours;
 (b) every configuration with >= 2 heads carries a penalty;
 (c) the smallest positive eigenvalue of H lies in [4^-k/(64 l^2), 4 pi^2 4^-k];
 (d) it equals 2 - 2cos(2 pi / N), N the length of the main orbit (measured by following the forward rules).
The equality test concerns only these finite samples. General theorems use the
full-counter bracket, not equality with the main cycle. The independent
component check includes explicit counterexamples to universal equality.
"""
import itertools, json, math, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

NL, NR, INC, RET = 0, 1, 2, 3
HEAD = (INC, RET)


def build(r, D):
    l = len(r)
    loc = 4 * D
    dim = loc ** l
    # encode site state s = d*4 + sym ; configuration index = sum s_j * loc^j
    def dec(idx):
        out = []
        for _ in range(l):
            out.append(idx % loc)
            idx //= loc
        return out
    def enc(states):
        idx = 0
        for j in reversed(range(l)):
            idx = idx * loc + states[j]
        return idx
    diag = np.zeros(dim)
    rows, cols = [], []
    pen = np.zeros(dim, dtype=bool)
    for idx in range(dim):
        st = dec(idx)
        d = [s // 4 for s in st]
        y = [s % 4 for s in st]
        p = 0.0
        for j in range(l):
            if d[j] >= r[j]:
                p += 1
        for j in range(l - 1):
            a, b = y[j], y[j + 1]
            if (a == NR and b == NL) or (a in HEAD and b == NL) or (a == NR and b in HEAD) or (a in HEAD and b in HEAD):
                p += 1
        diag[idx] += p
        pen[idx] = p > 0
    # transitions: list of (site set, pattern x, pattern y) applied wherever pattern x matches; add edge Laplacian
    edges = set()
    def add_edge(i1, i2):
        if i1 == i2:
            raise RuntimeError('self loop')
        e = (min(i1, i2), max(i1, i2))
        if e in edges:
            raise RuntimeError('double edge %s' % (e,))
        edges.add(e)
    # enumerate local terms literally: for each term and each global config matching side x, connect to y
    for idx in range(dim):
        st = dec(idx)
        d = [s // 4 for s in st]
        y = [s % 4 for s in st]
        for j in range(l):
            # T1 (1-site): (dd, INC) -> (dd+1, RET) if dd+1 < r_j  [forward side only, edge added once]
            if y[j] == INC and d[j] + 1 < r[j]:
                st2 = list(st); st2[j] = (d[j] + 1) * 4 + RET; add_edge(idx, enc(st2))
            # T3 (j last): (r_j - 1, INC) -> (0, RET)
            if j == l - 1 and y[j] == INC and d[j] == r[j] - 1:
                st2 = list(st); st2[j] = 0 * 4 + RET; add_edge(idx, enc(st2))
            # T5 (j first): (dd, RET) -> (dd, INC)
            if j == 0 and y[j] == RET:
                st2 = list(st); st2[j] = d[j] * 4 + INC; add_edge(idx, enc(st2))
            if j < l - 1:
                # T2: (r_j - 1, INC)(e, NR) -> (0, NL)(e, INC)
                if y[j] == INC and d[j] == r[j] - 1 and y[j + 1] == NR:
                    st2 = list(st); st2[j] = 0 * 4 + NL; st2[j + 1] = d[j + 1] * 4 + INC; add_edge(idx, enc(st2))
                # T4: (0, NL)(e, RET) -> (0, RET)(e, NR)
                if y[j] == NL and d[j] == 0 and y[j + 1] == RET:
                    st2 = list(st); st2[j] = 0 * 4 + RET; st2[j + 1] = d[j + 1] * 4 + NR; add_edge(idx, enc(st2))
    deg = np.zeros(dim, dtype=int)
    for (a, b) in edges:
        rows += [a, b]; cols += [b, a]
        deg[a] += 1; deg[b] += 1
    diag = diag + deg
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim))
    H = sp.diags(diag) - A
    heads = np.array([sum(1 for s in dec(i) if s % 4 in HEAD) for i in range(dim)])
    return H, A, deg, pen, heads, dec, enc


def main_orbit_length(r):
    """Follow the forward rules from (INC at cell 0, all digits 0) until return."""
    l = len(r)
    d = [0] * l
    pos, sym = 0, INC
    start = (tuple(d), pos, sym)
    n = 0
    while True:
        if sym == INC:
            if d[pos] + 1 < r[pos]:
                d[pos] += 1; sym = RET
            elif pos < l - 1:
                d[pos] = 0; pos += 1
            else:
                d[pos] = 0; sym = RET
        else:
            if pos == 0:
                sym = INC
            elif d[pos - 1] == 0:
                pos -= 1
            else:
                raise RuntimeError('stuck on main orbit')
        n += 1
        if (tuple(d), pos, sym) == start:
            return n
        if n > 10 ** 7:
            raise RuntimeError('no return')


def run_case(r):
    D = max(r)
    l = len(r)
    k = int(round(math.log2(np.prod(r))))
    t0 = time.time()
    H, A, deg, pen, heads, dec, enc = build(r, D)
    dim = H.shape[0]
    ok_deg = bool(np.all(deg[heads == 1] <= 2))
    ok_multi = bool(np.all(pen[heads >= 2]))
    ncomp, lab = connected_components(A, directed=False)
    Hc = H.tocsr()
    pos_min = math.inf
    for c in range(ncomp):
        verts = np.where(lab == c)[0]
        if len(verts) == 1:
            v = Hc[verts[0], verts[0]]
            if v > 1e-9:
                pos_min = min(pos_min, v)
            continue
        if len(verts) > 4000:
            raise RuntimeError('component too large %d' % len(verts))
        B = Hc[verts][:, verts].toarray()
        ev = np.linalg.eigvalsh(B)
        evp = ev[ev > 1e-9]
        if len(evp):
            pos_min = min(pos_min, float(evp[0]))
    N = main_orbit_length(r)
    cyc = 2 - 2 * math.cos(2 * math.pi / N)
    lo = 4.0 ** (-k) / (64 * l * l)
    hi = 4 * math.pi ** 2 * 4.0 ** (-k)
    res = dict(radices=r, l=l, k=k, dim=dim, components=int(ncomp), one_head_deg_le_2=ok_deg,
               multi_head_all_penalized=ok_multi, gap=pos_min, main_cycle_N=N, cycle_value=cyc,
               gap_equals_cycle=abs(pos_min - cyc) < 1e-10, lower=lo, upper=hi,
               in_bracket=(lo <= pos_min <= hi + 1e-12), seconds=round(time.time() - t0, 2))
    return res


if __name__ == '__main__':
    cases = [(2, 2), (1, 2), (2, 1, 2), (4, 2), (2, 4), (1, 1, 2), (2, 2, 2), (4, 1, 2), (1, 4, 1),
             (2, 2, 2, 2), (4, 4), (1, 2, 1, 2), (2, 1, 1, 2), (4, 2, 1), (4, 4, 1), (2, 4, 2)]
    out = []
    allok = True
    for r in cases:
        res = run_case(list(r))
        ok = res['one_head_deg_le_2'] and res['multi_head_all_penalized'] and res['in_bracket'] and res['gap_equals_cycle']
        allok &= ok
        print(json.dumps(res), flush=True)
        out.append(res)
    print('ALL PASS' if allok else 'SOME FAIL', flush=True)
    with open(sys.argv[1] if len(sys.argv) > 1 else 'check_film_counter.json', 'w') as fh:
        json.dump(dict(cases=out, all_pass=allok), fh, indent=1)
