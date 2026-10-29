"""Exhaustive floating-point check of the wall-layer selection/charging lemma (Theorem 1, Lemma CS).

Symbols: 0=W (horizontal wall), 1=V (vertical wall), 2=A (above signal), 3=S (signal),
4=D (below signal), 5=O (vacuum).
Energy E = -g #W + r #(V above W) + #(violated bonds).
Lemma CS claims, on every L1 x L2 torus (L1 columns, L2 rows):
    E(sigma) - (1 - 6r) V(sigma) >= L1 L2 f*,  f* = min(0, min_{p>=1} f(p)),  f(p) = -g/p + r/p^2,
i.e. with violation weight w = 6r the minimum is still >= L1 L2 f*.
Exhaustive floating-point min-plus transfer matrices over rows (width L1) and over columns (height L2).
Also reports the violation-free minimum (w large) and compares with the analytic prediction
  min over vacuum and over stripe partitions of L2 into pitches p | L1.
"""
import itertools, json, sys, time
import numpy as np

W, V, A, S, D, O = range(6)
NAMES = "WVASDO"
INF = 1e9

def pair_tables(r):
    # horizontal (left u, right v): allowed pairs cost 0, else violation 1
    hall = {(W, W), (O, O), (A, A), (S, A), (D, S), (D, D), (A, V), (S, V), (V, S), (V, D)}
    vall = {(W, S), (W, A), (W, V), (S, W), (D, W), (V, W), (A, A), (A, S), (S, D), (D, D), (V, V), (O, O)}
    hv = np.ones((6, 6)); vv = np.ones((6, 6))   # 1 = violation indicator
    for (a, b) in hall: hv[a, b] = 0
    for (a, b) in vall: vv[a, b] = 0
    vb = np.zeros((6, 6)); vb[V, W] = r          # corner cost on (V above W)
    return hv, vv, vb

def minplus(Am, Bm, chunk=24):
    n = Am.shape[0]
    C = np.empty((n, Bm.shape[1]))
    for i in range(0, n, chunk):
        C[i:i+chunk] = np.min(Am[i:i+chunk, :, None] + Bm[None, :, :], axis=1)
    return C

def transfer(width, g, r, w, along):
    """along='rows': states are rows of `width` sites (cyclic horizontally), stacking vertically.
       along='cols': states are columns of `width` sites (cyclic vertically), stacking horizontally."""
    hv, vv, vb = pair_tables(r)
    states = np.array(list(itertools.product(range(6), repeat=width)))
    n = len(states)
    site = np.where(states == W, -g, 0.0).sum(1)
    if along == 'rows':
        inner_v = sum(hv[states[:, i], states[:, (i+1) % width]] for i in range(width))  # horizontal bonds in row
        inner_c = np.zeros(n)
        cross_v = lambda a, b: vv[a, b]
        cross_c = lambda a, b: vb[a, b]
    else:
        inner_v = sum(vv[states[:, i], states[:, (i+1) % width]] for i in range(width))  # vertical bonds in column
        inner_c = sum(vb[states[:, i], states[:, (i+1) % width]] for i in range(width))
        cross_v = lambda a, b: hv[a, b]
        cross_c = lambda a, b: np.zeros_like(hv[a, b])
    T = np.zeros((n, n))
    for i in range(width):
        a = states[:, i][:, None]; b = states[:, i][None, :]
        T += w * cross_v(a, b) + cross_c(a, b)
    T += (site + w * inner_v + inner_c)[:, None]
    return T

def cyclic_mins(T, lengths):
    out = {}
    P = T.copy(); k = 1
    maxlen = max(lengths)
    powers = {1: T}
    cur = T
    for k in range(2, maxlen + 1):
        cur = minplus(cur, T)
        powers[k] = cur
    for Lk in lengths:
        out[Lk] = float(np.min(np.diag(powers[Lk])))
    return out

def fstar(g, r, pmax=10**4):
    p = np.arange(1, pmax + 1)
    return min(0.0, float(np.min(-g / p + r / p**2)))

def vf_prediction(L1, L2, g, r):
    # vacuum, or stripes: partition of L2 into pitches p >= 2 with p | L1; energy sum over stripes L1(-g + r/p)
    best = 0.0
    divs = [p for p in range(2, L2 + 1) if L1 % p == 0]
    # unbounded knapsack over total height L2
    dp = [INF] * (L2 + 1); dp[0] = 0.0
    for hgt in range(1, L2 + 1):
        for p in divs:
            if p <= hgt and dp[hgt - p] < INF:
                dp[hgt] = min(dp[hgt], dp[hgt - p] + L1 * (-g + r / p))
    return min(best, dp[L2])

def main():
    r = 1.0 / 16
    gs = [-0.05, 0.0, 0.01, 0.025, 0.03, 0.04, 0.045, 0.06, 0.08, 0.09]
    records = []; ok = True
    t0 = time.time()
    plan = [('rows', 3, list(range(2, 13)), gs), ('cols', 3, list(range(2, 13)), gs),
            ('rows', 4, list(range(2, 7)), [0.03, 0.045, 0.06]), ('cols', 4, list(range(2, 7)), [0.03, 0.045, 0.06])]
    for along, width, lengths, glist in plan:
        for g in glist:
            for w, tag in [(6 * r, 'w=6r'), (100.0, 'VF')]:
                T = transfer(width, g, r, w, along)
                mins = cyclic_mins(T, lengths)
                for Lk, m in mins.items():
                    L1, L2 = (width, Lk) if along == 'rows' else (Lk, width)
                    fs = fstar(g, r)
                    pred = vf_prediction(L1, L2, g, r)
                    rec = dict(along=along, L1=L1, L2=L2, g=g, weight=tag, min=m, bound=L1 * L2 * fs, vf_pred=pred)
                    if tag == 'w=6r':
                        rec['pass'] = bool(m >= L1 * L2 * fs - 1e-9)
                    else:
                        rec['pass'] = bool(abs(m - pred) < 1e-9)
                    ok &= rec['pass']
                    records.append(rec)
            print(f"{along} width={width} g={g:+.3f} done ({time.time()-t0:.1f}s)", flush=True)
    bad = [x for x in records if not x['pass']]
    print("total records", len(records), "failures", len(bad), flush=True)
    for x in bad[:20]:
        print("FAIL", x, flush=True)
    # tightness: smallest slack of the w=6r bound and how often the w=6r min equals the VF min
    sl = [x['min'] - x['bound'] for x in records if x['weight'] == 'w=6r']
    print("min slack (w=6r) =", min(sl), flush=True)
    eq = sum(1 for x in records if x['weight'] == 'w=6r' and abs(x['min'] - x['vf_pred']) < 1e-9)
    print("w=6r minimum equals violation-free prediction in", eq, "of", len(sl), "cases", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    json.dump(dict(r=r, records=records, passed=ok), open(sys.argv[0].replace('.py', '.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
