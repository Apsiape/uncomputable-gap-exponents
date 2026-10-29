"""Variant of check_wall_layer.py for the vacuum-free alphabet {W, V, A, S, D} actually used in Section 2.
Exhaustive floating-point min-plus transfer matrices on 3 x L and L x 3 tori (L = 2..12) and 4 x L, L x 4 tori (L = 2..6):
(i) violation weight 6r: minimum >= L1 L2 f*; (ii) weight 1 (the true model): reports min - VFmin;
(iii) weight 100: violation-free minimum equals the analytic prediction (all-A/all-D energy 0, or stripe stackings)."""
import itertools, json, sys, time
import numpy as np
import check_wall_layer as c

NS = 5  # W V A S D only

def transfer(width, g, r, w, along):
    hv, vv, vb = c.pair_tables(r)
    hv, vv, vb = hv[:NS, :NS], vv[:NS, :NS], vb[:NS, :NS]
    states = np.array(list(itertools.product(range(NS), repeat=width)))
    n = len(states)
    site = np.where(states == c.W, -g, 0.0).sum(1)
    if along == 'rows':
        inner_v = sum(hv[states[:, i], states[:, (i+1) % width]] for i in range(width)); inner_c = np.zeros(n)
        cv, cc = vv, vb
    else:
        inner_v = sum(vv[states[:, i], states[:, (i+1) % width]] for i in range(width))
        inner_c = sum(vb[states[:, i], states[:, (i+1) % width]] for i in range(width))
        cv, cc = hv, np.zeros_like(hv)
    T = np.zeros((n, n))
    for i in range(width):
        a = states[:, i][:, None]; b = states[:, i][None, :]
        T += w * cv[a, b] + cc[a, b]
    T += (site + w * inner_v + inner_c)[:, None]
    return T

def main():
    r = 1.0 / 16; ok = True; recs = []; t0 = time.time()
    plan = [('rows', 3, list(range(2, 13)), [-0.05, 0.0, 0.01, 0.03, 0.04, 0.045, 0.06, 0.08]),
            ('cols', 3, list(range(2, 13)), [-0.05, 0.0, 0.01, 0.03, 0.04, 0.045, 0.06, 0.08]),
            ('rows', 4, list(range(2, 7)), [0.03, 0.045]), ('cols', 4, list(range(2, 7)), [0.03, 0.045])]
    for along, width, lengths, gl in plan:
        for g in gl:
            res = {}
            for w in (6 * r, 1.0, 100.0):
                res[w] = c.cyclic_mins(transfer(width, g, r, w, along), lengths)
            for Lk in lengths:
                L1, L2 = (width, Lk) if along == 'rows' else (Lk, width)
                bound = L1 * L2 * c.fstar(g, r); pred = c.vf_prediction(L1, L2, g, r)
                p1 = res[6 * r][Lk] >= bound - 1e-9; p3 = abs(res[100.0][Lk] - pred) < 1e-9
                ok &= bool(p1 and p3)
                recs.append(dict(along=along, L1=L1, L2=L2, g=g, min_w6r=res[6 * r][Lk], bound=bound,
                                 min_true=res[1.0][Lk], vf_pred=pred, vf_min=res[100.0][Lk], pass_=bool(p1 and p3)))
            print(f"{along} width={width} g={g:+.3f} ({time.time()-t0:.0f}s)", flush=True)
    below = [x for x in recs if x['min_true'] < x['vf_pred'] - 1e-9]
    print("records", len(recs), "failures", sum(not x['pass_'] for x in recs), flush=True)
    print("true-model minima below the violation-free minimum (incommensurate tori):", len(below), flush=True)
    for x in below[:8]:
        print("  ", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in x.items()}, flush=True)
    assert ok, "Finite check condition failed"
    print("PASS" if ok else "FAIL", flush=True)
    json.dump(dict(r=r, records=recs, passed=ok), open('check_wall_layer_noO.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
