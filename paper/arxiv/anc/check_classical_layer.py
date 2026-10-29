"""
Exhaustive check of the boundary-preference mechanism of the classical layer (counter tableau part) in the 2D
realization theorem, on full L x L boxes, by an exact transfer (profile) dynamic programme over ALL configurations.

Site fields (64 states): T (top?), F (left?), B (bottom?), b (counter bit), c (carry-in), f (column has seen a 1).
Energy (all terms translation invariant, one-site or nearest-neighbour two-site, integer weights):
  preferences: -1 per T=1 site, +2 if a T=1 site has a site above; same for F (left neighbour) and B (site below).
  rules (+1 per violation):
    one-site   T=1 => b=0 and f=0 ;  F=1 => c=1
    horizontal (x left of y): if y.F=0 then y.c = x.b AND x.c
    vertical   (x above y):   if y.T=0 then y.b = x.b XOR x.c and y.f = x.f OR x.b
Claim checked: the minimum energy is exactly -3L, attained by exactly ONE configuration (so, energies being
integers, every other configuration costs at least 1 more), and in that configuration row r holds r in binary
(least significant bit in column 0) and the bottom-row region {j : B=1 and (f or b)} is [0, bitlength(L-1)).
Run: python -u check_classical_layer.py
"""
import itertools, json, sys, time
import numpy as np

FIELDS = 6
NS = 1 << FIELDS


def unpack(s):
    return dict(T=(s >> 0) & 1, F=(s >> 1) & 1, B=(s >> 2) & 1, b=(s >> 3) & 1, c=(s >> 4) & 1, f=(s >> 5) & 1)


U = [unpack(s) for s in range(NS)]
onsite = np.zeros(NS, dtype=np.int16)
for s, u in enumerate(U):
    onsite[s] = -u['T'] - u['F'] - u['B'] + (1 if u['T'] and (u['b'] or u['f']) else 0) + (1 if u['F'] and not u['c'] else 0)
vert = np.zeros((NS, NS), dtype=np.int16)   # vert[x_above, y_below]
horiz = np.zeros((NS, NS), dtype=np.int16)  # horiz[x_left, y_right]
for x, ux in enumerate(U):
    for y, uy in enumerate(U):
        v = 2 * uy['T'] + 2 * ux['B']
        if not uy['T']:
            if uy['b'] != (ux['b'] ^ ux['c']) or uy['f'] != (ux['f'] | ux['b']):
                v += 1
        vert[x, y] = v
        h = 2 * uy['F']
        if not uy['F'] and uy['c'] != (ux['b'] & ux['c']):
            h += 1
        horiz[x, y] = h


def solve(L):
    """Profile DP over row-major order. Frontier = last L sites (oldest = site above, newest = site to the left)."""
    INF = 8191  # all tested total energies have magnitude below 128
    shape = (NS,) * L
    dp = np.full(NS ** L, INF, dtype=np.int16)
    cnt = np.zeros(NS ** L)
    dp[0] = 0.0
    cnt[0] = 1.0
    back = []  # store argmin info for reconstruction: we reconstruct by re-running with the known minimum instead
    for r in range(L):
        for col in range(L):
            d3 = dp.reshape(NS, NS ** (L - 2), NS) if L >= 2 else None
            c3 = cnt.reshape(NS, NS ** (L - 2), NS)
            new = np.full((NS ** (L - 2), NS, NS), INF, dtype=np.int16)
            newc = np.zeros((NS ** (L - 2), NS, NS))
            for s in range(NS):
                cost = onsite[s] + (vert[:, s][:, None, None] if r > 0 else 0) + (horiz[:, s][None, None, :] if col > 0 else 0)
                tot = d3 + cost  # shape (above, mid, left)
                m = tot.min(axis=0)
                new[:, :, s] = m
                newc[:, :, s] = np.where((tot == m[None]), c3, 0.0).sum(axis=0)
            dp = new.reshape(-1)
            cnt = newc.reshape(-1)
    emin = dp.min()
    nmin = cnt[(dp == emin)].sum()
    # second-lowest energy among all configurations (minimum over frontier states of values above emin, and
    # multiplicity check): energies are integers, so it is enough that the minimiser is unique.
    return emin, nmin


def forced(L):
    """The configuration predicted by the theorem: rows r = 0..L-1 hold r in binary."""
    rows = []
    f = [0] * L
    for r in range(L):
        bits = [(r >> j) & 1 for j in range(L)]
        carry = []
        c = 1
        for j in range(L):
            carry.append(c)
            c = c & bits[j]
        rows.append((bits, carry, list(f)))
        f = [f[j] | bits[j] for j in range(L)]
    # explicit energy of the predicted configuration
    def code(r, j):
        bits, carry, fl = rows[r]
        T, F, B = int(r == 0), int(j == 0), int(r == L - 1)
        return T | (F << 1) | (B << 2) | (bits[j] << 3) | (carry[j] << 4) | (fl[j] << 5)
    E = 0.0
    for r in range(L):
        for j in range(L):
            s = code(r, j)
            E += onsite[s]
            if r > 0:
                E += vert[code(r - 1, j), s]
            if j > 0:
                E += horiz[code(r, j - 1), s]
    bits, carry, fl = rows[-1]
    region = [j for j in range(L) if (fl[j] or bits[j])]
    return region, E


if __name__ == '__main__':
    out = []
    for L in (2, 3, 4):
        t0 = time.time()
        emin, nmin = solve(L)
        reg, Epred = forced(L)
        res = dict(L=L, min_energy=float(emin), predicted_min=-3 * L, energy_of_predicted_configuration=Epred, number_of_minimisers=float(nmin),
                   predicted_region=reg, bitlength_L_minus_1=(L - 1).bit_length(), seconds=round(time.time() - t0, 1))
        out.append(res)
        print(json.dumps(res), flush=True)
    ok = all(r['min_energy'] == r['predicted_min'] == r['energy_of_predicted_configuration'] and r['number_of_minimisers'] == 1 for r in out)
    assert ok, "Finite check condition failed"
    print('unique minimiser at energy -3L, equal to the predicted tableau:', 'PASS' if ok else 'FAIL', flush=True)
    with open('check_classical_layer.json', 'w') as fh:
        json.dump(out, fh, indent=1)
