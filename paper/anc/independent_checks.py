"""Independent direct-state checks; no imports from supplied scripts. Single threaded."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import itertools
import json
import math
from collections import defaultdict
from fractions import Fraction
import numpy as np

NL, NR, INC, RET = range(4)


def counter(radices, outside=0, rmax=4):
    length = len(radices)
    nsites = length + outside
    q = 4 * rmax
    dim = q ** nsites
    adj = [defaultdict(int) for _ in range(dim)]
    penalties = np.zeros(dim, dtype=int)
    heads = np.zeros(dim, dtype=int)
    powers = [q ** (nsites - j - 1) for j in range(nsites)]

    def edge(v, conf, changes):
        w = v + sum((new - conf[j]) * powers[j] for j, new in changes.items())
        assert v != w
        adj[v][w] += 1
        adj[w][v] += 1

    for v, conf in enumerate(itertools.product(range(q), repeat=nsites)):
        d = [s // 4 for s in conf]
        h = [s % 4 for s in conf]
        heads[v] = sum(x >= INC for x in h[:length])
        penalties[v] = sum(d[j] >= radices[j] for j in range(length))
        penalties[v] += sum(conf[j] != 0 for j in range(length, nsites))
        for j in range(length - 1):
            a, b = h[j:j + 2]
            penalties[v] += ((a == NR and b == NL) or (a >= INC and b == NL)
                             or (a == NR and b >= INC) or (a >= INC and b >= INC))
        for j, radix in enumerate(radices):
            if h[j] == INC and d[j] + 1 < radix:
                edge(v, conf, {j: 4 * (d[j] + 1) + RET})
            if j + 1 < length and h[j] == INC and d[j] == radix - 1 and h[j + 1] == NR:
                edge(v, conf, {j: NL, j + 1: 4 * d[j + 1] + INC})
            if j == length - 1 and h[j] == INC and d[j] == radix - 1:
                edge(v, conf, {j: RET})
            if j + 1 < length and d[j] == 0 and h[j] == NL and h[j + 1] == RET:
                edge(v, conf, {j: RET, j + 1: 4 * d[j + 1] + NR})
            if j == 0 and h[j] == RET:
                edge(v, conf, {j: 4 * d[j] + INC})
    free = penalties == 0
    max_degree_one = max(sum(adj[v].values()) for v in range(dim) if heads[v] == 1)
    assert max_degree_one <= 2
    assert not np.any(free & (heads >= 2))
    product = math.prod(radices)
    assert sum(free & (heads == 1)) == 2 * length * product
    gap = math.inf
    seen = set()
    main = INC * powers[0] + sum(NR * powers[j] for j in range(1, length))
    main_n = None
    largest = 0
    for start in range(dim):
        if start in seen:
            continue
        component = [start]
        seen.add(start)
        for v in component:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    component.append(w)
        largest = max(largest, len(component))
        if main in component:
            main_n = len(component)
        if len(component) == 1:
            if penalties[start] > 0:
                gap = min(gap, float(penalties[start]))
            continue
        indices = {v: i for i, v in enumerate(component)}
        H = np.zeros((len(component), len(component)))
        for v, i in indices.items():
            H[i, i] = penalties[v] + sum(adj[v].values())
            for w, weight in adj[v].items():
                H[i, indices[w]] -= weight
        eig = np.linalg.eigvalsh(H)
        positive = eig[eig > 1e-8]
        if len(positive):
            gap = min(gap, float(positive[0]))
    expected_n = 2 * sum(product // math.prod(radices[:j]) for j in range(length))
    assert main_n == expected_n
    lower = 1 / (64 * length ** 2 * product ** 2)
    upper = 4 * math.pi ** 2 / product ** 2
    assert lower <= gap + 1e-10 and gap <= upper + 1e-10
    return dict(radices=radices, outside=outside, dimension=dim, gap=gap,
                main_n=main_n, largest_component=largest, lower=lower, upper=upper,
                lower_ratio=gap/lower, max_degree_one=max_degree_one)


def exhaustive_penalty():
    checked = 0
    worst = (math.inf, None)
    for n in range(2, 6):
        pairs = list(itertools.combinations(range(n), 2))
        for edge_mask in range(1, 1 << len(pairs)):
            edges = [pair for i, pair in enumerate(pairs) if edge_mask >> i & 1]
            neighbors = [set() for _ in range(n)]
            lap = np.zeros((n, n))
            for a, b in edges:
                neighbors[a].add(b); neighbors[b].add(a)
                lap[a, a] += 1; lap[b, b] += 1
                lap[a, b] -= 1; lap[b, a] -= 1
            reached = {0}
            stack = [0]
            for v in stack:
                for w in neighbors[v] - reached:
                    reached.add(w); stack.append(w)
            if len(reached) != n:
                continue
            D = max(map(len, neighbors))
            for mask in range(1, (1 << n) - 1):
                free = {v for v in range(n) if not mask >> v & 1}
                g = 0
                while free:
                    s = free.pop(); component = [s]
                    for v in component:
                        new = neighbors[v] & free
                        free -= new; component.extend(new)
                    g = max(g, len(component))
                V = np.array([(mask >> v) & 1 for v in range(n)])
                lam = float(np.linalg.eigvalsh(lap + np.diag(V))[0])
                ratio = lam * 8 * D * g * g
                checked += 1
                assert ratio >= 1 - 1e-10
                if ratio < worst[0]:
                    worst = (ratio, dict(n=n, edges=edges, penalty_mask=mask, lambda0=lam, D=D, g=g))
    return dict(checked=checked, worst_ratio=worst[0], witness=worst[1])


def classical_low_energy():
    # All configurations with flag energy at most ground+1 on a 2x2 box.
    # Every omitted flag pattern has flag energy >= ground+2; rules are nonnegative.
    L = 2
    expected = [(int(r == 0), int(c == 0), int(r == L-1), int(c == L-1))
                for r in range(L) for c in range(L)]
    flags_flat = sum((list(t) for t in expected), [])
    energies = defaultdict(int)
    for flip in [None] + list(range(4 * L * L)):
        fs = list(flags_flat)
        if flip is not None:
            fs[flip] ^= 1
        flags = [fs[i:i+4] for i in range(0, len(fs), 4)]
        for packed in itertools.product(range(8), repeat=L*L):
            b = [v & 1 for v in packed]
            carry = [(v >> 1) & 1 for v in packed]
            past = [(v >> 2) & 1 for v in packed]
            E = -4 * L + int(flip is not None)
            for r in range(L):
                for c in range(L):
                    j = r * L + c
                    T, F, B, R = flags[j]
                    E += bool(T and (b[j] or past[j]))
                    E += bool(F and not carry[j])
                    if c:
                        E += bool(not F and carry[j] != (b[j-1] & carry[j-1]))
                    if r:
                        E += bool(not T and ((b[j] != (b[j-L] ^ carry[j-L]))
                                           or (past[j] != (past[j-L] | b[j-L]))))
            energies[E] += 1
    assert energies[-8] == 1 and energies[-7] > 0
    # Independently track all carries and 'has shown a one' bits at modest sizes.
    tested = []
    for L in list(range(2, 65)) + [127, 128, 129, 255, 256, 257]:
        width = L
        bits = [0] * width
        past = [0] * width
        for row in range(L):
            assert sum(bit << j for j, bit in enumerate(bits)) == row
            if row == L - 1:
                region = [j for j in range(width) if bits[j] or past[j]]
                assert region == list(range((L - 1).bit_length()))
            carry = 1
            new = []
            for j in range(width):
                new.append(bits[j] ^ carry)
                carry &= bits[j]
                past[j] |= bits[j]
            bits = new
        tested.append(L)
    return dict(low_energy_enumerated=17*8**4, ground_energy=-8, ground_multiplicity=energies[-8],
                first_excited_energy=-7, first_excited_multiplicity_in_enumeration=energies[-7],
                tableau_sizes=tested)


def word_checks():
    records = []
    for mode in ('zero', 'one', 'third', 'oscillating'):
        N = 1
        ones = 0
        for s in range(40):
            slope = {'zero': Fraction(0), 'one': Fraction(1), 'third': Fraction(1,3),
                     'oscillating': Fraction(1 if s % 2 == 0 else 3, 4)}[mode]
            Nnext = (s + 2) * N
            length = Nnext - N
            count = (length * slope.numerator) // slope.denominator
            previous_density = Fraction(ones, N)
            ones += count
            density = Fraction(ones, Nnext)
            assert abs(density - slope) <= Fraction(1, s + 2) + Fraction(s + 1, Nnext)
            # Formula for an arbitrary within-stage prefix and its convex interpolation.
            t = length // 3
            partial = (t * slope.numerator) // slope.denominator
            actual = (previous_density * N + partial) / (N + t)
            interpolation = (previous_density * N + slope * t) / (N + t)
            assert abs(actual - interpolation) <= Fraction(1, N+t)
            if s >= 37:
                records.append(dict(mode=mode, stage=s, end_density=float(density), slope=float(slope)))
            N = Nnext
    return records


if __name__ == '__main__':
    out = {}
    out['penalty'] = exhaustive_penalty()
    print('Exhaustive penalty test:', json.dumps(out['penalty']), flush=True)
    out['counter'] = []
    for l in range(1, 4):
        for radices in itertools.product((1,2,4), repeat=l):
            out['counter'].append(counter(list(radices)))
    out['counter'].append(counter([2,2,2,2], rmax=2))
    for radices in ([1], [2], [1,1], [2,1], [1,2], [2,2]):
        out['counter'].append(counter(radices, outside=1))
    print('Independent counter cases:', len(out['counter']), 'PASS', flush=True)
    out['classical'] = classical_low_energy()
    print('Independent four-flag classical and carry checks:', json.dumps(out['classical']), flush=True)
    out['words'] = word_checks()
    print('Independent stage-density identities: PASS', flush=True)
    with open('independent_checks.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('ALL INDEPENDENT CHECKS PASS', flush=True)
