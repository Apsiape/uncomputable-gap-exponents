"""Checks for the classical layer of the film Hamiltonian (Theorem 12.1).

(1) Flag lemma in 3D (Section 12.1): for each of the six flag types separately (the flag energy is a sum of six
    independent parts), exhaustive minimisation over all 2^|box| assignments on small 3D boxes: the minimum is
    -(face area), attained only by the exact face.
(2) Thickness counter + output transport in one (x,z) plane, exact flags: count ALL solutions of the local rules by a
    depth-first search in a RANDOM variable order (so the functional order is not used); expect exactly one, whose
    top row marks the region {x : x < bitlength(M-1)} and carries the output bits upward unchanged.
"""
import itertools, json, random, sys


def flag_check(Lx, Ly, Lz):
    sites = [(x, y, z) for x in range(Lx) for y in range(Ly) for z in range(Lz)]
    idx = {s: i for i, s in enumerate(sites)}
    n = len(sites)
    res = {}
    dirs = {'X0': (0, -1), 'X1': (0, +1), 'Y0': (1, -1), 'Y1': (1, +1), 'Z0': (2, -1), 'Z1': (2, +1)}
    for name, (ax, sgn) in dirs.items():
        best, arg, cnt = None, None, 0
        nb = []  # for each site, does it have a neighbour in the outward direction
        for s in sites:
            t = list(s); t[ax] += sgn
            nb.append(tuple(t) in idx)
        for mask in range(1 << n):
            e = 0
            for i in range(n):
                if mask >> i & 1:
                    e += -1 + (2 if nb[i] else 0)
            if best is None or e < best:
                best, arg, cnt = e, mask, 1
            elif e == best:
                cnt += 1
        face = sum(1 << i for i in range(n) if not nb[i])
        area = bin(face).count('1')
        res[name] = dict(min=best, unique=(cnt == 1), is_face=(arg == face), area=area, ok=(cnt == 1 and arg == face and best == -area))
    return res


def plane_check(L, M, out, seed):
    # variables: (field, x, z) for field in b, c, f, o
    vars_ = [(fld, x, z) for fld in 'bcfo' for x in range(L) for z in range(M)]
    cons = []  # (list of vars, predicate on dict)
    for x in range(L):
        cons.append(([('b', x, 0)], lambda a, x=x: a[('b', x, 0)] == 0))
        cons.append(([('f', x, 0)], lambda a, x=x: a[('f', x, 0)] == 0))
        cons.append(([('o', x, 0)], lambda a, x=x: a[('o', x, 0)] == out[x]))
    for z in range(M):
        cons.append(([('c', 0, z)], lambda a, z=z: a[('c', 0, z)] == 1))
        for x in range(L - 1):
            v = [('c', x + 1, z), ('b', x, z), ('c', x, z)]
            cons.append((v, lambda a, x=x, z=z: a[('c', x + 1, z)] == (a[('b', x, z)] & a[('c', x, z)])))
    for z in range(M - 1):
        for x in range(L):
            cons.append(([('b', x, z + 1), ('b', x, z), ('c', x, z)], lambda a, x=x, z=z: a[('b', x, z + 1)] == (a[('b', x, z)] ^ a[('c', x, z)])))
            cons.append(([('f', x, z + 1), ('f', x, z), ('b', x, z)], lambda a, x=x, z=z: a[('f', x, z + 1)] == (a[('f', x, z)] | a[('b', x, z)])))
            cons.append(([('o', x, z + 1), ('o', x, z)], lambda a, x=x, z=z: a[('o', x, z + 1)] == a[('o', x, z)]))
    order = vars_[:]
    if seed is None:
        # reversed (anti-functional) order: top rows first, right to left
        order.sort(key=lambda v: (-v[2], -v[1], v[0]))
    else:
        random.Random(seed).shuffle(order)
    pos = {v: i for i, v in enumerate(order)}
    # attach each constraint to the latest variable (in the order) it mentions
    attach = {v: [] for v in order}
    for vs, pr in cons:
        attach[max(vs, key=lambda v: pos[v])].append(pr)
    sols = []
    a = {}
    def dfs(i):
        if len(sols) > 2:
            return
        if i == len(order):
            sols.append(dict(a)); return
        v = order[i]
        for val in (0, 1):
            a[v] = val
            if all(pr(a) for pr in attach[v]):
                dfs(i + 1)
            del a[v]
    dfs(0)
    ok = len(sols) == 1
    info = dict(L=L, M=M, solutions=len(sols))
    if ok:
        s = sols[0]
        region = [x for x in range(L) if (s[('f', x, M - 1)] | s[('b', x, M - 1)])]
        lM = (M - 1).bit_length()
        top_out = [s[('o', x, M - 1)] for x in range(L)]
        info.update(region=region, expected=list(range(min(lM, L))), region_ok=(region == list(range(min(lM, L)))),
                    transport_ok=(top_out == out))
        ok = info['region_ok'] and info['transport_ok']
    info['ok'] = ok
    return info


if __name__ == '__main__':
    allok = True
    out = dict(flags={}, planes=[])
    for box in [(2, 2, 2), (3, 2, 2), (3, 3, 2), (2, 2, 4)]:
        r = flag_check(*box)
        ok = all(v['ok'] for v in r.values())
        allok &= ok
        out['flags'][str(box)] = r
        assert ok, "Finite check condition failed"
        print('flags', box, 'ok' if ok else 'FAIL', {k: (v['min'], v['area']) for k, v in r.items()}, flush=True)
    rnd = random.Random(7)
    jobs = [(L, M, 100 * M + L) for M in range(1, 6) for L in (3, 4)] + [(L, M, None) for M in range(1, 10) for L in (4, 5)]
    for (L, M, sd) in jobs:
            o = [rnd.randint(0, 1) for _ in range(L)]
            info = plane_check(L, M, o, seed=sd)
            info['order'] = 'random' if sd is not None else 'reversed'
            allok &= info['ok']
            out['planes'].append(info)
            print('plane', info, flush=True)
    assert allok, "Finite check condition failed"
    print('ALL PASS' if allok else 'SOME FAIL', flush=True)
    out['all_pass'] = allok
    with open(sys.argv[1] if len(sys.argv) > 1 else 'check_film_classical_layer.json', 'w') as fh:
        json.dump(out, fh, indent=1)
