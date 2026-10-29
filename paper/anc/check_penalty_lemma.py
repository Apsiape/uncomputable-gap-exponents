"""
Random test of the penalty localization lemma (penalty localization): for a connected graph C of max degree D with a 0/1 penalty V that is
not identically 0, and largest penalty-free cluster g, the lowest eigenvalue of L_C + V is >= 1/(8 D g^2).
Also reports the worst observed ratio lambda_0 * 8 D g^2 (must be >= 1), and the tightest family found.
Graphs: random paths/cycles with sparse penalties (the case used in the theorem), random trees, random bounded-degree
graphs, stars (where the degree factor is necessary).  Run: python -u check_penalty_lemma.py
"""
import json
import numpy as np

rng = np.random.default_rng(20261028)


def lap(n, edges):
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] += 1; A[v, u] += 1
    return np.diag(A.sum(1)) - A


def clusters(n, edges, free):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for u, v in edges:
        if free[u] and free[v]:
            parent[find(u)] = find(v)
    sizes = {}
    for v in range(n):
        if free[v]:
            r = find(v); sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) if sizes else 0


def connected(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    seen = {0}; st = [0]
    while st:
        x = st.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y); st.append(y)
    return len(seen) == n


def test(n, edges, V):
    free = V == 0
    if free.all() or not connected(n, edges):
        return None
    g = max(clusters(n, edges, free), 1)
    deg = np.zeros(n)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    D = max(int(deg.max()), 1)
    lam0 = np.linalg.eigvalsh(lap(n, edges) + np.diag(V))[0]
    return lam0 * 8 * D * g * g, lam0, D, g


worst = {}
for family in ('path', 'cycle', 'tree', 'bounded', 'star'):
    w = (np.inf, None)
    for trial in range(400):
        n = int(rng.integers(3, 90))
        if family == 'path':
            edges = [(i, i + 1) for i in range(n - 1)]
        elif family == 'cycle':
            edges = [(i, (i + 1) % n) for i in range(n)]
        elif family == 'tree':
            edges = [(i, int(rng.integers(0, i))) for i in range(1, n)]
        elif family == 'bounded':
            edges = [(i, i + 1) for i in range(n - 1)]
            for _ in range(n // 2):
                u, v = rng.integers(0, n, 2)
                if u != v:
                    edges.append((int(u), int(v)))
        else:
            edges = [(0, i) for i in range(1, n)]
        V = (rng.random(n) < rng.choice([0.02, 0.05, 0.2, 0.5])).astype(float)
        if family == 'star':
            V = np.zeros(n); V[0] = 1.0
        if V.sum() == 0:
            V[int(rng.integers(0, n))] = 1.0
        r = test(n, edges, V)
        if r is None:
            continue
        if r[0] < w[0]:
            w = (r[0], dict(n=n, lambda0=r[1], D=r[2], g=r[3]))
    worst[family] = dict(min_ratio=float(w[0]), at=w[1])
    print(family, json.dumps(worst[family]), flush=True)
ok = all(v['min_ratio'] >= 1 - 1e-12 for v in worst.values())
assert ok, "Finite check condition failed"
print('the penalty localization lemma bound lambda0 >= 1/(8 D g^2) on all random instances:', 'PASS' if ok else 'FAIL', flush=True)
with open('check_penalty_lemma.json', 'w') as f:
    json.dump(worst, f, indent=1)
