"""Sequential, standard-library checks for reliable revising computation.
Run: python -u check_states.py. Related arguments are in Section 19.
No noncomputability or infinite-time claim is inferred from finite data.
"""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

OUT = Path(__file__).resolve().parent

def peak_bytes():
    if os.name != "nt":
        return None
    class COUNTERS(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = COUNTERS()
    c.cb = ctypes.sizeof(c)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(COUNTERS), ctypes.c_ulong]
    memory_info.restype = ctypes.c_int
    if not memory_info(get_process(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(c.PeakWorkingSetSize)

def binomial_tail(p, m):
    return sum(math.comb(m, k) * p**k * (1-p)**(m-k) for k in range(m//2+1, m+1))

def finite_ising(n, beta_j=0.3):
    values = []
    for left in (-1, 1):
        for right in (-1, 1):
            z = a = 0.0
            for word in range(1 << n):
                spin = [1 if (word >> i) & 1 else -1 for i in range(n)]
                energy = left*spin[0] + right*spin[-1] + sum(spin[i]*spin[i+1] for i in range(n-1))
                w = math.exp(beta_j*energy)
                z += w
                a += w*spin[n//2]
            values.append(a/z)
    width = max(values) - min(values)
    # For n=2L+1, plus/plus magnetization at its center is
    # 2*tanh(K)^(L+1)/(1+tanh(K)^(2L+2)).
    distance = n//2+1
    r = math.tanh(beta_j)
    analytic_width = 4*r**distance/(1+r**(2*distance))
    assert abs(width-analytic_width) < 2e-12
    return {"n": n, "width": width, "analytic_width": analytic_width}

def standin(t):
    return int(8 <= t < 24 or t >= 48)

def stacked_shift(p, seed, side=32, length=96, horizon=80):
    """F=shift; Toom-majority in each redundant plane, then shift.
    Finite torus: demonstration only. No temporal wrap reaches plane 0
    before horizon because length > horizon.
    """
    rng = random.Random(seed)
    area = side*side
    state = bytearray(length*area)
    for i in range(length):
        state[i*area:(i+1)*area] = bytes([standin(i)])*area
    north = [((k//side+1) % side)*side+k % side for k in range(area)]
    east = [(k//side)*side+(k % side+1) % side for k in range(area)]
    wanted = {4, 8, 16, 24, 32, 48, 64, 80}
    rows = []
    for t in range(1, horizon+1):
        nxt = bytearray(length*area)
        for i in range(length):
            src = ((i+1) % length)*area
            dst = i*area
            for k in range(area):
                bit = int(state[src+k]+state[src+north[k]]+state[src+east[k]] >= 2)
                nxt[dst+k] = bit ^ int(rng.random() < p)
        state = nxt
        if t in wanted:
            target = standin(t)
            ones = sum(state[:area])
            errors = ones if target == 0 else area-ones
            rows.append({"t": t, "intended": target, "fraction_wrong": errors/area,
                         "decoded_majority": int(ones > area/2),
                         "uncorrected_shift_expected_error": (1-(1-2*p)**t)/2})
    return {"p": p, "seed": seed, "side": side, "length": length,
            "horizon": horizon, "rows": rows}

def main():
    started = time.perf_counter()
    # A radius-zero two-state PCA: u=P(1|0), v=P(0|1).
    u, v = 0.12, 0.28
    p = u/(u+v)
    q = 0.0
    for _ in range(100):
        q = u+(1-u-v)*q
    assert abs(q-p) < 1e-14
    amp = []
    for m in (5, 11, 21, 41, 81):
        e = binomial_tail(p, m)
        bound = math.exp(-2*m*(0.5-p)**2)
        assert e <= bound+1e-14
        amp.append({"m": m, "exact_error": e, "hoeffding_upper": bound})
    ising = [finite_ising(n) for n in (1, 3, 5, 9, 13)]
    # Enumerate simple one-dimensional causal-cone intervals. The
    # redundant coordinate moves at speed <=1; separation 2T+1 suffices.
    for t in range(101):
        spacing = 2*t+1
        cones = [set(range(k*spacing-t, k*spacing+t+1)) for k in range(9)]
        assert all(not (cones[i] & cones[j]) for i in range(9) for j in range(i))
    # Constant-normalizer tableau identity, checked by brute force.
    beta = 1.7
    depth, width = 3, 3
    bottom = [1, 0, 1]
    z = 0.0
    max_delta = 0.0
    norm = 1+math.exp(-beta)
    for word in range(1 << (depth*width)):
        rows = [bottom]+[[(word >> (t*width+i)) & 1 for i in range(width)] for t in range(depth)]
        faults = sum(rows[t+1][i] != rows[t][(i+1) % width] for t in range(depth) for i in range(width))
        weight = math.exp(-beta*faults)
        prob = 1.0
        for t in range(depth):
            for i in range(width):
                prob *= (math.exp(-beta) if rows[t+1][i] != rows[t][(i+1) % width] else 1)/norm
        max_delta = max(max_delta, abs(weight/norm**(depth*width)-prob))
        z += weight
    assert abs(z-norm**(depth*width)) < 1e-11
    assert max_delta < 1e-14
    sims = [stacked_shift(p, 290026+i) for i,p in enumerate((0.01, 0.05, 0.12))]
    result = {"pca_zero_radius": {"u": u, "v": v, "stationary_p": p, "q_after_100": q},
              "majority_amplification": amp, "ising_boundary_brackets": ising,
              "disjoint_cone_checks": 101*36,
              "tableau": {"cases": 512, "z_enumerated": z, "z_factorized": norm**9,
                          "maximum_probability_discrepancy": max_delta},
              "stacked_shift": sims, "seconds": time.perf_counter()-started,
              "peak_working_set_bytes": peak_bytes(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    assert result["peak_working_set_bytes"] is None or result["peak_working_set_bytes"] < 1_000_000_000

if __name__ == "__main__":
    main()