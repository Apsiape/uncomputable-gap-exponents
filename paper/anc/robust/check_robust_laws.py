"""Finite illustrations for Uncomputable critical exponents, version 2.1.0.

Run: python -u check_robust_laws.py
One numerical thread; outputs are written beside this script.
Sections 20.6-20.8 contain the related analytic arguments.
C1 checks majority-tail comparisons for Theorem 20.9 and the restricted
forgetting comparison. C2 checks finite mechanical block endpoints and
dyadic counts related to Theorem 20.9. C3 illustrates finitely delayed
revisions related to Proposition 20.11, using a computable toy schedule.
C4 records a capped countdown on a finite torus and a contracting control.
Its plane majorities have overlapping cones; its settling is computable,
and its window/time table only illustrates a finite transient.
These checks do not certify an infinite reliability theorem, a universal
compiler, an uncomputable limiting number, noncommuting infinite limits,
or a robust intrinsic equilibrium exponent. Sample noise probabilities
are not certified members of the all-time reliability region.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, math, time
from fractions import Fraction
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
results = {}

def binom_tail_ge(m, p, k):
    """P(Bin(m,p) >= k), computed by a finite floating-point binomial sum."""
    s = 0.0
    for j in range(k, m + 1):
        s += math.comb(m, j) * (p ** j) * ((1 - p) ** (m - j))
    return s

# ---------------- C1: majority bounds ----------------
c1 = {"hoeffding_delta_quarter": [], "half_perturbation": []}
ok = True
for m in [1, 3, 5, 9, 17, 33, 65, 129]:
    exact = binom_tail_ge(m, 0.25, (m + 1) // 2)          # wrong majority when each copy wrong w.p. <= 1/4
    bound = math.exp(-m / 8.0)
    c1["hoeffding_delta_quarter"].append({"m": m, "exact": exact, "bound_exp(-m/8)": bound})
    ok &= exact <= bound + 1e-15
for m in [5, 17, 65, 257]:
    for eta in [1e-2, 1e-3, 1e-4]:
        dev = binom_tail_ge(m, 0.5 + eta, (m + 1) // 2) - 0.5   # monotone in each p_l, so this is the worst case
        bound = m * eta * math.comb(m - 1, (m - 1) // 2) / 2 ** (m - 1)  # m * eta * P(tie among m-1 at p=1/2), first order
        c1["half_perturbation"].append({"m": m, "eta": eta, "deviation": dev, "first_order_bound": bound,
                                        "sqrt(2m/pi)*eta": math.sqrt(2 * m / math.pi) * eta})
        ok &= dev <= 1.05 * bound
c1["pass"] = bool(ok)
results["C1_majority_bounds"] = c1
print("C1 majority bounds:", "PASS" if ok else "FAIL", flush=True)

# ---------------- C2: arithmetic constructions ----------------
# (a) density by mechanical bits on blocks N_{s+1} = (s+2) N_s, with a computable stand-in u_s -> rho
def density_construction(u, S):
    N = 1
    ones = 0
    rec = []
    for s in range(S):
        Nn = (s + 2) * N
        # bits A(N+t) = floor((t+1)u_s) - floor(t u_s), t = 0..Nn-N-1 ; count of ones = floor((Nn-N) u_s)
        ones += math.floor((Nn - N) * u(s))
        N = Nn
        rec.append((s, N, ones / N))
    return rec
rho = Fraction(3, 10)
u = lambda s: float(rho) + ((-1) ** s) / (s + 3)   # converges to 0.3, non-monotone
rec = density_construction(u, 14)
dens_err = [abs(d - float(rho)) for (_, _, d) in rec]
c2a = {"endpoint_densities": [{"s": s, "N_s": N, "density": d} for (s, N, d) in rec],
       "pass": bool(dens_err[-1] < 0.1 and dens_err[-1] < dens_err[3])}
# (b) counting exponent by dyadic blocks A = U_k [2^k, 2^k + M_k), M_k = floor(2^(k u_k)), u_k -> gamma
gamma = 0.6
uk = lambda k: gamma + 0.3 * ((-1) ** k) / math.sqrt(k + 1)
rows = []
cum = 0
for K in range(1, 61):
    Mk = math.floor(2 ** (K * min(1.0, max(0.0, uk(K)))))
    Mk = min(Mk, 2 ** K)
    cum += Mk
    N = 2 ** (K + 1)               # count of A below 2^(K+1)
    rows.append({"K": K, "log2(1+count)/log2(N)": math.log2(1 + cum) / math.log2(N)})
c2b = {"gamma": gamma, "tail": rows[-6:], "pass": bool(abs(rows[-1]["log2(1+count)/log2(N)"] - gamma) < 0.06)}
results["C2_arithmetic_constructions"] = {"density": c2a, "exponent": c2b}
print("C2 density construction:", "PASS" if c2a["pass"] else "FAIL",
      "| exponent construction:", "PASS" if c2b["pass"] else "FAIL", flush=True)

# ---------------- C3: rate collapse illustration ----------------
# toy 'machines' e = 0..5 halting at h_e = 2^(2^e) (e in a toy set H), r = sum_{e in H} 4^(-e-1)
H = {0: 2, 1: 4, 2: 16, 3: 256, 4: 65536, 5: 2 ** 32}
r = sum(4.0 ** (-e - 1) for e in H)
viol = []
for e, h in H.items():
    s = h - 1
    q = sum(4.0 ** (-f - 1) for f, hf in H.items() if hf <= s)
    viol.append({"e": e, "s": s, "|q_s - r|": r - q, "1/s": 1.0 / s, "exceeds_1/s": (r - q) > 1.0 / s})
c3 = {"rows": viol, "note": "late revisions (h_e >= 2^16) beat the computable rate 1/s; early ones need not: the infinite statement is Proposition 20.11; this toy is finite", "pass": all(v["exceeds_1/s"] for v in viol if v["s"] >= 2 ** 16 - 1)}
results["C3_rate_collapse_illustration"] = c3
print("C3 rate-collapse illustration:", "PASS" if c3["pass"] else "FAIL", flush=True)

# ---------------- C4: Toom-stacked countdown world ----------------
D = 40                      # countdown range
Q = 2 * (D + 1)             # alphabet size: symbol = 2*d + b
W, L = 24, 48               # transverse W x W torus, L program cells (radius-0 base rule)

def base_rule(sym):
    d = sym >> 1
    b = sym & 1
    nb = b ^ (d == 1)
    nd = np.maximum(d - 1, 0)
    return (2 * nd + nb).astype(np.int16)

def toom_major(x):
    """Full-symbol NEC majority of x[a,b], x[a+1,b], x[a,b+1] (torus); if all three differ keep x[a,b]."""
    y1 = np.roll(x, -1, axis=0)
    y2 = np.roll(x, -1, axis=1)
    out = x.copy()
    agree12 = (y1 == y2)
    out[agree12] = y1[agree12]     # if the two neighbours agree they form a majority (or all three agree)
    return out

def step(x, rng, kernel):
    g = base_rule(toom_major(x))
    kind, p, q_reset = kernel
    if kind == "uniform":
        hit = rng.random(x.shape) < p
        if hit.any():
            shift = rng.integers(1, Q, size=int(hit.sum()))
            g[hit] = (g[hit] + shift) % Q          # uniform among the Q-1 wrong symbols
    elif kind == "contract":
        uu = rng.random(x.shape)
        reset = uu < q_reset
        unif = (uu >= q_reset) & (uu < q_reset + p)
        g[reset] = 0
        if unif.any():
            g[unif] = rng.integers(0, Q, size=int(unif.sum()))
    return g

def program(name):
    tau = np.zeros(L, dtype=np.int16)
    for i in range(L):
        if name == "P_half" and i % 2 == 0:
            tau[i] = min(D, i + 1)        # register i is (re)written at time tau_i = i+1 (settling grows with i)
        if name == "P_quarter" and i % 4 == 0:
            tau[i] = min(D, i + 1)
    x = np.broadcast_to((2 * tau)[None, None, :], (W, W, L)).astype(np.int16).copy()
    return x, tau

def logical_bits(tau, t):
    return ((tau >= 1) & (tau <= t)).astype(np.int8)   # bit flips to 1 exactly when the countdown reaches 0

def readout(x):
    bits = (x & 1).astype(np.int32)
    frac = bits.mean(axis=(0, 1))
    return (frac > 0.5).astype(np.int8)                 # plane majority (finite illustration, not independent copies)

T_list = [4, 8, 16, 24, 32, 40, 48, 60]
N_list = [8, 16, 32, 48]
kernels = [("uniform", 0.0, 0.0), ("uniform", 0.01, 0.0), ("uniform", 0.03, 0.0), ("uniform", 0.06, 0.0),
           ("contract", 0.05, 0.75)]
c4 = {"W": W, "L": L, "D": D, "alphabet": Q, "runs": []}
all_ok = True
for pi_, prog in enumerate(["P_half", "P_quarter", "P_zero"]):
    for ki_, kern in enumerate(kernels):
        rng = np.random.default_rng(29102026 + 10 * pi_ + ki_)
        x, tau = program(prog)
        rho_tab = {}
        wrong_frac = {}
        first_flip = [None] * L
        for t in range(1, max(T_list) + 1):
            x = step(x, rng, kern)
            ro = readout(x)
            for i in range(L):
                if first_flip[i] is None and ro[i] == 1:
                    first_flip[i] = t
            if t in T_list:
                lb = logical_bits(tau, t)
                rho_tab[t] = {N: float(ro[:N].mean()) for N in N_list}
                rho_tab[t]["logical_N48"] = float(lb.mean())
                wrong_frac[t] = float(((x & 1) != lb[None, None, :]).mean())
        run = {"program": prog, "kernel": kern, "rho_N(t)": rho_tab, "single_site_bit_error_fraction": wrong_frac}
        if kern[0] == "uniform" and kern[1] <= 0.03:
            exp_flip = [int(tau[i]) if tau[i] >= 1 else None for i in range(L)]
            run["first_observed_flip_matches_countdown"] = (first_flip == exp_flip)
            all_ok &= run["first_observed_flip_matches_countdown"]
        if kern[0] == "contract":
            kappa_bound = 3 * (1 - kern[1] - kern[2])
            run["dobrushin_kappa_upper_bound"] = kappa_bound
            run["program_erased(all readouts 0 at t>=16)"] = all(rho_tab[t][48] == 0.0 for t in T_list if t >= 16)
            all_ok &= (kappa_bound < 1) and run["program_erased(all readouts 0 at t>=16)"]
        c4["runs"].append(run)
        print("C4", prog, kern, "rho_48(t):", [rho_tab[t][48] for t in T_list], flush=True)

# Finite window/time transient for P_half at p=0.03; no infinite-limit claim.
ph = [r for r in c4["runs"] if r["program"] == "P_half" and r["kernel"] == ("uniform", 0.03, 0.0)][0]
c4["finite_window_time_transient_P_half_p0.03"] = {
    "fixed_t=8, N=8,16,32,48": [ph["rho_N(t)"][8][N] for N in N_list],
    "fixed_N=8, t=8..60": [ph["rho_N(t)"][t][8] for t in T_list],
    "fixed_N=48, t=8..60": [ph["rho_N(t)"][t][48] for t in T_list]}
c4["pass"] = bool(all_ok)
results["C4_toom_countdown_world"] = c4
print("C4 Toom-stacked countdown world:", "PASS" if all_ok else "FAIL", flush=True)

results["runtime_seconds"] = time.time() - t0
try:
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32"); psapi = ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    results["peak_working_set_bytes"] = int(pmc.PeakWorkingSetSize)
except Exception as ex:  # non-Windows or API failure
    results["peak_working_set_bytes"] = None
    results["memory_probe_error"] = str(ex)
results["all_pass"] = bool(c1["pass"] and c2a["pass"] and c2b["pass"] and c3["pass"] and c4["pass"])
with open(os.path.join(OUT, "checks.json"), "w") as f:
    json.dump(results, f, indent=1, default=str)
assert results["all_pass"], "Robust-law finite check failed"
print("ALL PASS" if results["all_pass"] else "SOME CHECK FAILED", "| runtime %.2f s | peak WS %s bytes"
      % (results["runtime_seconds"], results["peak_working_set_bytes"]), flush=True)
