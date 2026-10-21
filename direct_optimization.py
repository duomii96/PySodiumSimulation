"""
direct_optimize.py
==================
Direct optimisation of spin-3/2 TQTPPI simulation against experimental
SQ and TQ FID data. Launches 16 parallel Nelder-Mead runs from randomised
starting points scattered around an initial parameter estimate.

Usage
-----
Edit the USER INPUTS section, then run:
    python direct_optimize.py

Dependencies: numpy, scipy, joblib, numba
             (same as your dictionary builder)
"""

import numpy as np
import numba
import time
from scipy.optimize import minimize
from joblib import Parallel, delayed
from pathlib import Path
from getSQTQ_acqs import getSQTQ_fixAcqsDim
from query_dictionary import query_dictionary

from commonImports import *
from Classes.getValues import getValues
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic

# ─────────────────────────────────────────────────────────────────────────────
# USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────

# --- Scanner / sequence settings (match your dictionary builder) ---
B0              = 9.4          # Tesla
tevo            = 10e-3        # evolution time [s] — set to your experimental value

# --- Dictionary pre-match (coarse grid search → starting point) ---
# If enabled, the best dictionary entry replaces T2f_est / T2s_est below as the
# centre of the starting points, and its gamma replaces GAMMA_FIXED.
USE_DICT_START        = True
DICT_NAME             = "dict_general_gamma2045.h5"
DICT_NORM             = "2"      # "2" = L2, otherwise L1 (see query_dictionary)
DICT_GAMMA_FROM_MATCH = True     # True: search all gamma, fix gamma to the match
                                 # False: only entries with |gamma - GAMMA_FIXED| <= DICT_GAMMA_TOL
DICT_GAMMA_TOL        = 5        # [rad/s]

# --- Initial parameter estimates (used when USE_DICT_START = False) ---
T2f_est         = 11e-3         # [s]  fast transverse relaxation time estimate
T2s_est         = 40e-3        # [s]  slow transverse relaxation time estimate
gamma_est       = 34.5         # [rad/s] Cauchy scale of wShiftdist (FWHM = 2*gamma)

# --- Search scatter around estimate (±fraction of estimate) ---
# Each of the 16 runs draws a random start within these bounds.
# Keep physical: T2f in [0.5, 15] ms, T2s in [5, 100] ms, gamma in [5, 100] rad/s
SCATTER_FRAC    = 0.3         # ±30 % scatter around estimate for each parameter
N_RUNS          = 16           # number of parallel minimisations
N_JOBS          = 16           # joblib workers  (set to -1 for all cores)
RANDOM_SEED     = 42           # master seed for reproducible starting points

# --- Hard parameter bounds (enforced as penalty inside cost function) ---
T2F_BOUNDS      = (3e-3,  15e-3)   # [s]
T2S_BOUNDS      = (35e-3,   55e-3)   # [s]
GAMMA_BOUNDS    = (20,   50)     # [rad/s] — matches dictionary gamma range
GAMMA_FIXED     = 40             # [rad/s]

# --- gamma: fixed or free ---
# False: optimise (T2f, T2s); gamma stays at the dictionary match (or GAMMA_FIXED).
# True : optimise (T2f, T2s, gamma); gamma starts at the same value, limited by GAMMA_BOUNDS.
OPTIMIZE_GAMMA     = False
GAMMA_SCATTER_FRAC = 0.2         # ±20 % scatter of the gamma start (only if OPTIMIZE_GAMMA)
# Internally gamma is optimised as gamma * GAMMA_SCALE so it has the same order of
# magnitude as T2 in seconds — keeps Nelder-Mead's initial simplex and xatol balanced.
GAMMA_SCALE        = 1e-3

# --- Experimental data ---
# Replace these with your actual measured arrays (numpy 1-D, length = dataPoints).
# exp_SQ and exp_TQ must be normalised the same way the simulation output is used.
# Example: load from a .npy or .mat file.
studyPath = Path("G:/Transfer/Dominik/DataTemp/DZ_ThimoAgar_nReco_1_44_20260119_160948/38")
#studyPath = Path("G:/Transfer/Dominik/DataTemp/DZ_hem6_rapidVol_thimo_1_63_20260512_082106/75")
#studyPath = Path("G:/Transfer/Dominik/DataTemp/DZ_Agar2Temperature_1_86_20240708_084113/15")
#   import scipy.io as sio
#   data   = sio.loadmat('my_experiment.mat')
#   exp_SQ = data['SQ'].ravel().astype(float)
#   exp_TQ = data['TQ'].ravel().astype(float)
#
# For a self-contained test we generate synthetic data from the estimate:
_SYNTHETIC_TEST = False   # <-- set False when using real data

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION WRAPPER  (mirrors _run_single from your dictionary builder)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(T2f, T2s, gamma, tevo,
              B0, wQbar, FreqShift, wShiftFID,
              nSpins, NumPhaseCycles, alphas,
              flip90, tmix, deadtimeFID, dwelltimeFID, dataPoints,
              nPtsevo, nPtsmix,
              pdistname,
              idxTQ, idxSQ,
              rng_seed):
    """
    Run one full fix TQTPPI simulation and return (SQ_norm, TQ_norm).
    rng_seed fixes the wShiftdist Monte Carlo draws so the cost
    function is deterministic within a single optimisation run.
    """
    np.random.seed(rng_seed)
    numba.set_num_threads(1)   # one Numba thread per worker; loky handles parallelism

    w0    = getValues.getw0(B0)
    T1s   = 1.175 * T2s
    T1f   = 0.825 * T2s
    Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)

    wShiftdist = getValues.getWshiftDist(pdistname, 0.0, gamma)
    wShiftRMS  = 0.0

    sim = PhaseCyclesVecDic(B0, tauC, wQ, wQbar, Jen,
                            FreqShift, wShiftRMS, wShiftFID)
    sim.nSpins         = nSpins
    sim.NumPhaseCycles = NumPhaseCycles
    sim.alphas         = alphas
    sim.flip90         = flip90
    sim.tmix           = tmix
    sim.wShiftdist     = wShiftdist
    sim.dataPoints     = dataPoints
    sim.deadtimeFID    = deadtimeFID
    sim.dwelltimeFID   = dwelltimeFID
    sim.tevo           = tevo
    sim.nPtsevo        = nPtsevo
    sim.nPtsmix        = nPtsmix

    _, _, fid, _ = sim.TQTPPIfixedwo180VJ()

    # Phase-cycle FFT → select coherence orders (same as dictionary builder)
    FTphase = np.fft.fftshift(np.fft.fft(fid[:, :], axis=0), axes=0)
    SQ_raw  = np.real(FTphase[idxSQ, :])
    TQ_raw  = np.real(FTphase[idxTQ, :])

    # Normalise to unit peak so fitting is scale-independent
    sq_max = np.max(np.abs(SQ_raw))
    tq_max = np.max(np.abs(TQ_raw))
    SQ_norm = SQ_raw / sq_max if sq_max > 0 else SQ_raw
    TQ_norm = TQ_raw / tq_max if tq_max > 0 else TQ_raw

    return SQ_norm, TQ_norm


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE OPTIMISATION RUN  (one joblib worker executes this)
# ─────────────────────────────────────────────────────────────────────────────

def _run_optimisation(run_id, x0,
                      exp_SQ, exp_TQ,
                      tevo, sim_cfg):
    """
    Minimise ||sim(T2f, T2s, gamma) - exp||^2 via Nelder-Mead.
    x0 = [T2f_s, T2s_s]                       if gamma is fixed (sim_cfg['gamma_fixed'])
    x0 = [T2f_s, T2s_s, gamma * GAMMA_SCALE]  if sim_cfg['optimize_gamma']
    sim_cfg : dict with all fixed scanner / sequence parameters
    """
    rng_seed = sim_cfg['master_seed'] + run_id   # unique, reproducible per run

    eval_counter = [0]

    def unpack(params):
        if sim_cfg['optimize_gamma']:
            return params[0], params[1], params[2] / GAMMA_SCALE
        return params[0], params[1], sim_cfg['gamma_fixed']

    def cost(params):
        T2f, T2s, gamma = unpack(params)
        # --- hard constraint penalty ---
        if (T2f < T2F_BOUNDS[0] or T2f > T2F_BOUNDS[1] or
            T2s < T2S_BOUNDS[0] or T2s > T2S_BOUNDS[1] or
            gamma < GAMMA_BOUNDS[0] or gamma > GAMMA_BOUNDS[1] or
            T2s < 2.0 * T2f):          # physical: T2s >= 2.0*T2f
            print(f"    PENALTY hit: T2f={T2f * 1e3:.3f} T2s={T2s * 1e3:.3f} gamma={gamma:.1f}")
            return 1e12

        sim_SQ, sim_TQ = _simulate(
            T2f, T2s, gamma, tevo,
            sim_cfg['B0'],
            sim_cfg['wQbar'],
            sim_cfg['FreqShift'],
            sim_cfg['wShiftFID'],
            sim_cfg['nSpins'],
            sim_cfg['NumPhaseCycles'],
            sim_cfg['alphas'],
            sim_cfg['flip90'],
            sim_cfg['tmix'],
            sim_cfg['deadtimeFID'],
            sim_cfg['dwelltimeFID'],
            sim_cfg['dataPoints'],
            sim_cfg['nPtsevo'],
            sim_cfg['nPtsmix'],
            sim_cfg['pdistname'],
            sim_cfg['idxTQ'],
            sim_cfg['idxSQ'],
            rng_seed,
        )

        res = np.concatenate([sim_SQ - exp_SQ, sim_TQ - exp_TQ])
        eval_counter[0] += 1
        return float(np.sum(res**2))

    t0  = time.time()
    res = minimize(
        cost, x0,
        method='Nelder-Mead',
        options={
            'xatol':   1e-6,
            'fatol':   1e-8,
            'maxiter': 500,
            'adaptive': True,   # adaptive Nelder-Mead — better for 3-D problems
        },
    )
    elapsed = time.time() - t0
    T2f_opt, T2s_opt, gamma_opt = unpack(res.x)

    print(f"  Run {run_id:2d} | fval={res.fun:.6e} | "
          f"T2f={T2f_opt * 1e3:.3f} ms  T2s={T2s_opt * 1e3:.3f} ms  "
          f"gamma={gamma_opt:.2f} rad/s | "
          f"{eval_counter[0]} evals | {elapsed:.0f} s | "
          f"{'OK' if res.success else 'WARN: ' + res.message}")

    return {
        'run_id': run_id, 'x0': x0,
        'T2f': T2f_opt, 'T2s': T2s_opt, 'gamma': gamma_opt,
        'fval': res.fun, 'success': res.success,
        'neval': eval_counter[0], 'elapsed': elapsed, 'result': res,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DICTIONARY PRE-MATCH  (coarse grid minimum → centre of the starting points)
# ─────────────────────────────────────────────────────────────────────────────

def _dictionary_start(exp_SQ, exp_TQ, tevo):
    """
    Match the measured SQ/TQ against the dictionary and return the best entry's
    INPUT parameters (T2f, T2s, gamma) — the same quantities _simulate takes —
    not its biexponentially fitted T2*.
    The start is clipped into the hard bounds so run 0 is not a penalty point.
    """
    gamma_tol = np.inf if DICT_GAMMA_FROM_MATCH else DICT_GAMMA_TOL

    print(f"\n{'─'*70}")
    print(f"Dictionary pre-match: {DICT_NAME}")
    res = query_dictionary(DICT_NAME, exp_SQ, exp_TQ, tevo * 1e3,
                           DICT_NORM, GAMMA_FIXED, gamma_tol)

    if abs(res.tevo_used_ms - tevo * 1e3) > 0.5:
        print(f"  WARNING: dictionary has no tevo = {tevo*1e3:.1f} ms, matched at "
              f"{res.tevo_used_ms:.1f} ms — starting point may be biased.")

    T2f   = res.T2f_input_ms * 1e-3
    T2s   = res.T2s_input_ms * 1e-3
    gamma = res.gamma
    print(f"  Best entry : T2f={T2f*1e3:.3f} ms  T2s={T2s*1e3:.3f} ms  "
          f"gamma={gamma:.1f} rad/s  (distance {np.min(res.dist):.4e})")

    T2f_c   = float(np.clip(T2f, *T2F_BOUNDS))
    T2s_c   = float(max(np.clip(T2s, *T2S_BOUNDS), 2.0 * T2f_c))
    gamma_c = float(np.clip(gamma, *GAMMA_BOUNDS))
    if (T2f_c, T2s_c, gamma_c) != (T2f, T2s, gamma):
        print(f"  Clipped into bounds → T2f={T2f_c*1e3:.3f} ms  "
              f"T2s={T2s_c*1e3:.3f} ms  gamma={gamma_c:.1f} rad/s")
    print(f"{'─'*70}")

    return T2f_c, T2s_c, gamma_c, res


# ─────────────────────────────────────────────────────────────────────────────
# STARTING POINTS  (scattered around the dictionary match / estimate)
# ─────────────────────────────────────────────────────────────────────────────

def _make_starting_points(T2f_est, T2s_est,
                          scatter_frac, n_runs, seed,
                          gamma_est=None, gamma_scatter_frac=0.0):
    """
    Draw n_runs random starting points within ±scatter_frac of each estimate.
    The first point IS the estimate itself (run 0), the rest are perturbed.
    If gamma_est is given (gamma optimised), each point gets a third entry
    gamma * GAMMA_SCALE, scattered by ±gamma_scatter_frac.
    """
    rng    = np.random.default_rng(seed)
    free_gamma = gamma_est is not None

    def point(T2f, T2s, gamma):
        return np.array([T2f, T2s, gamma * GAMMA_SCALE] if free_gamma else [T2f, T2s])

    x0s = [point(T2f_est, T2s_est, gamma_est)]


    for _ in range(n_runs - 1):
        while True:
            scale = 1.0 + rng.uniform(-scatter_frac, scatter_frac, size=2)
            T2f, T2s = T2f_est * scale[0], T2s_est * scale[1]
            gamma = (gamma_est * (1.0 + rng.uniform(-gamma_scatter_frac, gamma_scatter_frac))
                     if free_gamma else None)
            # Reject if outside hard bounds or violating T2s >= 2*T2f (same as cost)
            if (T2F_BOUNDS[0] <= T2f <= T2F_BOUNDS[1] and
                    T2S_BOUNDS[0] <= T2s <= T2S_BOUNDS[1] and
                    T2s >= 2.0 * T2f and
                    (not free_gamma or GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1])):
                x0s.append(point(T2f, T2s, gamma))
                break

    return x0s

def fit_T2star(best, sim_cfg, tevo_fit=1e-3):
    """
    1. Re-run the simulation with best-fit (T2f, T2s, gamma) at tevo_fit.
    2. Fit SQ and TQ simultaneously with biexponential transfer functions
       that share T2f and T2s as common parameters:

           SQ(t) = As * exp(-t/T2s) + Af * exp(-t/T2f) + offset_SQ
           TQ(t) = ATQ * (exp(-t/T2s) - exp(-t/T2f))   + offset_TQ

    3. Return and print the fitted T2f* and T2s*.

    Parameter vector x = [As, T2s_ms, Af, T2f_ms, offset_SQ, ATQ, offset_TQ]
    — identical layout to your dictionary builder's biexponential fit.
    """

    T2f_opt  = best['T2f']
    T2s_opt  = best['T2s']
    gamma_opt = best['gamma']

    print(f"\n{'─'*70}")
    print(f"Post-optimisation T2* fit  (tevo = {tevo_fit*1e3:.1f} ms)")
    print(f"  Using best parameters: T2f={T2f_opt*1e3:.4f} ms  "
          f"T2s={T2s_opt*1e3:.4f} ms  gamma={gamma_opt:.2f} rad/s")
    print("  Running simulation …")

    # Fixed seed: deterministic re-simulation
    SQ_sim, TQ_sim = _simulate(
        T2f_opt, T2s_opt, gamma_opt, tevo_fit,
        sim_cfg['B0'], sim_cfg['wQbar'], sim_cfg['FreqShift'],
        sim_cfg['wShiftFID'], sim_cfg['nSpins'],
        sim_cfg['NumPhaseCycles'], sim_cfg['alphas'],
        sim_cfg['flip90'], sim_cfg['tmix'],
        sim_cfg['deadtimeFID'], sim_cfg['dwelltimeFID'],
        sim_cfg['dataPoints'], sim_cfg['nPtsevo'], sim_cfg['nPtsmix'],
        sim_cfg['pdistname'], sim_cfg['idxTQ'], sim_cfg['idxSQ'],
        rng_seed=sim_cfg['master_seed'],
    )

    # Acquisition time axis in ms (better numerical scaling for the fitter)
    acqT_ms = (sim_cfg['deadtimeFID'] +
               sim_cfg['dwelltimeFID'] * np.arange(sim_cfg['dataPoints'])) * 1e3

    # Normalise to unit peak
    sqn = SQ_sim / np.max(np.abs(SQ_sim))
    tqn = TQ_sim / np.max(np.abs(TQ_sim))

    # ── Transfer functions (T2s and T2f shared between SQ and TQ) ────────────
    # x = [As, T2s_ms, Af, T2f_ms, offset_SQ, ATQ, offset_TQ]
    def funSQ(x, t): return x[0]*np.exp(-t/x[1]) + x[2]*np.exp(-t/x[3]) + x[4]
    def funTQ(x, t): return x[5]*(np.exp(-t/x[1]) - np.exp(-t/x[3]))    + x[6]

    def residuals(x):
        return np.concatenate([funSQ(x, acqT_ms) - sqn,
                               funTQ(x, acqT_ms) - tqn])

    # Initial guess: use optimised T2 values converted to ms
    x0     = np.array([0.4,  T2s_opt*1e3,  0.6,  T2f_opt*1e3,  0.0,  2.0,  0.0])
    x0_low = np.array([0.0,  0.0,          0.0,  0.0,          -0.5, 0.0, -0.5])
    x0_hi  = np.array([2.0,  200.0,        2.0,  50.0,          0.5, 20.0,  0.5])

    try:
        res = least_squares(residuals, x0, bounds=(x0_low, x0_hi),
                            method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10)
        x = res.x

        # Enforce T2f < T2s (swap if fitter crossed them)
        if x[3] > x[1]:
            x[0], x[2] = x[2], x[0]
            x[1], x[3] = x[3], x[1]

        T2s_fit_ms = x[1]
        T2f_fit_ms = x[3]
        As_fit     = x[0]
        Af_fit     = x[2]
        ATQ_fit    = x[5]
        converged  = res.success or res.cost < 1e-6

        print(f"\n  Biexponential fit result:")
        print(f"  {'Parameter':<18}  {'Value':>12}")
        print(f"  {'─'*32}")
        print(f"  {'T2s*':<18}  {T2s_fit_ms:>10.4f} ms")
        print(f"  {'T2f*':<18}  {T2f_fit_ms:>10.4f} ms")
        print(f"  {'As  (SQ slow amp)':<18}  {As_fit:>10.4f}")
        print(f"  {'Af  (SQ fast amp)':<18}  {Af_fit:>10.4f}")
        print(f"  {'ATQ (TQ amp)':<18}  {ATQ_fit:>10.4f}")
        print(f"  {'Residual norm':<18}  {res.cost:>10.4e}")
        print(f"  {'Converged':<18}  {'yes' if converged else 'check residual'}")
        print(f"{'─'*70}\n")

        return {
            'T2s_star_ms': T2s_fit_ms,
            'T2f_star_ms': T2f_fit_ms,
            'As':          As_fit,
            'Af':          Af_fit,
            'ATQ':         ATQ_fit,
            'residual':    res.cost,
            'converged':   converged,
            'fit_result':  res,
        }

    except Exception as e:
        print(f"T2* biexponential fit failed: {e}. "
                      "Returning optimisation T2 values as fallback.")
        print(f"  FALLBACK: T2s* = {T2s_opt*1e3:.4f} ms  T2f* = {T2f_opt*1e3:.4f} ms")
        return {
            'T2s_star_ms': T2s_opt * 1e3,
            'T2f_star_ms': T2f_opt * 1e3,
            'converged':   False,
        }

# ─────────────────────────────────────────────────────────────────────────────
# RESULT PLOT  — best simulation vs experimental data
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_result(best, exp_SQ, exp_TQ, tevo, sim_cfg, t2star=None):

    import matplotlib.pyplot as plt

    # Re-simulate with best parameters at experimental tevo
    sim_SQ, sim_TQ = _simulate(
        best['T2f'], best['T2s'], best['gamma'], tevo,
        sim_cfg['B0'], sim_cfg['wQbar'], sim_cfg['FreqShift'],
        sim_cfg['wShiftFID'], sim_cfg['nSpins'],
        sim_cfg['NumPhaseCycles'], sim_cfg['alphas'],
        sim_cfg['flip90'], sim_cfg['tmix'],
        sim_cfg['deadtimeFID'], sim_cfg['dwelltimeFID'],
        sim_cfg['dataPoints'], sim_cfg['nPtsevo'], sim_cfg['nPtsmix'],
        sim_cfg['pdistname'], sim_cfg['idxTQ'], sim_cfg['idxSQ'],
        rng_seed=sim_cfg['master_seed'],
    )

    # Acquisition time axis in ms
    acqT_ms = (sim_cfg['deadtimeFID'] +
               sim_cfg['dwelltimeFID'] * np.arange(sim_cfg['dataPoints'])) * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Best fit  —  T2f = {best['T2f']*1e3:.3f} ms  |  "
        f"T2s = {best['T2s']*1e3:.3f} ms  |  "
        f"gamma = {best['gamma']:.1f} rad/s\n"
        f"tevo = {tevo*1e3:.1f} ms  |  Cost = {best['fval']:.4e}",
        fontsize=11,
    )

    for ax, exp, sim, label, color in zip(
        axes,
        [exp_SQ, exp_TQ],
        [sim_SQ, sim_TQ],
        ['SQ', 'TQ'],
        ['tab:blue', 'tab:orange'],
    ):
        ax.plot(acqT_ms, exp, color=color, alpha=0.5, linewidth=1.2,
                label=f'{label} experimental')
        ax.plot(acqT_ms, sim, color=color, linewidth=2.0, linestyle='--',
                label=f'{label} simulation (best fit)')

        # Overlay T2* biexponential fit curves if available
        if t2star is not None and t2star.get('fit_result') is not None:
            x = t2star['fit_result'].x
            # funSQ / funTQ evaluated on the acqT_ms grid
            if label == 'SQ':
                fit_curve = x[0]*np.exp(-acqT_ms/x[1]) + x[2]*np.exp(-acqT_ms/x[3]) + x[4]
                ax.plot(acqT_ms, fit_curve, 'k:', linewidth=1.5,
                        label=f'T2* biexp fit  (T2s*={t2star["T2s_star_ms"]:.2f} ms,'
                              f' T2f*={t2star["T2f_star_ms"]:.2f} ms)')
            else:
                fit_curve = x[5]*(np.exp(-acqT_ms/x[1]) - np.exp(-acqT_ms/x[3])) + x[6]
                ax.plot(acqT_ms, fit_curve, 'k:', linewidth=1.5,
                        label=f'T2* biexp fit')

        ax.set_ylabel(f'{label} (normalised)', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(acqT_ms[0], acqT_ms[-1])

    axes[-1].set_xlabel('Acquisition time (ms)', fontsize=10)
    plt.tight_layout()

    fname = f'best_fit_tevo{tevo*1e3:.0f}ms.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Plot saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ------------------------------------------------------------------
    # Build a reference PC object to extract shared sequence settings
    # (mirrors your dictionary builder setup)
    # ------------------------------------------------------------------
    w0   = getValues.getw0(B0)
    T1s  = 1.175 * T2s_est
    T1f  = 0.825 * T2s_est
    Jen_ref, tauC_ref, wQ_ref, _ = getValues.getJenModel(T1f, T1s, T2f_est, T2s_est, w0)

    PC = PhaseCyclesVecDic(B0, tauC_ref, wQ_ref, 0.0, Jen_ref, 0.0, 0.0, 0.0)
    PC.dwelltimeFID    = 100e-6
    PC.dataPoints      = 2048
    PC.NumPhaseCycles  = 16
    PC.nSpins          = 1000
    PC.tmix            = 0.15e-3
    PC.wShiftdist      = getValues.getWshiftDist('Cauchy', 0, 45)
    PC.flip90          = 90.0

    nAlpha   = len(PC.alphas)
    idxTQ    = 1 * PC.NumPhaseCycles
    idxSQ    = 3 * PC.NumPhaseCycles
    pdistname = (PC.wShiftdist.dist.name
                 if PC.wShiftdist is not None else 'norm')

    sim_cfg = dict(
        B0             = B0,
        wQbar          = PC.wQbar,
        FreqShift      = PC.FreqShift,
        wShiftFID      = PC.wShiftFID,
        nSpins         = PC.nSpins,
        NumPhaseCycles = PC.NumPhaseCycles,
        alphas         = PC.alphas,
        flip90         = PC.flip90,
        tmix           = PC.tmix,
        deadtimeFID    = PC.deadtimeFID,
        dwelltimeFID   = PC.dwelltimeFID,
        dataPoints     = PC.dataPoints,
        nPtsevo        = PC.nPtsevo,
        nPtsmix        = PC.nPtsmix,
        pdistname      = pdistname,
        idxTQ          = idxTQ,
        idxSQ          = idxSQ,
        gamma_fixed    = GAMMA_FIXED,
        master_seed    = RANDOM_SEED,
    )

    # ------------------------------------------------------------------
    # Experimental data
    # ------------------------------------------------------------------
    if _SYNTHETIC_TEST:
        print("Generating synthetic experimental data from the estimate …")
        tevo = 10e-3
        exp_SQ, exp_TQ = _simulate(
            T2f_est, T2s_est, gamma_est, tevo,
            **{k: sim_cfg[k] for k in sim_cfg if k not in ('master_seed', 'gamma_fixed')},
            rng_seed=0,
        )
        # Add a tiny amount of noise to make it realistic
        rng    = np.random.default_rng(1)
        noise  = 0.005
        exp_SQ = exp_SQ + noise * rng.standard_normal(exp_SQ.shape)
        exp_TQ = exp_TQ + noise * rng.standard_normal(exp_TQ.shape)
        print(f"  Synthetic data generated  (nT={len(exp_SQ)}).")
    else:
        # ── Load real data here ──────────────────────────────────
        exp_SQ, exp_TQ, tevo = getSQTQ_fixAcqsDim(studyPath)
        #raise NotImplementedError("Set _SYNTHETIC_TEST=False and load your data above.")

    # ------------------------------------------------------------------
    # Dictionary pre-match → centre of the starting points
    # ------------------------------------------------------------------
    if USE_DICT_START:
        T2f_start, T2s_start, gamma_start, _ = _dictionary_start(exp_SQ, exp_TQ, tevo)
        start_src = "dictionary"
    else:
        T2f_start, T2s_start, gamma_start = T2f_est, T2s_est, GAMMA_FIXED
        start_src = "estimate"
    sim_cfg['gamma_fixed']    = gamma_start
    sim_cfg['optimize_gamma'] = OPTIMIZE_GAMMA

    # ------------------------------------------------------------------
    # Starting points
    # ------------------------------------------------------------------
    x0_list = _make_starting_points(
        T2f_start, T2s_start,
        SCATTER_FRAC, N_RUNS, RANDOM_SEED,
        gamma_est=gamma_start if OPTIMIZE_GAMMA else None,
        gamma_scatter_frac=GAMMA_SCATTER_FRAC,
    )

    gamma_mode = "free" if OPTIMIZE_GAMMA else "fixed"
    print(f"\n{'─'*70}")
    print(f"Launching {N_RUNS} parallel Nelder-Mead runs  (tevo = {tevo*1e3:.1f} ms)")
    print(f"Start ({start_src}) : T2f={T2f_start*1e3:.2f} ms  T2s={T2s_start*1e3:.2f} ms  "
          f"gamma={gamma_start:.1f} rad/s ({gamma_mode})")
    print(f"Scatter  : ±{SCATTER_FRAC*100:.0f} % T2"
          + (f", ±{GAMMA_SCATTER_FRAC*100:.0f} % gamma" if OPTIMIZE_GAMMA else "")
          + "  |  constraint: T2s ≥ 2·T2f")
    print(f"{'─'*70}\n")
    print("  Starting points (T2f ms / T2s ms / gamma rad/s):")
    for i, x0 in enumerate(x0_list):
        g = x0[2] / GAMMA_SCALE if OPTIMIZE_GAMMA else gamma_start
        print(f"  Run {i:2d}: {x0[0] * 1e3:.3f}  {x0[1] * 1e3:.3f}  {g:.2f}")

    t_total = time.time()

    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0)(
        delayed(_run_optimisation)(
            run_id, x0, exp_SQ, exp_TQ, tevo, sim_cfg
        )
        for run_id, x0 in enumerate(x0_list)
    )

    wall_time = time.time() - t_total

    # ------------------------------------------------------------------
    # Pick the best result
    # ------------------------------------------------------------------
    best = min(results, key=lambda r: r['fval'])

    print(f"\n{'─'*70}")
    print(f"All {N_RUNS} runs finished in {wall_time:.0f} s  (wall time)")
    print(f"{'─'*70}")
    print(f"\nBest result  (run {best['run_id']}):")
    print(f"  T2f  = {best['T2f']*1e3:.4f} ms")
    print(f"  T2s  = {best['T2s']*1e3:.4f} ms")
    print(f"  gamma = {best['gamma']:.2f} rad/s")
    print(f"  Cost = {best['fval']:.6e}")
    print(f"  Evaluations : {best['neval']}")

    # Summary table
    print(f"{'Run':>4}  {'T2f (ms)':>10}  {'T2s (ms)':>10}  {'gamma':>8}  {'Cost':>12}  {'OK':>4}")
    print(f"{'─' * 65}")
    for r in sorted(results, key=lambda x: x['fval']):
        ok = '✓' if r['success'] else '!'
        print(f"{r['run_id']:>4}  {r['T2f'] * 1e3:>10.4f}  {r['T2s'] * 1e3:>10.4f}  "
              f"{r['gamma']:>8.2f}  {r['fval']:>12.6e}  {ok:>4}")

    t2star = fit_T2star(best, sim_cfg, tevo_fit=1e-3)

    plot_best_result(best, exp_SQ, exp_TQ, tevo, sim_cfg, t2star=t2star)

    return best


if __name__ == '__main__':
    best = main()