"""
Loop wrapper around the single-spin-ensemble NMR simulation + fit.

Purpose
-------
Each call to PC.SpinEcho() / PC.TQTPPIWith180() draws a fresh set of
per-spin frequency offsets from PC.wShiftdist (Lorentzian/Cauchy here).
That randomness propagates through the fit, so a single run only gives
you one realization. This script repeats the whole
"build ensemble -> simulate -> fit" pipeline N_RUNS times and collects:

  - the fitted parameters (popt) from every run  -> mean / std per parameter
  - the raw signal curves from every run          -> mean curve +/- std band

Notes / assumptions (please sanity-check against your class internals)
------------------------------------------------------------------
* PC is rebuilt from scratch every iteration to *guarantee* independent
  random sampling, even if PhaseCyclesVecDic caches anything internally.
  If you know the sequence methods already resample on every call, you
  can move build_PC() outside the loop for a small speed-up.
* Per-run plotting and per-run save_TQTPPI() calls are disabled (100
  popup plots / files would not be useful) -- aggregated results are
  plotted/saved at the end instead.
* I don't know the exact parameter order returned by fit_biexponentialSQ
  / fit_TQTTPI_FID, so parameter labels below are generic placeholders
  (param0, param1, ...). Replace PARAM_NAMES_TQTPPI / PARAM_NAMES_SE
  with the real names for nicer plots/printouts.
"""

from Classes.getValues import getValues
import numpy as np
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic
import time
from commonImports import *
from diagnoseCheck import diagnose_sequence_with180
from Classes.getFunction import getFunction
from DataLoadingStoring.storeSimulRes import save_TQTPPI


# ============================================================
# Fixed (non-random) physical parameters -- computed once
# ============================================================
B0 = 9.4
w0 = getValues.getw0(B0)

# 4% Agarose, FWHM: 47  (currently active preset)
T1s = 42.9525e-3
T1f = 31.1021e-3   # T1f_8020
T2s = 37.8604e-3
T2f = 6.1239e-3

FWHM = 50

Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)


# ============================================================
# Run configuration
# ============================================================
spinEcho    = False   # True -> SpinEcho sequence, False -> TQTPPIWith180
storeResult = False   # per-run save_TQTPPI() is disabled regardless (see run_once)
N_RUNS      = 10

# Placeholder parameter names -- fill in with the real meaning of each
# entry of popt for your fit functions.
PARAM_NAMES_SE     = ["T2s", "T2f", "A_s", "A_f", "offset"]              # fit_biexponentialSQ
PARAM_NAMES_TQTPPI = [f"param{i}" for i in range(9)]                     # fit_TQTTPI_FID (9 params in p0)


# ============================================================
# Build a fresh PC instance (guarantees independent random sampling)
# ============================================================
def build_PC():
    PC = PhaseCyclesVecDic(B0, tauC, wQ, 0.0, Jen, 0.0, 0.0, 0.0)
    PC.dwelltimeFID   = 100e-6
    PC.dataPoints     = 2048
    PC.NumPhaseCycles = 65
    PC.nSpins         = 1000
    PC.tmix           = 0.14e-3
    PC.wShiftdist     = getValues.getWshiftDist('cauchy', 0, FWHM / 2)
    PC.flip90         = 90.0
    PC.TR             = 204.7e-3
    PC.tevo           = 30e-3
    PC.tevo0          = 0.0e-3
    PC.tevoStep       = 0.2e-3
    return PC


# ============================================================
# One full simulation + fit
# ============================================================
def run_once(run_idx):
    PC = build_PC()

    if spinEcho:
        t0 = time.time()
        acqTimeVec, echoFIDs, echoAmps, echoTimeVec = PC.SpinEcho()
        t1 = time.time()

        echoAmps = echoAmps / np.amax(echoAmps)
        popt, pcov = getFunction.fit_biexponentialSQ(
            echoTimeVec * 2, echoAmps, [T2s, T2f, 0.5, 0.5, 0.01])

        xdata, ydata = echoTimeVec * 2, echoAmps

    else:
        t0 = time.time()
        acqTimeVec, mqSpectrum, mqFID, FIDs, fAngles90, tevos = PC.TQTPPIWith180()
        t1 = time.time()

        mqFID = mqFID / np.amax(mqFID)

        # If you want every run's raw data stored, uncomment (will create
        # N_RUNS files):
        # if storeResult:
        #     save_TQTPPI(f"TQTPPI_AG4_simPy_run{run_idx:03d}", PC,
        #                 acqTimeVec, mqSpectrum, mqFID, FIDs, tevos)

        p0 = [T2s, T2f, 0.4, 0.6, 0.2,
              1 / (PC.tevoStep * len(PC.alphas)), np.pi, -np.pi, 0]
        popt, pcov = getFunction.fit_TQTTPI_FID(tevos, mqFID, p0)

        xdata, ydata = tevos, mqFID

    runtime = t1 - t0
    print(f"Run {run_idx + 1:3d}/{N_RUNS}  |  {runtime:6.2f} s  |  "
          f"popt = {np.array2string(np.asarray(popt), precision=4)}")

    return {"popt": np.asarray(popt), "pcov": pcov,
            "xdata": xdata, "ydata": ydata, "runtime": runtime}


# ============================================================
# Main loop
# ============================================================
all_results = []
failed_runs = []

t_start = time.time()
for i in range(N_RUNS):
    try:
        all_results.append(run_once(i))
    except Exception as e:
        print(f"Run {i + 1} FAILED: {e}")
        failed_runs.append(i)
t_end = time.time()

print(f"\nCompleted {len(all_results)}/{N_RUNS} runs "
      f"({len(failed_runs)} failed) in {t_end - t_start:.1f} s "
      f"({(t_end - t_start) / max(len(all_results), 1):.2f} s/run avg)")

if not all_results:
    raise RuntimeError("All runs failed -- nothing to analyze.")

param_names = PARAM_NAMES_SE if spinEcho else PARAM_NAMES_TQTPPI

# ---- Fit-parameter statistics ----
popt_array = np.array([r["popt"] for r in all_results])   # shape (n_runs, n_params)
mean_popt  = np.mean(popt_array, axis=0)
std_popt   = np.std(popt_array, axis=0, ddof=1)

print(f"\nFit parameter statistics over {len(all_results)} runs:")
for idx in range(popt_array.shape[1]):
    name = param_names[idx] if idx < len(param_names) else f"param{idx}"
    m, s = mean_popt[idx], std_popt[idx]
    rel = f"{100 * s / abs(m):.2f}%" if m != 0 else "n/a"
    print(f"  {name:10s}: mean = {m: .6g}   std = {s: .3g}   (rel. {rel})")

# ---- Save aggregated results for later analysis ----
np.savez("simulation_fluctuation_results.npz",
         popt_array=popt_array,
         mean_popt=mean_popt,
         std_popt=std_popt,
         param_names=np.array(param_names, dtype=object),
         failed_runs=np.array(failed_runs))
print("\nSaved aggregated popt array + stats to 'simulation_fluctuation_results.npz'")

# ============================================================
# Visualization
# ============================================================

# --- Mean signal +/- std band across all runs ---
xdata = all_results[0]["xdata"]
ydata_stack = np.array([r["ydata"] for r in all_results])
mean_curve = ydata_stack.mean(axis=0)
std_curve  = ydata_stack.std(axis=0, ddof=1)

plt.figure()
plt.plot(xdata, mean_curve, label="mean signal", color="C0")
plt.fill_between(xdata, mean_curve - std_curve, mean_curve + std_curve,
                  alpha=0.3, color="C0", label="+/- 1 std")
plt.xlabel("time" if not spinEcho else "echo time")
plt.ylabel("normalized signal")
plt.title(f"Mean +/- STD over {len(all_results)} runs")
plt.legend(loc="best")
plt.show()

# --- Histograms of each fit parameter ---
n_params = popt_array.shape[1]
ncols = 3
nrows = int(np.ceil(n_params / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = np.atleast_1d(axes).flatten()

for idx in range(n_params):
    name = param_names[idx] if idx < len(param_names) else f"param{idx}"
    axes[idx].hist(popt_array[:, idx], bins=15, color="C0", alpha=0.75)
    axes[idx].axvline(mean_popt[idx], color="k", linestyle="--", linewidth=1)
    axes[idx].set_title(f"{name}\nmean={mean_popt[idx]:.4g}, std={std_popt[idx]:.3g}")

for ax in axes[n_params:]:
    ax.axis("off")

plt.tight_layout()
plt.show()