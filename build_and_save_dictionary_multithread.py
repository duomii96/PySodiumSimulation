"""
build_and_save_dictionary.py
Generates the full coarse dictionary in parallel across all available CPU cores,
pre-fits T2f/T2s from low-evolution curves, and saves everything to HDF5.

Dictionary structure
--------------------
For each entry (T2f_param, T2s_param, gamma):
  gamma = scale of the Cauchy wShift distribution [rad/s] (HWHM, FWHM = 2*gamma)
          -> passed directly to getWshiftDist; FID dephasing ~ exp(-gamma*t)
  - SQ and TQ curves at all tevo values (used for high-evo matching)
  - T2ffit, T2sfit from low-evo fit (returned after matching)
  - Affit, Asfit amplitudes from fit

HDF5 layout
-----------
  params      N × 3          [T2fs, T2ss, gamma_rad_s] input grid
  tevo        1 × nEvo       evolution times [s]
  acqTime     1 × nT         acquisition time vector [s]
  SQ          N × nEvo × nT  SQ curves
  TQ          N × nEvo × nT  TQ curves
  T2ffit      N × 1          T2f from biexponential fit [s]
  T2sfit      N × 1          T2s from biexponential fit [s]
  Affit       N × 1          fast amplitude
  Asfit       N × 1          slow amplitude
  fittevoidx  1 × 1          tevo index used for fitting

Root attributes: B0, NumPhaseCycles, nSpins, tmixms,
                 alphaStepdeg, dwelltimeus, dataPoints, created
"""

import numpy as np
import h5py
import os
import time
import warnings
from datetime import datetime
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from commonImports import *

import numba
from Classes.getValues import getValues
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic


# ─────────────────────────────────────────────────────────────────────────────
# Worker — runs ONE (param-entry, tevo) simulation in its own process
# ─────────────────────────────────────────────────────────────────────────────
def _run_single(
    k, iEntry, iEvo,
    T2f, T2s, gamma, tevo,
    # basePC config — passed as plain scalars to avoid pickling the whole object
    B0, wQbar, FreqShift, wShiftFID,
    nSpins, NumPhaseCycles, alphas,
    flip90, tmix, deadtimeFID, dwelltimeFID, dataPoints,
    nPtsevo, nPtsmix,
    pdistname,
    idxTQ, idxSQ,
    n_numba_threads,
    rng_seed,
):
    """
    Simulate one (T2f, T2s, gamma, tevo) combination.
    rng_seed fixes the wShiftdist Monte Carlo draws so every job is reproducible.
    Returns (k, iEntry, iEvo, SQ_row, TQ_row).
    """
    np.random.seed(rng_seed)

    # Limit Numba threads per worker so we don\'t over-subscribe the CPU
    numba.set_num_threads(n_numba_threads)

    w0k     = getValues.getw0(B0)
    T1sest  = 1.175 * T2s
    T1fest  = 0.825 * T2s
    Jenk, tauCk, wQk, wShiftRMSk = getValues.getJenModel(T1fest, T1sest, T2f, T2s, w0k)

    wShiftdistk = getValues.getWshiftDist(pdistname, 0.0, gamma)
    wShiftRMSk  = 0.0   # consistent with Lorentzian linewidth contribution

    sim = PhaseCyclesVecDic(B0, tauCk, wQk, wQbar, Jenk,
                            FreqShift, wShiftRMSk, wShiftFID)
    sim.nSpins         = nSpins
    sim.NumPhaseCycles = NumPhaseCycles
    sim.alphas         = alphas
    sim.flip90         = flip90
    sim.tmix           = tmix
    sim.wShiftdist     = wShiftdistk
    sim.dataPoints     = dataPoints
    sim.deadtimeFID    = deadtimeFID
    sim.dwelltimeFID   = dwelltimeFID
    sim.tevo           = tevo
    sim.nPtsevo        = nPtsevo
    sim.nPtsmix        = nPtsmix

    _, _, fidk, _ = sim.TQTPPIfixedwo180VJ()

    # fidk shape: (nFIDs, nT, 2) → use the combined (already subtracted) channel
    # FFT along phase-cycle dimension (axis 0) to select coherence orders
    FTphase = np.fft.fftshift(np.fft.fft(fidk[:, :], axis=0), axes=0)
    SQ_row  = np.real(FTphase[idxSQ, :])
    TQ_row  = np.real(FTphase[idxTQ, :])

    return k, iEntry, iEvo, SQ_row, TQ_row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_and_save_dictionary(n_jobs: int = -1):
    """
    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. -1 = use all available cores.
        Set to 1 to disable parallelism (useful for debugging).
    """
    # ------------------------------------------------------------------
    # Reference physics (used to initialise PC object defaults only)
    # ------------------------------------------------------------------
    B0 = 9.4
    w0 = getValues.getw0(B0)

    T1s_ref = 39.129e-3
    T1f_ref = 27.6504e-3
    T2s_ref = 32.3578e-3
    T2f_ref = 3.9326e-3
    Jen_ref, tauC_ref, wQ_ref, wShiftRMS_ref = getValues.getJenModel(
        T1f_ref, T1s_ref, T2f_ref, T2s_ref, w0)

    # Base PC — only used to carry shared settings; physics is overridden per entry
    PC = PhaseCyclesVecDic(B0, tauC_ref, wQ_ref, 0.0, Jen_ref, 0.0, 0.0, 0.0)
    PC.dwelltimeFID   = 100e-6
    PC.dataPoints     = 2048
    PC.NumPhaseCycles = 16
    PC.nSpins         = 1000
    PC.tmix           = 0.15e-3
    PC.wShiftdist     = getValues.getWshiftDist('Cauchy', 0, 45)
    PC.flip90         = 90.0

    # ------------------------------------------------------------------
    # Parameter grid
    # ------------------------------------------------------------------
    T2fvec  = np.logspace(np.log10(4e-3),   np.log10(15e-3),  8)
    T2svec  = np.logspace(np.log10(20e-3), np.log10(55e-3), 12)
    gammavec = np.linspace(20, 60, 5, endpoint=True)   # Cauchy scale [rad/s] (old "FWHM" 60–90 / 2)

    params   = PhaseCyclesVecDic.make_param_grid(T2fvec, T2svec, gammavec)
    nEntries = params.shape[0]

    # ------------------------------------------------------------------
    # Evolution times
    # ------------------------------------------------------------------
    tevovec    = np.array([1, 2, 5, 10, 15, 20, 25, 30, 35]) * 1e-3
    fittevoidx = 0
    nEvo       = len(tevovec)
    total      = nEntries * nEvo

    print(f"Full grid before filter: {len(T2fvec) * len(T2svec) * len(gammavec)} × {nEvo} = "
          f"{len(T2fvec) * len(T2svec) * len(gammavec) * nEvo} total")
    print(f"After T2s≥2·T2f filter: {nEntries} × {nEvo} = {total} simulations")
    #print(f"Grid: {nEntries} entries × {nEvo} tevo = {total} simulations")
    print(f"Fitting will use tevo={tevovec[fittevoidx]*1e3:.0f} ms (index {fittevoidx})")

    # ------------------------------------------------------------------
    # Shared worker settings
    # ------------------------------------------------------------------
    acqTimeVec = PC.deadtimeFID + PC.dwelltimeFID * np.arange(PC.dataPoints)
    nT         = len(acqTimeVec)

    nFIDsdict  = PC.NumPhaseCycles * len(PC.alphas)
    idxTQ      = 1 * PC.NumPhaseCycles
    idxSQ      = 3 * PC.NumPhaseCycles

    pdistname  = (PC.wShiftdist.dist.name
                  if PC.wShiftdist is not None else 'norm')

    # How many Numba threads each worker gets.
    # With n_jobs workers × n_numba_threads each ≤ physical cores.
    import os as _os
    n_phys = _os.cpu_count() or 1
    n_workers = min(16, n_phys) if n_jobs == -1 else min(16, max(1, n_jobs))
    n_numba_threads = 1
    rng_seed_base   = 42   # job k uses seed rng_seed_base + k
    print(f"Using {n_workers} workers × {n_numba_threads} Numba thread(s) each "
          f"(total threads ≤ {n_workers * n_numba_threads})")


    # ------------------------------------------------------------------
    # Build job list  (iEntry × iEvo)
    # ------------------------------------------------------------------
    import itertools
    jobs = [
        (k, iEntry, iEvo,
         params[iEntry, 0], params[iEntry, 1], params[iEntry, 2], tevovec[iEvo])
        for k, (iEntry, iEvo) in enumerate(itertools.product(range(nEntries), range(nEvo)))
    ]

    # ------------------------------------------------------------------
    # Run in parallel
    # ------------------------------------------------------------------
    SQflat = np.zeros((total, nT))
    TQflat = np.zeros((total, nT))

    t0 = time.time()
    results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_run_single)(
            k, iEntry, iEvo, T2f, T2s, gamma, tevo,
            B0, PC.wQbar, PC.FreqShift, PC.wShiftFID,
            PC.nSpins, PC.NumPhaseCycles, PC.alphas,
            PC.flip90, PC.tmix, PC.deadtimeFID, PC.dwelltimeFID, PC.dataPoints,
            PC.nPtsevo, PC.nPtsmix,
            pdistname,
            idxTQ, idxSQ,
            n_numba_threads,
            rng_seed_base + k,
        )
        for k, iEntry, iEvo, T2f, T2s, gamma, tevo in jobs
    )

    for k, iEntry, iEvo, SQ_row, TQ_row in results:
        SQflat[k, :] = SQ_row
        TQflat[k, :] = TQ_row

    print(f"All simulations done in {time.time()-t0:.1f} s")

    # ------------------------------------------------------------------
    # Pre-fit T2f/T2s from low-evo SQ+TQ simultaneously
    # ------------------------------------------------------------------
    print(f"Fitting T2f/T2s from tevo={tevovec[fittevoidx]*1e3:.0f} ms ...")

    acqT_ms = acqTimeVec * 1e3   # ms — better numerical scaling for the fitter
    SQ3d = SQflat.reshape(nEntries, nEvo, nT)
    TQ3d = TQflat.reshape(nEntries, nEvo, nT)

    T2ffit = np.zeros(nEntries)
    T2sfit = np.zeros(nEntries)
    Affit  = np.zeros(nEntries)
    Asfit  = np.zeros(nEntries)

    x0     = np.array([0.4, 33.0, 0.6, 10.0, 0.0, 2.0, 0.0])
    x0_low = np.array([0.0,  0.0, 0.0,  0.0, -0.1, 0.0, -0.1])
    x0_hi  = np.array([2.0, 70.0, 2.0, 70.0,  0.1, 20.0, 0.1])

    def funSQ(x, t): return x[0]*np.exp(-t/x[1]) + x[2]*np.exp(-t/x[3]) + x[4]
    def funTQ(x, t): return x[5]*(np.exp(-t/x[1]) - np.exp(-t/x[3])) + x[6]

    for iEntry in range(nEntries):
        sq  = SQ3d[iEntry, fittevoidx, :]
        tq  = TQ3d[iEntry, fittevoidx, :]
        sqn = sq / max(np.abs(sq))
        tqn = tq / max(np.abs(tq))

        def fun(x):
            return np.concatenate([funSQ(x, acqT_ms) - sqn,
                                   funTQ(x, acqT_ms) - tqn])
        try:
            res = least_squares(fun, x0, bounds=(x0_low, x0_hi))
            x   = res.x
            # Enforce T2f < T2s
            if x[3] > x[1]:
                x[0], x[2] = x[2], x[0]
                x[1], x[3] = x[3], x[1]
            Asfit[iEntry]  = x[0]
            T2sfit[iEntry] = x[1] * 1e-3
            Affit[iEntry]  = x[2]
            T2ffit[iEntry] = x[3] * 1e-3
        except Exception:
            T2ffit[iEntry] = params[iEntry, 0]
            T2sfit[iEntry] = params[iEntry, 1]
            Affit[iEntry]  = 0.6
            Asfit[iEntry]  = 0.4
            warnings.warn(f"Fit failed for entry {iEntry}, using input params")

        if iEntry % 20 == 0:
            print(f"  fitted {iEntry+1}/{nEntries}")

    # ------------------------------------------------------------------
    # Save to HDF5
    # ------------------------------------------------------------------
    fname = os.path.join(os.getcwd(), 'dict_general_gamma2060.h5')
    if os.path.exists(fname):
        os.remove(fname)

    with h5py.File(fname, 'w') as f:
        f.create_dataset('params',     data=params)
        f.create_dataset('tevo',       data=tevovec.reshape(1, -1))
        f.create_dataset('acqTime',    data=acqTimeVec.reshape(1, -1))
        f.create_dataset('SQ',         data=SQ3d)
        f.create_dataset('TQ',         data=TQ3d)
        f.create_dataset('T2ffit',     data=T2ffit.reshape(-1, 1))
        f.create_dataset('T2sfit',     data=T2sfit.reshape(-1, 1))
        f.create_dataset('Affit',      data=Affit.reshape(-1, 1))
        f.create_dataset('Asfit',      data=Asfit.reshape(-1, 1))
        f.create_dataset('fittevoidx', data=np.array([[fittevoidx]]))

        f.attrs['B0']             = B0
        f.attrs['NumPhaseCycles'] = PC.NumPhaseCycles
        f.attrs['nSpins']         = PC.nSpins
        f.attrs['tmixms']         = PC.tmix * 1e3
        f.attrs['alphaStepdeg']   = PC.alphaStep
        f.attrs['dwelltimeus']    = PC.dwelltimeFID * 1e6
        f.attrs['dataPoints']     = PC.dataPoints
        f.attrs['created']        = datetime.now().isoformat()
        f.attrs['params_columns'] = 'T2f_s, T2s_s, gamma_rad_s (Cauchy scale, FWHM = 2*gamma)'
        f.attrs['rng_seed_base']  = rng_seed_base

    size_mb = os.path.getsize(fname) / 1e6
    print(f"\nSaved: {fname}  ({size_mb:.1f} MB)")
    with h5py.File(fname, 'r') as f:
        print(f"Datasets: {list(f.keys())}")

    # ------------------------------------------------------------------
    # Validation plot
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(params[:, 0]*1e3, T2ffit*1e3, s=20)
        axes[0].plot([0, 20], [0, 20], 'r--')
        axes[0].set_xlabel('T2f input (ms)'); axes[0].set_ylabel('T2f fitted (ms)')
        axes[0].set_title('T2f input vs fitted'); axes[0].axis('equal'); axes[0].grid(True)

        axes[1].scatter(params[:, 1]*1e3, T2sfit*1e3, s=20)
        axes[1].plot([0, 120], [0, 120], 'r--')
        axes[1].set_xlabel('T2s input (ms)'); axes[1].set_ylabel('T2s fitted (ms)')
        axes[1].set_title('T2s input vs fitted'); axes[1].axis('equal'); axes[1].grid(True)

        fig.suptitle(f'Dictionary {nEntries} entries, fit tevo={tevovec[fittevoidx]*1e3:.0f} ms')
        plt.tight_layout()
        plt.savefig('dictionary_fit_validation.png', dpi=120)
        plt.close()
        print("Validation plot saved → dictionary_fit_validation.png")
    except ImportError:
        pass


if __name__ == '__main__':
    build_and_save_dictionary(n_jobs=16)


"""with open('build_and_save_dictionary.py', 'w') as f:
    f.write(new_build)
print("Written.")"""