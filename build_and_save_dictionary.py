"""
build_and_save_dictionary.py
Generates the full coarse dictionary, pre-fits T2f/T2s from low-evolution curves,
and saves everything to HDF5 for efficient partial loading.

Dictionary structure
--------------------
For each entry (T2f_param, T2s_param, FWHM):
  - SQ and TQ curves at all tevo values (used for high-evo matching)
  - T2ffit, T2sfit from low-evo fit (returned after matching)
  - Affit, Asfit amplitudes from fit

HDF5 layout
-----------
  params      N × 3     [T2fs, T2ss, FWHM_Hz] input grid
  tevo        1 × nEvo  evolution times [s]
  acqTime     1 × nT    acquisition time vector [s]
  SQ          N × nEvo × nT   SQ curves (fftshift applied)
  TQ          N × nEvo × nT   TQ curves (fftshift applied)
  T2ffit      N × 1     T2f from biexponential fit of low-evo SQ [s]
  T2sfit      N × 1     T2s from biexponential fit of low-evo SQ [s]
  Affit       N × 1     fast amplitude
  Asfit       N × 1     slow amplitude
  fittevoidx  scalar    which tevo index was used for fitting

Attributes (stored on root group):
  B0, NumPhaseCycles, nSpins, tmixms, alphaStepdeg,
  dwelltimeus, dataPoints, created
"""

import numpy as np
import h5py
import os
from datetime import datetime
from scipy.optimize import least_squares

from Classes.getValues import getValues
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic


def build_and_save_dictionary():
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    B0 = 9.4
    w0 = getValues.getw0(B0)

    T1s = 39.129e-3
    T1f = 27.6504e-3
    T2s = 32.3578e-3
    T2f = 3.9326e-3

    Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)

    PC = PhaseCyclesVecDic(B0, tauC, wQ, 0.0, Jen, 0.0, 0.0, 0.0)
    PC.dwelltimeFID   = 100e-6
    PC.dataPoints     = 2048
    PC.NumPhaseCycles = 16
    PC.nSpins         = 1000
    PC.tmix           = 0.15e-3
    PC.wShiftdist     = getValues.getWshiftDist('tLocationScale', 0, 45)
    PC.flip90         = 90.0

    # ------------------------------------------------------------------
    # Parameter grid
    # ------------------------------------------------------------------
    T2fvec  = np.logspace(np.log10(1e-3),  np.log10(10e-3),  10)   # 2–10 ms
    T2svec  = np.logspace(np.log10(10.5e-3), np.log10(65e-3),  10)   # 11–80 ms
    FWHMvec = np.linspace(60, 80, 3, endpoint=True)                                # 30–70 Hz

    params = PhaseCyclesVecDic.make_param_grid(T2fvec, T2svec, FWHMvec)
    nEntries = params.shape[0]

    # ------------------------------------------------------------------
    # Evolution times
    # ------------------------------------------------------------------
    tevovec    = np.array([1, 2, 5, 10, 15, 20, 25, 30]) * 1e-3   # seconds
    #tevovec = np.array([30]) * 1e-3
    fittevoidx = 0   # use tevo[0] = 1 ms for T*2 fitting
    nEvo       = len(tevovec)

    print(f"Grid: {nEntries} entries × {nEvo} tevo = {nEntries*nEvo} simulations")
    print(f"Fitting will use tevo={tevovec[fittevoidx]*1e3:.0f} ms (index {fittevoidx})")

    # ------------------------------------------------------------------
    # Run dictionary
    # ------------------------------------------------------------------
    dict_data = PhaseCyclesVecDic.run_dictionary(PC, params, tevovec)

    # ------------------------------------------------------------------
    # Pre-fit T2f/T2s from low-evo SQ+TQ curves simultaneously
    # ------------------------------------------------------------------
    print(f"T2f/T2s from tevo={tevovec[fittevoidx]*1e3:.0f} ms SQ+TQ simultaneous...")

    acqT = dict_data['acqTime'] * 1e3  # convert to ms for better numerical scaling
    nT = len(acqT)

    T2ffit = np.zeros(nEntries)
    T2sfit = np.zeros(nEntries)
    Affit  = np.zeros(nEntries)
    Asfit  = np.zeros(nEntries)

    x0     = np.array([0.4, 33.0, 0.6, 10.0, 0.0, 2.0, 0.0])
    x0_low = np.array([0.0,  0.0, 0.0,  0.0, -0.1, 0.0, -0.1])
    x0_hi  = np.array([2.0, 70.0, 2.0, 70.0,  0.1, 20.0, 0.1])

    def funSQ(x, t): return x[0]*np.exp(-t/x[1]) + x[2]*np.exp(-t/x[3]) + x[4]
    def funTQ(x, t): return x[5]*(np.exp(-t/x[1]) - np.exp(-t/x[3])) + x[6]

    def fun(x):
        return np.concatenate([funSQ(x, acqT) - sqn, funTQ(x, acqT) - tqn])

    for iEntry in range(nEntries):
        sq = dict_data['SQ'][iEntry, fittevoidx, :]
        tq = dict_data['TQ'][iEntry, fittevoidx, :]
        sqn = sq / max(np.abs(sq))
        tqn = tq / max(np.abs(tq))

        try:
            res = least_squares(fun, x0, bounds=(x0_low, x0_hi))
            x = res.x

            # Enforce T2f < T2s (swap if needed)
            if x[3] > x[1]:  # x[3]=T2f, x[1]=T2s → swap fast/slow
                x[0], x[2] = x[2], x[0]
                x[1], x[3] = x[3], x[1]

            Asfit[iEntry]  = x[0]
            T2sfit[iEntry] = x[1] * 1e-3   # back to seconds
            Affit[iEntry]  = x[2]
            T2ffit[iEntry] = x[3] * 1e-3
        except Exception:
            T2ffit[iEntry] = params[iEntry, 0]
            T2sfit[iEntry] = params[iEntry, 1]
            Affit[iEntry]  = 0.6
            Asfit[iEntry]  = 0.4
            import warnings
            warnings.warn(f"Fit failed for entry {iEntry}, using input params")

        if iEntry % 20 == 0:
            print(f"  fitted {iEntry+1}/{nEntries}")

    # ------------------------------------------------------------------
    # Save to HDF5
    # ------------------------------------------------------------------
    fname = os.path.join(os.getcwd(), 'dictionaryadjFWHM3070.h5')
    if os.path.exists(fname):
        os.remove(fname)

    with h5py.File(fname, 'w') as f:
        f.create_dataset('params',     data=params)
        f.create_dataset('tevo',       data=dict_data['tevo'].reshape(1, -1))
        f.create_dataset('acqTime',    data=dict_data['acqTime'].reshape(1, -1))
        f.create_dataset('SQ',         data=dict_data['SQ'])
        f.create_dataset('TQ',         data=dict_data['TQ'])
        f.create_dataset('T2ffit',     data=T2ffit.reshape(-1, 1))
        f.create_dataset('T2sfit',     data=T2sfit.reshape(-1, 1))
        f.create_dataset('Affit',      data=Affit.reshape(-1, 1))
        f.create_dataset('Asfit',      data=Asfit.reshape(-1, 1))
        f.create_dataset('fittevoidx', data=np.array([[fittevoidx]]))

        # Metadata as attributes on root group
        f.attrs['B0']           = B0
        f.attrs['NumPhaseCycles'] = PC.NumPhaseCycles
        f.attrs['nSpins']       = PC.nSpins
        f.attrs['tmixms']       = PC.tmix * 1e3
        f.attrs['alphaStepdeg'] = PC.alphaStep
        f.attrs['dwelltimeus']  = PC.dwelltimeFID * 1e6
        f.attrs['dataPoints']   = PC.dataPoints
        f.attrs['created']      = datetime.now().isoformat()

    # Quick info
    size_mb = os.path.getsize(fname) / 1e6
    print(f"Saved: {fname}")
    with h5py.File(fname, 'r') as f:
        datasets = list(f.keys())
    print(f"Datasets: {', '.join(datasets)}")
    print(f"File size: {size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Quick validation plot
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(params[:, 0]*1e3, T2ffit*1e3, s=20)
        axes[0].plot([0, 10], [0, 10], 'r--')
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
        print("Validation plot saved.")
    except ImportError:
        pass


if __name__ == '__main__':
    build_and_save_dictionary()
