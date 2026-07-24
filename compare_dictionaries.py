"""
compare_dictionaries.py
Compare SQ and TQ curves between two HDF5 dictionaries for matching
(T2f, T2s, FWHM) parameter entries.

Usage
-----
Set fname_a, fname_b, target_FWHM, target_tevo_ms and optionally tol_ms
at the bottom of this file, then run:
    python compare_dictionaries.py
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load a dictionary — handles both Python and MATLAB HDF5 layouts
# ─────────────────────────────────────────────────────────────────────────────
def load_dict(fname: str) -> dict:
    """
    Load params, SQ, TQ, acqTime, tevo from an HDF5 dictionary file.

    MATLAB h5write stores arrays in Fortran (column-major) order, so h5py
    reads them transposed relative to Python convention:
      params  : (3, nEntries)   → transpose to (nEntries, 3)
      SQ / TQ : (nT, nEvo, nEntries) → transpose to (nEntries, nEvo, nT)

    Python-written files already have the correct layout:
      params  : (nEntries, 3)
      SQ / TQ : (nEntries, nEvo, nT)
    """
    with h5py.File(fname, "r") as f:
        keys = list(f.keys())

        params  = f["/params"][()]
        SQ_raw  = f["/SQ"][()]
        TQ_raw  = f["/TQ"][()]
        acqTime = f["/acqTime"][()].ravel()
        tevo    = f["/tevo"][()].ravel()

        T2ffit = (f["/T2ffit"][()] if "T2ffit" in keys else
                  f["/T2f_fit"][()] if "T2f_fit" in keys else None)
        T2sfit = (f["/T2sfit"][()] if "T2sfit" in keys else
                  f["/T2s_fit"][()] if "T2s_fit" in keys else None)

    # ── Detect and fix MATLAB transposition ──────────────────────────────
    # params: MATLAB writes (3, nEntries); Python writes (nEntries, 3)
    if params.ndim == 2 and params.shape[0] == 3 and params.shape[1] != 3:
        params = params.T                          # (3, N) → (N, 3)
        matlab_layout = True
    else:
        matlab_layout = False

    nEntries = params.shape[0]

    # SQ/TQ: MATLAB layout is (nT, nEvo, nEntries) → transpose to (nEntries, nEvo, nT)
    #        Python layout is (nEntries, nEvo, nT)  → no change needed
    if SQ_raw.ndim == 3:
        if matlab_layout or SQ_raw.shape[2] == nEntries:
            SQ = SQ_raw.transpose(2, 1, 0)   # (nT, nEvo, nEntries) → (nEntries, nEvo, nT)
            TQ = TQ_raw.transpose(2, 1, 0)
        else:
            SQ = SQ_raw   # already (nEntries, nEvo, nT)
            TQ = TQ_raw
    else:
        raise ValueError(f"Unexpected SQ shape {SQ_raw.shape} in {fname}")

    # Sanity check
    assert SQ.shape == (nEntries, len(tevo), len(acqTime)),         (f"Shape mismatch after layout correction in {fname}: "
         f"SQ={SQ.shape}, nEntries={nEntries}, nEvo={len(tevo)}, nT={len(acqTime)}")

    print(f"  {fname}: {nEntries} entries | "
          f"nEvo={len(tevo)} | nT={len(acqTime)} | "
          f"layout={'MATLAB' if matlab_layout else 'Python'}")

    return dict(params=params, SQ=SQ, TQ=TQ,
                acqTime=acqTime, tevo=tevo,
                T2ffit=T2ffit, T2sfit=T2sfit,
                fname=fname)


# ─────────────────────────────────────────────────────────────────────────────
# Core comparison function
# ─────────────────────────────────────────────────────────────────────────────
def compare_dictionaries(
    fname_a: str,
    fname_b: str,
    target_FWHM: float,           # Hz  — FWHM slice to compare
    target_tevo_ms,               # ms  — single value or list, e.g. 10 or [5, 10, 20]
    tol_ms: float = 1.5,          # ms  — tolerance for T2f / T2s matching
    max_pairs: int = 12,          # cap on individual pair figures
    label_a: str = None,
    label_b: str = None,
):
    """
    Find entries with the same FWHM and similar (T2f, T2s) in both dictionaries,
    then plot side-by-side SQ and TQ curves for each matched pair at the
    requested evolution time(s).

    Parameters
    ----------
    fname_a, fname_b   : paths to the two HDF5 files
    target_FWHM        : FWHM value (Hz) to filter on
    target_tevo_ms     : evolution time(s) in ms to plot, e.g. 10 or [5, 10, 20]
    tol_ms             : matching tolerance for T2f and T2s in milliseconds
    max_pairs          : max number of matched pairs to show individually
    label_a, label_b   : legend / title labels (defaults to filenames)
    """
    label_a = label_a or fname_a
    label_b = label_b or fname_b
    tol     = tol_ms * 1e-3

    target_tevo_list = (
        [target_tevo_ms] if np.isscalar(target_tevo_ms) else list(target_tevo_ms)
    )

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading dictionaries...")
    da = load_dict(fname_a)
    db = load_dict(fname_b)

    acqTime_ms = da["acqTime"] * 1e3   # ms for plotting
    tevo_a_ms  = da["tevo"]    * 1e3
    tevo_b_ms  = db["tevo"]    * 1e3

    # ── Resolve tevo indices in each file ─────────────────────────────────
    def find_tevo_idx(tevo_ms_vec, target_ms, fname):
        idx = int(np.argmin(np.abs(tevo_ms_vec - target_ms)))
        actual = tevo_ms_vec[idx]
        if abs(actual - target_ms) > 3:
            raise ValueError(
                f"tevo={target_ms} ms not found in {fname}. "
                f"Available: {tevo_ms_vec}")
        return idx, actual

    evo_idxs_a, evo_idxs_b, evo_labels = [], [], []
    for t_ms in target_tevo_list:
        ia, ta = find_tevo_idx(tevo_a_ms, t_ms, fname_a)
        ib, tb = find_tevo_idx(tevo_b_ms, t_ms, fname_b)
        evo_idxs_a.append(ia)
        evo_idxs_b.append(ib)
        evo_labels.append(f"{ta:.0f}/{tb:.0f} ms")
        print(f"  tevo={t_ms} ms → A index {ia} ({ta:.0f} ms), "
              f"B index {ib} ({tb:.0f} ms)")

    nEvoPlot = len(evo_idxs_a)

    # ── Filter by FWHM ────────────────────────────────────────────────────
    def fwhm_mask(d, fwhm_val):
        fwhm_col = d["params"][:, 2]
        closest  = fwhm_col[np.argmin(np.abs(fwhm_col - fwhm_val))]
        if abs(closest - fwhm_val) > 5:
            raise ValueError(
                f"FWHM={fwhm_val} Hz not found in {d['fname']}. "
                f"Available: {np.unique(fwhm_col)}")
        return fwhm_col == closest, closest

    mask_a, fwhm_used_a = fwhm_mask(da, target_FWHM)
    mask_b, fwhm_used_b = fwhm_mask(db, target_FWHM)

    pa  = da["params"][mask_a]
    pb  = db["params"][mask_b]
    SQa = da["SQ"][mask_a]
    TQa = da["TQ"][mask_a]
    SQb = db["SQ"][mask_b]
    TQb = db["TQ"][mask_b]

    print(f"FWHM filter: A={fwhm_used_a:.0f} Hz ({mask_a.sum()} entries), "
          f"B={fwhm_used_b:.0f} Hz ({mask_b.sum()} entries)")

    # ── Match entries by T2f and T2s ──────────────────────────────────────
    # Vectorised: (nA, 1) vs (1, nB) for each param
    dT2f = np.abs(pa[:, 0:1] - pb[np.newaxis, :, 0])   # (nA, nB)
    dT2s = np.abs(pa[:, 1:2] - pb[np.newaxis, :, 1])   # (nA, nB)
    within_tol = (dT2f <= tol) & (dT2s <= tol)           # (nA, nB)

    # For each A, pick the closest B match
    matches = []
    for iA in range(len(pa)):
        cands = np.where(within_tol[iA])[0]
        if len(cands) == 0:
            continue
        best_iB = cands[np.argmin(dT2f[iA, cands] + dT2s[iA, cands])]
        matches.append((iA, best_iB))

    n_found = len(matches)
    print(f"Matched pairs within ±{tol_ms} ms: {n_found}")
    if n_found == 0:
        print("  No matches found. Try increasing tol_ms.")
        return

    matches = matches[:max_pairs]
    print(f"Plotting {len(matches)} pairs (max_pairs={max_pairs})\n")

    colors_evo = plt.cm.viridis(np.linspace(0, 0.85, nEvoPlot))

    # ── Per-pair figures ───────────────────────────────────────────────────
    for pair_n, (iA, iB) in enumerate(matches):
        T2f_a_ms = pa[iA, 0] * 1e3
        T2s_a_ms = pa[iA, 1] * 1e3
        T2f_b_ms = pb[iB, 0] * 1e3
        T2s_b_ms = pb[iB, 1] * 1e3

        title = (f"Pair {pair_n+1}  |  FWHM={target_FWHM:.0f} Hz  |  "
                 f"T2f: {T2f_a_ms:.2f} / {T2f_b_ms:.2f} ms  "
                 f"T2s: {T2s_a_ms:.1f} / {T2s_b_ms:.1f} ms")
        print(f"  {title}")

        fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                                 num=f"Pair {pair_n+1}")
        fig.suptitle(title, fontsize=10)
        ax_sq_a, ax_sq_b = axes[0, 0], axes[0, 1]
        ax_tq_a, ax_tq_b = axes[1, 0], axes[1, 1]

        for col, (ei_a, ei_b, elbl) in enumerate(
                zip(evo_idxs_a, evo_idxs_b, evo_labels)):
            c = colors_evo[col]
            kw = dict(color=c, lw=1.4, label=f"tevo={elbl}")

            def norm(v): return v / (np.max(np.abs(v)) or 1)

            ax_sq_a.plot(acqTime_ms, norm(SQa[iA, ei_a, :]), **kw)
            ax_sq_b.plot(acqTime_ms, norm(SQb[iB, ei_b, :]), **kw)
            ax_tq_a.plot(acqTime_ms, norm(TQa[iA, ei_a, :]), **kw)
            ax_tq_b.plot(acqTime_ms, norm(TQb[iB, ei_b, :]), **kw)

        for ax, ttl in zip(
            [ax_sq_a, ax_sq_b, ax_tq_a, ax_tq_b],
            [f"SQ — {label_a}", f"SQ — {label_b}",
             f"TQ — {label_a}", f"TQ — {label_b}"]
        ):
            ax.set_xlabel("Acquisition time (ms)")
            ax.set_ylabel("Normalised amplitude")
            ax.set_title(ttl, fontsize=9)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.4)

        fig.tight_layout()

    # ── Overlay figure: all pairs, first tevo only ────────────────────────
    if len(matches) > 1:
        ei_a0, ei_b0 = evo_idxs_a[0], evo_idxs_b[0]
        colors_pair  = plt.cm.tab20(np.linspace(0, 1, len(matches)))

        fig_ov, axes_ov = plt.subplots(2, 2, figsize=(13, 8),
                                       num="Overlay — all matched pairs")
        fig_ov.suptitle(
            f"Overlay: {len(matches)} pairs  |  FWHM={target_FWHM:.0f} Hz  |  "
            f"tevo={evo_labels[0]}", fontsize=10)

        for col_p, (iA, iB) in enumerate(matches):
            c   = colors_pair[col_p]
            lbl = f"T2f={pa[iA,0]*1e3:.1f} T2s={pa[iA,1]*1e3:.1f} ms"

            def norm(v): return v / (np.max(np.abs(v)) or 1)

            axes_ov[0, 0].plot(acqTime_ms, norm(SQa[iA, ei_a0, :]), color=c, lw=1.1, label=lbl)
            axes_ov[0, 1].plot(acqTime_ms, norm(SQb[iB, ei_b0, :]), color=c, lw=1.1, label=lbl)
            axes_ov[1, 0].plot(acqTime_ms, norm(TQa[iA, ei_a0, :]), color=c, lw=1.1, label=lbl)
            axes_ov[1, 1].plot(acqTime_ms, norm(TQb[iB, ei_b0, :]), color=c, lw=1.1, label=lbl)

        for ax, ttl in zip(
            axes_ov.ravel(),
            [f"SQ — {label_a}", f"SQ — {label_b}",
             f"TQ — {label_a}", f"TQ — {label_b}"]
        ):
            ax.set_xlabel("Acquisition time (ms)")
            ax.set_ylabel("Normalised amplitude")
            ax.set_title(ttl, fontsize=9)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.4)

        fig_ov.tight_layout()

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — edit these settings
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compare_dictionaries(
        fname_a        = "pyMatlab_CompareDict.h5",
        fname_b        = "dictionary_adjFWHM30100.h5",
        target_FWHM    = 60,            # Hz
        target_tevo_ms = [10, 20, 25],   # ms — single value or list
        tol_ms         = 1.5,           # matching tolerance in ms
        max_pairs      = 12,
        label_a        = "Dict A (Python)",
        label_b        = "Dict B (MATLAB)",
    )
