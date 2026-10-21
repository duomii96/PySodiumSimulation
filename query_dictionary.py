# query_dictionary.py
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import h5py

from dict_params import gamma_column


@dataclass
class QueryResult:
    """Best-match result from dictionary lookup."""
    T2f_ms:         float
    T2s_ms:         float
    Af:             float
    As:             float
    gamma:          float               # Cauchy scale of wShiftdist [rad/s]
    tevo_used_ms:   float
    n_candidates:   int
    dist:           np.ndarray          # [nEntries]  full distance vector (np.inf for filtered-out)
    top10_T2f:      np.ndarray          # [K] T2f* of top-K candidates (ms)
    top10_T2s:      np.ndarray          # [K] T2s* of top-K candidates (ms)
    top10_params:   np.ndarray          # [K x 3] raw params of top-K candidates
    SQ_match:       np.ndarray          # [nT] best-match SQ curve
    TQ_match:       np.ndarray          # [nT] best-match TQ curve
    acqTime:        np.ndarray          # [nT] acquisition time vector (s)
    T2f_input_ms:   float = 0.0
    T2s_input_ms:   float = 0.0
    T1f_input_ms:   float = 0.0
    T1s_input_ms:   float = 0.0


def query_dictionary(
    fname: str,
    SQ_meas: np.ndarray,
    TQ_meas: np.ndarray,
    tevo_ms: float,
    matching_norm: str,
    gamma_meas: float,
    gamma_tol: float = 5.0,

) -> QueryResult:
    """
    Match measured SQ/TQ curves against an HDF5 dictionary.

    Parameters
    ----------
    fname        : path to HDF5 dictionary file
    SQ_meas      : measured SQ curve, shape (nT,)
    TQ_meas      : measured TQ curve, shape (nT,)
    tevo_ms      : evolution time of the measurement (ms)
    gamma_meas   : Cauchy scale of wShiftdist [rad/s] (FWHM = 2*gamma)
    gamma_tol    : gamma pre-filter tolerance [rad/s], default 5

    Returns
    -------
    QueryResult dataclass with best-match parameters and curves.
    """
    # ── Load HDF5 ─────────────────────────────────────────────────────────
    try:
        with h5py.File(fname, "r") as f:
            tevo_vec = f["/tevo"][:]
            acqTime  = f["/acqTime"][:]
            params   = f["/params"][:]      # [nEntries x 3]: [T2f, T2s, gamma (old: FWHM)]
            T2f_fit  = f["/T2ffit"][:]
            T2s_fit  = f["/T2sfit"][:]
            Af_fit   = f["/Affit"][:]
            As_fit   = f["/Asfit"][:]
            T2f_in   = params[:,0]
            T2s_in   = params[:,1]
            T1f_in   = T2s_in * 0.825
            T1s_in   = T2s_in * 1.175
            SQ_full  = f["/SQ"][:]         # [nEntries x nEvo x nT]
            TQ_full  = f["/TQ"][:]
            gamma_dict = gamma_column(params, f.attrs)


    except:
        # when dict was created with Matlab, shift rows and column
        with h5py.File(fname, "r") as f:
            tevo_vec = f["/tevo"][:]
            acqTime = f["/acqTime"][:]
            params = f["/params"][:].transpose()
            T2f_fit = f["/T2f_fit"][:]
            T2s_fit = f["/T2s_fit"][:]
            Af_fit = f["/Af_fit"][:]
            As_fit = f["/As_fit"][:]
            T2f_in = params[:,0]
            T2s_in = params[:,1]
            T1f_in = T2s_in * 0.825
            T1s_in = T2s_in * 1.175
            gamma_dict = gamma_column(params, f.attrs)
            SQ_full = f["/SQ"][:]
            TQ_full = f["/TQ"][:]



    # ── Normalise orientations ────────────────────────────────────────────
    tevo_vec = tevo_vec.ravel()
    acqTime  = acqTime.ravel()
    SQ_meas  = SQ_meas.ravel()
    TQ_meas  = TQ_meas.ravel()

    nEntries = params.shape[0]
    nT_dict  = acqTime.size
    nT_meas  = SQ_meas.size

    # ── Validate / trim time dimension ────────────────────────────────────
    if nT_meas != nT_dict:
        acqTime = acqTime[:nT_meas]

    # ── Find closest tevo ─────────────────────────────────────────────────
    tevo_idx = int(np.argmin(np.abs(tevo_vec * 1e3 - tevo_ms)))
    print(
        f"Matching at tevo={tevo_ms:.0f}ms  "
        f"(closest dict tevo={tevo_vec[tevo_idx]*1e3:.0f}ms, index={tevo_idx})"
    )

    # ── gamma pre-filtering ───────────────────────────────────────────────

    gamma_mask = np.abs(gamma_dict - gamma_meas) <= gamma_tol
    n_cand     = int(gamma_mask.sum())

    print(
        f"gamma filter: {gamma_meas:.1f} +/- {gamma_tol:.1f} rad/s "
        f"-> {n_cand}/{nEntries} entries pass"
    )

    if n_cand == 0:
        raise ValueError(
            f"No dictionary entries within {gamma_tol:.1f} rad/s of "
            f"gamma={gamma_meas:.1f} rad/s.\n"
            "Widen gamma_tol or extend the gamma range of the dictionary."
        )

    # ── Slice SQ/TQ at selected tevo ─────────────────────────────────────
    # SQ_full shape: [nEntries, nEvo, nT]
    if np.size(SQ_full,0) > np.size(SQ_full,-1):
        # When dict created in matlab
        SQ_full = np.swapaxes(SQ_full, 0, -1)
        TQ_full = np.swapaxes(TQ_full, 0, -1)
        D_SQ_all = SQ_full[:, tevo_idx, :nT_meas]   # [nEntries x nT]
        D_TQ_all = TQ_full[:, tevo_idx, :nT_meas]
    else:

        D_SQ_all = SQ_full[:, tevo_idx, :nT_meas]  # [nEntries x nT]
        D_TQ_all = TQ_full[:, tevo_idx, :nT_meas]


    D_SQ = D_SQ_all[gamma_mask]   # [n_cand x nT]
    D_TQ = D_TQ_all[gamma_mask]

    # ── Normalise ─────────────────────────────────────────────────────────
    q_SQ = SQ_meas / SQ_meas.max()
    q_TQ = TQ_meas / TQ_meas.max()

    # Avoid division by zero for degenerate rows
    D_SQ_n = D_SQ / np.maximum(D_SQ.max(axis=1, keepdims=True), 1e-30)
    D_TQ_n = D_TQ / np.maximum(D_TQ.max(axis=1, keepdims=True), 1e-30)


    if "2" in matching_norm:
        # ── L2 distance ───────────────────────────────────────────────────────
        dist_cand = (
            np.sum((D_SQ_n - q_SQ) ** 2, axis=1) +
            np.sum((D_TQ_n - q_TQ) ** 2, axis=1)
        )   # [n_cand]
    else:
        # L1
        dist_cand = (
                np.sum(np.abs(D_SQ_n - q_SQ) , axis=1) +
                np.sum(np.abs(D_TQ_n - q_TQ) , axis=1)
        )

    # Map back to full distance vector (np.inf for filtered-out entries)
    dist_full = np.full(nEntries, np.inf)
    dist_full[gamma_mask] = dist_cand

    sorted_cand   = np.argsort(dist_cand)
    cand_indices  = np.where(gamma_mask)[0]
    K             = min(10, n_cand)
    top_full_idx  = cand_indices[sorted_cand[:K]]
    best          = top_full_idx[0]

    # ── Build result ──────────────────────────────────────────────────────
    result = QueryResult(
        T2f_ms        = float(np.squeeze(T2f_fit)[best].item()) * 1e3,
        T2s_ms        = float(np.squeeze(T2s_fit)[best].item()) * 1e3,
        Af            = float(np.squeeze(Af_fit)[best].item()),
        As            = float(np.squeeze(As_fit)[best].item()),
        gamma         = float(gamma_dict[best].item()),
        tevo_used_ms  = float(tevo_vec[tevo_idx].item()) * 1e3,
        n_candidates  = n_cand,
        dist          = dist_full,
        top10_T2f     = np.squeeze(T2f_fit)[top_full_idx] * 1e3,
        top10_T2s     = np.squeeze(T2s_fit)[top_full_idx] * 1e3,
        top10_params  = params[top_full_idx],
        SQ_match      = D_SQ_all[best],
        TQ_match      = D_TQ_all[best],
        acqTime       = acqTime,
        T2f_input_ms  = float(np.squeeze(T2f_in)[best].item()) * 1e3,
        T2s_input_ms  = float(np.squeeze(T2s_in)[best].item()) * 1e3,
        T1f_input_ms  = float(np.squeeze(T1f_in)[best].item()) * 1e3,
        T1s_input_ms  = float(np.squeeze(T1s_in)[best].item()) * 1e3,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    print(
        f"Best match:  T2f*={result.T2f_ms:.2f}ms  "
        f"T2s*={result.T2s_ms:.1f}ms  gamma={result.gamma:.1f}rad/s"
    )
    print(f"Input T2:    T2f ={result.T2f_input_ms:.2f}ms  T2s ={result.T2s_input_ms:.1f}ms")
    print(
        f"Input T1:    T1f ={result.T1f_input_ms:.2f}ms  "
        f"T1s ={result.T1s_input_ms:.1f}ms  (estimated)"
    )
    print(f"Top-{K} T2f* range: {result.top10_T2f.min():.2f} - {result.top10_T2f.max():.2f}ms")
    print(f"Top-{K} T2s* range: {result.top10_T2s.min():.1f} - {result.top10_T2s.max():.1f}ms")

    return result