"""
merge_dictionaries.py
Combines two HDF5 dictionary files into one.

The two files must have been generated with the same:
  - acquisition parameters (dataPoints, dwelltimeFID, deadtimeFID)
  - tevo_vec
  - NumPhaseCycles, alphaStep
They can differ in their params (T2f, T2s, FWHM) grid — that is the point.

params layout: [T2f (s), T2s (s), FWHM (Hz)]  — input T2 values are
               read directly from params[:,0] and params[:,1].
"""

import numpy as np
import h5py
import os
from datetime import datetime


# Keys that travel with each dictionary entry (axis-0 indexed)
_ENTRY_KEYS = ["params", "SQ", "TQ", "T2ffit", "T2sfit", "Affit", "Asfit"]


def _load(fname: str) -> dict:
    print(f"Loading {fname} ...")
    with h5py.File(fname, "r") as f:
        d = {k: f[k][()] for k in f.keys()}
        d["_attrs"] = dict(f.attrs)
    return d


def _check_compatibility(a: dict, b: dict) -> None:
    assert np.array_equal(a["tevo"], b["tevo"]), \
        "tevo vectors differ — cannot merge."
    assert (a["acqTime"].shape == b["acqTime"].shape and
            np.max(np.abs(a["acqTime"].ravel() - b["acqTime"].ravel())) < 1e-12), \
        "acqTime vectors differ — files were generated with different acq parameters."
    assert a["SQ"].shape[1] == b["SQ"].shape[1], \
        "nEvo dimension mismatch — different number of evolution times."
    assert a["SQ"].shape[2] == b["SQ"].shape[2], \
        "nT dimension mismatch — different dataPoints or dwelltimeFID."


def _remove_duplicates(a: dict, b: dict) -> tuple[dict, int]:
    """Remove rows from b whose params already exist in a."""
    tol = 1e-9
    diffs = np.max(
        np.abs(b["params"][:, None, :] - a["params"][None, :, :]),
        axis=2,
    )                                      # (nB, nA)
    is_dup = np.any(diffs < tol, axis=1)  # (nB,)
    n_dup  = int(is_dup.sum())

    if n_dup:
        print(f"Removing {n_dup} duplicate entries from file B.")
        keep = ~is_dup
        for key in _ENTRY_KEYS:
            if key in b:
                b[key] = b[key][keep]
    else:
        print("No duplicate entries found.")

    return b, n_dup


def _sort_merged(data: dict) -> dict:
    """Sort all per-entry arrays by FWHM → T2s → T2f."""
    p = data["params"]
    idx = np.lexsort((p[:, 0], p[:, 1], p[:, 2]))  # FWHM outermost
    for key in _ENTRY_KEYS:
        if key in data:
            data[key] = data[key][idx]
    return data


def merge_dictionaries(
    fname_a:   str = "dictionary_1.h5",
    fname_b:   str = "dictionary_2.h5",
    fname_out: str = "dictionary_merged.h5",
) -> None:
    a = _load(fname_a)
    b = _load(fname_b)

    nA = a["params"].shape[0]
    nB = b["params"].shape[0]

    _check_compatibility(a, b)
    print(f"Compatibility check passed.  A: {nA} entries  |  B: {nB} entries")

    b, n_dup = _remove_duplicates(a, b)

    # ── Concatenate along entry axis (axis 0) ─────────────────────────────
    merged = {}
    for key in _ENTRY_KEYS:
        if key in a and key in b:
            merged[key] = np.concatenate([a[key], b[key]], axis=0)
        elif key in a:
            merged[key] = a[key]
        elif key in b:
            merged[key] = b[key]

    merged = _sort_merged(merged)

    nEntries_m = merged["SQ"].shape[0]
    print(f"Merged: {nEntries_m} entries total")

    # ── Write output file ─────────────────────────────────────────────────
    if os.path.exists(fname_out):
        os.remove(fname_out)

    with h5py.File(fname_out, "w") as f:
        # Shared / non-entry datasets
        f.create_dataset("tevo",    data=a["tevo"])
        f.create_dataset("acqTime", data=a["acqTime"])
        if "fittevoidx" in a:
            f.create_dataset("fittevoidx", data=a["fittevoidx"])

        # Per-entry datasets
        f.create_dataset("params",  data=merged["params"])
        f.create_dataset("SQ",      data=merged["SQ"])
        f.create_dataset("TQ",      data=merged["TQ"])
        f.create_dataset("T2ffit",  data=merged["T2ffit"].reshape(-1, 1))
        f.create_dataset("T2sfit",  data=merged["T2sfit"].reshape(-1, 1))
        f.create_dataset("Affit",   data=merged["Affit"].reshape(-1, 1))
        f.create_dataset("Asfit",   data=merged["Asfit"].reshape(-1, 1))

        # Metadata attributes from A + merge provenance
        for name, val in a["_attrs"].items():
            f.attrs[name] = val
        f.attrs["merged_from"]  = f"{fname_a} + {fname_b}"
        f.attrs["merged_date"]  = datetime.now().isoformat()
        f.attrs["n_entries_a"]  = nA
        f.attrs["n_entries_b"]  = nB - n_dup
        f.attrs["n_duplicates"] = n_dup

    # ── Summary ───────────────────────────────────────────────────────────
    p = merged["params"]
    fsize_mb  = os.path.getsize(fname_out) / 1e6
    fwhm_vals = np.unique(p[:, 2])

    print(f"\nSaved  {fname_out}  ({fsize_mb:.1f} MB)")
    print(f"  Total entries : {nEntries_m}  "
          f"(A: {nA}  +  B: {nB - n_dup}  −  {n_dup} duplicates)")
    print(f"  FWHM values   : {'  '.join(f'{v:.0f}' for v in fwhm_vals)} Hz")
    print(f"  T2f range     : {p[:,0].min()*1e3:.2f} – {p[:,0].max()*1e3:.2f} ms  (from params[:,0])")
    print(f"  T2s range     : {p[:,1].min()*1e3:.1f} – {p[:,1].max()*1e3:.1f} ms  (from params[:,1])")


if __name__ == "__main__":
    merge_dictionaries(
        fname_a="newDictPy_5090FWHM.h5",
        fname_b="newDictPy_6080FWHM.h5",
        fname_out="pyDict_5090FWHM.h5",
    )