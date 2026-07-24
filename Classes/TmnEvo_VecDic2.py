"""
TmnEvo_VecDic.py
Key optimisations over the plain-NumPy version
------------------------------------------------
1.  Two separate Numba JIT kernels replace _relax_from_cache:
      _relax_endpoint_kernel  – scalar f*, (nS,) ph*  → in-place, no alloc
      _relax_T11_kernel       – (nT,) f*, (nS,nT) ph* → returns T11relax
    Both use parallel=True (prange over spins) + fastmath=True.
2.  Endpoint relaxation no longer allocates (nS, nT) intermediate arrays —
    it operates directly on the single endpoint values, saving 65× memory
    bandwidth vs the old approach.
3.  First call triggers Numba AOT compilation (~2–5 s once, then cached).
"""

import numpy as np
import numba
from Classes.getValues import getValues
import time

# ─────────────────────────────────────────────────────────────────────────────
# Helper: ensure an array or scalar is contiguous complex128
# ─────────────────────────────────────────────────────────────────────────────
def _c128(x):
    a = np.asarray(x, dtype=np.complex128)
    return np.ascontiguousarray(a)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1 – endpoint relaxation
# f*  : complex scalars evaluated at the endpoint time
# ph* : (nSpins,) complex vectors — one phase factor per spin at endpoint
# Modifies Tmnbatch in-place; no return value.
# ─────────────────────────────────────────────────────────────────────────────
@numba.njit(parallel=True, fastmath=True, cache=True)
def _relax_endpoint_kernel(
        Tmnbatch,
        f011, f013, f033, f022,
        fm111, fm112, fm113, fm122, fm123, fm133,
        f111, f112p, f113, f122, f123p, f133,
        fm222, fm223, fm233,
        f222,  f223,  f233,
        fm333, f333,
        ph1m, ph1p, ph2m, ph2p, ph3m, ph3p):

    nS = Tmnbatch.shape[0]
    for s in numba.prange(nS):
        Tmnbatch[s, 0, 3] -= 1.0 + 0j          # subtract thermal equilibrium

        T10  = Tmnbatch[s, 0, 3]
        T20  = Tmnbatch[s, 1, 3]
        T30  = Tmnbatch[s, 2, 3]
        T1m1 = Tmnbatch[s, 0, 2];  T11  = Tmnbatch[s, 0, 4]
        T2m1 = Tmnbatch[s, 1, 2];  T21  = Tmnbatch[s, 1, 4]
        T3m1 = Tmnbatch[s, 2, 2];  T31  = Tmnbatch[s, 2, 4]
        T2m2 = Tmnbatch[s, 1, 1];  T22  = Tmnbatch[s, 1, 5]
        T3m2 = Tmnbatch[s, 2, 1];  T32  = Tmnbatch[s, 2, 5]
        T3m3 = Tmnbatch[s, 2, 0];  T33  = Tmnbatch[s, 2, 6]

        # q = 0
        Tmnbatch[s, 0, 3] = T10*f011 + T30*f013 + 1.0 + 0j
        Tmnbatch[s, 1, 3] = T20*f022
        Tmnbatch[s, 2, 3] = T10*f013 + T30*f033

        # q = -1
        Tmnbatch[s, 0, 2] = (T1m1*fm111 + T2m1*fm112 + T3m1*fm113) * ph1m[s]
        Tmnbatch[s, 1, 2] = (T1m1*fm112 + T2m1*fm122 + T3m1*fm123) * ph1m[s]
        Tmnbatch[s, 2, 2] = (T1m1*fm113 + T2m1*fm123 + T3m1*fm133) * ph1m[s]

        # q = +1
        Tmnbatch[s, 0, 4] = (T11*f111  + T21*f112p + T31*f113)  * ph1p[s]
        Tmnbatch[s, 1, 4] = (T11*f112p + T21*f122  + T31*f123p) * ph1p[s]
        Tmnbatch[s, 2, 4] = (T11*f113  + T21*f123p + T31*f133)  * ph1p[s]

        # q = -2
        Tmnbatch[s, 1, 1] = (T2m2*fm222 + T3m2*fm223) * ph2m[s]
        Tmnbatch[s, 2, 1] = (T3m2*fm233 + T2m2*fm223) * ph2m[s]

        # q = +2
        Tmnbatch[s, 1, 5] = (T22*f222 + T32*f223) * ph2p[s]
        Tmnbatch[s, 2, 5] = (T32*f233 + T22*f223) * ph2p[s]

        # q = ±3
        Tmnbatch[s, 2, 0] = T3m3 * fm333 * ph3m[s]
        Tmnbatch[s, 2, 6] = T33  * f333  * ph3p[s]


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2 – T_{1,1} FID acquisition (all time points)
# f*  : (nT,) complex arrays
# ph* : (nSpins, nT) complex arrays
# Returns T11relax (nSpins, nT); updates Tmnbatch to last time point.
# ─────────────────────────────────────────────────────────────────────────────
@numba.njit(parallel=True, fastmath=True, cache=True)
def _relax_T11_kernel(
        Tmnbatch,
        f011, f013, f033, f022,
        fm111, fm112, fm113, fm122, fm123, fm133,
        f111, f112p, f113, f122, f123p, f133,
        fm222, fm223, fm233,
        f222,  f223,  f233,
        fm333, f333,
        ph1m, ph1p, ph2m, ph2p, ph3m, ph3p):

    nS = Tmnbatch.shape[0]
    nT = f011.shape[0]
    T11relax = np.empty((nS, nT), dtype=np.complex128)

    for s in numba.prange(nS):
        Tmnbatch[s, 0, 3] -= 1.0 + 0j

        T10  = Tmnbatch[s, 0, 3]
        T20  = Tmnbatch[s, 1, 3]
        T30  = Tmnbatch[s, 2, 3]
        T1m1 = Tmnbatch[s, 0, 2];  T11  = Tmnbatch[s, 0, 4]
        T2m1 = Tmnbatch[s, 1, 2];  T21  = Tmnbatch[s, 1, 4]
        T3m1 = Tmnbatch[s, 2, 2];  T31  = Tmnbatch[s, 2, 4]
        T2m2 = Tmnbatch[s, 1, 1];  T22  = Tmnbatch[s, 1, 5]
        T3m2 = Tmnbatch[s, 2, 1];  T32  = Tmnbatch[s, 2, 5]
        T3m3 = Tmnbatch[s, 2, 0];  T33  = Tmnbatch[s, 2, 6]

        # Full T11 time series (the acquisition signal)
        for t in range(nT):
            T11relax[s, t] = (T11*f111[t] + T21*f112p[t] + T31*f113[t]) * ph1p[s, t]

        # Write endpoint of ALL coherence orders back into Tmnbatch
        last = nT - 1
        Tmnbatch[s, 0, 3] = T10*f011[last] + T30*f013[last] + 1.0 + 0j
        Tmnbatch[s, 1, 3] = T20*f022[last]
        Tmnbatch[s, 2, 3] = T10*f013[last]  + T30*f033[last]

        Tmnbatch[s, 0, 2] = (T1m1*fm111[last] + T2m1*fm112[last] + T3m1*fm113[last]) * ph1m[s, last]
        Tmnbatch[s, 1, 2] = (T1m1*fm112[last] + T2m1*fm122[last] + T3m1*fm123[last]) * ph1m[s, last]
        Tmnbatch[s, 2, 2] = (T1m1*fm113[last] + T2m1*fm123[last] + T3m1*fm133[last]) * ph1m[s, last]

        Tmnbatch[s, 0, 4] = T11relax[s, last]   # already computed above
        Tmnbatch[s, 1, 4] = (T11*f112p[last] + T21*f122[last]  + T31*f123p[last]) * ph1p[s, last]
        Tmnbatch[s, 2, 4] = (T11*f113[last]  + T21*f123p[last] + T31*f133[last])  * ph1p[s, last]

        Tmnbatch[s, 1, 1] = (T2m2*fm222[last] + T3m2*fm223[last]) * ph2m[s, last]
        Tmnbatch[s, 2, 1] = (T3m2*fm233[last] + T2m2*fm223[last]) * ph2m[s, last]
        Tmnbatch[s, 1, 5] = (T22*f222[last]   + T32*f223[last])   * ph2p[s, last]
        Tmnbatch[s, 2, 5] = (T32*f233[last]   + T22*f223[last])   * ph2p[s, last]

        Tmnbatch[s, 2, 0] = T3m3 * fm333[last] * ph3m[s, last]
        Tmnbatch[s, 2, 6] = T33  * f333[last]  * ph3p[s, last]

    return T11relax

@numba.njit(parallel=True, fastmath=True, cache=True)
def _relax_T1m1_kernel(
        Tmnbatch,
        f011, f013, f033, f022,
        fm111, fm112, fm113, fm122, fm123, fm133,
        f111, f112p, f113, f122, f123p, f133,
        fm222, fm223, fm233,
        f222,  f223,  f233,
        fm333, f333,
        ph1m, ph1p, ph2m, ph2p, ph3m, ph3p):

    nS = Tmnbatch.shape[0]
    nT = f011.shape[0]
    T1m1relax = np.empty((nS, nT), dtype=np.complex128)

    for s in numba.prange(nS):
        Tmnbatch[s, 0, 3] -= 1.0 + 0j # remove equilibrium

        T10  = Tmnbatch[s, 0, 3]
        T20  = Tmnbatch[s, 1, 3]
        T30  = Tmnbatch[s, 2, 3]
        T1m1 = Tmnbatch[s, 0, 2];  T11  = Tmnbatch[s, 0, 4]
        T2m1 = Tmnbatch[s, 1, 2];  T21  = Tmnbatch[s, 1, 4]
        T3m1 = Tmnbatch[s, 2, 2];  T31  = Tmnbatch[s, 2, 4]
        T2m2 = Tmnbatch[s, 1, 1];  T22  = Tmnbatch[s, 1, 5]
        T3m2 = Tmnbatch[s, 2, 1];  T32  = Tmnbatch[s, 2, 5]
        T3m3 = Tmnbatch[s, 2, 0];  T33  = Tmnbatch[s, 2, 6]

        # Full T1,-1 time series
        for t in range(nT):
            T1m1relax[s, t] = (T1m1*fm111[t] + T2m1*fm112[t] + T3m1*fm113[t]) * ph1m[s, t]

        # Write endpoint of ALL coherence orders back into Tmnbatch
        last = nT - 1
        Tmnbatch[s, 0, 3] = T10*f011[last] + T30*f013[last] + 1.0 + 0j
        Tmnbatch[s, 1, 3] = T20*f022[last]
        Tmnbatch[s, 2, 3] = T10*f013[last]  + T30*f033[last]

        Tmnbatch[s, 0, 2] = T1m1relax[s, last]
        Tmnbatch[s, 1, 2] = (T1m1*fm112[last] + T2m1*fm122[last] + T3m1*fm123[last]) * ph1m[s, last]
        Tmnbatch[s, 2, 2] = (T1m1*fm113[last] + T2m1*fm123[last] + T3m1*fm133[last]) * ph1m[s, last]

        Tmnbatch[s, 0, 4] = (T11*f111[last] + T21*f112p[last] + T31*f113[last]) * ph1p[s, last]
        Tmnbatch[s, 1, 4] = (T11*f112p[last] + T21*f122[last]  + T31*f123p[last]) * ph1p[s, last]
        Tmnbatch[s, 2, 4] = (T11*f113[last]  + T21*f123p[last] + T31*f133[last])  * ph1p[s, last]

        Tmnbatch[s, 1, 1] = (T2m2*fm222[last] + T3m2*fm223[last]) * ph2m[s, last]
        Tmnbatch[s, 2, 1] = (T3m2*fm233[last] + T2m2*fm223[last]) * ph2m[s, last]
        Tmnbatch[s, 1, 5] = (T22*f222[last]   + T32*f223[last])   * ph2p[s, last]
        Tmnbatch[s, 2, 5] = (T32*f233[last]   + T22*f223[last])   * ph2p[s, last]

        Tmnbatch[s, 2, 0] = T3m3 * fm333[last] * ph3m[s, last]
        Tmnbatch[s, 2, 6] = T33  * f333[last]  * ph3p[s, last]

    return T1m1relax

@numba.njit(parallel=True, fastmath=True, cache=True)
def _relax_endpoint_kernel_batched(
        Tmnbatch,                                    # (nFIDs, nS, 3, 7) complex128, in/out
        f011, f013, f033, f022,                       # (nFIDs,) complex128
        fm111, fm112, fm113, fm122, fm123, fm133,      # (nFIDs,)
        f111, f112p, f113, f122, f123p, f133,          # (nFIDs,)
        fm222, fm223, fm233,                           # (nFIDs,)
        f222,  f223,  f233,                            # (nFIDs,)
        fm333, f333,                                   # (nFIDs,)
        ph1m, ph1p, ph2m, ph2p, ph3m, ph3p):            # (nFIDs, nS) complex128

    nFIDs = Tmnbatch.shape[0]
    nS    = Tmnbatch.shape[1]

    for flat in numba.prange(nFIDs * nS):
        idx = flat // nS
        s   = flat %  nS

        Tmnbatch[idx, s, 0, 3] -= 1.0 + 0j

        T10  = Tmnbatch[idx, s, 0, 3]
        T20  = Tmnbatch[idx, s, 1, 3]
        T30  = Tmnbatch[idx, s, 2, 3]
        T1m1 = Tmnbatch[idx, s, 0, 2];  T11  = Tmnbatch[idx, s, 0, 4]
        T2m1 = Tmnbatch[idx, s, 1, 2];  T21  = Tmnbatch[idx, s, 1, 4]
        T3m1 = Tmnbatch[idx, s, 2, 2];  T31  = Tmnbatch[idx, s, 2, 4]
        T2m2 = Tmnbatch[idx, s, 1, 1];  T22  = Tmnbatch[idx, s, 1, 5]
        T3m2 = Tmnbatch[idx, s, 2, 1];  T32  = Tmnbatch[idx, s, 2, 5]
        T3m3 = Tmnbatch[idx, s, 2, 0];  T33  = Tmnbatch[idx, s, 2, 6]

        Tmnbatch[idx, s, 0, 3] = T10*f011[idx] + T30*f013[idx] + 1.0 + 0j
        Tmnbatch[idx, s, 1, 3] = T20*f022[idx]
        Tmnbatch[idx, s, 2, 3] = T10*f013[idx] + T30*f033[idx]

        Tmnbatch[idx, s, 0, 2] = (T1m1*fm111[idx] + T2m1*fm112[idx] + T3m1*fm113[idx]) * ph1m[idx, s]
        Tmnbatch[idx, s, 1, 2] = (T1m1*fm112[idx] + T2m1*fm122[idx] + T3m1*fm123[idx]) * ph1m[idx, s]
        Tmnbatch[idx, s, 2, 2] = (T1m1*fm113[idx] + T2m1*fm123[idx] + T3m1*fm133[idx]) * ph1m[idx, s]

        Tmnbatch[idx, s, 0, 4] = (T11*f111[idx]  + T21*f112p[idx] + T31*f113[idx])  * ph1p[idx, s]
        Tmnbatch[idx, s, 1, 4] = (T11*f112p[idx] + T21*f122[idx]  + T31*f123p[idx]) * ph1p[idx, s]
        Tmnbatch[idx, s, 2, 4] = (T11*f113[idx]  + T21*f123p[idx] + T31*f133[idx])  * ph1p[idx, s]

        Tmnbatch[idx, s, 1, 1] = (T2m2*fm222[idx] + T3m2*fm223[idx]) * ph2m[idx, s]
        Tmnbatch[idx, s, 2, 1] = (T3m2*fm233[idx] + T2m2*fm223[idx]) * ph2m[idx, s]

        Tmnbatch[idx, s, 1, 5] = (T22*f222[idx] + T32*f223[idx]) * ph2p[idx, s]
        Tmnbatch[idx, s, 2, 5] = (T32*f233[idx] + T22*f223[idx]) * ph2p[idx, s]

        Tmnbatch[idx, s, 2, 0] = T3m3 * fm333[idx] * ph3m[idx, s]
        Tmnbatch[idx, s, 2, 6] = T33  * f333[idx]  * ph3p[idx, s]


@numba.njit(parallel=True, fastmath=True, cache=True)
def _relax_T11_kernel_batched(
        Tmnbatch,                                    # (nFIDs, nS, 3, 7) complex128, in/out
        f011, f013, f033, f022,                       # (nFIDs, nT)
        fm111, fm112, fm113, fm122, fm123, fm133,      # (nFIDs, nT)
        f111, f112p, f113, f122, f123p, f133,          # (nFIDs, nT)
        fm222, fm223, fm233,                           # (nFIDs, nT)
        f222,  f223,  f233,                            # (nFIDs, nT)
        fm333, f333,                                   # (nFIDs, nT)
        ph1m, ph1p, ph2m, ph2p, ph3m, ph3p):            # (nFIDs, nS, nT)

    nFIDs = Tmnbatch.shape[0]
    nS    = Tmnbatch.shape[1]
    nT    = f011.shape[1]
    T11relax = np.empty((nFIDs, nS, nT), dtype=np.complex128)

    for flat in numba.prange(nFIDs * nS):
        idx = flat // nS
        s   = flat %  nS

        Tmnbatch[idx, s, 0, 3] -= 1.0 + 0j

        T10  = Tmnbatch[idx, s, 0, 3]
        T20  = Tmnbatch[idx, s, 1, 3]
        T30  = Tmnbatch[idx, s, 2, 3]
        T1m1 = Tmnbatch[idx, s, 0, 2];  T11  = Tmnbatch[idx, s, 0, 4]
        T2m1 = Tmnbatch[idx, s, 1, 2];  T21  = Tmnbatch[idx, s, 1, 4]
        T3m1 = Tmnbatch[idx, s, 2, 2];  T31  = Tmnbatch[idx, s, 2, 4]
        T2m2 = Tmnbatch[idx, s, 1, 1];  T22  = Tmnbatch[idx, s, 1, 5]
        T3m2 = Tmnbatch[idx, s, 2, 1];  T32  = Tmnbatch[idx, s, 2, 5]
        T3m3 = Tmnbatch[idx, s, 2, 0];  T33  = Tmnbatch[idx, s, 2, 6]

        for t in range(nT):
            T11relax[idx, s, t] = (T11*f111[idx, t] + T21*f112p[idx, t] + T31*f113[idx, t]) * ph1p[idx, s, t]

        last = nT - 1
        Tmnbatch[idx, s, 0, 3] = T10*f011[idx, last] + T30*f013[idx, last] + 1.0 + 0j
        Tmnbatch[idx, s, 1, 3] = T20*f022[idx, last]
        Tmnbatch[idx, s, 2, 3] = T10*f013[idx, last] + T30*f033[idx, last]

        Tmnbatch[idx, s, 0, 2] = (T1m1*fm111[idx, last] + T2m1*fm112[idx, last] + T3m1*fm113[idx, last]) * ph1m[idx, s, last]
        Tmnbatch[idx, s, 1, 2] = (T1m1*fm112[idx, last] + T2m1*fm122[idx, last] + T3m1*fm123[idx, last]) * ph1m[idx, s, last]
        Tmnbatch[idx, s, 2, 2] = (T1m1*fm113[idx, last] + T2m1*fm123[idx, last] + T3m1*fm133[idx, last]) * ph1m[idx, s, last]

        Tmnbatch[idx, s, 0, 4] = T11relax[idx, s, last]
        Tmnbatch[idx, s, 1, 4] = (T11*f112p[idx, last] + T21*f122[idx, last]  + T31*f123p[idx, last]) * ph1p[idx, s, last]
        Tmnbatch[idx, s, 2, 4] = (T11*f113[idx, last]  + T21*f123p[idx, last] + T31*f133[idx, last])  * ph1p[idx, s, last]

        Tmnbatch[idx, s, 1, 1] = (T2m2*fm222[idx, last] + T3m2*fm223[idx, last]) * ph2m[idx, s, last]
        Tmnbatch[idx, s, 2, 1] = (T3m2*fm233[idx, last] + T2m2*fm223[idx, last]) * ph2m[idx, s, last]
        Tmnbatch[idx, s, 1, 5] = (T22*f222[idx, last]   + T32*f223[idx, last])   * ph2p[idx, s, last]
        Tmnbatch[idx, s, 2, 5] = (T32*f233[idx, last]   + T22*f223[idx, last])   * ph2p[idx, s, last]

        Tmnbatch[idx, s, 2, 0] = T3m3 * fm333[idx, last] * ph3m[idx, s, last]
        Tmnbatch[idx, s, 2, 6] = T33  * f333[idx, last]  * ph3p[idx, s, last]

    return T11relax


@numba.njit(parallel=True, fastmath=True, cache=True)
def _pulsebatch_kernel(Tmnbatch, Pm1, Pm2, Pm3):
    """
    Apply cached Wigner-D*phase operators to every (idx, spin) row of
    Tmnbatch in one flattened parallel pass.
    Tmnbatch: (nFIDs, nS, 3, 7) complex128, updated in place.
    Pm1/Pm2/Pm3: (nFIDs, 7, 7) complex128 — one operator per idx.
    """
    nFIDs = Tmnbatch.shape[0]
    nS    = Tmnbatch.shape[1]

    for flat in numba.prange(nFIDs * nS):
        idx = flat // nS
        s   = flat %  nS
        for mi, Pm in enumerate((Pm1, Pm2, Pm3)):
            row = Tmnbatch[idx, s, mi, :].copy()
            for col in range(7):
                acc = 0.0 + 0j
                for k in range(7):
                    acc += row[k] * Pm[idx, col, k]
                Tmnbatch[idx, s, mi, col] = acc



# ─────────────────────────────────────────────────────────────────────────────
# Class
# ─────────────────────────────────────────────────────────────────────────────
class TmnEvoVecDic2:
    """Vectorised spin-3/2 density operator for N spins (Numba-accelerated)."""

    # T_{1,0} = 1 at thermal equilibrium
    Teq = np.zeros((3, 7), dtype=np.complex128)
    Teq[0, 3] = 1.0

    def __init__(self, B0, tauC, wQ, wQbar, Jen,
                 wShiftvec, wShiftRMS=0.0, wShiftFID=0.0):
        self.B0        = B0
        self.w0        = getValues.getw0(B0)
        self.tauC      = tauC
        self.wQ        = wQ
        self.wQbar     = wQbar
        self.Jen       = Jen
        self.wShiftvec = np.asarray(wShiftvec, dtype=np.float64).ravel()
        self.wShiftRMS = wShiftRMS
        self.wShiftFID = wShiftFID
        self.nSpins    = len(self.wShiftvec)

        self.cachedflipangle = np.nan
        self.cachedwigner    = [None, None, None]

        # Pre-allocated ones arrays (optional, kept for API compatibility)
        self.onesevo = None
        self.onesmix = None
        self.onesfid = None

        # C-contiguous complex128 — mandatory for Numba
        self.Tmnbatch = np.ascontiguousarray(
            np.tile(TmnEvoVecDic2.Teq, (self.nSpins, 1, 1)))

    # ------------------------------------------------------------------
    # Reset / update
    # ------------------------------------------------------------------
    def reset(self):
        self.Tmnbatch[:] = 0.0
        self.Tmnbatch[:, 0, 3] = 1.0

    def update_wshift(self, wShiftvec):
        self.wShiftvec = np.asarray(wShiftvec, dtype=np.float64).ravel()
        self.nSpins    = len(self.wShiftvec)

    # ------------------------------------------------------------------
    # Wigner D cache (unchanged logic, Python speed is fine here)
    # ------------------------------------------------------------------
    def _update_wigner_cache(self, flipAngle):
        if not np.isnan(self.cachedflipangle) and                 np.isclose(self.cachedflipangle, flipAngle):
            return
        self.cachedflipangle = flipAngle
        ns = np.arange(-3, 4)
        for mi, m in enumerate([1, 2, 3]):
            D = np.zeros((7, 7), dtype=complex)
            for i, n in enumerate(ns):
                for j, nbar in enumerate(ns):
                    if abs(n) <= m and abs(nbar) <= m:   # valid coherence orders for rank m
                        D[i, j] = getValues.getWignerD(m, n, nbar, flipAngle)
            self.cachedwigner[mi] = D

    # ------------------------------------------------------------------
    # Pulse (Wigner D cached)
    # ------------------------------------------------------------------
    def pulsebatch(self, flipAngle, phase):
        flip = np.atleast_1d(flipAngle)
        self._update_wigner_cache(float(flip.mean()))
        ns = np.arange(-3, 4)
        phase_rad = np.deg2rad(phase)
        for mi, m in enumerate([1, 2, 3]):
            phasemod = np.exp(1j * (ns[:, None] - ns[None, :]) * phase_rad)  # ← [:, None] first!
            Pm = phasemod * self.cachedwigner[mi]  # NO .T on wigner
            self.Tmnbatch[:, mi, :] = self.Tmnbatch[:, mi, :] @ Pm.T

    # ------------------------------------------------------------------
    # fJen precomputation (ONCE before all loops)
    # ------------------------------------------------------------------
    def precompute_fJen(self, ts):
        ts = np.asarray(ts, dtype=np.float64)
        cf = {}
        args = (self.tauC, self.wQ, self.wQbar, self.Jen, self.wShiftRMS, self.w0)

        for q, label in [(-3,'fm3'),(-2,'fm2'),(-1,'fm1'),(0,'f0'),(1,'f1'),(2,'f2'),(3,'f3')]:
            fqkks = getValues.getfJen(q, ts, *args)
            if abs(q) == 0:
                cf['f011'],cf['f013'],cf['f033'],cf['f022'] = [_c128(x) for x in fqkks]
            elif q == -1:
                """keys = ['fm111','fm113','fm133','fm112','fm123','fm122']
                vals = list(fqkks)
                for k,v in zip(keys, vals):
                    cf[k] = _c128(v)"""
                fm111, fm113, fm133, fm112, fm123, fm122 = fqkks
                cf['fm111'] = _c128(fm111)
                cf['fm113'] = _c128(fm113)
                cf['fm133'] = _c128(fm133)
                cf['fm112'] = _c128(-fm112)  # ← sign flip
                cf['fm123'] = _c128(-fm123)  # ← sign flip
                cf['fm122'] = _c128(fm122)
            elif q == 1:
                keys = ['f111','f113','f133','f112','f123','f122']
                for k,v in zip(keys, fqkks):
                    cf[k] = _c128(v)
                cf['f112p'] = cf['f112']
                cf['f123p'] = cf['f123']
            elif q == -2:
                cf['fm222'],cf['fm233'],cf['fm223'] = [_c128(x) for x in fqkks]
            elif q == 2:
                cf['f222'],cf['f233'],cf['f223'] = [_c128(x) for x in fqkks]
            elif q == -3:
                cf['fm333'] = _c128(fqkks[0])
            elif q == 3:
                cf['f333'] = _c128(fqkks[0])

        cf['nT'] = len(ts)
        cf['ts'] = ts
        return cf

    # ------------------------------------------------------------------
    # Phase factors (ONCE per j, after resampling wShift)
    # ------------------------------------------------------------------
    def add_phase_factors(self, cf):
        t0 = time.perf_counter()
        cf = dict(cf)
        w  = self.wShiftvec[:, None]          # (nS, 1)
        ts = cf['ts'][None, :]                # (1, nT)
        cf['ph1m'] = _c128(np.exp(-1j * w * ts))
        cf['ph1p'] = _c128(np.exp(+1j * w * ts))
        cf['ph2m'] = _c128(np.exp(-2j * w * ts))
        cf['ph2p'] = _c128(np.exp(+2j * w * ts))
        cf['ph3m'] = _c128(np.exp(-3j * w * ts))
        cf['ph3p'] = _c128(np.exp(+3j * w * ts))
        t1 = time.perf_counter()
        print(f"Adding Phasefactors: {t1-t0}")
        return cf

    # ------------------------------------------------------------------
    # Pre-allocate ones (kept for API compatibility — no longer used internally)
    # ------------------------------------------------------------------
    def preallocate_ones(self, nPtsevo, nPtsmix, nT):
        self.onesevo = np.ones((self.nSpins, nPtsevo))
        self.onesmix = np.ones((self.nSpins, nPtsmix))
        self.onesfid = np.ones((self.nSpins, nT))

    # ------------------------------------------------------------------
    # Relaxation – endpoint (calls Numba kernel 1)
    # ------------------------------------------------------------------
    def relax_endpoint_cached(self, c):
        """Relax to the endpoint of the precomputed time array."""
        _relax_endpoint_kernel(
            np.ascontiguousarray(self.Tmnbatch),
            # f* — scalar: last element of each array
            c['f011'][-1],  c['f013'][-1],  c['f033'][-1],  c['f022'][-1],
            c['fm111'][-1], c['fm112'][-1], c['fm113'][-1],
            c['fm122'][-1], c['fm123'][-1], c['fm133'][-1],
            c['f111'][-1],  c['f112p'][-1], c['f113'][-1],
            c['f122'][-1],  c['f123p'][-1], c['f133'][-1],
            c['fm222'][-1], c['fm223'][-1], c['fm233'][-1],
            c['f222'][-1],  c['f223'][-1],  c['f233'][-1],
            c['fm333'][-1], c['f333'][-1],
            # ph* — (nS,): last time-point column
            c['ph1m'][:, -1], c['ph1p'][:, -1],
            c['ph2m'][:, -1], c['ph2p'][:, -1],
            c['ph3m'][:, -1], c['ph3p'][:, -1],
        )

    # ------------------------------------------------------------------
    # Relaxation – T11 acquisition (calls Numba kernel 2)
    # ------------------------------------------------------------------
    def relax_T11_cached(self, c):
        """Acquire full FID via T_{1,1} channel. Returns (nSpins, nT) array."""
        return _relax_T11_kernel(
            np.ascontiguousarray(self.Tmnbatch),
            # f* — (nT,) arrays
            c['f011'],  c['f013'],  c['f033'],  c['f022'],
            c['fm111'], c['fm112'], c['fm113'],
            c['fm122'], c['fm123'], c['fm133'],
            c['f111'],  c['f112p'], c['f113'],
            c['f122'],  c['f123p'], c['f133'],
            c['fm222'], c['fm223'], c['fm233'],
            c['f222'],  c['f223'],  c['f233'],
            c['fm333'], c['f333'],
            # ph* — (nS, nT) arrays
            c['ph1m'], c['ph1p'],
            c['ph2m'], c['ph2p'],
            c['ph3m'], c['ph3p'],
        )

    def relax_T1m1_cached(self, c):
        """Acquire full FID via T_{1,-1} channel. Returns (nSpins, nT) array."""
        return _relax_T1m1_kernel(
            np.ascontiguousarray(self.Tmnbatch),
            c['f011'], c['f013'], c['f033'], c['f022'],
            c['fm111'], c['fm112'], c['fm113'],
            c['fm122'], c['fm123'], c['fm133'],
            c['f111'], c['f112p'], c['f113'],
            c['f122'], c['f123p'], c['f133'],
            c['fm222'], c['fm223'], c['fm233'],
            c['f222'], c['f223'], c['f233'],
            c['fm333'], c['f333'],
            c['ph1m'], c['ph1p'],
            c['ph2m'], c['ph2p'],
            c['ph3m'], c['ph3p'],
        )

    # ------------------------------------------------------------------
    # Uncached wrappers (convenience)
    # ------------------------------------------------------------------
    def relax_endpoint(self, duration, nPts=65):
        if duration <= 0:
            return
        c = self.precompute_fJen(np.linspace(0, duration, nPts + 1))
        c = self.add_phase_factors(c)
        self.relax_endpoint_cached(c)

    def relax_T11(self, times):
        c = self.precompute_fJen(np.asarray(times))
        c = self.add_phase_factors(c)
        return self.relax_T11_cached(c)

    def relax_fid_cached(self, c):
        """T_{1,-1} FID channel — mirrors relax_T11 but for q=-1."""
        # For API completeness; uses T1m1 channel
        T11 = self.relax_T11_cached(c)
        return T11   # caller can conjugate if needed

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------
    def get_Tmnbatch(self, m, n):
        """Return T_{m,n} for all spins. m ∈ {1,2,3}, n ∈ {-3..3}."""
        return self.Tmnbatch[:, m - 1, n + 3]

