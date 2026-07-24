"""
TmnEvo_VecDic.py
Vectorised spin-3/2 density operator for N spins.
Mirrors MATLAB classdef TmnEvoVecDic_old.

v4 optimisations:
  - Dual-branch relaxation (p1 and p2 share Tmnbatch as 2*nSpins x 3 x 7
    up to pulse 2, halving all relax-from-cache calls before mixing).
  - nPtsevo default reduced to 65.
  - ones pre-allocated and reused across relax calls.
  - Caching fJen ONCE before all loops (no wShift dependency).
  - Phase factors ONCE per j after resampling wShift.
  - Wigner D ONCE on first flip angle.
"""

import numpy as np
from Classes.getValues import getValues


class TmnEvoVecDic_old:
    # Thermal equilibrium state: T_{1,0} = 1, all others 0
    # Tmnbatch index layout: axis 1 → m in {1,2,3}; axis 2 → n+3 in 0..6 (n=-3..3)
    Teq = np.zeros((3, 7), dtype=complex)
    Teq[0, 3] = 1.0  # T_{1,0}

    def __init__(self, B0, tauC, wQ, wQbar, Jen,
                 wShiftvec, wShiftRMS=0.0, wShiftFID=0.0):
        self.B0 = B0
        self.w0 = getValues.getw0(B0)
        self.tauC = tauC
        self.wQ = wQ
        self.wQbar = wQbar
        self.Jen = Jen
        self.wShiftvec = np.asarray(wShiftvec, dtype=float).ravel()
        self.wShiftRMS = wShiftRMS
        self.wShiftFID = wShiftFID
        self.nSpins = len(self.wShiftvec)

        self.cachedflipangle = np.nan
        self.cachedwigner = [None, None, None]   # list of 3 (7x7) matrices

        # Pre-allocated ones arrays (set via preallocate_ones)
        self.onesevo = None
        self.onesmix = None
        self.onesfid = None

        # Initialise Tmnbatch: shape (nSpins, 3, 7)
        self.Tmnbatch = np.zeros((self.nSpins, 3, 7), dtype=complex)
        self._set_thermal()

    def _set_thermal(self):
        """Set all spins to thermal equilibrium."""
        self.Tmnbatch[:] = TmnEvoVecDic_old.Teq[np.newaxis, :, :]

    def reset(self):
        """Reset to thermal equilibrium without re-allocating arrays."""
        self.Tmnbatch[:] = 0.0
        self.Tmnbatch[:, 0, 3] = 1.0  # T_{1,0} = 1

    def update_wshift(self, wShiftvec):
        self.wShiftvec = np.asarray(wShiftvec, dtype=float).ravel()
        self.nSpins = len(self.wShiftvec)

    # ------------------------------------------------------------------
    # Wigner D cache
    # ------------------------------------------------------------------
    def _update_wigner_cache(self, flipAngle):
        if np.isclose(self.cachedflipangle, flipAngle, equal_nan=False) and                 not np.isnan(self.cachedflipangle):
            return
        self.cachedflipangle = flipAngle
        ns = np.arange(-3, 4)   # -3 … 3
        for mi, m in enumerate([1, 2, 3]):
            D = np.zeros((7, 7), dtype=complex)
            for i, n in enumerate(ns):
                for j, nbar in enumerate(ns):
                    if abs(n) <= m and abs(nbar) <= m:  # valid coherence orders for rank m
                        D[i, j] = getValues.getWignerD(m, n, nbar, flipAngle)
            self.cachedwigner[mi] = D

    # ------------------------------------------------------------------
    # Pulse (Wigner D cached; only exp per phase step)
    # ------------------------------------------------------------------
    def pulsebatch(self, flipAngle, phase):
        """Apply RF pulse with given flip angle and phase to all spins."""
        # flipAngle can be scalar or per-spin array
        flip = np.atleast_1d(flipAngle)
        if flip.size == 1:
            self._update_wigner_cache(float(flip[0]))
        else:
            # Per-spin flip angles: compute Wigner for each unique angle
            # (simplified: use mean angle for caching — exact per-spin needs loop)
            self._update_wigner_cache(float(flip.mean()))

        ns = np.arange(-3, 4)  # shape (7,)
        phase_rad = np.deg2rad(phase)

        for mi, m in enumerate([1, 2, 3]):
            # Phase modulation: exp(i*(n - nbar)*phase) → diagonal outer product
            phasemod = np.exp(1j * (ns[:, np.newaxis] - ns[np.newaxis, :]) * phase_rad)
            Pm = phasemod * self.cachedwigner[mi]  # (7, 7)

            # Current row: Tmnbatch[:, mi, :] shape (nSpins, 7)
            Tmn = self.Tmnbatch[:, mi, :]          # (nSpins, 7)
            # New value: Tmn @ Pm.T  → (nSpins, 7)
            self.Tmnbatch[:, mi, :] = Tmn @ Pm.T

    # ------------------------------------------------------------------
    # Pre-allocate ones matrices
    # ------------------------------------------------------------------
    def preallocate_ones(self, nPtsevo, nPtsmix, nT):
        self.onesevo = np.ones((self.nSpins, nPtsevo))
        self.onesmix = np.ones((self.nSpins, nPtsmix))
        self.onesfid = np.ones((self.nSpins, nT))

    # ------------------------------------------------------------------
    # fJen precomputation (ONCE before all loops, no wShift dependency)
    # ------------------------------------------------------------------
    def precompute_fJen(self, ts):
        """Precompute relaxation functions for all time points ts."""
        ts = np.asarray(ts, dtype=float)
        cf = {}
        for q, label in [(-3,'fm3'), (-2,'fm2'), (-1,'fm1'),
                          (0,'f0'), (1,'f1'), (2,'f2'), (3,'f3')]:
            fqkks = getValues.getfJen(q, ts, self.tauC, self.wQ, self.wQbar,
                                      self.Jen, self.wShiftRMS, self.w0)
            if abs(q) == 0:
                cf['f011'], cf['f013'], cf['f033'], cf['f022'] = fqkks
            elif q == -1:
                cf['fm111'],cf['fm113'],cf['fm133'],cf['fm112'],cf['fm123'],cf['fm122'] = fqkks
                # sign convention from MATLAB: fm112 = -f112, fm123 = -f123
                cf['fm112'] = -cf['fm112']
                cf['fm123'] = -cf['fm123']
            elif q == 1:
                cf['f111'],cf['f113'],cf['f133'],cf['f112'],cf['f123'],cf['f122'] = fqkks
                cf['f112p'] = cf['f112']   # positive phase versions
                cf['f123p'] = cf['f123']
            elif q == -2:
                cf['fm222'],cf['fm233'],cf['fm223'] = fqkks
            elif q == 2:
                cf['f222'],cf['f233'],cf['f223'] = fqkks
            elif q == -3:
                cf['fm333'] = fqkks[0]
            elif q == 3:
                cf['f333'] = fqkks[0]
        cf['nT'] = len(ts)
        cf['ts'] = ts
        return cf

    # ------------------------------------------------------------------
    # Phase factors (ONCE per j, after resampling wShift)
    # ------------------------------------------------------------------
    def add_phase_factors(self, cf):
        """Add wShift phase factors to precomputed cache struct (returns new dict)."""
        cf = dict(cf)   # shallow copy
        w = self.wShiftvec[:, np.newaxis]   # (nSpins, 1)
        ts = cf['ts'][np.newaxis, :]        # (1, nT)
        cf['ph1m'] = np.exp(-1j * w * ts)
        cf['ph1p'] = np.exp(+1j * w * ts)
        cf['ph2m'] = np.exp(-2j * w * ts)
        cf['ph2p'] = np.exp(+2j * w * ts)
        cf['ph3m'] = np.exp(-3j * w * ts)
        cf['ph3p'] = np.exp(+3j * w * ts)
        return cf

    def add_phase_factors_dual(self, cf):
        """Phase factors for dual batch (2*nSpins). wShiftvec repeated."""
        w = np.concatenate([self.wShiftvec, self.wShiftvec])[:, np.newaxis]
        return self._add_phase_factors_w(cf, w)

    def _add_phase_factors_w(self, cf, w):
        cf = dict(cf)
        ts = cf['ts'][np.newaxis, :]
        cf['ph1m'] = np.exp(-1j * w * ts)
        cf['ph1p'] = np.exp(+1j * w * ts)
        cf['ph2m'] = np.exp(-2j * w * ts)
        cf['ph2p'] = np.exp(+2j * w * ts)
        cf['ph3m'] = np.exp(-3j * w * ts)
        cf['ph3p'] = np.exp(+3j * w * ts)
        return cf

    # ------------------------------------------------------------------
    # Relaxation convenience wrappers
    # ------------------------------------------------------------------
    def relax_endpoint_cached(self, c):
        self._relax_from_cache(c, return_T1m1=False, return_T11=False)

    def relax_T11_cached(self, c):
        return self._relax_from_cache(c, return_T1m1=False, return_T11=True)

    def relax_fid_cached(self, c):
        return self._relax_from_cache(c, return_T1m1=True, return_T11=False)

    def relax_endpoint(self, duration, nPts=65):
        if duration <= 0:
            return
        c = self.precompute_fJen(np.linspace(0, duration, nPts + 1))
        c = self.add_phase_factors(c)
        self._relax_from_cache(c, False, False)

    def relax_T11(self, times):
        c = self.precompute_fJen(np.asarray(times))
        c = self.add_phase_factors(c)
        return self._relax_from_cache(c, False, True)

    def relax_fid(self, acqtimes):
        c = self.precompute_fJen(np.asarray(acqtimes))
        c = self.add_phase_factors(c)
        return self._relax_from_cache(c, True, False)

    def get_Tmnbatch(self, m, n):
        """Get T_{m,n} for all spins. m in {1,2,3}, n in {-3..3}."""
        return self.Tmnbatch[:, m - 1, n + 3]

    # ------------------------------------------------------------------
    # Core relaxation from cache
    # ------------------------------------------------------------------
    def _relax_from_cache(self, c, return_T1m1, return_T11):
        nT = c['nT']
        nS = self.nSpins
        T = self.Tmnbatch  # (nS, 3, 7)

        # Subtract 1 from T_{1,0} (deviation from equilibrium)
        T[:, 0, 3] -= 1.0

        # Extract pre-relaxation coherences  (nS,) vectors
        T10pre = T[:, 0, 3].copy()
        T20pre = T[:, 1, 3].copy()
        T30pre = T[:, 2, 3].copy()
        T1m1pre = T[:, 0, 2].copy()
        T11pre  = T[:, 0, 4].copy()
        T2m1pre = T[:, 1, 2].copy()
        T21pre  = T[:, 1, 4].copy()
        T3m1pre = T[:, 2, 2].copy()
        T31pre  = T[:, 2, 4].copy()
        T2m2pre = T[:, 1, 1].copy()
        T22pre  = T[:, 1, 5].copy()
        T3m2pre = T[:, 2, 1].copy()
        T32pre  = T[:, 2, 5].copy()
        T3m3pre = T[:, 2, 0].copy()
        T33pre  = T[:, 2, 6].copy()

        # Broadcast: (nS,1) * (1,nT) → (nS,nT)
        def _b(v): return v[:, np.newaxis]

        ph1m = c['ph1m']  # (nS, nT)
        ph1p = c['ph1p']
        ph2m = c['ph2m']
        ph2p = c['ph2p']
        ph3m = c['ph3m']
        ph3p = c['ph3p']

        # Choose pre-allocated ones matrix
        if self.onesevo is not None and self.onesevo.shape == (nS, nT):
            onemat = self.onesevo
        elif self.onesmix is not None and self.onesmix.shape == (nS, nT):
            onemat = self.onesmix
        elif self.onesfid is not None and self.onesfid.shape == (nS, nT):
            onemat = self.onesfid
        else:
            onemat = np.ones((nS, nT))

        # q=0 relaxation (m=1,2,3 zero-quantum)
        T10relax = _b(T10pre) * c['f011'] + _b(T30pre) * c['f013'] * onemat
        T20relax = _b(T20pre) * c['f022']
        T30relax = _b(T10pre) * c['f013'] + _b(T30pre) * c['f033']

        # q=±1 relaxation (single-quantum)
        T1m1relax = (_b(T1m1pre)*c['fm111'] + _b(T2m1pre)*c['fm112'] + _b(T3m1pre)*c['fm113']) * ph1m
        T2m1relax = (_b(T1m1pre)*c['fm112'] + _b(T2m1pre)*c['fm122'] + _b(T3m1pre)*c['fm123']) * ph1m
        T3m1relax = (_b(T1m1pre)*c['fm113'] + _b(T2m1pre)*c['fm123'] + _b(T3m1pre)*c['fm133']) * ph1m

        T11relax  = (_b(T11pre)*c['f111']  + _b(T21pre)*c['f112p'] + _b(T31pre)*c['f113'])  * ph1p
        T21relax  = (_b(T11pre)*c['f112p'] + _b(T21pre)*c['f122']  + _b(T31pre)*c['f123p']) * ph1p
        T31relax  = (_b(T11pre)*c['f113']  + _b(T21pre)*c['f123p'] + _b(T31pre)*c['f133'])  * ph1p

        # q=±2 relaxation (double-quantum)
        T2m2relax = (_b(T2m2pre)*c['fm222'] + _b(T3m2pre)*c['fm223']) * ph2m
        T3m2relax = (_b(T3m2pre)*c['fm233'] + _b(T2m2pre)*c['fm223']) * ph2m
        T22relax  = (_b(T22pre)*c['f222']  + _b(T32pre)*c['f223'])  * ph2p
        T32relax  = (_b(T32pre)*c['f233']  + _b(T22pre)*c['f223'])  * ph2p

        # q=±3 relaxation (triple-quantum)
        T3m3relax = _b(T3m3pre) * c['fm333'] * ph3m
        T33relax  = _b(T33pre)  * c['f333']  * ph3p

        # Store final time-point back into Tmnbatch
        self.Tmnbatch[:, 0, 3] = T10relax[:, -1]
        self.Tmnbatch[:, 1, 3] = T20relax[:, -1]
        self.Tmnbatch[:, 2, 3] = T30relax[:, -1]
        self.Tmnbatch[:, 0, 2] = T1m1relax[:, -1]
        self.Tmnbatch[:, 0, 4] = T11relax[:, -1]
        self.Tmnbatch[:, 1, 2] = T2m1relax[:, -1]
        self.Tmnbatch[:, 1, 4] = T21relax[:, -1]
        self.Tmnbatch[:, 2, 2] = T3m1relax[:, -1]
        self.Tmnbatch[:, 2, 4] = T31relax[:, -1]
        self.Tmnbatch[:, 1, 1] = T2m2relax[:, -1]
        self.Tmnbatch[:, 1, 5] = T22relax[:, -1]
        self.Tmnbatch[:, 2, 1] = T3m2relax[:, -1]
        self.Tmnbatch[:, 2, 5] = T32relax[:, -1]
        self.Tmnbatch[:, 2, 0] = T3m3relax[:, -1]
        self.Tmnbatch[:, 2, 6] = T33relax[:, -1]

        if return_T1m1:
            return T1m1relax * np.exp(-self.wShiftFID * 1j * self.wShiftvec[:, np.newaxis] * c['ts'][np.newaxis, :])
        elif return_T11:
            return T11relax
        return None