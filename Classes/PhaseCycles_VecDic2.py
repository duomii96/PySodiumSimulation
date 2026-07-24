"""
PhaseCycles_VecDic.py
Vectorised simulation for dictionary generation.
Mirrors MATLAB classdef PhaseCycles.

"""

import numpy as np
from Classes.getValues import getValues
from Classes.TmnEvo_VecDic2 import (
    TmnEvoVecDic2,
    _relax_endpoint_kernel_batched,
    _relax_T11_kernel_batched,
    _pulsebatch_kernel,
)

ENDPOINT_KEYS = ['f011','f013','f033','f022',
                 'fm111','fm112','fm113','fm122','fm123','fm133',
                 'f111','f112p','f113','f122','f123p','f133',
                 'fm222','fm223','fm233','f222','f223','f233',
                 'fm333','f333']
PH_KEYS = ['ph1m','ph1p','ph2m','ph2p','ph3m','ph3p']

CHUNK_J = 4  # number of PHASE CYCLES (j) processed per chunk, tune for RAM


def _build_wigner_block(flipAngle, m):
    ns = np.arange(-3, 4)
    D = np.zeros((7, 7), dtype=complex)
    for i, n in enumerate(ns):
        for j, nbar in enumerate(ns):
            if abs(n) <= m and abs(nbar) <= m:
                D[i, j] = getValues.getWignerD(m, n, nbar, flipAngle)
    return D


def _stack_pulse_operator(flipAngle_scalar, phases, n):
    phases = np.broadcast_to(np.asarray(phases, dtype=float), (n,))
    ns = np.arange(-3, 4)
    Ds = [_build_wigner_block(flipAngle_scalar, m) for m in (1, 2, 3)]
    phase_rad = np.deg2rad(phases)
    dn = (ns[:, None] - ns[None, :])
    phasemod = np.exp(1j * phase_rad[:, None, None] * dn[None, :, :])
    out = []
    for D in Ds:
        Pm = phasemod * D[None, :, :]
        out.append(np.ascontiguousarray(np.transpose(Pm, (0, 2, 1))))
    return out


class _PhaseFactorBuffers:
    """
    Preallocated, REUSED buffers for endpoint (nS,) and full (nS,nT)
    phase-factor arrays, sized once for a given chunk and overwritten
    in place on every call via np.exp(..., out=buf). Avoids the
    malloc/free churn that caused the 0.008s-0.6s variance per call.
    """
    def __init__(self, nS, nT_evo, nT_mix, nT_fid, nAlpha):
        self.nS = nS
        # evo-half buffers: one slot per alpha within a phase cycle
        self.evo = {k: np.empty((nAlpha, nS, nT_evo), dtype=np.complex128) for k in PH_KEYS}
        # mix / fid buffers: one set per phase cycle (shared across alphas)
        self.mix = {k: np.empty((nS, nT_mix), dtype=np.complex128) for k in PH_KEYS}
        self.fid = {k: np.empty((nS, nT_fid), dtype=np.complex128) for k in PH_KEYS}

    def fill_mix(self, wShiftvec, ts_mix):
        w = wShiftvec[:, None]
        ts = ts_mix[None, :]
        np.exp(-1j * w * ts, out=self.mix["ph1m"])
        np.exp(+1j * w * ts, out=self.mix["ph1p"])
        np.exp(-2j * w * ts, out=self.mix["ph2m"])
        np.exp(+2j * w * ts, out=self.mix["ph2p"])
        np.exp(-3j * w * ts, out=self.mix["ph3m"])
        np.exp(+3j * w * ts, out=self.mix["ph3p"])
        return self.mix

    def fill_fid(self, wShiftvec, ts_fid):
        w = wShiftvec[:, None]
        ts = ts_fid[None, :]
        np.exp(-1j * w * ts, out=self.fid["ph1m"])
        np.exp(+1j * w * ts, out=self.fid["ph1p"])
        np.exp(-2j * w * ts, out=self.fid["ph2m"])
        np.exp(+2j * w * ts, out=self.fid["ph2p"])
        np.exp(-3j * w * ts, out=self.fid["ph3m"])
        np.exp(+3j * w * ts, out=self.fid["ph3p"])
        return self.fid

    def fill_evo(self, wShiftvec, ts_batch):
        """ts_batch: (n_per_j, nT_evo) -- one row per alpha, ALL computed
        in one broadcasted call per phase cycle (not per idx)."""
        w = wShiftvec[None, :, None]
        ts = ts_batch[:, None, :]
        np.exp(-1j * w * ts, out=self.evo["ph1m"])
        np.exp(+1j * w * ts, out=self.evo["ph1p"])
        np.exp(-2j * w * ts, out=self.evo["ph2m"])
        np.exp(+2j * w * ts, out=self.evo["ph2p"])
        np.exp(-3j * w * ts, out=self.evo["ph3m"])
        np.exp(+3j * w * ts, out=self.evo["ph3p"])
        return self.evo


def _endpoint_scalars(cf_template, key):
    return cf_template[key][-1]

class PhaseCyclesVecDic2:
    def __init__(self, B0, tauC, wQ, wQbar, Jen,
                 wShift=0.0, wShiftRMS=0.0, wShiftFID=0.0):
        self.B0 = B0
        self.w0 = getValues.getw0(B0)
        self.tauC = tauC
        self.wQ = wQ
        self.wQbar = wQbar
        self.Jen = Jen
        self.wShift = wShift
        self.FreqShift = wShift
        self.wShiftRMS = wShiftRMS
        self.wShiftFID = wShiftFID
        self.wShiftdist = None  # set externally (scipy frozen distribution)

        # RF parameters
        self.flip90 = 90.0
        self.flip180 = 180.0
        self.varFlip90 = False
        self.flipShiftdist90 = None
        self.flipShiftSig90 = 5  # Sigma of Gaussian for FlipAngle Variation

        # Phase cycling
        self.nSpins = 1000
        self.NumPhaseCycles = 16
        self.startPhase = 90.0
        self.alphaStep = 45.0
        self.alphas = np.arange(0, 360, self.alphaStep) + self.startPhase  # after __init__

        # Timing
        self.tevo = 0.2e-3
        self.tevoStep = 0.2e-3
        self.tmix = 0.1e-3
        self.TR = 300e-3
        self.dataPoints = 2048
        self.deadtimeFID = 6e-6
        self.dwelltimeFID = 50e-6
        self.tauVec = np.linspace(0.0, 70e-3, 210)  # half echo time

        # Simulation accuracy
        self.nPtsevo = 65  # endpoint accuracy ≡ 257 for smooth T2 decays
        self.nPtsmix = 33

        # Rebuild alphas after default values are set
        self.alphas = np.arange(0, 360, self.alphaStep) + self.startPhase



    # ------------------------------------------------------------------
    # TQTPPI (with 180°)
    # ------------------------------------------------------------------
    def TQTPPIWith180_batched(self, chunk_j=32):
        acqTimeVec = (self.deadtimeFID +
                      self.dwelltimeFID * np.arange(self.dataPoints))
        nAlpha = len(self.alphas)
        nFIDs = self.NumPhaseCycles * nAlpha
        nT = len(acqTimeVec)
        nS = self.nSpins

        warm = TmnEvoVecDic2(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        ts_mix = np.linspace(0, self.tmix, self.nPtsmix)
        cfmix_template = warm.precompute_fJen(ts_mix)
        cffid_template = warm.precompute_fJen(acqTimeVec)

        all_tevos = self.tevo0 + np.arange(nFIDs) * self.tevoStep
        all_ts_half = [np.linspace(0, t / 2, self.nPtsevo) for t in all_tevos]
        all_cf_half_template = [warm.precompute_fJen(ts) for ts in all_ts_half]
        # ts grids are identical across idx within the SAME idx-offset pattern;
        # we need the actual ts arrays per idx for evo buffers:
        all_ts_half_arr = np.stack(all_ts_half, axis=0)  # (nFIDs, nPtsevo)

        fAngles90 = self.flip90 * np.ones(self.nSpins)
        alphas_full = np.array([self.alphas[i % nAlpha] for i in range(nFIDs)])

        FIDsbeta = np.zeros((nFIDs, nT, 2), dtype=complex)

        buffers = _PhaseFactorBuffers(nS, self.nPtsevo, self.nPtsmix, nT, nAlpha)

        for jstart in range(0, self.NumPhaseCycles, chunk_j):
            jend = min(jstart + chunk_j, self.NumPhaseCycles)
            n_j = jend - jstart
            n = n_j * nAlpha
            idx0 = jstart * nAlpha
            idx1 = jend * nAlpha
            sl = slice(idx0, idx1)

            # Per-chunk containers (allocated once per chunk, not per idx)
            ph_half_all = {k: np.empty((n, nS), dtype=np.complex128) for k in PH_KEYS}
            ph_mix_all = {k: np.empty((n, nS), dtype=np.complex128) for k in PH_KEYS}
            ph_fid_all = {k: np.empty((n, nS, nT), dtype=np.complex128) for k in PH_KEYS}

            for jj in range(n_j):
                j = jstart + jj
                wShiftvec = (self.FreqShift + self.wShiftdist.rvs(nS)
                             if self.wShiftdist is not None
                             else self.FreqShift * np.ones(nS))
                if self.varFlip90 and self.flipShiftdist90 is not None:
                    flip1vec = self.flip90 + self.flipShiftdist90.rvs(nS)
                    fAngles90 = flip1vec

                ts_batch_j = all_ts_half_arr[j * nAlpha:(j + 1) * nAlpha]  # (nAlpha, nPtsevo)
                evo_pf = buffers.fill_evo(wShiftvec, ts_batch_j)  # ONE call/j, not nAlpha calls
                mix_pf = buffers.fill_mix(wShiftvec, ts_mix)
                fid_pf = buffers.fill_fid(wShiftvec, acqTimeVec)

                rows = slice(jj * nAlpha, (jj + 1) * nAlpha)
                for k in PH_KEYS:
                    ph_half_all[k][rows] = evo_pf[k][:, :, -1]  # endpoint = last time col
                    ph_mix_all[k][rows] = mix_pf[k][None, :, -1]  # same mix factor, all alphas
                    ph_fid_all[k][rows] = fid_pf[k][None, :, :]  # same fid factor, all alphas

            endpoint_half = {k: np.full(n, _endpoint_scalars(all_cf_half_template[idx0], k),
                                        dtype=np.complex128) for k in []}  # placeholder, see note below
            # NOTE: f011/f013/... endpoint scalars for the evo-half ARE idx-dependent
            # (they depend on tevo/2, which differs per idx), unlike ph_half above where
            # only the wShift differs per j (angle-independent). Build them per idx:
            endpoint_half = {k: np.array([all_cf_half_template[idx0 + i][k][-1] for i in range(n)],
                                         dtype=np.complex128)
                             for k in ENDPOINT_KEYS}
            endpoint_mix = {k: np.full(n, cfmix_template[k][-1], dtype=np.complex128)
                            for k in ENDPOINT_KEYS}
            fid_arrays = {k: np.tile(cffid_template[k][None, :], (n, 1)) for k in ENDPOINT_KEYS}

            alphas_chunk = alphas_full[sl]

            P90_1 = _stack_pulse_operator(self.flip90, alphas_chunk, n)
            P180 = _stack_pulse_operator(self.flip180, alphas_chunk + 90, n)
            P90_2a = _stack_pulse_operator(self.flip90, alphas_chunk + 90, n)
            P90_2b = _stack_pulse_operator(self.flip90, alphas_chunk - 90, n)
            P90_3 = _stack_pulse_operator(self.flip90, np.zeros(n), n)

            Tmn1 = np.zeros((n, nS, 3, 7), dtype=np.complex128);
            Tmn1[:, :, 0, 3] = 1.0
            Tmn2 = np.zeros((n, nS, 3, 7), dtype=np.complex128);
            Tmn2[:, :, 0, 3] = 1.0

            args_half = [endpoint_half[k] for k in ENDPOINT_KEYS] + [ph_half_all[k] for k in PH_KEYS]
            args_mix = [endpoint_mix[k] for k in ENDPOINT_KEYS] + [ph_mix_all[k] for k in PH_KEYS]
            args_fid = [fid_arrays[k] for k in ENDPOINT_KEYS] + [ph_fid_all[k] for k in PH_KEYS]

            _pulsebatch_kernel(Tmn1, *P90_1)
            _pulsebatch_kernel(Tmn2, *P90_1)

            _relax_endpoint_kernel_batched(Tmn1, *args_half)
            _relax_endpoint_kernel_batched(Tmn2, *args_half)

            _pulsebatch_kernel(Tmn1, *P180)
            _pulsebatch_kernel(Tmn2, *P180)

            _relax_endpoint_kernel_batched(Tmn1, *args_half)
            _relax_endpoint_kernel_batched(Tmn2, *args_half)

            _pulsebatch_kernel(Tmn1, *P90_2a)
            _pulsebatch_kernel(Tmn2, *P90_2b)

            _relax_endpoint_kernel_batched(Tmn1, *args_mix)
            _relax_endpoint_kernel_batched(Tmn2, *args_mix)

            _pulsebatch_kernel(Tmn1, *P90_3)
            _pulsebatch_kernel(Tmn2, *P90_3)

            T11_1 = _relax_T11_kernel_batched(Tmn1, *args_fid)
            T11_2 = _relax_T11_kernel_batched(Tmn2, *args_fid)

            FIDsbeta[sl, :, 0] = np.sum(T11_1, axis=1)
            FIDsbeta[sl, :, 1] = np.sum(T11_2, axis=1)

            del T11_1, T11_2, Tmn1, Tmn2, ph_fid_all, fid_arrays

        FIDs = getValues.switchReImFID(FIDsbeta[:, :, 0] + FIDsbeta[:, :, 1])
        mqFID = np.real(np.squeeze(np.sum(FIDs, axis=1)))
        mqSpectrum = np.fft.fftshift(np.fft.fft(mqFID))
        return acqTimeVec, mqSpectrum, mqFID, FIDs, fAngles90, all_tevos

