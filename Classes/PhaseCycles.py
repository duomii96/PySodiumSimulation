"""
PhaseCycles_VecDic.py
Vectorised fixed TQTPPI simulation for dictionary generation.
Mirrors MATLAB classdef PhaseCycles.

To test if sequences are working and if parallelization effective and to find bugs
Uses Vec Dic old without @Jit
"""

import numpy as np
from getValues import getValues
from TmnEvo_VecDic_old import TmnEvoVecDic_old


class PhaseCycles:
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

        # Phase cycling
        self.nSpins = 1000
        self.NumPhaseCycles = 100
        self.startPhase = 90.0
        self.alphaStep = 45.0
        self.alphas = np.arange(0, 360, self.alphaStep) + self.startPhase  # after __init__

        # Timing
        self.tevo = 0.2e-3
        self.tmix = 0.1e-3
        self.TR = 300e-3
        self.dataPoints = 2048
        self.deadtimeFID = 6e-6
        self.dwelltimeFID = 50e-6

        # Simulation accuracy
        self.nPtsevo = 65   # endpoint accuracy ≡ 257 for smooth T2 decays
        self.nPtsmix = 33

        # Rebuild alphas after default values are set
        self.alphas = np.arange(0, 360, self.alphaStep) + self.startPhase

    # ------------------------------------------------------------------
    # Single-pulse validation
    # ------------------------------------------------------------------
    def SinglePulseVec(self):
        times = (self.deadtimeFID +
                      self.dwelltimeFID * np.arange(self.dataPoints))
        wShiftvec = self.FreqShift + self.wShiftdist.rvs(self.nSpins)
        st = TmnEvoVecDic_old(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                          wShiftvec, self.wShiftRMS, self.wShiftFID)
        st.pulsebatch(self.flip90, 0.0)
        TQ = np.sum(st.get_Tmnbatch(3, 3) + st.get_Tmnbatch(3, -3))
        SQ = np.sum(st.get_Tmnbatch(1, 1) + st.get_Tmnbatch(1, -1))
        c = st.precompute_fJen(times)
        c = st.add_phase_factors(c)
        FID = np.squeeze(np.sum(st.relax_T11_cached(c), axis=0))
        return times, FID, TQ, SQ

    # ------------------------------------------------------------------
    # TQTPPI fixed (without 180°)
    # ------------------------------------------------------------------
    def TQTPPIfixedwo180VJ(self):
        acqTimeVec = (self.deadtimeFID +
                      self.dwelltimeFID * np.arange(self.dataPoints))
        nAlpha = len(self.alphas)
        nFIDs = self.NumPhaseCycles * nAlpha
        nT = len(acqTimeVec)

        fAngles90 = self.flip90 * np.ones(self.nSpins)

        verbose = True  # simplified (MATLAB checks getCurrentTask)

        # Wigner D cache — computed ONCE
        warm = TmnEvoVecDic_old(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        warm.pulsebatch(self.flip90, 0.0)
        cachedflip = warm.cachedflipangle
        cachedwig = warm.cachedwigner

        # fJen — ONCE before all loops
        tsevo = np.linspace(0, self.tevo, self.nPtsevo + 1)
        tsmix = np.linspace(0, self.tmix, self.nPtsmix + 1)
        cfevo = warm.precompute_fJen(tsevo)
        cfmix = warm.precompute_fJen(tsmix)
        cffid = warm.precompute_fJen(acqTimeVec)

        warm.preallocate_ones(self.nPtsevo + 1, self.nPtsmix + 1, nT)

        # Two reusable state objects
        stp1 = TmnEvoVecDic_old(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        stp2 = TmnEvoVecDic_old(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        for sp in [stp1, stp2]:
            sp.cachedflipangle = cachedflip
            sp.cachedwigner = cachedwig
            sp.onesevo = warm.onesevo
            sp.onesmix = warm.onesmix
            sp.onesfid = warm.onesfid

        FIDsbeta = np.zeros((nFIDs, nT, 2), dtype=complex)

        for j in range(self.NumPhaseCycles):
            wShiftvec = (self.FreqShift +
                         self.wShiftdist.rvs(self.nSpins) if self.wShiftdist is not None
                         else self.FreqShift * np.ones(self.nSpins))
            stp1.update_wshift(wShiftvec)
            stp2.update_wshift(wShiftvec)




            if self.varFlip90 and self.flipShiftdist90 is not None:
                flip1vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip2vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip3vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                fAngles90 = flip1vec
            else:
                flip1vec = self.flip90
                flip2vec = self.flip90
                flip3vec = self.flip90



            # Phase factors ONCE per j
            cfevo_ph = stp1.add_phase_factors(cfevo)
            cfmix_ph = stp1.add_phase_factors(cfmix)
            cffid_ph = stp1.add_phase_factors(cffid)

            for alphaIdx, alpha in enumerate(self.alphas):

                idx = j * nAlpha + alphaIdx

                # Reset both states
                stp1.reset(); stp1.update_wshift(wShiftvec)
                stp2.reset(); stp2.update_wshift(wShiftvec)

                # Pulse 1 — identical for p1 and p2
                stp1.pulsebatch(flip1vec, alpha)
                stp2.pulsebatch(flip1vec, alpha)

                # Evolution — identical
                stp1.relax_endpoint_cached(cfevo_ph)
                stp2.relax_endpoint_cached(cfevo_ph)

                # Pulse 2 — branches diverge
                stp1.pulsebatch(flip2vec, alpha + 90.0)
                stp2.pulsebatch(flip2vec, alpha - 90.0)

                # Mixing
                stp1.relax_endpoint_cached(cfmix_ph)
                stp2.relax_endpoint_cached(cfmix_ph)

                # Pulse 3
                stp1.pulsebatch(flip3vec, 0.0)
                stp2.pulsebatch(flip3vec, 0.0)

                # Acquisition T_{1,1} channel
                FIDsbeta[idx, :, 0] = np.sum(stp1.relax_T11_cached(cffid_ph), axis=0)
                FIDsbeta[idx, :, 1] = np.sum(stp2.relax_T11_cached(cffid_ph), axis=0)

                if verbose and idx % 10 == 0:
                    print(f"  phase step {idx+1}/{nFIDs}")

        FIDs = getValues.switchReImFID(FIDsbeta[:, :, 0] + FIDsbeta[:, :, 1])
        mqFID = np.squeeze(np.sum(FIDs, axis=1))
        mqSpectrum = np.fft.fftshift(np.fft.fft(mqFID))
        return acqTimeVec, mqSpectrum, FIDs, fAngles90

  