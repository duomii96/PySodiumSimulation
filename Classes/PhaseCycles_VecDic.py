"""
PhaseCycles_VecDic.py
Vectorised simulation for dictionary generation.
Mirrors MATLAB classdef PhaseCycles.

"""

import numpy as np
from Classes.getValues import getValues
from Classes.TmnEvo_VecDic import TmnEvoVecDic
import copy



class PhaseCyclesVecDic:
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
        self.flipShiftSig90 = 5 # Sigma of Gaussian for FlipAngle Variation

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
        self.tauVec = np.linspace(0.0,70e-3,210) # half echo time

        # Simulation accuracy
        self.nPtsevo = 65   # endpoint accuracy ≡ 257 for smooth T2 decays
        self.nPtsmix = 33

        # Rebuild alphas after default values are set
        self.alphas = np.arange(0, 360, self.alphaStep) + self.startPhase

    # ------------------------------------------------------------------
    # Single-pulse validation
    # ------------------------------------------------------------------
    def SinglePulse(self):
        times = self.deadtimeFID + self.TR * np.arange(self.dataPoints) / self.dataPoints
        wShiftvec = self.FreqShift + self.wShiftdist.rvs(self.nSpins)
        st = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                          wShiftvec, self.wShiftRMS, self.wShiftFID)
        st.pulsebatch(self.flip90, 0.0)
        TQ = np.sum(st.get_Tmnbatch(3, 3) + st.get_Tmnbatch(3, -3))
        SQ = np.sum(st.get_Tmnbatch(1, 1) + st.get_Tmnbatch(1, -1))
        c = st.precompute_fJen(times)
        c = st.add_phase_factors(c)
        FID = np.squeeze(np.sum(st.relax_T11_cached(c), axis=0))
        FID *= np.exp(1j * np.pi)
        return times, FID, TQ, SQ

    def SpinEcho(self):
        """
        Multi-echo spin echo sequence for spin-3/2.
        Structure:
          90° pulse → [τ → 180° → τ → acquire] × nEchoes

        self.tauVec : array of τ values (half echo times), length nEchoes
                      e.g. np.linspace(1e-3, 30e-3, 16)
        self.nEchoes: number of echoes (= len(self.tauVec))

        Each echo is acquired as a full FID starting at the echo centre.
        Returns:
          acqTimeVec  : (nT,)        acquisition time vector
          echoFIDs    : (nEchoes, nT) complex FID per echo
          echoAmps    : (nEchoes,)    on-resonance echo amplitude per echo
          tauVec      : (nEchoes,)    τ values used
        """
        acqTimeVec = (self.deadtimeFID +
                      self.dwelltimeFID * np.arange(self.dataPoints))
        nT = len(acqTimeVec)
        nEcho = len(self.tauVec)
        verbose = False

        # ── Wigner D cache — 90° and 180°, computed ONCE ──────────────────────────
        warm = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        warm.pulsebatch(self.flip90, 0)
        cached_flip90 = warm.cachedflipangle
        cached_wig90 = [w.copy() for w in warm.cachedwigner]

        warm.pulsebatch(self.flip180, 0)
        cached_flip180 = warm.cachedflipangle
        cached_wig180 = [w.copy() for w in warm.cachedwigner]

        # ── Precompute ALL τ evolution caches upfront ─────────────────────────────
        # Each τ is unique → one cf per echo
        all_ts_tau = [np.linspace(0, tau, self.nPtsevo) for tau in self.tauVec]
        all_cf_tau = [warm.precompute_fJen(ts) for ts in all_ts_tau]

        # FID acquisition cache — fixed
        cffid = warm.precompute_fJen(acqTimeVec)

        # ── State object — single spin ensemble, reused across echoes ─────────────
        stp = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                           np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        warm.preallocate_ones(self.nPtsevo, self.nPtsevo, nT)  # reuse evo-sized ones
        stp.onesevo = warm.onesevo
        stp.onesmix = warm.onesmix  # second τ half uses same size
        stp.onesfid = warm.onesfid

        # ── Sample spin ensemble ONCE (single-shot experiment) ────────────────────
        wShiftvec = (self.FreqShift +
                     self.wShiftdist.rvs(self.nSpins) if self.wShiftdist is not None
                     else self.FreqShift * np.ones(self.nSpins))

        if self.varFlip90 and self.flipShiftdist90 is not None:
            flip90vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
            flip180vec = self.flip180 + self.flipShiftdist90.rvs(self.nSpins)
        else:
            flip90vec = self.flip90
            flip180vec = self.flip180

        stp.update_wshift(wShiftvec)

        # ── Add phase factors for FID ──────────────────────────────────────────────
        cffid_pf = stp.add_phase_factors(cffid)

        # ── Output arrays ─────────────────────────────────────────────────────────
        echoFIDs = np.zeros((nEcho, nT), dtype=complex)
        echoAmps = np.zeros(nEcho)

        # ── 90° excitation pulse — ONCE, state then propagated echo by echo ───────
        # We need a clean starting state after 90° for ALL echoes.
        # Since each echo uses a DIFFERENT τ we cannot share a single evolved state.
        # → Reset to thermal eq, apply 90°, then independently evolve each echo.

        for echoIdx in range(nEcho):
            tau = self.tauVec[echoIdx]

            # ── Reset to thermal equilibrium ──────────────────────────────────────
            stp.reset()
            stp.update_wshift(wShiftvec)

            # ── Phase factors for this τ ──────────────────────────────────────────
            cftau_pf = stp.add_phase_factors(all_cf_tau[echoIdx])

            # ── 90° excitation pulse (phase = 0°) ────────────────────────────────
            stp.cachedflipangle = cached_flip90
            stp.cachedwigner = cached_wig90
            stp.pulsebatch(flip90vec, 0)

            # ── First τ interval (90° → 180°) ─────────────────────────────────────
            stp.relax_endpoint_cached(cftau_pf)

            # ── 180° refocusing pulse (phase = 90°, standard phase cycle) ─────────
            stp.cachedflipangle = cached_flip180
            stp.cachedwigner = cached_wig180
            stp.pulsebatch(flip180vec, 90)

            # ── Second τ interval (180° → echo centre) ────────────────────────────
            #stp.relax_endpoint_cached(cftau_pf)

            # ── Acquire FID directly after 180 Pulse ───────────────────────────────
            relaxT11 = np.sum(stp.relax_T11_cached(cffid_pf), axis=0) *  np.exp(1j * np.pi) #sum over spins
            echoFIDs[echoIdx, :] = relaxT11
            echoAmps[echoIdx] = np.amax(np.real(relaxT11))




            if verbose:
                print(f"  echo {echoIdx + 1:3d}/{nEcho}  τ={tau * 1e3:.3f} ms"
                      f"  TE={2 * tau * 1e3:.3f} ms  amp={echoAmps[echoIdx]:.4f}")

        #echoFIDs = getValues.switchReImFID(echoFIDs)

        return acqTimeVec, echoFIDs, echoAmps, self.tauVec
    # ------------------------------------------------------------------
    # TQTPPI (with 180°)
    # ------------------------------------------------------------------
    def TQTPPIWith180(self):
        """
        Non-fixed TQTPPI with 180° refocusing pulse.
        Same structure as TQTPPIfixWith180dz but tevo increments
        by self.tevoStep at every phase cycle step.
        State is RESET each phase cycle (spins resampled), matching
        the fixed-sequence style.
        """
        acqTimeVec = (self.deadtimeFID +
                      self.dwelltimeFID * np.arange(self.dataPoints))
        nAlpha = len(self.alphas)
        nFIDs = self.NumPhaseCycles * nAlpha
        nT = len(acqTimeVec)

        fAngles90 = self.flip90 * np.ones(self.nSpins)
        verbose = False

        # ── Wigner D cache — warm up BOTH 90° and 180°, ONCE ─────────────────────
        warm = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        warm.pulsebatch(self.flip90, 0)
        cached_flip90 = warm.cachedflipangle
        cached_wig90 = [w.copy() for w in warm.cachedwigner]

        warm.pulsebatch(self.flip180, 0)
        cached_flip180 = warm.cachedflipangle
        cached_wig180 = [w.copy() for w in warm.cachedwigner]

        # ── Fixed-duration caches — precomputed ONCE ──────────────────────────────
        ts_mix = np.linspace(0, self.tmix, self.nPtsmix)
        cfmix = warm.precompute_fJen(ts_mix)
        cffid = warm.precompute_fJen(acqTimeVec)

        # ── tevo grid — precompute ALL unique tevo/2 values upfront ──────────────
        # tevo[idx] = tevo0 + idx * tevoStep  for idx = 0..nFIDs-1
        # Precompute one cf_half per unique tevo so the inner loop stays cache-only
        all_tevos = self.tevo0 + np.arange(nFIDs) * self.tevoStep  # (nFIDs,)
        all_ts_half = [np.linspace(0, t / 2, self.nPtsevo) for t in all_tevos]
        all_cf_half = [warm.precompute_fJen(ts) for ts in all_ts_half]

        # ── Two reusable state objects ────────────────────────────────────────────
        stp1 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        stp2 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        # Propagate ones arrays
        warm.preallocate_ones(self.nPtsevo, self.nPtsmix, nT)
        stp1.onesevo = warm.onesevo;  stp2.onesevo = warm.onesevo
        stp1.onesmix = warm.onesmix;  stp2.onesmix = warm.onesmix
        stp1.onesfid = warm.onesfid;  stp2.onesfid = warm.onesfid

        FIDsbeta = np.zeros((nFIDs, nT, 2), dtype=complex)
        tevos = np.zeros(nFIDs)

        # ── Outer loop: resample spins each phase cycle ───────────────────────────
        for j in range(self.NumPhaseCycles):
            wShiftvec = (self.FreqShift +
                         self.wShiftdist.rvs(self.nSpins) if self.wShiftdist is not None
                         else self.FreqShift * np.ones(self.nSpins))

            if self.varFlip90 and self.flipShiftdist90 is not None:
                flip1vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip2vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip3vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                fAngles90 = flip1vec
            else:
                flip1vec = self.flip90
                flip2vec = self.flip90
                flip3vec = self.flip90

            stp1.update_wshift(wShiftvec)
            stp2.update_wshift(wShiftvec)

            # Phase factors for fixed caches — once per j
            cfmix_pf = stp1.add_phase_factors(cfmix)
            cffid_pf = stp1.add_phase_factors(cffid)

            # ── Inner loop: alpha steps ───────────────────────────────────────────
            for alphaIdx, alpha in enumerate(self.alphas):
                idx = j * nAlpha + alphaIdx
                tevos[idx] = all_tevos[idx]

                # Phase factors for this step's tevo — add wShift for current j
                cfhalf_pf = stp1.add_phase_factors(all_cf_half[idx])

                stp1.reset(); stp1.update_wshift(wShiftvec)
                stp2.reset(); stp2.update_wshift(wShiftvec)

                # ── Pulse 1 (90°, alpha) ──────────────────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip1vec, alpha)
                stp2.pulsebatch(flip1vec, alpha)

                # ── Evolution first half (tevo/2) ─────────────────────────────────
                stp1.relax_endpoint_cached(cfhalf_pf)
                stp2.relax_endpoint_cached(cfhalf_pf)

                # ── 180° pulse (phase = alpha + 90°) ─────────────────────────────
                stp1.cachedflipangle = cached_flip180; stp1.cachedwigner = cached_wig180
                stp2.cachedflipangle = cached_flip180; stp2.cachedwigner = cached_wig180
                stp1.pulsebatch(self.flip180, alpha + 90)
                stp2.pulsebatch(self.flip180, alpha + 90)

                # ── Evolution second half (tevo/2) ────────────────────────────────
                stp1.relax_endpoint_cached(cfhalf_pf)
                stp2.relax_endpoint_cached(cfhalf_pf)

                # ── Pulse 2 (90°) — branches diverge ─────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip2vec, alpha + 90)
                stp2.pulsebatch(flip2vec, alpha - 90)

                # ── Mixing ────────────────────────────────────────────────────────
                stp1.relax_endpoint_cached(cfmix_pf)
                stp2.relax_endpoint_cached(cfmix_pf)

                # ── Pulse 3 (90°, phase=0°) ───────────────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip3vec, 0)
                stp2.pulsebatch(flip3vec, 0)

                # ── Acquisition ───────────────────────────────────────────────────
                FIDsbeta[idx, :, 0] = np.sum(stp1.relax_T1m1_cached(cffid_pf), axis=0)
                FIDsbeta[idx, :, 1] = np.sum(stp2.relax_T1m1_cached(cffid_pf), axis=0)

                if verbose and idx % 160 == 0:
                    print(f"  phase step {idx:3d}/{nFIDs}")

        FIDs = getValues.switchReImFID(FIDsbeta[:, :, 0] + FIDsbeta[:, :, 1])
        mqFID = np.real(np.squeeze(np.sum(FIDs, axis=1)))
        mqSpectrum = np.fft.fftshift(np.fft.fft(mqFID))
        return acqTimeVec, mqSpectrum, mqFID, FIDs, fAngles90, tevos


    def TQTPPIwo180(self):
        """
        Non-fixed TQTPPI without 180° refocusing pulse.
        Same structure as TQTPPIWith180 but with a single full tevo
        evolution instead of two tevo/2 halves.
        tevo increments by self.tevoStep at every phase cycle step.
        """
        acqTimeVec = (self.deadtimeFID +
                      self.dwelltimeFID * np.arange(self.dataPoints))
        nAlpha = len(self.alphas)
        nFIDs = self.NumPhaseCycles * nAlpha
        nT = len(acqTimeVec)

        fAngles90 = self.flip90 * np.ones(self.nSpins)
        verbose = False

        # ── Wigner D cache — 90° only ─────────────────────────────────────────────
        warm = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        warm.pulsebatch(self.flip90, 0)
        cached_flip90 = warm.cachedflipangle
        cached_wig90 = [w.copy() for w in warm.cachedwigner]

        # ── Fixed-duration caches — precomputed ONCE ──────────────────────────────
        ts_mix = np.linspace(0, self.tmix, self.nPtsmix)
        cfmix = warm.precompute_fJen(ts_mix)
        cffid = warm.precompute_fJen(acqTimeVec)

        # ── tevo grid — precompute ALL unique tevo values upfront ─────────────────
        all_tevos = self.tevo0 + np.arange(nFIDs) * self.tevoStep  # (nFIDs,)
        all_ts_evo = [np.linspace(0, t, self.nPtsevo) for t in all_tevos]
        all_cf_evo = [warm.precompute_fJen(ts) for ts in all_ts_evo]

        # ── Two reusable state objects ────────────────────────────────────────────
        stp1 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        stp2 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        # Propagate ones arrays
        warm.preallocate_ones(self.nPtsevo, self.nPtsmix, nT)
        stp1.onesevo = warm.onesevo; stp2.onesevo = warm.onesevo
        stp1.onesmix = warm.onesmix; stp2.onesmix = warm.onesmix
        stp1.onesfid = warm.onesfid; stp2.onesfid = warm.onesfid

        FIDsbeta = np.zeros((nFIDs, nT, 2), dtype=complex)
        tevos = np.zeros(nFIDs)

        # ── Outer loop: resample spins each phase cycle ───────────────────────────
        for j in range(self.NumPhaseCycles):
            wShiftvec = (self.FreqShift +
                         self.wShiftdist.rvs(self.nSpins) if self.wShiftdist is not None
                         else self.FreqShift * np.ones(self.nSpins))

            if self.varFlip90 and self.flipShiftdist90 is not None:
                flip1vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip2vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip3vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                fAngles90 = flip1vec
            else:
                flip1vec = self.flip90
                flip2vec = self.flip90
                flip3vec = self.flip90

            stp1.update_wshift(wShiftvec)
            stp2.update_wshift(wShiftvec)

            # Phase factors for fixed caches — once per j
            cfmix_pf = stp1.add_phase_factors(cfmix)
            cffid_pf = stp1.add_phase_factors(cffid)

            # ── Inner loop: alpha steps ───────────────────────────────────────────
            for alphaIdx, alpha in enumerate(self.alphas):
                idx = j * nAlpha + alphaIdx
                tevos[idx] = all_tevos[idx]

                # Phase factors for this step's tevo
                cfevo_pf = stp1.add_phase_factors(all_cf_evo[idx])

                stp1.reset();
                stp1.update_wshift(wShiftvec)
                stp2.reset();
                stp2.update_wshift(wShiftvec)

                # ── Pulse 1 (90°, alpha) ──────────────────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip1vec, alpha)
                stp2.pulsebatch(flip1vec, alpha)

                # ── Full evolution (tevo) ─────────────────────────────────────────
                stp1.relax_endpoint_cached(cfevo_pf)
                stp2.relax_endpoint_cached(cfevo_pf)

                # ── Pulse 2 (90°) — branches diverge ─────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip2vec, alpha + 90)
                stp2.pulsebatch(flip2vec, alpha - 90)

                # ── Mixing ────────────────────────────────────────────────────────
                stp1.relax_endpoint_cached(cfmix_pf)
                stp2.relax_endpoint_cached(cfmix_pf)

                # ── Pulse 3 (90°, phase=0°) ───────────────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip3vec, 0)
                stp2.pulsebatch(flip3vec, 0)

                # ── Acquisition ───────────────────────────────────────────────────
                FIDsbeta[idx, :, 0] = np.sum(stp1.relax_T11_cached(cffid_pf), axis=0)
                FIDsbeta[idx, :, 1] = np.sum(stp2.relax_T11_cached(cffid_pf), axis=0)

                if verbose and idx % 160 == 0:
                    print(f"  phase step {idx:3d}/{nFIDs}")

        FIDs = getValues.switchReImFID(FIDsbeta[:, :, 0] + FIDsbeta[:, :, 1])
        mqFID = np.squeeze(np.sum(FIDs, axis=1))
        mqSpectrum = np.fft.fftshift(np.fft.fft(mqFID))
        return acqTimeVec, mqSpectrum, mqFID, fAngles90, tevos
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

        verbose = False  # simplified (MATLAB checks getCurrentTask)

        # Wigner D cache — computed ONCE
        warm = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
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
        stp1 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        stp2 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        for sp in [stp1, stp2]:
            sp.cachedflipangle = cachedflip
            sp.cachedwigner = cachedwig
            sp.onesevo = warm.onesevo
            sp.onesmix = warm.onesmix
            sp.onesfid = warm.onesfid

        FIDsbeta = np.zeros((nFIDs, nT, 2), dtype=complex)
        FIDsbeta_m = np.zeros((nFIDs, nT, 2), dtype=complex)
        for j in range(self.NumPhaseCycles):
            wShiftvec = (self.FreqShift +
                         self.wShiftdist.rvs(self.nSpins) if self.wShiftdist is not None
                         else self.FreqShift * np.ones(self.nSpins))


            if self.varFlip90 and self.flipShiftdist90 is not None:

                if isinstance(self.flipShiftdist90, str):
                    self.flipShiftdist90 = getValues.getFlipShiftDist90(
                        self.flipShiftdist90, 0.0, self.flipShiftSig90
                    )
                flip1vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip2vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip3vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                fAngles90 = flip1vec
            else:
                flip1vec = self.flip90
                flip2vec = self.flip90
                flip3vec = self.flip90

            stp1.update_wshift(wShiftvec)
            stp2.update_wshift(wShiftvec)

            #Phase factors ONCE per j
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

                """  st1_p = copy.deepcopy(stp1)
                st1_m = copy.deepcopy(stp1)
                st2_p = copy.deepcopy(stp2)
                st2_m = copy.deepcopy(stp2)

                #fid1_p = np.sum(st1_p.relax_T11_cached(cffid_ph), axis=0)
                fid1_m = np.sum(st1_m.relax_T1m1_cached(cffid_ph), axis=0)
                #fid2_p = np.sum(st2_p.relax_T11_cached(cffid_ph), axis=0)
                fid2_m = np.sum(st2_m.relax_T1m1_cached(cffid_ph), axis=0)

                # Acquistion T_1,-1 channel
                FIDsbeta_m[idx, :, 0] = fid1_m
                FIDsbeta_m[idx, :, 1] = fid2_m"""

                # Acquisition T_{1,1} channel
                FIDsbeta[idx, :, 0] = np.sum(stp1.relax_T11_cached(cffid_ph), axis=0)
                FIDsbeta[idx, :, 1] = np.sum(stp2.relax_T11_cached(cffid_ph), axis=0)




                if verbose and idx % 160 == 0:
                    print(f"  phase step {idx+1}/{nFIDs}")
        #FIDs_m = getValues.switchReImFID(FIDsbeta_m[:,:,0]+FIDsbeta_m[:,:,1])

        FIDs = getValues.switchReImFID(FIDsbeta[:, :, 0] + FIDsbeta[:, :, 1])
        mqFID = np.squeeze(np.sum(FIDs, axis=1))
        mqSpectrum = np.fft.fftshift(np.fft.fft(mqFID))
        return acqTimeVec, mqSpectrum, FIDs, fAngles90

    def TQTPPIfixWith180dz(self):
        acqTimeVec = (self.deadtimeFID +
                      self.dwelltimeFID * np.arange(self.dataPoints))
        nAlpha = len(self.alphas)
        nFIDs = self.NumPhaseCycles * nAlpha
        nT = len(acqTimeVec)

        fAngles90 = self.flip90 * np.ones(self.nSpins)
        verbose = False

        # ── Wigner D cache — warm up BOTH 90° and 180°, ONCE ─────────────────────
        warm = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        warm.pulsebatch(self.flip90, 0)
        cached_flip90 = warm.cachedflipangle
        cached_wig90 = [w.copy() for w in warm.cachedwigner]  # ← deep copy

        warm.pulsebatch(self.flip180, 0)
        cached_flip180 = warm.cachedflipangle
        cached_wig180 = [w.copy() for w in warm.cachedwigner]  # ← deep copy

        # ── fJen precomputation — ONCE ────────────────────────────────────────────
        ts_evo_half = np.linspace(0, self.tevo / 2, self.nPtsevo)
        ts_mix = np.linspace(0, self.tmix, self.nPtsmix)
        cf_evo_half = warm.precompute_fJen(ts_evo_half)
        cfmix = warm.precompute_fJen(ts_mix)
        cffid = warm.precompute_fJen(acqTimeVec)

        # ── Two reusable state objects ────────────────────────────────────────────
        stp1 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)
        stp2 = TmnEvoVecDic(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
                            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID)

        # Propagate ones arrays to both state objects
        warm.preallocate_ones(self.nPtsevo, self.nPtsmix, nT)
        stp1.onesevo = warm.onesevo
        stp1.onesmix = warm.onesmix
        stp1.onesfid = warm.onesfid
        stp2.onesevo = warm.onesevo
        stp2.onesmix = warm.onesmix
        stp2.onesfid = warm.onesfid

        FIDsbeta = np.zeros((nFIDs, nT, 2), dtype=complex)

        # ── Outer loop: resample spins each phase cycle ───────────────────────────
        for j in range(self.NumPhaseCycles):
            wShiftvec = (self.FreqShift +
                         self.wShiftdist.rvs(self.nSpins) if self.wShiftdist is not None
                         else self.FreqShift * np.ones(self.nSpins))

            if self.varFlip90 and self.flipShiftdist90 is not None:
                flip1vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip2vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip3vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                fAngles90 = flip1vec
            else:
                flip1vec = self.flip90
                flip2vec = self.flip90
                flip3vec = self.flip90

            stp1.update_wshift(wShiftvec)
            stp2.update_wshift(wShiftvec)

            # Phase factors: once per j
            cfhalf_pf = stp1.add_phase_factors(cf_evo_half)
            cfmix_pf = stp1.add_phase_factors(cfmix)
            cffid_pf = stp1.add_phase_factors(cffid)

            # ── Inner loop: alpha steps ───────────────────────────────────────────
            for alphaIdx, alpha in enumerate(self.alphas):
                idx = j * nAlpha + alphaIdx

                stp1.reset();
                stp1.update_wshift(wShiftvec)
                stp2.reset();
                stp2.update_wshift(wShiftvec)

                # ── Pulse 1 (90°, alpha) — identical for both branches ────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip1vec, alpha)
                stp2.pulsebatch(flip1vec, alpha)

                # ── Evolution first half (tevo/2) ─────────────────────────────────
                stp1.relax_endpoint_cached(cfhalf_pf)
                stp2.relax_endpoint_cached(cfhalf_pf)

                # ── 180° pulse (phase = alpha+90°) — identical for both ───────────
                stp1.cachedflipangle = cached_flip180; stp1.cachedwigner = cached_wig180
                stp2.cachedflipangle = cached_flip180; stp2.cachedwigner = cached_wig180
                stp1.pulsebatch(self.flip180, alpha + 90)
                stp2.pulsebatch(self.flip180, alpha + 90)

                # ── Evolution second half (tevo/2) ────────────────────────────────
                stp1.relax_endpoint_cached(cfhalf_pf)
                stp2.relax_endpoint_cached(cfhalf_pf)

                # ── Pulse 2 (90°) — DQ filter -------─────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip2vec, alpha + 90)
                stp2.pulsebatch(flip2vec, alpha - 90)

                # ── Mixing ────────────────────────────────────────────────────────
                stp1.relax_endpoint_cached(cfmix_pf)
                stp2.relax_endpoint_cached(cfmix_pf)

                # ── Pulse 3 (90°, phase=0°) ───────────────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip3vec, 0)
                stp2.pulsebatch(flip3vec, 0)

                # ── Acquisition ───────────────────────────────────────────────────
                FIDsbeta[idx, :, 0] = np.sum(stp1.relax_T11_cached(cffid_pf), axis=0)
                FIDsbeta[idx, :, 1] = np.sum(stp2.relax_T11_cached(cffid_pf), axis=0)

                if verbose and idx % 160 == 0:
                    print(f"  phase step {idx:3d}/{nFIDs}")

        FIDs = getValues.switchReImFID(FIDsbeta[:, :, 0] + FIDsbeta[:, :, 1])
        mqFID = np.squeeze(np.sum(FIDs, axis=1))
        mqSpectrum = np.fft.fftshift(np.fft.fft(mqFID))
        return acqTimeVec, mqSpectrum, FIDs, fAngles90

    def IRTQTPPI_sr(self, skip=True):
        """
        Inversion-Recovery TQTPPI based on TQTPPI.

        Sequence per phase step:
            180(0) -> tevo -> 90(beta = alpha ± 135) -> tmix -> 90(0) -> acquire
            This would be PC2 from Simon ...

        Differences vs old MATLAB sr version:
            - state is reset for every phase step
            - no TR carry-over between steps
            - relaxation during acquisition is handled by relax_T11_cached(...)
            - tevo increments by self.tevoStep each step
        """
        acqTimeVec = self.deadtimeFID + self.dwelltimeFID * np.arange(self.dataPoints)
        nAlpha = len(self.alphas)
        nFIDs = self.NumPhaseCycles * nAlpha
        nT = len(acqTimeVec)
        verbose = False

        # ── Warm-up Wigner caches ──────────────────────────────────────────────────
        warm = TmnEvoVecDic(
            self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID
        )

        warm.pulsebatch(self.flip90, 0)
        cached_flip90 = warm.cachedflipangle
        cached_wig90 = [w.copy() for w in warm.cachedwigner]

        warm.pulsebatch(self.flip180, 0)
        cached_flip180 = warm.cachedflipangle
        cached_wig180 = [w.copy() for w in warm.cachedwigner]

        # ── Fixed caches ───────────────────────────────────────────────────────────
        ts_mix = np.linspace(0, self.tmix, self.nPtsmix)
        cfmix = warm.precompute_fJen(ts_mix)
        cffid = warm.precompute_fJen(acqTimeVec)

        # ── tevo grid ──────────────────────────────────────────────────────────────
        if skip:


            x1 = np.arange(1,nFIDs*5+1)

            # 2. Extract subsets and concatenate (MATLAB indexing is 1-based, Python is 0-based)
            half_len = int(np.floor(nFIDs / 2))
            # MATLAB x1(1:half_len) -> Python x1[0:half_len]
            # MATLAB x1(half_len+9:9:end) -> Python x1[half_len+8::9] (8-based offset for 9th element due to 0-indexing)
            x = np.concatenate([x1[0:half_len], x1[half_len + 8:: 9]])

            # 3. Apply transformation similar to the MATLAB script
            x = x.astype(float)  # Ensure float type for calculations if needed
            x[0] = self.tevo0
            x[1:] = (x[1:] - 1) * self.tevoStep + x[0]
            tevos = x
            all_ts_evo = [np.linspace(0, t, self.nPtsevo) for t in x]
            all_cf_evo = [warm.precompute_fJen(ts) for ts in all_ts_evo]


        else:
            tevos = self.tevo0 + np.arange(nFIDs) * self.tevoStep
            all_ts_evo = [np.linspace(0, t, self.nPtsevo) for t in tevos]
            all_cf_evo = [warm.precompute_fJen(ts) for ts in all_ts_evo]

        # ── Reusable state objects ─────────────────────────────────────────────────
        stp1 = TmnEvoVecDic(
            self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID
        )
        stp2 = TmnEvoVecDic(
            self.B0, self.tauC, self.wQ, self.wQbar, self.Jen,
            np.zeros(self.nSpins), self.wShiftRMS, self.wShiftFID
        )

        warm.preallocate_ones(self.nPtsevo, self.nPtsmix, nT)
        stp1.onesevo = warm.onesevo; stp2.onesevo = warm.onesevo
        stp1.onesmix = warm.onesmix; stp2.onesmix = warm.onesmix
        stp1.onesfid = warm.onesfid; stp2.onesfid = warm.onesfid

        FIDs_p1 = np.zeros((nFIDs, nT), dtype=complex)
        FIDs_p2 = np.zeros((nFIDs, nT), dtype=complex)
        TQs_p1 = np.zeros(nFIDs, dtype=complex)
        TQs_p2 = np.zeros(nFIDs, dtype=complex)
        SQs_p1 = np.zeros(nFIDs, dtype=complex)
        SQs_p2 = np.zeros(nFIDs, dtype=complex)

        # ── Outer loop: resample spins each phase cycle ────────────────────────────
        for j in range(self.NumPhaseCycles):
            wShiftvec = (
                self.FreqShift + self.wShiftdist.rvs(self.nSpins)
                if self.wShiftdist is not None
                else self.FreqShift * np.ones(self.nSpins)
            )

            if self.varFlip90 and self.flipShiftdist90 is not None:
                flip180_1_vec = self.flip180 + self.flipShiftdist90.rvs(self.nSpins)
                flip90_2_vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
                flip90_3_vec = self.flip90 + self.flipShiftdist90.rvs(self.nSpins)
            else:
                flip180_1_vec = self.flip180
                flip90_2_vec = self.flip90
                flip90_3_vec = self.flip90

            stp1.update_wshift(wShiftvec)
            stp2.update_wshift(wShiftvec)

            cfmix_pf = stp1.add_phase_factors(cfmix)
            cffid_pf = stp1.add_phase_factors(cffid)

            # ── Inner loop: alpha steps ────────────────────────────────────────────
            for alphaIdx, alpha in enumerate(self.alphas):
                idx = j * nAlpha + alphaIdx

                alpha_p1 = 0.0
                alpha_p2 = 0.0
                beta_p1 = alpha + 135.0
                beta_p2 = alpha - 135.0

                cfevo_pf = stp1.add_phase_factors(all_cf_evo[idx])

                # Reset both branches for this phase step
                stp1.reset(); stp1.update_wshift(wShiftvec)
                stp2.reset(); stp2.update_wshift(wShiftvec)

                # ── Pulse 1: 180° at 0° ────────────────────────────────────────────
                stp1.cachedflipangle = cached_flip180; stp1.cachedwigner = cached_wig180
                stp2.cachedflipangle = cached_flip180; stp2.cachedwigner = cached_wig180
                stp1.pulsebatch(flip180_1_vec, alpha_p1)
                stp2.pulsebatch(flip180_1_vec, alpha_p2)

                # ── Evolution ───────────────────────────────────────────────────────
                stp1.relax_endpoint_cached(cfevo_pf)
                stp2.relax_endpoint_cached(cfevo_pf)

                # Diagnostics after tevo
                #TQs_p1[idx] = np.sum(stp1.get_Tmnbatch(3, 3) + stp1.get_Tmnbatch(3, -3))
                #TQs_p2[idx] = np.sum(stp2.get_Tmnbatch(3, 3) + stp2.get_Tmnbatch(3, -3))
                #SQs_p1[idx] = np.sum(stp1.get_Tmnbatch(1, 1) + stp1.get_Tmnbatch(1, -1))
                #SQs_p2[idx] = np.sum(stp2.get_Tmnbatch(1, 1) + stp2.get_Tmnbatch(1, -1))


                # ── Pulse 2: 90° at beta = alpha ± 135° ───────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip90_2_vec, beta_p1)
                stp2.pulsebatch(flip90_2_vec, beta_p2)

                # ── Mixing ──────────────────────────────────────────────────────────
                stp1.relax_endpoint_cached(cfmix_pf)
                stp2.relax_endpoint_cached(cfmix_pf)

                # ── Pulse 3: 90° at 0° ─────────────────────────────────────────────
                stp1.cachedflipangle = cached_flip90; stp1.cachedwigner = cached_wig90
                stp2.cachedflipangle = cached_flip90; stp2.cachedwigner = cached_wig90
                stp1.pulsebatch(flip90_3_vec, 0.0)
                stp2.pulsebatch(flip90_3_vec, 0.0)

                # ── Acquisition ────────────────────────────────────────────────────
                FIDs_p1[idx, :] = np.sum(stp1.relax_T11_cached(cffid_pf), axis=0)
                FIDs_p2[idx, :] = np.sum(stp2.relax_T11_cached(cffid_pf), axis=0)


                if verbose and idx % 160 == 0:
                    print(f"  phase step {idx:3d}/{nFIDs}, tevo = {tevos[idx] * 1e3:.3f} ms")

        FIDs_p1 = getValues.switchReImFID(FIDs_p1)
        FIDs_p2 = getValues.switchReImFID(FIDs_p2)
        mqFID = np.sum(FIDs_p1 + FIDs_p2,1) * np.exp(1j*np.pi)
        mqSpectrum = np.fft.fftshift(np.fft.fft(mqFID))
        return mqFID,  mqSpectrum, tevos, TQs_p1, TQs_p2, SQs_p1, SQs_p2





    # ------------------------------------------------------------------
    # Static / class-level dictionary methods
    # ------------------------------------------------------------------



    @staticmethod
    def make_param_grid(T2fvec, T2svec, FWHMvec):
        """Create valid parameter grid (T2f, T2s, FWHM)."""
        T2fg, T2sg, FWHMg = np.meshgrid(T2fvec, T2svec, FWHMvec, indexing='ij')
        valid = T2sg >= 2 * T2fg
        params = np.column_stack([T2fg[valid], T2sg[valid], FWHMg[valid]])
        print(f"Parameter grid: {params.shape[0]} valid combinations")
        return params

