"""
getValues.py
Helper class to calculate values like spectral density functions, relaxation rates,
Jen model parameters, Wigner D-matrix elements, and other NMR constants.
Mirrors MATLAB classdef getValues (static methods only).
"""

import numpy as np
from scipy.optimize import fmin, least_squares
from scipy.stats import norm as scipy_norm
import numpy as np
from scipy.stats import rv_continuous, norm, cauchy
import math


class getValues:
    # Gyromagnetic ratio for 23Na [Hz/T]
    Nagamma_reduced = 11.262 * 1e6  # rad/(s·T)
    Nagamma = 11.262*1e6 #2 * np.pi  # 1/(T·s) — same value, different label

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    @staticmethod
    def getRandomAB(a, b, N):
        """Get N random numbers in interval [a, b]."""
        return a + (b - a) * np.random.rand(N, 1)

    @staticmethod
    def getwPnorm(vec, w, p):
        """Weighted p-norm of vector."""
        return np.linalg.norm(vec * w, p)

    # -------------------------------------------------------------------------
    # Larmor frequency
    # -------------------------------------------------------------------------
    @staticmethod
    def getw0(B0):
        """Calculate Larmor frequency for given B0 [T]."""
        return getValues.Nagamma * B0

    # -------------------------------------------------------------------------
    # Correlation time distribution
    # -------------------------------------------------------------------------
    @staticmethod
    def getptauC(tauCs, a, b, tcm):
        """
        Calculate p(tauC) for asymmetric log-Gauss distribution (Rooney et al. 1991).
        """
        Norm = 2 * a * b / (np.sqrt(np.pi) * (a + b) * tauCs)
        ptauCs = np.where(
            tauCs <= tcm,
            Norm * np.exp(-a**2 * np.log(tauCs / tcm)**2),
            (Norm * tauCs / tcm) * np.exp(-b**2 * np.log(tauCs / tcm)**2),
        )
        return ptauCs

    @staticmethod
    def getwQs(tauCs, tauC0, wQ0, wQ1):
        """Step function for wQ(tauC)."""
        wQs = np.ones(len(tauCs)) * wQ0
        wQs[tauCs > tauC0] = wQ1
        return wQs

    # -------------------------------------------------------------------------
    # Log range
    # -------------------------------------------------------------------------
    @staticmethod
    def getlogRange(startExp, endExp, stepSize):
        """
        Log range of the form 10^startExp, ..., 10^endExp
        with stepSize sub-divisions per decade.
        """
        rtmp = np.arange(startExp, endExp)
        ttmp = 10.0 ** rtmp
        steps = np.arange(1, stepSize * 9 + 1) / (stepSize * 9)
        # build each decade
        chunks = [t * steps for t in ttmp]
        logRange = np.concatenate(chunks)
        logRange = np.append(logRange, 10.0 ** endExp)
        return logRange

    # -------------------------------------------------------------------------
    # Statistical distributions
    # -------------------------------------------------------------------------
    @staticmethod
    def getWshiftDist(distname: str, cen: float, sig: float,
                  eta: float = 0.5,
                  sig_g: float = 2.0,
                  sig_l: float = 1.0):
        """
        Return a scipy frozen distribution object.

        Parameters
        ----------
        distname : "normal" | "t location-scale" (and variants) | "cauchy" | "lognormal"
        cen      : location parameter (centre)
        sig      : scale parameter
        """
        from scipy.stats import norm, cauchy, lognorm

        distname_clean = distname.replace("-", " ").strip().lower()

        if distname_clean == "normal":
            return norm(loc=cen, scale=sig)

        elif distname_clean in ("tlocationscale", "t location scale", "t",
                                "cauchy"):
            # Cauchy = Student-t with df=1; scipy.stats.cauchy is the
            # standard parameterisation: loc=centre, scale=half-width
            return cauchy(loc=cen, scale=sig)

        elif distname_clean == "lognormal":
            # MATLAB lognormal: mu=cen, sigma=sig in log-space
            return lognorm(s=sig, scale=np.exp(cen))

        elif distname_clean in ("voigt", "pseudo voigt", "pseudo-voigt",
                                "pseudovoigt", "mixed"):
            pseudo_voigt = _PseudoVoigtGen(name="pseudo_voigt")
            if not (0.0 <= eta <= 1.0):
                raise ValueError(f"eta must be in [0, 1], got {eta}")
            if sig_g <= 0 or sig_l <= 0:
                raise ValueError("sig_g and sig_l must be positive")
            # rv_continuous applies loc/scale as:  Y = loc + scale * X
            # So we pass scale=sig and encode the two widths as shape params b, c
            return pseudo_voigt(loc=cen, scale=sig, a=eta, b=sig_g, c=sig_l)

        else:
            raise ValueError(f"Unknown distribution: '{distname}'")

    @staticmethod
    def getFlipShiftDist90(distname, cen, sig):
        """Same as getWshiftDist — flip angle distribution."""
        return getValues.getWshiftDist(distname, cen, sig)

    @staticmethod
    def getDistFWHM(dist):
        """Estimate FWHM and normalised PDF for a distribution."""
        x = np.linspace(-100, 100, 1000)
        pd = dist.pdf(x)
        pdnorm = pd / pd.max()
        pddiff = np.abs(pdnorm - 0.5)
        idx1 = np.argmin(pddiff[:500])
        idx2 = 500 + np.argmin(pddiff[500:])
        FWHM = np.abs(x[idx1] - x[idx2])
        return x, pdnorm, FWHM

    # -------------------------------------------------------------------------
    # Signal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def getCompSignal(times2, tevo, TmnPhase1, TmnPhase2):
        """Composite signal from two coherence pathways."""
        times1 = np.arange(0, tevo, tevo / 2048)
        CompTimes = np.concatenate([-np.flip(times1), times2])
        nPts = len(CompTimes)
        nSpins = TmnPhase1.shape[0]
        Signal = np.zeros((nSpins, nPts))
        for idx in range(nSpins):
            Signal[idx, :len(times1)] = TmnPhase1[idx]
            Signal[idx, len(times1):] = TmnPhase2[idx]
        Signalres = Signal.sum(axis=0)
        Signalnorm = Signalres / np.max(np.abs(Signalres))
        return CompTimes, Signalnorm

    @staticmethod
    def getTimes(tstart, tstop, datapoints):
        """Linearly spaced time vector from tstart to tstop (inclusive)."""
        return np.linspace(tstart, tstop, datapoints)

    @staticmethod
    def switchReImFID(FID):
        """Switch real/imaginary parts of FID."""
        FIDre = -np.imag(FID)
        FIDim = -np.real(FID)
        return FIDre + 1j * FIDim

    @staticmethod
    def SE(lb, ub, conflevel=95):
        """Convert confidence interval bounds to standard error."""
        if conflevel == 95:
            divider = 3.92
        elif conflevel == 99:
            divider = 5.15
        else:
            divider = 3.29
        return np.abs(lb - ub) / divider

    # -------------------------------------------------------------------------
    # Spectral density — Jen model (m can be scalar or array)
    # -------------------------------------------------------------------------
    @staticmethod
    def getJJen(m, tauC, wQ, Jen, w0):
        """Spectral density J_m and K_m for Jen model."""
        x = (w0 * tauC) ** 2
        Jm = wQ**2 / 5 * tauC / (1 + m**2 * x) + Jen
        Km = m * w0 * tauC * Jm
        return Jm, Km

    # -------------------------------------------------------------------------
    # Spectral density — isotropic (dk20)
    # -------------------------------------------------------------------------
    @staticmethod
    def getJ(m, tauC, wQ, w0):
        """Spectral density J_m and K_m for isotropic environment."""
        x = (w0 * tauC) ** 2
        Jm = wQ**2 / 5 * tauC / (1 + m**2 * x)
        Km = m * w0 * tauC * Jm
        return Jm, Km

    # -------------------------------------------------------------------------
    # Relaxation rates
    # -------------------------------------------------------------------------
    @staticmethod
    def getRsJen(q, tauC, wQ, wQbar, Jen, wShiftRMS, w0):
        """Relaxation rates for Jen model (anisotropic environment)."""
        J0, K0 = getValues.getJJen(0, tauC, wQ, Jen, w0)
        J1, K1 = getValues.getJJen(1, tauC, wQ, Jen, w0)
        J2, K2 = getValues.getJJen(2, tauC, wQ, Jen, w0)
        q = abs(q)
        if q == 0:
            R01 = 2 * J1
            R02 = 2 * J2
            R03 = 2 * J1 + 2 * J2
            Rs = np.array([R01, R02, R03])
        elif q == 1 or q == -1:
            R11 = J0 + J1 + J2 - np.sqrt(np.maximum(J2**2 - wQbar**2, 0)) + abs(q) * wShiftRMS
            R12 = J1 + J2 + abs(q) * wShiftRMS
            R13 = J0 + J1 + J2 + np.sqrt(np.maximum(J2**2 - wQbar**2, 0)) + abs(q) * wShiftRMS
            Rs = np.array([R11, R12, R13])
        elif q == 2 or q == -2:
            R21 = J0 + J1 + J2 + np.sqrt(np.maximum(J1**2 - wQbar**2, 0)) + abs(q) * wShiftRMS
            R22 = J0 + J1 + J2 - np.sqrt(np.maximum(J1**2 - wQbar**2, 0)) + abs(q) * wShiftRMS
            Rs = np.array([R21, R22])
        elif q == 3 or q == -3:
            R31 = J1 + J2 + abs(q) * wShiftRMS
            Rs = np.array([R31])
        else:
            raise ValueError(f"Unsupported coherence order q={q}")
        return Rs

    @staticmethod
    def getRs(q, tauC, wQ, w0):
        """Relaxation rates for isotropic environment."""
        J0, _ = getValues.getJ(0, tauC, wQ, w0)
        J1, _ = getValues.getJ(1, tauC, wQ, w0)
        J2, _ = getValues.getJ(2, tauC, wQ, w0)
        q = abs(q)
        if q == 0:
            Rs = np.array([2*J1, 2*J2, 2*J1+2*J2])
        elif q == 1:
            Rs = np.array([J0+J1, J1+J2, J0+J1+2*J2])
        elif q == 2:
            Rs = np.array([J0+2*J1+J2, J0+J2])
        elif q == 3:
            Rs = np.array([J1+J2])
        else:
            raise ValueError(f"Unsupported q={q}")
        return Rs

    @staticmethod
    def getRsaniso(q, tauC, wQ, wQbar, w0):
        """Relaxation rates for anisotropic environment."""
        J0, _ = getValues.getJ(0, tauC, wQ, w0)
        J1, _ = getValues.getJ(1, tauC, wQ, w0)
        J2, _ = getValues.getJ(2, tauC, wQ, w0)
        q = abs(q)
        if q == 0:
            Rs = np.array([2*J1, 2*J2, 2*J1+2*J2])
        elif q == 1:
            R11 = J0 + J1 + J2 - np.sqrt(np.maximum(J2**2 - wQbar**2, 0))
            R12 = J1 + J2
            R13 = J0 + J1 + J2 + np.sqrt(np.maximum(J2**2 - wQbar**2, 0))
            Rs = np.array([R11, R12, R13])
        elif q == 2:
            R21 = J0 + J1 + J2 + np.sqrt(np.maximum(J1**2 - wQbar**2, 0))
            R22 = J0 + J1 + J2 - np.sqrt(np.maximum(J1**2 - wQbar**2, 0))
            Rs = np.array([R21, R22])
        elif q == 3:
            Rs = np.array([J1 + J2])
        return Rs

    @staticmethod
    def getRsanisoptauC(q, a, b, tcm, wQ, wQbar, w0):
        """Relaxation rates for anisotropic environment with tauC distribution."""
        tauCs = getValues.getlogRange(-14, -6, 0.1)
        ptauCs = getValues.getptauC(tauCs, a, b, tcm)
        deltaTauC = np.diff(tauCs)[0]
        wQs = wQ * np.ones_like(tauCs)
        J0s, _ = getValues.getJ(0, tauCs, wQs, w0)
        J1s, _ = getValues.getJ(1, tauCs, wQs, w0)
        J2s, _ = getValues.getJ(2, tauCs, wQs, w0)
        J0 = np.sum(J0s * ptauCs * deltaTauC)
        J1 = np.sum(J1s * ptauCs * deltaTauC)
        J2 = np.sum(J2s * ptauCs * deltaTauC)
        q = abs(q)
        if q == 0:
            Rs = np.array([2*J1, 2*J2, 2*J1+2*J2])
        elif q == 1:
            R11 = J0 + J1 + J2 - np.sqrt(max(J2**2 - wQbar**2, 0))
            Rs = np.array([R11, J1+J2, J0+J1+J2+np.sqrt(max(J2**2-wQbar**2,0))])
        elif q == 2:
            Rs = np.array([J0+J1+J2+np.sqrt(max(J1**2-wQbar**2,0)),
                           J0+J1+J2-np.sqrt(max(J1**2-wQbar**2,0))])
        elif q == 3:
            Rs = np.array([J1+J2])
        return Rs

    @staticmethod
    def getRsanisoptauCwQ(q, a, b, tcm, wQ0, wQ1, tauC0, wQbar, w0):
        """Relaxation rates for anisotropic env with tauC and wQ distributions."""
        tauCs = getValues.getlogRange(-14, -6, 0.1)
        ptauCs = getValues.getptauC(tauCs, a, b, tcm)
        deltaTauC = np.diff(tauCs)[0]
        wQs = getValues.getwQs(tauCs, tauC0, wQ0, wQ1)
        J0s, _ = getValues.getJ(0, tauCs, wQs, w0)
        J1s, _ = getValues.getJ(1, tauCs, wQs, w0)
        J2s, _ = getValues.getJ(2, tauCs, wQs, w0)
        J0 = np.sum(J0s * ptauCs * deltaTauC)
        J1 = np.sum(J1s * ptauCs * deltaTauC)
        J2 = np.sum(J2s * ptauCs * deltaTauC)
        q = abs(q)
        if q == 0:
            Rs = np.array([2*J1, 2*J2, 2*J1+2*J2])
        elif q == 1:
            Rs = np.array([J0+J1+J2-np.sqrt(max(J2**2-wQbar**2,0)),
                           J1+J2,
                           J0+J1+J2+np.sqrt(max(J2**2-wQbar**2,0))])
        elif q == 2:
            Rs = np.array([J0+J1+J2+np.sqrt(max(J1**2-wQbar**2,0)),
                           J0+J1+J2-np.sqrt(max(J1**2-wQbar**2,0))])
        elif q == 3:
            Rs = np.array([J1+J2])
        return Rs

    @staticmethod
    def getRsZQSQJen(Jen, tauC, wQ, wQbar, wShiftRMS, w0):
        """Simultaneous SQ and ZQ relaxation rates for Jen model."""
        R0s = getValues.getRsJen(0, tauC, wQ, wQbar, Jen, wShiftRMS, w0)
        R1s = getValues.getRsJen(1, tauC, wQ, wQbar, Jen, wShiftRMS, w0)
        return np.concatenate([R0s[[0, 1]], R1s[[0, 1]]])  # R0[2], R0[3], R1[2], R1[3]

    # -------------------------------------------------------------------------
    # Relaxation times
    # -------------------------------------------------------------------------
    @staticmethod
    def getrelaxationTimesJen(tauC, wQ, wQbar, Jen, wShiftRMS, w0):
        """Relaxation times T1s, T1f, T2s, T2f for Jen model."""
        R0s = getValues.getRsJen(0, tauC, wQ, wQbar, Jen, wShiftRMS, w0)
        R1s = getValues.getRsJen(1, tauC, wQ, wQbar, Jen, wShiftRMS, w0)
        T1s = 1.0 / R0s[1]
        T1f = 1.0 / R0s[0]
        T2s = 1.0 / R1s[1]
        T2f = 1.0 / R1s[0]
        return T1s, T1f, T2s, T2f

    @staticmethod
    def getrelaxationTimes(tauC, wQ, w0):
        """Relaxation times based on dk20 definitions."""
        tauCs = np.atleast_1d(tauC)
        wQs = np.atleast_1d(wQ)
        x = (w0 * tauCs) ** 2
        a0 = (1 + 4*x) / (1 + x)
        b0 = 6 / 5 * x / tauCs * wQs**2 / ((4*x + 25*x + 1))
        a1 = (2 + 9*x + 4*x**2) / (2 + 5*x)
        b1 = 1 / 5 * tauCs * wQs**2 / (4*x + 1) / (4*x)
        T1f = (a0 - 1) / (a0 * b0)
        T1s = (a0 - 1) / b0
        T2f = (a1 - 1) / (a1 * b1)
        T2s = (a1 - 1) / b1
        return T1s, T1f, T2s, T2f

    @staticmethod
    def gettauOptT(Tif, Tis):
        """Optimal tau for TQTPPI or IRTQTPPI."""
        return np.log(Tis / Tif) / (1.0 / Tif - 1.0 / Tis)

    # -------------------------------------------------------------------------
    # Jen model fitting
    # -------------------------------------------------------------------------
    @staticmethod
    def getJenModel(T1f, T1s, T2f, T2s, w0):
        """
        Fit Jen, tauC, wQ, wShiftRMS from relaxation times using fminsearch.
        Returns Jen, tauC, wQ, wShiftRMS.
        """
        Tsmess = np.array([T1f, T1s, T2f, T2s])
        Rsmess = 1.0 / Tsmess
        weights = Tsmess
        p = 2

        def MinFun(JenVal):
            Rs_pred = getValues.getRsZQSQJen(JenVal[0], JenVal[1], JenVal[2], 0.0, JenVal[3], w0)
            return getValues.getwPnorm(Rs_pred - Rsmess, weights, p)

        x0 = np.array([10.0, 5e-8, 1e5, 0.0])
        result = fmin(MinFun, x0, disp=False)
        return result[0], result[1], result[2], result[3]

    # -------------------------------------------------------------------------
    # tauC / wQ inversion
    # -------------------------------------------------------------------------
    @staticmethod
    def gettauCwQ(T2f, T2s, w0):
        """Calculate tauC and wQ from T2f and T2s (dk20)."""
        R11 = 1.0 / np.atleast_1d(T2f)
        R12 = 1.0 / np.atleast_1d(T2s)
        a1 = R11 * R12
        b1 = R11 - R12
        b1[b1 == 0] = 0
        a1[b1 == 0] = 0
        tauC = np.real(1.0 / w0 * np.sqrt(
            1.0 / 8 - 9/5 * a1 + np.sqrt((9/5 * a1)**2 - (5*8) * a1 + 49)
        ))
        x = (w0 * tauC) ** 2
        wQ = np.sqrt(5 * b1 / (1 + 4*x) / (4*x) * tauC)
        wQ[np.isnan(wQ)] = 0
        return tauC, wQ

    @staticmethod
    def gettauCwQT1(T1f, T1s, w0):
        """Calculate tauC and wQ from T1f and T1s (dk20)."""
        R01 = 1.0 / np.atleast_1d(T1f)
        R02 = 1.0 / np.atleast_1d(T1s)
        a0 = R01 * R02
        b0 = R01 - R02
        b0[b0 == 0] = 0
        a0[b0 == 0] = 0
        tauC = np.real(1.0 / w0 * np.sqrt((a0 - 1.0) / (4.0 - a0)))
        x = (w0 * tauC) ** 2
        wQ = np.sqrt(b0 * 6/5 * x * tauC / (4*x * (25*x + 1)))
        wQ[np.isnan(wQ)] = 0
        return tauC, wQ

    # -------------------------------------------------------------------------
    # Relaxation functions f(t)
    # -------------------------------------------------------------------------
    @staticmethod
    def getf(q, t, tauC, wQ, w0):
        """Relaxation functions f_{q,kk'} for isotropic environment."""
        J0, _ = getValues.getJ(0, tauC, wQ, w0)
        J1, _ = getValues.getJ(1, tauC, wQ, w0)
        J2, _ = getValues.getJ(2, tauC, wQ, w0)
        q = abs(q)
        if q == 0:
            R01 = 2*J1; R02 = 2*J2; R03 = 2*J1+2*J2
            f011 = (1/5) * (np.exp(-R01*t) + 4*np.exp(-R02*t))
            f013 = (2/5) * (np.exp(-R01*t) - np.exp(-R02*t))
            f033 = (1/5) * (4*np.exp(-R01*t) + np.exp(-R02*t))
            f022 = np.exp(-R03*t)
            return f011, f013, f033, f022
        elif q == 1:
            R11 = J0+J1; R12 = J1+J2; R13 = J0+J1+2*J2
            f111 = (1/5) * (3*np.exp(-R11*t) + 2*np.exp(-R12*t))
            f113 = np.sqrt(6/5) * (np.exp(-R11*t) - np.exp(-R12*t))
            f133 = (1/5) * (2*np.exp(-R11*t) + 3*np.exp(-R12*t))
            f122 = np.exp(-R13*t)
            return f111, f113, f133, f122
        elif q == 2:
            R21 = J0+2*J1+J2; R22 = J0+J2
            f222 = np.exp(-R21*t)
            f233 = np.exp(-R22*t)
            return f222, f233
        elif q == 3:
            R31 = J1+J2
            f333 = np.exp(-R31*t)
            return (f333,)

    @staticmethod
    def getfaniso(q, t, tauC, wQ, wQbar, w0):
        """Relaxation functions for anisotropic environment."""
        J0, _ = getValues.getJ(0, tauC, wQ, w0)
        J1, _ = getValues.getJ(1, tauC, wQ, w0)
        J2, _ = getValues.getJ(2, tauC, wQ, w0)
        Rs = getValues.getRsaniso(q, tauC, wQ, wQbar, w0)
        sign_q = np.sign(q) if q != 0 else 1
        q = abs(q)
        if q == 0:
            R01, R02, R03 = Rs
            f011 = (1/5) * (np.exp(-R01*t) + 4*np.exp(-R02*t))
            f013 = (2/5) * (np.exp(-R01*t) - np.exp(-R02*t))
            f033 = (1/5) * (4*np.exp(-R01*t) + np.exp(-R02*t))
            f022 = np.exp(-R03*t)
            return f011, f013, f033, f022
        elif q == 1:
            R11, R12, R13 = Rs
            mu = J2 / np.sqrt(max(J2**2 - wQbar**2, 1e-100))
            v  = wQbar / np.sqrt(max(J2**2 - wQbar**2, 1e-100))
            f111 = (1/5) * ((3/2*(1+mu))*np.exp(-R11*t) + 2*np.exp(-R12*t) + (3/2*(1-mu))*np.exp(-R13*t))
            f122 = (1/2) * ((1-mu)*np.exp(-R11*t) + (1+mu)*np.exp(-R13*t))
            f133 = (1/5) * ((1+mu)*np.exp(-R11*t) + 3*np.exp(-R12*t) + (1-mu)*np.exp(-R13*t))  # approx
            f113 = np.sqrt(6/5) * ((1/2*(1+mu))*np.exp(-R11*t) - np.exp(-R12*t) + (1/2*(1-mu))*np.exp(-R13*t))
            f112 = (1j/2) * np.sqrt(3/5) * v * (sign_q*np.exp(-R11*t) - sign_q*np.exp(-R13*t))
            f123 = (1j) * np.sqrt(10) * v * (sign_q*np.exp(-R11*t) - sign_q*np.exp(-R13*t))
            return f111, f113, f133, f112, f123, f122
        elif q == 2:
            R21, R22 = Rs
            mu = J1 / np.sqrt(max(J1**2 - wQbar**2, 1e-100))
            v  = wQbar / np.sqrt(max(J1**2 - wQbar**2, 1e-100))
            f222 = (1/2) * ((1+mu)*np.exp(-R21*t) + (1-mu)*np.exp(-R22*t))
            f233 = (1/2) * ((1+mu)*np.exp(-R21*t) + (1-mu)*np.exp(-R22*t))
            f223 = (-1j/2) * v * (sign_q*np.exp(-R21*t) - sign_q*np.exp(-R22*t))
            return f222, f233, f223
        elif q == 3:
            R31 = Rs[0]
            f333 = np.exp(-R31*t)
            return (f333,)

    @staticmethod
    def getfJen(q, t, tauC, wQ, wQbar, Jen, wShiftRMS, w0):
        """Relaxation functions for Jen model (anisotropic)."""
        J0, K0 = getValues.getJJen(0, tauC, wQ, Jen, w0)
        J1, K1 = getValues.getJJen(1, tauC, wQ, Jen, w0)
        J2, K2 = getValues.getJJen(2, tauC, wQ, Jen, w0)
        Rs = getValues.getRsJen(q, tauC, wQ, wQbar, Jen, wShiftRMS, w0)
        sign_q = np.sign(q) if q != 0 else 1
        q_abs = abs(q)
        if q_abs == 0:
            R01, R02, R03 = Rs
            f011 = (1/5) * (np.exp(-R01*t) + 4*np.exp(-R02*t))
            f013 = (2/5) * (np.exp(-R01*t) - np.exp(-R02*t))
            f033 = (1/5) * (4*np.exp(-R01*t) + np.exp(-R02*t))
            f022 = np.exp(-R03*t)
            return f011, f013, f033, f022
        elif q_abs == 1:
            R11, R12, R13 = Rs
            mu = J2 / np.sqrt(np.maximum(J2**2 - wQbar**2, 1e-100))
            v  = wQbar / np.sqrt(np.maximum(J2**2 - wQbar**2, 1e-100))
            f111 = (1/5)*((3/2*(1+mu))*np.exp(-R11*t) + 2*np.exp(-R12*t) + (3/2*(1-mu))*np.exp(-R13*t))
            f122 = (1/2)*((1-mu)*np.exp(-R11*t) + (1+mu)*np.exp(-R13*t))
            f133 = (1/5)*((1+mu)*np.exp(-R11*t) + 3*np.exp(-R12*t) + (1-mu)*np.exp(-R11*t))
            f113 = (np.sqrt(6)/5)*((1/2*(1+mu))*np.exp(-R11*t) - np.exp(-R12*t) + (1/2*(1-mu))*np.exp(-R13*t))
            f112 = (1j/2)*np.sqrt(3/5)*v*(sign_q*np.exp(-R11*t) - sign_q*np.exp(-R13*t))
            f123 = (1j / np.sqrt(10)) * v *(sign_q*np.exp(-R11*t) - sign_q*np.exp(-R13*t))
            return f111, f113, f133, f112, f123, f122
        elif q_abs == 2:
            R21, R22 = Rs
            mu = J1 / np.sqrt(np.maximum(J1**2 - wQbar**2, 1e-100))
            v  = wQbar / np.sqrt(np.maximum(J1**2 - wQbar**2, 1e-100))
            f222 = (1/2)*((1+mu)*np.exp(-R21*t) + (1-mu)*np.exp(-R22*t))
            f233 = (1/2)*((1+mu)*np.exp(-R21*t) + (1-mu)*np.exp(-R22*t))
            f223 = (-1j/2)*v*(sign_q*np.exp(-R21*t) - sign_q*np.exp(-R22*t))
            return f222, f233, f223
        elif q_abs == 3:
            R31 = Rs[0]
            f333 = np.exp(-R31*t)
            return (f333,)

    # -------------------------------------------------------------------------
    # TQ signal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def getT31ts(ts, fids):
        """Extract TQ and SQ FIDs from phase-cycled FIDs."""
        NumTevoPoints = len(fids) // 2
        FIDwTQ = np.zeros(NumTevoPoints)
        FIDwoTQ = np.zeros(NumTevoPoints)
        tevos = ts[:NumTevoPoints]
        for icut in range(NumTevoPoints):
            fid = fids[icut + NumTevoPoints - icut:]
            fid4fft = fid.copy()
            fid4fft[0] = 0.5 * fid[0]
            spec = np.fft.fftshift(np.fft.fft(fid4fft))
            signalwTQ = np.sum(fid)
            signalwoTQ = np.max(np.abs(spec))
            FIDwTQ[icut] = np.abs(signalwTQ)
            FIDwoTQ[icut] = signalwoTQ
        fidwTQ = FIDwTQ / FIDwTQ[0]
        fidwoTQ = FIDwoTQ / FIDwoTQ[0]
        fidTQ = fidwTQ - fidwoTQ
        return fidwTQ, fidwoTQ, fidTQ, tevos

    @staticmethod
    def getT31norm(ts, SQ, TQ):
        """TQ normalisation factor from SQ curve fit."""
        from scipy.optimize import least_squares as lsq
        fitY = np.real(SQ) / np.max(np.real(SQ))
        fitX = ts

        def fun(x, t):
            return x[0]*np.exp(-t/x[1]) + x[2]*np.exp(-t/x[3]) + x[4]

        x0 = [0.4, 33e-3, 0.6, 10e-3, 0.0]
        bounds = ([0,0,0,0,-0.1], [2,70e-3,2,70e-3,0.1])
        res = lsq(lambda x: fun(x, fitX) - fitY, x0, bounds=bounds)
        X = res.x
        As, T2s, Af, T2f = X[0], X[1], X[2], X[3]
        sumSQ6040 = 2 * (0.4*T2s + 0.6*T2f)
        norm1 = 1 / (np.sqrt(6/5) * (T2s - T2f) * As * T2s * Af * T2f)
        norm2 = 1 / (np.sqrt(6/5) * (T2s - T2f) * sumSQ6040)
        norm3 = 1 / np.sum(TQ) * np.sum(TQ)
        return norm1, norm2, norm3, sumSQ6040

    @staticmethod
    def getT31normSQTQ(ts, SQ, TQ):
        """TQ normalisation from simultaneous SQ+TQ fit."""
        from scipy.optimize import least_squares as lsq
        fitY = np.column_stack([
            np.real(SQ) / np.max(np.abs(SQ)),
            np.real(TQ) / np.max(np.abs(TQ))
        ])
        fitX = ts

        def funSQ(x, t): return x[0]*np.exp(-t/x[1]) + x[2]*np.exp(-t/x[3]) + x[4]
        def funTQ(x, t): return x[5]*np.exp(-t/x[1]) - x[5]*np.exp(-t/x[3]) + x[6]

        def fun(x):
            return np.concatenate([funSQ(x, fitX) - fitY[:,0],
                                   funTQ(x, fitX) - fitY[:,1]])

        x0 = [0.4,33e-3,0.6,10e-3,0.0,2.0,0.0]
        bounds = ([0,0,0,0,-0.1,0,-0.1],[1,70e-3,1,70e-3,0.1,20,0.1])
        res = lsq(fun, x0, bounds=bounds)
        X = res.x
        As, T2s, Af, T2f = X[0], X[1], X[2], X[3]
        sumSQ6040 = 2 * (0.4*T2s + 0.6*T2f)
        norm1 = 1 / (np.sqrt(6/5) * (T2s - T2f) * As * T2s * Af * T2f)
        norm2 = 1 / (np.sqrt(6/5) * (T2s - T2f) * sumSQ6040)
        norm3 = 1 / np.sum(TQ) * np.sum(TQ)
        return norm1, norm2, norm3, sumSQ6040

    # -------------------------------------------------------------------------
    # Wigner D-matrix
    # -------------------------------------------------------------------------
    @staticmethod
    def getWignerD(j, mbar, m, theta):
        """Wigner small d-matrix element d^j_{mbar,m}(theta).
        Returns 0 immediately if |mbar| > j or |m| > j (unphysical).
        """
        if abs(mbar) > j or abs(m) > j:
            return 0.0
        smax = max(m - mbar, j - mbar, j + m)
        smax = max(smax, 0)
        djmm = 0.0
        for s in range(smax + 1):
            if (mbar - m + s) < 0: continue
            if (j + m - s) < 0: continue
            if (j - mbar - s) < 0: continue
            denom = (math.factorial(j + m - s) * math.factorial(s) *
                     math.factorial(mbar - m + s) * math.factorial(j - mbar - s))
            djmm += ((-1) ** (mbar - m + s) / denom *
                     np.cos(np.deg2rad(theta) / 2) ** (2 * j + m - mbar - 2 * s) *
                     np.sin(np.deg2rad(theta) / 2) ** (mbar - m + 2 * s))
        djmm *= np.sqrt(math.factorial(j + mbar) * math.factorial(j - mbar) *
                        math.factorial(j + m) * math.factorial(j - m))
        return djmm

    # -------------------------------------------------------------------------
    # Pulse superoperator
    # -------------------------------------------------------------------------
    @staticmethod
    def getPulseOperator(flipAngle, phase, m):
        """n×n pulse superoperator for coherence order m."""
        ns = np.arange(-3, 4)  # -3 to 3
        N = len(ns)
        pulseOperator = np.zeros((N, N), dtype=complex)
        for nIdx, n in enumerate(ns):
            for nBarIdx, nbar in enumerate(ns):
                if abs(n + m) <= 3 and abs(nbar + m) <= 3:
                    pulseOperator[nIdx, nBarIdx] = (
                        np.exp(1j * (n - nbar) * np.deg2rad(phase)) *
                        getValues.getWignerD(m, n, nbar, flipAngle)
                    )
        return pulseOperator



class _PseudoVoigtGen(rv_continuous):
    """
    Pseudo-Voigt distribution: a linear mixture of a Gaussian (Normal)
    and a Lorentzian (Cauchy), sharing the same center and scale.

        pdf(x) = (1 - eta) * Normal(x; loc, sig_g)
               +      eta  * Cauchy(x; loc, sig_l)

    Shape parameters
    ----------------
    a : eta   in [0, 1]  — Lorentzian fraction
    b : sig_g > 0        — Gaussian sigma (relative to the frozen scale `sig`)
    c : sig_l > 0        — Lorentzian HWHM (relative to the frozen scale `sig`)

    When frozen as  dist(loc=cen, scale=sig, a=eta, b=sig_g, c=sig_l):
      - Gaussian  : loc=cen, scale=sig * sig_g
      - Lorentzian: loc=cen, scale=sig * sig_l
    """

    def _argcheck(self, a, b, c):
        return (0 <= a) & (a <= 1) & (b > 0) & (c > 0)

    def _pdf(self, x, a, b, c):
        # x is already shifted/scaled by rv_continuous (loc/scale applied)
        return (1.0 - a) * norm.pdf(x, scale=b) + a * cauchy.pdf(x, scale=c)

    def _cdf(self, x, a, b, c):
        return (1.0 - a) * norm.cdf(x, scale=b) + a * cauchy.cdf(x, scale=c)

    def _ppf(self, q, a, b, c):
        # No closed form — use inherited numerical inversion via _cdf
        return super()._ppf(q, a, b, c)

    def _rvs(self, a, b, c, size=None, random_state=None):
        rng = self._random_state if random_state is None else random_state
        # Draw component membership (Bernoulli with p=eta)
        mask = rng.uniform(size=size) < a
        samples = np.where(
            mask,
            cauchy.rvs(scale=c, size=size, random_state=rng),
            norm.rvs(scale=b,   size=size, random_state=rng),
        )
        return samples