
from scipy.optimize import fmin, least_squares
from scipy.stats import norm as scipy_norm
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


"""
Function library with corresponding fit routine. Call via getFunction.<insertFunctionName>
"""



class getFunction:
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

    @staticmethod
    def getFunc_biexp(t, T2a, T2b, A1, A2, C):
        return A1 * np.exp(-t / T2a) + A2 * np.exp(-t / T2b) + C

    def getFunc_biexp_oneAmp(t, T2a, T2b, A1, C):
        return A1 * np.exp(-t / T2a) + (1-A1) * np.exp(-t / T2b) + C

    @staticmethod
    def getFunc_TQTPPI_FID(t, T2s, T2f, As, Af, Atq, omega, ph1, ph2, C):
        return np.sin(t * omega * 2 * np.pi + ph1) * (As * np.exp(-t / T2s) + Af * np.exp(-t / T2f)) + Atq * np.sin(
            3 * 2 * np.pi * omega * t + ph2) * (np.exp(-t / T2s) - np.exp(-t / T2f)) + C

    @staticmethod
    def getFunc_IRTQTPPI_FID(t, T1s, T1f, A1s, A1f, Atq, omega, ph1, ph2, C):
        return np.sin(t * omega * 2 * np.pi + ph1) * (
                    1 - 2 * (A1s * np.exp(-t / T1s) + A1f * np.exp(-t / T1f))) + Atq * np.sin(
            3 * 2 * np.pi * omega * t + ph2) * (np.exp(-t / T1s) - np.exp(-t / T1f)) + C

    def fit_biexponentialSQ(t, y, p0=None, bounds=None, plot=False, singleAmp=False):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        #check if data is normalized
        if np.amax(y) >1:
            y /= np.amax(y)

        if p0 is None:
            span = t.ptp() if t.ptp() > 0 else 1.0
            offset = y[-1]
            amp = y[0] - offset
            p0 = (span / 5, span / 2, 0.6 * amp, 0.4 * amp, offset)

        if bounds is None and not(singleAmp):
            bounds = (
                [0, 0, 0, 0, -0.1],
                [0.07, 0.070, 1, 1, 0.1]
            )
            popt, pcov = curve_fit(getFunction.getFunc_biexp, t, y, p0=p0, bounds=bounds, maxfev=20000)

            plt.figure(figsize=(9, 16), dpi=200)
            plt.plot(t, y, '<-', label="Data")
            plt.plot(t, getFunction.getFunc_biexp(t, *popt))
            plt.grid(True)
            plt.xlabel("Time (s)")
            plt.show()
        elif bounds is None and singleAmp:
            bounds = (
                [0, 0, 0, -0.1],
                [0.07, 0.070,  1, 0.1]
            )
            popt, pcov = curve_fit(getFunction.getFunc_biexp_oneAmp, t, y, p0=p0, bounds=bounds, maxfev=20000)
            plt.figure(figsize=(9, 16), dpi=200)
            plt.plot(t, y, '<-', label="Data")
            plt.plot(t, getFunction.getFunc_biexp_oneAmp(t, *popt))
            plt.grid(True)
            plt.xlabel("Time (s)")
            plt.show()
        else:
            # bounds are specified // 4 par fit
            popt, pcov = curve_fit(getFunction.getFunc_biexp_oneAmp, t, y, p0=p0, bounds=bounds, maxfev=20000)
            plt.figure(figsize=(9, 16), dpi=200)
            plt.plot(t, y, '<-', label="Data")
            plt.plot(t, getFunction.getFunc_biexp_oneAmp(t, *popt))
            plt.grid(True)
            plt.xlabel("Time (s)")
            plt.show()







        return popt, pcov

    def fit_TQTTPI_FID(t, y, p0=None, bounds=None):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if bounds is None:
            bounds = (
                [0, 0, 0, 0, 0, 0, -2 * np.pi, -2 * np.pi, -0.1],
                [0.070, 0.050, 1, 1, 1, 1e5, 3 * np.pi, 3 * np.pi, 0.1]
            )

        popt, pcov = curve_fit(getFunction.getFunc_TQTPPI_FID, t, y, p0=p0, bounds=bounds, maxfev=20000)

        return popt, pcov

    def fit_IRTQTTPI_FID(t, y, p0=None, fixAmp=True, bounds=None):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if bounds is None:

            if fixAmp:

                bounds = (
                    [0, 0, 0.79999, 0.1999, 0, 0, -np.pi, -np.pi, -0.1],
                    [0.060, 0.035, 0.8, 0.2, 0.5, 1e5, 2 * np.pi, 2 * np.pi, 0.1]
                )
            else:

                bounds = (
                    [0, 0, 0, 0, 0, 0, -np.pi, -np.pi, -0.1],
                    [0.060, 0.035, 1, 1, 1, 1e5, np.pi, np.pi, 0.1]
                )

        popt, pcov = curve_fit(getFunction.getFunc_IRTQTPPI_FID, t, y, p0=p0, bounds=bounds, maxfev=20000)

        return popt, pcov
