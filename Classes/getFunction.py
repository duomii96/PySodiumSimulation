
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
    def getFunc_biexp_gamma(t, T2a, T2b, A1, A2, C, gamma):
        """
        Biexponential decay with an additional common e^(-gamma*t) factor
        multiplied onto both relaxation exponentials, accounting for extra
        dephasing from B0 inhomogeneity (gamma = scale parameter used for
        sampling the B0 inhomogeneity distribution).
        """
        return (A1 * np.exp(-t / T2a) + A2 * np.exp(-t / T2b)) * np.exp(-gamma * t) + C

    @staticmethod
    def getFunc_TQTPPI_FID(t, T2s, T2f, As, Af, Atq, omega, ph1, ph2, C):
        return np.sin(t * omega * 2 * np.pi + ph1) * (As * np.exp(-t / T2s) + Af * np.exp(-t / T2f)) + Atq * np.sin(
            3 * 2 * np.pi * omega * t + ph2) * (np.exp(-t / T2s) - np.exp(-t / T2f)) + C

    @staticmethod
    def getFunc_IRTQTPPI_FID(t, T1s, T1f, A1s, A1f, Atq, omega, ph1, ph2, C):
        return np.sin(t * omega * 2 * np.pi + ph1) * (
                    1 - 2 * (A1s * np.exp(-t / T1s) + A1f * np.exp(-t / T1f))) + Atq * np.sin(
            3 * 2 * np.pi * omega * t + ph2) * (np.exp(-t / T1s) - np.exp(-t / T1f)) + C

    @staticmethod
    def fit_IRTQTPPI_FID(t, y, p0=None, fixAmp=True, bounds=None):
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

    def fit_biexponentialSQ(t, y, p0=None, bounds=None, plot=False, singleAmp=False):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        #check if data is normalized
        if np.amax(y) >1:
            y /= np.amax(y)

        if p0 is None:

            p0 = [0.030, 0.010, 0.4 , 0.6, 0]

        if bounds is None and not(singleAmp):
            bounds = (
                [0, 0, 0, 0, -0.1],
                [0.07, 0.070, 1, 1, 0.1]
            )
            popt, pcov = curve_fit(getFunction.getFunc_biexp, t, y, p0=p0, bounds=bounds, maxfev=20000)

            plt.figure(figsize=(9, 5), dpi=200)
            plt.plot(t, y, 'o-', label="Data")
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
            plt.figure(figsize=(9, 5), dpi=200)
            plt.plot(t, y, '<-', label="Data")
            plt.plot(t, getFunction.getFunc_biexp_oneAmp(t, *popt))
            plt.grid(True)
            plt.xlabel("Time (s)")
            plt.show()
        else:
            # bounds are specified // 4 par fit
            popt, pcov = curve_fit(getFunction.getFunc_biexp_oneAmp, t, y, p0=p0, bounds=bounds, maxfev=20000)
            plt.figure(figsize=(9, 5), dpi=200)
            plt.plot(t, y, '<-', label="Data")
            plt.plot(t, getFunction.getFunc_biexp_oneAmp(t, *popt))
            plt.grid(True)
            plt.xlabel("Time (s)")
            plt.show()







        return popt, pcov

    @staticmethod
    def fit_biexp_gamma(t, y, p0=None, bounds=None, fixed=None, plot=False):
        """
        Fit getFunc_biexp_gamma to data.

        fixed : dict, optional
            Any subset of {'T2a', 'T2b', 'A1', 'A2', 'C'} with already-known
            values. Parameters given here are held fixed during the fit;
            only the remaining ones (gamma always included) are optimized.
            e.g. fixed={'T2a': 0.030, 'T2b': 0.010, 'A1': 0.4, 'A2': 0.6, 'C': 0}
            leaves gamma as the only free parameter.

        Returns
        -------
        popt, pcov : optimized values (free params only) and their covariance
        fullParams : dict with all six parameter names/values (fixed + fitted)
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.amax(y) > 1:
            y = y / np.amax(y)

        if fixed is None:
            fixed = {}

        allNames = ['T2a', 'T2b', 'A1', 'A2', 'C']
        freeNames = [n for n in allNames if n not in fixed] + ['gamma']

        p0default = {'T2a': 0.030, 'T2b': 0.010, 'A1': 0.4, 'A2': 0.6, 'C': 0.0, 'gamma': 1000.0}
        lowerAll = {'T2a': 0, 'T2b': 0, 'A1': 0, 'A2': 0, 'C': -0.1, 'gamma': 0}
        upperAll = {'T2a': 0.07, 'T2b': 0.07, 'A1': 1, 'A2': 1, 'C': 0.1, 'gamma': 1e5}

        if p0 is None:
            p0 = [p0default[n] for n in freeNames]

        if bounds is None:
            bounds = ([lowerAll[n] for n in freeNames],
                       [upperAll[n] for n in freeNames])

        def _wrapped(t, *params):
            kwargs = dict(fixed)
            kwargs.update(zip(freeNames, params))
            return getFunction.getFunc_biexp_gamma(
                t, kwargs['T2a'], kwargs['T2b'], kwargs['A1'], kwargs['A2'],
                kwargs['C'], kwargs['gamma']
            )

        popt, pcov = curve_fit(_wrapped, t, y, p0=p0, bounds=bounds, maxfev=20000)

        fullParams = dict(fixed)
        fullParams.update(zip(freeNames, popt))

        if plot:
            plt.figure(figsize=(9, 5), dpi=200)
            plt.plot(t, y, 'o-', label="Data")
            plt.plot(t, _wrapped(t, *popt), label="Fit")
            plt.grid(True)
            plt.xlabel("Time (s)")
            plt.legend()
            plt.show()

        return popt, pcov, fullParams

    def fit_TQTTPI_FID(t, y, p0=None, bounds=None):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if bounds is None:
            bounds = (
                [0, 0, 0, 0, 0, 0, -2 * np.pi, -2 * np.pi, -0.1],
                [0.070, 0.050, 1.0 , 1.0, 1, 1e5, 3 * np.pi, 3 * np.pi, 0.1]
            )

        popt, pcov = curve_fit(getFunction.getFunc_TQTPPI_FID, t, y, p0=p0, bounds=bounds, maxfev=20000)

        return popt, pcov


