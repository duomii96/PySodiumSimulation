import numpy as np
from scipy.signal import savgol_filter

class SpectrumSampler:
    """
    Empirical sampler built from a (normalized) Fourier spectrum S(nu).

    Compatible with the existing framework's usage pattern:
        wShiftvec = self.FreqShift + self.wShiftdist.rvs(self.nSpins)

    Parameters
    ----------
    nu : array_like
        Frequency axis (Hz), monotonically increasing, length N.
    spectrum : array_like
        Spectral intensity at each nu point, length N. Does not need to
        be pre-normalized; it will be normalized internally. Negative
        values (e.g. from noisy experimental data) are clipped to zero.
    bin_size : int, optional
        Number of original spectral points to group into one bin.
        bin_size=1 keeps full resolution (no binning).
    to_angular : bool, optional
        If True, samples are returned in rad/s (i.e. multiplied by 2*pi)
        so they can be assigned directly to wShiftvec when the phase
        factors in add_phase_factors expect angular frequency.
    """

    def __init__(self, nu, spectrum, bin_size=1, to_angular=True):
        nu = np.asarray(nu, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)

        if nu.shape != spectrum.shape:
            raise ValueError("nu and spectrum must have the same shape")

        spectrum = np.clip(spectrum, 0, None)

        n_bins = len(nu) // bin_size
        trimmed = n_bins * bin_size
        if trimmed == 0:
            raise ValueError("bin_size too large for given spectrum length")

        nu_trim = nu[:trimmed]
        spec_trim = spectrum[:trimmed]

        nu_bins = nu_trim.reshape(n_bins, bin_size)
        spec_bins = spec_trim.reshape(n_bins, bin_size)

        self.bin_centers = nu_bins.mean(axis=1)
        weights = spec_bins.sum(axis=1)

        total = weights.sum()
        if total <= 0:
            raise ValueError("Spectrum integrates to zero; check input data")
        self.weights = weights / total

        self.to_angular = to_angular
        self._scale = 2 * np.pi if to_angular else 1.0

        self.mean = np.sum(self.bin_centers * self.weights) * self._scale
        self.var = np.sum(
            (self.bin_centers * self._scale - self.mean) ** 2 * self.weights
        )
        self.std = np.sqrt(self.var)

    def rvs(self, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(len(self.bin_centers), size=size, p=self.weights)
        picks = self.bin_centers[idx]
        if len(self.bin_centers) > 1:
            bin_width = np.diff(self.bin_centers).mean()
            picks = picks + rng.uniform(-bin_width / 2, bin_width / 2, size=size)
        return picks * self._scale

    def pdf_table(self):
        """Return (bin_centers_hz, weights) for inspection/plotting."""
        return self.bin_centers, self.weights

class AdaptiveSpectrumSampler:
    """
    Empirical sampler using equal-probability-mass (quantile) binning,
    so bin width automatically shrinks where spectral density is high
    (e.g. narrow central peak) and widens in low-density wings.
    """

    def __init__(self, nu, spectrum, n_bins=400, to_angular=True, jitter=True):
        nu = np.asarray(nu, dtype=float)
        spectrum = np.clip(np.asarray(spectrum, dtype=float), 0, None)

        order = np.argsort(nu)
        nu_s, spec_s = nu[order], spectrum[order]

        cdf = np.cumsum(spec_s)
        cdf = cdf / cdf[-1]

        quantile_edges = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.interp(quantile_edges, cdf, nu_s)
        bin_edges = np.unique(bin_edges)

        idx = np.searchsorted(bin_edges, nu_s, side='right') - 1
        idx = np.clip(idx, 0, len(bin_edges) - 2)

        n_actual = len(bin_edges) - 1
        weights = np.array([spec_s[idx == b].sum() for b in range(n_actual)])
        weights = weights / weights.sum()

        self.bin_edges = bin_edges
        self.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.bin_widths = np.diff(bin_edges)
        self.weights = weights

        self.jitter = jitter
        self.to_angular = to_angular
        self._scale = 2 * np.pi if to_angular else 1.0

    def rvs(self, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(len(self.bin_centers), size=size, p=self.weights)
        picks = self.bin_centers[idx]
        if self.jitter:
            picks = picks + rng.uniform(-0.5, 0.5, size=size) * self.bin_widths[idx]
        return picks * self._scale

    def pdf_table(self):
        return self.bin_centers, self.weights

def sampler_from_fid(fid, dt, bin_size=1, to_angular=True, freq_range=None):
    """
    Build a SpectrumSampler directly from a time-domain FID by FFT.

    Parameters
    ----------
    fid : array_like (complex)
        Time-domain signal, uniformly sampled.
    dt : float
        Sampling interval in seconds.
    bin_size : int
        Points per bin passed to SpectrumSampler.
    to_angular : bool
        Passed through to SpectrumSampler.
    freq_range : tuple(float, float), optional
        Restrict to (nu_min, nu_max) in Hz before building sampler,
        e.g. to exclude noise-dominated far wings.
    """
    fid = np.asarray(fid)
    n = len(fid)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
    nu = np.fft.fftshift(np.fft.fftfreq(n, d=dt))

    if freq_range is not None:
        mask = (nu >= freq_range[0]) & (nu <= freq_range[1])
        nu, spectrum = nu[mask], spectrum[mask]

    #return SpectrumSampler(nu, spectrum, bin_size=bin_size, to_angular=to_angular)
    return AdaptiveSpectrumSampler(nu, spectrum, to_angular=to_angular)

def smooth_spectrum(nu, spectrum, method="savgol", window=21, polyorder=3,
                     noise_threshold_frac=0.01, clip_negative=True):
    """
    Smooth and/or denoise a spectrum before adaptive binning.

    Parameters
    ----------
    nu : array_like
        Frequency axis (Hz).
    spectrum : array_like
        Raw spectral intensity (magnitude spectrum), same length as nu.
    method : "savgol" | "moving_average" | "gaussian"
        Smoothing method.
    window : int
        Window length (must be odd for savgol). Larger = more smoothing.
    polyorder : int
        Polynomial order for Savitzky-Golay (must be < window).
    noise_threshold_frac : float
        Fraction of peak intensity below which values are zeroed out
        after smoothing (set to 0 to disable thresholding).
    clip_negative : bool
        Clip negative values to zero (smoothing can introduce small
        negative ringing near sharp peaks).
    """
    nu = np.asarray(nu, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    if method == "savgol":
        if window % 2 == 0:
            window += 1
        window = min(window, len(spectrum) - (1 - len(spectrum) % 2))
        spectrum_smooth = savgol_filter(spectrum, window_length=window,
                                         polyorder=polyorder)
    elif method == "moving_average":
        kernel = np.ones(window) / window
        spectrum_smooth = np.convolve(spectrum, kernel, mode="same")
    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        sigma = window / 6.0  # heuristic: window ~ 6 sigma
        spectrum_smooth = gaussian_filter1d(spectrum, sigma=sigma)
    else:
        raise ValueError(f"Unknown method: {method}")

    if clip_negative:
        spectrum_smooth = np.clip(spectrum_smooth, 0, None)

    if noise_threshold_frac > 0:
        thresh = noise_threshold_frac * spectrum_smooth.max()
        spectrum_smooth[spectrum_smooth < thresh] = 0.0

    return nu, spectrum_smooth