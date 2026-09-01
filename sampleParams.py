"""
Relaxation-time parameter sets for calibrated Agarose phantom samples.

Values are biexponential-fit T1s/T1f/T2s/T2f (in seconds) at 9.4 T.
Use as: T1s, T1f, T2s, T2f = SAMPLES["Agar2"]
"""

from collections import namedtuple

SampleParams = namedtuple("SampleParams", ["T1s", "T1f", "T2s", "T2f"])

SAMPLES = {
    # 2% Agarose, FWHM 47 Hz
    "Agar2": SampleParams(T1s=49.1431e-3, T1f=35.1757e-3, T2s=43.3500e-3, T2f=10.2693e-3),
    # 2% Agarose, alternate calibration, FWHM 47 Hz
    "Agar2_alt": SampleParams(T1s=46.65000934e-3, T1f=41.92039465e-3, T2s=43.35286305e-3, T2f=9.137434931e-3),
    # 2% Agarose, SR
    "Agar2_sr": SampleParams(T1s=48.3581e-3, T1f=35.9234e-3, T2s=43.87286305e-3, T2f=8.9369e-3),

    # 4% Agarose, FWHM 47 Hz
    "Agar4": SampleParams(T1s=42.9525e-3, T1f=31.1021e-3, T2s=37.8604e-3, T2f=6.1239e-3),
    # 4% Agarose, alternate calibration
    "Agar4_alt": SampleParams(T1s=41.38000934e-3, T1f=33.62039465e-3, T2s=37.22286305e-3, T2f=5.127434931e-3),
    # 4% Agarose, SR
    "Agar4_sr": SampleParams(T1s=41.1338e-3, T1f=30.6446e-3, T2s=36.4629e-3, T2f=4.8999e-3),

    # 6% Agarose, FWHM 47 Hz
    "Agar6": SampleParams(T1s=39.079334e-3, T1f=27.22039465e-3, T2s=32.025e-3, T2f=4.1242e-3),
    # 6% Agarose, alternate calibration
    "Agar6_alt": SampleParams(T1s=39.129e-3, T1f=27.6504e-3, T2s=32.3578e-3, T2f=3.9326e-3),
    # 6% Agarose, SR
    "Agar6_sr": SampleParams(T1s=36.5995e-3, T1f=27.5178e-3, T2s=32.0251e-3, T2f=3.6569e-3)
}