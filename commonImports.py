"""
File to import all necessary libraries and to set certain global vars.
"""


import matplotlib.pyplot as plt
import numpy as np
from Classes.getValues import *
from numpy import real
from numpy import imag
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy.optimize import curve_fit

plt.rcParams.update({
    "lines.linewidth": 1.0,
    "errorbar.capsize": 2.0
})


"""plt.rcParams.update({
    "lines.linewidth": 1.0,
    "errorbar.capsize": 2.0,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": "Palatino",
})"""

