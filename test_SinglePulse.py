from cProfile import label

import matplotlib.pyplot as plt

from Classes.getValues import getValues
import numpy as np
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic
import time
from commonImports import *
from diagnoseCheck import diagnose_sequence_with180
from Classes.getFunction import getFunction
from DataLoadingStoring.NEOreadIn import NEOreadInSE
from Classes.spectralSampling import sampler_from_fid, smooth_spectrum, AdaptiveSpectrumSampler
import plotly.graph_objects as go
from sampleParams import SAMPLES


B0 = 9.4
w0 = getValues.getw0(B0)

_sampleFWHMfromSP = False

T1s, T1f, T2s, T2f = SAMPLES["Agar6"]

Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)



PC = PhaseCyclesVecDic(B0, tauC, wQ,wQbar=0, Jen=Jen, wShiftRMS=wShiftRMS,wShiftFID=0)



if _sampleFWHMfromSP:
    complexData, method, _ = NEOreadInSE(r"C:\Users\dz8\Data\Messdaten\DZ_Agar6_newReco_1_106_20250609_125343\68")
    FIDGibbs_reduced = np.squeeze(complexData)[78:]
    dt = 0.1e-03 # dwell time in seconds
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(FIDGibbs_reduced)))

    n = len(spectrum)
    nu = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    nu_s, spec_s = smooth_spectrum(nu, spectrum, method="savgol",
                                        window=9, polyorder=3,
                                        noise_threshold_frac=0.01)

    plt.figure()
    plt.grid(True)
    plt.plot(spectrum, label='Input spectrum')
    plt.plot(spec_s, label='Smooth spectrum')
    plt.legend()
    #plt.savefig("SmoothedSpectrum_window5.png")

    sampler = AdaptiveSpectrumSampler(nu_s, spec_s, n_bins=401, to_angular=False, jitter=True)

    centers, weights = sampler.pdf_table()
    print("Peak bin:", centers[np.argmax(weights)])
    print("Mean (Hz):", np.sum(centers * weights))

    """samples = sampler.rvs(200000)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=samples, histnorm='probability density', nbinsx=100, name="Drawn samples"))
    fig2.add_trace(go.Scatter(x=centers, y=weights / np.diff(centers).mean(), mode='lines', name="pdf_table (density)"))
    fig2.write_image("Sampler.png")
    fig2.show()"""



    PC.wShiftdist = sampler

else:
    FWHM = 50
    PC.wShiftdist = getValues.getWshiftDist('cauchy', 0, FWHM / 2)

PC.dwelltimeFID   = 100e-6
PC.dataPoints     = 2048
PC.nSpins         = 1000
PC.tmix           = 0.14e-3
#PC.wShiftdist     = getValues.getWshiftDist('cauchy', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist('normal', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist("voigt", cen=0.0, sig=FWHM/2, eta=0.7)
PC.flip90         = 90.0
PC.TR             = 400.0e-3


FWHM_est= getValues.getDistFWHM(PC.wShiftdist)

times, FID, TQ, SQ = PC.SinglePulse()

FID = real(FID)

FID /= np.amax(FID)




# ------ FIT ------------

p0 = [0.020,0.005, 0.4, 0.6, 0]
popt, pcov = getFunction.fit_biexponentialSQ(times, FID, plot=True)

plt.plot(times, FID)

plt.show()

print(popt)
