import matplotlib.pyplot as plt
import sys
from Classes.getFunction import getFunction
from Classes.getValues import getValues
import numpy as np
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic
import time

from DataLoadingStoring.storeSimulRes import save_IRTQTPPI
from commonImports import *
from Classes.getFunction import getFunction
from sampleParams import SAMPLES


B0 = 9.4
w0 = getValues.getw0(B0)

storeResult = False


# usage: python script.py Agar2_sr
"""if len(sys.argv) < 2:
    sys.exit(f"Usage: python {sys.argv[0]} <sampleName>  (one of {list(SAMPLES)})")
sampleName = sys.argv[1]
if sampleName not in SAMPLES:
    sys.exit(f"Unknown sampleName '{sampleName}'. Available: {list(SAMPLES)}")"""


sampleName = "Agar2_sr"
# Sim Paper TQTPPI, Inversion TQTPPI, and Spin Echo SR all used "Agar2";
# swap the key to "Agar4" / "Agar6" (or their "_alt" calibrations) as needed.
T1s, T1f, T2s, T2f, scale = SAMPLES[sampleName]
T1f = 35.92e-3
print(f"Simulating with: \n T1s ={T1s} s \n T1f ={T1f} s \n T2s ={T2s} s \n T2f ={T2f} s \n Scale ={scale} rad/s \n  ")

#FWHM_half = np.finfo(float).eps # rad/s
FWHM_half = scale
print(f"B0-inhomogeneity: {FWHM_half} \n")


#Select model
Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)
PC = PhaseCyclesVecDic(B0, tauC, wQ,wQbar=0, Jen=Jen, wShiftRMS=0,wShiftFID=0)

"""tauC, wQ = getValues.gettauCwQT1(T1f, T1s, w0)  # or gettauCwQ(T2f, T2s, w0)
PC = PhaseCyclesVecDic(B0, tauC[0], wQ[0], wQbar=0.0, Jen=0.0, wShiftRMS=0.0)"""

"""tauC, wQ = getValues.gettauCwQ(T2f, T2s, w0)  # or gettauCwQ(T2f, T2s, w0)
PC = PhaseCyclesVecDic(B0, tauC[0], wQ[0], wQbar=0.0, Jen=0.0, wShiftRMS=0.0)"""


PC.dwelltimeFID   = 100e-6
PC.dataPoints     = 2048
PC.NumPhaseCycles = 80
PC.nSpins         = 1
PC.tmix           = 0.14e-3
#PC.wShiftdist     = getValues.getWshiftDist('cauchy', 0, FWHM_half)
#PC.wShiftdist     = getValues.getWshiftDist('normal', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist("voigt", cen=0.0, sig=FWHM/2, eta=0.7)
PC.flip90         = 90.0
PC.TR             = 400e-3
PC.tevo           = 30e-3
PC.tevo0          = 0.207e-3
PC.tevoStep       = 0.2e-3

t0 = time.time()
mqFID, mqSpectrum, tevos, _, _, _, _ = PC.IRTQTPPI_sr()
t1 = time.time()
print("Total Time: s ", t1 - t0)

if storeResult:
    save_IRTQTPPI(f"IRTQTPPI_{sampleName}_{PC.nSpins}_T2debeye", PC, mqFID, mqSpectrum, tevos)

plt.figure()
plt.subplot(211)
plt.plot(tevos,real(mqFID))
plt.grid()
plt.subplot(212)
plt.plot(real(mqSpectrum))
plt.grid()
plt.show(block=False)
plt.pause(0.5)

"""mqFID /= np.amax(np.abs(mqFID))

p0 = [T1s, T1f,  0.8, 0.2, 0.1,1 / (PC.tevoStep * len(PC.alphas)), np.pi, -np.pi, 0]
popt, pcov = getFunction.fit_IRTQTPPI_FID(tevos, mqFID, p0, fixAmp=False)

print(popt)

plt.figure()
plt.plot(tevos, mqFID, 'o-', label="mqFID")
plt.plot(tevos, getFunction.getFunc_IRTQTPPI_FID(tevos,*popt), label="FIT")
plt.grid()
plt.legend(loc='best')
plt.show()"""