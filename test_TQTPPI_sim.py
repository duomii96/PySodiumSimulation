from Classes.getValues import getValues
import numpy as np
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic
import time
from commonImports import *
from diagnoseCheck import diagnose_sequence_with180
from Classes.getFunction import getFunction
from DataLoadingStoring.storeSimulRes import save_TQTPPI, save_SpinEcho
from sampleParams import SAMPLES



B0 = 9.4
w0 = getValues.getw0(B0)

sampleName = "Agar6_sr"
T1s, T1f, T2s, T2f, scale = SAMPLES[sampleName]
print(f"Simulating with: \n T1s ={T1s} s \n T1f ={T1f} s \n T2s ={T2s} s \n T2f ={T2f} s \n Scale ={scale} rad/s \n  ")

scale = scale # rad/s

# Select model
# Jen Model
#Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)
#PC = PhaseCyclesVecDic(B0, tauC, wQ,wQbar=0, Jen=Jen, wShiftRMS=0,wShiftFID=0)
# T2 debye
"""tauC, wQ = getValues.gettauCwQ(T2f, T2s, w0)  # or gettauCwQT1(T2f, T2s, w0)
PC = PhaseCyclesVecDic(B0, tauC[0], wQ[0], wQbar=0.0, Jen=0.0, wShiftRMS=0.0)"""
# T1 debye
tauC, wQ = getValues.gettauCwQT1(T1f, T1s, w0)  # or gettauCwQT1(T2f, T2s, w0)
PC = PhaseCyclesVecDic(B0, tauC[0], wQ[0], wQbar=0.0, Jen=0.0, wShiftRMS=0.0)

PC.dwelltimeFID   = 100e-6
PC.dataPoints     = 2048
PC.NumPhaseCycles = 80
PC.nSpins         = 1
PC.tmix           = 0.14e-3
#PC.wShiftdist     = getValues.getWshiftDist('cauchy', 0, scale)
#PC.wShiftdist     = getValues.getWshiftDist('normal', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist("voigt", cen=0.0, sig=FWHM/2, eta=0.7)
PC.flip90         = 90.0
PC.TR             = 400.0e-3
PC.tevo           = 30e-3
PC.tevo0          = 0.0e-3
PC.tevoStep       = 0.2e-3


spinEcho = False
storeResult = True


if spinEcho:
    t0 = time.time()
    acqTimeVec, echoFIDs, echoAmps, echoTimeVec = PC.SpinEcho()
    t1 = time.time()
    print("Total Time: s ", t1 - t0)
    echoAmps /= np.amax(echoAmps)
    popt, pcov = getFunction.fit_biexponentialSQ(echoTimeVec * 2, echoAmps,
                                     [T2s, T2f, 0.5, 0.5, 0.01])

    if storeResult:
        save_SpinEcho(f"SpinEcho_{sampleName}", PC, acqTimeVec, echoFIDs, echoAmps, echoTimeVec)

    plt.figure()
    plt.plot(echoTimeVec * 2, echoAmps, label="Echo Data")
    plt.plot(echoTimeVec * 2, getFunction.getFunc_biexp(echoTimeVec * 2, *popt), label="biexponential")
    plt.show()

    print(popt)

else:
    t0 = time.time()
    acqTimeVec, mqSpectrum, mqFID, FIDs, fAngles90, tevos = PC.TQTPPIWith180()
    t1 = time.time()
    print("Total Time: s ", t1 - t0)

    # normalize FID
    mqFID /= np.amax(mqFID)

    if storeResult:
        save_TQTPPI(f"TQTPPI_{sampleName}_{PC.nSpins}_T1debye", PC, acqTimeVec, mqSpectrum, mqFID, FIDs, tevos)

    p0 = [T2s, T2f, 0.4, 0.6, 0.2, 1/(PC.tevoStep * len(PC.alphas)), np.pi, -np.pi,0]
    popt, pcov = getFunction.fit_TQTTPI_FID(tevos, mqFID, p0)

    print(popt)

    plt.figure()
    plt.plot(tevos, mqFID,'o-', label="mqFID")
    plt.plot(tevos, getFunction.getFunc_TQTPPI_FID(tevos, *popt), label="FIT")
    plt.legend(loc='best')
    plt.show()








#times, FID, TQ, SQ = PC.SinglePulseVec()

