from Classes.getValues import getValues
import numpy as np
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic
import time
from commonImports import *
from diagnoseCheck import diagnose_sequence_with180
from Classes.getFunction import getFunction



B0 = 9.4
w0 = getValues.getw0(B0)


T1s =  36.18000934e-3
T1f =  26.22039465e-3
T2s =  32.57286305e-3
T2f =   3.657434931e-3
# Agar 6%
"""T1s = 39.129e-3
T1f = 27.6504e-3
T2s = 32.3578e-3
T2f = 3.9326e-3"""
# agar 4%
"""T1s =  41.38000934e-3
T1f =  33.62039465e-3
T2s =  37.22286305e-3
T2f =   5.127434931e-3"""
# Relaxationrates 2% @ 9.4T FHWM: 47
"""T1s =  46.65000934e-3
T1f =  41.92039465e-3
T2s =  43.35286305e-3
T2f =   9.137434931e-3"""

#----------- For Sim Paper TQTPPI and Inversion TQTPPI -----------
#6% Agarose  FWHM: 47
T1s =  39.079334e-3
T1f =  27.22039465e-3 # T1f_8020
T2s =  32.025e-3
T2f =   4.1242e-3


#4% Agarose  FWHM: 47
T1s =  42.9525e-3
T1f =  31.1021e-3 # T1f_8020
T2s =  37.8604e-3
T2f =   6.1239e-3

#2% Agarose  FWHM: 47
T1s =  49.1431e-3
T1f =  35.1757e-3 # T1f_8020
T2s =  43.3500e-3
T2f =  10.2693e-3

#----------- For Sim Paper Spin Echo SR-----------
#6% Agarose  FWHM: 47
T1s =  39.079334e-3
T1f =  27.22039465e-3 # T1f_8020
T2s =  32.025e-3
T2f =   4.1242e-3


#4% Agarose  FWHM: 47
T1s =  42.9525e-3
T1f =  31.1021e-3 # T1f_8020
T2s =  37.8604e-3
T2f =   6.1239e-3

#2% Agarose  FWHM: 47
T1s =  49.1431e-3
T1f =  35.1757e-3 # T1f_8020
T2s =  43.3500e-3
T2f =  10.2693e-3



FWHM = 50

Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)

"""
T1s_r, T1f_r, T2s_r, T2f_r = getValues.getrelaxationTimesJen(
    tauC, wQ, 0.0, Jen, 0.0, w0)

print(f"Input:     T2f={T2f*1e3:.2f} ms,  T2s={T2s*1e3:.1f} ms")
print(f"Recovered: T2f={T2f_r*1e3:.2f} ms, T2s={T2s_r*1e3:.1f} ms")"""

PC = PhaseCyclesVecDic(B0, tauC, wQ, 0.0, Jen, 0.0, 0.0, 0.0)
PC.dwelltimeFID   = 100e-6
PC.dataPoints     = 2048
PC.NumPhaseCycles = 60
PC.nSpins         = 1000
PC.tmix           = 0.15e-3
PC.wShiftdist     = getValues.getWshiftDist('cauchy', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist('normal', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist("voigt", cen=0.0, sig=FWHM/2, eta=0.7)
PC.flip90         = 90.0
PC.TR             = 204.7e-3
PC.tevo           = 30e-3
PC.tevo0          = 0.0e-3
PC.tevoStep       = 0.2e-3


spinEcho = False


if spinEcho:
    t0 = time.time()
    acqTimeVec, echoFIDs, echoAmps, echoTimeVec = PC.SpinEcho()
    t1 = time.time()
    print("Total Time: s ", t1 - t0)
    echoAmps /= np.amax(echoAmps)
    popt, pcov = getFunction.fit_biexponentialSQ(echoTimeVec * 2, echoAmps,
                                     [T2s, T2f, 0.5, 0.5, 0.01])

    plt.figure()
    plt.plot(echoTimeVec * 2, echoAmps, label="Echo Data")
    plt.plot(echoTimeVec * 2, getFunction.getFitFunc_biexp(echoTimeVec * 2, *popt), label="biexponential")
    plt.show()

    print(popt)

else:
    t0 = time.time()
    acqTimeVec, mqSpectrum, mqFID, FIDs, fAngles90, tevos = PC.TQTPPIWith180()
    t1 = time.time()
    print("Total Time: s ", t1 - t0)

    # normalize FID
    mqFID /= np.amax(mqFID)

    p0 = [T2s, T2f, 0.4, 0.6, 0.2, 1/(PC.tevoStep * len(PC.alphas)), np.pi, -np.pi,0]
    popt, pcov = getFunction.fit_TQTTPI_FID(tevos, mqFID, p0)

    print(popt)

    plt.figure()
    plt.plot(tevos, mqFID,'o-', label="mqFID")
    plt.plot(tevos, getFunction.getFunc_TQTPPI_FID(tevos, *popt), label="FIT")
    plt.legend(loc='best')
    plt.show()


#acqTimeVec, mqSpectrum, mqFID, FIDs, fAngles90, tevos = PC.TQTPPIWith180()
#acqTimeVec, echoFIDs, echoAmps, echoTimeVec = PC.SpinEcho()
#acqTimeVec, FIDs, TQ, SQ = PC.SinglePulseVec()






#times, FID, TQ, SQ = PC.SinglePulseVec()

