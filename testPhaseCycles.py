from Classes.getValues import getValues
import numpy as np
from Classes.PhaseCycles_VecDic import PhaseCyclesVecDic
import time
from commonImports import *
from diagnoseCheck import diagnose_sequence_with180
from Classes.getFunction import getFunction

def diagnose_sequence_with_relax(B0, tauC, wQ, wQbar, Jen,
                                  flip90=90.0, alpha=0.0,
                                  tevo=0.1e-3, tmix=0.1e-3,
                                  wShiftRMS=0.0, nPtsevo=66, nPtsmix=34):
    """
    Traces the full TQTPPI state vector at every key checkpoint WITH relaxation.
    Uses a single spin (wShift=0) for clarity.
    Checkpoints:
      1. After Pulse 1
      2. After Evolution (tevo)
      3. After Pulse 2 (st1: alpha+90, st2: alpha-90)
      4. After Mixing (tmix)
      5. After Pulse 3 (phase=0)
    """
    import numpy as np
    from Classes.TmnEvo_VecDic import TmnEvoVecDic

    ns_labels = [-3, -2, -1, 0, 1, 2, 3]
    m_labels  = [1, 2, 3]

    def print_state(st, label):
        print(f"\n--- {label} ---")
        for mi, m in enumerate(m_labels):
            row = st.Tmnbatch[0, mi, :]
            nonzero = [(n, v) for n, v in zip(ns_labels, row) if abs(v) > 1e-10]
            if nonzero:
                parts = ", ".join(f"T_{{{m},{n}}}={v:.5f}" for n, v in nonzero)
                print(f"  m={m}: {parts}")
            else:
                print(f"  m={m}: (all zero)")

    def copy_state(src, dst):
        dst.Tmnbatch[:] = src.Tmnbatch[:]

    wShiftvec = np.array([0.0])

    st1 = TmnEvoVecDic(B0, tauC, wQ, wQbar, Jen, wShiftvec, wShiftRMS, 0.0)
    st2 = TmnEvoVecDic(B0, tauC, wQ, wQbar, Jen, wShiftvec, wShiftRMS, 0.0)

    # Precompute relaxation caches
    ts_evo = np.linspace(0, tevo, nPtsevo)
    ts_mix = np.linspace(0, tmix, nPtsmix)
    cfevo = st1.precompute_fJen(ts_evo)
    cfevo = st1.add_phase_factors(cfevo)
    cfmix = st1.precompute_fJen(ts_mix)
    cfmix = st1.add_phase_factors(cfmix)

    print(f"\n{'='*60}")
    print(f"SEQUENCE DIAGNOSTICS")
    print(f"  flip={flip90}°, alpha={alpha}°, tevo={tevo*1e3:.3f}ms, tmix={tmix*1e3:.3f}ms")
    print(f"  Pulse1: phase={alpha}°")
    print(f"  Pulse2: st1={alpha+90}°, st2={alpha-90}°")
    print(f"  Pulse3: 0°")
    print(f"{'='*60}")

    # ── Pulse 1 ──
    st1.reset()
    st2.reset()
    print_state(st1, "INITIAL (thermal eq)")

    st1.pulsebatch(flip90, alpha)
    copy_state(st1, st2)
    print_state(st1, f"[1] After Pulse 1 (flip={flip90}°, phase={alpha}°)")

    # ── Evolution ──
    st1.relax_endpoint_cached(cfevo)
    copy_state(st1, st2)   # st1 and st2 are identical until pulse 2
    print_state(st1, f"[2] After Evolution (tevo={tevo*1e3:.3f} ms)")

    # ── Pulse 2 (branches diverge) ──
    st1.pulsebatch(flip90, alpha + 90)
    st2.pulsebatch(flip90, alpha - 90)
    print_state(st1, f"[3a] After Pulse 2 st1 (phase={alpha+90}°)")
    print_state(st2, f"[3b] After Pulse 2 st2 (phase={alpha-90}°)")

    # ── Mixing ──
    st1.relax_endpoint_cached(cfmix)
    st2.relax_endpoint_cached(cfmix)
    print_state(st1, f"[4a] After Mixing st1 (tmix={tmix*1e3:.3f} ms)")
    print_state(st2, f"[4b] After Mixing st2 (tmix={tmix*1e3:.3f} ms)")

    # ── Pulse 3 ──
    st1.pulsebatch(flip90, 0)
    st2.pulsebatch(flip90, 0)
    print_state(st1, "[5a] After Pulse 3 st1 (phase=0°)")
    print_state(st2, "[5b] After Pulse 3 st2 (phase=0°)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("KEY COHERENCES AFTER PULSE 3:")
    for label, st in [("st1", st1), ("st2", st2)]:
        T11  = st.Tmnbatch[0, 0, 4]
        T1m1 = st.Tmnbatch[0, 0, 2]
        T31  = st.Tmnbatch[0, 2, 4]
        T33  = st.Tmnbatch[0, 2, 6]
        T3m3 = st.Tmnbatch[0, 2, 0]
        print(f"  {label}: T_{{1,+1}}={T11:.5f}  T_{{1,-1}}={T1m1:.5f}  "
              f"T_{{3,+1}}={T31:.5f}  T_{{3,+3}}={T33:.5f}  T_{{3,-3}}={T3m3:.5f}")

    diff_T11 = st1.Tmnbatch[0,0,4] - st2.Tmnbatch[0,0,4]
    sum_T11  = st1.Tmnbatch[0,0,4] + st2.Tmnbatch[0,0,4]
    print(f"\n  st1 - st2  →  T_{{1,1}} = {diff_T11:.5f}  (TQ pathway signal)")
    print(f"  st1 + st2  →  T_{{1,1}} = {sum_T11:.5f}  (SQ pathway signal)")
    print(f"{'='*60}")

    return st1, st2



B0 = 9.4
w0 = getValues.getw0(B0)


"""T1s =  46.18000934e-3
T1f =  36.22039465e-3
T2s =  32.57286305e-3
T2f =   3.657434931e-3"""
# Agar 6%
T1s = 39.129e-3
T1f = 27.6504e-3
T2s = 32.3578e-3
T2f = 3.9326e-3
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





FWHM = 50

Jen, tauC, wQ, wShiftRMS = getValues.getJenModel(T1f, T1s, T2f, T2s, w0)

#st1, st2 = diagnose_sequence_with180(B0=B0, tauC=tauC, wQ=wQ, wQbar=0, Jen=Jen,
#    flip90=90.0, flip180=180,alpha=45, tevo=30e-3, tmix=0.15e-3)

"""
T1s_r, T1f_r, T2s_r, T2f_r = getValues.getrelaxationTimesJen(
    tauC, wQ, 0.0, Jen, 0.0, w0)

print(f"Input:     T2f={T2f*1e3:.2f} ms,  T2s={T2s*1e3:.1f} ms")
print(f"Recovered: T2f={T2f_r*1e3:.2f} ms, T2s={T2s_r*1e3:.1f} ms")"""

PC = PhaseCyclesVecDic(B0, tauC, wQ, 0.0, Jen, 0.0, 0.0, 0.0)
PC.dwelltimeFID   = 100e-6
PC.dataPoints     = 2048
PC.NumPhaseCycles = 16
PC.nSpins         = 1000
PC.tmix           = 0.15e-3
PC.wShiftdist     = getValues.getWshiftDist('cauchy', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist('normal', 0, FWHM/2)
#PC.wShiftdist     = getValues.getWshiftDist("voigt", cen=0.0, sig=FWHM/2, eta=0.7)
PC.flip90         = 90.0
PC.TR             = 204.7e-3
PC.tevo           = 25e-3
PC.varFlip90      = False
PC.flipShiftdist90 = "Cauchy"



newReco = True
singlePulse=False

t0 = time.time()
acqTimeVec, mqSpectrum, FIDs, fAngles90 = PC.TQTPPIfixedwo180VJ()
#acqTimeVec, mqSpectrum, FIDs, fAngles90 = PC.TQTPPIfixWith180dz()
#times, FIDs, TQ, SQ = PC.SinglePulseVec()

if newReco:
    ftFIDs = np.fft.fftshift(np.fft.fft(FIDs, axis=0), 0)
    TQacqs = np.real(ftFIDs[16,:])
    SQacqs = np.real(ftFIDs[48, :])

    SQ_n = SQacqs / np.amax(SQacqs)
    TQ_n = TQacqs / np.amax(TQacqs)

    plt.figure()
    plt.hist(fAngles90, bins=5)
    plt.title("Flip Angle Variation")
    plt.show()


    plt.figure()
    plt.subplot(211)
    plt.plot(acqTimeVec,SQ_n, label="SQ")
    plt.grid(True)
    plt.subplot(212)
    plt.plot(acqTimeVec,TQ_n, label="TQ")
    plt.grid()
    plt.show()
elif newReco == False and singlePulse ==True:

    #times, FID, TQ, SQ = PC.SinglePulseVec()
    imgFID = imag(FIDs)
    FID = real(FIDs)

    FID /= np.amax(real(FID))

    p0 = [T2s, T2f, 0.4, 0.6, 0.01]
    popt, pcov = getFunction.fit_biexponentialSQ(times, FID, p0)

    print(popt)

    plt.figure()
    plt.plot(times, real(FID),  label="FID")
    plt.plot(times, getFunction.getFunc_biexp(times, *popt), label="FIT")
    plt.grid()
    plt.legend(loc='best')
    plt.show()





t1 = time.time()
print("Total Time: s ", t1 - t0)


