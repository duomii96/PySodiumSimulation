import numpy as np


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




def diagnose_sequence_with180(B0, tauC, wQ, wQbar, Jen,
                               flip90=90.0, flip180=180.0, alpha=0.0,
                               tevo=0.1e-3, tmix=0.1e-3,
                               wShiftRMS=0.0, nPtsevo=66, nPtsmix=34):
    """
    Traces the full TQTPPI+180 state vector at every key checkpoint WITH relaxation.
    Sequence:
      Pulse1 (90°, alpha) → tevo/2 → Pulse180 (180°, alpha+90°) → tevo/2
      → Pulse2 (90°, alpha±90°) → tmix → Pulse3 (90°, 0°) → FID
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

    def set_cache(st, flip, cached_flip, cached_wig):
        st.cachedflipangle = cached_flip
        st.cachedwigner    = cached_wig

    wShiftvec = np.array([0.0])

    st1 = TmnEvoVecDic(B0, tauC, wQ, wQbar, Jen, wShiftvec, wShiftRMS, 0.0)
    st2 = TmnEvoVecDic(B0, tauC, wQ, wQbar, Jen, wShiftvec, wShiftRMS, 0.0)

    # ── Warm up both Wigner caches ────────────────────────────────────────────
    warm = TmnEvoVecDic(B0, tauC, wQ, wQbar, Jen, wShiftvec, wShiftRMS, 0.0)
    warm.pulsebatch(flip90, 0)
    cached_flip90  = warm.cachedflipangle
    cached_wig90   = [w.copy() for w in warm.cachedwigner]

    warm.pulsebatch(flip180, 0)
    cached_flip180 = warm.cachedflipangle
    cached_wig180  = [w.copy() for w in warm.cachedwigner]

    # ── Precompute relaxation caches ──────────────────────────────────────────
    ts_half = np.linspace(0, tevo / 2, nPtsevo)
    ts_mix  = np.linspace(0, tmix,     nPtsmix)

    cfhalf = st1.precompute_fJen(ts_half)
    cfhalf = st1.add_phase_factors(cfhalf)
    cfmix  = st1.precompute_fJen(ts_mix)
    cfmix  = st1.add_phase_factors(cfmix)

    print(f"\n{'='*62}")
    print(f"SEQUENCE DIAGNOSTICS (with 180° pulse)")
    print(f"  flip90={flip90}°  flip180={flip180}°  alpha={alpha}°")
    print(f"  tevo={tevo*1e3:.3f}ms (2×{tevo/2*1e3:.3f}ms)  tmix={tmix*1e3:.3f}ms")
    print(f"  Pulse1: {alpha}° | Pulse180: {alpha+90}° | "
          f"Pulse2: st1={alpha+90}°/st2={alpha-90}° | Pulse3: 0°")
    print(f"{'='*62}")

    # ── [0] Initial ───────────────────────────────────────────────────────────
    st1.reset(); st2.reset()
    print_state(st1, "[0] INITIAL (thermal equilibrium)")

    # ── [1] Pulse 1 (90°) ─────────────────────────────────────────────────────
    set_cache(st1, flip90, cached_flip90, cached_wig90)
    set_cache(st2, flip90, cached_flip90, cached_wig90)
    st1.pulsebatch(flip90, alpha)
    copy_state(st1, st2)
    print_state(st1, f"[1] After Pulse 1 (90°, phase={alpha}°)")

    # ── [2] Evolution first half ───────────────────────────────────────────────
    st1.relax_endpoint_cached(cfhalf)
    copy_state(st1, st2)
    print_state(st1, f"[2] After Evolution 1st half (tevo/2={tevo/2*1e3:.3f}ms)")

    # ── [3] 180° pulse ────────────────────────────────────────────────────────
    set_cache(st1, flip180, cached_flip180, cached_wig180)
    set_cache(st2, flip180, cached_flip180, cached_wig180)
    st1.pulsebatch(flip180, alpha + 90)
    copy_state(st1, st2)
    print_state(st1, f"[3] After Pulse 180° (phase={alpha+90}°)")

    # ── [4] Evolution second half ─────────────────────────────────────────────
    st1.relax_endpoint_cached(cfhalf)
    copy_state(st1, st2)
    print_state(st1, f"[4] After Evolution 2nd half (tevo/2={tevo/2*1e3:.3f}ms)")

    # ── [5] Pulse 2 — branches diverge ───────────────────────────────────────
    set_cache(st1, flip90, cached_flip90, cached_wig90)
    set_cache(st2, flip90, cached_flip90, cached_wig90)
    st1.pulsebatch(flip90, alpha + 90)
    st2.pulsebatch(flip90, alpha - 90)
    print_state(st1, f"[5a] After Pulse 2 st1 (90°, phase={alpha+90}°)")
    print_state(st2, f"[5b] After Pulse 2 st2 (90°, phase={alpha-90}°)")

    # ── [6] Mixing ────────────────────────────────────────────────────────────
    st1.relax_endpoint_cached(cfmix)
    st2.relax_endpoint_cached(cfmix)
    print_state(st1, f"[6a] After Mixing st1 (tmix={tmix*1e3:.3f}ms)")
    print_state(st2, f"[6b] After Mixing st2 (tmix={tmix*1e3:.3f}ms)")

    # ── [7] Pulse 3 ───────────────────────────────────────────────────────────
    set_cache(st1, flip90, cached_flip90, cached_wig90)
    set_cache(st2, flip90, cached_flip90, cached_wig90)
    st1.pulsebatch(flip90, 0)
    st2.pulsebatch(flip90, 0)
    print_state(st1, "[7a] After Pulse 3 st1 (90°, phase=0°)")
    print_state(st2, "[7b] After Pulse 3 st2 (90°, phase=0°)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("KEY COHERENCES AFTER PULSE 3:")
    for label, st in [("st1", st1), ("st2", st2)]:
        T11  = st.Tmnbatch[0, 0, 4]
        T1m1 = st.Tmnbatch[0, 0, 2]
        T31  = st.Tmnbatch[0, 2, 4]
        T33  = st.Tmnbatch[0, 2, 6]
        T3m3 = st.Tmnbatch[0, 2, 0]
        print(f"  {label}: T_{{1,+1}}={T11:.5f}  T_{{1,-1}}={T1m1:.5f}  "
              f"T_{{3,+1}}={T31:.5f}  T_{{3,+3}}={T33:.5f}  T_{{3,-3}}={T3m3:.5f}")

    diff = st1.Tmnbatch[0,0,4] - st2.Tmnbatch[0,0,4]
    sumv = st1.Tmnbatch[0,0,4] + st2.Tmnbatch[0,0,4]
    print(f"\n  st1 - st2  →  T_{{1,1}} = {diff:.5f}  (TQ pathway signal)")
    print(f"  st1 + st2  →  T_{{1,1}} = {sumv:.5f}  (SQ pathway signal)")
    print(f"{'='*62}")

    return st1, st2