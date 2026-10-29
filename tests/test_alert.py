from posture.alert import ABSENT, ALERT, OK, WARNING, AlertParams, AlertState


def run(seq, **kw):
    st = AlertState(AlertParams(**kw))
    return [st.update(t, p) for t, p in seq], st


def test_alert_after_trigger_duration():
    seq = [(t, 0.9) for t in range(0, 40)]
    out, _ = run(seq, alpha=1.0, trigger_s=30)
    states = [s for s, _ in out]
    assert states[0] == WARNING  # premiere image deja au-dessus du seuil
    assert states[29] == WARNING and states[30] == ALERT
    assert sum(b for _, b in out) == 1  # un seul bip (intervalle 60 s)


def test_hysteresis_no_flicker():
    seq = [(t, 0.9) for t in range(0, 35)] + [(t, 0.55) for t in range(35, 60)]
    out, _ = run(seq, alpha=1.0, trigger_s=30)
    assert all(s == ALERT for s, _ in out[30:])  # 0.55 entre seuil bas et haut : on reste


def test_recovery_needs_hold():
    seq = [(t, 0.9) for t in range(0, 35)] + [(t, 0.1) for t in range(35, 45)]
    out, _ = run(seq, alpha=1.0, trigger_s=30, exit_hold_s=5)
    states = [s for s, _ in out]
    assert states[39] == ALERT and states[40] == OK


def test_absent_and_snooze():
    st = AlertState(AlertParams(alpha=1.0, trigger_s=0, absent_timeout_s=10))
    assert st.update(0, None)[0] == ABSENT
    st.snooze(0)
    s, beep = st.update(1, 0.95)
    assert s == ALERT and not beep
    assert st.update(5, None)[0] == ALERT   # absence courte : on garde l'etat
    assert st.update(20, None)[0] == ABSENT
    assert st.update(21, 0.1)[0] == OK
