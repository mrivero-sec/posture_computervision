// Machine a etats d'alerte (miroir de posture/alert.py).
#pragma once

namespace posture {

enum class State { ABSENT, OK, WARNING, ALERT };

struct AlertParams {
    double alpha = 0.3;
    double enter_thr = 0.70;
    double exit_thr = 0.40;
    double trigger_s = 30.0;
    double exit_hold_s = 5.0;
    double beep_interval_s = 60.0;
    double absent_timeout_s = 10.0;
    double snooze_s = 900.0;
    bool sound = true;
};

class AlertState {
public:
    explicit AlertState(const AlertParams &p) : p_(p) {}
    void set_params(const AlertParams &p) { p_ = p; reset(); }
    void reset();
    void snooze(double now) { snooze_until_ = now + p_.snooze_s; }
    bool snoozed(double now) const { return now < snooze_until_; }
    // has_person = false si personne absente. Retourne true s'il faut biper.
    bool update(double now, bool has_person, double prob);
    State state() const { return state_; }
    double ema() const { return ema_; }
    const AlertParams &params() const { return p_; }

private:
    AlertParams p_;
    State state_ = State::ABSENT;
    double ema_ = 0.0;
    bool bad_set_ = false, good_set_ = false, seen_ = false;
    double bad_since_ = 0.0, good_since_ = 0.0, last_seen_ = 0.0;
    double last_beep_ = -1e18, snooze_until_ = -1e18;
};

const char *state_name(State s);

}  // namespace posture
