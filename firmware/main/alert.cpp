#include "alert.hpp"

namespace posture {

const char *state_name(State s) {
    switch (s) {
        case State::ABSENT: return "ABSENT";
        case State::OK: return "OK";
        case State::WARNING: return "WARNING";
        case State::ALERT: return "ALERT";
    }
    return "?";
}

void AlertState::reset() {
    state_ = State::ABSENT;
    ema_ = 0.0;
    bad_set_ = good_set_ = seen_ = false;
}

bool AlertState::update(double now, bool has_person, double prob) {
    if (!has_person) {
        if (!seen_ || now - last_seen_ >= p_.absent_timeout_s) {
            state_ = State::ABSENT;
            ema_ = 0.0;
            bad_set_ = good_set_ = false;
        }
        return false;
    }
    seen_ = true;
    last_seen_ = now;
    if (state_ == State::ABSENT) {
        ema_ = prob;
        state_ = State::OK;
    } else {
        ema_ = p_.alpha * prob + (1.0 - p_.alpha) * ema_;
    }

    if (ema_ >= p_.enter_thr) {
        good_set_ = false;
        if (!bad_set_) { bad_set_ = true; bad_since_ = now; }
        if (state_ == State::OK) state_ = State::WARNING;
        if (state_ == State::WARNING && now - bad_since_ >= p_.trigger_s) state_ = State::ALERT;
    } else if (ema_ <= p_.exit_thr) {
        bad_set_ = false;
        if (!good_set_) { good_set_ = true; good_since_ = now; }
        if (state_ == State::WARNING || (state_ == State::ALERT && now - good_since_ >= p_.exit_hold_s))
            state_ = State::OK;
    } else {
        good_set_ = false;
    }

    if (state_ == State::ALERT && now >= snooze_until_ && now - last_beep_ >= p_.beep_interval_s) {
        last_beep_ = now;
        return true;
    }
    return false;
}

}  // namespace posture
