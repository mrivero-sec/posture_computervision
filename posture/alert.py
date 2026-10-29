"""Machine a etats d'alerte : lissage EMA + hysteresis + duree minimale.

Miroir de firmware/main/alert.cpp.
Etats : ABSENT -> OK -> WARNING -> ALERT, retour OK apres `exit_hold_s`
sous le seuil bas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ABSENT, OK, WARNING, ALERT = "ABSENT", "OK", "WARNING", "ALERT"


@dataclass
class AlertParams:
    alpha: float = 0.3          # lissage EMA (1 = pas de lissage)
    enter_thr: float = 0.70     # seuil haut
    exit_thr: float = 0.40      # seuil bas
    trigger_s: float = 30.0     # duree au-dessus du seuil haut avant ALERT
    exit_hold_s: float = 5.0    # duree sous le seuil bas avant retour OK
    beep_interval_s: float = 60.0
    absent_timeout_s: float = 10.0
    snooze_s: float = 900.0

    @classmethod
    def from_dict(cls, d: dict) -> "AlertParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AlertState:
    p: AlertParams = field(default_factory=AlertParams)
    state: str = ABSENT
    ema: float = 0.0
    bad_since: float | None = None
    good_since: float | None = None
    last_seen: float | None = None
    last_beep: float = -1e18
    snooze_until: float = -1e18

    def snooze(self, now: float) -> None:
        self.snooze_until = now + self.p.snooze_s

    def update(self, now: float, prob: float | None) -> tuple[str, bool]:
        """prob = None si personne absente. Retourne (etat, bip_a_jouer)."""
        p = self.p
        if prob is None:
            if self.last_seen is None or now - self.last_seen >= p.absent_timeout_s:
                self.state, self.ema = ABSENT, 0.0
                self.bad_since = self.good_since = None
            return self.state, False

        self.last_seen = now
        if self.state == ABSENT:
            self.ema = prob
            self.state = OK
        else:
            self.ema = p.alpha * prob + (1.0 - p.alpha) * self.ema

        if self.ema >= p.enter_thr:
            self.good_since = None
            if self.bad_since is None:
                self.bad_since = now
            if self.state == OK:
                self.state = WARNING
            if self.state == WARNING and now - self.bad_since >= p.trigger_s:
                self.state = ALERT
        elif self.ema <= p.exit_thr:
            self.bad_since = None
            if self.good_since is None:
                self.good_since = now
            if self.state == WARNING or (self.state == ALERT and now - self.good_since >= p.exit_hold_s):
                self.state = OK
        else:
            # zone d'hysteresis : on garde l'etat, le chrono "mauvais" continue en WARNING
            self.good_since = None

        beep = False
        if self.state == ALERT and now >= self.snooze_until and now - self.last_beep >= p.beep_interval_s:
            beep = True
            self.last_beep = now
        return self.state, beep
