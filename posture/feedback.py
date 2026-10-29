"""Retour utilisateur cote PC : couleur type LED + bip (simule le materiel)."""
from __future__ import annotations

import sys
import threading

from .alert import ABSENT, ALERT, OK, WARNING

# BGR (OpenCV)
STATE_COLOR = {ABSENT: (90, 90, 90), OK: (0, 200, 0), WARNING: (0, 165, 255), ALERT: (0, 0, 255)}


def beep(freq: int = 1800, ms: int = 250) -> None:
    def _run():
        if sys.platform.startswith("win"):
            import winsound

            winsound.Beep(freq, ms)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()

    threading.Thread(target=_run, daemon=True).start()
