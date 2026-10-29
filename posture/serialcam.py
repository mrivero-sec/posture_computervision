"""Camera de la carte ESP32-S3 lue par le cable USB (sketch arduino/cam_serial).

Protocole : le PC envoie 'c', la carte repond "FRAME" + longueur (uint32 LE) + JPEG.
SerialCam imite cv2.VideoCapture (read / isOpened / release / get) pour etre
utilisable comme n'importe quelle source video dans collect.py et run.py.
"""
from __future__ import annotations

import struct
import time

import numpy as np

MAGIC = b"FRAME"
MAX_JPEG = 2_000_000


def open_serial(port: str, baud: int):
    import serial

    if not hasattr(serial, "Serial"):
        raise SystemExit(
            f"Le module 'serial' charge ({getattr(serial, '__file__', '?')}) n'est pas pyserial.\n"
            "Corrige avec :  pip uninstall -y serial  puis  pip install --force-reinstall pyserial"
        )
    # DTR/RTS a False AVANT l'ouverture : sinon le circuit d'auto-reset
    # peut redemarrer la carte ou la bloquer en mode telechargement.
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, 2
    ser.dtr = False
    ser.rts = False
    try:
        ser.open()
    except serial.SerialException as e:
        raise SystemExit(f"Impossible d'ouvrir {port} : {e}\n"
                         "Port deja utilise ? (Moniteur serie Arduino ouvert, autre script en cours)")
    return ser


def read_jpeg(ser, timeout: float = 3.0, log=print) -> bytes | None:
    """Demande une image et renvoie les octets JPEG (None si rien recu)."""
    ser.reset_input_buffer()
    ser.write(b"c")
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:            # cherche l'en-tete, affiche les lignes texte
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if buf.endswith(MAGIC):
            break
        if b == b"\n":
            line = buf.decode("utf-8", "ignore").strip()
            if line and log:
                log(f"[carte] {line}")
            buf = b""
    else:
        return None
    head = ser.read(4)
    if len(head) != 4:
        return None
    (n,) = struct.unpack("<I", head)
    if n == 0 or n > MAX_JPEG:
        return None
    data = ser.read(n)
    return data if len(data) == n else None


class SerialCam:
    def __init__(self, port: str, baud: int = 2_000_000, ser=None, retries: int = 5):
        self.ser = ser if ser is not None else open_serial(port, baud)
        self.retries = retries
        self.last_jpeg: bytes | None = None
        if ser is None:
            time.sleep(1.0)

    def isOpened(self) -> bool:
        return self.ser is not None

    def read(self):
        import cv2

        for i in range(self.retries):
            jpg = read_jpeg(self.ser)
            if jpg is not None:
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.last_jpeg = jpg
                    return True, img
            if i == 1:
                print("Pas d'image recue (sketch cam_serial televerse ? bon port ? Moniteur serie ferme ?)")
        return False, None

    def set(self, *_):
        return False

    def get(self, *_):
        return 0.0

    def release(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None


class Rotated:
    """Enveloppe une source video et tourne chaque image (sens horaire)."""

    def __init__(self, cap, degrees: int):
        import cv2

        self.cap = cap
        self.code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees]

    def read(self):
        import cv2

        ok, img = self.cap.read()
        return (ok, cv2.rotate(img, self.code)) if ok else (ok, img)

    def __getattr__(self, name):
        return getattr(self.cap, name)
