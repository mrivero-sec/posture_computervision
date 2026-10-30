"""ESP32-S3 board camera read over the USB cable (sketch arduino/cam_serial).

Protocol: the PC sends 'c', the board replies "FRAME" + length (uint32 LE) + JPEG.
SerialCam mimics cv2.VideoCapture (read / isOpened / release / get) so it can be
used like any other video source in collect.py and run.py.
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
            f"The loaded 'serial' module ({getattr(serial, '__file__', '?')}) is not pyserial.\n"
            "Fix it with:  pip uninstall -y serial  then  pip install --force-reinstall pyserial"
        )
    # DTR/RTS set to False BEFORE opening: otherwise the auto-reset circuit
    # may reboot the board or hold it in download mode.
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, 2
    ser.dtr = False
    ser.rts = False
    try:
        ser.open()
    except serial.SerialException as e:
        raise SystemExit(f"Cannot open {port}: {e}\n"
                         "Port already in use? (Arduino Serial Monitor open, another script running)")
    return ser


def read_jpeg(ser, timeout: float = 3.0, log=print) -> bytes | None:
    """Request a frame and return the JPEG bytes (None if nothing was received)."""
    ser.reset_input_buffer()
    ser.write(b"c")
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:            # look for the header, print text lines
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if buf.endswith(MAGIC):
            break
        if b == b"\n":
            line = buf.decode("utf-8", "ignore").strip()
            if line and log:
                log(f"[board] {line}")
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
                print("No frame received (cam_serial sketch uploaded? right port? Serial Monitor closed?)")
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
    """Wraps a video source and rotates each frame (clockwise)."""

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
