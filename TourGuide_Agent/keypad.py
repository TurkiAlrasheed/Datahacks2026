"""4x4 matrix keypad reader for Arduino UNO Q via Linux sysfs GPIO.

Keypad layout:
    1  2  3  A
    4  5  6  B
    7  8  9  C
    *  0  #  D

Wiring (matches the Arduino sketch):
    Row pins: D9, D8, D7, D6  (outputs, active LOW)
    Col pins: D5, D4, D3, D2  (inputs, pulled HIGH)

Usage:
    from keypad import Keypad
    kp = Keypad()
    key = kp.read()  # returns pressed key or None

Requires: the Linux GPIO numbers for D2-D9 on the UNO Q.
Run `gpioinfo` or `cat /sys/kernel/debug/gpio` on the board to find them.
Adjust GPIO_MAP below if the numbers differ on your board.
"""

from __future__ import annotations

import time
from pathlib import Path

# ── GPIO pin mapping ──────────────────────────────────────────────────────────
# Map Arduino digital pin names to Linux GPIO numbers.
# >>> IMPORTANT: You may need to adjust these! <<<
# SSH into the Arduino and run:  gpioinfo  or  cat /sys/kernel/debug/gpio
# to find the actual Linux GPIO number for each digital pin.
#
# Common UNO Q mappings (verify on your board):
GPIO_MAP = {
    "D2": 2,
    "D3": 3,
    "D4": 4,
    "D5": 5,
    "D6": 6,
    "D7": 7,
    "D8": 8,
    "D9": 9,
}

ROWS_KEYS = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"],
]

ROW_PINS = [GPIO_MAP["D9"], GPIO_MAP["D8"], GPIO_MAP["D7"], GPIO_MAP["D6"]]
COL_PINS = [GPIO_MAP["D5"], GPIO_MAP["D4"], GPIO_MAP["D3"], GPIO_MAP["D2"]]

SYSFS = Path("/sys/class/gpio")


def _export(gpio: int) -> None:
    pin_path = SYSFS / f"gpio{gpio}"
    if not pin_path.exists():
        (SYSFS / "export").write_text(str(gpio))
        time.sleep(0.05)


def _unexport(gpio: int) -> None:
    pin_path = SYSFS / f"gpio{gpio}"
    if pin_path.exists():
        (SYSFS / "unexport").write_text(str(gpio))


def _set_direction(gpio: int, direction: str) -> None:
    (SYSFS / f"gpio{gpio}" / "direction").write_text(direction)


def _write(gpio: int, value: int) -> None:
    (SYSFS / f"gpio{gpio}" / "value").write_text(str(value))


def _read(gpio: int) -> int:
    return int((SYSFS / f"gpio{gpio}" / "value").read_text().strip())


class Keypad:
    """4x4 matrix keypad scanner using Linux sysfs GPIO."""

    def __init__(self) -> None:
        for pin in ROW_PINS:
            _export(pin)
            _set_direction(pin, "out")
            _write(pin, 1)

        for pin in COL_PINS:
            _export(pin)
            _set_direction(pin, "in")

    def read(self) -> str | None:
        """Scan the keypad and return the pressed key, or None."""
        for row_idx, row_pin in enumerate(ROW_PINS):
            _write(row_pin, 0)
            time.sleep(0.001)

            for col_idx, col_pin in enumerate(COL_PINS):
                if _read(col_pin) == 0:
                    _write(row_pin, 1)
                    return ROWS_KEYS[row_idx][col_idx]

            _write(row_pin, 1)

        return None

    def wait_for_key(self, poll_interval: float = 0.05) -> str:
        """Block until a key is pressed, then wait for release. Returns the key."""
        while True:
            key = self.read()
            if key is not None:
                while self.read() is not None:
                    time.sleep(poll_interval)
                return key
            time.sleep(poll_interval)

    def cleanup(self) -> None:
        """Unexport all GPIO pins."""
        for pin in ROW_PINS + COL_PINS:
            try:
                _unexport(pin)
            except Exception:
                pass

    def __del__(self) -> None:
        self.cleanup()


if __name__ == "__main__":
    print("Keypad test — press keys (Ctrl+C to exit)")
    print(f"Row GPIO pins: {ROW_PINS}")
    print(f"Col GPIO pins: {COL_PINS}")
    print()

    kp = Keypad()
    try:
        while True:
            key = kp.read()
            if key:
                print(f"Key pressed: {key}")
                while kp.read() is not None:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        kp.cleanup()
