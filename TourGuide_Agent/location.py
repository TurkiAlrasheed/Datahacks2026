"""Location detection and reverse geocoding for the tour guide.

Primary target hardware is the Arduino UNO Q (Linux Debian + Python 3 on a
quad-core SoC). The device can detect location either from attached GPS
hardware (USB/UART, NMEA sentences) or, if none is present and Wi-Fi is
available, by IP-based geolocation. Coordinates are then reverse-geocoded to
a human-readable place name via OpenStreetMap's Nominatim — free, no API
key, but subject to a 1 req/sec policy and requires a descriptive
User-Agent.

Typical use:

    from location import get_location
    loc = get_location()
    print(loc.display_name, loc.lat, loc.lon)

All network calls have short timeouts and fail silently, cascading to the
next method. If every method fails, `Location.unknown()` is returned so the
agent can keep running without geographic context.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests


NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
# ipapi.co — free, no key, ~1000 anonymous req/day. Swap for another provider
# (ip-api.com, ipinfo.io) here if the quota becomes an issue.
IP_GEOLOCATE_URL = "https://ipapi.co/json/"
# Nominatim's usage policy requires identifying the app.
USER_AGENT = "EcoTourGuide/1.0 (DataHacks 2026 hackathon project)"

_DEFAULT_TIMEOUT = 5.0


@dataclass
class Location:
    display_name: str
    lat: float | None = None
    lon: float | None = None
    country: str | None = None
    region: str | None = None

    @classmethod
    def unknown(cls) -> "Location":
        return cls(display_name="Unknown location")

    def __str__(self) -> str:
        return self.display_name

    @property
    def has_coords(self) -> bool:
        return self.lat is not None and self.lon is not None


def _read_gps_serial(port: str, baud: int = 9600, timeout: float = 5.0) -> tuple[float, float] | None:
    """Parse NMEA from a GPS module on `port`. Returns (lat, lon) or None.

    Reads lines until a valid GGA/RMC fix or the timeout elapses. Needs
    `pyserial` and `pynmea2` — both optional; install only if GPS is wired in.
    """
    try:
        import serial  # pyserial
        import pynmea2
    except ImportError:
        return None

    try:
        with serial.Serial(port, baud, timeout=1.0) as ser:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    raw = ser.readline().decode("ascii", errors="ignore").strip()
                except Exception:
                    continue
                if not raw.startswith("$") or ("GGA" not in raw and "RMC" not in raw):
                    continue
                try:
                    msg = pynmea2.parse(raw)
                except pynmea2.ParseError:
                    continue
                lat = getattr(msg, "latitude", None)
                lon = getattr(msg, "longitude", None)
                if lat and lon:
                    return float(lat), float(lon)
    except Exception:
        return None
    return None


def _ip_geolocate() -> tuple[float, float] | None:
    try:
        r = requests.get(IP_GEOLOCATE_URL, timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except Exception:
        return None


def _reverse_geocode(lat: float, lon: float) -> dict | None:
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14},
            headers={"User-Agent": USER_AGENT},
            timeout=_DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_location(
    lat: float | None = None,
    lon: float | None = None,
    *,
    gps_port: str | None = None,
) -> Location:
    """Resolve the current location and reverse-geocode to a place name.

    Resolution order:
      1. Explicit `lat`/`lon` arguments.
      2. GPS on `gps_port` (or `GPS_PORT` env var) — only if pyserial/pynmea2 are installed.
      3. IP-based geolocation via ipapi.co.

    Returns `Location.unknown()` if every method fails.
    """
    coords: tuple[float, float] | None = None
    if lat is not None and lon is not None:
        coords = (float(lat), float(lon))
    else:
        port = gps_port or os.environ.get("GPS_PORT")
        if port:
            coords = _read_gps_serial(port)
        if coords is None:
            coords = _ip_geolocate()

    if coords is None:
        return Location.unknown()

    lat_, lon_ = coords
    geo = _reverse_geocode(lat_, lon_) or {}
    display = geo.get("display_name") or f"{lat_:.5f}, {lon_:.5f}"
    addr = geo.get("address") or {}
    return Location(
        display_name=display,
        lat=lat_,
        lon=lon_,
        country=addr.get("country"),
        region=addr.get("state") or addr.get("region") or addr.get("county"),
    )
