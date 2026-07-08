"""Hardware tuner for the reSpeaker XMOS XVF3800 (macOS / libusb).

This module controls the XVF3800's on-chip DSP parameters over USB vendor
control transfers (VID 0x2886, PID 0x001A) -- NOT the CDC serial port, which
does not expose a console on this firmware. The parameter table below was
enumerated from the official Seeed `xvf_host.py --list` output and verified by
reading the live device (firmware 2.0.6). Every RESID/CMDID pair is real.

Honesty rules (per VISION.md "don't fake it"):
- We only ever write parameters the firmware actually exposes. Unknown or
  read-only params raise `HardwareTunerError` -- no silent no-ops.
- Writes are VOLATILE by default (the chip reverts to flash defaults on
  reboot). We never call REBOOT or SAVE_CONFIGURATION unless explicitly asked,
  so tuning experiments cannot brick or persistently mutate the board.
- Every write is verified by an immediate read-back; a mismatch raises.

The two capabilities the research scaffold needs:
1. RAW_LINEAR profile: assert AGC off and noise-suppression floors at 0, so
   channel energies keep a linear relationship with acoustic pressure.
2. Manual beam steering: enable fixed-beam mode and point the "acoustic laser"
   at a top-down-selected azimuth. The Python saliency loop can compute the
   target angle and call `steer_to(azimuth_deg)`.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

try:  # pyserial-free; backend resolved explicitly for macOS libusb
    import libusb_package  # noqa: F401  (side-effect: extracts bundled dylib)
    import usb.backend.libusb1
    import usb.core
    import usb.util

    _HAVE_USB = True
except Exception:  # pragma: no cover - import guard for headless test envs
    _HAVE_USB = False

VENDOR_ID = 0x2886
PRODUCT_ID = 0x001A

# Word byte widths per type (from the firmware param table).

@dataclass(frozen=True)
class ParamSpec:
    name: str
    resid: int
    cmdid: int
    length: int  # in words
    type: str
    access: str  # "ro", "wo", "rw"


# Verified subset of the XVF3800 parameter registry (Seeed xvf_host --list).
# Only params the scaffold actually uses are included; extend from --list as needed.
PARAMETERS: dict[str, ParamSpec] = {
    "VERSION": ParamSpec("VERSION", 48, 0, 3, "uint8", "ro"),
    "REBOOT": ParamSpec("REBOOT", 48, 7, 1, "uint8", "wo"),
    "SAVE_CONFIGURATION": ParamSpec("SAVE_CONFIGURATION", 48, 9, 1, "uint8", "wo"),
    "CLEAR_CONFIGURATION": ParamSpec("CLEAR_CONFIGURATION", 48, 10, 1, "uint8", "wo"),
    "AEC_HPFONOFF": ParamSpec("AEC_HPFONOFF", 33, 1, 1, "int32", "rw"),
    "AEC_FIXEDBEAMSONOFF": ParamSpec("AEC_FIXEDBEAMSONOFF", 33, 37, 1, "int32", "rw"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": ParamSpec(
        "AEC_FIXEDBEAMSAZIMUTH_VALUES", 33, 81, 2, "radians", "rw"
    ),
    "AEC_FIXEDBEAMSGATING": ParamSpec("AEC_FIXEDBEAMSGATING", 33, 83, 1, "uint8", "rw"),
    "DOA_VALUE": ParamSpec("DOA_VALUE", 20, 18, 2, "float", "ro"),
    "AUDIO_MGR_SELECTED_AZIMUTHS": ParamSpec(
        "AUDIO_MGR_SELECTED_AZIMUTHS", 35, 11, 2, "radians", "ro"
    ),
    "PP_AGCONOFF": ParamSpec("PP_AGCONOFF", 17, 10, 1, "int32", "rw"),
    "PP_AGCMAXGAIN": ParamSpec("PP_AGCMAXGAIN", 17, 11, 1, "float", "rw"),
    "PP_AGCDESIREDLEVEL": ParamSpec("PP_AGCDESIREDLEVEL", 17, 12, 1, "float", "rw"),
    "PP_MIN_NS": ParamSpec("PP_MIN_NS", 17, 21, 1, "float", "rw"),
    "PP_MIN_NN": ParamSpec("PP_MIN_NN", 17, 22, 1, "float", "rw"),
}

# Persisted/mutation commands -- never touched unless explicitly requested.
_PROTECTED = {"REBOOT", "SAVE_CONFIGURATION", "CLEAR_CONFIGURATION"}


class HardwareTunerError(Exception):
    """Raised on any unsupported, read-only, or unverified tuner operation."""


@dataclass
class DoAReading:
    """Firmware DOA_VALUE register readback.

    On XVF3800 firmware 2.0.6 this behaves as beam-azimuth state, not a proven
    acoustic-source bearing. Keep the name for the firmware register, but do not
    interpret it as source localization without a separate geometry proof.
    """

    azimuth_deg: float
    speech_detected: bool


def _resolve_backend() -> Any:
    if not _HAVE_USB:
        raise HardwareTunerError("pyusb/libusb not available in this environment")
    import os

    pkg_path = os.path.dirname(libusb_package.__file__)  # type: ignore[arg-type]
    dylib = os.path.join(pkg_path, "libusb-1.0.dylib")
    return usb.backend.libusb1.get_backend(find_library=lambda _: dylib)  # type: ignore[union-attr]


def _decode(resp: bytes, typ: str) -> list[float]:
    b = resp
    if typ == "uint8":
        return list(b[1:])
    if typ in ("int32", "float", "radians"):
        n = (len(b) - 1) // 4
        fmt = "<" + ("i" if typ == "int32" else "f") * n
        return [round(v, 4) for v in struct.unpack(fmt, b[1 : 1 + n * 4])]
    if typ == "uint16":
        n = (len(b) - 1) // 2
        return list(struct.unpack("<" + "H" * n, b[1 : 1 + n * 2]))
    return list(b)


def _encode(values: list[float], typ: str) -> bytes:
    out = b""
    if typ in ("int32",):
        for v in values:
            out += struct.pack("<i", int(v))
    elif typ in ("float", "radians"):
        for v in values:
            out += struct.pack("<f", float(v))
    elif typ == "uint8":
        for v in values:
            out += int(v).to_bytes(1, "little")
    elif typ == "uint16":
        for v in values:
            out += int(v).to_bytes(2, "little")
    return out


class HardwareTuner:
    """Thin, honest wrapper over the XVF3800 USB control interface."""

    # Distinguishes a real USB-backed tuner from the bus-free _NullTuner used in
    # synthetic/test runs. The LiveListener reports this so a JSON run can state
    # honestly whether the gaze loop touched real silicon.
    is_real_hardware = True

    def __init__(self, dev: Any = None) -> None:
        if dev is not None:
            self._dev = dev
            self._owns_device = False
        else:
            if not _HAVE_USB:
                raise HardwareTunerError("pyusb/libusb not available")
            backend = _resolve_backend()
            found = usb.core.find(
                idVendor=VENDOR_ID, idProduct=PRODUCT_ID, backend=backend
            )
            if found is None:
                raise HardwareTunerError(
                    f"XVF3800 not found (VID={VENDOR_ID:#06x} PID={PRODUCT_ID:#06x})"
                )
            self._dev = found
            self._owns_device = True

    # -- low-level ---------------------------------------------------------
    def read(self, name: str) -> list[float]:
        spec = self._spec(name, allow_write=False)
        if spec.type in ("uint8", "char"):
            size = spec.length + 1
        elif spec.type == "uint16":
            size = spec.length * 2 + 1
        else:
            size = spec.length * 4 + 1
        resp = self._dev.ctrl_transfer(
            usb.util.CTRL_IN
            | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            0x80 | spec.cmdid,
            spec.resid,
            size,
            2000,
        )
        raw = bytes(resp)
        # Status-byte retry loop (mirrors Seeed xvf_host.py ground truth).
        # Status 64 = SERVICER_COMMAND_RETRY -> re-read. Status 66 is the
        # firmware's normal "data present" code for the localization params
        # (DOA_VALUE / selected azimuths) on 2.0.6; the Seeed SDK ignores the
        # status byte entirely and decodes the payload, so we accept 0 AND 66
        # as valid data. Other non-zero statuses are treated as errors.
        attempts = 0
        while raw[0] == 64 and attempts < 100:  # SERVICER_COMMAND_RETRY
            attempts += 1
            import time

            time.sleep(0.01)
            raw = bytes(
                self._dev.ctrl_transfer(
                    usb.util.CTRL_IN
                    | usb.util.CTRL_TYPE_VENDOR
                    | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,
                    0x80 | spec.cmdid,
                    spec.resid,
                    size,
                    2000,
                )
            )
        if raw[0] not in (0, 66):
            raise HardwareTunerError(
                f"read {name} returned status {raw[0]} (not success)"
            )
        return _decode(raw, spec.type)

    def write(self, name: str, values: list[float]) -> list[float]:
        spec = self._spec(name, allow_write=True)
        if name in _PROTECTED:
            raise HardwareTunerError(
                f"{name} is a protected/mutation command; not written automatically"
            )
        if spec.access == "ro":
            raise HardwareTunerError(f"{name} is read-only")
        if len(values) != spec.length:
            raise HardwareTunerError(
                f"{name} expects {spec.length} value(s), got {len(values)}"
            )
        payload = _encode(values, spec.type)
        self._dev.ctrl_transfer(
            usb.util.CTRL_OUT
            | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            spec.cmdid,
            spec.resid,
            payload,
            2000,
        )
        # Verify by read-back.
        actual = self.read(name)
        if [round(v, 4) for v in actual] != [round(v, 4) for v in values]:
            raise HardwareTunerError(
                f"write to {name} unverified: wrote {values}, read back {actual}"
            )
        return actual

    def _spec(self, name: str, allow_write: bool) -> ParamSpec:
        spec = PARAMETERS.get(name)
        if spec is None:
            raise HardwareTunerError(f"unknown parameter: {name}")
        return spec

    # -- profiles ----------------------------------------------------------
    def apply_raw_linear(self) -> dict[str, list[float]]:
        """Assert AGC off and noise-suppression floors at 0 (linear capture)."""
        result: dict[str, list[float]] = {}
        result["PP_AGCONOFF"] = self.write("PP_AGCONOFF", [0])
        result["PP_MIN_NS"] = self.write("PP_MIN_NS", [0.0])
        result["PP_MIN_NN"] = self.write("PP_MIN_NN", [0.0])
        return result

    def steer_to(self, azimuth_deg: float) -> list[float]:
        """Enable fixed-beam mode and point beam 1 at `azimuth_deg` (0..359)."""
        if not 0 <= azimuth_deg < 360:
            raise HardwareTunerError("azimuth_deg must be in [0, 360)")
        rad = math.radians(azimuth_deg)
        # Set beam 1 azimuth; leave beam 2 at 0 (unused for single-target steer).
        self.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [rad, 0.0])
        self.write("AEC_FIXEDBEAMSONOFF", [1])
        return self.read("AEC_FIXEDBEAMSAZIMUTH_VALUES")

    def release_beam(self) -> list[float]:
        """Return to automatic (free-running) beam selection."""
        return self.write("AEC_FIXEDBEAMSONOFF", [0])

    def snapshot(self) -> dict[str, Any]:
        """Read back the research-relevant params for logging/audit."""
        keys = [
            "VERSION",
            "PP_AGCONOFF",
            "PP_MIN_NS",
            "PP_MIN_NN",
            "AEC_FIXEDBEAMSONOFF",
            "AEC_FIXEDBEAMSAZIMUTH_VALUES",
            "AEC_HPFONOFF",
        ]
        return {k: self.read(k) for k in keys}

    def read_doa(self) -> "DoAReading":
        """Read the firmware's DOA_VALUE register.

        Geometry testing showed that on firmware 2.0.6 this register echoes the
        beam's own steered azimuth (released -> 0, steer 350 -> 350, steer 100
        -> 100). It is useful as beam-state readback, but it is NOT evidence of
        acoustic-source localization. The real processed source-DoA is exposed
        at `AUDIO_MGR_SELECTED_AZIMUTHS[0]`; it stayed NAN for tone stimuli but
        became finite when a macOS TTS speech stimulus fired the firmware VAD.

        NOTE on the firmware wire format (firmware 2.0.6): DOA_VALUE is served
        as a 2-float payload [azimuth_rad, flag] with status byte 66 (NOT 0).
        The Seeed SDK decodes the payload ignoring status; we mirror that. So
        raw[0] = azimuth in RADIANS (convert to degrees), raw[1] = speech flag
        (non-zero => detected).
        """
        raw = self.read("DOA_VALUE")  # [azimuth_rad, flag]
        azimuth_rad = float(raw[0]) if len(raw) > 0 else 0.0
        speech = bool(raw[1]) if len(raw) > 1 else False
        azimuth_deg = (math.degrees(azimuth_rad) % 360 + 360) % 360
        return DoAReading(azimuth_deg=azimuth_deg, speech_detected=speech)

    def read_selected_azimuths(self) -> tuple[float, float]:
        """Processed azimuths in radians: [processed_DoA, auto_select_beam].

        Returns NAN for a component if neither fixed beam contains speech
        (per firmware semantics). Useful as a richer localization signal.
        """
        vals = self.read("AUDIO_MGR_SELECTED_AZIMUTHS")
        return (float(vals[0]), float(vals[1]))

