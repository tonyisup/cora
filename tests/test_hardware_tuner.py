"""Tests for the XVF3800 hardware tuner (mock USB backend, no hardware)."""

import pytest

from cora_infans.io.hardware_tuner import (
    HardwareTuner,
    HardwareTunerError,
    PARAMETERS,
    _decode,
    _encode,
)


class _MockDevice:
    """In-memory XVF3800: stores param values keyed by (resid, cmdid)."""

    def __init__(self) -> None:
        self.store: dict[tuple[int, int], list[float]] = {}
        # Seed factory defaults observed on the real board (firmware 2.0.6).
        self.store[(17, 10)] = [0]          # PP_AGCONOFF = 0
        self.store[(17, 21)] = [0.0]        # PP_MIN_NS = 0.0
        self.store[(17, 22)] = [0.0]        # PP_MIN_NN = 0.0
        self.store[(33, 37)] = [0]          # AEC_FIXEDBEAMSONOFF = 0
        self.store[(33, 81)] = [0.0, 0.0]   # AEC_FIXEDBEAMSAZIMUTH_VALUES
        self.store[(33, 83)] = [0]          # AEC_FIXEDBEAMSGATING = 0
        self.store[(33, 1)] = [0]           # AEC_HPFONOFF = 0
        self.store[(48, 0)] = [2, 0, 6]     # VERSION
        # DOA_VALUE length 2 float: [azimuth_rad, speech_flag]
        import math
        self.store[(20, 18)] = [math.radians(315.0), 1.0]  # DoA=315deg, speech detected

    def ctrl_transfer(self, bm_request_type, b_request, w_value, w_index, data_or_length, timeout=None):
        import struct

        spec = next(
            s for s in PARAMETERS.values() if s.resid == w_index and s.cmdid == (w_value & 0x7F)
        )
        if bm_request_type & 0x80:  # IN (read)
            vals = self.store.get((w_index, spec.cmdid), [0] * spec.length)
            out = bytes([0])  # status SUCCESS
            for v in vals:
                if spec.type in ("int32",):
                    out += struct.pack("<i", int(v))
                elif spec.type in ("float", "radians"):
                    out += struct.pack("<f", float(v))
                elif spec.type == "uint8":
                    out += int(v).to_bytes(1, "little")
                elif spec.type == "uint16":
                    out += int(v).to_bytes(2, "little")
            return out
        else:  # OUT (write)
            payload = data_or_length if isinstance(data_or_length, (bytes, bytearray)) else b""
            vals = []
            if spec.type in ("int32",):
                vals = list(struct.unpack("<" + "i" * (len(payload) // 4), payload))
            elif spec.type in ("float", "radians"):
                vals = list(struct.unpack("<" + "f" * (len(payload) // 4), payload))
            elif spec.type == "uint8":
                vals = list(payload)
            self.store[(w_index, spec.cmdid)] = vals
            return len(payload)


def _tuner() -> HardwareTuner:
    return HardwareTuner(dev=_MockDevice())


def test_decode_encode_roundtrip() -> None:
    raw = _encode([1.5, -2.0], "float")
    assert _decode(bytes([0]) + raw, "float") == [1.5, -2.0]
    raw_i = _encode([7], "int32")
    assert _decode(bytes([0]) + raw_i, "int32") == [7]


def test_read_returns_factory_defaults() -> None:
    t = _tuner()
    assert t.read("PP_AGCONOFF") == [0]
    assert t.read("PP_MIN_NS") == [0.0]
    assert t.read("VERSION") == [2, 0, 6]


def test_apply_raw_linear_asserts_agc_and_ns_off() -> None:
    t = _tuner()
    res = t.apply_raw_linear()
    assert res["PP_AGCONOFF"] == [0]
    assert res["PP_MIN_NS"] == [0.0]
    assert res["PP_MIN_NN"] == [0.0]
    # Idempotent: re-applying does not error and keeps values.
    assert t.apply_raw_linear()["PP_AGCONOFF"] == [0]


def test_steer_to_enables_fixed_beam_and_sets_azimuth() -> None:
    import math

    t = _tuner()
    got = t.steer_to(90.0)
    # Beam 1 should be at ~90 deg in radians; beam 2 unchanged at 0.
    assert abs(got[0] - math.radians(90.0)) < 1e-3
    assert t.read("AEC_FIXEDBEAMSONOFF") == [1]
    # release returns to auto.
    assert t.release_beam() == [0]


def test_write_verifies_and_rejects_mismatch() -> None:
    # A device that lies on read-back should raise.
    class _LyingDevice(_MockDevice):
        def ctrl_transfer(self, *a, **k):
            out = super().ctrl_transfer(*a, **k)
            # On reads, corrupt the value.
            if a[0] & 0x80:
                return bytes([0, 0xFF, 0xFF, 0xFF, 0xFF]) if len(out) > 1 else out
            return out

    t = HardwareTuner(dev=_LyingDevice())
    try:
        t.write("PP_AGCONOFF", [0])
        assert False, "expected HardwareTunerError on unverified write"
    except HardwareTunerError:
        pass


def test_protected_commands_rejected() -> None:
    t = _tuner()
    for name in ("REBOOT", "SAVE_CONFIGURATION", "CLEAR_CONFIGURATION"):
        try:
            t.write(name, [1])
            assert False, f"expected rejection of {name}"
        except HardwareTunerError:
            pass


def test_read_doa_decodes_true_bearing_and_speech_flag() -> None:
    from cora_infans.io.hardware_tuner import DoAReading

    t = _tuner()
    doa = t.read_doa()
    assert isinstance(doa, DoAReading)
    assert doa.azimuth_deg == pytest.approx(315.0, abs=0.01)
    assert doa.speech_detected is True


def test_unknown_and_readonly_params_rejected() -> None:
    t = _tuner()
    try:
        t.write("NONEXISTENT", [0])
        assert False, "expected rejection of unknown param"
    except HardwareTunerError:
        pass
    try:
        t.write("VERSION", [1, 2, 3])  # read-only
        assert False, "expected rejection of read-only param"
    except HardwareTunerError:
        pass
