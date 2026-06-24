"""Unit + integration tests for the channel shift simulator."""
import numpy as np
import pytest
from misregistration.simulation import ChannelShiftSimulator, SimulationResult


def _make_rgb(size=64):
    rng = np.random.default_rng(0)
    return rng.integers(40, 220, (size, size, 3), dtype=np.uint8)


def test_simulator_returns_result():
    rgb = _make_rgb()
    sim = ChannelShiftSimulator(shifts_px={"C":(1,0),"M":(0,1),"Y":(-1,0),"K":(0,0)})
    result = sim.simulate(rgb, "test")
    assert isinstance(result, SimulationResult)
    assert result.reference_rgb.shape == rgb.shape
    assert result.shifted_rgb.shape == rgb.shape


def test_zero_shift_gives_low_dE():
    rgb = _make_rgb()
    sim = ChannelShiftSimulator(shifts_px={"C":(0,0),"M":(0,0),"Y":(0,0),"K":(0,0)})
    result = sim.simulate(rgb)
    assert result.dE_image < 3.0


def test_large_shift_gives_high_dE():
    rgb = _make_rgb()
    sim = ChannelShiftSimulator(shifts_px={"C":(10,10),"M":(-10,-10),"Y":(10,-10),"K":(0,0)})
    result = sim.simulate(rgb)
    assert result.dE_image > 0.5


def test_rms_keys_present():
    rgb = _make_rgb()
    sim = ChannelShiftSimulator()
    result = sim.simulate(rgb)
    for ch in ("C", "M", "Y"):
        assert ch in result.rms_px


def test_dE_map_shape():
    rgb = _make_rgb()
    sim = ChannelShiftSimulator()
    result = sim.simulate(rgb)
    assert result.dE_map.shape == rgb.shape[:2]
