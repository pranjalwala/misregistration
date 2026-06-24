"""Simulation: synthetic CMYK misregistration for demo/smoke-test mode."""
from .channel_shift_simulator import (
    ChannelShiftSimulator,
    simulate_from_png,
    simulate_from_jpg,
    simulate_from_folder,
    SimulationResult,
)

__all__ = [
    "ChannelShiftSimulator",
    "simulate_from_png",
    "simulate_from_jpg",
    "simulate_from_folder",
    "SimulationResult",
]
