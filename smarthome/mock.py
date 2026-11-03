"""A simulated home.

Runs with no hardware and no account, so the whole agent can be exercised on a
laptop. State lives in memory and reacts to commands exactly as a real home
would.

Extra for the agent: `fail_device_ids` makes chosen devices raise on execute,
so we can test the "a device fails -> the model decides skip or retry" path
without breaking real hardware.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable, Optional

from smarthome.model import Device, HomeGraph, apply_command_to_state


class DeviceError(Exception):
    """Raised when a device won't accept a command (simulated failure)."""


def _default_devices() -> list[Device]:
    return [
        Device("lr-ceiling", "Living Room Ceiling Light", "Living Room",
               ["switch", "switchLevel", "colorTemperature"],
               {"switch": "on", "level": 80, "colorTemperature": 4000}),
        Device("lr-lamp", "Living Room Lamp", "Living Room",
               ["switch", "switchLevel", "colorTemperature"],
               {"switch": "on", "level": 70, "colorTemperature": 3500}),
        Device("lr-shades", "Living Room Shades", "Living Room",
               ["windowShade"], {"windowShade": "open"}),
        Device("lr-tv", "Living Room TV", "Living Room",
               ["switch", "mediaPlayback", "audioVolume"],
               {"switch": "off", "playbackStatus": "stopped", "volume": 20}),
        Device("thermostat", "Thermostat", "Hallway",
               ["thermostatHeatingSetpoint", "thermostatCoolingSetpoint"],
               {"heatingSetpoint": 68, "coolingSetpoint": 74}),
        Device("br-light", "Bedroom Light", "Bedroom",
               ["switch", "switchLevel", "colorTemperature"],
               {"switch": "off", "level": 60, "colorTemperature": 3000}),
        Device("br-shades", "Bedroom Shades", "Bedroom",
               ["windowShade"], {"windowShade": "open"}),
        Device("kitchen-light", "Kitchen Light", "Kitchen",
               ["switch", "switchLevel"], {"switch": "off", "level": 100}),
        Device("coffee-maker", "Coffee Maker", "Kitchen",
               ["switch"], {"switch": "off"}),
        Device("front-lock", "Front Door Lock", "Hallway",
               ["lock"], {"lock": "unlocked"}),
    ]


class MockHome:
    def __init__(
        self,
        devices: Optional[list[Device]] = None,
        fail_device_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self._devices: dict[str, Device] = {
            d.device_id: d for d in (devices or _default_devices())
        }
        # Devices that will refuse every command, to simulate a dead/unreachable
        # device in tests and demos.
        self._fail: set[str] = set(fail_device_ids or [])

    def get_home_graph(self) -> HomeGraph:
        # Hand back deep copies so callers can't mutate our internal state.
        devices = [copy.deepcopy(d) for d in self._devices.values()]
        rooms = sorted({d.room for d in devices})
        return HomeGraph(rooms=rooms, devices=devices)

    def execute_command(
        self, device_id: str, capability: str, command: str, args: dict[str, Any]
    ) -> None:
        device = self._devices.get(device_id)
        if device is None:
            raise DeviceError(f"unknown device: {device_id}")
        if device_id in self._fail:
            raise DeviceError(f"{device.label} is not responding")
        device.state = apply_command_to_state(device.state, capability, command, args)
