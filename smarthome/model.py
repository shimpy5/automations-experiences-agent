"""The smart-home model: what a home is, and how a command changes it.

Pure data and rules — no network, no AI. Everything the agent eventually does
reads or changes the things defined here. Kept dependency-free (plain
dataclasses) so it's easy to read and test.

Ported from the smartthings-experience-engine project, trimmed to what the
agent needs and made synchronous.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# --- Capability schema -------------------------------------------------------
# A small, demo-sized subset of the real SmartThings capability catalogue. Each
# capability lists the commands it accepts and how each command's arguments are
# constrained. This is the contract the model is held to: validation checks
# every command against it, so a made-up or out-of-range command never runs.
CAPABILITY_SCHEMAS: dict[str, dict[str, Any]] = {
    "switch": {
        "commands": {"on": {}, "off": {}},
        "attribute": "switch",
    },
    "switchLevel": {
        "commands": {"setLevel": {"level": {"type": "int", "min": 0, "max": 100}}},
        "attribute": "level",
    },
    "colorTemperature": {
        "commands": {
            "setColorTemperature": {"temperature": {"type": "int", "min": 2200, "max": 6500}}
        },
        "attribute": "colorTemperature",
    },
    "windowShade": {
        "commands": {"open": {}, "close": {}, "pause": {}},
        "attribute": "windowShade",
    },
    "thermostatHeatingSetpoint": {
        "commands": {"setHeatingSetpoint": {"setpoint": {"type": "number", "min": 40, "max": 90}}},
        "attribute": "heatingSetpoint",
    },
    "thermostatCoolingSetpoint": {
        "commands": {"setCoolingSetpoint": {"setpoint": {"type": "number", "min": 50, "max": 90}}},
        "attribute": "coolingSetpoint",
    },
    "lock": {
        "commands": {"lock": {}, "unlock": {}},
        "attribute": "lock",
    },
    "mediaPlayback": {
        "commands": {"play": {}, "pause": {}, "stop": {}},
        "attribute": "playbackStatus",
    },
    "audioVolume": {
        "commands": {"setVolume": {"volume": {"type": "int", "min": 0, "max": 100}}},
        "attribute": "volume",
    },
}


# --- Domain types ------------------------------------------------------------
@dataclass
class Device:
    device_id: str
    label: str
    room: str
    capabilities: list[str]
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HomeGraph:
    rooms: list[str]
    devices: list[Device]

    def by_id(self) -> dict[str, Device]:
        return {d.device_id: d for d in self.devices}

    def to_dict(self) -> dict[str, Any]:
        return {"rooms": self.rooms, "devices": [d.to_dict() for d in self.devices]}


@dataclass
class Command:
    """A concrete device command, bound to a real device."""

    device_id: str
    device_label: str
    room: str
    capability: str
    command: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- State transitions -------------------------------------------------------
# How each command changes a device's observable state. The mock home uses this
# to update its in-memory devices; restore_command uses it in reverse to put a
# device back from a saved snapshot.
def apply_command_to_state(
    state: dict[str, Any], capability: str, command: str, args: dict[str, Any]
) -> dict[str, Any]:
    new = dict(state)
    if capability == "switch":
        new["switch"] = "on" if command == "on" else "off"
    elif capability == "switchLevel":
        new["level"] = int(args.get("level", new.get("level", 0)))
        new["switch"] = "on"  # setting a level implies the light is on
    elif capability == "colorTemperature":
        new["colorTemperature"] = int(args.get("temperature", new.get("colorTemperature", 4000)))
    elif capability == "windowShade":
        new["windowShade"] = {"open": "open", "close": "closed", "pause": "partially open"}.get(
            command, new.get("windowShade", "open")
        )
    elif capability == "thermostatHeatingSetpoint":
        new["heatingSetpoint"] = args.get("setpoint", new.get("heatingSetpoint"))
    elif capability == "thermostatCoolingSetpoint":
        new["coolingSetpoint"] = args.get("setpoint", new.get("coolingSetpoint"))
    elif capability == "lock":
        new["lock"] = "locked" if command == "lock" else "unlocked"
    elif capability == "mediaPlayback":
        new["playbackStatus"] = {"play": "playing", "pause": "paused", "stop": "stopped"}.get(
            command, new.get("playbackStatus", "stopped")
        )
    elif capability == "audioVolume":
        new["volume"] = int(args.get("volume", new.get("volume", 0)))
    return new


def restore_command(capability: str, before: dict[str, Any]):
    """Given a device's prior state, return the (command, args) that puts this
    capability back the way it was — or None if it can't be restored."""
    if capability == "switch":
        return ("on", {}) if before.get("switch") == "on" else ("off", {})
    if capability == "switchLevel" and "level" in before:
        return "setLevel", {"level": int(before["level"])}
    if capability == "colorTemperature" and "colorTemperature" in before:
        return "setColorTemperature", {"temperature": int(before["colorTemperature"])}
    if capability == "windowShade":
        cmd = {"open": "open", "closed": "close", "partially open": "pause"}.get(
            before.get("windowShade", "open")
        )
        return (cmd, {}) if cmd else None
    if capability == "thermostatHeatingSetpoint" and "heatingSetpoint" in before:
        return "setHeatingSetpoint", {"setpoint": before["heatingSetpoint"]}
    if capability == "thermostatCoolingSetpoint" and "coolingSetpoint" in before:
        return "setCoolingSetpoint", {"setpoint": before["coolingSetpoint"]}
    if capability == "lock":
        return ("lock", {}) if before.get("lock") == "locked" else ("unlock", {})
    if capability == "mediaPlayback":
        cmd = {"playing": "play", "paused": "pause", "stopped": "stop"}.get(
            before.get("playbackStatus", "stopped")
        )
        return (cmd, {}) if cmd else None
    if capability == "audioVolume" and "volume" in before:
        return "setVolume", {"volume": int(before["volume"])}
    return None
