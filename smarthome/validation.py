"""The guardrail between the model and the devices.

Whatever the model proposes, every command is checked against the capability
schema and the live home before it can run: the device must exist, support the
capability, the command must be defined for that capability, and the arguments
must be present and in range. Anything else is rejected with a reason — the
model gets the reason back and can fix it, but the bad command never reaches a
device.
"""
from __future__ import annotations

from smarthome.model import CAPABILITY_SCHEMAS, Command, Device


def validate_command(cmd: Command, by_id: dict[str, Device]) -> tuple[bool, str]:
    device = by_id.get(cmd.device_id)
    if device is None:
        return False, f"unknown device {cmd.device_id}"
    if cmd.capability not in device.capabilities:
        return False, f"{device.label} does not support {cmd.capability}"
    schema = CAPABILITY_SCHEMAS.get(cmd.capability)
    if schema is None:
        return False, f"unknown capability {cmd.capability}"
    commands = schema["commands"]
    if cmd.command not in commands:
        return False, f"{cmd.capability} has no command {cmd.command}"
    for name, rule in commands[cmd.command].items():
        if name not in cmd.args:
            return False, f"{cmd.command} missing argument {name}"
        value = cmd.args[name]
        if not isinstance(value, (int, float)):
            return False, f"{name} must be numeric"
        if "min" in rule and value < rule["min"]:
            return False, f"{name}={value} below min {rule['min']}"
        if "max" in rule and value > rule["max"]:
            return False, f"{name}={value} above max {rule['max']}"
    return True, ""


def clean(commands: list[Command], by_id: dict[str, Device]) -> tuple[list[Command], list[dict]]:
    """Split a list of commands into (valid, rejected). Each rejected entry
    carries a `why` so the caller — or the model — can see what was wrong."""
    valid: list[Command] = []
    dropped: list[dict] = []
    for cmd in commands:
        ok, why = validate_command(cmd, by_id)
        if ok:
            valid.append(cmd)
        else:
            dropped.append({
                "device_id": cmd.device_id,
                "capability": cmd.capability,
                "command": cmd.command,
                "why": why,
            })
    return valid, dropped
