"""The tools the model can call.

Each tool is two things:
  1. a DEFINITION in TOOL_DEFS — the name, description, and input shape the
     model reads to decide what to call. The model can only call these, and
     only with arguments that fit the schema.
  2. a HANDLER on the Tools class — the plain Python that runs when the model
     calls that tool. Handlers wrap the foundation (mock home, validation,
     recipes); none of them call the model.

Design notes (Option C — hybrid):
  * apply_scene runs the WHOLE scene at once (the fast, happy path). It does
    NOT roll back on a partial failure: it applies what it can and reports
    which devices failed, so the model can decide what to do next.
  * execute_command runs ONE device, for when the model wants to retry a failed
    device, substitute a different one, or make a fix.
  * Every command is validated before it runs; an invalid command is reported
    back, never executed.
  * Each device's state is remembered the first time we touch it, so the loop
    can put the whole home back if something goes wrong (a safety backstop —
    not a tool the model controls).
"""
from __future__ import annotations

from typing import Any

from experiences.recipes import RECIPES, match_recipe
from smarthome.mock import DeviceError, MockHome
from smarthome.model import Command, restore_command
from smarthome.validation import clean

# How many times a single command is retried automatically before it's reported
# as failed. This is the mechanical retry (for a flaky device); the *policy*
# retry — give up, skip, try another device — is the model's decision.
AUTO_RETRIES = 1


# --- Tool definitions (what the model reads) --------------------------------
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "get_experience",
        "description": (
            "Look up the experience pattern that best matches the user's request "
            "(e.g. 'movie night', 'wind down for bed'). Returns an abstract recipe "
            "— a list of steps described by capability and room, not real devices. "
            "Use it as a guide, then call get_home to see the actual devices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
        },
    },
    {
        "name": "get_home",
        "description": (
            "List the user's real devices with their capabilities and current "
            "state. Only use device_id values that appear here, and only "
            "capabilities a device actually lists."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "apply_scene",
        "description": (
            "Run a whole scene at once: the full list of commands for the "
            "experience. Each command is validated first; valid ones run, and "
            "the result reports which devices succeeded, which failed, and which "
            "were invalid. It does NOT undo successful commands when another "
            "fails — if something fails, use execute_command to retry it, pick a "
            "different device, or skip it, then finish. Call this once to start."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "device_id": {"type": "string"},
                            "capability": {"type": "string"},
                            "command": {"type": "string"},
                            "args": {"type": "object"},
                        },
                        "required": ["device_id", "capability", "command"],
                    },
                }
            },
            "required": ["commands"],
        },
    },
    {
        "name": "execute_command",
        "description": (
            "Run a single command on one device. Use this to retry a device that "
            "failed in apply_scene, to try a different device instead, or to make "
            "a one-off fix. Returns whether it succeeded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "capability": {"type": "string"},
                "command": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["device_id", "capability", "command"],
        },
    },
    {
        "name": "finish",
        "description": (
            "End the task. Give a short summary of what the experience did, and "
            "mention any device you skipped or couldn't set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


# --- Tool handlers (what actually runs) -------------------------------------
class Tools:
    def __init__(self, home: MockHome) -> None:
        self.home = home
        # device_id -> its state before we first changed it (for the restore backstop)
        self._before: dict[str, dict[str, Any]] = {}

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route a tool call to its handler and return a JSON-able result.
        (finish is handled by the loop, not here.)"""
        if name == "get_experience":
            return self._get_experience(args["request"])
        if name == "get_home":
            return self._get_home()
        if name == "apply_scene":
            return self._apply_scene(args.get("commands", []))
        if name == "execute_command":
            return self._execute_command(args)
        return {"error": f"unknown tool {name}"}

    # -- individual tools --
    def _get_experience(self, request: str) -> dict[str, Any]:
        recipe, score = match_recipe(request)
        return {"recipe": recipe.to_dict(), "confidence": round(score, 2)}

    def _get_home(self) -> dict[str, Any]:
        return self.home.get_home_graph().to_dict()

    def _apply_scene(self, raw_commands: list[dict]) -> dict[str, Any]:
        by_id = self.home.get_home_graph().by_id()
        cmds = [self._build(c, by_id) for c in raw_commands]
        valid, invalid = clean(cmds, by_id)

        applied, failed = [], []
        for cmd in valid:
            ok, why = self._run_one(cmd, by_id)
            if ok:
                applied.append(cmd.device_id)
            else:
                failed.append({"device_id": cmd.device_id,
                               "device_label": cmd.device_label, "why": why})
        return {"applied": applied, "failed": failed, "invalid": invalid}

    def _execute_command(self, raw: dict) -> dict[str, Any]:
        by_id = self.home.get_home_graph().by_id()
        cmd = self._build(raw, by_id)
        valid, invalid = clean([cmd], by_id)
        if invalid:
            return {"device_id": cmd.device_id, "status": "invalid", "why": invalid[0]["why"]}
        ok, why = self._run_one(cmd, by_id)
        return {"device_id": cmd.device_id,
                "status": "ok" if ok else "failed", "why": why}

    # -- helpers --
    @staticmethod
    def _build(raw: dict, by_id: dict) -> Command:
        """Turn the model's raw command dict into a Command. Unknown devices get
        blank label/room and are caught by validation as 'unknown device'."""
        dev = by_id.get(raw.get("device_id"))
        return Command(
            device_id=raw.get("device_id", ""),
            device_label=dev.label if dev else raw.get("device_id", ""),
            room=dev.room if dev else "",
            capability=raw.get("capability", ""),
            command=raw.get("command", ""),
            args=raw.get("args", {}) or {},
        )

    def _run_one(self, cmd: Command, by_id: dict) -> tuple[bool, str]:
        """Execute one validated command, with a small automatic retry.
        Remembers the device's prior state first, so it can be restored later."""
        if cmd.device_id not in self._before and cmd.device_id in by_id:
            self._before[cmd.device_id] = dict(by_id[cmd.device_id].state)
        attempt = 0
        while True:
            try:
                self.home.execute_command(cmd.device_id, cmd.capability, cmd.command, cmd.args)
                return True, ""
            except DeviceError as e:
                attempt += 1
                if attempt > AUTO_RETRIES:  # real code would back off between tries
                    return False, str(e)

    def restore(self) -> list[str]:
        """Safety backstop: put every device we touched back to its starting
        state. Called by the loop if the run aborts — not a tool the model sees."""
        restored = []
        by_id = self.home.get_home_graph().by_id()
        for device_id, before in self._before.items():
            dev = by_id.get(device_id)
            if dev is None:
                continue
            for cap in dev.capabilities:
                rc = restore_command(cap, before)
                if rc is None:
                    continue
                command, args = rc
                try:
                    self.home.execute_command(device_id, cap, command, args)
                except DeviceError:
                    pass  # best effort
            restored.append(device_id)
        return restored
