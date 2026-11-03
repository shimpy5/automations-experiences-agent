"""The agent loop — the part that ties the model and the tools together.

The model is text-in/text-out: it can only *say* which tool to call. This loop
is the body that carries out those decisions:

    call the model -> run the tool it asked for -> feed the result back -> repeat

The model decides WHICH tool and in what order (and how to handle failures);
the loop just executes and remembers. It ends when the model calls `finish`,
when the model stops asking for tools, or when MAX_STEPS runs out. If the run
aborts (step limit or an exception), the loop calls the restore backstop so a
half-applied home is put back.

Written as a hand loop on purpose, so every step is visible. You could later
swap in the SDK's tool runner and keep the same shape.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from agent.tools import TOOL_DEFS, Tools
from smarthome.mock import MockHome

MAX_STEPS = 10
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")

SYSTEM = """You are a smart-home agent. The user names an experience they want
(e.g. "cozy movie night", "wind down for bed") and you set it up in their real home.

How to work (a guide, not a strict script):
1. Call get_experience to get a recipe pattern for the request.
2. Call get_home to see the real devices, their capabilities, and current state.
3. Build the scene for THIS home — pick real device_ids from get_home and
   sensible values — and send it all at once with apply_scene.
4. apply_scene reports what applied, what failed, and what was invalid:
   - invalid: you wrote a command the device can't do; fix it and try again.
   - failed: the device didn't respond. Decide per device — retry it with
     execute_command, use a different device, or skip it.
5. When the scene is set up (or you've decided to skip what you can't do), call
   finish with a short summary, naming anything you skipped.

Rules:
- Only use device_id values and capabilities that appear in get_home. Never
  invent devices.
- Don't retry a dead device forever — a couple of tries, then skip it.
- Call finish exactly once, on its own, when you're done.
"""


class ExperienceAgent:
    def __init__(
        self,
        home: MockHome,
        client: Optional[Any] = None,
        model: str = MODEL,
        verbose: bool = False,
    ) -> None:
        self.home = home
        self.tools = Tools(home)
        self.model = model
        self.verbose = verbose
        # Injectable so an eval (or a test) can pass a fake client. By default we
        # create the real Anthropic client, which reads ANTHROPIC_API_KEY.
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def run(self, request: str) -> str:
        messages: list[dict] = [{"role": "user", "content": request}]
        self._trace(f"USER: {request}")
        try:
            for _ in range(MAX_STEPS):
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM,
                    tools=TOOL_DEFS,
                    messages=messages,
                )
                # Keep the model's whole turn (tool_use + any thinking blocks).
                messages.append({"role": "assistant", "content": resp.content})

                # No tool call -> the model is just talking -> it's done.
                if resp.stop_reason != "tool_use":
                    return self._text(resp)

                # Run every tool the model asked for; collect the results.
                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    if block.name == "finish":
                        summary = block.input.get("summary", "done")
                        self._trace(f"FINISH: {summary}")
                        return summary  # clean end -> leave the scene in place
                    output = self.tools.dispatch(block.name, dict(block.input))
                    self._trace(f"TOOL {block.name}({_short(block.input)}) -> {_short(output)}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output),
                    })

                # Feed the results back and loop for the next decision.
                messages.append({"role": "user", "content": results})

            # Ran out of steps without finishing -> abort and put the home back.
            self._trace("ABORT: hit the step limit; restoring")
            self.tools.restore()
            return "Stopped: the agent hit the step limit, so the home was put back the way it was."
        except Exception:
            # Any crash mid-run: don't strand a half-applied home.
            self.tools.restore()
            raise

    def _text(self, resp) -> str:
        return "".join(b.text for b in resp.content if b.type == "text") or "done"

    def _trace(self, line: str) -> None:
        if self.verbose:
            print(line)


def _short(obj, limit: int = 160) -> str:
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= limit else s[: limit - 1] + "…"


if __name__ == "__main__":
    import sys

    request = " ".join(sys.argv[1:]) or "make it cozy for movie night"
    # Optional: set FAIL_DEVICES=lr-tv,br-light to simulate dead devices.
    fail = [d for d in os.getenv("FAIL_DEVICES", "").split(",") if d]
    agent = ExperienceAgent(MockHome(fail_device_ids=fail), verbose=True)
    print("\nRESULT:", agent.run(request))
