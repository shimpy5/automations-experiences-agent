"""The eval test set.

Each Case is a request plus a list of Checks. A Check is a plain predicate over
the home's final state (device_id -> Device). We check RANGES and intent, not
exact values, because the model chooses the exact numbers — "dimmed" means
level <= 30, not level == 15. A reasonable scene passes; a wrong one fails.

Keep the set small (a few dozen cases at most): every case runs the real model,
which costs a little.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# A check receives `by_id`: {device_id -> Device}. Returns True if the home
# looks the way this part of the experience should.
Predicate = Callable[[dict], bool]


@dataclass
class Check:
    desc: str
    ok: Predicate


@dataclass
class Case:
    name: str
    request: str
    fail_devices: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)


def st(by_id: dict, device_id: str, attr: str, default=None):
    """Safe read of one device's state attribute."""
    dev = by_id.get(device_id)
    return dev.state.get(attr, default) if dev else default


CASES: list[Case] = [
    Case(
        name="movie_night",
        request="make it cozy for movie night",
        checks=[
            Check("living room ceiling dimmed (<=30)", lambda h: st(h, "lr-ceiling", "level", 100) <= 30),
            Check("living room ceiling warm (<=3000K)", lambda h: st(h, "lr-ceiling", "colorTemperature", 6500) <= 3000),
            Check("living room shades closed", lambda h: st(h, "lr-shades", "windowShade") == "closed"),
            Check("TV turned on", lambda h: st(h, "lr-tv", "switch") == "on"),
        ],
    ),
    Case(
        name="movie_night_dead_tv",
        request="set up movie night",
        fail_devices=["lr-tv"],
        checks=[
            # Lights still dimmed proves the scene applied AND wasn't rolled back.
            Check("living room ceiling dimmed (<=30)", lambda h: st(h, "lr-ceiling", "level", 100) <= 30),
            Check("living room shades closed", lambda h: st(h, "lr-shades", "windowShade") == "closed"),
            # TV is dead, so it should be left off — not falsely reported on.
            Check("dead TV left off", lambda h: st(h, "lr-tv", "switch") == "off"),
        ],
    ),
    Case(
        name="wind_down",
        request="help me wind down for bed",
        checks=[
            Check("bedroom light dimmed (<=30)", lambda h: st(h, "br-light", "level", 100) <= 30),
            Check("bedroom light very warm (<=2700K)", lambda h: st(h, "br-light", "colorTemperature", 6500) <= 2700),
            Check("bedroom shades closed", lambda h: st(h, "br-shades", "windowShade") == "closed"),
            Check("front door locked", lambda h: st(h, "front-lock", "lock") == "locked"),
        ],
    ),
    Case(
        name="good_morning",
        request="good morning, start my day",
        checks=[
            Check("bedroom shades open", lambda h: st(h, "br-shades", "windowShade") == "open"),
            Check("kitchen light on", lambda h: st(h, "kitchen-light", "switch") == "on"),
            Check("coffee maker on", lambda h: st(h, "coffee-maker", "switch") == "on"),
        ],
    ),
    Case(
        name="leaving_home",
        request="I'm leaving home, lock up",
        checks=[
            Check("living room ceiling off", lambda h: st(h, "lr-ceiling", "switch") == "off"),
            Check("kitchen light off", lambda h: st(h, "kitchen-light", "switch") == "off"),
            Check("living room shades closed", lambda h: st(h, "lr-shades", "windowShade") == "closed"),
            Check("front door locked", lambda h: st(h, "front-lock", "lock") == "locked"),
        ],
    ),
    Case(
        name="focus",
        request="I need to focus and get work done",
        checks=[
            Check("living room ceiling bright (>=80)", lambda h: st(h, "lr-ceiling", "level", 0) >= 80),
            Check("living room ceiling cool (>=4500K)", lambda h: st(h, "lr-ceiling", "colorTemperature", 0) >= 4500),
            Check("living room shades open", lambda h: st(h, "lr-shades", "windowShade") == "open"),
        ],
    ),
]
