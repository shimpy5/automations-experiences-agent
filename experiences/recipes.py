"""The experience recipe library + a simple way to find the right one.

A recipe is a home-agnostic *pattern*: it describes an experience in abstract
capabilities and rooms ("dim the Living Room lights, warm, shades down"),
naming no specific device. The agent's `get_experience` tool hands one of these
to the model, and the model turns it into concrete commands for the real home.

Retrieval here is plain keyword scoring — zero dependencies. The seam is clean
if you later swap in embeddings/a vector store.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_RECIPES_PATH = Path(__file__).resolve().parent / "recipes.json"


@dataclass
class RecipeStep:
    capability: str
    command: str
    args: dict[str, Any] = field(default_factory=dict)
    room: Optional[str] = None
    device_hint: Optional[str] = None  # label substring to pick among same-capability devices
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {"capability": self.capability, "command": self.command,
             "args": self.args, "reason": self.reason}
        if self.room:
            d["room"] = self.room
        if self.device_hint:
            d["device_hint"] = self.device_hint
        return d


@dataclass
class Recipe:
    id: str
    name: str
    keywords: list[str]
    description: str
    steps: list[RecipeStep]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "description": self.description,
                "steps": [s.to_dict() for s in self.steps]}


def load_recipes() -> list[Recipe]:
    data = json.loads(_RECIPES_PATH.read_text())
    recipes = []
    for r in data["recipes"]:
        steps = [RecipeStep(**s) for s in r["steps"]]
        recipes.append(Recipe(id=r["id"], name=r["name"], keywords=r["keywords"],
                              description=r["description"], steps=steps))
    return recipes


RECIPES: list[Recipe] = load_recipes()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def match_recipe(request: str, recipes: Optional[list[Recipe]] = None) -> tuple[Recipe, float]:
    """Find the recipe whose keywords best match the request. Returns the recipe
    and a rough 0..1 confidence score."""
    recipes = recipes or RECIPES
    words = _tokens(request)
    best, best_score = recipes[0], 0.0
    for recipe in recipes:
        score = 0.0
        for kw in recipe.keywords:
            if kw.lower() in request.lower():
                score += 2.0  # whole phrase present is a strong signal
            elif _tokens(kw) & words:
                score += 1.0
        norm = min(1.0, score / 4.0)
        if norm > best_score:
            best, best_score = recipe, norm
    return best, best_score
