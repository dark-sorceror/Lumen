"""Durable per-task configuration.

A configuration is the unit that makes a task's behaviour repeatable: saved
context plus steering (and, later, grounding rules), versioned and reapplied
across sessions instead of rebuilt from a prompt each time. These controls sit
BELOW the prompt layer -- they are directions in activation space and cache
state, not text -- which is exactly what a hosted API cannot expose.

Stored as plain JSON on purpose. A configuration a user cannot read is the kind
of black box this project exists to avoid, and a diffable file makes a change
in behaviour traceable to a change in the config."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx


@dataclass
class SteeringSpec:
    """One direction added to the residual stream at a given block."""
    layer: int
    vector: list[float]
    strength: float
    label: str = ""

    @classmethod
    def from_vector(cls, layer: int, vector, strength: float,
                    label: str = "") -> "SteeringSpec":
        """Build a spec from a derived (mx.array) direction."""
        return cls(layer=layer, vector=[float(x) for x in vector.tolist()],
                   strength=float(strength), label=label)


@dataclass
class TaskConfig:
    name: str
    steering: list[SteeringSpec] = field(default_factory=list)

    def as_steering(self) -> dict:
        """The form the engine takes: layer -> (direction, strength)."""
        return {s.layer: (mx.array(s.vector), s.strength) for s in self.steering}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "TaskConfig":
        raw = json.loads(Path(path).read_text())
        return cls(name=raw["name"],
                   steering=[SteeringSpec(**s) for s in raw.get("steering", [])])
