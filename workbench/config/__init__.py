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
import re
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


# Names arrive over the wire, so they are restricted rather than sanitised:
# quietly rewriting a name would let two different requests collide on one file.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


class ConfigStore:
    """A directory of named configurations.

    Flat and file-per-config on purpose: `ls` and a text editor are valid tools
    for inspecting what a task is configured to do."""

    def __init__(self, root):
        self.root = Path(root)

    def _path_for(self, name: str) -> Path:
        if not isinstance(name, str) or not _SAFE_NAME.match(name):
            raise ValueError(f"unsafe configuration name: {name!r}")
        return self.root / f"{name}.json"

    def save(self, cfg: TaskConfig) -> Path:
        path = self._path_for(cfg.name)
        self.root.mkdir(parents=True, exist_ok=True)
        cfg.save(path)
        return path

    def load(self, name: str) -> TaskConfig:
        path = self._path_for(name)
        if not path.is_file():
            raise FileNotFoundError(f"no configuration named {name!r}")
        return TaskConfig.load(path)

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))
