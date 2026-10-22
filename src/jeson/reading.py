"""What every reader returns: a probability distribution over the allowed values of each typed leaf."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assemble import assemble
from .schema import Compiled


@dataclass
class Reading:
    compiled: Compiled
    dists: dict[str, dict]                     # leaf key -> {value: probability}, over the leaf's allowed values only
    features: dict[str, np.ndarray] = field(default_factory=dict)  # leaf key -> hidden state (what a Lock fits on)
    latency_ms: float | None = None

    def answers(self) -> dict:
        """Leaf key -> the most probable allowed value."""
        return {k: max(d, key=d.get) for k, d in self.dists.items()}

    def confidence(self) -> dict:
        """Leaf key -> the probability of its answer."""
        return {k: max(d.values()) for k, d in self.dists.items()}

    @property
    def json(self) -> dict:
        """The nested JSON answer, valid by construction."""
        return assemble(self.compiled, self.answers())

    def with_dists(self, dists: dict) -> Reading:
        return Reading(self.compiled, {**self.dists, **dists}, self.features, self.latency_ms)
