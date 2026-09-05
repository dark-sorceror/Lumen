"""Maps the ContextObject onto the token sequence the engine consumes."""
from __future__ import annotations

import copy
from dataclasses import dataclass

from workbench.context.model import ContextObject, EditEvent


@dataclass
class TokenizedContext:
    tokens: list[int]
    spans: dict[str, tuple[int, int]]


@dataclass
class CacheImpact:
    first_invalid_token: int
    tokens_to_reprefill: int


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class ContextManager:
    def __init__(self, ctx: ContextObject, tokenizer):
        self.ctx = ctx
        self.tokenizer = tokenizer

    def to_tokens(self) -> TokenizedContext:
        tokens: list[int] = []
        spans: dict[str, tuple[int, int]] = {}
        for seg in self.ctx.segments:
            start = len(tokens)
            tokens.extend(self.tokenizer.encode(seg.text))
            spans[seg.id] = (start, len(tokens))
        return TokenizedContext(tokens=tokens, spans=spans)

    def _impact_against(self, new_ctx: ContextObject) -> CacheImpact:
        old = self.to_tokens().tokens
        new = ContextManager(new_ctx, self.tokenizer).to_tokens().tokens
        keep = _common_prefix_len(old, new)
        return CacheImpact(first_invalid_token=keep,
                           tokens_to_reprefill=len(new) - keep)

    def impact_of(self, other: ContextObject) -> CacheImpact:
        """What switching this session to `other` would cost -- the public form
        of the same common-prefix comparison every edit is priced with."""
        return self._impact_against(other)

    def preview_edit(self, event: EditEvent) -> CacheImpact:
        trial = copy.deepcopy(self.ctx)
        trial.apply(event)
        return self._impact_against(trial)

    def apply_edit(self, event: EditEvent) -> CacheImpact:
        impact = self.preview_edit(event)
        self.ctx.apply(event)
        return impact

    def preview_rewind(self, n_events: int) -> CacheImpact:
        """What rewinding to `n_events` would cost, without doing it."""
        trial = copy.deepcopy(self.ctx)
        trial.rewind_to(n_events)
        return self._impact_against(trial)

    def apply_rewind(self, n_events: int) -> CacheImpact:
        """Rewind the context, returning the cache impact.

        Priced exactly like a forward edit: undo is not free, because the KV
        entries after the divergence point were computed against text that no
        longer exists."""
        impact = self.preview_rewind(n_events)
        self.ctx.rewind_to(n_events)
        return impact
