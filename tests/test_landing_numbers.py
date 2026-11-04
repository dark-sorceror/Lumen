"""The numbers the landing's drawings print: the token estimate and the
short token counts."""
from __future__ import annotations

from tests.node_ts import needs_node, run_ts

pytestmark = needs_node


def test_estimate_is_four_characters_a_token_rounded_up_and_zero_when_blank():
    out = run_ts("lib/landing.ts", """
      return ["", "   ", "a", "abcd", "abcde", "  abcdefgh  ", "x".repeat(4001)].map(m.estimateTokens);
    """)
    assert out == [0, 0, 1, 1, 2, 2, 1001]


def test_tokens_text_is_short_and_never_says_1000k():
    out = run_ts("lib/landing.ts", """
      return [0, 640, 1000, 1600, 9400, 9960, 10000, 61300, 200000, 999499, 999500, 1000000, 1250000]
        .map(m.tokensText);
    """)
    assert out == ["0", "640", "1k", "1.6k", "9.4k", "10k", "10k", "61k", "200k", "999k", "1M", "1M", "1.3M"]
