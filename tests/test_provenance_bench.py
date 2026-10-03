from experiments.provenance_bench import collinearity
from workbench.context.model import Provenance, Segment, SegmentKind


def _segs():
    out = [Segment(id=f"s{i}", kind=SegmentKind.USER_MSG, text="x",
                   provenance=Provenance(author="a")) for i in range(6)]
    out[2] = Segment(id="s2", kind=SegmentKind.ASSISTANT_MSG, text="x",
                     provenance=Provenance(author="model", derived_from=("s0",)))
    return out


def test_collinearity_ignores_pinned_segments():
    segs = _segs()

    assert collinearity(segs, pinned=("s4", "s5")) != collinearity(segs)
