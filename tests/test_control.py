import threading
import time

from workbench.engine.control import Control, ControlQueue
from workbench.engine.engine import Engine, GenParams


def test_abort_stops_generation(fake_model, fake_tokenizer):
    control = ControlQueue()
    control.post(Control.ABORT)
    engine = Engine(fake_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=100), control=control))
    assert len(events) == 1
    assert events[0].finish_reason == "aborted"


def test_pause_blocks_until_resume(fake_model, fake_tokenizer):
    control = ControlQueue()
    engine = Engine(fake_model, fake_tokenizer)
    out = []

    def run():
        out.extend(engine.generate([1], GenParams(max_tokens=200), control=control))

    control.post(Control.PAUSE)
    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.2)
    assert len(out) == 0          # paused before first token
    control.post(Control.RESUME)
    t.join(timeout=5)
    assert not t.is_alive()
    assert len(out) == 200


def test_abort_while_paused(fake_model, fake_tokenizer):
    control = ControlQueue()
    engine = Engine(fake_model, fake_tokenizer)
    out = []
    control.post(Control.PAUSE)
    t = threading.Thread(
        target=lambda: out.extend(engine.generate([1], GenParams(max_tokens=200), control=control))
    )
    t.start()
    time.sleep(0.2)
    control.post(Control.ABORT)
    t.join(timeout=5)
    assert not t.is_alive()
    assert out[-1].finish_reason == "aborted"
