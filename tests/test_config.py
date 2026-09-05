import json

import mlx.core as mx
import pytest

from workbench.config import SteeringSpec, TaskConfig


def test_config_round_trips_through_json(tmp_path):
    """A saved configuration must be readable back byte-for-byte in meaning --
    it is the durable unit the whole 'not starting fresh' idea rests on."""
    cfg = TaskConfig(name="concise",
                     steering=[SteeringSpec(layer=18, vector=[0.5, -0.25, 1.0],
                                            strength=0.15, label="terse vs verbose")])
    path = tmp_path / "concise.json"
    cfg.save(path)

    loaded = TaskConfig.load(path)

    assert loaded.name == "concise"
    assert loaded.steering[0].layer == 18
    assert loaded.steering[0].strength == pytest.approx(0.15)
    assert loaded.steering[0].vector == [0.5, -0.25, 1.0]
    assert loaded.steering[0].label == "terse vs verbose"


def test_saved_config_is_plain_inspectable_json(tmp_path):
    """Inspectable is a project value: a config a user cannot read is a
    black box of exactly the kind this project exists to avoid."""
    cfg = TaskConfig(name="x", steering=[SteeringSpec(layer=1, vector=[1.0], strength=2.0)])
    path = tmp_path / "x.json"
    cfg.save(path)

    raw = json.loads(path.read_text())
    assert raw["name"] == "x"
    assert raw["steering"][0]["layer"] == 1


def test_config_converts_to_engine_steering_params():
    cfg = TaskConfig(name="x", steering=[SteeringSpec(layer=3, vector=[1.0, 2.0], strength=0.5)])

    params = cfg.as_steering()

    assert set(params) == {3}
    vector, strength = params[3]
    assert strength == pytest.approx(0.5)
    assert [float(v) for v in vector.tolist()] == [1.0, 2.0]


def test_empty_config_steers_nothing():
    assert TaskConfig(name="none").as_steering() == {}


def test_spec_from_a_derived_vector_keeps_the_values():
    v = mx.array([1.5, -2.5])
    spec = SteeringSpec.from_vector(layer=7, vector=v, strength=0.3, label="lbl")
    assert spec.vector == [1.5, -2.5]
    assert spec.layer == 7
