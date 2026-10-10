import numpy as np
import pytest
from world_discovery.env import PushWorld, WorldSpec, model_input


def test_force_moves_free_object_but_not_fixed():
    free = PushWorld(WorldSpec(1, 0.2))
    fixed = PushWorld(WorldSpec(1, 0.2, True))
    assert free.step([12, 0.15])[0] > 0.01
    assert abs(fixed.step([12, 0.15])[0]) < 1e-12


def test_mass_and_friction_affect_actual_mujoco_contact():
    light = PushWorld(WorldSpec(0.5, 0.2)).step([12, 0.15])
    heavy = PushWorld(WorldSpec(2, 0.2)).step([12, 0.15])
    slippery = PushWorld(WorldSpec(1, 0.1)).step([6, 0.15])
    rough = PushWorld(WorldSpec(1, 0.7)).step([6, 0.15])
    assert light[0] > heavy[0]
    assert slippery[0] > rough[0]


def test_snapshot_is_exact_and_observation_is_not_a_view():
    env = PushWorld(WorldSpec(1, 0.3))
    original = env.observe()
    state = env.snapshot()
    first = env.step([6, 0.15])
    env.restore(state)
    second = env.step([6, 0.15])
    np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)
    assert original[0] == 0
    assert model_input(original, [6, 0.15]).shape == (3,)


def test_invalid_action_rejected():
    env = PushWorld(WorldSpec(1, 0.3))
    with pytest.raises(ValueError):
        env.step([1, 0.001])
