"""Sanity checks for the v0.2.1 stage-4 noise wrapper and policy wiring."""
import numpy as np

from world_discovery.env import WorldSpec
from world_discovery.stage4 import NoisyWorld, sigma_total_of

CONFIG = {
    'particles': 200, 'forces': list(range(-12, 13)), 'duration': 0.15,
    'timestep': 0.002, 'exploration_probability': 0.0, 'cost_lambda': 0.05,
    'train_mass': [0.4, 1.6], 'train_friction': [0.15, 0.7], 'fixed_probability': 0.2,
    'obs_noise': [0.005, 0.08],
}


def test_noisy_observation_statistics():
    rng = np.random.default_rng(0)
    world = NoisyWorld(WorldSpec(1.0, 0.4, False), CONFIG, rng)
    world.reset()
    clean = world.env.observe()
    draws = np.array([world.observe() for _ in range(400)])
    np.testing.assert_allclose(draws.mean(axis=0), clean, atol=3e-3)
    assert abs(draws[:, 1].std() - CONFIG['obs_noise'][1]) < 0.02
    assert abs(draws[:, 0].std() - CONFIG['obs_noise'][0]) < 0.002


def test_noise_applied_to_step_observation():
    rng = np.random.default_rng(1)
    world = NoisyWorld(WorldSpec(0.5, 0.3, False), CONFIG, rng)
    world.reset()
    obs = world.observe()
    nxt = world.step([6.0, 0.15])
    clean_delta = np.asarray(nxt) - np.asarray(obs)
    assert abs(clean_delta[1] - 1.6) < 0.4  # (6 - 0.3*0.5*9.81)/0.5 * 0.15 ≈ 1.66 + noise


def test_sigma_total_combines_family_and_twice_noise():
    family = np.array([0.004, 0.042])
    total = sigma_total_of(family, CONFIG)
    expected = np.sqrt(family ** 2 + 2 * np.asarray(CONFIG['obs_noise']) ** 2)
    np.testing.assert_allclose(total, expected)
