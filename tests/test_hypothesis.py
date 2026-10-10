"""Sanity checks for the v0.2 analytic hypothesis family and particle reader.

These use training-distribution worlds with known specs; no test-world truth
is involved (protocol v0.2 anti-cheat boundary).
"""
import numpy as np

from world_discovery.env import PushWorld, WorldSpec
from world_discovery.hypothesis import ParticleHypotheses, analytic_rollout

CONFIG = {
    'particles': 200, 'forces': [0, -12, -6, -2, 2, 6, 12], 'duration': 0.15,
    'timestep': 0.002, 'exploration_probability': 0.0, 'cost_lambda': 0.05,
    'train_mass': [0.4, 1.6], 'train_friction': [0.15, 0.7], 'fixed_probability': 0.2,
}


def test_analytic_matches_mujoco_sliding():
    spec = WorldSpec(0.5, 0.3, False)
    env = PushWorld(spec, 0.002)
    env.reset()
    obs = env.observe()
    nxt = env.step([6.0, 0.15])
    x1, v1 = analytic_rollout(np.array([obs[0]]), np.array([obs[1]]), 6.0, 0.15,
                              np.array([spec.mass]), np.array([spec.friction]),
                              np.array([spec.fixed]), 0.002)
    assert abs((v1[0] - obs[1]) - (nxt[1] - obs[1])) < 0.05  # dv
    assert abs((x1[0] - obs[0]) - (nxt[0] - obs[0])) < 0.02  # dx


def test_analytic_fixed_world_is_motionless():
    spec = WorldSpec(1.2, 0.4, True)
    env = PushWorld(spec, 0.002)
    env.reset()
    obs = env.observe()
    nxt = env.step([12.0, 0.15])
    assert nxt[0] == obs[0] and nxt[1] == obs[1]  # MuJoCo: fixed world cannot move
    x1, v1 = analytic_rollout(np.array([0.0]), np.array([0.0]), 12.0, 0.15,
                              np.array([spec.mass]), np.array([spec.friction]),
                              np.array([True]), 0.002)
    assert x1[0] == 0.0 and v1[0] == 0.0


def test_particle_posterior_converges_on_known_world():
    spec = WorldSpec(0.9, 0.45, False)
    env = PushWorld(spec, 0.002)
    env.reset()
    sigma = np.array([0.01, 0.15])
    reader = ParticleHypotheses(CONFIG, np.random.default_rng(0), sigma)
    forces = [12.0, -12.0, 6.0, -6.0, 0.0, 2.0, -2.0, 12.0, -12.0, 6.0, 0.0, -6.0]
    for f in forces:
        obs = env.observe()
        nxt = env.step([f, 0.15])
        reader.update(obs, [f, 0.15], nxt)
    s = reader.posterior_summary()
    assert abs(s['mu_mean'] - 0.45) < 0.08
    assert abs(s['mass_mean'] - 0.9) < 0.15
    assert s['p_fixed'] < 0.05


def test_particle_reader_detects_fixed_world():
    spec = WorldSpec(0.7, 0.3, True)
    env = PushWorld(spec, 0.002)
    env.reset()
    sigma = np.array([0.01, 0.15])
    reader = ParticleHypotheses(CONFIG, np.random.default_rng(1), sigma)
    for f in [12.0, -12.0, 12.0, 6.0]:
        obs = env.observe()
        nxt = env.step([f, 0.15])
        reader.update(obs, [f, 0.15], nxt)
    assert reader.posterior_summary()['p_fixed'] > 0.9


def test_ig_prefers_discriminating_action():
    sigma = np.array([0.01, 0.15])
    reader = ParticleHypotheses(CONFIG, np.random.default_rng(2), sigma)
    cands = np.array([[f, 0.15] for f in CONFIG['forces']])
    obs = np.array([0.0, 0.0])
    ig = reader.information_gain(obs, cands)
    push = float(ig[CONFIG['forces'].index(12.0)])
    idle = float(ig[CONFIG['forces'].index(0.0)])
    assert push > idle  # a hard push separates fixed from moving particles
