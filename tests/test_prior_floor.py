"""Tests for the v0.2.2 prior-floor machinery and the vectorized IG score."""
import numpy as np
import pytest

from world_discovery.hypothesis import ParticleHypotheses, analytic_rollout
from world_discovery.prior_floor import AnalyticWorld, in_family_queries, oracle_choose

CONFIG = {
    'particles': 60, 'train_mass': [0.4, 1.6], 'train_friction': [0.15, 0.7],
    'fixed_probability': 0.2, 'exploration_probability': 0.0, 'cost_lambda': 0.05,
    'forces': list(range(-12, 13)), 'duration': 0.15, 'timestep': 0.002,
    'obs_noise': [0.005, 0.08],
}


def _candidates():
    return np.array([[f, 0.15] for f in CONFIG['forces']], dtype=float)


def test_information_gain_equivalence():
    """Vectorized batch IG must match the per-candidate loop to 1e-12 nats."""
    rng = np.random.default_rng(3)
    reader = ParticleHypotheses(CONFIG, rng, np.array([0.009, 0.125]))
    obs = np.array([0.01, 0.4])
    cands = _candidates()
    got = reader.information_gain(obs, cands)
    reference = np.zeros(len(cands))
    for i, action in enumerate(cands):
        d = reader.delta_prediction(obs, action)
        mean = d.T @ reader.w
        var = ((d - mean) ** 2).T @ reader.w
        reference[i] = 0.5 * float(np.sum(np.log(1.0 + var / reader.sigma ** 2)))
    assert np.allclose(got, reference, atol=1e-12, rtol=1e-12)


def test_analytic_world_matches_rollout():
    rng = np.random.default_rng(5)
    spec = {'mass': 1.0, 'friction': 0.4, 'fixed': False}
    world = AnalyticWorld(spec, CONFIG, rng)
    world.reset()
    world.step(np.array([10.0, 0.15]))
    x, v = analytic_rollout(np.array([0.0]), np.array([0.0]), 10.0, 0.15,
                            spec['mass'], spec['friction'], spec['fixed'], 0.002)
    # Internal state is noise-free and must equal the rollout exactly.
    assert world.x == pytest.approx(float(x[0]), abs=1e-9)
    assert world.v == pytest.approx(float(v[0]), abs=1e-9)


def test_in_family_queries_shape_and_static_world():
    spec = {'mass': 1.5, 'friction': 0.7, 'fixed': False}
    y = in_family_queries(spec, CONFIG)
    assert y.shape == (75, 2)
    # Impulses +-6 and all forces below mu*m*g = 10.3 cannot move this world:
    # only |F| in {11, 12} produce nonzero deltas (4 forces x 3 groups).
    moving = np.any(np.abs(y) > 1e-9, axis=1)
    assert moving.sum() == 12


def test_oracle_choose_prefers_discriminating_action():
    rng = np.random.default_rng(7)
    spec = {'mass': 1.0, 'friction': 0.3, 'fixed': False}
    other = {'mass': 1.0, 'friction': 0.6, 'fixed': False}
    sigma = np.array([0.009, 0.125])
    obs = np.array([0.0, 0.0])
    action, score = oracle_choose(obs, _candidates(), spec, other, CONFIG, sigma)
    assert score > 0.0
    assert abs(float(action[0])) > 0  # a zero force cannot discriminate
