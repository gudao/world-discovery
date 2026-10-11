"""Explicit hypothesis set H for protocol v0.2 (docs/PROTOCOL-v002.md section 1).

Particles over (mass, sliding friction, fixed) with an analytic Coulomb
sliding model as the hypothesis family. The functional form is prior
knowledge -- the same level of prior as v0.1 knowing the input is
(v, F, dt). Parameter VALUES are never calibrated against any test truth;
sigma is calibrated on the training distribution only (section 8.3).

Information gain uses the moment-matching approximation
    0.5 * sum_d log(1 + Var_w(particle prediction_d) / sigma_d^2)
per frozen protocol section 8.1.
"""
from __future__ import annotations

import numpy as np

G = 9.81
V_EPS = 1e-9


def analytic_rollout(x, v, force, duration, mass, mu, fixed, dt=0.002):
    """Coulomb sliding integration, vectorized over particles.

    x, v: shape (P,). mass, mu: broadcastable to (P,). fixed: boolean (P,).
    force: scalar or shape (P,) -- per-particle external force.
    Returns next (x, v), each shape (P,). Substeps match the MuJoCo timestep
    so integration error is of the same order as the simulator's.
    """
    x = np.array(x, dtype=float)
    v = np.array(v, dtype=float)
    mass = np.broadcast_to(np.asarray(mass, dtype=float), x.shape).astype(float).copy()
    mu = np.broadcast_to(np.asarray(mu, dtype=float), x.shape).astype(float).copy()
    fixed = np.broadcast_to(np.asarray(fixed, dtype=bool), x.shape).copy()
    force = np.broadcast_to(np.asarray(force, dtype=float), x.shape)
    steps = int(round(float(duration) / dt))
    for _ in range(steps):
        moving = np.abs(v) > V_EPS
        friction = mu * mass * G
        a = np.zeros_like(v)
        kin = moving & ~fixed
        a[kin] = (force[kin] - np.sign(v[kin]) * friction[kin]) / mass[kin]
        stat = (~moving) & (~fixed) & (np.abs(force) > friction)
        a[stat] = (force[stat] - np.sign(force[stat]) * friction[stat]) / mass[stat]
        v_new = v + a * dt
        # Stick within the substep when deceleration passes through zero.
        stop = moving & (np.sign(a) == -np.sign(v)) & (np.abs(v) <= np.abs(a) * dt)
        v_new[stop] = 0.0
        v = v_new
        x = x + v * dt
    return x, v


class ParticleHypotheses:
    """Reader posterior shared by every policy in the stage-1 test.

    Policies differ only in action selection; this class turns the chosen
    (observation, action, next_observation) triples into a posterior over
    hidden parameters, so discrimination is scored on identical machinery.
    """

    def __init__(self, config, rng, sigma):
        self.config = config
        self.rng = rng
        self.sigma = np.asarray(sigma, dtype=float)  # per-dim (dx, dv)
        n = int(config['particles'])
        self.mass = rng.uniform(*config['train_mass'], n)
        self.mu = rng.uniform(*config['train_friction'], n)
        self.fixed = rng.random(n) < config['fixed_probability']
        self.w = np.full(n, 1.0 / n)
        self.ess = float(n)

    # -- prediction -----------------------------------------------------
    def delta_prediction(self, obs, action):
        """Predicted (dx, dv) per particle, shape (P, 2)."""
        x0, v0 = float(obs[0]), float(obs[1])
        force, duration = float(action[0]), float(action[1])
        ones = np.ones(len(self.w))
        x1, v1 = analytic_rollout(ones * x0, ones * v0, force, duration,
                                  self.mass, self.mu, self.fixed,
                                  self.config['timestep'])
        return np.stack([x1 - x0, v1 - v0], axis=1)

    def predict_mean(self, obs, action):
        return self.delta_prediction(obs, action).T @ self.w

    # -- Bayesian update ------------------------------------------------
    def update(self, obs, action, next_obs):
        d = self.delta_prediction(obs, action)
        observed = np.asarray(next_obs, dtype=float) - np.asarray(obs, dtype=float)
        log_lik = -0.5 * np.sum(((observed[None, :] - d) / self.sigma) ** 2, axis=1)
        log_lik -= 0.5 * float(np.log(2 * np.pi * self.sigma ** 2).sum())
        log_w = np.log(np.maximum(self.w, 1e-300)) + log_lik
        log_w -= log_w.max()
        self.w = np.exp(log_w)
        self.w /= self.w.sum()
        self.ess = float(1.0 / np.sum(self.w ** 2))
        if self.ess < len(self.w) / 2:
            self._resample()

    def _resample(self):
        n = len(self.w)
        positions = (self.rng.random() + np.arange(n)) / n
        cdf = np.cumsum(self.w)
        idx = np.clip(np.searchsorted(cdf, positions), 0, n - 1)
        self.mass = self.mass[idx]
        self.mu = self.mu[idx]
        self.fixed = self.fixed[idx]
        self.w = np.full(n, 1.0 / n)
        self.ess = float(n)

    # -- experimental design ---------------------------------------------
    def information_gain(self, obs, candidates):
        """Moment-matching EIG per candidate action, in nats (protocol 8.1).

        All candidates are rolled out in one vectorized call; the math is
        identical to a per-candidate loop (validated by
        tests/test_prior_floor.py::test_information_gain_equivalence).
        """
        n, p = len(candidates), len(self.w)
        x = np.full(n * p, float(obs[0]))
        v = np.full(n * p, float(obs[1]))
        force = np.repeat(np.asarray(candidates[:, 0], dtype=float), p)
        mass = np.tile(self.mass, n)
        mu = np.tile(self.mu, n)
        fixed = np.tile(self.fixed, n)
        x1, v1 = analytic_rollout(x, v, force, float(candidates[0, 1]), mass, mu,
                                  fixed, self.config['timestep'])
        d = np.stack([x1 - float(obs[0]), v1 - float(obs[1])], axis=1).reshape(n, p, 2)
        mean = np.einsum('npd,p->nd', d, self.w)
        var = np.einsum('npd,p->nd', (d - mean[:, None, :]) ** 2, self.w)
        return 0.5 * np.sum(np.log(1.0 + var / self.sigma ** 2), axis=1)

    def choose(self, obs, candidates, prev_action):
        """Frozen scoring rule (protocol sections 2 and 8.1).

        eps-branch repeats the previous action (verification of
        reproducibility); otherwise argmax of information gain minus cost.
        """
        if prev_action is not None and self.rng.random() < self.config['exploration_probability']:
            return np.array(prev_action, dtype=float), {'branch': 'verify'}
        ig = self.information_gain(obs, candidates)
        cost = self.config['cost_lambda'] * np.abs(candidates[:, 0]) * candidates[:, 1]
        total = ig - cost
        best = np.flatnonzero(np.isclose(total, total.max(), atol=1e-12, rtol=1e-9))
        i = int(self.rng.choice(best))
        info = {'branch': 'ig', 'action_index': i, 'ig': float(ig[i]),
                'cost': float(cost[i]), 'total': float(total[i]),
                'all_ig': ig.tolist(), 'all_cost': cost.tolist()}
        return candidates[i].copy(), info

    # -- posterior summaries ----------------------------------------------
    def posterior_summary(self):
        def wstat(values):
            mean = float(self.w @ values)
            var = float(self.w @ (values - mean) ** 2)
            return mean, max(var, 0.0) ** 0.5
        m_mean, m_std = wstat(self.mass)
        mu_mean, mu_std = wstat(self.mu)
        return {'mass_mean': m_mean, 'mass_std': m_std,
                'mu_mean': mu_mean, 'mu_std': mu_std,
                'p_fixed': float(self.w @ self.fixed), 'ess': self.ess}

    def prob_greater(self, other, dim):
        """P(this world's parameter > other's) under the two posteriors.

        dim in {'mass', 'mu', 'fixed'}; exact weighted double sum.
        """
        keys = {'mass': (self.mass, other.mass), 'friction': (self.mu, other.mu),
                'fixed': (self.fixed.astype(float), other.fixed.astype(float))}
        a, b = keys[dim]
        mat = (a[:, None] > b[None, :]).astype(float)
        return float(np.einsum('i,ij,j->', self.w, mat, other.w))
