"""Pre-freeze prior information-floor calculation for protocol v0.2.2.

In-family synthetic simulation: the analytic Coulomb family serves BOTH as
environment truth and as the reader model (MuJoCo is replaced by
`analytic_rollout`). Because the reader's model equals the truth here, the
resulting accuracies are an UPPER BOUND on what the real (MuJoCo) run can
achieve; the gap is the family error, already measured by sigma_family.

This is a design calculation run BEFORE freezing v0.2.2. Its purpose is to
verify that the information-scarce regime actually separates information
selecting policies (ig, oracle) from non-selecting ones (random, fixed7)
before any MuJoCo test world is probed. All sweep conditions are archived
in the output; the frozen amendment records the chosen parameters.

Physical rationale for the heavy/high-friction strata: from rest the
per-probe velocity signal is capped at |dmu|*g*dt; the only channels that
can exceed it are carried-velocity amplification and position accumulation.
Heavy, high-friction worlds push static thresholds (mu*m*g) high enough that
most of the force grid is uninformative from rest, so wasted probes cost
real information and action selection becomes load-bearing.

Policies simulated: random, fixed7, ig (real frozen machinery including the
eps-verify branch), and oracle_greedy -- a cheating upper bound that knows
both true specs and greedily maximizes the two-hypothesis discrimination
information of the next observation. v01_active is not simulated: it is a
frozen negative control whose ranking is not a design input.

Run:
    python -m world_discovery.prior_floor --pairs 50 --out runs/v0022-prior-floor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .hypothesis import ParticleHypotheses, analytic_rollout
from .stage4 import candidate_actions, sigma_total_of

STRATA = ('friction', 'mass')
POLICIES = ('random', 'fixed7', 'ig', 'oracle')
SIGMA_FAMILY = [0.005234461095870429, 0.05301680557738392]  # frozen v0.2.1 file

HEAVY = {'friction_mass_range': [1.0, 1.6], 'friction_mu_range': [0.4, 0.7],
         'mass_range': [1.1, 1.6]}
CONTROL = {'friction_mass_range': [0.4, 1.6], 'friction_mu_range': [0.15, 0.7],
           'mass_range': [0.4, 0.9]}

SWEEP = [
    {'label': 'G1_heavy_s10', 'obs_noise': [0.005, 0.10], **HEAVY},
    {'label': 'G2_heavy_s12', 'obs_noise': [0.005, 0.12], **HEAVY},
    {'label': 'G3_heavy_s15', 'obs_noise': [0.005, 0.15], **HEAVY},
    {'label': 'G4_heavy_x20_s12', 'obs_noise': [0.02, 0.12], **HEAVY},
    {'label': 'G5_heavy_x20_s15', 'obs_noise': [0.02, 0.15], **HEAVY},
    {'label': 'G6_heavy_s20', 'obs_noise': [0.005, 0.20], **HEAVY},
    {'label': 'CTRL_light_s15', 'obs_noise': [0.005, 0.15], **CONTROL},
]


class AnalyticWorld:
    """In-family environment: analytic rollout truth + sensor noise."""

    def __init__(self, spec, config, rng):
        self.spec = spec
        self.noise = np.asarray(config['obs_noise'], dtype=float)
        self.dt = config['timestep']
        self.rng = rng
        self.x, self.v = 0.0, 0.0

    def _noisy(self, obs):
        return np.asarray(obs, dtype=float) + self.rng.normal(0.0, self.noise)

    def observe(self):
        return self._noisy([self.x, self.v])

    def step(self, action):
        force, duration = float(action[0]), float(action[1])
        x, v = analytic_rollout(np.array([self.x]), np.array([self.v]), force,
                                duration, self.spec['mass'], self.spec['friction'],
                                self.spec['fixed'], self.dt)
        self.x, self.v = float(x[0]), float(v[0])
        return self._noisy([self.x, self.v])

    def reset(self):
        self.x, self.v = 0.0, 0.0
        return self.observe()


def in_family_queries(spec, config):
    """75 clean queries mirroring build_evaluation_queries (3 impulses x 25 forces).

    Vectorized: one rollout call over all 75 (state, force) combinations.
    """
    forces = np.asarray(config['forces'], dtype=float)
    x0 = np.zeros(3 * len(forces))
    v0 = np.zeros(3 * len(forces))
    fr = np.tile(forces, 3)
    ya = []
    for impulse in [0, -6, 6]:
        if impulse:
            x1, v1 = analytic_rollout(np.array([0.0]), np.array([0.0]), impulse,
                                      config['duration'], spec['mass'],
                                      spec['friction'], spec['fixed'], config['timestep'])
        else:
            x1, v1 = np.zeros(1), np.zeros(1)
        ya.append((float(x1[0]), float(v1[0])))
    base = np.repeat(np.asarray(ya), len(forces), axis=0)
    xn, vn = analytic_rollout(base[:, 0].copy(), base[:, 1].copy(), fr,
                              config['duration'], spec['mass'], spec['friction'],
                              spec['fixed'], config['timestep'])
    return np.stack([xn - base[:, 0], vn - base[:, 1]], axis=1)


def sample_pair(rng, config, stratum, sigma_total, diagnostics):
    if stratum == 'friction':
        mass_range, mu_range = config['friction_mass_range'], config['friction_mu_range']
    else:
        mass_range, mu_range = config['mass_range'], config['train_friction']
    for attempt in range(20000):
        a = {'mass': float(rng.uniform(*mass_range)),
             'friction': float(rng.uniform(*mu_range)), 'fixed': False}
        b = {'mass': float(rng.uniform(*mass_range)),
             'friction': float(rng.uniform(*mu_range)), 'fixed': False}
        if stratum == 'friction':
            ok = (abs(a['friction'] - b['friction']) >= config['min_friction_delta']
                  and abs(a['mass'] - b['mass']) <= config['friction_mass_guard'])
        else:
            ok = (abs(a['mass'] - b['mass']) >= config['min_mass_delta']
                  and abs(a['friction'] - b['friction']) <= config['mass_friction_guard'])
        if not ok:
            continue
        ya = in_family_queries(a, config)
        yb = in_family_queries(b, config)
        d = float(np.sum((ya - yb) ** 2 / (2.0 * sigma_total[None, :] ** 2)))
        diagnostics['attempts'][stratum] = diagnostics['attempts'].get(stratum, 0) + 1
        if d >= config['identifiability_nats']:
            diagnostics['screen_d'][stratum] = diagnostics['screen_d'].get(stratum, []) + [d]
            return a, b, d
    raise RuntimeError(f'Could not sample {stratum} pair')


def oracle_choose(obs, cands, spec, other, config, sigma_total):
    """Greedy two-hypothesis discrimination information (cheating upper bound).

    Vectorized: one rollout call over 25 candidates x 2 specs.
    """
    n = len(cands)
    x = np.full(2 * n, float(obs[0]))
    v = np.full(2 * n, float(obs[1]))
    force = np.concatenate([cands[:, 0], cands[:, 0]])
    mass = np.concatenate([np.full(n, spec['mass']), np.full(n, other['mass'])])
    mu = np.concatenate([np.full(n, spec['friction']), np.full(n, other['friction'])])
    fixed = np.zeros(2 * n, dtype=bool)
    xn, vn = analytic_rollout(x, v, force, float(cands[0][1]), mass, mu, fixed,
                              config['timestep'])
    delta = np.stack([xn - x, vn - v], axis=1)
    diff = delta[:n] - delta[n:]
    disc = np.sum(diff ** 2 / (2.0 * sigma_total ** 2), axis=1)
    best = int(np.argmax(disc))
    return cands[best].copy(), float(disc[best])


def probe_synthetic(policy, pair, label, config, rng_policy, rng_noise, sigma_total):
    spec = pair[label]
    other = pair['b' if label == 'a' else 'a']
    world = AnalyticWorld(spec, config, rng_noise)
    world.reset()
    cands = candidate_actions(config)
    reader = ParticleHypotheses(config, rng_policy, sigma_total)
    fixed7 = config['fixed7_forces']
    prev, effort, oracle_d = None, 0.0, 0.0
    for step in range(config['probe_budget']):
        obs = world.observe()
        if policy == 'random':
            action = cands[int(rng_policy.integers(len(cands)))].copy()
        elif policy == 'fixed7':
            action = np.array([fixed7[step % len(fixed7)], config['duration']], dtype=float)
        elif policy == 'ig':
            action, _ = reader.choose(obs, cands, prev)
        elif policy == 'oracle':
            action, d = oracle_choose(obs, cands, spec, other, config, sigma_total)
            oracle_d += d
        else:
            raise ValueError(policy)
        nxt = world.step(action)
        reader.update(obs, action, nxt)
        effort += abs(float(action[0])) * float(action[1])
        prev = action
    return reader, effort, oracle_d


def run_condition(cond, base_config, seeds, pairs_per_stratum):
    config = dict(base_config)
    config['obs_noise'] = cond['obs_noise']
    config.update({k: cond[k] for k in
                   ['friction_mass_range', 'friction_mu_range', 'mass_range']})
    sigma_total = sigma_total_of(SIGMA_FAMILY, config)
    records = []
    diagnostics = {'attempts': {}, 'screen_d': {}}
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed]))
        pairs = []
        for stratum in STRATA:
            for _ in range(pairs_per_stratum):
                a, b, d = sample_pair(rng, config, stratum, sigma_total, diagnostics)
                pairs.append({'type': stratum, 'a': a, 'b': b, 'identifiability_nats': d})
        for i, pair in enumerate(pairs):
            for policy in POLICIES:
                results = {}
                for label in ('a', 'b'):
                    ss = np.random.SeedSequence([seed, 9000, i, POLICIES.index(policy),
                                                 0 if label == 'a' else 1])
                    rng_policy, rng_noise = [np.random.default_rng(s) for s in ss.spawn(2)]
                    results[label] = probe_synthetic(
                        policy, pair, label, config, rng_policy, rng_noise, sigma_total)
                dim = pair['type']
                p_gt = results['a'][0].prob_greater(results['b'][0], dim)
                success = (p_gt > 0.5) == bool(pair['a'][dim] > pair['b'][dim])
                records.append({'seed': seed, 'pair': i, 'type': dim, 'policy': policy,
                                'success': bool(success),
                                'effort': (results['a'][1] + results['b'][1]) / 2,
                                'oracle_d': (results['a'][2] + results['b'][2]) / 2})
    summary = {'obs_noise': cond['obs_noise'], 'sigma_total': sigma_total.tolist(),
               'strata_ranges': {k: cond[k] for k in
                                 ['friction_mass_range', 'friction_mu_range', 'mass_range']},
               'pairs': len(records) // len(POLICIES),
               'screen_d_mean': {t: float(np.mean(v)) for t, v in diagnostics['screen_d'].items()},
               'attempts_per_pair': {t: diagnostics['attempts'][t] / (pairs_per_stratum * len(seeds))
                                     for t in STRATA},
               'policies': {}}
    for policy in POLICIES:
        rows = [r for r in records if r['policy'] == policy]
        summary['policies'][policy] = {
            'accuracy': float(np.mean([r['success'] for r in rows])),
            'effort': float(np.mean([r['effort'] for r in rows])),
            'by_type': {t: float(np.mean([r['success'] for r in rows if r['type'] == t]))
                        for t in STRATA}}
    summary['oracle_d_mean'] = float(np.mean(
        [r['oracle_d'] for r in records if r['policy'] == 'oracle']))
    return summary, records


def main(argv=None):
    parser = argparse.ArgumentParser(description='v0.2.2 pre-freeze prior information floor')
    parser.add_argument('--config', default='configs/v002-stage4.json')
    parser.add_argument('--pairs', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+', default=[31])
    parser.add_argument('--out', default='runs/v0022-prior-floor')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f'Refusing to overwrite non-empty {out}; pass --overwrite')
    out.mkdir(parents=True, exist_ok=True)

    base = json.loads(Path(args.config).read_text())
    base.update({'probe_budget': 5, 'min_friction_delta': 0.12,
                 'friction_mass_guard': 0.06, 'min_mass_delta': 0.25,
                 'mass_friction_guard': 0.03, 'identifiability_nats': 4.0})

    results = []
    for cond in SWEEP:
        summary, _ = run_condition(cond, base, args.seeds, args.pairs)
        results.append(summary)
        pol = {p: round(v['accuracy'], 3) for p, v in summary['policies'].items()}
        by_type = {p: {t: round(v, 3) for t, v in summary['policies'][p]['by_type'].items()}
                   for p in ('random', 'ig')}
        print(f"{cond['label']} sigma={cond['obs_noise']} "
              f"oracle_D={summary['oracle_d_mean']:.1f} -> {pol} | rnd/ig by type {by_type}",
              flush=True)

    out_json = {'sigma_family': SIGMA_FAMILY, 'design_targets': {
                    'oracle_ceiling': '>= 0.85', 'random': '<= 0.75'},
                'base_overrides': {k: base[k] for k in
                                   ['probe_budget', 'min_friction_delta', 'friction_mass_guard',
                                    'min_mass_delta', 'mass_friction_guard',
                                    'identifiability_nats']},
                'seeds': args.seeds, 'pairs_per_stratum': args.pairs,
                'conditions': results}
    (out / 'prior_floor.json').write_text(json.dumps(out_json, ensure_ascii=False, indent=2))
    print(f'wrote {out / "prior_floor.json"}')


if __name__ == '__main__':
    main()
