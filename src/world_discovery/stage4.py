"""Stage-4 hardened discrimination test (docs/PROTOCOL-v002.1.md).

Environment changes vs v0.2.0 stage 1: observation noise, a 25-level fine
force grid, a smaller probe budget (8) and smaller parameter deltas, so
that information gain varies strongly across candidate actions.

Run:
    python -m world_discovery.stage4 --config configs/v002-stage4.json

Anti-cheat boundaries are unchanged: evaluator-side truth (pair sampling,
identifiability, verdicts) never reaches policies or the particle reader;
sigma_family is calibrated on the training distribution only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np

from .env import PushWorld, WorldSpec, model_input
from .experiment import build_evaluation_queries
from .explorer import choose_action
from .hypothesis import ParticleHypotheses
from .model import LocalMemory, NeuralEnsemble
from .discrimination import calibrate_sigma, write_json

T_SEED = 2.776  # two-sided 95%, df=4 (5 seeds)
STRATA = ('friction', 'mass')


class NoisyWorld:
    """Sensor-spec observation noise on top of the frozen v0.1 environment."""

    def __init__(self, spec, config, rng):
        self.env = PushWorld(spec, config['timestep'])
        self.noise = np.asarray(config['obs_noise'], dtype=float)
        self.rng = rng
        self.env.reset()

    def _noisy(self, obs):
        return np.asarray(obs, dtype=float) + self.rng.normal(0.0, self.noise)

    def observe(self):
        return self._noisy(self.env.observe())

    def step(self, action):
        return self._noisy(self.env.step(action))

    def reset(self):
        return self._noisy(self.env.reset())


def candidate_actions(config):
    return np.array([[f, config['duration']] for f in config['forces']], dtype=float)


def sigma_total_of(sigma_family, config):
    """Deltas are differences of two noisy observations (protocol 2.2)."""
    return np.sqrt(np.asarray(sigma_family, float) ** 2
                   + 2.0 * np.asarray(config['obs_noise'], float) ** 2)


# --------------------------------------------------------------------------
# Pair construction (evaluator side)
# --------------------------------------------------------------------------
def sample_pair(rng, config, stratum, sigma_total):
    for _ in range(20000):
        a = WorldSpec(float(rng.uniform(*config['train_mass'])),
                      float(rng.uniform(*config['train_friction'])), False)
        b = WorldSpec(float(rng.uniform(*config['train_mass'])),
                      float(rng.uniform(*config['train_friction'])), False)
        if stratum == 'friction':
            ok = (abs(a.friction - b.friction) >= config['min_friction_delta']
                  and abs(a.mass - b.mass) <= config['friction_mass_guard'])
        else:
            ok = (abs(a.mass - b.mass) >= config['min_mass_delta']
                  and abs(a.friction - b.friction) <= config['mass_friction_guard'])
        if not ok:
            continue
        ya = build_evaluation_queries(a, config)[1]
        yb = build_evaluation_queries(b, config)[1]
        d = float(np.sum((ya - yb) ** 2 / (2.0 * sigma_total[None, :] ** 2)))
        if d >= config['identifiability_nats']:
            return ({'mass': a.mass, 'friction': a.friction, 'fixed': a.fixed},
                    {'mass': b.mass, 'friction': b.friction, 'fixed': b.fixed}, d)
    raise RuntimeError(f'Could not sample {stratum} pair')


def make_pairs(rng, config, sigma_total):
    pairs = []
    for stratum in STRATA:
        n = config['pair_strata'][stratum]
        for _ in range(n):
            a, b, d = sample_pair(rng, config, stratum, sigma_total)
            pairs.append({'type': stratum, 'a': a, 'b': b, 'identifiability_nats': d})
    return pairs


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------
def probe_world(policy, spec_dict, config, rng_policy, rng_noise, model, threshold,
                sigma_total, v01cfg, score_rows, tags):
    spec = WorldSpec(**spec_dict)
    world = NoisyWorld(spec, config, rng_noise)
    world.reset()
    cands = candidate_actions(config)
    reader = ParticleHypotheses(config, rng_policy, sigma_total)
    memory = LocalMemory(model, v01cfg, 'small') if policy == 'v01_active' else None
    fixed7 = config['fixed7_forces']
    prev, effort, forces_taken = None, 0.0, []
    for step in range(config['probe_budget']):
        obs = world.observe()
        if policy == 'random':
            action, branch = cands[int(rng_policy.integers(len(cands)))].copy(), 'random'
        elif policy == 'fixed7':
            # 2026-10-11 defect fix: the original version mapped indices of
            # fixed7_forces into the 25-level candidate grid, silently executing
            # a descending single-sided scan [-12..-8] instead of the protocol's
            # classic sweep [0,-12,-6,-2,2,6,12]. Both stage 4 and the first
            # stage-5 run used the buggy baseline; see the errata in their
            # reports. The force is now taken directly from the frozen list.
            action = np.array([fixed7[step % len(fixed7)], config['duration']], dtype=float)
            branch = 'fixed7'
        elif policy == 'v01_active':
            action = choose_action('active', memory, obs, cands, rng_policy, step, v01cfg)
            branch = 'v01_active'
        elif policy == 'ig':
            action, info = reader.choose(obs, cands, prev)
            branch = info['branch']
            score_rows.append({**tags, 'step': step, 'action_index': info.get('action_index'),
                               'force': float(action[0]), 'branch': branch,
                               'ig': info.get('ig'), 'cost': info.get('cost')})
        else:
            raise ValueError(policy)
        nxt = world.step(action)
        reader.update(obs, action, nxt)
        if memory is not None:
            memory.update(model_input(obs, action), nxt - obs, threshold)
        effort += abs(float(action[0])) * float(action[1])
        forces_taken.append(float(action[0]))
        prev = action
    return reader, {'effort': effort, 'forces': forces_taken}


def run_seed(seed, config, model, threshold, sigma_total, v01cfg, score_rows):
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    pairs = make_pairs(rng, config, sigma_total)
    records = []
    for i, pair in enumerate(pairs):
        for policy in config['policies']:
            row = {'seed': seed, 'pair': i, 'type': pair['type'], 'policy': policy}
            results = {}
            for label in ('a', 'b'):
                ss = np.random.SeedSequence([seed, 8000, i, config['policies'].index(policy),
                                             0 if label == 'a' else 1])
                rng_policy, rng_noise = [np.random.default_rng(s) for s in ss.spawn(2)]
                reader, extra = probe_world(
                    policy, pair[label], config, rng_policy, rng_noise, model, threshold,
                    sigma_total, v01cfg, score_rows,
                    {'seed': seed, 'pair': i, 'world': label, 'policy': policy})
                results[label] = (reader, extra)
            dim = pair['type']  # 'friction' or 'mass'
            p_gt = results['a'][0].prob_greater(results['b'][0], dim)
            direction = pair['a'][dim] > pair['b'][dim]
            success = (p_gt > 0.5) == bool(direction)
            row.update({'p_a_gt_b': p_gt, 'success': bool(success),
                        'post_a': results['a'][0].posterior_summary(),
                        'post_b': results['b'][0].posterior_summary(),
                        'effort_a': results['a'][1]['effort'], 'effort_b': results['b'][1]['effort'],
                        'forces_a': results['a'][1]['forces'], 'forces_b': results['b'][1]['forces']})
            records.append(row)
    return pairs, records


# --------------------------------------------------------------------------
# Aggregation and verdict
# --------------------------------------------------------------------------
def aggregate(records, seeds):
    def acc(rows):
        return float(np.mean([r['success'] for r in rows])) if rows else float('nan')

    by = {}
    for policy in sorted({r['policy'] for r in records}):
        rows = [r for r in records if r['policy'] == policy]
        by[policy] = {'accuracy': acc(rows), 'pairs': len(rows),
                      'effort': float(np.mean([(r['effort_a'] + r['effort_b']) / 2 for r in rows])),
                      'by_type': {t: acc([r for r in rows if r['type'] == t]) for t in STRATA},
                      'by_seed': {s: acc([r for r in rows if r['seed'] == s]) for s in seeds}}
    keyed = {}
    for r in records:
        keyed.setdefault((r['seed'], r['pair']), {})[r['policy']] = r
    paired = {}
    for other in [p for p in by if p != 'ig']:
        per_seed = []
        for s in seeds:
            ids = sorted({i for (ss, i) in keyed if ss == s})
            diff = np.array([keyed[(s, i)]['ig']['success'] for i in ids], float) \
                 - np.array([keyed[(s, i)][other]['success'] for i in ids], float)
            per_seed.append(float(diff.mean()))
        mean = float(np.mean(per_seed))
        se = float(np.std(per_seed, ddof=1) / math.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
        paired[f'ig_minus_{other}'] = {'per_seed': per_seed, 'mean': mean, 'seed_se': se,
                                       'ci95': [mean - T_SEED * se, mean + T_SEED * se]}
    return by, paired


def verdict(by, paired):
    lines = []
    fixed7 = by.get('fixed7', {}).get('accuracy', float('nan'))
    ig = by.get('ig', {}).get('accuracy', float('nan'))
    best = max(v['accuracy'] for v in by.values())
    edge = paired.get('ig_minus_fixed7', {}).get('mean', float('nan'))
    if fixed7 >= 0.95:
        lines.append(f'环境过便宜条款触发：fixed7 = {fixed7:.3f} >= 0.95。不做策略排名，环境仍不足以检验原则。')
    elif best < 0.60:
        lines.append(f'环境过难条款触发：四策略最大准确率 {best:.3f} < 0.60。区分落于噪声之下不可达，不做策略排名。')
    elif ig >= 0.85 and edge >= 0.10:
        lines.append(f'及格线通过：IG 绝对准确率 {ig:.3f} >= 0.85，对 fixed7 优势 {edge:+.3f} >= 0.10。')
        ci = paired['ig_minus_fixed7']['ci95']
        lines.append(f"种子级 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                     + ('，排除 0。' if ci[0] > 0 else '，但未排除 0。'))
    else:
        lines.append(f'及格线未通过：IG 绝对准确率 {ig:.3f}（门槛 0.85），对 fixed7 优势 {edge:+.3f}（门槛 0.10）。')
        lines.append('按协议失败解读：审计评分日志与粒子族误差定位，修正后用新采样世界对重测；不得回调阈值。')
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description='v0.2.1 stage-4 hardened discrimination test')
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='runs/v002-stage4')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f'Refusing to overwrite non-empty {out}; pass --overwrite')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', config)

    family_path = out / 'sigma_family.json'
    if not family_path.exists():
        write_json(family_path, calibrate_sigma(config))
    family = np.asarray(json.loads(family_path.read_text())['sigma'], dtype=float)
    sigma_total = sigma_total_of(family, config)
    write_json(out / 'sigma_total.json',
               {'sigma_family': family.tolist(), 'obs_noise': config['obs_noise'],
                'sigma_total': sigma_total.tolist()})

    frozen_dir = Path(config['frozen_dir'])
    model = NeuralEnsemble.load(frozen_dir / 'model.npz')
    threshold = json.loads((frozen_dir / 'calibration.json').read_text())['normalized_error_threshold']
    v01cfg = json.loads((frozen_dir / 'config.json').read_text())

    score_rows, records, pairs_by_seed = [], [], {}
    for seed in config['seeds']:
        pairs, recs = run_seed(seed, config, model, threshold, sigma_total, v01cfg, score_rows)
        pairs_by_seed[str(seed)] = pairs
        records.extend(recs)
        print(f'seed {seed}: {len(recs)} policy-runs over {len(pairs)} pairs', flush=True)

    write_json(out / 'pairs.json', pairs_by_seed)
    with (out / 'score_log.jsonl').open('w', encoding='utf-8') as f:
        for row in score_rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(out / 'records.json', records)

    by, paired = aggregate(records, config['seeds'])
    summary = {'protocol': config['protocol'], 'policies': by, 'paired': paired,
               'sigma_total': sigma_total.tolist(), 'verdict': verdict(by, paired),
               'pre_registered': {'ig_accuracy': 0.85, 'edge_over_fixed7': 0.10,
                                  'fixed7_too_easy': 0.95, 'all_too_hard': 0.60}}
    write_json(out / 'summary.json', summary)

    versions = {n: importlib.metadata.version(n) for n in ['mujoco', 'numpy']}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), Path(__file__).parent / 'hypothesis.py',
                        Path(config['protocol_doc'])]}
    write_json(out / 'manifest.json', {'python_packages': versions, 'sha256': hashes})

    print(json.dumps({'policies': {k: round(v['accuracy'], 3) for k, v in by.items()},
                      'by_type': {k: {t: round(v, 3) for t, v in by[k]['by_type'].items()} for k in by},
                      'verdict': summary['verdict']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
