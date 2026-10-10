"""Stage-1 discrimination test (docs/PROTOCOL-v002.md section 3).

Run:
    python -m world_discovery.discrimination --config configs/v002-discrimination.json --calibrate
    python -m world_discovery.discrimination --config configs/v002-discrimination.json

Frozen v0.1 artifacts (examples/quick-A) provide the "old representation"
for screening and the v0.1-active negative control. No v0.1 file is read
for anything other than the global model, its calibration threshold and
its action-scoring config. Evaluator-side truth (pair construction,
identifiability, verdicts) never reaches policies or the particle reader.
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
from .hypothesis import ParticleHypotheses, analytic_rollout
from .model import LocalMemory, NeuralEnsemble

T_SEED = 2.776  # two-sided 95%, df=4 (5 seeds)
PAIR_TYPES = ('friction', 'mass', 'fixed')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def candidate_actions(config):
    return np.array([[f, config['duration']] for f in config['forces']], dtype=float)


# --------------------------------------------------------------------------
# Sigma calibration (training distribution only; protocol section 8.3)
# --------------------------------------------------------------------------
def calibrate_sigma(config):
    rng = np.random.default_rng(config['calibration_seed'])
    residuals = []
    for _ in range(config['calibration_worlds']):
        spec = WorldSpec(float(rng.uniform(*config['train_mass'])),
                         float(rng.uniform(*config['train_friction'])),
                         bool(rng.random() < config['fixed_probability']))
        env = PushWorld(spec, config['timestep'])
        env.reset()
        forces = config['forces']
        for _ in range(config['calibration_steps']):
            obs = env.observe()
            force = float(forces[int(rng.integers(len(forces)))])
            action = [force, config['duration']]
            nxt = env.step(action)
            x1, v1 = analytic_rollout(np.array([obs[0]]), np.array([obs[1]]),
                                      force, config['duration'],
                                      np.array([spec.mass]), np.array([spec.friction]),
                                      np.array([spec.fixed]), config['timestep'])
            residuals.append([(x1[0] - obs[0]) - (nxt[0] - obs[0]),
                              (v1[0] - obs[1]) - (nxt[1] - obs[1])])
    resid = np.array(residuals)
    raw_std = resid.std(axis=0)
    sigma = raw_std * config['sigma_inflation']
    return {'sigma': sigma.tolist(), 'raw_std': raw_std.tolist(),
            'residual_mean': resid.mean(axis=0).tolist(),
            'n_samples': int(len(resid)),
            'inflation': config['sigma_inflation'],
            'note': 'Analytic family with TRUE parameters vs MuJoCo on the '
                    'training distribution; sigma never touches test truth.'}


# --------------------------------------------------------------------------
# Pair construction (evaluator side)
# --------------------------------------------------------------------------
def sample_spec_pair(rng, config, pair_type):
    """Rejection-sample a typed pair. Typing rule for '分型记录':

    friction: both moving, |d_mu| >= 0.20, |d_mass| <= 0.25
    mass:     both moving, |d_mass| >= 0.50, |d_mu| <= 0.10
    fixed:    exactly one fixed, the other moving, parameters free
    """
    lo_m, hi_m = config['train_mass']
    lo_f, hi_f = config['train_friction']
    for _ in range(10000):
        a = WorldSpec(float(rng.uniform(lo_m, hi_m)), float(rng.uniform(lo_f, hi_f)),
                      pair_type == 'fixed' and rng.random() < 0.5)
        b = WorldSpec(float(rng.uniform(lo_m, hi_m)), float(rng.uniform(lo_f, hi_f)),
                      pair_type == 'fixed' and not a.fixed)
        if pair_type == 'friction':
            ok = (not a.fixed and not b.fixed and abs(a.friction - b.friction) >= config['min_friction_delta']
                  and abs(a.mass - b.mass) <= 0.25)
        elif pair_type == 'mass':
            ok = (not a.fixed and not b.fixed and abs(a.mass - b.mass) >= config['min_mass_delta']
                  and abs(a.friction - b.friction) <= 0.10)
        else:
            ok = a.fixed != b.fixed
        if ok:
            return a, b
    raise RuntimeError(f'Could not sample pair of type {pair_type}')


def identifiable(a, b, config, sigma):
    """At least `min_queries` standard queries where the two worlds differ by
    >= 6 sigma in some output dimension (evaluator-side oracle check)."""
    need = config['identifiability_sigma_multiple'] * np.asarray(sigma)
    count = 0
    ya = build_evaluation_queries(a, config)[1]
    yb = build_evaluation_queries(b, config)[1]
    diff = np.abs(ya - yb)
    count = int(np.sum(np.any(diff >= need[None, :], axis=1)))
    return count >= config['identifiability_min_queries'], count


def make_pairs(rng, config, sigma):
    pairs = []
    per_type = config['pairs_per_seed'] // len(PAIR_TYPES)
    extra = config['pairs_per_seed'] - per_type * len(PAIR_TYPES)
    for i, pair_type in enumerate(PAIR_TYPES):
        n = per_type + (1 if i < extra else 0)
        made = 0
        while made < n:
            a, b = sample_spec_pair(rng, config, pair_type)
            ok, count = identifiable(a, b, config, sigma)
            if ok:
                pairs.append({'type': pair_type, 'a': {'mass': a.mass, 'friction': a.friction, 'fixed': a.fixed},
                              'b': {'mass': b.mass, 'friction': b.friction, 'fixed': b.fixed},
                              'identifiable_queries': count})
                made += 1
    return pairs


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------
def probe_world(policy, spec_dict, config, rng, model, threshold, sigma, v01cfg,
                score_rows, tags):
    spec = WorldSpec(**spec_dict)
    env = PushWorld(spec, config['timestep'])
    env.reset()
    cands = candidate_actions(config)
    reader = ParticleHypotheses(config, rng, sigma)
    memory = LocalMemory(model, v01cfg, 'small') if policy == 'v01_active' else None
    prev, effort, forces_taken = None, 0.0, []
    for step in range(config['probe_budget']):
        obs = env.observe()
        if policy == 'random':
            action, branch = cands[int(rng.integers(len(cands)))].copy(), 'random'
        elif policy == 'fixed':
            action, branch = cands[step % len(cands)].copy(), 'fixed'
        elif policy == 'v01_active':
            action = choose_action('active', memory, obs, cands, rng, step, v01cfg)
            branch = 'v01_active'
        elif policy == 'ig':
            action, info = reader.choose(obs, cands, prev)
            branch = info['branch']
            score_rows.append({**tags, 'step': step, 'action_index': info.get('action_index'),
                               'force': float(action[0]), 'branch': branch,
                               'ig': info.get('ig'), 'cost': info.get('cost'),
                               'all_ig': info.get('all_ig'), 'all_cost': info.get('all_cost')})
        else:
            raise ValueError(policy)
        nxt = env.step(action)
        reader.update(obs, action, nxt)
        if memory is not None:
            memory.update(model_input(obs, action), nxt - obs, threshold)
        effort += abs(float(action[0])) * float(action[1])
        forces_taken.append(float(action[0]))
        prev = action
    return reader, {'effort': effort, 'forces': forces_taken}


def dimension_of(pair_type):
    return {'friction': 'friction', 'mass': 'mass', 'fixed': 'fixed'}[pair_type]


def true_direction(pair, pair_type):
    dim = dimension_of(pair_type)
    return pair['a'][dim] > pair['b'][dim]


def run_seed(seed, config, model, threshold, sigma, v01cfg, score_rows):
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    pairs = make_pairs(rng, config, sigma)
    records = []
    for i, pair in enumerate(pairs):
        for policy in config['policies']:
            row = {'seed': seed, 'pair': i, 'type': pair['type'], 'policy': policy}
            results = {}
            for label in ('a', 'b'):
                world_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, 7000, i, config['policies'].index(policy),
                                            0 if label == 'a' else 1]))
                reader, extra = probe_world(
                    policy, pair[label], config, world_rng, model, threshold, sigma,
                    v01cfg, score_rows,
                    {'seed': seed, 'pair': i, 'world': label, 'policy': policy})
                results[label] = (reader, extra)
            dim = dimension_of(pair['type'])
            p_gt = results['a'][0].prob_greater(results['b'][0], dim)
            direction = true_direction(pair, pair['type'])
            success = (p_gt > 0.5) == bool(direction)
            row.update({'p_a_gt_b': p_gt, 'success': bool(success),
                        'post_a': results['a'][0].posterior_summary(),
                        'post_b': results['b'][0].posterior_summary(),
                        'effort_a': results['a'][1]['effort'], 'effort_b': results['b'][1]['effort'],
                        'forces_a': results['a'][1]['forces'], 'forces_b': results['b'][1]['forces']})
            records.append(row)
    return pairs, records


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate(records, seeds):
    def acc(rows):
        return float(np.mean([r['success'] for r in rows])) if rows else float('nan')

    by = {}
    for policy in sorted({r['policy'] for r in records}):
        rows = [r for r in records if r['policy'] == policy]
        by[policy] = {'accuracy': acc(rows), 'pairs': len(rows),
                      'effort': float(np.mean([r['effort_a'] + r['effort_b'] for r in rows]) / 2),
                      'by_type': {t: acc([r for r in rows if r['type'] == t]) for t in PAIR_TYPES},
                      'by_seed': {s: acc([r for r in rows if r['seed'] == s]) for s in seeds}}
    paired = {}
    keyed = {}
    for r in records:
        keyed.setdefault((r['seed'], r['pair']), {})[r['policy']] = r
    for other in [p for p in by if p != 'ig']:
        per_seed = []
        for s in seeds:
            succ_ig = [keyed[(s, i)]['ig']['success'] for i in sorted(
                {i for (ss, i) in keyed if ss == s})]
            succ_o = [keyed[(s, i)][other]['success'] for i in sorted(
                {i for (ss, i) in keyed if ss == s})]
            per_seed.append(float(np.mean(np.array(succ_ig, float) - np.array(succ_o, float))))
        mean = float(np.mean(per_seed))
        se = float(np.std(per_seed, ddof=1) / math.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
        paired[f'ig_minus_{other}'] = {'per_seed': per_seed, 'mean': mean,
                                       'seed_se': se,
                                       'ci95': [mean - T_SEED * se, mean + T_SEED * se]}
    return by, paired


def verdict(by, paired):
    lines = []
    fixed_acc = by.get('fixed', {}).get('accuracy', float('nan'))
    ig_acc = by.get('ig', {}).get('accuracy', float('nan'))
    edge = paired.get('ig_minus_fixed', {}).get('mean', float('nan'))
    if fixed_acc >= 0.95:
        lines.append('环境过便宜条款触发：fixed >= 0.95。不做策略排名，进入协议第 7 节阶段 4 环境升级。')
    elif ig_acc >= 0.85 and edge >= 0.10:
        lines.append(f'及格线通过：IG 绝对准确率 {ig_acc:.3f} >= 0.85，对 fixed 优势 {edge:+.3f} >= 0.10。')
        ci = paired['ig_minus_fixed']['ci95']
        lines.append(f"种子级 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]，{'' if ci[0] > 0 else '但未排除 0'}。")
    else:
        lines.append(f'及格线未通过：IG 绝对准确率 {ig_acc:.3f}（门槛 0.85），对 fixed 优势 {edge:+.3f}（门槛 0.10）。')
        lines.append('按协议第 3 节失败解读：先审计评分日志与粒子族误差定位，修正后用新采样世界对重测；不得回调阈值。')
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description='v0.2 stage-1 discrimination test')
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='runs/v002-discrimination')
    parser.add_argument('--calibrate', action='store_true', help='(re)compute sigma on the training distribution')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f'Refusing to overwrite non-empty {out}; pass --overwrite')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', config)

    sigma_path = out / 'sigma.json'
    if args.calibrate or not sigma_path.exists():
        write_json(sigma_path, calibrate_sigma(config))
    sigma = np.asarray(json.loads(sigma_path.read_text())['sigma'], dtype=float)

    frozen_dir = Path(config['frozen_dir'])
    model = NeuralEnsemble.load(frozen_dir / 'model.npz')
    threshold = json.loads((frozen_dir / 'calibration.json').read_text())['normalized_error_threshold']
    v01cfg = json.loads((frozen_dir / 'config.json').read_text())

    score_rows, records, pairs_by_seed = [], [], {}
    for seed in config['seeds']:
        pairs, recs = run_seed(seed, config, model, threshold, sigma, v01cfg, score_rows)
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
               'sigma': sigma.tolist(), 'verdict': verdict(by, paired),
               'pre_registered': {'ig_accuracy': 0.85, 'edge_over_fixed': 0.10,
                                  'fixed_too_easy': 0.95}}
    write_json(out / 'summary.json', summary)

    versions = {n: importlib.metadata.version(n) for n in ['mujoco', 'numpy']}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), Path(__file__).parent / 'hypothesis.py',
                        Path(config.get('protocol_doc', 'docs/PROTOCOL-v002.md'))]}
    write_json(out / 'manifest.json', {'python_packages': versions, 'sha256': hashes})

    print(json.dumps({'policies': {k: round(v['accuracy'], 3) for k, v in by.items()},
                      'verdict': summary['verdict']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
