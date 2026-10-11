"""Stage-5 information-scarce discrimination test (docs/PROTOCOL-v002.2.md).

Environment changes vs v0.2.1 stage 4: probe budget 5, higher velocity
observation noise, and heavy/high-friction world-pair strata whose static
thresholds (mu*m*g) cover most of the force grid, so most from-rest actions
are uninformative and wasted probes cost real information. The design was
frozen only after the in-family prior information-floor calculation
(`runs/v0022-prior-floor/`, `src/world_discovery/prior_floor.py`) confirmed
the regime separates information-selecting from non-selecting policies.

Run:
    python -m world_discovery.stage5 --config configs/v002-stage5.json

Anti-cheat boundaries unchanged: evaluator-side truth (pair sampling,
identifiability, verdicts) never reaches policies or the particle reader;
sigma_family is calibrated on the training distribution only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from .env import WorldSpec
from .experiment import build_evaluation_queries
from .hypothesis import ParticleHypotheses
from .model import NeuralEnsemble
from .stage4 import (STRATA, aggregate, probe_world, sigma_total_of,
                     write_json)
from .discrimination import calibrate_sigma

T_SEED = 2.776  # two-sided 95%, df=4 (5 seeds)


# --------------------------------------------------------------------------
# Pair construction (evaluator side): scarce-information strata
# --------------------------------------------------------------------------
def sample_pair(rng, config, stratum, sigma_total):
    if stratum == 'friction':
        mass_range, mu_range = config['friction_mass_range'], config['friction_mu_range']
    else:
        mass_range, mu_range = config['mass_range'], config['train_friction']
    for _ in range(20000):
        a = WorldSpec(float(rng.uniform(*mass_range)),
                      float(rng.uniform(*mu_range)), False)
        b = WorldSpec(float(rng.uniform(*mass_range)),
                      float(rng.uniform(*mu_range)), False)
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
            dim = pair['type']
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
# Verdict (pre-registered in docs/PROTOCOL-v002.2.md; set after sweep)
# --------------------------------------------------------------------------
def verdict(by, paired, pre_registered):
    lines = []
    ig = by.get('ig', {}).get('accuracy', float('nan'))
    best = max(v['accuracy'] for v in by.values())
    edge_random = paired.get('ig_minus_random', {}).get('mean', float('nan'))
    edge_fixed7 = paired.get('ig_minus_fixed7', {}).get('mean', float('nan'))
    floor = pre_registered['ig_floor']
    if best < pre_registered['too_hard']:
        lines.append(f'环境过难条款触发：四策略最大准确率 {best:.3f} < {pre_registered["too_hard"]}。'
                     '不做策略排名。')
    elif ig < floor:
        lines.append(f'及格线未通过（绝对）：IG 绝对准确率 {ig:.3f} < 信息底门槛 {floor}。'
                     '按失败解读路径处理。')
    elif edge_random < pre_registered['edge_random']:
        lines.append(f'及格线未通过（对照）：IG 对 random 配对优势 {edge_random:+.3f} '
                     f'< {pre_registered["edge_random"]}。按失败解读路径处理。')
    else:
        lines.append(f'及格线通过：IG 绝对准确率 {ig:.3f} >= {floor}，对 random 优势 '
                     f'{edge_random:+.3f} >= {pre_registered["edge_random"]}，'
                     f'对 fixed7 优势 {edge_fixed7:+.3f}。')
        for key in ('ig_minus_random', 'ig_minus_fixed7'):
            ci = paired[key]['ci95']
            lines.append(f"{key} 种子级 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                         + ('，排除 0。' if ci[0] > 0 else '，但未排除 0。'))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description='v0.2.2 stage-5 scarce-information discrimination test')
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='runs/v002-stage5')
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
    pre_registered = config['pre_registered']
    summary = {'protocol': config['protocol'], 'policies': by, 'paired': paired,
               'sigma_total': sigma_total.tolist(), 'verdict': verdict(by, paired, pre_registered),
               'pre_registered': pre_registered}
    write_json(out / 'summary.json', summary)

    versions = {n: importlib.metadata.version(n) for n in ['mujoco', 'numpy']}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), Path(__file__).parent / 'hypothesis.py',
                        Path(__file__).parent / 'stage4.py',
                        Path(config['protocol_doc'])]}
    write_json(out / 'manifest.json', {'python_packages': versions, 'sha256': hashes})

    print(json.dumps({'policies': {k: round(v['accuracy'], 3) for k, v in by.items()},
                      'by_type': {k: {t: round(v, 3) for t, v in by[k]['by_type'].items()} for k in by},
                      'verdict': summary['verdict']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
