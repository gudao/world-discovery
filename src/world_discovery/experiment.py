import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import importlib.metadata
import numpy as np
from .env import PushWorld, WorldSpec, actions, model_input, sample_spec
from .model import NeuralEnsemble, LocalMemory
from .explorer import choose_action


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate_config(c):
    required = ['seed', 'train_worlds', 'validation_worlds', 'test_worlds', 'steps_per_world',
                'ensemble', 'hidden', 'epochs', 'learning_rate', 'batch_size', 'probe_budget',
                'budgets', 'forces', 'duration', 'timestep', 'ridge', 'base_features',
                'wide_features', 'exploration_probability', 'cost_weight', 'growth_min_samples',
                'growth_holdout', 'growth_min_gain', 'growth_penalty', 'error_quantile',
                'train_mass', 'test_mass', 'train_friction', 'test_friction', 'fixed_probability']
    for k in required:
        if k not in c:
            raise ValueError(f'Missing configuration: {k}')
    for k in ['train_worlds', 'validation_worlds', 'test_worlds', 'steps_per_world', 'ensemble',
              'hidden', 'epochs', 'batch_size', 'probe_budget', 'growth_holdout']:
        if not isinstance(c[k], int) or c[k] <= 0:
            raise ValueError(f'{k} must be a positive integer')
    if c['ridge'] <= 0 or c['learning_rate'] <= 0 or c['timestep'] <= 0:
        raise ValueError('ridge, learning_rate and timestep must be positive')
    if not 0 <= c['base_features'] < c['wide_features']:
        raise ValueError('Require 0 <= base_features < wide_features')
    if not c['growth_holdout'] < c['growth_min_samples'] <= c['probe_budget']:
        raise ValueError('Require holdout < growth_min_samples <= probe_budget')
    if not c['budgets'] or max(c['budgets']) != c['probe_budget'] or min(c['budgets']) != 0:
        raise ValueError('budgets must include 0 and probe_budget')
    if any(not isinstance(b, int) or b < 0 for b in c['budgets']):
        raise ValueError('budgets must be nonnegative integers')
    for k in ['exploration_probability', 'fixed_probability']:
        if not 0 <= c[k] <= 1:
            raise ValueError(f'{k} must be in [0, 1]')
    if not 0 < c['error_quantile'] < 1:
        raise ValueError('error_quantile must be in (0,1)')
    for k in ['train_mass', 'test_mass', 'train_friction', 'test_friction']:
        if len(c[k]) != 2 or c[k][0] <= 0 or c[k][1] < c[k][0]:
            raise ValueError(f'Invalid range: {k}')
    if not np.isclose(c['duration'] / c['timestep'], round(c['duration'] / c['timestep'])):
        raise ValueError('duration must be a multiple of timestep')
    if c['duration'] <= 0 or not c['forces'] or not np.isfinite(c['forces']).all():
        raise ValueError('Invalid actions')
    return c


def collect(config, out, experiment):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    validate_config(config)
    candidates = actions(config)
    audit = {}
    split_ids = []
    for split, offset, count in [('train', 1000, config['train_worlds']),
                                  ('validation', 2000, config['validation_worlds']),
                                  ('test', 3000, config['test_worlds'])]:
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], offset]))
        specs, x, y, episode, observations, action_log, next_obs = [], [], [], [], [], [], []
        for i in range(count):
            spec = sample_spec(rng, config, split, experiment)
            specs.append(spec.__dict__)
            identity = f'{split}:{i}'
            split_ids.append(identity)
            if split == 'test':
                continue
            env = PushWorld(spec, config['timestep'])
            for step in range(config['steps_per_world']):
                if step % 5 == 0:
                    env.reset()
                o = env.observe()
                a = candidates[int(rng.integers(len(candidates)))]
                after = env.step(a)
                x.append(model_input(o, a))
                y.append(after - o)
                episode.append(i)
                observations.append(o)
                action_log.append(a)
                next_obs.append(after)
        audit[split] = specs
        if split != 'test':
            np.savez_compressed(out / f'{split}.npz', x=np.array(x), y=np.array(y),
                                episode_id=np.array(episode), observation=np.array(observations),
                                action=np.array(action_log), next_observation=np.array(next_obs))
    write_json(out / 'evaluator_only_worlds.json', audit)
    write_json(out / 'config.json', config)
    write_json(out / 'protocol.json', {'experiment': experiment,
               'split_ids_unique': len(split_ids) == len(set(split_ids)),
               'hidden_truth_location': 'evaluator_only_worlds.json',
               'observation': ['ideal_x_position', 'ideal_x_velocity'],
               'input': ['x_velocity', 'force', 'duration'],
               'prediction': ['delta_x_position', 'delta_x_velocity']})


def train(config, out):
    out = Path(out)
    with np.load(out / 'train.npz', allow_pickle=False) as data:
        x, y = data['x'], data['y']
    model = NeuralEnsemble()
    losses = model.fit(x, y, config)
    model.save(out / 'model.npz')
    with np.load(out / 'validation.npz', allow_pickle=False) as data:
        pred = model.predict_members(data['x']).mean(0)
        residual = pred - model.normalize_y(data['y'])
        errors = np.mean(residual ** 2, axis=1)
    threshold = float(np.quantile(errors, config['error_quantile']))
    write_json(out / 'calibration.json', {'normalized_error_threshold': threshold,
               'validation_nmse': float(errors.mean()), 'training_member_nmse': losses,
               'note': 'Global validation threshold; conservative heuristic, not calibrated post-adaptation.'})


def build_evaluation_queries(spec, config):
    """Evaluator-only counterfactuals; agent sees none of these labels or states.

    Same queries for all policies. Three initial conditions are generated using
    a fixed prior impulse; each action is tested from an exact restored state.
    """
    env = PushWorld(spec, config['timestep'])
    x, y = [], []
    for impulse in [0, -6, 6]:
        env.reset()
        if impulse:
            env.step([impulse, config['duration']])
        state = env.snapshot()
        for a in actions(config):
            env.restore(state)
            o = env.observe()
            x.append(model_input(o, a))
            y.append(env.step(a) - o)
    return np.array(x), np.array(y)


def evaluate(config, out, experiment):
    out = Path(out)
    model = NeuralEnsemble.load(out / 'model.npz')
    threshold = json.loads((out / 'calibration.json').read_text())['normalized_error_threshold']
    specs = json.loads((out / 'evaluator_only_worlds.json').read_text())['test']
    candidates = actions(config)
    groups = [('random', 'small'), ('fixed', 'small'), ('active', 'small'), ('active', 'none')]
    if experiment == 'B':
        # Both random and active schedules: separates capacity from acquisition.
        groups += [('active', 'wide'), ('active', 'grow'), ('random', 'wide'), ('random', 'grow')]
    rows, logs, growth = [], [], []
    budgets = set(config['budgets'])
    for i, item in enumerate(specs):
        spec = WorldSpec(**item)
        eval_x, eval_y = build_evaluation_queries(spec, config)
        for policy, mode in groups:
            # Same per-world RNG stream for randomized policies. Different policy
            # consumption can diverge; fixed-world queries remain exactly paired.
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 4000, i]))
            env = PushWorld(spec, config['timestep'])
            memory = LocalMemory(model, config, mode)
            effort = 0.0
            for step in range(config['probe_budget'] + 1):
                if step in budgets:
                    residual = memory.predict(eval_x) - eval_y
                    rows.append({'seed': config['seed'], 'world': i, 'fixed': spec.fixed,
                                 'policy': policy, 'memory': mode, 'budget': step,
                                 'nmse': float(np.mean((residual / model.y_scale) ** 2)),
                                 'position_mse': float(np.mean(residual[:, 0] ** 2)),
                                 'velocity_mse': float(np.mean(residual[:, 1] ** 2)),
                                 'feature_width': memory.width,
                                 'adapter_parameters': 0 if mode == 'none' else (4 + memory.width) * 2 * len(model.members),
                                 'interaction_seconds': step * config['duration'],
                                 'force_time_effort': effort})
                if step == config['probe_budget']:
                    break
                observation = env.observe()
                action = choose_action(policy, memory, observation, candidates, rng, step, config)
                after = env.step(action)
                memory.update(model_input(observation, action), after - observation, threshold)
                effort += abs(float(action[0])) * float(action[1])
                logs.append({'world': i, 'policy': policy, 'memory': mode, 'step': step,
                             'observation': observation.tolist(), 'action': action.tolist(),
                             'next_observation': after.tolist(), 'prequential_nmse': memory.errors[-1],
                             'feature_width': memory.width})
            growth += [dict(event, world=i, policy=policy, memory=mode) for event in memory.events]
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / 'interactions.jsonl').open('w', encoding='utf-8') as f:
        for log in logs:
            f.write(json.dumps(log, allow_nan=False) + '\n')
    write_json(out / 'growth_events.json', growth)
    summaries = []
    for policy, mode in groups:
        for budget in sorted(budgets):
            selected = [r for r in rows if r['policy'] == policy and r['memory'] == mode and r['budget'] == budget]
            values = np.array([r['nmse'] for r in selected])
            summaries.append({'policy': policy, 'memory': mode, 'budget': budget,
                              'mean_nmse': float(values.mean()),
                              'world_standard_error': float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0,
                              'worlds': len(values)})
    # Paired-world difference: positive means active improves over random.
    final = config['probe_budget']
    random = np.array([r['nmse'] for r in rows if r['policy'] == 'random' and r['memory'] == 'small' and r['budget'] == final])
    active = np.array([r['nmse'] for r in rows if r['policy'] == 'active' and r['memory'] == 'small' and r['budget'] == final])
    difference = random - active
    write_json(out / 'summary.json', {'curves': summaries,
               'paired_random_minus_active': float(difference.mean()),
               'paired_world_standard_error': float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else 0,
               'accepted_growth_events': sum(e['accepted'] for e in growth),
               'growth_tests': len(growth),
               'interpretation': 'Single-seed descriptive result; no claim of new concepts or superiority.'})
    from .report import make_report
    make_report(out, summaries, experiment)


def manifest(out, elapsed):
    out = Path(out)
    versions = {name: importlib.metadata.version(name) for name in ['mujoco', 'numpy', 'matplotlib']}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.iterdir() if p.is_file() and p.suffix in ['.npz', '.json', '.csv', '.jsonl'] and p.name != 'manifest.json'}
    root = Path(__file__).resolve().parents[2]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root / 'src').rglob('*.py'))}
    write_json(out / 'manifest.json', {'python': platform.python_version(), 'platform': platform.platform(),
               'versions': versions, 'elapsed_seconds': elapsed, 'sha256': hashes, 'source_sha256': source_hashes})


def pipeline(config, out, experiment):
    start = time.perf_counter()
    print(f'[{experiment}] collecting MuJoCo trajectories...', flush=True)
    collect(config, out, experiment)
    print(f'[{experiment}] training neural ensemble...', flush=True)
    train(config, out)
    print(f'[{experiment}] evaluating paired worlds...', flush=True)
    evaluate(config, out, experiment)
    manifest(out, time.perf_counter() - start)
    print(f'Complete: {Path(out).resolve()}', flush=True)
