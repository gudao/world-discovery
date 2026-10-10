import numpy as np
from .env import model_input


def choose_action(policy, memory, observation, candidates, rng, step, config):
    if policy == 'random':
        return candidates[int(rng.integers(len(candidates)))].copy()
    if policy == 'fixed':
        return candidates[step % len(candidates)].copy()
    if policy != 'active':
        raise ValueError(f'Unknown policy: {policy}')
    if rng.random() < config['exploration_probability']:
        return candidates[int(rng.integers(len(candidates)))].copy()
    x = np.stack([model_input(observation, a) for a in candidates])
    predictions = memory.predict_members(x)
    disagreement = predictions.var(axis=0).mean(axis=1)
    # The leverage term covers cases where the ensemble agrees but local evidence
    # is scarce. It is an uncertainty heuristic, not an exact IG estimator.
    leverage = memory.parameter_uncertainty(x)
    scores = disagreement + 0.01 * leverage
    scores -= config['cost_weight'] * np.abs(candidates[:, 0]) * candidates[:, 1]
    best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=1e-8))
    return candidates[int(rng.choice(best))].copy()
