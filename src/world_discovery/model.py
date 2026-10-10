"""Small NumPy neural ensemble plus bounded per-world residual memory.

Global tanh MLPs are trained by Adam. Local adapters use ridge regression.
Growth enables extra fixed random features: capacity growth, not semantic
concept discovery. Shared feature banks make capacity comparisons auditable.
"""
import numpy as np


class NeuralEnsemble:
    def fit(self, x, y, config):
        self.x_mean, self.x_scale = x.mean(0), np.maximum(x.std(0), 0.1)
        self.y_mean, self.y_scale = y.mean(0), np.maximum(y.std(0), 0.01)
        xn, yn = self.normalize_x(x), self.normalize_y(y)
        self.members = []
        history = []
        for k in range(config['ensemble']):
            rng = np.random.default_rng(config['seed'] + 500 + k)
            ids = rng.integers(len(x), size=len(x))
            xb, yb = xn[ids], yn[ids]
            width = config['hidden']
            p = [rng.normal(0, 0.4, (3, width)), np.zeros(width),
                 rng.normal(0, 0.15, (width, 2)), np.zeros(2)]
            m, v = [np.zeros_like(a) for a in p], [np.zeros_like(a) for a in p]
            iteration = 0
            for epoch in range(config['epochs']):
                order = rng.permutation(len(x))
                for start in range(0, len(x), config['batch_size']):
                    ix = order[start:start + config['batch_size']]
                    a, target = xb[ix], yb[ix]
                    hidden = np.tanh(a @ p[0] + p[1])
                    pred = hidden @ p[2] + p[3]
                    dp = (pred - target) / len(ix)  # mean over two output dimensions
                    dh = (dp @ p[2].T) * (1 - hidden ** 2)
                    gradients = [a.T @ dh, dh.sum(0), hidden.T @ dp, dp.sum(0)]
                    iteration += 1
                    for j, g in enumerate(gradients):
                        g = np.clip(g, -10, 10)
                        m[j] = 0.9 * m[j] + 0.1 * g
                        v[j] = 0.999 * v[j] + 0.001 * g * g
                        p[j] -= config['learning_rate'] * (m[j] / (1 - 0.9 ** iteration)) / (
                            np.sqrt(v[j] / (1 - 0.999 ** iteration)) + 1e-8)
                if epoch == config['epochs'] - 1:
                    history.append(float(np.mean((np.tanh(xn @ p[0] + p[1]) @ p[2] + p[3] - yn) ** 2)))
            self.members.append(p)
        return history

    def normalize_x(self, x):
        return (np.asarray(x) - self.x_mean) / self.x_scale

    def normalize_y(self, y):
        return (np.asarray(y) - self.y_mean) / self.y_scale

    def predict_members(self, x):
        xn = self.normalize_x(np.atleast_2d(x))
        return np.stack([np.tanh(xn @ p[0] + p[1]) @ p[2] + p[3] for p in self.members])

    def save(self, path):
        values = {n: getattr(self, n) for n in ['x_mean', 'x_scale', 'y_mean', 'y_scale']}
        values['ensemble_size'] = np.array(len(self.members))
        for k, p in enumerate(self.members):
            for j, a in enumerate(p):
                values[f'member_{k}_{j}'] = a
        np.savez_compressed(path, **values)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            for n in ['x_mean', 'x_scale', 'y_mean', 'y_scale']:
                setattr(obj, n, data[n].copy())
            obj.members = [[data[f'member_{k}_{j}'].copy() for j in range(4)]
                           for k in range(int(data['ensemble_size']))]
        return obj


class LocalMemory:
    def __init__(self, model, config, mode='small'):
        self.model, self.config, self.mode = model, config, mode
        self.width = config['wide_features'] if mode == 'wide' else config['base_features']
        rng = np.random.default_rng(config['seed'] + 900)
        self.bank = rng.normal(0, 0.8, (3, config['wide_features']))
        self.bias = rng.normal(0, 0.5, config['wide_features'])
        self.x, self.y, self.errors, self.events = [], [], [], []
        self.weights = self._fit([], [], self.width)
        self.trigger_streak = 0

    def features(self, x, width=None):
        width = self.width if width is None else width
        xn = self.model.normalize_x(np.atleast_2d(x))
        return np.concatenate([np.ones((len(xn), 1)), xn,
                               np.tanh(xn @ self.bank[:, :width] + self.bias[:width])], axis=1)

    def _fit(self, x, y, width):
        dim = 4 + width
        if len(x) == 0:
            return np.zeros((len(self.model.members), dim, 2))
        phi = self.features(np.array(x), width)
        prior = self.model.predict_members(np.array(x))
        target = self.model.normalize_y(np.array(y))
        gram = phi.T @ phi + self.config['ridge'] * np.eye(dim)
        return np.stack([np.linalg.solve(gram, phi.T @ (target - p)) for p in prior])

    def predict_members(self, x):
        prior = self.model.predict_members(x)
        if self.mode == 'none':
            return prior
        return prior + np.einsum('nd,kdo->kno', self.features(x), self.weights)

    def predict(self, x):
        normalized = self.predict_members(x).mean(0)
        return normalized * self.model.y_scale + self.model.y_mean

    def parameter_uncertainty(self, x):
        """Ridge leverage heuristic; not a calibrated Bayesian information gain."""
        phi = self.features(x)
        dim = phi.shape[1]
        if self.x:
            past = self.features(self.x)
            gram = past.T @ past + self.config['ridge'] * np.eye(dim)
        else:
            gram = self.config['ridge'] * np.eye(dim)
        solved = np.linalg.solve(gram, phi.T).T
        return np.einsum('nd,nd->n', phi, solved)

    def update(self, x, y, threshold):
        error = float(np.mean(((self.predict(x)[0] - y) / self.model.y_scale) ** 2))
        self.errors.append(error)  # prequential: prediction made before this observation
        self.x.append(np.array(x).copy())
        self.y.append(np.array(y).copy())
        # Memory is bounded by probe_budget: one adapter is discarded per world.
        if len(self.x) > self.config['probe_budget']:
            self.x.pop(0)
            self.y.pop(0)
        self.trigger_streak = self.trigger_streak + 1 if error > threshold else 0
        if (self.mode == 'grow' and self.width < self.config['wide_features']
                and len(self.x) >= self.config['growth_min_samples'] and self.trigger_streak >= 2):
            nval = self.config['growth_holdout']
            tx, ty = self.x[:-nval], self.y[:-nval]
            vx, vy = np.array(self.x[-nval:]), np.array(self.y[-nval:])
            target, prior = self.model.normalize_y(vy), self.model.predict_members(vx)
            losses = []
            for width in [self.width, self.config['wide_features']]:
                weights = self._fit(tx, ty, width)
                pred = prior + np.einsum('nd,kdo->kno', self.features(vx, width), weights)
                losses.append(float(np.mean((pred.mean(0) - target) ** 2)))
            gain = losses[0] - losses[1]
            extra_parameters = (self.config['wide_features'] - self.width) * 2 * len(self.model.members)
            gate = self.config['growth_min_gain'] + self.config['growth_penalty'] * extra_parameters
            accepted = gain > gate
            self.events.append({'step': len(self.errors), 'small_loss': losses[0],
                                'wide_loss': losses[1], 'gain': gain, 'gate': gate,
                                'accepted': accepted})
            if accepted:
                self.width = self.config['wide_features']
        if self.mode != 'none':
            self.weights = self._fit(self.x, self.y, self.width)
