import json
from pathlib import Path
import numpy as np
from world_discovery.model import NeuralEnsemble, LocalMemory
from world_discovery.experiment import collect, train, evaluate


def config():
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / 'configs/quick.json').read_text())
    c.update(train_worlds=4, validation_worlds=2, test_worlds=2,
             steps_per_world=10, ensemble=2, hidden=12, epochs=8,
             probe_budget=8, budgets=[0, 2, 8])
    return c


def test_model_save_load_and_local_update(tmp_path):
    c = config()
    rng = np.random.default_rng(10)
    x = rng.normal(size=(60, 3))
    y = x[:, :2] * 0.2
    m = NeuralEnsemble()
    m.fit(x, y, c)
    m.save(tmp_path / 'model.npz')
    restored = NeuralEnsemble.load(tmp_path / 'model.npz')
    np.testing.assert_allclose(m.predict_members(x), restored.predict_members(x))
    memory = LocalMemory(m, c)
    target = memory.predict(x[:1])[0] + np.array([0.2, -0.1])
    before = np.linalg.norm(memory.predict(x[:1])[0] - target)
    memory.update(x[0], target, threshold=100)
    assert np.linalg.norm(memory.predict(x[:1])[0] - target) < before


def test_growth_gate_can_accept_useful_capacity():
    c = config()
    c.update(probe_budget=30, growth_min_samples=20, growth_holdout=5,
             growth_min_gain=0.0, growth_penalty=0.0)
    m = NeuralEnsemble()
    m.x_mean, m.x_scale = np.zeros(3), np.ones(3)
    m.y_mean, m.y_scale = np.zeros(2), np.ones(2)
    m.members = [[np.zeros((3, 2)), np.zeros(2), np.zeros((2, 2)), np.zeros(2)]]
    memory = LocalMemory(m, c, 'grow')
    rng = np.random.default_rng(81)
    for _ in range(30):
        x = rng.normal(size=3)
        # Controlled nonlinear target contained in wide but not small bank.
        value = float(np.tanh(x @ memory.bank[:, 10] + memory.bias[10]))
        memory.update(x, np.array([value, -value]), threshold=-1)
    assert memory.events
    assert any(e['accepted'] for e in memory.events)
    assert memory.width == c['wide_features']


def test_end_to_end_split_isolation_and_no_evaluation_leak(tmp_path):
    c = config()
    collect(c, tmp_path, 'B')
    train(c, tmp_path)
    evaluate(c, tmp_path, 'B')
    truth = json.loads((tmp_path / 'evaluator_only_worlds.json').read_text())
    assert all(s['friction'] == 0.25 for s in truth['train'])
    assert any(s['friction'] != 0.25 for s in truth['test'])
    with np.load(tmp_path / 'train.npz') as data:
        assert not {'mass', 'friction', 'fixed'} & set(data.files)
        assert data['x'].shape[1] == 3
    logs = [json.loads(line) for line in (tmp_path / 'interactions.jsonl').read_text().splitlines()]
    assert len(logs) == c['test_worlds'] * 8 * c['probe_budget']
    assert all(not {'mass', 'friction', 'fixed'} & set(row) for row in logs)
    assert (tmp_path / 'report.html').exists()
    summary = json.loads((tmp_path / 'summary.json').read_text())
    zero = [s['mean_nmse'] for s in summary['curves'] if s['budget'] == 0]
    np.testing.assert_allclose(zero, zero[0], atol=1e-12)
