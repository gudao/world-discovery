"""MuJoCo planar puck with genuine floor contact; no analytic friction substitute."""
from dataclasses import dataclass, asdict
import numpy as np
import mujoco


@dataclass(frozen=True)
class WorldSpec:
    mass: float
    friction: float
    fixed: bool = False


class PushWorld:
    """Ideal x-position/velocity sensors. Model/spec are never passed to learners.

    Two slide joints prevent rotation and lateral motion. A fixed world omits
    the x joint, while retaining vertical contact. This is a restricted scene,
    not a full free-body manipulation benchmark.
    """
    def __init__(self, spec: WorldSpec, timestep=0.002):
        self._spec = spec
        x_joint = '' if spec.fixed else '<joint name="x" type="slide" axis="1 0 0"/>'
        xml = f'''<mujoco model="discovery_puck">
          <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
          <default><geom condim="3" solref="0.01 1"/></default>
          <worldbody>
            <light pos="0 0 3"/>
            <geom name="floor" type="plane" size="0 0 0.1"
                  friction="{spec.friction} 0.005 0.0001" rgba="0.8 0.85 0.9 1"/>
            <body name="puck" pos="0 0 0.055">
              {x_joint}<joint name="z" type="slide" axis="0 0 1"/>
              <geom name="puck_geom" type="box" size="0.05 0.05 0.05"
                    mass="{spec.mass}" friction="{spec.friction} 0.005 0.0001"
                    rgba="0.1 0.45 0.95 1"/>
            </body>
            <camera name="overview" pos="0 -1.4 1" xyaxes="1 0 0 0 0.58 0.81"/>
          </worldbody></mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'puck')
        self.x_dof = None if spec.fixed else int(self.model.jnt_dofadr[0])
        mujoco.mj_step(self.model, self.data, nstep=250)
        mujoco.mj_forward(self.model, self.data)
        self._initial = self.snapshot()

    def observe(self):
        v = 0.0 if self.x_dof is None else float(self.data.qvel[self.x_dof])
        return np.array([self.data.xpos[self.body_id, 0], v], dtype=np.float64)

    def step(self, action):
        force, duration = map(float, action)
        if not np.isfinite([force, duration]).all() or duration <= 0:
            raise ValueError('Action must contain a finite force and positive duration')
        count = round(duration / self.model.opt.timestep)
        if count < 1 or not np.isclose(count * self.model.opt.timestep, duration):
            raise ValueError('duration must be a positive integer multiple of timestep')
        self.data.xfrc_applied[:] = 0
        self.data.xfrc_applied[self.body_id, 0] = force
        mujoco.mj_step(self.model, self.data, nstep=count)
        self.data.xfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)
        obs = self.observe()
        if not np.isfinite(obs).all():
            raise FloatingPointError('Non-finite MuJoCo observation')
        return obs

    def snapshot(self):
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, kind))
        mujoco.mj_getState(self.model, self.data, state, kind)
        return state.copy()

    def restore(self, state):
        mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        self.restore(self._initial)
        return self.observe()

    def evaluator_truth(self):
        """For isolated auditing only. Never supplied to model or explorer."""
        return asdict(self._spec)


def model_input(obs, action):
    # Translation invariant; excludes hidden mass, friction and fixed flag.
    return np.array([obs[1], action[0], action[1]], dtype=np.float64)


def actions(config):
    return np.array([[f, config['duration']] for f in config['forces']], dtype=float)


def sample_spec(rng, config, split, experiment):
    masses = config['test_mass'] if split == 'test' else config['train_mass']
    frictions = config['test_friction'] if split == 'test' else config['train_friction']
    # Experiment B: pretraining only encounters one friction coefficient.
    friction = 0.25 if experiment == 'B' and split != 'test' else rng.uniform(*frictions)
    return WorldSpec(float(rng.uniform(*masses)), float(friction),
                     bool(rng.random() < config['fixed_probability']))
