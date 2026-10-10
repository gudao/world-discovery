import time
import mujoco.viewer
from .env import PushWorld, WorldSpec


def view():
    """Desktop demonstration. The experiments themselves require no renderer."""
    env = PushWorld(WorldSpec(mass=1.0, friction=0.3))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start = time.monotonic()
        while viewer.is_running():
            elapsed = time.monotonic() - start
            phase = int(elapsed) % 6
            force = 6 if phase == 1 else -6 if phase == 3 else 0
            env.step([force, 0.02])
            viewer.sync()
            time.sleep(0.02)
