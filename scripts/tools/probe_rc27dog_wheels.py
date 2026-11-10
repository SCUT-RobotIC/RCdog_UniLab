"""Probe 27dog wheel actuator authority on the flat scene.

Plain MuJoCo, no RL stack: load the home keyframe, hold the 12 leg position
targets at the home pose, command the 4 wheels to a constant velocity target,
and report how fast the base actually travels.

The A/B on wheel joint damping separates "the asset cannot propel" from
"the policy does not use the wheels":

    damping = 0  -> the current asset
    damping = 2  -> the old <default class="robot"> value that the 4 wheel
                    joints never overrode

Usage (repo root, Linux server):

    uv run python scripts/tools/probe_rc27dog_wheels.py
"""

from __future__ import annotations

import numpy as np
import mujoco

MODEL = "src/unilab/assets/robots/rc27dog/scene.xml"
WHEELS = ("FR_feet_joint", "FL_feet_joint", "RR_feet_joint", "RL_feet_joint")
LEG_HOME = np.array([0.0, 0.7, -1.5] * 4)
SIM_DT = 0.005
SETTLE_SECONDS = 0.5
DRIVE_SECONDS = 3.0


def wheel_dofs(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in WHEELS
    ]


def build(damping: float) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(MODEL)
    model.opt.timestep = SIM_DT
    for dof in wheel_dofs(model):
        model.dof_damping[dof] = damping
    return model


def drive(model: mujoco.MjModel, target: float) -> tuple[float, list[float], float]:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:12] = LEG_HOME
    data.ctrl[12:] = 0.0
    for _ in range(int(round(SETTLE_SECONDS / SIM_DT))):
        mujoco.mj_step(model, data)

    x0 = float(data.qpos[0])
    for _ in range(int(round(DRIVE_SECONDS / SIM_DT))):
        data.ctrl[12:] = target
        mujoco.mj_step(model, data)

    speed = (float(data.qpos[0]) - x0) / DRIVE_SECONDS
    wheel = [float(data.qvel[dof]) for dof in wheel_dofs(model)]
    return speed, wheel, float(data.qpos[2])


def main() -> None:
    loaded = mujoco.MjModel.from_xml_path(MODEL)
    print("wheel dof damping in the loaded MJCF:", [round(float(loaded.dof_damping[d]), 4) for d in wheel_dofs(loaded)])
    print(f"{'case':<34}{'base vx [m/s]':>15}{'wheel w [rad/s]':>17}{'base z [m]':>12}")
    for label, damping, target in (
        ("current asset, target 5 rad/s", 0.0, 5.0),
        ("current asset, target 12 rad/s", 0.0, 12.0),
        ("old damping=2, target 5 rad/s", 2.0, 5.0),
        ("old damping=2, target 12 rad/s", 2.0, 12.0),
    ):
        speed, wheel, height = drive(build(damping), target)
        print(f"{label:<34}{speed:>15.3f}{float(np.mean(wheel)):>17.2f}{height:>12.3f}")


if __name__ == "__main__":
    main()
