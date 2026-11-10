"""Probe 27dog wheel authority and standing stance on the flat scene.

Plain MuJoCo, no RL stack. Two sections:

1. stance: hold the keyframe ``home`` targets and let the robot settle. Reports
   the height it actually rests at, the pitch, the knee tracking error and which
   geoms touch the floor.
2. drive: same stance, four wheels commanded to a constant velocity target.

Run from the repository root:

    uv run python scripts/tools/probe_rc27dog_wheels.py
"""

from __future__ import annotations

import numpy as np
import mujoco

MODEL = "src/unilab/assets/robots/rc27dog/scene.xml"
LEGS = ("FR", "FL", "RR", "RL")
WHEELS = tuple(f"{leg}_feet_joint" for leg in LEGS)
CALVES = tuple(f"{leg}_calf_joint" for leg in LEGS)
LEG_HOME = np.array([0.0, 0.7, -1.5] * 4)
SIM_DT = 0.005
SETTLE_SECONDS = 1.0
DRIVE_SECONDS = 3.0


def wheel_dofs(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in WHEELS
    ]


def joint_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[jid]])


def wheel_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(np.mean([data.actuator_force[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] for name in WHEELS]))


def knee_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return min(
        float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf_link")][2])
        for leg in LEGS
    )


def pitch_deg(data: mujoco.MjData) -> float:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(data.qpos[3:7], dtype=np.float64))
    mat = mat.reshape(3, 3)
    return float(np.degrees(np.arctan2(mat[0, 2], mat[2, 2])))


def floor_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    names: list[str] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geoms = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
        }
        if "floor" in geoms:
            other = (geoms - {"floor"}).pop()
            names.append(other or "unnamed")
    return names


def build(damping: float) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(MODEL)
    model.opt.timestep = SIM_DT
    for dof in wheel_dofs(model):
        model.dof_damping[dof] = damping
    return model


def settle(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:12] = LEG_HOME
    data.ctrl[12:] = 0.0
    for _ in range(int(round(SETTLE_SECONDS / SIM_DT))):
        mujoco.mj_step(model, data)
    return data


def report_stance(model: mujoco.MjModel) -> None:
    data = settle(model)
    print("\n[stance] home targets held for ", SETTLE_SECONDS, " s")
    print(f"  base z          : {float(data.qpos[2]):.3f} m   (keyframe 0.420)")
    print(f"  lowest knee z   : {knee_z(model, data):.3f} m   (calf box half-height 0.043)")
    print(f"  pitch           : {pitch_deg(data):+.1f} deg  (+ = nose down)")
    for name in CALVES:
        target = float(data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)])
        print(f"  {name:<14}: target {target:+.2f}  actual {joint_pos(model, data, name):+.2f}"
              f"  error {joint_pos(model, data, name) - target:+.2f} rad")
    print(f"  floor contacts  : {sorted(floor_contacts(model, data))}")


def drive(model: mujoco.MjModel, target: float) -> None:
    data = settle(model)
    x0 = float(data.qpos[0])
    for _ in range(int(round(DRIVE_SECONDS / SIM_DT))):
        data.ctrl[12:] = target
        mujoco.mj_step(model, data)
    speed = (float(data.qpos[0]) - x0) / DRIVE_SECONDS
    wheel = float(np.mean([data.qvel[dof] for dof in wheel_dofs(model)]))
    print(f"  {target:>5.1f} rad/s -> base {speed:+.3f} m/s  wheel {wheel:5.2f} rad/s"
          f"  torque {wheel_force(model, data):+5.2f} Nm  z {float(data.qpos[2]):.3f} m"
          f"  knee_z {knee_z(model, data):.3f} m  pitch {pitch_deg(data):+.1f} deg")
    print(f"          floor contacts: {sorted(floor_contacts(model, data))}")


def main() -> None:
    loaded = mujoco.MjModel.from_xml_path(MODEL)
    print("wheel dof damping in the loaded MJCF:",
          [round(float(loaded.dof_damping[d]), 4) for d in wheel_dofs(loaded)])
    report_stance(build(0.0))
    for damping, label in ((0.0, "current asset"), (2.0, "old damping=2")):
        model = build(damping)
        print(f"\n[drive] {label}")
        for target in (5.0, 12.0):
            drive(model, target)


if __name__ == "__main__":
    main()
