#!/usr/bin/env python3
"""Finish the 27dog wheel-legged MJCF asset.

The URDF carries no ``<transmission>``, so ``import_robot.py`` produces a model
without actuators, and even when it derives them it maps every ``<motor>`` to a
``<position>`` servo. A wheel joint driven by a position servo cannot roll, so
this script applies the parameters already tuned on the IsaacLab side
(``robot_lab/assets/rc.py`` and ``rc_wheeled.py``):

  legs    position servo  kp=25 kv=2  effort 20 (hip) / 30 (thigh, calf)
  wheels  velocity servo  kv=0.5      effort 3.69, ctrl +-61.47 rad/s

It also adds ``armature`` / ``frictionloss`` to the joints, writes the ``home``
keyframe into ``scene.xml`` and compiles the result as a final check.

Run it after ``import_robot.py``; it is idempotent.

Usage:
  uv run scripts/tools/finalize_rc27dog_asset.py [--height 0.42] [--robot rc27dog]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ROOT = REPO_ROOT / "src" / "unilab" / "assets" / "robots"

# 27dog leg layout: four legs, each hip / thigh / calf revolute and one wheel.
LEG_PREFIXES = ("FR", "FL", "RR", "RL")

# IsaacLab actuator parameters (robot_lab/assets/rc.py, rc_wheeled.py).
HIP_JOINT = "hip"
THIGH_JOINT = "thigh"
CALF_JOINT = "calf"
WHEEL_JOINT = "feet"

LEG_GAINS = {
    HIP_JOINT: {"kp": 25.0, "kv": 2.0, "effort": 20.0},
    THIGH_JOINT: {"kp": 25.0, "kv": 2.0, "effort": 30.0},
    CALF_JOINT: {"kp": 25.0, "kv": 2.0, "effort": 30.0},
}
WHEEL_GAINS = {"kv": 0.5, "effort": 3.69, "ctrl": 61.470496255240}

LEG_ARMATURE = 0.02
LEG_FRICTIONLOSS = 0.05
WHEEL_ARMATURE = 0.001

# Standing pose: hip 0, thigh 0.7, calf -1.5, wheel 0 (robot_lab init_state).
STANDING_POSE = {HIP_JOINT: 0.0, THIGH_JOINT: 0.7, CALF_JOINT: -1.5, WHEEL_JOINT: 0.0}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="rc27dog", help="Robot asset directory name.")
    parser.add_argument(
        "--height",
        type=float,
        default=0.42,
        help="Free-joint z of the home keyframe in metres (RCWHEEL_STANDING_HEIGHT).",
    )
    return parser.parse_args(argv)


def _joint_kind(name: str) -> str | None:
    for kind in (WHEEL_JOINT, HIP_JOINT, THIGH_JOINT, CALF_JOINT):
        if name.endswith(f"_{kind}_joint"):
            return kind
    return None


def _is_leg_joint(name: str) -> bool:
    return _joint_kind(name) in LEG_GAINS


def _is_wheel_joint(name: str) -> bool:
    return _joint_kind(name) == WHEEL_JOINT


def _expected_joint_names(prefixes: Sequence[str]) -> list[str]:
    names: list[str] = []
    for prefix in prefixes:
        names.extend(
            f"{prefix}_{kind}_joint"
            for kind in (HIP_JOINT, THIGH_JOINT, CALF_JOINT)
        )
    for prefix in prefixes:
        names.append(f"{prefix}_{WHEEL_JOINT}_joint")
    return names


def _apply_joint_params(root: ET.Element) -> list[str]:
    """Set armature / frictionloss on every joint; return the joint names seen."""
    seen: list[str] = []
    for joint in root.iter("joint"):
        name = joint.get("name")
        if not name:
            continue
        seen.append(name)
        if _is_wheel_joint(name):
            joint.set("limited", "false")
            joint.set("armature", f"{WHEEL_ARMATURE:g}")
            joint.set("frictionloss", "0")
        elif _is_leg_joint(name):
            joint.set("armature", f"{LEG_ARMATURE:g}")
            joint.set("frictionloss", f"{LEG_FRICTIONLOSS:g}")
    return seen


def _replace_actuators(root: ET.Element, joint_names: Sequence[str]) -> None:
    """Replace any converted actuator block with the wheel-aware contract."""
    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)

    actuator = ET.SubElement(root, "actuator")

    def _add_position(name: str, kind: str) -> None:
        gains = LEG_GAINS[kind]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": name,
                "kp": f"{gains['kp']:g}",
                "kv": f"{gains['kv']:g}",
                "forcerange": f"-{gains['effort']:g} {gains['effort']:g}",
            },
        )

    for prefix in LEG_PREFIXES:
        for kind in (HIP_JOINT, THIGH_JOINT, CALF_JOINT):
            _add_position(f"{prefix}_{kind}_joint", kind)
    for prefix in LEG_PREFIXES:
        name = f"{prefix}_{WHEEL_JOINT}_joint"
        ET.SubElement(
            actuator,
            "velocity",
            {
                "name": name,
                "joint": name,
                "kv": f"{WHEEL_GAINS['kv']:g}",
                "ctrlrange": (
                    f"-{WHEEL_GAINS['ctrl']:.6f} {WHEEL_GAINS['ctrl']:.6f}"
                ),
                "forcerange": f"-{WHEEL_GAINS['effort']:g} {WHEEL_GAINS['effort']:g}",
            },
        )


def _write_robot_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)


def _scene_includes_robot(root: ET.Element, robot_name: str) -> bool:
    """Whether a scene compiles as a standalone model (i.e. pulls in the robot)."""
    expected = f"{robot_name}.xml"
    return any(
        (include.get("file") or "").strip() == expected for include in root.iter("include")
    )


def _build_scene(robot_name: str) -> ET.Element:
    """Minimal flat-ground scene used when the importer did not write one."""
    root = ET.Element("mujoco", {"model": f"{robot_name} scene"})
    ET.SubElement(root, "include", {"file": f"{robot_name}.xml"})
    ET.SubElement(root, "statistic", {"center": "0 0 0.3", "extent": "0.9"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.1 0.1 0.1", "specular": "0.9 0.9 0.9"},
    )
    ET.SubElement(visual, "map", {"fogstart": "100.0", "fogend": "300.0"})
    ET.SubElement(visual, "rgba", {"haze": "0.75 0.75 0.77 1"})
    ET.SubElement(visual, "global", {"azimuth": "-130", "elevation": "-20"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.55 0.55 0.57",
            "rgb2": "0.55 0.55 0.57",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "rgb1": "0.62 0.62 0.63",
            "rgb2": "0.49 0.49 0.51",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "1.25 1.25",
            "reflectance": "0.0",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true",
                             "diffuse": "0.8 0.8 0.8"}
    )
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"},
    )
    return root


def _format(values: Sequence[float]) -> str:
    return " ".join(f"{value:.6f}".rstrip("0").rstrip(".") for value in values)


def _write_keyframe(
    scene_root: ET.Element, model: "object", height: float
) -> None:
    import mujoco

    mujoco_api = mujoco

    for existing in scene_root.findall("keyframe"):
        scene_root.remove(existing)
    keyframe = ET.SubElement(scene_root, "keyframe")

    qpos = [0.0] * model.nq  # type: ignore[attr-defined]
    qpos[0:3] = [0.0, 0.0, height]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for joint_id in range(1, model.njnt):  # type: ignore[attr-defined]
        name = mujoco_api.mj_id2name(
            model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id
        )
        kind = _joint_kind(name or "")
        if kind is None:
            continue
        qpos[int(model.jnt_qposadr[joint_id])] = STANDING_POSE[kind]  # type: ignore[attr-defined]

    ctrl = [0.0] * model.nu  # type: ignore[attr-defined]
    for actuator_id in range(model.nu):  # type: ignore[attr-defined]
        joint_id = int(model.actuator_trnid[actuator_id, 0])  # type: ignore[attr-defined]
        name = mujoco_api.mj_id2name(
            model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id
        )
        kind = _joint_kind(name or "")
        ctrl[actuator_id] = STANDING_POSE[kind] if kind else 0.0

    ET.SubElement(keyframe, "key", {"name": "home", "qpos": _format(qpos), "ctrl": _format(ctrl)})


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    robot_dir = ROBOT_ROOT / args.robot
    robot_xml = robot_dir / f"{args.robot}.xml"
    scene_xml = robot_dir / "scene.xml"

    if not robot_xml.is_file():
        print(f"error: {robot_xml} not found; run import_robot.py first", file=sys.stderr)
        return 1

    try:
        import mujoco
    except Exception as exc:  # pragma: no cover - environment guard
        print(f"error: mujoco is required for this step: {exc}", file=sys.stderr)
        return 1

    tree = ET.parse(robot_xml)
    root = tree.getroot()

    joint_names = _apply_joint_params(root)
    expected = _expected_joint_names(LEG_PREFIXES)
    missing = [name for name in expected if name not in joint_names]
    if missing:
        print(
            "error: converted model is missing expected joints: "
            + ", ".join(missing)
            + f"\n       found: {', '.join(joint_names)}",
            file=sys.stderr,
        )
        return 1

    _replace_actuators(root, joint_names)
    _write_robot_xml(root, robot_xml)

    # Compile the robot alone: mesh paths and the actuator block are checked here.
    robot_model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if robot_model.nu != len(expected):
        print(
            f"error: expected {len(expected)} actuators, model has {robot_model.nu}",
            file=sys.stderr,
        )
        return 1

    # import_robot.py writes a keyframe-only fragment without <include>, so a
    # standalone compile of it has nq=0. Rebuild the scene whenever the robot is
    # not included; keep a scene the caller already composed by hand.
    scene_tree = ET.parse(scene_xml) if scene_xml.is_file() else None
    if scene_tree is not None and _scene_includes_robot(scene_tree.getroot(), args.robot):
        scene_root = scene_tree.getroot()
        rebuilt = False
    else:
        scene_root = _build_scene(args.robot)
        scene_tree = ET.ElementTree(scene_root)
        rebuilt = True

    _write_keyframe(scene_root, robot_model, args.height)
    ET.indent(scene_tree, space="  ")
    scene_tree.write(scene_xml, encoding="utf-8", xml_declaration=False)

    # Final check on the composed scene.
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
    except Exception as exc:
        print(f"error: composed scene does not compile: {exc}", file=sys.stderr)
        print(f"       inspect {scene_xml}", file=sys.stderr)
        return 1
    print(f"robot xml : {robot_xml.relative_to(REPO_ROOT)}")
    print(f"scene xml : {scene_xml.relative_to(REPO_ROOT)}"
          + ("  (rebuilt: include + ground + light + keyframe)" if rebuilt else ""))
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt}")
    print(f"actuators : {model.nu} (12 position + 4 velocity)")
    print(f"home height: {args.height} m")
    if model.nq != 23 or model.nu != 16:
        print(
            f"error: expected nq=23 nu=16, got nq={model.nq} nu={model.nu}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
