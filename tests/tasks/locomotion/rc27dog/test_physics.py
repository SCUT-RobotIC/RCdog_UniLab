"""Compiled RC27Dog physics must preserve the original URDF, including cross inertia."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

# Golden link-frame properties from robot_lab 82595e4:
# data/Robots/RC/27dog urdf v1/27dog urdf v1/urdf/27dog urdf v1.urdf.
# Values are independent of the converted XML. Inertia order: xx yy zz xy xz yz.
_URDF = {
    "base_link": (
        8.54711286,
        (-0.0114530629, 0.000184145189, 0.0108079466),
        (0.0454076359, 0.133474942, 0.150237356, 7.18465329e-05, 0.00326273311, 7.56194551e-05),
    ),
    "FL_hip_link": (
        0.979024572,
        (0.0668567141, 0.00709367375, 5.20014991e-05),
        (
            0.000181948054,
            0.000468245871,
            0.000425764599,
            1.60347569e-05,
            -1.81798032e-07,
            1.83019039e-07,
        ),
    ),
    "FL_thigh_link": (
        1.67712744,
        (-0.00107568226, -0.0323365411, -0.0368182499),
        (0.0126972557, 0.0120288322, 0.0019987605, 5.90095887e-05, -0.000118261261, 0.00199390507),
    ),
    "FL_calf_link": (
        0.28687129,
        (0.00337151812, -0.0308641979, -0.18911897),
        (
            0.00217614791,
            0.00225723253,
            0.000127158305,
            1.86644429e-06,
            -1.29944889e-05,
            1.26899194e-05,
        ),
    ),
    "FL_feet_link": (
        0.75408584,
        (-7.95024319e-05, 0.00690478198, -0.000145043428),
        (
            0.00138124617,
            0.00223987548,
            0.00138287505,
            2.71090251e-07,
            1.94578639e-06,
            -5.59476119e-07,
        ),
    ),
    "FR_hip_link": (
        0.979024572,
        (0.0668567141, -0.00709367375, -5.20014992e-05),
        (
            0.000181948054,
            0.000468245871,
            0.000425764599,
            -1.6034757e-05,
            1.81798025e-07,
            1.83019043e-07,
        ),
    ),
    "FR_thigh_link": (
        1.68668998,
        (-0.00123683343, 0.0322736059, -0.036834739),
        (
            0.0127221716,
            0.0120771885,
            0.00200108397,
            -5.80070202e-05,
            -0.000131592564,
            -0.00199386636,
        ),
    ),
    "FR_calf_link": (
        0.283703379,
        (0.00323406393, 0.0309203865, -0.189454782),
        (
            0.00217341929,
            0.00225388133,
            0.000126403611,
            -1.91830554e-06,
            -1.21410206e-05,
            -1.26485063e-05,
        ),
    ),
    "FR_feet_link": (
        0.75408584,
        (-2.03306144e-05, -0.00690478198, -0.000164149013),
        (
            0.00137995716,
            0.00223987548,
            0.00138416406,
            5.47083593e-07,
            1.57797557e-07,
            2.95301531e-07,
        ),
    ),
    "RL_hip_link": (
        0.979024572,
        (-0.0668567141, 0.00709367375, -5.20014991e-05),
        (
            0.000181948054,
            0.000468245871,
            0.000425764599,
            -1.60347569e-05,
            -1.81798032e-07,
            -1.83019039e-07,
        ),
    ),
    "RL_thigh_link": (
        1.67712744,
        (-0.00107568226, -0.0323365411, -0.0368182499),
        (0.0126972557, 0.0120288322, 0.0019987605, 5.90095887e-05, -0.000118261261, 0.00199390507),
    ),
    "RL_calf_link": (
        0.28687129,
        (0.00337151812, -0.0308641979, -0.18911897),
        (
            0.00217614791,
            0.00225723253,
            0.000127158305,
            1.86644429e-06,
            -1.29944889e-05,
            1.26899194e-05,
        ),
    ),
    "RL_feet_link": (
        0.75408584,
        (-7.95024317e-05, 0.00690478198, -0.000145043428),
        (
            0.00138124617,
            0.00223987548,
            0.00138287505,
            2.71090251e-07,
            1.94578639e-06,
            -5.59476119e-07,
        ),
    ),
    "RR_hip_link": (
        0.979024572,
        (-0.0668567141, -0.00709367375, 5.20014992e-05),
        (
            0.000181948054,
            0.000468245871,
            0.000425764599,
            1.6034757e-05,
            1.81798025e-07,
            -1.83019043e-07,
        ),
    ),
    "RR_thigh_link": (
        1.68668998,
        (-0.00123683343, 0.0322736059, -0.036834739),
        (
            0.0127221716,
            0.0120771885,
            0.00200108397,
            -5.80070202e-05,
            -0.000131592564,
            -0.00199386636,
        ),
    ),
    "RR_calf_link": (
        0.283703379,
        (0.00323406393, 0.0309203865, -0.189454782),
        (
            0.00217341929,
            0.00225388133,
            0.000126403611,
            -1.91830554e-06,
            -1.21410206e-05,
            -1.26485063e-05,
        ),
    ),
    "RR_feet_link": (
        0.75408584,
        (-2.03306143e-05, -0.00690478198, -0.000164149013),
        (
            0.00137995716,
            0.00223987548,
            0.00138416406,
            5.47083593e-07,
            1.57797557e-07,
            2.95301531e-07,
        ),
    ),
}


def _model() -> mujoco.MjModel:
    path = Path(__file__).parents[4] / "src/unilab/assets/robots/rc27dog/rc27dog.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def test_compiled_mass_com_and_full_inertia_match_urdf() -> None:
    model = _model()
    assert model.body_mass.sum() == pytest.approx(23.348338686, abs=1e-10)
    for name, (mass, com, inertia) in _URDF.items():
        body = model.body(name)
        assert float(body.mass[0]) == pytest.approx(mass, abs=1e-10)
        np.testing.assert_allclose(body.ipos, com, atol=1e-10, rtol=0)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        rotation = rotation.reshape(3, 3)
        compiled = rotation @ np.diag(body.inertia) @ rotation.T
        xx, yy, zz, xy, xz, yz = inertia
        expected = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        # MuJoCo diagonalizes fullinertia with a finite eigenvector tolerance.
        np.testing.assert_allclose(compiled, expected, atol=1e-9, rtol=1e-6, err_msg=name)


def test_joint_and_actuator_physics_match_wheeled_recipe() -> None:
    model = _model()
    for leg in ("FR", "FL", "RR", "RL"):
        for kind in ("hip", "thigh", "calf", "feet"):
            name = f"{leg}_{kind}_joint"
            joint = model.joint(name)
            dof = int(joint.dofadr[0])
            wheel = kind == "feet"
            assert model.dof_damping[dof] == 0, "actuator kd must not be counted twice"
            assert model.dof_armature[dof] == pytest.approx(0.001 if wheel else 0.02)
            assert model.dof_frictionloss[dof] == pytest.approx(0 if wheel else 0.05)
            assert bool(joint.limited[0]) is not wheel
            actuator = model.actuator(name)
            cap = 3.69 if wheel else 20.0 if kind == "hip" else 30.0
            np.testing.assert_allclose(actuator.forcerange, [-cap, cap])
            assert actuator.gainprm[0] == pytest.approx(0.5 if wheel else 25.0)
            assert actuator.biasprm[2] == pytest.approx(-0.5 if wheel else -2.0)
            assert actuator.biasprm[1] == pytest.approx(0 if wheel else -25.0)
            if wheel:
                limit = 587.0 * 2.0 * np.pi / 60.0
                np.testing.assert_allclose(actuator.ctrlrange, [-limit, limit], rtol=1e-12)


def test_compiled_servos_apply_expected_velocity_damping_and_effort_caps() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for name in ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_feet_joint"):
        joint = model.joint(name)
        actuator = model.actuator(name)
        dof = int(joint.dofadr[0])
        data.qvel[dof] = 1.0
        mujoco.mj_forward(model, data)
        damping = 0.5 if name.endswith("feet_joint") else 2.0
        assert data.actuator_force[actuator.id] == pytest.approx(-damping, abs=1e-10)
        data.ctrl[actuator.id] = 1e6
        mujoco.mj_forward(model, data)
        assert data.actuator_force[actuator.id] == pytest.approx(actuator.forcerange[1])
        data.qvel[dof] = 0
        data.ctrl[:] = model.key_ctrl[model.key("home").id]
