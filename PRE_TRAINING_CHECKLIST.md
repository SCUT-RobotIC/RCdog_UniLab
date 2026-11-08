# 27dog 轮足机器人：UniLab 训练前工作清单（v2）

> 仓库：`SCUT-RobotIC/RCdog_UniLab`（UniLab `1.5.9` fork）
> 机器人：`robot_lab_vision/.../RC/27dog urdf v1/urdf/27dog urdf v1.urdf`
> 服务器：Ubuntu 22.04 + RTX 5080 16 GB，驱动 `580.178.04` / CUDA 13.0
> 算法：**FlashSAC + mjwarp**（GPU 物理 + off-policy 异步），MuJoCo 先验证资产

---

## 0. 已确认的事实

| 项 | 结论 |
|---|---|
| 机器人自由度 | **16**：12 个 `revolute` 腿关节 + 4 个 `continuous` 轮关节（`*_feet_joint`，轴 Y） |
| 质量 | 总计 **23.35 kg**（base 8.55 kg） |
| 轮子 | 半径 **0.085 m**，半径圆柱长 0.0773 m；`effort=3.69 N·m`，`velocity=61.47 rad/s` → 极速 **5.22 m/s** |
| 腿 | `effort=53.5 N·m`、`vel=11 rad/s`；hip 限位 ±0.977，thigh `[-1.047, 4.189]`，calf `[-2.801, -0.742]` |
| 碰撞体 | **URDF 里已经是基本体**（box/cylinder），不需要重建，只需微调 |
| 网格 | 共 ~60 MB（base_link 24.9 MB），只做视觉，不参与碰撞 |
| 站高 | `RCWHEEL_STANDING_HEIGHT = 0.42 m`，初始 0.48 m（IsaacLab 侧定义） |
| 驱动门槛 | 580 ≥ CUDA 13.x 下限，`torch 2.14.0+cu130` 可用 ✅ |

IsaacLab 侧已调好的参数（`robot_lab/assets/rc.py`、`rc_wheeled.py`、`config/wheeled/rc/`）：

| 关节组 | effort | 速度限 | 刚度 kp | 阻尼 kd | armature | friction |
|---|---|---|---|---|---|---|
| hip | 20 | 30 | 25 | 2 | 0.02 | 0.05 |
| thigh / calf | 30 | 10 | 25 | 2 | 0.02 | 0.05 |
| wheel | 3.69 | 61.47 | 0 | 0.5 | 0.001 | 0 |

---

## 1. 算法与后端选择

### 决定：FlashSAC + mjwarp，MuJoCo 作为资产调试后端

| 组合 | 加速点 | 仓库证据等级 | 用法 |
|---|---|---|---|
| **FlashSAC + mjwarp** | GPU 常驻物理 + 采集/学习重叠 + compile/CUDA graph | `Configured`（g1 有 owner，无自动化测试） | **主训练** |
| SAC + mjwarp | 同上，去掉 compile/graph | `Tested`（唯一有测试证据的 mjwarp 组合） | FlashSAC 不稳时一行命令降级 |
| FlashSAC + mujoco | CPU 物理 + CUDA learner | `Tested` | **先验证 MJCF / 任务** |
| PPO + 任意 | on-policy，无 replay | Tested | 不选：同墙钟时间样本效率低一个量级 |

不选 PPO 的理由：UniLab 的加速来自 **off-policy 把采集和更新重叠**，以及 **mjwarp 把物理搬到 GPU**。PPO 是纯 on-policy，加速收益只剩后端一项，浪费了框架的主要特性。

16 GB 显存的预算（FlashSAC 默认网络只有 128/256，很省）：

| 项 | `num_envs=4096`, `replay_buffer_n=256` |
|---|---|
| replay 行数 | 4096 × 256 = 1.05 M 行 |
| 每行宽 | ≈ 237 float（2×obs 53 + act 16 + 3 + 2×critic 56） |
| replay 显存 | ≈ **1.0 GB** |
| mjwarp 状态 + 接触缓冲 | < 0.5 GB |
| learner（batch 4096，128/256 MLP） | < 1 GB |
| 余量 | 充足，`num_envs` 可上探 8192 |

### 降级阶梯（出问题按这个顺序退）

1. `--sim mjwarp` 起不来 → `--sim mujoco`（owner 已建好，同一份任务配置）
2. FlashSAC 报错/不收敛 → 复制成 SAC 树：`cp -r src/unilab/conf/flashsac/task/rc27dog_flat src/unilab/conf/sac/task/`
3. 轮子执行器不被 mjwarp 支持 → 见 §3.3 的 motor 兜底

---

## 2. 服务器环境（门槛已过，剩三步）

```bash
# 1) 系统依赖
sudo apt-get update
sudo apt-get install -y build-essential git curl cmake ffmpeg \
  libgl1 libglib2.0-0 libxrender1 libxext6 libsm6

# 2) uv + Python
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.13
uv python pin 3.13

# 3) 同步（mujoco + mjwarp + RL 栈）
cd RCdog_UniLab
git checkout -b feat/rc27dog
uv sync --extra mujoco --extra mjwarp --extra unirl
```

验证（缺一不可）：

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -c "import warp; print('warp', warp.__version__)"
# 内置任务冒烟：验证 mjbatch / mjwarp / warp 在 5080 上都正常，与我们的机器人无关
uv run train --algo flashsac --task g1_walk_flat --sim mjwarp algo.num_envs=64 algo.max_iterations=2 training.no_play=true
```

> 第 3 条是**关键的去风险步骤**：先用仓库自带任务确认 `warp + sm_120 + 驱动 580` 这条链路通，
> 再去查我们自己的 MJCF。不要拿新机器人当第一个小白鼠。

国内网络：`export HF_ENDPOINT=https://hf-mirror.com`、`export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

---

## 3. 资产：URDF → MJCF（唯一剩下的硬工作，约 0.5~1 天）

碰撞体已经是基本体，所以这一步只剩 **执行器 + 关键帧 + 少量关节参数**。

### 3.1 用仓库脚本转换（可无显示器）

```bash
uv run scripts/tools/import_robot.py \
  "/data/27dog urdf v1/27dog urdf v1/urdf/27dog urdf v1.urdf" rc27dog
```

- 脚本会写 `src/unilab/assets/robots/rc27dog/{rc27dog.xml, scene.xml, assets/}`。
- 最后会尝试开 MuJoCo 查看器调 `home`：**无显示器会在这一步抛错，但前面的文件已经写好了**，直接忽略。
  （它抛错前已经 `MjModel.from_xml_path` 编译过一遍，所以这一步也顺带验证了 MJCF 合法性。）
- 有桌面的话就不用忽略，直接在查看器里拖姿态，关窗口自动写回 `scene.xml`。

### 3.2 一次补全：执行器 + 关节参数 + 关键帧

仓库里已经放好 [`scripts/tools/finalize_rc27dog_asset.py`](scripts/tools/finalize_rc27dog_asset.py)：
它按 IsaacLab 那份调好的参数写入 12 个 `<position>` + 4 个 `<velocity>`、给关节补
`armature/frictionloss`、生成 `home` 关键帧，最后编译校验（要求 `nq=23`、`nu=16`）。
幂等，可以反复跑；关节名对不上时它会报 `missing expected joints` 并退出，不会写坏文件。

```bash
uv run scripts/tools/finalize_rc27dog_asset.py --height 0.42
```

3.2 ~ 3.4 余下内容就是脚本写入的具体东西，留作对照或手工微调时参考。

#### 执行器（URDF 没有 `<transmission>`，转换结果里没有执行器）

```xml
<actuator>
  <!-- 腿：位置伺服 kp=25 kv=2（robot_lab/assets/rc.py） -->
  <position name="FR_hip_joint"   joint="FR_hip_joint"   kp="25" kv="2" forcerange="-20 20"/>
  <position name="FR_thigh_joint" joint="FR_thigh_joint" kp="25" kv="2" forcerange="-30 30"/>
  <position name="FR_calf_joint"  joint="FR_calf_joint"  kp="25" kv="2" forcerange="-30 30"/>
  <position name="FL_hip_joint"   joint="FL_hip_joint"   kp="25" kv="2" forcerange="-20 20"/>
  <position name="FL_thigh_joint" joint="FL_thigh_joint" kp="25" kv="2" forcerange="-30 30"/>
  <position name="FL_calf_joint"  joint="FL_calf_joint"  kp="25" kv="2" forcerange="-30 30"/>
  <position name="RR_hip_joint"   joint="RR_hip_joint"   kp="25" kv="2" forcerange="-20 20"/>
  <position name="RR_thigh_joint" joint="RR_thigh_joint" kp="25" kv="2" forcerange="-30 30"/>
  <position name="RR_calf_joint"  joint="RR_calf_joint"  kp="25" kv="2" forcerange="-30 30"/>
  <position name="RL_hip_joint"   joint="RL_hip_joint"   kp="25" kv="2" forcerange="-20 20"/>
  <position name="RL_thigh_joint" joint="RL_thigh_joint" kp="25" kv="2" forcerange="-30 30"/>
  <position name="RL_calf_joint"  joint="RL_calf_joint"  kp="25" kv="2" forcerange="-30 30"/>
  <!-- 轮：速度伺服 kv=0.5，ctrl 是目标角速度（rc_wheeled.py） -->
  <velocity name="FR_feet_joint" joint="FR_feet_joint" kv="0.5" ctrlrange="-61.47 61.47" forcerange="-3.69 3.69"/>
  <velocity name="FL_feet_joint" joint="FL_feet_joint" kv="0.5" ctrlrange="-61.47 61.47" forcerange="-3.69 3.69"/>
  <velocity name="RR_feet_joint" joint="RR_feet_joint" kv="0.5" ctrlrange="-61.47 61.47" forcerange="-3.69 3.69"/>
  <velocity name="RL_feet_joint" joint="RL_feet_joint" kv="0.5" ctrlrange="-61.47 61.47" forcerange="-3.69 3.69"/>
</actuator>
```

要点：

- **轮子绝不能是 `<position>`**：`JointVelocityActionCfg` 只是把 ctrl 写进去，执行器类型决定语义。
- MuJoCo 的 `<position>` 自带阻尼项 `kv`，所以关节本体不要再写 `damping`，否则阻尼叠加一倍。
- 执行器顺序 = 策略动作顺序（腿 12 + 轮 4），也决定 `keyframe` 的 `ctrl` 顺序。
- 若 mjwarp 不认 `<velocity>`（概率低），兜底：改成 `<motor joint=... forcerange=.../>`，YAML 里把 `JointVelocityActionCfg` 换成 `JointEffortActionCfg`（此时需要自己给轮速环，控制难度上升，仅作最后手段）。

### 3.3 关节本体参数（脚本已写入）

```xml
<!-- 12 条腿关节：armature / frictionloss 从 rc.py 搬过来，damping 保持 0 -->
<joint name="FR_hip_joint" type="hinge" axis="1 0 0" range="-0.977 0.977"
       armature="0.02" frictionloss="0.05"/>
<!-- 4 个轮关节：continuous，无 range -->
<joint name="FR_feet_joint" type="hinge" axis="0 1 0" limited="false" armature="0.001"/>
```

### 3.4 `scene.xml` 的 `home` 关键帧（脚本已写入）

转换脚本会写一个全零的 `home`，必须改成真实站姿。四个腿参数相同，所以顺序不影响：

```xml
<keyframe>
  <key name="home"
    qpos="0 0 0.42  1 0 0 0
          0 0.7 -1.5 0   0 0.7 -1.5 0   0 0.7 -1.5 0   0 0.7 -1.5 0"
    ctrl="0 0.7 -1.5   0 0.7 -1.5   0 0.7 -1.5   0 0.7 -1.5   0 0 0 0"/>
</keyframe>
```

- `qpos` 前 7 位是浮动基（`0 0 0.42` + 四元数 `1 0 0 0`），后面按**模型关节顺序**每腿 4 个（hip / thigh / calf / wheel），共 16。
- `ctrl` 按**执行器顺序** 16 个：12 个腿目标角 + 4 个轮目标角速度（0）。
- 高度 0.42 m 是 `RCWHEEL_STANDING_HEIGHT`；如果轮子悬空或穿地，±0.02 微调。
- 只写 `qpos` 不写 `ctrl` 也能跑，但重置后第一步的执行器目标会是旧值。

### 3.5 验收

`finalize_rc27dog_asset.py` 结尾已经打印并校验过一遍，这里只是独立复核：

```bash
# 结构检查：nq=23, nv=22, nu=16
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/rc27dog/scene.xml'); print('nq',m.nq,'nv',m.nv,'nu',m.nu,'njnt',m.njnt)"
# 有桌面时肉眼看站姿
uv run python -m mujoco.viewer --mjcf=src/unilab/assets/robots/rc27dog/scene.xml
```

`nu` 必须是 16；查看器里 `home` 应四轮着地、机身水平。

### 3.6 传到服务器

```bash
rsync -av src/unilab/assets/robots/rc27dog/ user@server:~/RCdog_UniLab/src/unilab/assets/robots/rc27dog/
```

---

## 4. 已落地的任务配置

### 4.1 仓库改动清单

| 文件 | 作用 |
|---|---|
| `src/unilab/tasks/locomotion/rc27dog/__init__.py` | 注册 `RC27DogWheelFlat`（mujoco + mjwarp） |
| `src/unilab/tasks/__init__.py` | 把新模块加进 `__unilab_registry_modules__` |
| `src/unilab/conf/flashsac/task/rc27dog_flat/base.yaml` | 16 DoF 任务契约（观测/动作/奖励/终止/事件） |
| `src/unilab/conf/flashsac/task/rc27dog_flat/mujoco.yaml` | MuJoCo owner（先验证资产） |
| `src/unilab/conf/flashsac/task/rc27dog_flat/mjwarp.yaml` | MJWarp owner（主训练，4096 env） |
| `scripts/tools/finalize_rc27dog_asset.py` | 转换后一次补全执行器/关节参数/关键帧并校验 |

训练命令：

```bash
uv run train --algo flashsac --task rc27dog_flat --sim mjwarp    # 主
uv run train --algo flashsac --task rc27dog_flat --sim mujoco    # 验证
```

### 4.2 与 IsaacLab 配置的对应关系

| robot_lab（`wheeled/rc`） | UniLab YAML |
|---|---|
| `LEG_JOINT_NAMES` = FR, FL, RR, RL | `actuator_names` 前 12 项（顺序一致） |
| `WHEEL_JOINT_NAMES` = FR, FL, RR, RL | `actuator_names` 后 4 项 |
| `joint_pos.scale = {hip:0.125, 其它:0.25}` | `scale` 字典（同样用正则键） |
| `joint_vel(轮).scale=5.0`, clip ±61.47 | `wheel_vel` term |
| policy 不含 `base_lin_vel` | policy 不写该 term；critic 写 |
| policy `joint_pos` 只含腿、`joint_vel` 含全部 16 | `asset_cfg.joint_names` 只列 12 腿 |
| `sim.dt=0.005`, `decimation=4` | `sim_dt: 0.005`, `ctrl_dt: 0.02` |
| `base_height_l2 target=0.42` | 同值 |
| `track_lin_vel_xy=5.0`, `track_ang_vel_z=3.0`, `is_terminated=-100` | 同值 |
| `heading_command=True`, `rel_heading_envs=1.0`, 指令范围 ±1.0 | 同值 |

### 4.3 这一版**没**迁过来的奖励（需要的话第二步再加）

`lin_vel_z_l2`、`lin_vel_xy_delta_l2`、`joint_torques_l2`、`joint_acc_l2`、`joint_power`、
`joint_vel_limits`、`joint_pos_penalty`、`stand_still`、`joint_mirror`、`action_sync`、
`undesired_contacts`、`contact_forces`、`feet_*`。

原因：UniLab 的内置 mdp 里没有对应 term（接触类奖励还需要在 `scene.xml` 里加 contact sensor）。
先把「能走起来」跑通，再按需写 `src/unilab/tasks/locomotion/rc27dog/manager_terms.py` 补。

---

## 5. 从零到训练的阶梯

| 级别 | 命令 | 通过标准 | 预计 |
|---|---|---|---|
| L0 环境 | `uv run python -c "import torch, warp"` | 打印 5080 | 0.5 天 |
| L1 后端 | `... --task g1_walk_flat --sim mjwarp algo.num_envs=64 algo.max_iterations=2` | 跑完不报错 | 10 分钟 |
| L2 资产 | §3.5 的 `nq/nv/nu` 检查 + viewer | `nu=16`，站姿正确 | 0.5~1 天 |
| L3 任务 | `... --task rc27dog_flat --sim mujoco algo.num_envs=16 algo.max_iterations=1 training.no_play=true` | 跑完、存 checkpoint、无 NaN | 10 分钟 |
| L4 短跑 | `... --sim mujoco algo.num_envs=512 algo.max_iterations=20` | reward 上升、不立即终止 | 半小时 |
| L5 全速 | `... --sim mjwarp algo.num_envs=4096` | `Perf/total_fps` 达到预期 | 数小时~数天 |

启动与监控：

```bash
mkdir -p logs
tmux new -s train
uv run train --algo flashsac --task rc27dog_flat --sim mjwarp training.no_play=true \
  2>&1 | tee logs/rc27dog_mjwarp.log

uv run tensorboard --logdir logs --port 6006     # ssh -L 6006:localhost:6006
```

看板重点：`Perf/total_fps`、`Train/mean_reward`、`Train/mean_episode_length`、
`Episode/timeout_rate`、`reward/track_lin_vel`、`reward/track_ang_vel`。

回放（服务器无显示器）：

```bash
uv run eval --algo flashsac --task rc27dog_flat --sim mjwarp --load-run -1 --render-mode viser
# ssh -L 8080:localhost:8080，浏览器开 http://localhost:8080
```

---

## 6. 风险清单

| 风险 | 触发条件 | 应对 |
|---|---|---|
| mjwarp 不认 `<velocity>` 执行器 | 启动即报 actuator/ctrl 维度错 | 换 `<motor>` + `JointEffortActionCfg`（§3.2 末尾） |
| warp 对 sm_120 编译失败 | L1 冒烟报 NVCC 错 | 退回 `--sim mujoco`，同时报 issue |
| mjwarp 不支持我们 MJCF 的某个元素 | 模型加载报错 | 删掉该类元素（可视化网格可剥离）或退回 mujoco |
| 16 GB 不够 4096 env | OOM | `algo.num_envs=2048`，`batch_size=2048` |
| 轮子位置项混进观测 | 复制配置时漏改 | `joint_pos` 的 `asset_cfg` 必须只列 12 腿 |
| 奖励尺度导致原地不动 | 初期常见 | 先缩小指令范围 `env.commands.twist.ranges.lin_vel_x=[0,0.6]` |
| `/dev/shm` 太小（容器） | off-policy 启动报共享内存不足 | 裸机一般够；容器加 `--shm-size=8g` |
| 资产路径相对启动目录 | 换目录启动找不到 scene.xml | 固定在仓库根运行 |

---

## 7. 还需要你确认/决定的

1. **`home` 姿态**：`0 0 0.42` + `thigh 0.7 / calf -1.5` 是 IsaacLab 的初始位；实际站高以查看器里四轮着地为准，可能要 ±0.02。
2. **指令范围**：现在照搬 IsaacLab 的 ±1.0 m/s、±1.0 rad/s。第一次训练建议先收到 `[0, 0.6]` 让策略先学会走。
3. **`num_envs`**：mjwarp owner 先给 4096；L5 若 OOM 或 fps 不升反降，降到 2048。
4. **是否补接触类奖励**：需要往 `scene.xml` 加 `<contact>` sensor（每轮一条），再写自定义 term，属于第二阶段。
