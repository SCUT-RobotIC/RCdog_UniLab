# 评估与回放

```bash
# 最近一次运行
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

# 无头视频导出
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco \
    --load-run -1 --render-mode record

# Off-policy 回放可以跳过 ONNX 导出，但仍然录制 MP4
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1 \
    --render-mode record training.export_onnx=false

# 演示（首次运行会从 HF 下载检查点）
uv run demo dance
```

渲染模式：

- `interactive` — 打开查看器窗口（macOS Motrix 上的默认值）。
- `record` — 将 MP4 写入 `runs/<run>/playback/`。
- `viser` — 在基于浏览器的 viser viewer 中展示回放
  （支持所有具备 physics-state playback 的后端：`mujoco`、`mjwarp`、`newton`、`drake`、`superdex`）。
- `none` — 跳过渲染，仅计算指标。

当使用 `--sim mujoco --render-mode interactive` 时，`uv run eval` 会直接启动
`play_interactive.py` MuJoCo viewer。该模式只 rollout 一个环境以保持相机和交互控制；
`training.play_env_num` 在此模式下会被忽略。

当使用 `--sim mujoco` / `--sim mjwarp --render-mode viser` 时，`uv run eval` 改为启动
`play_viser.py`，在 `http://localhost:8080` 提供浏览器回放（可通过 `viser.*`
配置组调整）。该模式不需要本地显示，在 macOS 上也不需要 `mjpython`。

`training.export_onnx=false` 目前仅适用于 off-policy 回放路径
（`src/unilab/scripts/train_sac.py` / `src/unilab/scripts/train_flashsac.py`
以及使用 `--algo sac|flashsac` 的 CLI 运行）。它会跳过
`policy.onnx` 的导出与校验，但仍会执行回放和视频录制。

## MuJoCo viewer 可视化脚本

常规评估和视频导出优先使用 `uv run eval`。需要直接打开 `mujoco.viewer`
调试策略时，可以使用低层脚本 `src/unilab/scripts/play_interactive.py`。

`src/unilab/scripts/play_interactive.py` 是通用 MuJoCo viewer 入口，适合 PPO、APPO、
SAC 和 FlashSAC 的策略可视化。它使用 `--algo / --task / --sim`
选择算法和 owner config；无论 `--sim` 选择 MuJoCo 还是 Motrix，窗口都使用
`mujoco.viewer` 可视化，`--sim` 只决定读取哪份配置。

```bash
# 使用 owner config 中的 interactive.action_mode；全局默认是 zero action
uv run src/unilab/scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco

# 随机动作
uv run src/unilab/scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    interactive.action_mode=random

# 策略动作
uv run src/unilab/scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    algo.load_run=-1 interactive.action_mode=policy

uv run src/unilab/scripts/play_interactive.py --algo flashsac --task g1_walk_flat --sim mujoco \
    algo.load_run=-1 interactive.action_mode=policy

uv run src/unilab/scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    interactive.action_mode=policy interactive.keyboard=true
```

动作模式通过 `interactive.action_mode=zero|random|policy` 选择；不传时使用
owner config 中的设置（全局默认是 `zero`，部分任务 YAML 会覆盖为 `policy`）。
键盘控制通过 Hydra override 开启：`interactive.action_mode=policy
interactive.keyboard=true`。开启键盘控制时会检查 policy obs 是否包含 velocity
command，不满足时退出。处于 `policy` 模式、且 locomotion 速度指令任务的 policy
obs 中包含 velocity command 时，窗口会自动显示绿色目标速度和蓝色当前速度箭头；
`zero` 和 `random` 模式不显示速度箭头。

底层 API 请参阅 `unilab.visualization.playback`。

## 小键盘速度控制

例如回放训练好的 RC27 碎石地形策略：

```bash
uv run eval --algo flashsac --task rc27dog_rough_stones --sim mujoco \
    --load-run -1 --render-mode interactive interactive.keyboard=true
```

点击本地 MuJoCo 窗口取得焦点，小键盘打开 NumLock；主键盘同名数字键也可用。

| 按键 | 功能 |
| --- | --- |
| `8` / `2` 或 ↑ / ↓ | vx 增加 / 减少（前 / 后）|
| `4` / `6` | vy 增加 / 减少（左 / 右）|
| `7` / `9` 或 ← / → | 转向角速度增加 / 减少（左转 / 右转）|
| `5`、`0` 或 Enter | 全部速度指令清零 |
| Backspace | 重置机器人并清零指令 |
| Space / N | 暂停或继续 / 暂停并单步 |
| `+` / `-` | 提高 / 降低回放速度 |
| Esc | 退出 |

每次按键累加一档，松开后保持指令。默认线速度步长 0.1 m/s、角速度步长
0.2 rad/s，分别用 `interactive.keyboard_step_lin`、`interactive.keyboard_step_ang`
修改。限幅来自任务速度 command term 的 ranges。速度在机器人机体系下定义，
正 vx 向前，正 vy 向左，正转向为左转；清零指令由策略减速，并非物理急停。

键盘控制要求恰好一个 `UniformVelocityCommand`（含子类）和策略组中的
`generated_commands` 观测项，支持观测缩放与历史堆叠。手动命令持续接管随机
重采样及航向控制，历史和延迟按正常控制步推进；新指令在下次观测更新后供策略
使用。起始指令为零；键盘模式会关闭任务的 episode 时限，正常行走可持续运行。
摔倒或越出地形边界后窗口暂停并等待 Backspace；重置后自动继续。
地形与物理随机化继续使用任务配置。这套按键适用于本地 interactive 窗口。
