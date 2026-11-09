# 从 Isaac Lab 迁移

把 Isaac Lab Manager-Based task 迁入 UniLab 时，应保留 manager 与 term 结构，只在各自
owner 边界适配配置、数值执行和场景访问；不要把 task 重写成单体 environment 子类。

这是基于源码的兼容迁移，不代表任意 Isaac Lab task 都能不修改直接运行。目标路径是：

```text
Hydra owner YAML
  -> plain ManagerBasedRlEnvCfg
  -> Registry + make_manager_based_rl_env
  -> NumPy/SimBackend runtime 上的 ManagerBasedRlEnv
  -> 交给现有 training 和 IPC 路径的 TorchEnvState
```

## 兼容边界

```{list-table}
:header-rows: 1
:widths: 28 24 48

* - Isaac Lab 表面
  - UniLab 状态
  - 迁移规则
* - Manager 类别、term 名称和字典顺序
  - Compatible
  - 保持 observation、action、event、reward、termination、command 与
    curriculum term 的顺序。
* - Function/class term 与 `func + params`
  - Compatible
  - 把 import 改为 `unilab.managers`；保留 term 边界和局部
    `reset(env_ids)` 语义。
* - `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`
  - Compatible 拼写 alias
  - UniLab canonical 名称是 `ManagerBasedRlEnv` 与 `ManagerBasedRlEnvCfg`；alias
    指向同一份实现。
* - Tensor 数值与运算
  - Adapted
  - 把 `torch.Tensor` 换成 `np.ndarray` 并使用向量化 NumPy；manager-facing
    API 没有 device 接口。
* - 嵌套 `@configclass` task 配置
  - Adapted
  - 把完整 task 声明迁入唯一 Hydra owner YAML；用 `_target_` 选择具体 config
    dataclass，用 dotted `func` 选择 term。
* - `InteractiveSceneCfg`、USD 与 PhysX view
  - Adapted 或 Unsupported
  - 声明 task-owned `SceneCfg` 与 `EntityCfg`；状态和控制只通过
    `SceneEntityCfg` 与公共 entity facade 访问。不支持的能力在冷路径绑定时报错。
* - Omniverse、Isaac renderer 与 Torch/PhysX mutation
  - Unsupported
  - UniLab 不安装这些 runtime，也不提供静默模拟或回退。
```

规范边界见 {doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`。
只有已经被 registry、配置和测试覆盖的表面才能声明为 Compatible。

## 迁移步骤

### 1. 盘点来源 task

固定 Isaac Lab revision，并列出来源 manager group、term 名称与顺序、参数、observation
维度、action 维度、reset 行为和 episode timing。写代码前逐项分类：

- 复用已有 `unilab.managers` config 或 `unilab.envs.mdp` term；
- 把 task-specific term 从 Torch 适配为 NumPy；
- 如果 term 依赖公共 entity 或 `SimBackend` contract 尚未提供的能力，立即停止。

不能用 `getattr`/`hasattr` 探测 backend 对象、返回零，或把 task 路由回 legacy env。

### 2. 在冷路径迁移 scene 与 asset

用 task-owned `SceneCfg` 代替 Isaac Lab 的 USD/`InteractiveSceneCfg` 声明，显式声明 term
需要的每个 entity 与 selector。`SceneEntityCfg` 在 materialization 时只解析一次名称和
正则表达式；reset/step 复用缓存 ID 与 NumPy view。

Cartpole fixture 使用最小 task-owned MJCF。更复杂的 asset 必须遵守
{doc}`场景组合 <../../4-developer_guide/1-architecture/4-scene_composition>`，并只使用所选
backend 的正式能力。

### 3. 迁移 term 代码，不改 manager 结构

保留每个 function/class term 及其参数，机械地把 Torch 类型与运算改为 NumPy，保持 batch
shape，并在来源 term 返回每环境数值时继续返回每环境数值。Stateful term 在构造时解析
selector、分配 buffer，热路径只更新 NumPy buffer。

Python 只拥有 term 实现和可复用 config dataclass，不能再保存一份 task-specific term
启停清单或默认 weight。

### 4. 让 Hydra 成为唯一 task 配置 owner

在 owner YAML 中声明 scene、timing、group、term、具体 config 类型、callable、参数、
weight 和 observation mapping。例如：

```yaml
env:
  observations:
    policy:
      terms:
        joint_pos_rel:
          func: unilab.envs.mdp.joint_pos_rel
  terminations:
    time_out:
      func: unilab.envs.mdp.time_out
      time_out: true
  policy_observation_group: policy
  critic_observation_group: null

reward:
  alive:
    func: unilab.envs.mdp.is_alive
    weight: 1.0
```

值类型唯一且具体的 manager mapping（observations / events / rewards /
terminations / curriculum / metrics / recorders）可以省略 `_target_`，物化时按字段
类型注解推断；`actions` / `commands` 的基类是抽象的，必须显式声明具体 `_target_`
（如 `unilab.envs.mdp.JointPositionActionCfg`）。`unilab.managers.` 下的 config 类
（如 `SceneEntityCfg`）可以直接写裸类名。

Hydra compose 在冷路径把这份声明物化为 plain typed config。未知字段、无法解析的
`_target_`/`func` 和错误 config 类型都会在 reset/step 之前报错。直接用 Python 构造
config 只用于 focused 底层测试。

#### 执行器模型与指令延迟

执行器动力学保留在机器人或任务的 MJCF 中，不要恢复 `EntityCfg` 执行器层。
对于转子惯量 `J` 和减速比 `N`，将反射惯量 `I = J * N**2` 写入关节的
`armature`。对于固有频率 `omega` 和阻尼比 `zeta`，推导
`kp = I * omega**2`、`kv = 2 * zeta * I * omega`，并在 position 执行器上声明：

```xml
<joint name="knee" armature="0.02"/>
<actuator>
  <position name="knee_drive" joint="knee" kp="40" kv="1.2"/>
</actuator>
```

`JointPositionActionCfg` 与 `MotionJointPositionActionCfg` 还支持
`delay_min_lag` 和 `delay_max_lag`。lag 是包含端点的物理子步数，对应时间范围为
`lag * env.sim_dt`，而不是 `ctrl_dt` 的倍数：

```yaml
env:
  actions:
    joint_pos:
      _target_: unilab.envs.mdp.JointPositionActionCfg
      entity_name: robot
      actuator_names: [".*"]
      delay_min_lag: 0
      delay_max_lag: 2
```

Manager 按环境、按物理子步独立采样延迟，局部 reset 只清除所选环境的历史。
初始化或 reset 后的首个目标会回填历史。这些字段覆盖指令延迟，不覆盖 DC
电机的转矩—转速饱和曲线。

### 5. 只注册一条通用 runtime 路径

Task module 为仓库已经实际支持的每个 backend 注册 `ManagerBasedRlEnvCfg` 与
`make_manager_based_rl_env`。Backend owner YAML 只承载 backend 身份与 tuning。用户通过
标准 CLI 选择 compose owner，例如：

```bash
uv run train --algo ppo --task <task> --sim mujoco
```

不要增加 task-specific 训练脚本分支、env factory、runner 或 IPC 路径。

generic factory 规则只有一个 maintainer 批准的已注册例外：
`make_g1_walk_env`（`src/unilab/tasks/locomotion/g1/manager_terms.py`）构造
`G1WalkManagerBasedEnv` 子类，承载 G1 walk 的 manager-based 生产 runtime。其余所有
Compatible task 都直接注册
`make_manager_based_rl_env`。

### 6. 在适配风险附近验证

测试 Hydra compose 与 typed materialization、term 顺序与数学、selector 失败、
observation/action shape、局部 reset，以及至少一个真实已注册 backend 的 transition。行为
应与固定来源 task 对比；完成语义迁移后再做性能 benchmark。

## 任务迁移最终状态

任务迁移状态由 registry 与迁移矩阵维护。fail-closed 的 source of truth 是
`src/unilab/tasks/migration_matrix.py`：`migration_record()` 对没有
entry 的 production task 名称抛出 `KeyError`，因此新增 production 注册必须显式做出
迁移决策。

- 36 个 task 为 **Compatible**（`target=complete`）：Hydra owner YAML 物化 canonical
  tensor-only Manager-Based runtime。

## 仓库证据

## RC27Dog 轮足迁移

`robot_lab_vision` 的 velocity/RCwheel 分支包含六个任务：Flat、Rough、RoughStones、
LowBar、LowWall 和 BrokenBridge。它们现在由 UniLab 的 FlashSAC owner 配置管理，任务
名称和后端入口如下：

```text
rc27dog_flat          -> RC27DogWheelFlat
rc27dog_rough         -> RC27DogWheelRough
rc27dog_rough_stones  -> RC27DogWheelRoughStones
rc27dog_low_bar       -> RC27DogWheelLowBar
rc27dog_low_wall      -> RC27DogWheelLowWall
rc27dog_broken_bridge -> RC27DogWheelBrokenBridge
```

每个任务都有 `mujoco` 和 `mjwarp` owner。训练入口使用仓库顶层 CLI，例如先用一个环境
做启动检查：

```bash
uv run train --algo flashsac --task rc27dog_flat --sim mujoco \
  algo.num_envs=1 algo.max_iterations=1 training.no_play=true
```

迁移代码位于 `src/unilab/tasks/locomotion/rc27dog/`，机器人和障碍场景位于
`src/unilab/assets/robots/rc27dog/`，Hydra owner 位于
`src/unilab/conf/flashsac/task/rc27dog_*/`。轮足动作是 16 维：前 12 维按
FR、FL、RR、RL 排列的腿部位置目标，后 4 维是轮速目标。旧 Isaac Lab checkpoint
不能直接恢复，因为动作布局、观测历史和算法运行时都必须逐项匹配。

正式训练前必须逐项完成以下检查：

1. 在目标机器上用真实 MuJoCo 和 MJWarp 分别完成环境构造、`reset`、至少一个完整
   `step` 和局部 reset；仅 Hydra compose 或 `registry.make()` 成功不能证明物理运行正常。
2. 核对 `rc27dog.xml` 中的 16 个关节、执行器限位、轮胎碰撞体和默认 keyframe。当前
   轮速使用 3.69 N·m、587 rpm 的峰值边界，尚未建模转速-力矩曲线、热限制或 C620
   电流限制；需要用实机响应校准轮子的阻尼和 armature。
3. 在 Flat、Rough 和 RoughStones 上检查初始姿态扰动端点、轮胎和地形的最小间隙、
   楼梯/碎石碰撞以及长时间运行中的穿透。名义 `base z=0.48 m` 的轮胎几何间隙不
   代表带扰动和完整碰撞体时的安全间隙。
4. 为接触、力矩、关节加速度和障碍几何检查实际传感器及 body 名称。LowBar、LowWall
   和 BrokenBridge 的奖励/终止依赖这些公开 sensor；缺失时应在构造阶段失败，不能
   用零值掩盖。
5. 对六个任务分别记录 policy/critic observation shape、16 维 action shape、命令
   范围、终止条件和 reward 数值范围，并确认 MuJoCo 与 MJWarp 的读数一致。尤其检查
   LowBar 的低姿态、LowWall 的越墙进度和 BrokenBridge 的沟槽相位与碰撞几何一致。
6. 完成一轮短时稳定性实验后再打开大规模 FlashSAC 配置：确认没有 NaN、动作饱和、
   接触爆炸或 episode 无故提前结束，并保存配置快照和日志。当前 smoke 命令只验证
   启动路径，不能作为收敛或性能结果。
7. 训练出新策略后，先做 MuJoCo/MJWarp sim-to-sim 对比，再按
   `docs/sphinx/source/zh_CN/3-deployment/2-sim_to_sim/` 的 contract 检查流程导出和
   回放；没有完成这些检查前不要接入实机。

### RoughStones 对齐与训练（2026-11-08）

本次只对 `RC27DogWheelRoughStones` 对齐原版最终生效的机器人参数和任务配置，
地形按近似效果验收。源仓库为 `/home/haji/RClab/robot_lab_vision`，HEAD 为
`82595e4`；UniLab 修改位于当前工作区。其他 RC 任务不在本次对齐验收范围内。

核对来源为 `docs/27dog_wheeled_training_zh_CN.md`、velocity 目录的
`velocity_env_cfg.py`、`config/wheeled/rc/rough_env_cfg.py`、
`rough_stones_env_cfg.py`、`common.py`、资产 `rc.py`、`rc_wheeled.py`，
以及 `data/Robots/RC/27dog urdf v1/27dog urdf v1/urdf/27dog urdf v1.urdf`。
核对顺序包含最后一次 `configure_wheeled_rc()`：站立奖励高度最终为 **0.42 m**，
初始化高度为 **0.48 m**，不能使用父类中随后被覆盖的 0.33 m。

#### 机器人参数

`src/unilab/assets/robots/rc27dog/rc27dog.xml` 已修复 base 缺少显式惯量、
四肢丢失惯量交叉项，以及重复叠加被动阻尼的问题。比较 MuJoCo 编译结果时，
将主惯量按惯性坐标系姿态旋转回 link 坐标系，再与原 URDF 的完整惯量张量比较。

| 检查 | 对齐结果 |
| --- | --- |
| 总质量 | 原 URDF 和编译结果均为 23.348338686 kg；修复前约 61.8567 kg |
| base 质量 | 8.54711286 kg |
| base 质心 | (-0.0114530629, 0.000184145189, 0.0108079466) m |
| 17 个 link | 名义质量、质心、完整惯量逐项一致 |
| 16 个关节 | 父子关系、原点、轴向、腿关节位置边界和轮 continuous 属性一致 |
| 碰撞体 | 类型、尺寸、位置、旋转一致；轮半径 0.085 m；禁用机器人自碰撞 |
| 腿执行器 | kp=25、kd=2；hip 力矩上限 20 N·m，thigh/calf 为 30 N·m |
| 腿关节参数 | armature=0.02、frictionloss=0.05；移除额外被动 damping=2 |
| 腿目标延迟 | hip/thigh/calf 三组各自每环境采样；每回合保持 2–7 个物理步 |
| 轮执行器 | kd=0.5、力矩上限 3.69 N·m、目标速度上限 61.470496255240 rad/s |
| 轮关节参数 | armature=0.001、frictionloss=0；被动 damping=0 |

腿速度阈值按原版 hip 30、thigh/calf 10 rad/s 用于速度越界奖励。
MuJoCo position/velocity servo 与 PhysX 显式腿 PD、隐式轮驱动的积分和约束求解
不同；轮目标速度截断不等于 PhysX 的关节求解器速度约束，不能宣称跨引擎轨迹
逐步相同。轮子的转速—力矩曲线、热限制等在源配方中也没有建模。

四肢 16 个 visual STL 文件的 SHA-256 相同；base visual STL 与源文件不同。
本次保留既有 visual 文件，物理质量和碰撞体已独立核对，visual 差异不参与质量推导。

#### 奖励与观测

独立 owner 为 `src/unilab/conf/flashsac/task/rc27dog_rough_stones/base.yaml`。
下表按原版最终启用的 23 项对齐权重；公式通过任务模块
`source_terms.py` 和 `manager_terms.py` 实现。

| 原版奖励 | UniLab owner 名称 | 权重 |
| --- | --- | ---: |
| is_terminated | terminated | -100 |
| lin_vel_z_l2 | lin_vel_z | -1 |
| ang_vel_xy_l2 | ang_vel_xy | -1 |
| flat_orientation_l2 | flat_orientation | -0.2 |
| lin_vel_xy_delta_l2 | lin_vel_xy_delta | -1.2 |
| base_height_l2 | base_height | -6 |
| joint_torques_l2 | joint_torques | -0.00003 |
| joint_acc_l2 | joint_acc | -0.000003 |
| joint_pos_limits | joint_pos_limits | -4 |
| joint_vel_limits | joint_vel_limits | -3 |
| joint_power | joint_power | -0.000025 |
| stand_still | stand_still | -0.7 |
| joint_pos_penalty | joint_pos_penalty | -0.8 |
| joint_mirror | joint_mirror | -0.04 |
| action_rate_l2 | action_rate | -0.02 |
| undesired_contacts | undesired_contacts | -1.5 |
| contact_forces | contact_forces | -0.00025 |
| track_lin_vel_xy_exp | track_lin_vel | 5.5 |
| track_ang_vel_z_exp | track_ang_vel | 2.5 |
| feet_contact_without_cmd | feet_contact_without_cmd | 0.05 |
| upward | upward | 0.35 |
| stance_hip_zero | stance_hip_zero | 1 |
| stance_calf_singularity | stance_calf_singularity | -1 |

线速度只跟踪机体系 xy，角速度只跟踪机体系 z；相关原版项保留直立门控。
高度奖励使用 base 下方 0.1×0.1 m 网格的平均地面高度加 0.42 m，
主 critic 扫描为 1.6×1.0 m、0.1 m 分辨率。腿软位置范围按原边界的 0.9 收缩；
速度越界按每关节最多 1 rad/s 累加。接触奖励读取三维 net force，
惩罚项和支撑项保留最近 3 个物理步的历史；轮部力阈值为 80 N。
关节加速度按最后物理步的速度差分计算，未以控制周期差分代替。
`rough_stones_task.xml` 提供关节力矩、关节速度和接触历史传感器。

保持原版禁用的 `feet_air_time`、`feet_slide`、`feet_height`、
`feet_height_body`、`feet_gait`、`swing_feet_clearance`、`wheel_vel_penalty`
为未启用状态。

| 项目 | 当前配方 |
| --- | --- |
| 周期/时限 | 物理 0.005 s、控制 0.02 s、回合 20 s |
| 动作 | FR、FL、RR、RL；12 腿位置后接 4 轮速度 |
| 动作缩放 | hip 0.125、其他腿 0.25、轮 5 |
| 命令范围 | x ±0.8、y ±0.6 m/s、yaw ±0.8 rad/s |
| 命令采样 | 5–15 s；站立比例 0.02；xy 模长 ≤0.2 m/s 置零 |
| 朝向控制 | heading=true，刚度 0.5 |
| policy | 角速度、投影重力、指令、腿位置、全部关节速度、上次动作；各 5 帧，265 维 |
| policy 噪声 | 上述角速度 ±0.2、重力 ±0.05、位置 ±0.03、速度 ±1.5，按原版缩放/裁剪 |
| critic | 原版顺序与缩放；增加世界 z、扫描、关节力矩；552 维，无噪声 |
| 终止 | timeout、地图外边界、dead_down(threshold=0)；不再用世界 z<0.15 m 终止 |
| 地形课程 | 按位移和命令速度升级/降级；起始等级 0，仅更新 reset 的环境 |

#### 随机化与地形近似

`events.py` 实现原版随机化范围，并通过公开 Entity/SimBackend 接口写入。
材质、质量/惯量、关节参数每个环境仅采样一次，在 reset 事务中重放，避免每回合
重采样或累乘；COM 使用 startup 事件。增益和外力按回合重采样。

| 随机化 | 范围 |
| --- | --- |
| base 质量/其余 link 质量 | 加 [-1,3] kg / 缩放 [0.8,1.2]，按质量比例重算惯量 |
| base COM | x [-0.05,0.08]、y ±0.05、z [-0.02,0.04] m |
| 关节 friction/armature | 缩放 [0.2,1.5] / [0.8,1.2] |
| 全部执行器 kp/kd | 缩放 [0.75,1.25] / [0.6,1.5]；轮 kp=0 保持为 0 |
| 材质 64 桶 | static [0.35,1.4]、dynamic [0.25,1.1]、restitution [0,0.2] |
| root reset | 地形 origin +0.48+[0,0.1] m；xy ±0.5 m；roll/pitch ±0.05；yaw ±3.14 rad |
| root reset 速度 | xyz ±0.5/±0.5/±0.2 m/s；roll/pitch ±0.25、yaw ±0.5 rad/s |
| 腿 reset | 位置加 ±0.05 rad、速度 ±0.5 rad/s |
| 轮 reset | 位置和速度均为 0 |
| 持续外力/力矩 | base 局部坐标系 ±6 N / ±4 N·m，reset 采样并在每步持续施加 |
| 推扰 | 每 10–16 s 覆盖 xy 速度至 ±0.25 m/s、roll/pitch 角速度至 ±0.12 rad/s |

MuJoCo 没有独立静/动库仑摩擦系数，当前以 dynamic 样本写入 sliding friction，
保留原有 torsional/rolling friction；恢复系数用 `solref` 阻尼近似。
这保留源采样范围，不能视作 PhysX 材质求解等价。

地形保留七类、25/30/15/10/10/5/5% 配比、8×8 m tile、10 行难度、
0.05/0.005 m 尺度、0.15/0.05 m 边脊/对角脊、0.10 m 台阶、20° 坡。
当前生成器按实际七类分列，不能将源 `num_cols=20` 视为逐块复刻；
heightfield 的网格与楼梯立面、随机采样、材质接触以及小地图保底配比均为近似。
地形效果按本次约定允许近似，未声称 mesh 或随机种子逐点一致。

#### 已执行验证与开始训练

已执行配置、公式、延迟、物理参数、随机化、课程、实际环境 reset/局部 reset/step
检查。16 环境零动作连续运行 1000 控制步，共 16000 条 transition，观测和奖励均
有限；局部 reset 保留其他环境位置。统计为 16 次超时、2 次 dead_down，
无地图越界；修复前同类检查有 3009 次世界高度误终止。

MuJoCo FlashSAC 已完成 1 次迭代、4112 env steps，正常保存 `model_1.pt`，
日志为 `logs/flash_sac/RC27DogWheelRoughStones/2026-11-08_22-39-25_mujoco/`，
平均已结束回合长度为 196 控制步。随后以 16 环境完成 5 次迭代、4176 env steps，
优化器正常更新并保存 `model_5.pt`，日志为
`logs/flash_sac/RC27DogWheelRoughStones/2026-11-08_22-43-28_mujoco/`。
另以正式建议规模 256 环境完成 1 次迭代、25344 env steps，并保存 `model_1.pt`，
日志为 `logs/flash_sac/RC27DogWheelRoughStones/2026-11-08_22-45-50_mujoco/`。
复现命令为下方正式训练命令将 `algo.max_iterations` 改为 `1`。

本次验证命令与结果：

```bash
uv run pytest tests/tasks/locomotion/rc27dog tests/config/test_rc27dog_owner_configs.py \
  tests/config/test_rc27dog_rough_stones_source_contract.py \
  tests/tasks/test_package_boundary.py -q
make check
make test
```

专项测试最终为 **81 passed**；`make check` 通过，Pyright 留有现有的
`base/__init__.py` 导出警告。全仓 `make test` 的一次记录为 2890 passed、21 failed：
其中 3 项 RC 注册/NumPy 边界问题随后已修复并通过上述专项回归；其余失败涉及
Go2 mesh 缺失、Hugging Face 资产下载失败和 Motrix 日志捕获。
全仓测试未全部通过，不能将本任务通过理解为整个仓库通过。
追加文档检查后，专项与文档测试合计 **101 passed**；Sphinx HTML 构建通过，
保留 7 条本次未修改页面的 toctree 警告。

从仓库根目录启动，先以 256 环境进行正式训练：

```bash
uv run train --algo flashsac --task rc27dog_rough_stones --sim mujoco \
  algo.num_envs=256 algo.max_iterations=15000 training.no_play=true
```

上述 256 环境已通过启动测试，尚未做长期吞吐与稳定性测试；可按本机 CPU、显存与日志调整。
先查看早期回合长度、跟踪奖励、动作饱和和策略回放，再决定是否长时间运行。
短测试证明构造、采样、优化和保存链路可运行，不证明已经学会走路或最终收敛；
FlashSAC 与源 PPO 的收敛也不能按相同超参数假定一致。

当前请使用 **MuJoCo**。MJWarp 实测在生成地形阶段抛出
`host_numpy profile does not support generated terrain or height-field scanners`，
没有进入训练。对应 owner 的存在不代表本任务能在当前安装的 MJWarp 后端运行。

`tests/fixtures/isaac_lab_cartpole/` 迁移了 Isaac Lab commit
`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` 的 Manager-Based Cartpole task。它保留
全部 12 个来源 term 的名称和顺序，同时把 Torch 适配为 NumPy、嵌套 config object 适配
为 Hydra YAML，并用 fixture-local MJCF 实现 scene/action/reset 边界。这只是 test-only
证据，不是 production task 或 Isaac Lab 全量支持声明。

## 另请参阅

- {doc}`Manager-Based API <../../4-developer_guide/1-architecture/6-manager_based_api>`
- {doc}`Env contract <../../4-developer_guide/2-contracts/1-env_contract>`
- {doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`
