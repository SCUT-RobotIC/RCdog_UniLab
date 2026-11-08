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

`tests/fixtures/isaac_lab_cartpole/` 迁移了 Isaac Lab commit
`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` 的 Manager-Based Cartpole task。它保留
全部 12 个来源 term 的名称和顺序，同时把 Torch 适配为 NumPy、嵌套 config object 适配
为 Hydra YAML，并用 fixture-local MJCF 实现 scene/action/reset 边界。这只是 test-only
证据，不是 production task 或 Isaac Lab 全量支持声明。

## 另请参阅

- {doc}`Manager-Based API <../../4-developer_guide/1-architecture/6-manager_based_api>`
- {doc}`Env contract <../../4-developer_guide/2-contracts/1-env_contract>`
- {doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`
