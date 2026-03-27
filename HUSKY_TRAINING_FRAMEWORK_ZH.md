# HUSKY 训练框架讲解（从简到深）

本文只讲训练框架，不展开部署/推理脚本。目标是让你先拿到大图，再逐层下潜到源码细节。

## 第 0 层：一页总览（先记住这个）

1. 训练入口是 `uv run train Mjlab-Skater-Flat-Unitree-G1 ...`，命令映射在 `pyproject.toml`，真正入口在 `src/mjlab_husky/scripts/train.py`。  
2. `train.py` 会先拿到 `task_id`，再从任务注册表取到三件事：环境配置 `env_cfg`、算法配置 `rl_cfg`、runner 类。  
3. 这个任务注册在 `src/mjlab_husky/tasks/skater/config/g1/__init__.py`，runner 是 `SkaterOnPolicyRunner`。  
4. 训练环境是 `G1SkaterManagerBasedRlEnv`，通过 `RslRlVecEnvWrapper` 暴露给 RSL-RL 的统一接口。  
5. 训练主循环在 `rsl_rl/runners/amp_on_policy_runner.py::learn`：采样动作、环境步进、存储 rollout、计算 returns、执行 update。  
6. 策略主体是 PPO（`AMP_PPO`），策略网络是 Actor-Critic（AC），其中 actor 负责动作分布，critic 负责价值估计。  
7. 这个项目的关键增量是 AMP：除了任务奖励，还用判别器对动作风格进行约束。  
8. AMP 的 expert 数据来自 `dataset/skate_push`，policy 数据来自环境轨迹，两者在 update 阶段共同训练判别器。  
9. 输出结果有两类：训练时用于继续训练/评估的 `model_*.pt`，以及（在 wandb logger 下）自动导出的 ONNX。  

### 模块地图（训练主线）

| 模块 | 责任 | 主要输入 | 主要输出 | 实现位置 |
|---|---|---|---|---|
| 训练入口 | 解析 task 和超参，启动训练 | CLI 参数、`task_id` | `run_train()` 执行 | `src/mjlab_husky/scripts/train.py` |
| 任务注册 | 绑定任务到 env/rl/runner | `task_id` | `env_cfg`、`rl_cfg`、`runner_cls` | `src/mjlab_husky/tasks/registry.py` + `src/mjlab_husky/tasks/skater/config/g1/__init__.py` |
| 环境配置 | 定义 obs/action/command/event/reward/termination | 任务设计 | `G1SkaterManagerBasedRlEnvCfg` | `src/mjlab_husky/tasks/skater/skater_env_cfg.py` |
| 环境执行 | 按 manager 顺序执行仿真与奖励 | 动作、命令、事件、传感器 | obs/reward/done/extras | `src/mjlab_husky/envs/g1_skate_rl_env.py` |
| Vec 包装器 | 适配 rsl_rl 的 VecEnv 协议 | 原始 env | `get_observations/step/get_amp_observations` | `src/mjlab_husky/rl/vecenv_wrapper.py` |
| AMP Runner | 训练主循环与日志/保存调度 | env + cfg | rollout、loss、checkpoint | `rsl_rl/runners/amp_on_policy_runner.py` |
| AMP_PPO | PPO + AMP 联合优化 | rollout、expert/policy AMP 样本 | 参数更新、loss 字典 | `rsl_rl/algorithms/amp_ppo.py` |
| AC 网络 | actor/critic 前向与分布 | policy/critic obs | action/value/logprob | `rsl_rl/modules/actor_critic.py` |
| AMP 判别器 | 计算风格奖励与判别损失 | 多帧 AMP 状态 | `disc_reward`、`amp_loss` | `rsl_rl/modules/discriminator_multi.py` |
| AMP 数据加载 | 从 npy 轨迹构建 expert 批次 | `dataset/skate_push` | `[batch, num_frames, amp_dim]` | `rsl_rl/utils/motion_loader_g1.py` |

---

## 第 1 层：PPO + AC 主架构

### 1.1 PPO 在这个项目中的角色

- PPO 负责策略更新（surrogate loss）和价值回归（value loss），并做 entropy 正则。  
- 在本项目中，PPO 的 loss 会额外叠加 AMP 的判别器损失和梯度惩罚。  

对应代码：

- 入口调度：`src/mjlab_husky/scripts/train.py`  
- 训练循环：`rsl_rl/runners/amp_on_policy_runner.py::learn`  
- 更新逻辑：`rsl_rl/algorithms/amp_ppo.py::update`  

### 1.2 Actor / Critic 各自组成

- Actor 与 Critic 都是 MLP（`rsl_rl/networks/mlp.py`），隐藏层维度来自任务配置。  
- 当前任务在 `src/mjlab_husky/tasks/skater/config/g1/rl_cfg.py` 里设置为 `actor_hidden_dims=(512,256,128)`、`critic_hidden_dims=(512,256,128)`。  
- Actor 输出动作分布参数：训练时采样 `Normal(mean, std)`，推理时 `act_inference` 直接输出均值。  
- Critic 输出单标量状态价值。  
- Actor/Critic 都支持经验归一化（`actor_obs_normalization`、`critic_obs_normalization`）。  

对应代码：

- `rsl_rl/modules/actor_critic.py`（`act` / `evaluate` / `act_inference` / obs 归一化）  
- `rsl_rl/networks/mlp.py`（层结构定义）  

### 1.3 训练核心张量流（最简版）

```text
obs = env.get_observations()
actions = policy.act(obs)
next_obs, rewards, dones, extras = env.step(actions)
alg.process_env_step(...)
...
alg.compute_returns(last_obs)
loss_dict = alg.update()
```

对应代码：

- rollout：`rsl_rl/runners/amp_on_policy_runner.py`（`learn` 循环）  
- returns：`rsl_rl/algorithms/amp_ppo.py::compute_returns`  
- update：`rsl_rl/algorithms/amp_ppo.py::update`  

---

## 第 2 层：Obs / Action / Reward 设计

### 2.1 Obs 设计（policy vs critic）

`src/mjlab_husky/tasks/skater/skater_env_cfg.py` 里把观测分为两组：

- `policy`：`command`, `heading`, `base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel`, `actions`, `phase`  
- `critic`：包含全部 `policy` 项，再加 `base_lin_vel`、接触力、过渡目标等特权信息  

关键点：

- `policy` 组启用 `history_length=5` 且 `flatten_history_dim=True`。  
- `policy` 组默认有噪声/扰动（训练），而 `play=True` 时会在配置里关闭 corruption。  

### 2.2 Action 语义

- 动作类型是 `JointPositionActionCfg(..., use_default_offset=True)`。  
- 在 `src/mjlab_husky/tasks/skater/config/g1/env_cfgs.py` 中，`joint_pos` 的 scale 被覆盖为 `G1_23Dof_ACTION_SCALE`。  
- 这意味着策略输出是“关节位置目标增量语义”，不是直接力矩控制。  

### 2.3 Reward 分组与约束

在 `skater_env_cfg.py` 中奖励分 4 组：

- `push_rewards`
- `steer_rewards`
- `transition_rewards`
- `regularization_rewards`

在环境 `step` 里真正组合时，会按 `contact_phase` 做相位门控（而不是简单全时段加和）：

```text
reward = push * phase_push
       + steer * phase_steer
       + regularization
       + transition * (phase_push2steer or phase_steer2push)
```

对应代码：

- 奖励项定义：`src/mjlab_husky/tasks/skater/mdp/rewards.py`  
- 相位与组合：`src/mjlab_husky/envs/g1_skate_rl_env.py`（`step` / `_resample_contact_phases`）  

### 2.4 termination / event / command 对分布的影响

- `termination`：包括 `fell_over`、`feet_off_board`、`illegal_contact` 等。  
- `event`：如 `push_robot`、摩擦/质心随机化，会改变训练数据分布。  
- `command`：`SkateUniformVelocityCommand` 会重采样速度与 heading，直接影响任务目标。  

对应代码：

- termination：`src/mjlab_husky/tasks/skater/mdp/terminations.py`  
- command：`src/mjlab_husky/tasks/skater/mdp/velocity_command.py`  
- 事件与配置：`src/mjlab_husky/tasks/skater/skater_env_cfg.py`  

---

## 第 3 层：AMP 子系统

### 3.1 AMP 的定位

- AMP 不是替代 PPO。  
- PPO 仍是主优化器；AMP 额外提供“动作风格一致性”约束。  

### 3.2 AMP 输入来源

policy 侧输入：

- `env.get_amp_observations()`，当前实现返回 `robot.data.joint_pos`（23 维）。  

expert 侧输入：

- `G1_AMPLoader` 从 `dataset/skate_push` 读取动作数据，构建多帧样本。  
- 训练配置默认 `amp_num_frames=5`，`amp_motion_files="dataset/skate_push"`。  

对应代码：

- `src/mjlab_husky/envs/g1_skate_rl_env.py::get_amp_observations`  
- `rsl_rl/utils/motion_loader_g1.py`  
- `src/mjlab_husky/rl/config.py`（AMP 配置项）  

### 3.3 AMP 输出与奖励注入

rollout 阶段：

- runner 调用 `discriminator.predict_amp_reward(...)` 产出 `reward, logit, disc_reward`。  
- 当前实现按 `contact_phase[:,0] == 1`（push 相位）才注入 AMP reward。  

update 阶段：

- 计算 expert/policy 判别器 MSE loss + grad penalty。  
- `amp_loss + grad_pen_loss` 直接加到 PPO 总 loss。  

对应代码：

- 注入点：`rsl_rl/runners/amp_on_policy_runner.py`  
- 联合优化：`rsl_rl/algorithms/amp_ppo.py`  
- 判别器结构：`rsl_rl/modules/discriminator_multi.py`  

### 3.4 AMP 训练闭环（简版）

1. 环境给 policy AMP 观测（多帧拼接）。  
2. policy AMP 轨迹进 replay buffer。  
3. expert loader 采样 expert AMP 批次。  
4. 判别器区分 policy/expert，生成风格奖励与判别损失。  
5. 判别器损失与 PPO 损失一起反向传播。  

---

## 第 4 层：理论流程 vs 本仓库实现（差异表）

| 主题 | 理论上常见理解 | 本仓库实际实现 | 影响 |
|---|---|---|---|
| AMP 奖励注入时机 | AMP 奖励全时段参与 | rollout 中仅 push 相位（`contact_phase[:,0]`）注入 | AMP 更聚焦 push 段风格约束 |
| AMP update 触发 | 始终参与 | `update()` 中用 `mask_push.any()` 决定 AMP 分支是否启用 | 某些 batch 可能无 AMP 梯度 |
| reward 组合 | 各奖励直接线性加和 | push/steer/transition 受 contact_phase 门控 | 奖励解释必须结合相位 |
| obs 组装 | “actor/critic 固定字段” | 由 `obs_groups` + manager 实时拼接；policy 含 5 帧历史 | 改观测会联动网络输入维度 |
| play 与 train 配置 | 默认一致 | `play=True` 时关闭 corruption、改 termination、去掉 push event | train/play 行为分布不同 |
| 训练产物 | 只保存 checkpoint | 保存 `model_*.pt`；wandb 模式下 runner 额外导出 ONNX | 续训与部署产物路径不同 |

---

## Public APIs / Interfaces（仅解释，不新增）

- 训练入口参数接口：`TrainConfig`（`src/mjlab_husky/scripts/train.py`）  
- 环境对算法接口：`get_observations` / `step` / `get_amp_observations`（`src/mjlab_husky/rl/vecenv_wrapper.py`）  
- runner 与算法接口：`learn` / `act` / `process_env_step` / `update`（`amp_on_policy_runner.py` + `amp_ppo.py`）  

---

## “改哪里会动到什么”最小路径

### 改 Obs

- 主要改：`src/mjlab_husky/tasks/skater/skater_env_cfg.py`（terms、history、noise）  
- 连锁影响：AC 输入维度、normalizer 统计、checkpoint 兼容性  

### 改 Reward

- 主要改：`src/mjlab_husky/tasks/skater/mdp/rewards.py` + `skater_env_cfg.py`（权重）  
- 连锁影响：contact_phase 门控下的实际有效区间  

### 改 AMP

- 主要改：`src/mjlab_husky/rl/config.py`、`g1_skate_rl_env.py::get_amp_observations`、`rsl_rl/modules/discriminator_multi.py`、`rsl_rl/algorithms/amp_ppo.py`  
- 连锁影响：expert/policy 维度对齐、reward 注入幅度、loss 稳定性  

### 改网络结构（AC）

- 主要改：`src/mjlab_husky/tasks/skater/config/g1/rl_cfg.py`（隐藏层、归一化开关）  
- 连锁影响：训练稳定性、收敛速度、旧模型加载兼容  

---

## Test Plan（读完后自检）

1. 结构一致性：能把 `train.py -> runner.learn -> alg.update` 主链完整复述。  
2. 数据流一致性：能说清 obs/action/reward/amp 四条流的输入输出。  
3. 差异核对：能指出至少 3 条“理论 vs 实现”差异及其影响。  
4. 可用性：能独立回答以下 5 个问题：  
   - 训练从哪开始？  
   - Actor/Critic 分别吃什么、产什么？  
   - reward 分几组、为什么这样分？  
   - AMP 的输入输出和注入点是什么？  
   - 最终保存了什么模型，如何续训？  

