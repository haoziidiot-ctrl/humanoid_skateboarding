# Findings & Decisions

## Requirements
- 用户要一个项目分析，重点是确认 `bash test_scene/sim.sh your-onnx-path` 是否直接运行策略。
- 需要给出项目主要模块、启动链路和策略加载方式。

## Research Findings
- 仓库根目录包含 `test_scene/`、`src/mjlab_husky/`、`rsl_rl/`、`ckpts/`、`dataset/`。
- `test_scene/sim.sh` 和 `test_scene/sim.py` 很可能是最直接的测试入口。
- 仓库同时包含训练脚本 `src/mjlab_husky/scripts/train.py`、播放脚本 `src/mjlab_husky/scripts/play.py` 和导出相关代码 `src/mjlab_husky/tasks/skater/rl/exporter.py`。
- `README.md` 明确把 `bash test_scene/sim.sh your-onnx-path` 描述为 “lite MuJoCo simulation script for evaluation”。
- `test_scene/sim.sh` 只做参数检查，然后执行 `uv run python test_scene/sim.py --xml ... --policy <onnx> --device cuda --policy_frequency 50`。
- `test_scene/sim.py` 在 `RealTimePolicyController.__init__` 中调用 `load_onnx_policy(policy_path, device)` 创建 `onnxruntime.InferenceSession`。
- `test_scene/sim.py` 的主循环会从 MuJoCo 状态拼装观测 `obs_buf`，随后执行 `raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()`。
- 推理得到的 `raw_action` 会经过 `action_scale` 和默认关节位姿偏置，形成 PD target，再写入 `self.data.ctrl[:-7]` 驱动仿真。
- 这说明 `sim.sh` 对应的是“直接加载 ONNX 策略做逐步推理，并把输出动作送入 MuJoCo 控制器”的评估流程，不是训练脚本，也不是通过完整 mjlab 环境播放器运行。
- `src/mjlab_husky/scripts/play.py` 的 `play` 流程与 `test_scene/sim.py` 不同：它加载训练 checkpoint，构建 `G1SkaterManagerBasedRlEnv + RslRlVecEnvWrapper`，再从 runner 获取 inference policy。
- `src/mjlab_husky/scripts/train.py` 负责训练，并通过 `load_runner_cls(task_id)` 构建 runner。
- `src/mjlab_husky/tasks/skater/rl/runner.py` 的 `SkaterOnPolicyRunner.save()` 在保存训练模型时会额外导出 ONNX 文件。
- `src/mjlab_husky/tasks/skater/rl/exporter.py` 使用 `_OnnxPolicyExporter` 把 actor-critic 导出为 ONNX。
- `src/mjlab_husky/tasks/skater/config/g1/__init__.py` 把任务 `Mjlab-Skater-Flat-Unitree-G1` 注册到任务表，并指定 `runner_cls=SkaterOnPolicyRunner`。
- `pyproject.toml` 将 `train` 和 `play` 暴露为命令行脚本，分别指向 `mjlab_husky.scripts.train:main` 与 `mjlab_husky.scripts.play:main`。
- `src/mjlab_husky/tasks/skater/skater_env_cfg.py` 中的 policy 观测项依次是 `command / heading / base_ang_vel / projected_gravity / joint_pos / joint_vel / actions / phase`，且 `history_length=5`。
- `test_scene/sim.py` 的 `obs_proprio` 和 `history_len=5` 与训练环境的 policy observation 结构一致，说明它是在手工复现 actor 的输入接口。
- `skater_env_cfg.py` 中动作类型是 `JointPositionActionCfg(..., use_default_offset=True)`，而 `sim.py` 里也把网络输出乘以 `action_scale` 后加上默认关节位姿，作为 PD target 写入控制量。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 先追踪测试入口，再回看训练和导出链路 | 用户问题聚焦“这个命令是否直接运行策略”，入口链路是核心证据 |
| 将 `test_scene/sim.py` 视为独立轻量评估器，而不是训练环境的一部分 | 代码直接操作 MuJoCo 和 ONNXRuntime，没有经过 `play.py` 的 runner/env 封装 |
| 将 `sim.py` 认定为“同一策略的轻量复刻推理端” | 它复用了训练时的 policy 观测结构、历史长度和 joint position action 语义 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `session-catchup.py` 没有返回额外上下文 | 继续以当前仓库状态为准开展分析 |

## Resources
- `/home/jerry/humanoid_skateboarding/test_scene/sim.sh`
- `/home/jerry/humanoid_skateboarding/test_scene/sim.py`
- `/home/jerry/humanoid_skateboarding/README.md`
- `/home/jerry/humanoid_skateboarding/src/mjlab_husky/scripts/train.py`
- `/home/jerry/humanoid_skateboarding/src/mjlab_husky/scripts/play.py`
- `/home/jerry/humanoid_skateboarding/src/mjlab_husky/tasks/skater/rl/runner.py`
- `/home/jerry/humanoid_skateboarding/src/mjlab_husky/tasks/skater/rl/exporter.py`
- `/home/jerry/humanoid_skateboarding/src/mjlab_husky/tasks/skater/skater_env_cfg.py`
- `/home/jerry/humanoid_skateboarding/src/mjlab_husky/tasks/skater/config/g1/__init__.py`
- `/home/jerry/humanoid_skateboarding/pyproject.toml`

## Visual/Browser Findings
- 本任务未使用浏览器或图像分析。
