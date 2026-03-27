"""任务注册系统：用于管理环境任务的注册与读取。"""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg
  play_env_cfg: ManagerBasedRlEnvCfg
  rl_cfg: RslRlOnPolicyRunnerCfg
  runner_cls: type | None


# 模块级私有注册表：task_id -> 任务配置。
_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """注册一个环境任务。

  参数:
    task_id: 任务唯一标识（例如 "Mjlab-Velocity-Rough-Unitree-Go1"）。
    env_cfg: 训练阶段使用的环境配置。
    play_env_cfg: play 阶段使用的环境配置。
    rl_cfg: RL 训练器配置。
    runner_cls: 可选的自定义 runner 类；若为 None，则使用默认 OnPolicyRunner。
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """返回所有已注册任务的 task_id 列表。"""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """加载指定任务的环境配置。

  返回深拷贝，避免外部修改污染注册表中的原始配置。
  """
  return deepcopy(
    _REGISTRY[task_name].env_cfg if not play else _REGISTRY[task_name].play_env_cfg
  )


def load_rl_cfg(task_name: str) -> RslRlOnPolicyRunnerCfg:
  """加载指定任务的 RL 配置。

  返回深拷贝，避免外部修改污染注册表中的原始配置。
  """
  return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """加载指定任务绑定的 runner 类。

  若返回 None，则后续流程会使用默认 OnPolicyRunner。
  """
  return _REGISTRY[task_name].runner_cls
