# Task Plan: Analyze Project and Startup Path

## Goal
分析 `humanoid_skateboarding` 项目结构，并确认 `bash test_scene/sim.sh your-onnx-path` 是否是在直接运行导出的策略。

## Current Phase
Phase 5

## Phases
### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints and requirements
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Startup Chain Inspection
- [x] Read README and runtime entry points
- [x] Trace `test_scene/sim.sh -> test_scene/sim.py`
- [x] Identify model load path and input/output flow
- **Status:** complete

### Phase 3: RL Pipeline Inspection
- [x] Inspect training/export scripts
- [x] Map relationship between training checkpoint and ONNX export
- [x] Determine whether simulation runs policy directly or via wrapper
- **Status:** complete

### Phase 4: Verification & Synthesis
- [x] Cross-check conclusions against code
- [x] Summarize project modules and execution flow
- [x] Answer user question directly
- **Status:** complete

### Phase 5: Delivery
- [x] Deliver concise explanation with file references
- [ ] Note any assumptions or unverified runtime behavior
- **Status:** in_progress

## Key Questions
1. `test_scene/sim.sh` 具体启动了哪个 Python 入口？
2. 传入的 `your-onnx-path` 是不是被直接作为策略网络加载并参与每步动作推理？
3. 这个仓库的训练、导出、测试三部分分别由哪些模块负责？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用 planning files 记录分析过程 | 该任务需要多步追踪入口、配置和推理链路，避免结论遗漏 |
| 将 `sim.sh` 与 `play` 区分为两条推理路径 | 前者直接跑导出的 ONNX，后者通过 mjlab 环境和 runner 加载 checkpoint |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |

## Notes
- 优先确认运行链路，再总结项目整体结构
- 如果需要，用样例模型 `ckpts/test.onnx` 作为辅助证据
