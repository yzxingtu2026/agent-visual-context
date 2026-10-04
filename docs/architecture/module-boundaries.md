# 模块边界与依赖方向

本文记录 `agent-visual-context` 的目录结构、模块职责、依赖方向与降级原则，是后续所有检测器、
关系推理器和输入源接入时必须遵守的边界约定。对应 Issue #9（PoC-0 项目骨架）。

## 目录结构

```text
src/agent_visual_context/
├── domain/       # Frame、Detection、TrackedObject、Relation、Observation、Event、Snapshot
├── input/        # 图片、视频、摄像头与合成数据源接口及 Mock 实现
├── perception/   # Detector、Tracker、RelationReasoner 抽象及 Mock 实现
├── temporal/     # 去抖、持续时间、TTL、有界时间线
├── context/      # 场景摘要与时间窗口快照
├── policies/     # 迎宾候选等规则；只产出候选事件，不执行业务动作
├── runtime/      # 流水线编排、事件总线、生命周期与降级状态
├── api/          # Agent 查询/订阅接口边界
├── config.py     # 统一配置模型（环境变量前缀 AVC_）
├── logging_setup.py  # 统一日志配置
├── errors.py     # 统一错误类型
└── cli.py        # `avc` 命令行入口

tests/
├── unit/         # 领域模型、时间线、去抖、配置等纯逻辑测试
├── contract/     # 组件协议契约测试：Mock 与真实适配器都必须满足
└── integration/  # 流水线端到端、API 查询、CLI 行为

docs/
├── architecture/ # 本文所在的架构与边界说明
├── api/          # 对外接口文档（后续补充）
├── decisions/    # 架构决策记录（后续补充）
└── research/     # 模型与方案调研（后续补充）

examples/         # 可运行的最小示例
```

## 依赖方向

依赖只能由外向内，内层不得反向依赖外层：

```text
cli / api  ->  runtime  ->  context / policies / temporal  ->  perception / input  ->  domain
                    \______________________ config / logging_setup / errors ______________________/
```

硬约束：

- `domain` 只依赖标准库与 Pydantic，**不得**引用 OpenCV、YOLO-World、RelateAnything 或任何 Agent 框架；
  图像等不可序列化的原始数据通过 `RawFrameData`（对领域层透明）承载。
- `input`、`perception` 的具体实现只能出现在各自模块的适配器文件中，并通过协议注入 `runtime`；
  真实模型接入（PoC-3 / PoC-4）必须新增适配器，而不是修改流水线。
- `policies` 只消费 `Snapshot` 并产出 `Event`，不得直接调用业务后端、TTS 或大屏渲染。
- `api` 是唯一对 Agent 暴露的入口，不得绕过时间线直接读取模型输出。

## 可替换组件协议

| 组件 | 协议位置 | Mock 实现 | 真实实现（后续） |
| --- | --- | --- | --- |
| 输入源 | `input/base.py::FrameSource` | `ScriptedFrameSource` | 图片/视频/摄像头适配器（PoC-2、PoC-5） |
| 目标检测 | `perception/base.py::Detector` | `StaticSceneDetector`、`ScriptedDetector` | YOLO-World 适配器（PoC-3） |
| 目标跟踪 | `perception/base.py::Tracker` | `MockTracker` | IoU/外观匹配跟踪器（PoC-3） |
| 关系推理 | `perception/base.py::RelationReasoner` | `MockRelationReasoner` | RelateAnything 适配器（PoC-4） |
| 策略 | `policies/base.py::Policy` | `GreetingCandidatePolicy` | 迎宾状态机与冷却策略 |

`tests/contract/` 中的契约测试以协议类型标注函数签名，任何新适配器都应当能直接替换 Mock 通过同一组测试。

## 数据分型

三类信息必须分型，视觉观察不得伪装成用户意图或业务事实：

1. `Observation`（`epistemic_status = visual-observation`）：模型看到或推理出的内容，带置信度与有效期；
2. `Event`（`epistemic_status = rule-event`）：由策略从观察派生的候选事件，例如迎宾候选；
3. `user-assertion`：来自语音或触控的用户明确表达，由 Agent 侧写入，本项目不生成。

每条观察必须包含：`scene_id`、`observed_at`、`expires_at`、`confidence`、`source_id`、`model_version`。

## 时间与有界性

- 时间线 `BoundedTimeline` 同时受**容量**与 **TTL** 约束：超量丢最旧，超期由 `prune()` 清理；
- 写入时间线时，`expires_at` 不得超过 `observed_at + default_ttl`，避免出现无限期条目；
- 单帧关系不直接进入时间线：`PersistenceGate` 要求最小命中次数、最小持续时间，
  并在超过最大间隔时重置计数，用于抑制抖动；
- 快照 `Snapshot` 始终限定在配置窗口内，并做去重（同一 subject+predicate+target 只保留最新一条）。

## 降级原则

视觉处理必须可降级，不得阻塞大屏渲染、音频采集、WebSocket 或 TTS 链路：

- 组件级异常在 `Pipeline._call()` 内被捕获，记录到 `PipelineStatus.components[*]`（`degraded` / `failed`），
  该帧对应环节退化为空结果，流水线继续处理后续帧；
- 输入源读取失败时降级为“无帧”，不抛异常；
- `EventBus` 隔离订阅者异常，单个订阅者崩溃不影响流水线；
- 配置 `fail_fast = true`（`AVC_FAIL_FAST`）时改为抛出 `PerceptionError`，仅用于调试与测试；
- `PipelineState` 与 `ComponentState` 是对外唯一的状态口径，CLI、日志与 Agent 都应读取它，
  不要另建一套状态字段。

## 开发约定

- **配置**：所有可调参数集中在 `config.py::AppConfig`，环境变量前缀 `AVC_`；
  禁止在模块内直接读 `os.environ`。
- **日志**：运行时日志统一使用 `logging_setup.get_logger(__name__)`；
  Ruff 规则 `T20` 会在 `src/` 中拦截 `print`，CLI 与 `examples/` 中的终端报告输出除外。
- **错误**：新增异常必须继承 `AgentVisualContextError`，可恢复异常由 `runtime` 转降级，
  不可恢复异常由 CLI 转退出码（配置错误 `2`，运行失败 `1`，中断 `130`）。
- **装配**：组件接线统一放在 `runtime/factory.py`，CLI、示例与测试都通过工厂构造流水线；
  需要替换组件时用参数注入，不要在业务代码中直接实例化具体实现。
- **时间**：领域层通过 `utc_now()` 取时间，流水线与时间线支持注入 `clock`，测试必须使用 `FakeClock`。
