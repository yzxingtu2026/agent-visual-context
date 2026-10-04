# 模块边界与依赖方向

本文记录 `agent-visual-context` 的目录结构、模块职责、依赖方向与降级原则，是后续所有检测器、
关系推理器和输入源接入时必须遵守的边界约定。对应 Issue #9（PoC-0 项目骨架），
关系生命周期与两类快照服务在 Issue #4（PoC-1 Mock 视觉时间线）中补充，
离线图片/视频输入适配在 Issue #5（PoC-2）中补充。

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
├── fixtures/     # 固定离线素材（一张 PNG 图片与一段 2s MP4 短视频）
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
| 输入源 | `input/base.py::FrameSource` | `ScriptedFrameSource` | `ImageFrameSource`、`VideoFrameSource`（PoC-2 已落地）；摄像头适配器（PoC-5） |
| 目标检测 | `perception/base.py::Detector` | `StaticSceneDetector`、`ScriptedDetector` | YOLO-World 适配器（PoC-3） |
| 目标跟踪 | `perception/base.py::Tracker` | `MockTracker` | IoU/外观匹配跟踪器（PoC-3） |
| 关系推理 | `perception/base.py::RelationReasoner` | `MockRelationReasoner` | RelateAnything 适配器（PoC-4） |
| 策略 | `policies/base.py::Policy` | `GreetingCandidatePolicy` | 迎宾状态机与冷却策略 |

`tests/contract/` 中的契约测试以协议类型标注函数签名，任何新适配器都应当能直接替换 Mock 通过同一组测试。

### 离线素材输入（PoC-2）

图片与视频适配器统一遵守以下约定，摄像头适配器（PoC-5）接入时应当复用同一套语义：

- **装配**：CLI、示例与测试通过 `input/loader.py::frame_source_from_path()` 把素材路径
  （单张图片、图片目录或视频文件）转换为 `FrameSource`，再用
  `runtime/factory.py::build_offline_pipeline()` 注入流水线；不要在业务代码中直接实例化适配器。
- **编号与时间戳**：`frame_id = "{source_id}-{原始帧编号:05d}"`；视频帧
  `captured_at = start + 原始帧号 / 源帧率`，图片序列按 `1 / target_fps` 等间隔推进。
- **锚定**：未显式给定 `start` 时，以“素材末帧即打开时刻”锚定回放（与合成帧一致），
  保证离线素材落入默认时间窗口。
- **采样**：`input/sampling.py::plan_sampling()` 是抽帧的唯一口径：源帧率高于 `target_fps`
  时等间隔取整（首帧必选、编号不重复），否则逐帧全选；视频按原始帧号顺序读取并跳过未采样帧，
  不使用 seek，避免不同编码的兼容性问题。
- **依赖隔离**：OpenCV 通过 `input/_opencv.py::require_cv2()` 惰性加载，包导入不强制依赖
  cv2；numpy 像素数组放入 `Frame.data`（`RawFrameData`），OpenCV 类型不得进入 `domain/`。
- **降级**：素材不存在、目录无图片、解码失败与空视频在 `open()`/`read()` 抛出
  `FrameSourceError`；流水线将输入源读取失败降级为“无帧”并记录组件状态，`fail_fast` 时上抛。

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

### 关系生命周期（开始 / 持续 / 结束）

`PersistenceGate.update_with_lifecycle()` 是关系生命周期的唯一口径，返回 `GateUpdate`：

- **开始 / 持续**：某关系键（subject+predicate+target）命中累积到 `min_hits`
  且首末命中间隔达到 `min_seconds` 后进入 `stable`，此时才被流水线提升为 `Observation`；
- **短时丢帧容错**：相邻命中间隔只要不超过 `max_gap_seconds`，已累积的命中与首次时间
  不被重置，关系仍可继续提升；
- **结束**：曾经稳定（`promoted`）的关系超过 `max_gap_seconds` 未再命中时被遗忘，
  并以结束前的最后一条证据进入 `ended`；从未成形即消失的抖动不产生 `ended`，避免误报。

`update()` 是 `update_with_lifecycle().stable` 的便捷别名，保持既有调用方不变。

### 两类快照服务

`context/` 提供两个互补且**只读**时间线的快照服务，Agent 侧不得自行重复实现时间线查询：

- `SceneSummarizer`：以“当前时刻往前 `window_seconds`”生成**当前场景摘要**，会先 `prune()`；
- `WindowSnapshotBuilder`：按**调用方显式给定的 `[start, end]` 窗口**生成快照（例如某次语音
  话轮的起止），同样做去重与过期过滤但不修改时间线。`api/queries.py::get_turn_context()`
  复用它，避免在 API 层内联时间线逻辑。


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
