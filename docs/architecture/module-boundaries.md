# 模块边界与依赖方向

本文记录 `agent-visual-context` 的目录结构、模块职责、依赖方向与降级原则，是后续所有检测器、
关系推理器和输入源接入时必须遵守的边界约定。对应 Issue #9（PoC-0 项目骨架），
关系生命周期与两类快照服务在 Issue #4（PoC-1 Mock 视觉时间线）中补充，
离线图片/视频输入适配在 Issue #5（PoC-2）中补充，
YOLO-World 目标检测适配器在 Issue #6（PoC-3）中补充，
RelateAnything 关系推理适配器在 Issue #7（PoC-4）中补充，
本地摄像头实时链路与 Sidecar 循环在 Issue #8（PoC-5）中补充。
人体检测人数摘要与查询契约在 Issue #16 中补充。

## 目录结构

```text
src/agent_visual_context/
├── domain/       # Frame、Detection、TrackedObject、Relation、Observation、Event、Snapshot
├── input/        # 图片、视频、摄像头与合成数据源接口及 Mock 实现
├── perception/   # Detector、Tracker、RelationReasoner 抽象及 Mock 实现
├── temporal/     # 去抖、持续时间、TTL、有界时间线
├── context/      # 场景摘要与时间窗口快照
├── policies/     # 迎宾候选等规则；只产出候选事件，不执行业务动作
├── runtime/      # 流水线编排、事件总线、生命周期与降级状态；实时 Sidecar 循环、有界帧缓冲与运行指标
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
| 输入源 | `input/base.py::FrameSource` | `ScriptedFrameSource` | `ImageFrameSource`、`VideoFrameSource`（PoC-2）；`CameraSource`（PoC-5 已落地，`input/camera.py`） |
| 目标检测 | `perception/base.py::Detector` | `StaticSceneDetector`、`ScriptedDetector` | `YoloWorldDetector`（PoC-3 已落地，`perception/yolo_world.py`） |
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

### YOLO-World 目标检测（PoC-3）

真实检测器与 Mock 满足同一 `Detector` 协议，接入时遵守以下约定（详见 `docs/research/yolo-world.md`）：

- **落点唯一**：模型调用只能出现在 `perception/yolo_world.py`（及惰性加载器
  `perception/_ultralytics.py`）；**不得**写入 `domain/`、`temporal/`、`context/`。
- **装配**：`runtime/factory.py::build_detector(config)` 按 `detector_backend` 装配
  （`mock` -> `StaticSceneDetector`，`yolo-world` -> `YoloWorldDetector.from_config`）；
  `build_mock_pipeline`/`build_offline_pipeline` 在未显式注入 `detector` 时调用它。
  CLI `run --detector yolo-world` 走同一路径。
- **可插拔后端**：`YoloWorldDetector` 通过 `backend`/`backend_factory` 注入推理后端，
  真实实现 `UltralyticsYoloWorldBackend` 惰性加载 ultralytics；测试注入假后端即可离线验证，
  无需安装 torch 或联网下载权重。
- **依赖隔离**：ultralytics 通过 `perception/_ultralytics.py::require_ultralytics()` 惰性加载，
  作为 `yolo-world` extra（AGPL-3.0），包导入不强制依赖；权重二进制不入仓库，部署时单独分发。
- **归一化**：后端把模型张量转换为纯 Python 的 `RawDetection`，检测器再映射为领域 `Detection`
  （bbox 裁剪到帧内、退化框丢弃、置信度过滤、时间戳取自当前帧、携带模型版本）；
  torch/ultralytics 类型不得进入 `domain/`。
- **降级**：模型初始化失败、推理异常、推理超时、缺少像素数据都抛 `PerceptionError`，
  由流水线降级为“该帧无检测”并标记 `detector` 组件状态；空检测结果是正常空列表，不算失败。
- **可观测**：每帧记录模型版本、推理耗时、输入尺寸、检测数量与类别分布（logger
  `perception.yolo_world`），并累积到 `YoloWorldDetector.stats`。

### 本地摄像头实时链路（PoC-5）

摄像头是**无限实时流**，与离线源"跑到耗尽"的语义不同。实时链路的落点与约定如下：

- **落点唯一**：摄像头采集只在 `input/camera.py::CameraSource`；实时采集/消费循环只在
  `runtime/live.py::LiveRuntime`；有界帧缓冲在 `runtime/buffer.py::LatestFrameBuffer`；
  运行指标在 `runtime/metrics.py::LiveMetrics`。**摄像头循环不得写进模型适配器或 CLI**。
- **装配**：`runtime/factory.py::build_camera_source()` 与 `build_live_runtime()` 是唯一装配口径，
  CLI `avc camera`、示例与测试都通过工厂构造；不要在业务代码里直接实例化 `CameraSource`/`LiveRuntime`。
- **可插拔后端**：`CameraSource` 通过 `backend`/`backend_factory` 注入 `CameraBackend`，
  真实实现 `OpenCVCameraBackend` 惰性加载 cv2（`vision` extra）；测试注入假后端即可离线验证，
  无需真实摄像头。设备选择/分辨率/采样频率取自 `config.camera_*` 与 `target_fps`。
- **生命周期**：直接复用 `AbstractFrameSource` 的 `open()`/`close()`（即启动/停止采集）与上下文管理器；
  `LiveRuntime.start()/stop()` 管理后台采集线程与设备，二者均幂等。
- **采集/消费解耦**：后台采集线程（daemon）持续 `read()` 写入 `LatestFrameBuffer`（有界、丢旧保最新），
  主消费循环取最新帧交给 `Pipeline.process_frame()`；高帧率采集与低频（1~2 FPS）推理互不阻塞，
  缓冲永不无限堆积。低频采样由 `CameraSource` 按 `target_fps` 节流（`sleeper` 可注入便于测试）。
- **降级不阻塞宿主**：摄像头打不开、读帧连续失败（达 `live_max_capture_failures`）、模型异常/超时都只降级，
  `LiveRuntime.run()` **绝不向上抛出**，返回 `degraded` 状态，确保不阻塞大屏渲染、麦克风采集、
  语音 WebSocket 与 TTS 播放；推理超时由各适配器 `*_timeout_seconds` 与组件级降级覆盖，
  消费循环再对兜底异常做一次捕获，防止实时循环因单帧崩溃退出。
- **可观测**：`LiveMetrics` 累积有效 FPS、端到端延迟（min/max/mean）、丢帧率与模型失败率，
  周期性写结构化日志（logger `runtime.live`）；CPU/内存（`cpu_percent`/`rss_mb`）为**预留字段**，
  由目标设备真机压测脚本填充——真机实测与 PoC 结论在后续 Issue 完成，本模块不引入 `psutil` 等额外依赖。
- **健康查询**：`api/queries.py::VisualContextApi.from_live_runtime()` 暴露 `get_health()`（组件+整体状态）
  与 `get_metrics()`，是宿主链路判断视觉是否可用的唯一口径；视觉降级时宿主读到 `degraded` 即可跳过视觉增强，
  无需等待或阻塞。所有输出仍是视觉辅助观察，不触发任何业务写操作。
- **可视化落点**：逐帧画框/关系线/中文上下文面板只存在于 `examples/run_camera_visual.py`，属于**示例层**，
  不进入 `src/`。它通过 `Pipeline.process_frame()` 返回的 `FrameResult.detection_items/tracked_items/
  relation_items` 拿到本帧可绘制对象（单次推理，不重复调用适配器）；中文渲染依赖 Pillow + 系统中文字体
  （`viz` extra，仅示例使用），缺失时降级为 ASCII 面板。生产实时链路仍走 `LiveRuntime`，示例为前台单循环
  以便逐帧绘制（cv2 窗口须在主线程）。

## 数据分型

### 人数摘要（Issue #16）

`PersonSceneSummary` 是单帧人体检测的有界视觉摘要，独立于关系观察。`Pipeline` 仅对通过
`min_detection_confidence` 的 `person` 检测计数；背对镜头但人体可检测时仍计入。
`current_person_count` 是最新有效采样帧的人数，绝不跨帧相加。检测成功但未见人时为 `0`，
`quality=no-detection` 表示“本帧未检出”，不保证现场无人；检测器失败时人数为 `null`，
`quality=degraded`，并覆盖之前的计数。有人检出时 `quality=detected`，`confidence` 为
本帧人体检测置信度的最小值；空检测或降级时置信度为 `null`。
未配置 `person` 检测类别时人数为 `null`、`quality=unavailable`，不把未检测当成 0。

摘要带 `scene_id`、`source_id`、`sampled_at`、`expires_at`，有效期受时间线默认 TTL 限制。
`recognizable_face_count` 在未接入人脸分析时为 `null`，绝不使用人体数或 `0` 代替。
`window_distinct_person_count` 在可靠跨帧跟踪能力接入前为 `null`：当前 `MockTracker` 的
类别顺序 ID 不能证明跨帧同一人，不能用于窗口去重。时间线最多保存 `timeline_capacity` 条
人数样本，快照只暴露窗口内最新且在查询时刻未过期的一条；无有效样本时 `persons=null`。
`get_scene_snapshot()` 和 `get_turn_context()` 都使用这一字段，过期样本不会成为当前场景。
这些人数是视觉估计，不是身份或业务事实，也不直接触发业务动作。

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
