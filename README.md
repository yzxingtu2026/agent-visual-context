# 面向 Agent 的实时视觉关系上下文处理器

面向 Agent 的实时视觉关系上下文处理器，将摄像头画面转换为**带时间、置信度、有效期和来源的结构化视觉观察**，为现代 Agent 提供可查询、可订阅、可注入话轮的实时视觉上下文。

本项目的首个真实应用场景是阿国饭店 AI 点菜大屏：利用本地摄像头低频观察顾客与大屏、手机、手部和交互区域之间的关系，实现视觉辅助主动迎宾，以及在语音点菜过程中提供连续的辅助观察上下文。详见阿国饭店项目 [Issue #446](https://github.com/yzxingtu2026/aguo-ai-dining/issues/446)。

> 本项目输出的是视觉观察，不是用户指令、身份事实或业务事实。任何订单、支付、服务请求等业务动作都必须经过用户明确表达、Agent 安全策略和业务后端权威校验。

## 快速开始：主要演示入口

想最快看到整条链路跑起来，直接运行**主要演示入口** `examples/run_camera_visual.py`：摄像头实时画面
经检测 -> 跟踪 -> 关系推理 -> 时间线/摘要后，叠加目标检测框、主宾语关系连线与右下角中文上下文面板
（如「人拿着手机」「人看着屏幕」）。

```bash
# 一次装齐演示所需依赖（uv extras 不叠加，需一次列全）
uv sync --extra vision --extra viz --extra yolo-world --extra relate-anything

# 办公室场景（笔记本自带摄像头）：真实检测 + 真实关系推理 + 中文上下文
uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \
    --classes person,hand,phone,laptop --relations holding,looking_at,touching,near
```

窗口内 `q`/`ESC` 退出、`s` 保存当前帧；无摄像头或无显示器时加 `--input 图片/视频 --save out.png`
即可用离线素材跑同一套可视化。更多场景类别组合、关系词与参数见下文「摄像头全链路可视化」小节。

## 为什么需要这个项目？

单帧目标检测只能回答“画面里有什么”，而 Agent 往往还需要知道：

- 谁正在看向或接近什么？
- 谁拿着什么、指向什么或触摸什么？
- 某个关系是否持续了几秒？
- 最近一个语音话轮期间观察到了哪些变化？
- 当前观察是否仍然有效，是否只是低置信度推断？

本项目把模型推理结果加工成 Agent 可以安全使用的上下文：

```text
摄像头
  -> 目标检测/分割
  -> 目标跟踪
  -> 视觉关系推理
  -> 跨帧去抖与关系持续性
  -> 有界视觉事件时间线
  -> 场景摘要 / 话轮快照
  -> Agent 查询、订阅或上下文注入
```

## 核心模型组合

首期计划采用：

- **YOLO-World 或其他可替换目标检测器**：发现人、手、手机、屏幕、物品等目标并提供目标框；分割模型可选，用于需要 mask 的场景。
- **RelateAnything**：基于图像和目标区域预测开放词汇的目标关系，例如 `person -> holding -> phone`、`person -> looking at -> screen`。
- **跟踪与上下文引擎**：为目标分配跨帧 track id，合并短时观察，处理置信度、TTL、去抖、场景和话轮边界。

RelateAnything 主要负责视觉关系推理，并不替代目标检测、目标跟踪或 Agent 上下文处理。模型适配层会保持可替换，以便在不同设备、许可证和性能条件下切换检测器与推理后端。

## 面向 Agent 的能力

计划提供最小而稳定的通用接口：

```text
get_scene_snapshot()                 获取当前场景摘要
get_observations(since, until)       查询时间窗口内的观察
subscribe_events()                   订阅视觉事件
get_turn_context(turn_id)             获取某个语音话轮的视觉快照
```

每条观察应尽量包含：

```json
{
  "scene_id": "scene-01",
  "observed_at": "2026-10-04T12:00:01.200Z",
  "expires_at": "2026-10-04T12:00:04.200Z",
  "subject": {"track_id": "person-01", "label": "person"},
  "predicate": "looking_at",
  "object": {"track_id": "screen-01", "label": "screen"},
  "confidence": 0.82,
  "source": "local-camera",
  "epistemic_status": "visual-observation"
}
```

输出需要区分：

1. **模型观察**：当前画面中检测到或推理出的目标与关系；
2. **规则事件**：例如“持续停留超过 2 秒”“关系在最近 3 秒重复出现”；
3. **用户明确表达**：由语音或触控确认的意图，不能被视觉观察覆盖。

## 阿国饭店大屏首个应用

### 视觉辅助主动迎宾

```text
检测到顾客
  -> 顾客进入大屏交互区域
  -> 持续停留并大致面向大屏
  -> 满足阈值和冷却条件
  -> 数字人主动打招呼
  -> 顾客回应后进入正式双工语音点菜
```

视觉触发只负责提出主动迎宾候选，不应因顾客路过或短暂停留反复播报，也不应在已有语音会话、维护页或不适合打扰的页面中触发。

### 长期辅助观察上下文

在语音倾听和点菜过程中，处理器以低频方式维护最近时间窗口，例如：

```text
最近 4 秒观察到：
- 2 位顾客持续位于大屏交互区域；
- 其中 1 位面向菜单区域；
- 检测到 1 部手机；
- 未检测到明确指向某个菜品的动作。
```

提交语音话轮时，只发送经过截断、去重和 TTL 过滤的视觉快照，不发送无界原始视频或完整逐帧结果。

## 设计原则

- **本地优先**：摄像头画面默认在设备本地处理，减少不必要的视频上传。
- **低频优先**：面向普通 CPU 设备，优先验证 640p、1–2 FPS 关系推理，再按设备能力调节。
- **模型可替换**：检测器、跟踪器、关系推理后端和设备执行提供方通过适配层解耦。
- **观察不等于事实**：任何视觉结果都带来源、置信度和有效期，不能伪装成用户确认或后端事实。
- **语音优先**：用户明确说出的内容优先于视觉估计；视觉只作为辅助上下文。
- **业务后端权威**：视觉观察不得直接执行下单、加菜、催菜、呼叫服务或其他业务写操作。
- **资源隔离**：视觉推理异常、超时或退出时，不得阻塞大屏渲染、麦克风采集、语音 WebSocket 和 TTS 播放。
- **可观测可回放**：记录模型版本、推理延迟、丢帧、置信度和事件生命周期，支持现场问题复盘。

## 项目阶段

当前处于项目启动与技术方案阶段：

- [x] 从 `agent-project-skeleton` 模板初始化项目；
- [x] 确立与阿国饭店大屏 Issue #446 的首个应用关系；
- [x] 建立 Python 项目骨架、核心抽象接口、测试基线与最小 CLI（Issue #9）；
- [x] 用 Mock 检测/关系结果跑通视觉时间线（Issue #4）；
- [x] 完成图片/短视频最小闭环：统一帧输入 -> Mock 检测/关系 -> 时间线（Issue #5）；
- [x] 接入 YOLO-World 目标检测适配器：可配置类别/阈值/设备，模型不可用时降级（Issue #6）；
- [x] 接入 RelateAnything 关系推理适配器：可配置关系词/阈值/设备，模型不可用时降级（Issue #7）；
- [x] 接入本地摄像头实时链路：CameraSource + 有界保最新帧队列 + 实时 Sidecar 循环 + 健康/指标（Issue #8）；
- [x] 摄像头全链路可视化主要演示入口：检测框 + 关系连线 + 中文上下文面板，可自配类别/关系词（Issue #8）；
- [ ] 在目标 Windows 10 一体机上执行摄像头真机实测、资源隔离验证与 PoC 结论（Issue #8 后续）；
- [ ] 设计视觉观察数据模型和 Agent API；
- [ ] 实现主动迎宾候选事件与冷却状态机；
- [ ] 接入阿国饭店双工语音上下文；
- [ ] 发布可复用的 Agent Skills 适配包。

## 本地开发与 PoC 运行

项目使用 Python 3.12 与 [uv](https://docs.astral.sh/uv/) 管理依赖，正式代码位于 `src/agent_visual_context/`。

### 环境准备

```bash
uv sync                     # 创建 .venv，安装项目与开发依赖（pytest / ruff / mypy / opencv）
uv sync --extra vision      # 运行环境需要 OpenCV 图片/视频输入时安装
uv sync --extra yolo-world  # 需要真实 YOLO-World 检测时安装（引入 torch 等重依赖，AGPL-3.0，需联网下载权重）
uv sync --extra viz         # 摄像头可视化示例的右下角中文面板需要（Pillow + 系统中文字体）
# 主要演示入口一次装齐（extras 不叠加，需一次列全）：
uv sync --extra vision --extra viz --extra yolo-world --extra relate-anything
```

### 验证基线

```bash
uv run pytest                 # 单元 / 契约 / 集成测试
uv run ruff check .           # 静态检查
uv run ruff format --check .  # 格式检查
uv run mypy                   # 类型检查（strict，覆盖 src 与 tests）
```

### 主要演示入口：摄像头全链路可视化（PoC-5）

`examples/run_camera_visual.py` 是本项目的主要演示入口，把整条链路跑在一张实时画面上：摄像头帧经
检测（可切 YOLO-World）-> 跟踪 -> 关系推理（可切 RelateAnything）-> 时间线/摘要后，在画面上叠加
每个目标的检测框（类别 + track_id + 置信度）、主宾语之间的关系连线，以及右下角的中文上下文面板
（链路状态、有效 FPS、目标/关系/观察/事件计数，以及「人拿着手机」「人看着屏幕」这类自然中文观察句）。

```bash
uv sync --extra vision --extra viz                 # 摄像头 + 中文面板（Pillow）
uv run python examples/run_camera_visual.py        # 默认设备 0，Mock 检测/关系，q/ESC 退出、s 存帧
uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything
# 自配检测类别与关系词（逗号分隔）：
uv run python examples/run_camera_visual.py --classes person,phone,screen \
    --relations looking_at,holding,pointing_at,near
# 无摄像头/无显示器时用离线素材跑同一套可视化并保存标注帧：
uv run python examples/run_camera_visual.py --input tests/fixtures/sample_detection.png --save out.png
```

办公室场景（笔记本自带摄像头）常用类别组合，从轻量到全量（类别越多单帧越慢，CPU 上建议 `--fps 1`）：

```bash
# 人手与手持设备（最轻量，最适合看「人拿着手机」效果）
uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \
    --classes person,hand,phone,laptop --relations holding,looking_at,touching,near
# 桌面物品
uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \
    --classes person,laptop,keyboard,mouse,cup,bottle --relations using,looking_at,touching,next_to,near
# 会议/协作
uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \
    --classes person,chair,table,laptop,screen,whiteboard --relations sitting_on,facing,looking_at,pointing_at,near
# 阅读/书写
uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \
    --classes person,book,notebook,paper,pen,phone --relations holding,reading,looking_at,typing_on,near
```

中文面板依赖 Pillow 与系统中文字体（`viz` extra）；OpenCV 的 `cv2.putText` 无法渲染中文，缺字体或
未装 Pillow 时面板自动降级为 ASCII，检测框与关系连线照常绘制。生产实时链路仍用 `avc camera` /
`run_camera_live.py`（采集/消费解耦的 Sidecar 循环）；本示例为前台单循环以便逐帧绘制（cv2 窗口须在主线程）。

### 运行最小 PoC

```bash
uv run avc version                            # 版本与运行环境
uv run avc healthcheck                        # 用 Mock 组件跑通流水线并输出组件健康状态
uv run avc run --frames 6                     # 输出场景摘要
uv run avc run --frames 6 --json              # 输出可注入话轮上下文的快照 JSON
uv run python examples/run_mock_pipeline.py   # 示例：组件注入、事件订阅与快照输出
```

### 运行离线图片/视频 PoC

`avc run --input` 支持单张图片、图片目录或视频文件（需安装 `vision` extra），
检测/跟踪/关系推理仍为 Mock 组件，输出与合成帧完全同构：

```bash
uv run avc run --input tests/fixtures/sample_image.png --json   # 单张图片 -> 1 帧闭环
uv run avc run --input tests/fixtures/sample_video.mp4 --json   # 2 秒视频按 --fps 采样
uv run python examples/run_offline_pipeline.py                  # 示例：图片与视频跑通同一套时间线
uv run python examples/run_offline_pipeline.py path/to/video.mp4
```

固定测试素材位于 `tests/fixtures/`：`sample_image.png`（320x240 合成 PNG）与 `sample_video.mp4`
（2s/10fps MP4）用于验证接线与降级；`sample_detection.png`（1024x768 真实场景图，含 person/hand/phone/screen）
用于人工核对 YOLO-World 真实检测质量。集成测试不依赖摄像头、GPU 或网络模型下载。视频帧时间戳按
`start + 原始帧号 / 源帧率` 统一映射，默认以“素材末帧即打开时刻”锚定回放，保证离线素材落入时间线窗口。

### 运行 YOLO-World 真实检测（可选）

安装 `yolo-world` extra 并联网下载权重后，`avc run --detector yolo-world` 会把离线素材交给真实
YOLO-World 适配器检测，输出统一 `Detection` 并进入同一套时间线；跟踪与关系推理仍为 Mock：

```bash
uv sync --extra yolo-world
uv run avc run --input tests/fixtures/sample_image.png --detector yolo-world --json
uv run python examples/run_yolo_world_detection.py    # 逐帧打印检测框/类别/置信度/耗时与统计
```

目标类别、置信度阈值、输入尺寸、推理设备与超时均可配置（`AVC_DETECTOR_*`）；模型不可用或推理
失败时流水线降级为“该帧无检测”而不崩溃。模型版本、权重来源、许可证（AGPL-3.0/GPL-3.0，商用前
须完成合规确认）与真机基准口径见 [docs/research/yolo-world.md](docs/research/yolo-world.md)。

### 运行摄像头实时链路 PoC（PoC-5）

安装 `vision` extra 后，`avc camera` 会启动本地摄像头的实时 Sidecar 链路：后台采集线程持续抓帧写入
有界“保最新帧”缓冲，主循环按 `--fps` 低频取最新帧推理；摄像头打不开、读帧失败或模型超时都只降级、
不阻塞宿主（大屏与语音链路），运行结束输出健康状态、运行指标与场景快照：

```bash
uv sync --extra vision
uv run avc camera --duration 10                     # 默认设备 0，运行 10s 后输出健康/指标/摘要
uv run avc camera --device 1 --fps 2 --frames 20    # 指定设备、采样帧率与最大处理帧数
uv run avc camera --duration 5 --json               # 输出健康状态 + 指标 + 快照 JSON
uv run python examples/run_camera_live.py           # 示例：实时链路 + 健康/指标/快照输出
```

设备选择、请求分辨率、缓冲容量、采集失败上限与运行时长均可配置（`AVC_CAMERA_*`、`AVC_LIVE_*`）；
摄像头循环只存在于 `runtime/live.py`，不写进模型适配器或 CLI。**目标设备（Windows 一体机）真机实测、
CPU/内存/有效 FPS 等指标记录与 PoC 结论在后续 Issue 完成**；`LiveMetrics` 已预留 `cpu_percent`/`rss_mb`
字段，由真机压测脚本填充。

常用环境变量统一使用前缀 `AVC_`，例如 `AVC_SCENE_ID`、`AVC_SOURCE_ID`、`AVC_MAX_FRAMES`、
`AVC_TARGET_FPS`、`AVC_OBSERVATION_TTL_SECONDS`、`AVC_WINDOW_SECONDS`、`AVC_LOG_LEVEL`、`AVC_FAIL_FAST`，
以及检测器相关的 `AVC_DETECTOR_BACKEND`、`AVC_DETECTOR_CLASSES`、`AVC_DETECTOR_DEVICE` 等。

当前 PoC 进度：输入支持离线图片/视频文件（Issue #5）与本地摄像头实时流（Issue #8）；目标检测可选接入
真实 YOLO-World 适配器（Issue #6，默认仍为 Mock），关系推理可选接入真实 RelateAnything 适配器
（Issue #7，默认仍为 Mock），跟踪仍由 Mock 组件提供，用于验证模块边界、时间线约束与降级机制。
摄像头链路的真机性能实测、资源隔离验证与 PoC 结论在后续 Issue 完成。所有视觉输出均为辅助观察，
不触发任何业务写操作。

### 架构与边界

目录结构、依赖方向、可替换组件协议、数据分型与降级原则见
[docs/architecture/module-boundaries.md](docs/architecture/module-boundaries.md)。

## 合规与许可证

本项目会分别核查源代码、模型权重、训练数据、检测器、运行时依赖和分发方式的许可证。RelateAnything 当前仓库代码采用 AGPL-3.0-only，其模型权重、数据集和第三方检测组件可能具有独立条款；“YOLO-World”名称本身也不足以确定具体实现和权利义务。

PoC-3 接入的 YOLO-World 适配器经 Ultralytics 运行时（AGPL-3.0，或商业 Enterprise 授权）加载 `yolov8s-worldv2.pt` 权重，原始 YOLO-World 代码为 GPL-3.0；`ultralytics` 作为可选 extra 提供、权重不入仓库。在完成合规确认前，不将 ultralytics、YOLO-World 代码或权重打入商业发布包。详见 [docs/research/yolo-world.md](docs/research/yolo-world.md)。

在许可证结论完成前，不将模型、权重或相关依赖承诺为可直接用于商业交付的默认组件。正式发布时将提供完整的第三方声明、来源链接和适用的源代码/署名信息。

## 关联项目

- [阿国饭店 AI 点菜互动系统](https://github.com/yzxingtu2026/aguo-ai-dining)
- [阿国饭店 Issue #446：引入 RelateAnything 实现主动迎宾唤醒与长期视觉辅助上下文](https://github.com/yzxingtu2026/aguo-ai-dining/issues/446)
- [RelateAnything](https://github.com/Maelic/RelateAnything)
- [agent-project-skeleton](https://github.com/yzxingtu2026/agent-project-skeleton)
