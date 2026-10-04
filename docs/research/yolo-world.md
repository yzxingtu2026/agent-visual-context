# YOLO-World 目标检测适配器调研与合规记录（PoC-3 / Issue #6）

本文记录 `perception/yolo_world.py` 适配器所选模型的版本、权重来源、许可证、依赖与分发方式，
以及离线验证与真机基准的口径。对应 Issue #6，衔接 PoC-0 的 `Detector` 抽象与 PoC-2 的图片/视频流水线。

## 选型结论

- **模型**：YOLO-World（开放词汇目标检测），通过 Ultralytics 集成运行时接入。
- **默认权重**：`yolov8s-worldv2.pt`（YOLOv8-WorldV2 的 s 号，速度与效果折中，适合 CPU 一体机 PoC）。
  可按需替换为 `yolov8s-world.pt` 或 `m/l/x` 更大规格；权重名由配置项 `detector_weights` 指定。
- **理由**：首批目标（person、hand、phone、screen/menu-area）不是固定闭集，开放词汇检测可用类别名直接提示，
  无需重训即可调整目标集合，契合"可替换目标检测接口"的目标。

## 首批目标类别

默认类别取自配置 `detector_classes`（环境变量 `AVC_DETECTOR_CLASSES`，JSON 数组）：

```
person, hand, phone, screen
```

`screen` 用于近似覆盖 issue 中的 `screen/menu-area`；开放词汇下也可写入 `menu`、`monitor`、
`display` 等更贴近的提示词，通过配置调整而无需改代码。类别经 `model.set_classes([...])` 下发到模型。

## 配置项（`config.py::AppConfig`，前缀 `AVC_`）

| 配置项 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `detector_backend` | `AVC_DETECTOR_BACKEND` | `mock` | `mock` 或 `yolo-world`；决定工厂装配哪种检测器 |
| `detector_classes` | `AVC_DETECTOR_CLASSES` | `["person","hand","phone","screen"]` | 开放词汇目标类别（JSON 数组） |
| `detector_conf_threshold` | `AVC_DETECTOR_CONF_THRESHOLD` | `0.05` | 模型级置信度阈值；开放词汇检测置信度普遍偏低，故取值较小 |
| `detector_iou_threshold` | `AVC_DETECTOR_IOU_THRESHOLD` | `0.45` | NMS IoU 阈值 |
| `detector_imgsz` | `AVC_DETECTOR_IMGSZ` | `640` | 推理输入尺寸 |
| `detector_device` | `AVC_DETECTOR_DEVICE` | `cpu` | 推理设备：`cpu` / `cuda` / `cuda:0` / `mps` |
| `detector_timeout_seconds` | `AVC_DETECTOR_TIMEOUT_SECONDS` | `10.0` | 单帧推理超时，超时降级 |
| `detector_weights` | `AVC_DETECTOR_WEIGHTS` | `yolov8s-worldv2.pt` | 权重文件名或本地路径 |

说明：`detector_conf_threshold` 是模型级过滤；流水线仍会用既有的 `min_detection_confidence`（默认 0.3）
做二次过滤，两者职责不同——前者控制模型输出，后者控制进入时间线的检测质量。

## 依赖与分发

- **运行时依赖**：`ultralytics>=8.2`（会引入 `torch`、`torchvision`、`opencv-python` 等重依赖）。
  作为可选 extra 提供，默认不安装，保持骨架环境轻量：

  ```bash
  uv sync --extra yolo-world
  ```

- **惰性加载**：与 OpenCV 一致，`perception/_ultralytics.py::require_ultralytics()` 只在构造真实后端时
  导入 ultralytics；未安装时抛出 `PerceptionError`（带安装指引），由流水线降级，包导入本身不依赖它。
- **权重分发**：ultralytics 首次加载 `yolov8s-worldv2.pt` 时会从官方源联网下载到本地缓存。
  离线一体机部署需预先把权重放入运行目录或缓存，并将 `detector_weights` 指向本地路径；
  权重二进制**不入本仓库**（体积与许可证原因），部署时单独分发。

## 许可证与合规（未完成确认前不进入商业发布包）

| 组成 | 许可证 | 备注 |
| --- | --- | --- |
| Ultralytics 运行时（`ultralytics` 包） | **AGPL-3.0** 或商业 Enterprise 授权 | 网络服务分发会触发 AGPL 传染，商用须购买 Enterprise 授权 |
| YOLO-World 原始实现（AILab-CVC/YOLO-World） | **GPL-3.0** | 论文与原始代码库许可 |
| 预训练权重（`yolov8*-world*.pt`） | 随其训练代码/数据许可，通常与上述一致 | 需按具体权重来源逐一确认 |

**合规结论**：AGPL-3.0/GPL-3.0 对闭源商业分发有强约束。PoC 阶段仅用于内部离线验证；
**在完成许可证合规确认（购买 Enterprise 授权或替换为宽松许可模型/权重）之前，不得将 ultralytics、
YOLO-World 代码或权重打入商业发布包**。此约束记录于本文件，随代码一同交付。

## 降级行为

适配器把以下情况统一转换为 `PerceptionError`，由 `runtime.Pipeline._call()` 捕获并把 `detector`
组件标记为 `degraded`，该帧检测退化为空、流水线继续运行（`fail_fast=true` 时改为上抛）：

- **模型初始化失败**：ultralytics 未安装、权重缺失/下载失败、设备不可用等；失败被缓存，后续帧不再重复加载。
- **推理异常**：模型 predict 抛出的任何异常（如显存不足）。
- **推理超时**：单帧超过 `detector_timeout_seconds`；推理在独立线程执行，超时后放弃该帧结果。
- **缺少像素数据**：`Frame.data` 为空（例如合成帧）时报错，提示改用图片/视频输入或 Mock 检测器。
- **空检测结果**：属于正常情况，返回空列表，不算失败、不降级。

## 可观测信息

每帧通过 `logging`（logger 名 `agent_visual_context.perception.yolo_world`）输出结构化字段：
`frame`、`model`（模型版本）、输入尺寸、`imgsz`、`device`、耗时（秒）、检测数量与类别分布。
检测器同时保留：

- `YoloWorldDetector.last_metrics`：最近一帧的 `DetectionMetrics`；
- `YoloWorldDetector.stats`：跨帧累积的 `DetectionStats`（帧数、检测总数、总/最小/最大/平均延迟）。

## 离线验证与真机基准

### 离线可复现验证（无需 torch/联网）

单元测试与契约/集成测试均通过**注入假后端**验证归一化、bbox 裁剪、置信度过滤、空结果、
初始化失败、推理异常、超时与流水线降级，全部 hermetic：

```bash
uv run pytest tests/unit/test_yolo_world_detector.py \
              tests/contract/test_component_contracts.py \
              tests/integration/test_yolo_world_pipeline.py
```

### 真机检测（安装 extra 并联网下载权重后）

对固定图片/短视频执行真实检测，示例脚本输出统一 Detection 与延迟/数量指标：

```bash
uv sync --extra yolo-world
uv run python examples/run_yolo_world_detection.py            # 默认跑 tests/fixtures 素材
uv run python examples/run_yolo_world_detection.py path/to/image.png
# 或经 CLI：
uv run avc run --input tests/fixtures/sample_image.png --detector yolo-world --json
```

### 基准记录（待在目标设备实测填写）

> 以下为占位表格，需在 Windows 10 一体机（或指定目标设备）实测后填写；
> issue #6 不承诺一体机实时性能，本 PoC 仅要求记录至少一轮数据。

| 设备 | 权重 | imgsz | device | 素材 | 单帧延迟(min/mean/max, s) | CPU/内存峰值 | 检测数量/质量备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填写 | yolov8s-worldv2.pt | 640 | cpu | sample_image.png | 待填写 | 待填写 | 待填写 |
| 待填写 | yolov8s-worldv2.pt | 640 | cpu | sample_video.mp4 | 待填写 | 待填写 | 待填写 |

质量评估口径：人工核对固定素材上的检测框是否覆盖 person/hand/phone/screen，记录漏检、误检与
置信度分布，作为是否进入真实摄像头阶段（PoC-5）的依据。

## 边界约束（务必遵守）

- YOLO-World 只能实现在 `perception/yolo_world.py`（及其惰性加载器 `perception/_ultralytics.py`）；
  **不得**把模型调用写入 `domain/`、`temporal/` 或 `context/`。
- 检测器满足 `perception/base.py::Detector` 协议，可被 Mock 无缝替换；上层只依赖协议，不感知具体模型。
- 检测结果只是视觉观察，**不得**在本 Issue 中直接解释为用户意图或业务事实；关系推理由 PoC-4 负责。

## 参考来源

- Ultralytics YOLO-World 模型文档（权重名、`set_classes`/`predict` 用法）
- Ultralytics 许可证说明（AGPL-3.0 / Enterprise）
- YOLO-World 原始论文与代码库（GPL-3.0）
