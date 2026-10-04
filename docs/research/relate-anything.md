# RelateAnything 关系推理适配器调研与合规记录（PoC-4 / Issue #7）

本文记录 `perception/relate_anything.py` 适配器所选模型的版本、权重来源、许可证、依赖与分发方式，
以及离线验证与真机基准的口径。对应 Issue #7，衔接 PoC-0 的 `RelationReasoner` 抽象、
PoC-1 的 `RelationObservation`/Timeline，以及 PoC-3 的检测结果。

## 选型结论

- **模型**：RelateAnything（实时开放词汇关系推理），通过 `relsgg` 包接入。
- **默认权重**：`maelic/relsgg-vits16plus`（ViT-S/16+ 视觉编码器 + 关系 transformer，
  速度与效果折中，适合 CPU PoC 验证）。权重托管于 HuggingFace，无需 gated 认证。
- **理由**：首批关系词（looking_at、facing、holding、pointing_at、touching、near）不是固定闭集，
  开放词汇关系推理可用自然语言谓词直接提示，无需重训即可调整关系词集合，
  契合"可替换关系推理接口"的目标。模型支持动态 `set_vocabulary()` 切换关系词，
  单次前向传播即可输出排序后的 (subject, relation, object) 三元组。

## 首批关系词白名单

默认关系词取自配置 `reasoner_vocabulary`（环境变量 `AVC_REASONER_VOCABULARY`，JSON 数组）：

```
looking_at, facing, holding, pointing_at, touching, near
```

覆盖大屏场景中人-屏幕、人-手机、人-人之间的空间与交互关系。
关系词经 `model.set_vocabulary([...])` 下发到模型；可通过配置调整而无需改代码。
适配器在归一化阶段对模型输出做白名单过滤（大小写不敏感、空格转下划线），
不在白名单内的谓词被丢弃，保证进入时间线的关系都是业务关心的。

## 配置项（`config.py::AppConfig`，前缀 `AVC_`）

| 配置项 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `reasoner_backend` | `AVC_REASONER_BACKEND` | `mock` | `mock` 或 `relate-anything`；决定工厂装配哪种推理器 |
| `reasoner_vocabulary` | `AVC_REASONER_VOCABULARY` | `["looking_at","facing","holding","pointing_at","touching","near"]` | 关系词白名单（JSON 数组） |
| `reasoner_conf_threshold` | `AVC_REASONER_CONF_THRESHOLD` | `0.1` | 模型级置信度阈值；开放词汇关系推理置信度普遍偏低，故取值较小 |
| `reasoner_topk` | `AVC_REASONER_TOPK` | `10` | 每帧返回的最大关系三元组数量 |
| `reasoner_device` | `AVC_REASONER_DEVICE` | `cpu` | 推理设备：`cpu` / `cuda` / `cuda:0` / `mps` |
| `reasoner_timeout_seconds` | `AVC_REASONER_TIMEOUT_SECONDS` | `15.0` | 单帧推理超时，超时降级 |
| `reasoner_model_name` | `AVC_REASONER_MODEL_NAME` | `maelic/relsgg-vits16plus` | HuggingFace 模型名或本地路径 |

说明：`reasoner_conf_threshold` 是模型级过滤；流水线仍会用既有的 `min_relation_confidence`（默认 0.4）
做二次过滤，两者职责不同——前者控制模型输出（开放词汇关系推理置信度普遍偏低，默认 0.1 只滤噪声），
后者控制进入时间线的关系质量。

## 依赖与分发

- **运行时依赖**：`relsgg>=0.1`（会引入 `torch`、`torchvision`、`transformers`、`numpy` 等重依赖），
  以及 `Pillow>=10.0`（图像加载）和 `opencv-python>=4.10`（帧数据转换）。
  作为可选 extra 提供，默认不安装，保持骨架环境轻量：

  ```bash
  uv sync --extra relate-anything
  ```

- **惰性加载**：与 ultralytics 一致，`perception/_relsgg.py::require_relsgg()` 只在构造真实后端时
  导入 relsgg；未安装时抛出 `PerceptionError`（带安装指引），由流水线降级，包导入本身不依赖它。
- **权重分发**：relsgg 首次加载 `maelic/relsgg-vits16plus` 时会从 HuggingFace Hub 下载模型权重
  （ViT-S/16+ 约 90MB + 关系 transformer 约 50MB）。离线一体机部署需预先把权重缓存到
  `~/.cache/huggingface/` 或将 `reasoner_model_name` 指向本地路径；
  权重二进制**不入本仓库**——`*.pt`、`*.bin`、`*.safetensors`、`/weights/` 等已在 `.gitignore` 排除，
  部署时单独分发。
- **首帧超时提示**：上述一次性下载发生在首帧推理期间，可能超过默认 `reasoner_timeout_seconds`（15s）
  而使首帧降级。首次运行请用较大的 `AVC_REASONER_TIMEOUT_SECONDS`（如 600）预热
  下载并缓存权重，之后各帧延迟回落到亚秒级。

## 许可证与合规（未完成确认前不进入商业发布包）

| 组成 | 许可证 | 备注 |
| --- | --- | --- |
| RelateAnything 代码（`relsgg` 包） | **AGPL-3.0-only** | 网络服务分发会触发 AGPL 传染，商用须购买商业授权 |
| 预训练权重（ViT 编码器） | **Meta DINOv2/v3 许可** | 继承 Meta 的模型许可条款，需逐一确认 |
| 训练标注数据 | **Google Gemma 使用条款** | 标注数据许可，可能影响衍生模型分发 |
| PyTorch 运行时 | BSD-3-Clause | 宽松许可，无传染风险 |
| Transformers（HuggingFace） | Apache-2.0 | 宽松许可，无传染风险 |

**合规结论**：AGPL-3.0 对闭源商业分发有强约束，权重另有 Meta/Google 独立条款。
PoC 阶段仅用于内部离线验证；**在完成许可证合规确认（购买商业授权或替换为宽松许可模型/权重）
之前，不得将 relsgg 代码、RelateAnything 权重或 Meta/Google 许可组件打入商业发布包**。
此约束记录于本文件，随代码一同交付。

## 降级行为

适配器把以下情况统一转换为 `PerceptionError`，由 `runtime.Pipeline._call()` 捕获并把 `reasoner`
组件标记为 `degraded`，该帧关系推理退化为空、流水线继续运行（`fail_fast=true` 时改为上抛）：

- **模型初始化失败**：relsgg 未安装、权重缺失/下载失败、设备不可用等；失败被缓存，后续帧不再重复加载。
- **推理异常**：模型 predict 抛出的任何异常（如显存不足）。
- **推理超时**：单帧超过 `reasoner_timeout_seconds`；推理在独立线程执行，超时后放弃该帧结果。
- **缺少像素数据**：`Frame.data` 为空（例如合成帧）时报错，提示改用图片/视频输入或 Mock 推理器。
- **空目标列表**：属于正常情况，直接返回空列表，不触发模型推理、不算失败、不降级。

## 可观测信息

每帧通过 `logging`（logger 名 `agent_visual_context.perception.relate_anything`）输出结构化字段：
`frame`、`model`（模型版本）、目标数、`device`、耗时（秒）、关系数量与谓词分布。
推理器同时保留：

- `RelateAnythingReasoner.last_metrics`：最近一帧的 `RelationMetrics`；
- `RelateAnythingReasoner.stats`：跨帧累积的 `RelationStats`（帧数、关系总数、总/最小/最大/平均延迟）。

## 离线验证与真机基准

### 离线可复现验证（无需 torch/联网）

单元测试与契约/集成测试均通过**注入假后端**验证归一化、白名单过滤、置信度裁剪、空目标、
初始化失败、推理异常、超时与流水线降级，全部 hermetic：

```bash
uv run pytest tests/unit/test_relate_anything.py \
              tests/contract/test_component_contracts.py
```

### 真机关系推理（安装 extra 并联网下载权重后）

对固定图片/短视频执行真实关系推理，示例脚本输出统一 Relation 与延迟/数量指标：

```bash
uv sync --extra relate-anything
uv run python examples/run_relate_anything.py            # 默认跑 tests/fixtures 素材
uv run python examples/run_relate_anything.py path/to/image.png
# 或经 CLI（reasoner_backend=relate-anything）：
uv run avc run --input tests/fixtures/sample_detection.png --reasoner relate-anything --json
```

### 基准记录

| 设备 | 权重 | device | 素材 | 目标数 | 单帧延迟(min/mean/max, s) | CPU/内存峰值 | 关系数量/质量备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MacBook Air (macOS) | relsgg-vits16plus | cpu | sample_detection.png（1024x768，真实场景） | 3（Mock 固定区域） | 0.466 / 0.495 / 0.495 | 未采集 | 检出 3 个关系：holding 0.31、near 0.19、near 0.14；正确识别 person-holding-phone |
| Windows 10 一体机（待填写） | relsgg-vits16plus | cpu | sample_detection.png | 待填写 | 待填写 | 待填写 | 待填写 |

质量评估口径：人工核对固定素材上的关系三元组是否覆盖 looking_at/facing/holding/pointing_at/touching/near，
记录漏检、误检与置信度分布，作为是否进入真实摄像头阶段（PoC-5）的依据。

### 首轮真机验证结论

已在 macOS 开发机（MacBook Air，`device=cpu`）实测，命令为
`uv run python examples/run_relate_anything.py`（使用 Mock 固定目标区域，ultralytics 未安装）：

- 模型从 HuggingFace 加载权重约 96s（首次），后续帧无下载开销；
- 单帧 CPU 推理约 0.47–0.50s（3 个目标区域、6 个关系词）；
- 正确识别 `person --holding--> phone`（conf=0.31），符合 sample_detection.png 的实际内容；
- `person --near--> screen`（0.19）和 `screen --near--> person`（0.14）也合理；
- `looking_at`、`facing`、`pointing_at`、`touching` 未过阈值（均 < 0.1），
  可能因 Mock 目标区域与真实人体/屏幕位置偏差较大，待接入 YOLO-World 真实检测框后复测；
- 整体置信度偏低（最高 0.31），开放词汇关系推理在 CPU 小模型上属正常表现，
  流水线级 `min_relation_confidence`（默认 0.4）需根据实测分布调低或保持，
  建议 PoC-5 摄像头阶段结合真实检测框重新标定阈值。

## 边界约束（务必遵守）

- RelateAnything 只能实现在 `perception/relate_anything.py`（及其惰性加载器 `perception/_relsgg.py`）；
  **不得**把模型调用写入 `domain/`、`temporal/` 或 `context/`。
- 推理器满足 `perception/base.py::RelationReasoner` 协议，可被 Mock 无缝替换；上层只依赖协议，不感知具体模型。
- 关系推理依赖目标区域输入（TrackedObject），不能替代目标检测和跟踪；
  检测结果由 PoC-3 的 YOLO-World 适配器提供。
- 关系结果只是视觉观察，**不得**在本 Issue 中直接解释为用户意图或触发业务动作。

## 参考来源

- [RelateAnything GitHub 仓库](https://github.com/Maelic/RelateAnything)（代码、API 用法、模型架构）
- HuggingFace 模型卡 `maelic/relsgg-vits16plus`（权重下载、推理示例）
- RelateAnything 论文：Real-Time Open-Vocabulary Relation Prediction
- Meta DINOv2 许可证说明
- Google Gemma 使用条款
