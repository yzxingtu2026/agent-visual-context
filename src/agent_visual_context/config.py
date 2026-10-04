"""统一配置模型。

配置来源优先级：显式传入参数 > 环境变量（前缀 `AVC_`）> 默认值。
骨架阶段不引入配置文件解析，后续需要时在此扩展，避免各模块各自读环境变量。
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigurationError


class AppConfig(BaseSettings):
    """流水线运行所需的最小配置集合。"""

    model_config = SettingsConfigDict(env_prefix="AVC_", extra="ignore")

    # 场景与输入
    scene_id: str = "scene-01"
    source_id: str = "mock-source"
    frame_width: int = Field(default=640, ge=1)
    frame_height: int = Field(default=480, ge=1)
    target_fps: float = Field(default=2.0, gt=0)
    max_frames: int = Field(default=10, ge=0)

    # 时间线与去抖
    timeline_capacity: int = Field(default=512, ge=1)
    observation_ttl_seconds: float = Field(default=4.0, gt=0)
    window_seconds: float = Field(default=4.0, gt=0)
    persistence_min_hits: int = Field(default=2, ge=1)
    persistence_min_seconds: float = Field(default=1.0, ge=0)

    # 关系与策略阈值
    min_detection_confidence: float = Field(default=0.3, ge=0, le=1)
    min_relation_confidence: float = Field(default=0.4, ge=0, le=1)
    greeting_cooldown_seconds: float = Field(default=30.0, ge=0)

    # 目标检测器（PoC-3：YOLO-World 适配器）
    # backend=mock 时使用 StaticSceneDetector；backend=yolo-world 时装配真实适配器。
    detector_backend: str = "mock"
    # 首批开放词汇目标类别，覆盖 person/hand/phone/screen(menu-area) 验证路径。
    detector_classes: list[str] = Field(
        default_factory=lambda: ["person", "hand", "phone", "screen"]
    )
    # 模型级置信度阈值：开放词汇检测置信度普遍偏低，默认取值较小，
    # 流水线仍会用 min_detection_confidence 做二次过滤。
    detector_conf_threshold: float = Field(default=0.05, ge=0, le=1)
    detector_iou_threshold: float = Field(default=0.45, ge=0, le=1)
    detector_imgsz: int = Field(default=640, ge=32)
    detector_device: str = "cpu"
    # 单帧推理超时；超时按可恢复异常降级，不中断流水线。
    detector_timeout_seconds: float = Field(default=10.0, gt=0)
    detector_weights: str = "yolov8s-worldv2.pt"

    # 关系推理器（PoC-4：RelateAnything 适配器）
    # backend=mock 时使用 MockRelationReasoner；backend=relate-anything 时装配真实适配器。
    reasoner_backend: str = "mock"
    # 首批关系词白名单，覆盖大屏场景相关的空间与交互关系。
    reasoner_vocabulary: list[str] = Field(
        default_factory=lambda: [
            "looking_at",
            "facing",
            "holding",
            "pointing_at",
            "touching",
            "near",
        ]
    )
    # 模型级置信度阈值；低于此值的关系三元组在适配器层即被过滤。
    reasoner_conf_threshold: float = Field(default=0.3, ge=0, le=1)
    # 每帧返回的最大关系三元组数量。
    reasoner_topk: int = Field(default=10, ge=1)
    reasoner_device: str = "cpu"
    # 单帧推理超时；超时按可恢复异常降级，不中断流水线。
    reasoner_timeout_seconds: float = Field(default=15.0, gt=0)
    reasoner_model_name: str = "maelic/relsgg-vits16plus"

    # 运行与可观测性
    log_level: str = "INFO"
    fail_fast: bool = False

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        normalized = value.upper()
        if normalized not in allowed:
            msg = f"log_level 必须是 {sorted(allowed)} 之一，收到 {value!r}"
            raise ConfigurationError(msg)
        return normalized

    @field_validator("detector_backend")
    @classmethod
    def _validate_detector_backend(cls, value: str) -> str:
        allowed = {"mock", "yolo-world"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            msg = f"detector_backend 必须是 {sorted(allowed)} 之一，收到 {value!r}"
            raise ConfigurationError(msg)
        return normalized

    @field_validator("detector_device")
    @classmethod
    def _validate_detector_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            msg = "detector_device 不能为空（例如 cpu、cuda、cuda:0、mps）"
            raise ConfigurationError(msg)
        return normalized

    @field_validator("detector_classes")
    @classmethod
    def _validate_detector_classes(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            msg = "detector_classes 至少需要一个非空目标类别"
            raise ConfigurationError(msg)
        return cleaned

    @field_validator("reasoner_backend")
    @classmethod
    def _validate_reasoner_backend(cls, value: str) -> str:
        allowed = {"mock", "relate-anything"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            msg = f"reasoner_backend 必须是 {sorted(allowed)} 之一，收到 {value!r}"
            raise ConfigurationError(msg)
        return normalized

    @field_validator("reasoner_vocabulary")
    @classmethod
    def _validate_reasoner_vocabulary(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            msg = "reasoner_vocabulary 至少需要一个非空关系词"
            raise ConfigurationError(msg)
        return cleaned

    @field_validator("reasoner_device")
    @classmethod
    def _validate_reasoner_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            msg = "reasoner_device 不能为空（例如 cpu、cuda、cuda:0、mps）"
            raise ConfigurationError(msg)
        return normalized


def load_config(**overrides: object) -> AppConfig:
    """加载配置；`overrides` 中的 None 值会被忽略。

    配置非法时抛出 `ConfigurationError`，由 CLI 转换为非零退出码。
    """
    payload = {key: value for key, value in overrides.items() if value is not None}
    try:
        return AppConfig(**payload)  # type: ignore[arg-type]
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError 等
        msg = f"配置加载失败：{exc}"
        raise ConfigurationError(msg) from exc
