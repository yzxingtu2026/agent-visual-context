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
