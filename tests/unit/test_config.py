"""配置模型单元测试：环境变量前缀、默认值与非法配置。"""

from __future__ import annotations

import pytest

from agent_visual_context.config import AppConfig, load_config
from agent_visual_context.errors import ConfigurationError


def test_defaults_match_poc_constraints() -> None:
    config = AppConfig()

    assert config.target_fps == pytest.approx(2.0)
    assert config.frame_width == 640
    assert config.observation_ttl_seconds > 0
    assert config.log_level == "INFO"


def test_environment_variables_use_avc_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVC_SCENE_ID", "scene-env")
    monkeypatch.setenv("AVC_MAX_FRAMES", "3")
    monkeypatch.setenv("AVC_LOG_LEVEL", "debug")

    config = AppConfig()

    assert config.scene_id == "scene-env"
    assert config.max_frames == 3
    assert config.log_level == "DEBUG"


def test_load_config_ignores_none_overrides() -> None:
    config = load_config(scene_id=None, max_frames=5)

    assert config.scene_id == "scene-01"
    assert config.max_frames == 5


def test_invalid_log_level_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        load_config(log_level="verbose")


def test_invalid_value_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        load_config(target_fps=0)
