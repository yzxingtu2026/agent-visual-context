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


def test_detector_defaults_cover_first_batch_classes() -> None:
    config = AppConfig()

    assert config.detector_backend == "mock"
    assert {"person", "hand", "phone", "screen"} <= set(config.detector_classes)
    assert 0 <= config.detector_conf_threshold <= 1
    assert config.detector_imgsz >= 32
    assert config.detector_device == "cpu"
    assert config.detector_timeout_seconds > 0


def test_detector_backend_normalized_and_validated() -> None:
    assert load_config(detector_backend="YOLO-World").detector_backend == "yolo-world"
    with pytest.raises(ConfigurationError):
        load_config(detector_backend="yolo-v9")


def test_detector_classes_from_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVC_DETECTOR_CLASSES", '["person", "cup"]')

    config = AppConfig()

    assert config.detector_classes == ["person", "cup"]


def test_empty_detector_classes_rejected() -> None:
    with pytest.raises(ConfigurationError):
        load_config(detector_classes=[])


def test_empty_detector_device_rejected() -> None:
    with pytest.raises(ConfigurationError):
        load_config(detector_device="   ")
