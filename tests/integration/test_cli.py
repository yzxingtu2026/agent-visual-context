"""集成测试：CLI 入口可安装、可执行、可输出结构化结果。"""

from __future__ import annotations

import json

import pytest

from agent_visual_context import __version__
from agent_visual_context.cli import main


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["version"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert __version__ in output


def test_healthcheck_command_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--log-level", "ERROR", "healthcheck", "--frames", "4"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "结果：正常" in output
    assert "detector" in output
    assert "reasoner" in output


def test_run_command_outputs_text_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--log-level", "ERROR", "run", "--frames", "6", "--scene-id", "scene-cli"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "场景 scene-cli 摘要" in output


def test_run_command_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--log-level", "ERROR", "run", "--frames", "6", "--json"])

    output = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(output)
    assert payload["scene_id"] == "scene-01"
    assert payload["observations"]
    observation = payload["observations"][0]
    assert observation["epistemic_status"] == "visual-observation"
    assert observation["expires_at"] > observation["observed_at"]


def test_invalid_config_exits_with_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--fps", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "错误" in captured.err
