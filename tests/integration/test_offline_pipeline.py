"""集成测试：离线图片/视频素材跑通 Mock 流水线闭环。

不依赖摄像头、GPU 或网络模型下载：素材为 tests/fixtures 下的固定文件，
感知组件全部为 Mock。
"""

from __future__ import annotations

import json

import pytest

from agent_visual_context.cli import main
from agent_visual_context.config import AppConfig
from agent_visual_context.domain import Observation
from agent_visual_context.runtime import PipelineState, build_offline_pipeline
from tests.conftest import SAMPLE_IMAGE, SAMPLE_VIDEO, FakeClock


def test_single_image_enters_pipeline_closed_loop(clock: FakeClock) -> None:
    # 单帧素材放宽去抖门槛（1 次命中、0 秒持续），验证完整闭环
    config = AppConfig(
        scene_id="scene-image",
        max_frames=4,
        target_fps=2.0,
        persistence_min_hits=1,
        persistence_min_seconds=0.0,
    )
    pipeline = build_offline_pipeline(config, SAMPLE_IMAGE, clock=clock)

    result = pipeline.run()

    assert result.frames_processed == 1
    assert pipeline.status.state is PipelineState.STOPPED
    assert len(result.observations) == 1
    observation: Observation = result.observations[0]
    assert observation.predicate == "looking_at"
    assert observation.subject.track_id == "person-01"
    assert observation.target is not None
    assert observation.target.track_id == "screen-01"
    assert observation.source_id == "sample_image"
    assert observation.epistemic_status.value == "visual-observation"
    assert [event.kind for event in result.events] == ["greeting-candidate"]
    assert result.snapshot is not None
    assert len(result.snapshot.observations) == 1


def test_short_video_produces_bounded_timeline(config: AppConfig, clock: FakeClock) -> None:
    pipeline = build_offline_pipeline(config, SAMPLE_VIDEO, clock=clock)

    result = pipeline.run()

    # 2s/10fps 视频按 target_fps=2 采样出 4 帧
    assert result.frames_processed == 4
    assert pipeline.status.state is PipelineState.STOPPED
    assert not pipeline.status.is_degraded
    # 去抖要求 2 次命中且持续 1 秒：第 3、4 个采样帧产出观察
    assert 0 < len(result.observations) < result.frames_processed
    assert all(item.source_id == "sample_video" for item in result.observations)
    assert all(item.expires_at > item.observed_at for item in result.observations)
    assert [event.kind for event in result.events] == ["greeting-candidate"]

    snapshot = result.snapshot
    assert snapshot is not None
    assert snapshot.duration_seconds == pytest.approx(config.window_seconds)
    assert snapshot.window_end == clock()
    assert snapshot.observations


def test_offline_input_reuses_same_domain_models(clock: FakeClock) -> None:
    """图片与视频入口复用同一套领域模型与时间线逻辑。"""
    config = AppConfig(
        scene_id="scene-shared",
        max_frames=8,
        persistence_min_hits=1,
        persistence_min_seconds=0.0,
    )
    image_run = build_offline_pipeline(config, SAMPLE_IMAGE, clock=clock).run()
    video_run = build_offline_pipeline(config, SAMPLE_VIDEO, clock=clock).run()

    assert image_run.observations and video_run.observations
    for run in (image_run, video_run):
        for observation in run.observations:
            assert isinstance(observation, Observation)
            assert observation.scene_id == "scene-shared"
    # 时间线同样有界
    assert len(video_run.observations) <= video_run.frames_processed


def test_cli_run_with_video_outputs_structured_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--log-level",
            "ERROR",
            "run",
            "--input",
            str(SAMPLE_VIDEO),
            "--scene-id",
            "scene-cli",
            "--json",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(output)
    assert payload["scene_id"] == "scene-cli"
    assert payload["observations"]
    observation = payload["observations"][0]
    assert observation["source_id"] == "sample_video"
    assert observation["epistemic_status"] == "visual-observation"


def test_cli_run_with_image_outputs_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["--log-level", "ERROR", "run", "--input", str(SAMPLE_IMAGE), "--scene-id", "scene-cli"]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "场景 scene-cli 摘要" in output
    assert "帧：1" in output


def test_cli_run_with_missing_input_exits_with_code_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["--log-level", "ERROR", "run", "--input", str(SAMPLE_IMAGE.parent / "absent.mp4")]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "不存在" in captured.err
