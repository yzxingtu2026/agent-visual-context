"""单元测试：图片/视频输入源与路径分发（固定素材，不依赖摄像头/GPU/网络）。"""

from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import pytest

from agent_visual_context.errors import FrameSourceError
from agent_visual_context.input import (
    ImageFrameSource,
    VideoFrameSource,
    frame_source_from_path,
)
from tests.conftest import SAMPLE_IMAGE, SAMPLE_VIDEO, T0, FakeClock

# ---------- 图片输入源 ----------


def test_single_image_produces_one_frame(clock: FakeClock) -> None:
    source = ImageFrameSource([SAMPLE_IMAGE], clock=clock, target_fps=2.0)

    with source:
        frame = source.read()
        assert frame is not None
        assert source.read() is None

    assert frame.frame_id == "sample_image-00000"
    assert frame.source_id == "sample_image"
    # 单帧末帧锚定：captured_at 即打开时刻
    assert frame.captured_at == clock()
    assert (frame.width, frame.height) == (320, 240)
    assert frame.data is not None
    assert frame.metadata["kind"] == "image"
    assert frame.metadata["index"] == 0
    assert frame.metadata["path"] == str(SAMPLE_IMAGE)


def test_image_sequence_uses_uniform_timestamps(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        target = tmp_path / f"frame_{index}.png"
        shutil.copy(SAMPLE_IMAGE, target)
        paths.append(target)
    source = ImageFrameSource(paths, target_fps=2.0, start=T0)

    with source:
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)

    assert len(frames) == 3
    assert [frame.frame_id for frame in frames] == [
        "frame_0-00000",
        "frame_0-00001",
        "frame_0-00002",
    ]
    assert [frame.captured_at for frame in frames] == [
        T0,
        T0 + timedelta(seconds=0.5),
        T0 + timedelta(seconds=1.0),
    ]


def test_empty_image_list_raises() -> None:
    with pytest.raises(FrameSourceError, match="图片素材为空"):
        ImageFrameSource([])


def test_missing_image_file_raises_on_open(tmp_path: Path) -> None:
    source = ImageFrameSource([tmp_path / "absent.png"])

    with pytest.raises(FrameSourceError, match="不存在"):
        source.open()


def test_corrupt_image_raises_on_read(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not-an-image")
    source = ImageFrameSource([broken])

    with source, pytest.raises(FrameSourceError, match="解码失败"):
        source.read()


# ---------- 视频输入源 ----------


def test_video_is_sampled_at_target_fps(clock: FakeClock) -> None:
    source = VideoFrameSource(SAMPLE_VIDEO, target_fps=2.0, start=T0, clock=clock)

    with source:
        assert source.sampling_plan == [0, 5, 10, 15]
        assert source.source_fps == pytest.approx(10.0)
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)

    assert [frame.frame_id for frame in frames] == [
        "sample_video-00000",
        "sample_video-00005",
        "sample_video-00010",
        "sample_video-00015",
    ]
    # captured_at = start + 原始帧号 / 源帧率
    assert [frame.captured_at for frame in frames] == [
        T0 + timedelta(seconds=index / 10.0) for index in (0, 5, 10, 15)
    ]
    assert (frames[0].width, frames[0].height) == (160, 120)
    assert frames[0].data is not None
    assert frames[1].metadata["kind"] == "video"
    assert frames[1].metadata["index"] == 5
    assert frames[1].metadata["sampled"] == 1


def test_video_target_fps_above_source_keeps_all_frames() -> None:
    source = VideoFrameSource(SAMPLE_VIDEO, target_fps=30.0, start=T0)

    with source:
        count = sum(1 for _ in iter(source.read, None))

    assert count == 20


def test_video_end_anchored_to_open_moment(clock: FakeClock) -> None:
    source = VideoFrameSource(SAMPLE_VIDEO, target_fps=2.0, clock=clock)

    with source:
        last = None
        while (frame := source.read()) is not None:
            last = frame

    assert last is not None
    # 末帧（1.5s 处）距打开时刻 0.5s，即整段 2s 视频锚定在打开时刻结束
    assert last.captured_at == clock() - timedelta(seconds=0.5)


def test_missing_video_raises_on_open(tmp_path: Path) -> None:
    source = VideoFrameSource(tmp_path / "absent.mp4")

    with pytest.raises(FrameSourceError, match="不存在"):
        source.open()


def test_corrupt_video_raises_on_open(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not-a-video")
    source = VideoFrameSource(broken)

    with pytest.raises(FrameSourceError, match="无法打开"):
        source.open()


# ---------- 路径分发 ----------


def test_loader_dispatches_image_and_video() -> None:
    image_source = frame_source_from_path(SAMPLE_IMAGE)
    video_source = frame_source_from_path(SAMPLE_VIDEO)

    assert isinstance(image_source, ImageFrameSource)
    assert isinstance(video_source, VideoFrameSource)
    assert video_source.source_id == "sample_video"


def test_loader_dispatches_directory_of_images() -> None:
    source = frame_source_from_path(SAMPLE_IMAGE.parent)

    assert isinstance(source, ImageFrameSource)
    assert source.source_id == "fixtures"
    # fixtures 目录含合成小图与一张真实检测样本，按文件名排序回放
    assert [path.name for path in source.paths] == [
        "sample_detection.png",
        "sample_image.png",
    ]


def test_loader_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FrameSourceError, match="不存在"):
        frame_source_from_path(tmp_path / "absent.bin")


def test_loader_rejects_unsupported_suffix(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")

    with pytest.raises(FrameSourceError, match="不支持的素材类型"):
        frame_source_from_path(notes)


def test_loader_rejects_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FrameSourceError, match="没有可用图片"):
        frame_source_from_path(empty)
