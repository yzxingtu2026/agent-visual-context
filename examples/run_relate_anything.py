"""RelateAnything 离线关系推理示例：在固定图片/视频上验证关系三元组、谓词、置信度与耗时。

运行前需安装重依赖并联网下载权重（详见 docs/research/relate-anything.md）：

    uv sync --extra relate-anything --extra yolo-world
    uv run python examples/run_relate_anything.py                 # 默认 tests/fixtures 素材
    uv run python examples/run_relate_anything.py path/to/img.png
    uv run python examples/run_relate_anything.py path/to/video.mp4

示例聚焦"关系推理"这一层：先用 YOLO-World 检测 + MockTracker 获取跟踪目标，
再驱动 `RelateAnythingReasoner` 逐帧输出统一 Relation 与可观测指标，
演示模型不可用（未安装 relsgg / 权重缺失）时如何降级为提示而非崩溃。
若 YOLO-World 也不可用，则回退到 Mock 检测器提供固定目标区域。
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_visual_context.config import load_config
from agent_visual_context.domain import BBox
from agent_visual_context.errors import PerceptionError
from agent_visual_context.input import frame_source_from_path
from agent_visual_context.logging_setup import configure_logging, get_logger
from agent_visual_context.perception import (
    MockTarget,
    MockTracker,
    RelateAnythingReasoner,
    StaticSceneDetector,
    YoloWorldDetector,
)
from agent_visual_context.perception.base import Detector

logger = get_logger("examples.relate_anything")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATERIALS = [
    REPO_ROOT / "tests" / "fixtures" / "sample_detection.png",
]

# 当 YOLO-World 不可用时，使用固定目标区域模拟检测结果（基于 sample_detection.png 的典型布局）
FALLBACK_TARGETS = [
    MockTarget(label="person", bbox=BBox(x=80, y=100, width=200, height=400), confidence=0.92),
    MockTarget(label="screen", bbox=BBox(x=350, y=60, width=300, height=250), confidence=0.85),
    MockTarget(label="phone", bbox=BBox(x=180, y=280, width=60, height=100), confidence=0.70),
]


def build_detector(config: object) -> Detector:
    """尝试构造 YOLO-World 检测器；ultralytics 不可用时回退到 Mock。"""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("  [提示] ultralytics 未安装，回退到 Mock 检测器提供固定目标区域")
        return StaticSceneDetector(FALLBACK_TARGETS)
    return YoloWorldDetector.from_config(config)  # type: ignore[arg-type]


def infer_relations(path: Path, detector: Detector, reasoner: RelateAnythingReasoner) -> None:
    print(f"\n=== {path.name} ===")
    tracker = MockTracker()
    source = frame_source_from_path(path, target_fps=2.0)
    source.open()
    try:
        while (frame := source.read()) is not None:
            # 检测
            try:
                detections = detector.detect(frame)
            except PerceptionError as exc:
                print(f"  [检测降级] {frame.frame_id}: {exc}")
                detections = []

            # 跟踪
            tracked = tracker.update(frame, detections)
            if not tracked:
                print(f"  帧 {frame.frame_id}：无跟踪目标，跳过关系推理")
                continue

            # 关系推理
            try:
                relations = reasoner.infer(frame, tracked)
            except PerceptionError as exc:
                print(f"  [推理降级] {frame.frame_id}: {exc}")
                return

            print(
                f"  帧 {frame.frame_id}（{frame.width}x{frame.height}）"
                f"目标={len(tracked)} 关系={len(relations)}"
            )
            for rel in relations:
                print(
                    f"    - {rel.subject.label}({rel.subject.track_id}) "
                    f"--[{rel.predicate}]--> "
                    f"{rel.target.label}({rel.target.track_id}) "
                    f"conf={rel.confidence:.2f}"
                )
    finally:
        source.close()

    stats = reasoner.stats
    print(
        f"  统计：帧={stats.frames} 关系={stats.relations} "
        f"延迟(min/mean/max)={stats.min_latency_seconds or 0:.3f}/"
        f"{stats.mean_latency_seconds:.3f}/{stats.max_latency_seconds or 0:.3f}s"
    )


def main() -> None:
    configure_logging("WARNING")
    materials = [Path(arg) for arg in sys.argv[1:]] or DEFAULT_MATERIALS
    config = load_config(
        detector_backend="yolo-world",
        detector_classes=["person", "hand", "phone", "screen"],
        detector_device="cpu",
        reasoner_backend="relate-anything",
        reasoner_vocabulary=["looking_at", "facing", "holding", "pointing_at", "touching", "near"],
        reasoner_device="cpu",
        reasoner_timeout_seconds=120.0,  # 首帧含权重下载，给足超时
    )

    detector = build_detector(config)
    reasoner = RelateAnythingReasoner.from_config(config)
    print(f"检测器：{detector.model_version}")
    print(f"推理器：{reasoner.model_version}  关系词：{list(reasoner.settings.vocabulary)}")

    for path in materials:
        if not path.exists():
            print(f"\n[跳过] 素材不存在：{path}")
            continue
        infer_relations(path, detector, reasoner)

    reasoner.close()
    if hasattr(detector, "close"):
        detector.close()


if __name__ == "__main__":
    main()
