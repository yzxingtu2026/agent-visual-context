"""命令行入口（`avc`）。

骨架阶段提供三个命令：
- `avc version`：打印版本与运行环境；
- `avc healthcheck`：用 Mock 组件跑一条最小流水线，验证安装、装配与降级机制；
- `avc run`：以合成帧驱动 Mock 流水线，输出场景摘要（可选 JSON）。

约定：终端报告用 print，运行时日志用 logging，两者不混用。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime

from . import __version__
from .api import VisualContextApi
from .config import AppConfig, load_config
from .errors import AgentVisualContextError
from .logging_setup import configure_logging, get_logger
from .runtime import ComponentState, PipelineState, RunResult, build_mock_pipeline

logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avc",
        description="面向 Agent 的实时视觉关系上下文处理器（PoC 骨架）",
    )
    parser.add_argument("--log-level", default=None, help="日志级别，默认 INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="打印版本信息")

    health = subparsers.add_parser("healthcheck", help="运行最小流水线健康检查")
    _add_common_args(health)
    health.add_argument("--frames", type=int, default=4, help="健康检查使用的帧数，默认 4")

    run = subparsers.add_parser("run", help="以合成帧运行 Mock 流水线")
    _add_common_args(run)
    run.add_argument("--frames", type=int, default=None, help="处理帧数，默认取配置 max_frames")
    run.add_argument("--json", action="store_true", help="以 JSON 输出场景摘要")

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-id", default=None, help="场景 ID，默认 scene-01")
    parser.add_argument("--source-id", default=None, help="输入源 ID，默认 mock-source")
    parser.add_argument("--fps", type=float, default=None, help="目标帧率，默认 2.0")
    parser.add_argument("--width", type=int, default=None, help="帧宽度，默认 640")
    parser.add_argument("--height", type=int, default=None, help="帧高度，默认 480")


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    return load_config(
        scene_id=args.scene_id,
        source_id=args.source_id,
        target_fps=args.fps,
        frame_width=args.width,
        frame_height=args.height,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "version":
            _cmd_version()
            return 0

        config = _config_from_args(args)
        configure_logging(config.log_level)

        if args.command == "healthcheck":
            return _cmd_healthcheck(config, frames=args.frames)
        return _cmd_run(config, frames=args.frames, as_json=args.json)
    except AgentVisualContextError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130


def _cmd_version() -> None:
    print(f"agent-visual-context {__version__}")
    print(f"python {sys.version.split()[0]}")


def _cmd_healthcheck(config: AppConfig, *, frames: int) -> int:
    pipeline = build_mock_pipeline(config.model_copy(update={"max_frames": frames}))
    api = VisualContextApi.from_pipeline(pipeline)
    result = pipeline.run(max_frames=frames)

    print(f"agent-visual-context {__version__} 健康检查")
    print(f"场景：{config.scene_id} 输入源：{config.source_id} 帧数：{result.frames_processed}")
    print("组件状态：")
    for name, health in pipeline.status.components.items():
        detail = health.model_version
        if health.last_error:
            detail += f" 最近错误={health.last_error}"
        print(f"  - {name:<9} {health.state.value:<8} {detail}")

    snapshot = api.get_scene_snapshot(now=_snapshot_moment(result))
    print(f"时间线条目：{len(pipeline.timeline)} 观察：{len(snapshot.observations)}")
    for highlight in snapshot.highlights:
        print(f"  {highlight}")

    failed = any(
        health.state == ComponentState.FAILED for health in pipeline.status.components.values()
    )
    if failed or pipeline.status.state == PipelineState.FAILED:
        print("结果：失败")
        return 1
    degraded = pipeline.status.state == PipelineState.DEGRADED
    print("结果：降级运行" if degraded else "结果：正常")
    return 0


def _cmd_run(config: AppConfig, *, frames: int | None, as_json: bool) -> int:
    effective_config = (
        config if frames is None else config.model_copy(update={"max_frames": frames})
    )
    pipeline = build_mock_pipeline(effective_config)
    api = VisualContextApi.from_pipeline(pipeline)
    result = pipeline.run(max_frames=frames)
    snapshot = api.get_scene_snapshot(now=_snapshot_moment(result))

    if as_json:
        print(json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    print(f"场景 {snapshot.scene_id} 摘要（窗口 {snapshot.duration_seconds:.1f}s）")
    for highlight in snapshot.highlights:
        print(f"  {highlight}")
    print(
        f"状态：{pipeline.status.state.value} 帧：{result.frames_processed} "
        f"观察：{len(snapshot.observations)} 事件：{len(snapshot.events)}"
    )
    return 0 if pipeline.status.state != PipelineState.FAILED else 1


def _snapshot_moment(result: RunResult) -> datetime | None:
    """取运行结果中的快照时刻，避免用真实时钟过滤掉合成帧。"""
    return result.snapshot.generated_at if result.snapshot is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
