"""摄像头全链路可视化示例：实时采集 -> 检测/跟踪/关系推理 -> 画框 + 右下角中文上下文。

这条示例把整条链路跑在一张实时画面上，用于人工核对端到端效果：

    摄像头帧 -> Detector（可切 YOLO-World）-> Tracker -> RelationReasoner（可切 RelateAnything）
             -> 去抖/时间线/摘要 -> 场景快照 -> 叠加绘制

画面元素：
- 每个跟踪目标画检测框 + 类别 + track_id + 置信度；
- 每条关系在主语/宾语框心之间画带箭头连线 + 谓词；
- 右下角中文上下文面板：链路状态、有效 FPS、目标/关系/观察/事件计数与最近若干条视觉观察。

运行（需本机摄像头，并安装可视化依赖）：

    uv sync --extra vision --extra viz                     # 摄像头 + 中文面板（Pillow）
    uv run python examples/run_camera_visual.py            # 默认设备 0，Mock 检测/关系，按 q 退出
    uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything
    uv run python examples/run_camera_visual.py --classes person,phone,screen \\
        --relations looking_at,holding,pointing_at,near    # 自配类别与关系词

办公室场景（笔记本自带摄像头）常用类别组合，可按需增减：

    # 人手与手持设备（最轻量，适合验证「人拿着手机」这类中文上下文）
    uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \\
        --classes person,hand,phone,laptop --relations holding,looking_at,touching,near
    # 桌面物品
    uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \\
        --classes person,laptop,keyboard,mouse,cup,bottle \\
        --relations using,looking_at,touching,next_to,near
    # 会议/协作场景
    uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \\
        --classes person,chair,table,laptop,screen,whiteboard \\
        --relations sitting_on,facing,looking_at,pointing_at,near
    # 阅读/书写场景
    uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \\
        --classes person,book,notebook,paper,pen,phone \\
        --relations holding,reading,looking_at,typing_on,near
    # 全量办公（类别多、推理更慢，CPU 上建议 --fps 1）
    uv run python examples/run_camera_visual.py --detector yolo-world --reasoner relate-anything \\
        --fps 1 --classes person,hand,phone,laptop,keyboard,mouse,monitor,cup,bottle,book,backpack,chair,table \\
        --relations holding,looking_at,using,touching,next_to,near

无摄像头/无显示器时，可用离线素材跑同一套可视化并保存标注帧（便于验证与 CI 人工核对）：

    uv run python examples/run_camera_visual.py --input tests/fixtures/sample_detection.png \\
        --save avc_visual.png

依赖与降级：
- 画面绘制依赖 OpenCV（`vision` extra）；缺失时直接提示安装并退出（可视化硬前提）。
- 右下角中文依赖 Pillow + 系统中文字体（`viz` extra）；OpenCV 的 `cv2.putText` 不能渲染中文，
  缺失时面板自动降级为 ASCII，检测框与关系连线照常绘制，不崩溃。
- 摄像头打不开/读帧失败按可恢复异常降级退出，绝不抛异常阻塞进程。

真实生产链路请用 `examples/run_camera_live.py`（采集/消费解耦的 Sidecar 循环）；本示例为了让
每一帧推理结果都能即时绘制，采用前台单循环（cv2 窗口必须在主线程）。

所有画面均为视觉辅助观察，不触发任何业务写操作。
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from agent_visual_context.config import load_config
from agent_visual_context.domain import Observation, Relation, Snapshot, TrackedObject
from agent_visual_context.input import frame_source_from_path
from agent_visual_context.input.base import FrameSource
from agent_visual_context.logging_setup import configure_logging, get_logger
from agent_visual_context.runtime import FrameResult, build_camera_source, build_mock_pipeline
from agent_visual_context.runtime.status import PipelineState, PipelineStatus

logger = get_logger("examples.camera_visual")

try:  # 可视化硬前提：OpenCV + numpy（vision extra）
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - 环境缺依赖时的降级提示
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

# 类别配色（BGR），按标签稳定取色，保证同一类别颜色一致。
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (80, 200, 80),
    (80, 80, 220),
    (220, 160, 60),
    (200, 80, 200),
    (60, 180, 220),
    (180, 120, 80),
    (120, 120, 240),
    (240, 200, 120),
)

# 英文标签 -> 中文名词，用于把上下文面板的观察渲染成自然中文（如「人拿着手机」）。
# 未命中时回退原文，保证任何类别都有可读输出。覆盖办公室与门店常见目标。
_LABEL_ZH: dict[str, str] = {
    "person": "人",
    "people": "人",
    "hand": "手",
    "phone": "手机",
    "cell phone": "手机",
    "mobile phone": "手机",
    "laptop": "笔记本电脑",
    "computer": "电脑",
    "keyboard": "键盘",
    "mouse": "鼠标",
    "monitor": "显示器",
    "screen": "屏幕",
    "display": "屏幕",
    "tv": "电视",
    "cup": "杯子",
    "mug": "马克杯",
    "bottle": "水瓶",
    "book": "书",
    "notebook": "笔记本",
    "paper": "纸张",
    "document": "文件",
    "pen": "笔",
    "chair": "椅子",
    "table": "桌子",
    "desk": "办公桌",
    "backpack": "背包",
    "bag": "包",
    "headset": "耳机",
    "earphone": "耳机",
    "whiteboard": "白板",
    "projector": "投影仪",
    "plant": "绿植",
    "window": "窗户",
    "door": "门",
    "menu": "菜单",
    "food": "食物",
    "plate": "餐盘",
}

# 英文谓词 -> 中文动词，与 _LABEL_ZH 组合成「人拿着手机」这类自然句。
_PREDICATE_ZH: dict[str, str] = {
    "looking_at": "看着",
    "watching": "看着",
    "facing": "面向",
    "holding": "拿着",
    "carrying": "携带",
    "pointing_at": "指着",
    "touching": "触摸",
    "using": "使用",
    "typing_on": "敲击",
    "reading": "阅读",
    "drinking": "喝",
    "talking_to": "交谈",
    "sitting_on": "坐在",
    "standing_on": "站在",
    "on": "在…上",
    "on_top_of": "在…顶部",
    "next_to": "挨着",
    "beside": "在…旁边",
    "near": "靠近",
    "behind": "在…后面",
    "in_front_of": "在…前面",
    "under": "在…下面",
    "above": "在…上面",
    "wearing": "佩戴",
}


def _zh(token: str, table: dict[str, str]) -> str:
    """查中文映射表；未命中（含大小写差异）时回退原文。"""
    return table.get(token) or table.get(token.lower()) or token


def _observation_sentence(observation: Observation) -> str:
    """把一条观察渲染成自然中文句，如「人拿着手机」；无宾语时仅主谓。"""
    subject = _zh(observation.subject.label, _LABEL_ZH)
    predicate = _zh(observation.predicate, _PREDICATE_ZH)
    if observation.target is None:
        return f"{subject}{predicate}"
    return f"{subject}{predicate}{_zh(observation.target.label, _LABEL_ZH)}"


# 中文字体候选路径，覆盖 macOS / Windows / 常见 Linux 发行版。
_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
_FONT_GLOBS: tuple[str, ...] = (
    "/usr/share/fonts/**/Noto*CJK*.t?c",
    "/usr/share/fonts/**/wqy*.t?c",
    "/usr/share/fonts/**/*Hei*.t?c",
)


def _find_cjk_font(user_font: str | None) -> str | None:
    """按 用户指定 > 已知路径 > 通配搜索 的顺序定位一个可用中文字体。"""
    if user_font:
        return user_font if Path(user_font).exists() else None
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for pattern in _FONT_GLOBS:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def _color_for(label: str) -> tuple[int, int, int]:
    return _PALETTE[sum(ord(ch) for ch in label) % len(_PALETTE)]


def _split_csv(value: str | None) -> list[str] | None:
    """把逗号分隔的命令行参数解析成去空的列表；未给出时返回 None（用配置默认值）。"""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


class ContextPanel:
    """右下角上下文面板：优先用 Pillow + 中文字体渲染，缺失时降级为 ASCII（cv2.putText）。"""

    def __init__(self, *, font_path: str | None = None, font_size: int = 18) -> None:
        self._font = None
        self._font_size = font_size
        path = _find_cjk_font(font_path)
        if path is not None:
            try:
                from PIL import ImageFont

                self._font = ImageFont.truetype(path, font_size)
                logger.info("中文面板字体：%s", path)
            except Exception as exc:  # 字体损坏/格式不支持：降级
                logger.warning("加载中文字体失败，面板降级为 ASCII：%s", exc)
        if self._font is None:
            logger.warning(
                "未找到可用中文字体或未安装 Pillow（uv sync --extra viz），"
                "右下角面板降级为 ASCII，检测框不受影响"
            )

    @property
    def chinese(self) -> bool:
        return self._font is not None

    def draw(self, img: object, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
        """在 img（BGR numpy）右下角就地绘制面板；小帧按可用高度截断行数。"""
        img_h = img.shape[0]  # type: ignore[attr-defined]
        row_h = self._font_size + 6 if self._font is not None else 20
        pad = 10
        # 顶部留出状态条约 48px + 边距，避免面板与状态条重叠
        max_h = max(row_h + 2 * pad, img_h - 48 - 12)
        max_lines = max(1, int((max_h - 2 * pad) // row_h))
        lines = list(lines)[:max_lines]
        if self._font is not None:
            self._draw_pil(img, lines)
        else:
            self._draw_cv2(img, lines)

    # -- 内部实现 ---------------------------------------------------------

    def _geometry(
        self, width: int, height: int, img_w: int, img_h: int
    ) -> tuple[int, int, int, int]:
        margin = 12
        width = min(width, int(img_w * 0.8))
        height = min(height, img_h - 2 * margin)
        x0 = img_w - width - margin
        y0 = img_h - height - margin
        return x0, y0, width, height

    def _draw_pil(self, img: object, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
        from PIL import Image, ImageDraw

        line_h = self._font_size + 6
        pad = 10
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        text_w = max((int(probe.textlength(text, font=self._font)) for text, _ in lines), default=0)
        box_w = text_w + pad * 2
        box_h = line_h * len(lines) + pad * 2
        img_h, img_w = img.shape[:2]  # type: ignore[attr-defined]
        x0, y0, box_w, box_h = self._geometry(box_w, box_h, img_w, img_h)

        overlay = img.copy()  # type: ignore[attr-defined]
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)  # type: ignore[arg-type]

        region = cv2.cvtColor(img[y0 : y0 + box_h, x0 : x0 + box_w], cv2.COLOR_BGR2RGB)  # type: ignore[index]
        canvas = Image.fromarray(region)
        drawer = ImageDraw.Draw(canvas)
        y = pad
        for text, color_bgr in lines:
            drawer.text(
                (pad, y), text, font=self._font, fill=(color_bgr[2], color_bgr[1], color_bgr[0])
            )
            y += line_h
        img[y0 : y0 + box_h, x0 : x0 + box_w] = cv2.cvtColor(  # type: ignore[index]
            np.asarray(canvas), cv2.COLOR_RGB2BGR
        )

    def _draw_cv2(self, img: object, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
        scale, thickness, gap, pad = 0.5, 1, 8, 10
        sizes = [
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
            for text, _ in lines
        ]
        box_w = max((size[0] for size in sizes), default=0) + pad * 2
        box_h = sum(size[1] + gap for size in sizes) + pad * 2
        img_h, img_w = img.shape[:2]  # type: ignore[attr-defined]
        x0, y0, box_w, box_h = self._geometry(box_w, box_h, img_w, img_h)

        overlay = img.copy()  # type: ignore[attr-defined]
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)  # type: ignore[arg-type]

        y = y0 + pad
        for (text, color_bgr), size in zip(lines, sizes, strict=False):
            y += size[1]
            cv2.putText(
                img,  # type: ignore[arg-type]
                text,
                (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color_bgr,
                thickness,
                cv2.LINE_AA,
            )
            y += gap


def _draw_tracked(
    img: object, tracked: Sequence[TrackedObject]
) -> dict[str, tuple[int, int, tuple[int, int, int]]]:
    """绘制检测框并返回 track_id -> (中心 x, 中心 y, 颜色)，供关系连线复用。"""
    centers: dict[str, tuple[int, int, tuple[int, int, int]]] = {}
    for obj in tracked:
        x, y = int(obj.bbox.x), int(obj.bbox.y)
        w, h = int(obj.bbox.width), int(obj.bbox.height)
        color = _color_for(obj.label)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # type: ignore[arg-type]
        centers[obj.track_id] = (x + w // 2, y + h // 2, color)

        text = f"{obj.label} #{obj.track_id.split('-')[-1]} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y - th - 6)
        cv2.rectangle(img, (x, ty), (x + tw + 6, ty + th + 6), color, -1)  # type: ignore[arg-type]
        cv2.putText(  # type: ignore[arg-type]
            img,
            text,
            (x + 3, ty + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return centers


def _draw_relations(
    img: object,
    relations: Sequence[Relation],
    centers: dict[str, tuple[int, int, tuple[int, int, int]]],
) -> None:
    """在主语/宾语框心之间画带箭头连线，中点标注谓词与置信度。"""
    for rel in relations:
        start = centers.get(rel.subject.track_id)
        end = centers.get(rel.target.track_id)
        if start is None or end is None:
            continue
        color = start[2]
        cv2.arrowedLine(  # type: ignore[arg-type]
            img, (start[0], start[1]), (end[0], end[1]), color, 2, tipLength=0.06
        )
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        label = f"{rel.predicate} {rel.confidence:.2f}"
        cv2.putText(  # type: ignore[arg-type]
            img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )


def _draw_header(img: object, status: PipelineStatus, fps: float) -> None:
    """左上角 ASCII 状态条：链路状态与有效 FPS。"""
    text = f"state={status.state.value} degraded={'Y' if status.is_degraded else 'N'} fps={fps:.1f}"
    cv2.rectangle(img, (8, 8), (8 + 12 * len(text), 34), (0, 0, 0), -1)  # type: ignore[arg-type]
    cv2.putText(  # type: ignore[arg-type]
        img, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 1, cv2.LINE_AA
    )


def _context_lines(
    result: FrameResult,
    snapshot: Snapshot,
    status: PipelineStatus,
    fps: float,
    *,
    chinese: bool,
) -> list[tuple[str, tuple[int, int, int]]]:
    """构造右下角面板文本（中文优先，无字体时用 ASCII 兜底）。"""
    state = status.state.value
    if not chinese:
        return [
            (f"[context] last {snapshot.duration_seconds:.0f}s", (120, 220, 120)),
            (
                f"state={state} degraded={'Y' if status.is_degraded else 'N'} fps={fps:.1f}",
                (255, 255, 255),
            ),
            (
                f"tracks={len(result.tracked_items)} relations={len(result.relation_items)} "
                f"obs={len(snapshot.observations)} events={len(snapshot.events)}",
                (255, 255, 255),
            ),
            ("install viz extra + CJK font for Chinese", (150, 180, 255)),
        ]

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"视觉上下文 · 最近 {snapshot.duration_seconds:.0f} 秒", (120, 220, 120)),
        (
            f"状态：{state}　降级：{'是' if status.is_degraded else '否'}　有效FPS：{fps:.1f}",
            (255, 255, 255),
        ),
        (
            f"目标 {len(result.tracked_items)} · 关系 {len(result.relation_items)} · "
            f"观察 {len(snapshot.observations)} · 事件 {len(snapshot.events)}",
            (255, 255, 255),
        ),
    ]
    # 观察明细渲染成自然中文句（如「人拿着手机」），比英文 highlights 更贴近真实语义。
    for observation in snapshot.observations[:4]:
        lines.append(
            (
                f"- {_observation_sentence(observation)}（置信度 {observation.confidence:.2f}）",
                (235, 235, 235),
            )
        )
    lines.append(("视觉辅助观察 · 不触发业务写操作", (150, 180, 255)))
    return lines


def _annotate(
    frame_bgr: object,
    result: FrameResult,
    snapshot: Snapshot,
    status: PipelineStatus,
    fps: float,
    panel: ContextPanel,
) -> object:
    img = frame_bgr.copy()  # type: ignore[attr-defined]
    centers = _draw_tracked(img, result.tracked_items)
    _draw_relations(img, result.relation_items, centers)
    _draw_header(img, status, fps)
    panel.draw(img, _context_lines(result, snapshot, status, fps, chinese=panel.chinese))
    return img


def _build_source(args: argparse.Namespace, config: object) -> FrameSource:
    if args.input:
        return frame_source_from_path(Path(args.input), target_fps=args.fps)
    return build_camera_source(config)  # type: ignore[arg-type]


def run(args: argparse.Namespace) -> int:
    if cv2 is None or np is None:
        print("缺少 OpenCV/numpy，无法可视化。请先安装：uv sync --extra vision --extra viz")
        return 1

    configure_logging(args.log_level)
    config = load_config(
        scene_id="scene-camera-visual",
        target_fps=args.fps,
        camera_device_index=args.device,
        camera_width=args.width,
        camera_height=args.height,
        detector_backend=args.detector,
        reasoner_backend=args.reasoner,
        detector_classes=_split_csv(args.classes),
        reasoner_vocabulary=_split_csv(args.relations),
        min_detection_confidence=args.conf,
        # 放宽去抖门槛，便于短演示/单帧离线素材也能立即产出观察
        persistence_min_hits=1,
        persistence_min_seconds=0.0,
    )
    source = _build_source(args, config)
    pipeline = build_mock_pipeline(config, source=source)
    panel = ContextPanel(font_path=args.font, font_size=args.font_size)

    try:
        source.open()
    except Exception as exc:
        # 摄像头打不开/被占用：降级退出，不抛异常阻塞进程。
        print(f"[降级] 输入源打开失败：{exc}")
        return 0
    pipeline.status.state = PipelineState.RUNNING

    processed = 0
    saved: Path | None = None
    started = time.monotonic()
    print(
        f"链路：{source.source_id} -> {config.detector_backend}/{config.reasoner_backend}"
        f"　关系词：{config.reasoner_vocabulary}"
    )
    print("操作：q/ESC 退出，s 保存当前帧" if not args.save else f"保存模式：{args.save}")
    try:
        while True:
            if args.duration > 0 and time.monotonic() - started > args.duration:
                break
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            try:
                frame = source.read()
            except Exception as exc:
                print(f"[降级] 读帧失败，停止：{exc}")
                break
            if frame is None:  # 离线素材耗尽
                break
            if frame.data is None:
                continue

            result = pipeline.process_frame(frame)
            processed += 1
            snapshot = pipeline.summarizer.build(
                pipeline.timeline,
                now=frame.captured_at,
                degraded=pipeline.status.is_degraded,
            )
            elapsed = max(1e-6, time.monotonic() - started)
            # 前两帧用于预热，避免单帧/首帧算出失真的 FPS
            fps = processed / elapsed if processed >= 2 else 0.0
            img = _annotate(frame.data, result, snapshot, pipeline.status, fps, panel)

            if args.save:
                saved = _save_frame(img, args.save, processed)
            else:
                if not _show(img, args.window, processed):
                    break
    finally:
        source.close()
        if not args.save:
            cv2.destroyAllWindows()

    pipeline.status.state = (
        PipelineState.DEGRADED if pipeline.status.is_degraded else PipelineState.STOPPED
    )
    print(
        f"\n结束：处理 {processed} 帧，最终状态 {pipeline.status.state.value}"
        f"（降级={'是' if pipeline.status.is_degraded else '否'}）"
    )
    if saved is not None:
        print(f"已保存标注帧：{saved}")
    print("提示：以上均为视觉辅助观察，不触发任何业务写操作。")
    return 0


def _save_frame(img: object, target: str, index: int) -> Path:
    out = Path(target)
    if out.suffix:  # 指定文件名：逐帧覆盖，最终留下最后一帧
        out.parent.mkdir(parents=True, exist_ok=True)
        path = out
    else:  # 指定目录：按序号保存每一帧
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"frame_{index:05d}.png"
    cv2.imwrite(str(path), img)  # type: ignore[arg-type]
    return path


def _show(img: object, window: str, index: int) -> bool:
    """显示一帧并处理按键；返回 False 表示应退出循环。"""
    del index
    try:
        cv2.imshow(window, img)  # type: ignore[arg-type]
        key = cv2.waitKey(1) & 0xFF
    except Exception as exc:
        print(f"[降级] 无法打开显示窗口（{exc}）；无显示环境请改用 --save 保存标注帧")
        return False
    if key in (ord("q"), 27):  # q 或 ESC
        return False
    if key == ord("s"):
        path = Path(f"avc_visual_{time.strftime('%H%M%S')}.png")
        cv2.imwrite(str(path), img)  # type: ignore[arg-type]
        print(f"已保存 {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="摄像头全链路可视化示例（画框 + 中文上下文）")
    parser.add_argument("--device", type=int, default=0, help="摄像头设备索引，默认 0")
    parser.add_argument("--fps", type=float, default=2.0, help="采样帧率，默认 2.0")
    parser.add_argument("--width", type=int, default=640, help="请求分辨率宽，默认 640")
    parser.add_argument("--height", type=int, default=480, help="请求分辨率高，默认 480")
    parser.add_argument(
        "--duration", type=float, default=0.0, help="运行时长（秒），0 表示直到退出，默认 0"
    )
    parser.add_argument(
        "--max-frames",
        dest="max_frames",
        type=int,
        default=0,
        help="最多处理帧数，0 表示不限，默认 0",
    )
    parser.add_argument(
        "--detector", default="mock", choices=["mock", "yolo-world"], help="检测后端，默认 mock"
    )
    parser.add_argument(
        "--reasoner",
        default="mock",
        choices=["mock", "relate-anything"],
        help="关系推理后端，默认 mock",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="检测类别，逗号分隔（如 person,phone,screen）；默认用配置内置",
    )
    parser.add_argument(
        "--relations",
        default=None,
        help="关系词，逗号分隔（如 looking_at,holding,near）；默认用配置内置",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="检测置信度下限，默认 0.3")
    parser.add_argument(
        "--input", default=None, help="离线图片/视频路径，替代摄像头（便于无摄像头验证）"
    )
    parser.add_argument("--save", default=None, help="保存标注帧到文件（.png）或目录，不弹窗")
    parser.add_argument("--window", default="agent-visual-context", help="显示窗口标题")
    parser.add_argument("--font", default=None, help="指定中文字体文件路径（默认自动探测）")
    parser.add_argument(
        "--font-size", dest="font_size", type=int, default=18, help="中文字号，默认 18"
    )
    parser.add_argument(
        "--log-level", dest="log_level", default="WARNING", help="日志级别，默认 WARNING"
    )
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
