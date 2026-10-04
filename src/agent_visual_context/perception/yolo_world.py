"""YOLO-World 目标检测适配器（PoC-3）。

本模块是模型调用的唯一落点：`domain/`、`temporal/`、`context/` 不得引用此处或任何
具体模型。适配器遵守 `perception/base.py::Detector` 协议，因此可被 Mock 检测器无缝替换，
时间线与上层模块无需修改。

设计要点：
- **可插拔后端**：真实推理走 `UltralyticsYoloWorldBackend`（惰性加载 ultralytics），
  测试与离线验证可注入假后端，无需安装 torch 或联网下载权重；
- **归一化**：模型输出统一映射为领域层 `Detection`（bbox 裁剪到帧内、置信度过滤、
  时间戳取自当前帧、携带模型版本）；
- **降级**：模型初始化失败、推理异常、推理超时、缺少像素数据都抛出 `PerceptionError`，
  由 `runtime.Pipeline._call()` 捕获并降级为“该帧无检测”，流水线继续运行；空检测结果
  是正常的空列表，不算失败；
- **可观测**：每帧记录模型版本、推理耗时、输入尺寸、检测数量与类别分布，
  通过 `logging` 输出并累积到 `YoloWorldDetector.stats`。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import AppConfig
from ..domain import BBox, Detection, Frame
from ..errors import PerceptionError
from ..logging_setup import get_logger
from ._ultralytics import require_ultralytics

logger = get_logger("perception.yolo_world")


@dataclass(frozen=True, slots=True)
class RawDetection:
    """后端返回的原始检测框（像素坐标 xyxy + 置信度 + 类别标签）。

    后端负责把模型张量转换为纯 Python 数值，使 `YoloWorldDetector` 不感知 torch/ultralytics 类型。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


@dataclass(frozen=True, slots=True)
class YoloWorldSettings:
    """YOLO-World 检测器的可配置项，可由 `AppConfig` 派生。"""

    classes: tuple[str, ...] = ("person", "hand", "phone", "screen")
    conf_threshold: float = 0.05
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    timeout_seconds: float = 10.0
    weights: str = "yolov8s-worldv2.pt"

    @property
    def default_model_version(self) -> str:
        return f"yolo-world:{self.weights}"

    @classmethod
    def from_config(cls, config: AppConfig) -> YoloWorldSettings:
        return cls(
            classes=tuple(config.detector_classes),
            conf_threshold=config.detector_conf_threshold,
            iou_threshold=config.detector_iou_threshold,
            imgsz=config.detector_imgsz,
            device=config.detector_device,
            timeout_seconds=config.detector_timeout_seconds,
            weights=config.detector_weights,
        )


@dataclass(slots=True)
class DetectionMetrics:
    """单帧检测的可观测信息。"""

    frame_id: str
    model_version: str
    input_width: int
    input_height: int
    imgsz: int
    device: str
    latency_seconds: float
    detection_count: int
    label_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionStats:
    """跨帧累积的检测统计，用于记录一轮延迟/数量数据。"""

    frames: int = 0
    detections: int = 0
    total_latency_seconds: float = 0.0
    min_latency_seconds: float | None = None
    max_latency_seconds: float | None = None

    @property
    def mean_latency_seconds(self) -> float:
        return self.total_latency_seconds / self.frames if self.frames else 0.0

    def record(self, metrics: DetectionMetrics) -> None:
        self.frames += 1
        self.detections += metrics.detection_count
        self.total_latency_seconds += metrics.latency_seconds
        self.min_latency_seconds = (
            metrics.latency_seconds
            if self.min_latency_seconds is None
            else min(self.min_latency_seconds, metrics.latency_seconds)
        )
        self.max_latency_seconds = (
            metrics.latency_seconds
            if self.max_latency_seconds is None
            else max(self.max_latency_seconds, metrics.latency_seconds)
        )


class YoloWorldBackend(Protocol):
    """YOLO-World 推理后端协议：真实实现惰性加载 ultralytics，测试可注入假后端。"""

    @property
    def model_version(self) -> str: ...

    def predict(
        self,
        image: Any,
        *,
        classes: Sequence[str],
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
        device: str,
    ) -> Sequence[RawDetection]: ...


class UltralyticsYoloWorldBackend:
    """基于 ultralytics 的真实 YOLO-World 后端。

    构造时即加载模型（首次会按 `weights` 联网下载权重），失败抛出异常，
    由 `YoloWorldDetector` 转换为 `PerceptionError` 触发降级。
    """

    def __init__(self, settings: YoloWorldSettings) -> None:
        self._settings = settings
        ultralytics = require_ultralytics()
        # YOLO(weights) 会加载/下载权重；set_classes 设定开放词汇类别。
        self._model: Any = ultralytics.YOLO(settings.weights)
        self._model_version = f"yolo-world:{settings.weights}"
        logger.info("YOLO-World 后端就绪 weights=%s", settings.weights)

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(
        self,
        image: Any,
        *,
        classes: Sequence[str],
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
        device: str,
    ) -> Sequence[RawDetection]:
        self._model.set_classes(list(classes))
        results = self._model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        return self._normalize_results(results)

    @staticmethod
    def _normalize_results(results: Any) -> list[RawDetection]:
        detections: list[RawDetection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            clss = boxes.cls.tolist()
            for box, conf, cls_id in zip(xyxy, confs, clss, strict=False):
                label = names.get(int(cls_id), str(int(cls_id)))
                detections.append(
                    RawDetection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(conf),
                        label=str(label),
                    )
                )
        return detections


class YoloWorldDetector:
    """满足 `Detector` 协议的 YOLO-World 检测器。

    参数：
    - `settings`：类别、阈值、输入尺寸、设备与超时配置；
    - `backend`：已构造的后端实例（测试注入用），给出后不再惰性创建；
    - `backend_factory`：后端构造函数，默认 `UltralyticsYoloWorldBackend`，
      测试可注入以模拟初始化失败。
    """

    def __init__(
        self,
        settings: YoloWorldSettings | None = None,
        *,
        backend: YoloWorldBackend | None = None,
        backend_factory: Callable[[YoloWorldSettings], YoloWorldBackend] | None = None,
    ) -> None:
        self._settings = settings or YoloWorldSettings()
        self._backend = backend
        self._backend_factory = backend_factory or UltralyticsYoloWorldBackend
        self._init_failed = False
        self._executor: ThreadPoolExecutor | None = None
        self.last_metrics: DetectionMetrics | None = None
        self.stats = DetectionStats()

    @property
    def settings(self) -> YoloWorldSettings:
        return self._settings

    @property
    def model_version(self) -> str:
        if self._backend is not None:
            return self._backend.model_version
        return self._settings.default_model_version

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        backend: YoloWorldBackend | None = None,
        backend_factory: Callable[[YoloWorldSettings], YoloWorldBackend] | None = None,
    ) -> YoloWorldDetector:
        return cls(
            YoloWorldSettings.from_config(config), backend=backend, backend_factory=backend_factory
        )

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.data
        if image is None:
            msg = (
                f"YOLO-World 需要真实像素数据，但帧 {frame.frame_id} 的 data 为空；"
                "请使用图片/视频输入源，或改用 Mock 检测器。"
            )
            raise PerceptionError(msg)

        backend = self._ensure_backend()
        start = time.perf_counter()
        try:
            raw = self._predict_with_timeout(backend, image)
        finally:
            latency = time.perf_counter() - start

        detections = self._normalize(raw, frame)
        metrics = self._record_metrics(frame, detections, latency)
        logger.info(
            "yolo-world 检测 frame=%s model=%s 输入=%sx%s imgsz=%s device=%s "
            "耗时=%.3fs 检测数=%s 类别=%s",
            frame.frame_id,
            metrics.model_version,
            metrics.input_width,
            metrics.input_height,
            metrics.imgsz,
            metrics.device,
            metrics.latency_seconds,
            metrics.detection_count,
            metrics.label_counts,
        )
        return detections

    def close(self) -> None:
        """释放推理线程池；流水线不强制调用，进程退出时兜底回收。"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def _ensure_backend(self) -> YoloWorldBackend:
        if self._backend is not None:
            return self._backend
        if self._init_failed:
            msg = "YOLO-World 后端此前初始化失败，已降级；跳过重复加载。"
            raise PerceptionError(msg)
        try:
            self._backend = self._backend_factory(self._settings)
        except PerceptionError:
            self._init_failed = True
            raise
        except Exception as exc:
            self._init_failed = True
            msg = f"YOLO-World 模型初始化失败，已降级：{exc}"
            raise PerceptionError(msg) from exc
        return self._backend

    def _predict_with_timeout(
        self, backend: YoloWorldBackend, image: Any
    ) -> Sequence[RawDetection]:
        settings = self._settings
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="yolo-world-infer"
            )
        future = self._executor.submit(
            backend.predict,
            image,
            classes=settings.classes,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            imgsz=settings.imgsz,
            device=settings.device,
        )
        try:
            return future.result(timeout=settings.timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            msg = f"YOLO-World 推理超过 {settings.timeout_seconds:.2f}s，已降级放弃该帧。"
            raise PerceptionError(msg) from exc
        except PerceptionError:
            raise
        except Exception as exc:
            msg = f"YOLO-World 推理失败，已降级：{exc}"
            raise PerceptionError(msg) from exc

    def _normalize(self, raw: Sequence[RawDetection], frame: Frame) -> list[Detection]:
        model_version = self.model_version
        detections: list[Detection] = []
        for item in raw:
            if item.confidence < self._settings.conf_threshold:
                continue
            bbox = self._clamp_bbox(item, frame)
            if bbox is None:
                continue
            detections.append(
                Detection(
                    label=item.label,
                    bbox=bbox,
                    confidence=min(1.0, max(0.0, item.confidence)),
                    model_version=model_version,
                    detected_at=frame.captured_at,
                )
            )
        return detections

    @staticmethod
    def _clamp_bbox(item: RawDetection, frame: Frame) -> BBox | None:
        """把 xyxy 裁剪到帧内并转换为 BBox；退化框（宽或高 <= 0）丢弃。"""
        x1 = max(0.0, min(item.x1, frame.width))
        y1 = max(0.0, min(item.y1, frame.height))
        x2 = max(0.0, min(item.x2, frame.width))
        y2 = max(0.0, min(item.y2, frame.height))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None
        return BBox(x=x1, y=y1, width=width, height=height)

    def _record_metrics(
        self, frame: Frame, detections: Sequence[Detection], latency: float
    ) -> DetectionMetrics:
        label_counts: dict[str, int] = {}
        for detection in detections:
            label_counts[detection.label] = label_counts.get(detection.label, 0) + 1
        metrics = DetectionMetrics(
            frame_id=frame.frame_id,
            model_version=self.model_version,
            input_width=frame.width,
            input_height=frame.height,
            imgsz=self._settings.imgsz,
            device=self._settings.device,
            latency_seconds=latency,
            detection_count=len(detections),
            label_counts=label_counts,
        )
        self.last_metrics = metrics
        self.stats.record(metrics)
        return metrics
