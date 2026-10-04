"""RelateAnything 关系推理适配器（PoC-4）。

本模块是模型调用的唯一落点：`domain/`、`temporal/`、`context/` 不得引用此处或任何
具体模型。适配器遵守 `perception/base.py::RelationReasoner` 协议，因此可被 Mock 推理器
无缝替换，时间线与上层模块无需修改。

设计要点：
- **可插拔后端**：真实推理走 `RelSggBackend`（惰性加载 relsgg），
  测试与离线验证可注入假后端，无需安装 torch 或联网下载权重；
- **归一化**：模型输出统一映射为领域层 `Relation`（subject/target 绑定 track_id、
  predicate 经白名单过滤、置信度裁剪、时间戳取自当前帧、携带模型版本）；
- **降级**：模型初始化失败、推理异常、推理超时、缺少像素数据都抛出 `PerceptionError`，
  由 `runtime.Pipeline._call()` 捕获并降级为"该帧无关系"，流水线继续运行；
  空目标列表是正常情况，直接返回空结果不触发推理；
- **可观测**：每帧记录模型版本、推理耗时、输入目标数、关系数量与谓词分布，
  通过 `logging` 输出并累积到 `RelateAnythingReasoner.stats`。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import AppConfig
from ..domain import Frame, Relation, TrackedObject, TrackRef
from ..errors import PerceptionError
from ..logging_setup import get_logger
from ._relsgg import require_relsgg

logger = get_logger("perception.relate_anything")

# 首批关系词白名单：覆盖大屏场景相关的空间与交互关系。
DEFAULT_VOCABULARY: tuple[str, ...] = (
    "looking_at",
    "facing",
    "holding",
    "pointing_at",
    "touching",
    "near",
)


@dataclass(frozen=True, slots=True)
class RawTriplet:
    """后端返回的原始关系三元组。

    后端负责把模型张量转换为纯 Python 数值，使 `RelateAnythingReasoner` 不感知
    torch/relsgg 类型。subject_index 和 object_index 对应输入 boxes 数组的行索引。
    """

    subject_index: int
    predicate: str
    object_index: int
    confidence: float


@dataclass(frozen=True, slots=True)
class RelateAnythingSettings:
    """RelateAnything 关系推理器的可配置项，可由 `AppConfig` 派生。"""

    vocabulary: tuple[str, ...] = DEFAULT_VOCABULARY
    conf_threshold: float = 0.3
    topk: int = 10
    device: str = "cpu"
    timeout_seconds: float = 15.0
    model_name: str = "maelic/relsgg-vits16plus"

    @property
    def default_model_version(self) -> str:
        return f"relate-anything:{self.model_name}"

    @classmethod
    def from_config(cls, config: AppConfig) -> RelateAnythingSettings:
        return cls(
            vocabulary=tuple(config.reasoner_vocabulary),
            conf_threshold=config.reasoner_conf_threshold,
            topk=config.reasoner_topk,
            device=config.reasoner_device,
            timeout_seconds=config.reasoner_timeout_seconds,
            model_name=config.reasoner_model_name,
        )


@dataclass(slots=True)
class RelationMetrics:
    """单帧关系推理的可观测信息。"""

    frame_id: str
    model_version: str
    input_objects: int
    device: str
    latency_seconds: float
    relation_count: int
    predicate_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class RelationStats:
    """跨帧累积的关系推理统计，用于记录一轮延迟/数量数据。"""

    frames: int = 0
    relations: int = 0
    total_latency_seconds: float = 0.0
    min_latency_seconds: float | None = None
    max_latency_seconds: float | None = None

    @property
    def mean_latency_seconds(self) -> float:
        return self.total_latency_seconds / self.frames if self.frames else 0.0

    def record(self, metrics: RelationMetrics) -> None:
        self.frames += 1
        self.relations += metrics.relation_count
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


class RelateAnythingBackend(Protocol):
    """RelateAnything 推理后端协议：真实实现惰性加载 relsgg，测试可注入假后端。"""

    @property
    def model_version(self) -> str: ...

    def predict(
        self,
        image: Any,
        boxes: Any,
        box_labels: Sequence[str],
        *,
        vocabulary: Sequence[str],
        conf_threshold: float,
        topk: int,
    ) -> Sequence[RawTriplet]: ...


class RelSggBackend:
    """基于 relsgg 的真实 RelateAnything 后端。

    构造时即加载模型（首次会从 HuggingFace 下载权重），失败抛出异常，
    由 `RelateAnythingReasoner` 转换为 `PerceptionError` 触发降级。
    """

    def __init__(self, settings: RelateAnythingSettings) -> None:
        self._settings = settings
        relsgg = require_relsgg()
        self._model: Any = relsgg.RelateAnything.from_pretrained(
            settings.model_name, device=settings.device
        )
        self._model.set_vocabulary(list(settings.vocabulary))
        self._model_version = f"relate-anything:{settings.model_name}"
        logger.info(
            "RelateAnything 后端就绪 model=%s device=%s vocabulary=%s",
            settings.model_name,
            settings.device,
            settings.vocabulary,
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(
        self,
        image: Any,
        boxes: Any,
        box_labels: Sequence[str],
        *,
        vocabulary: Sequence[str],
        conf_threshold: float,
        topk: int,
    ) -> Sequence[RawTriplet]:
        self._model.set_vocabulary(list(vocabulary))
        raw_results = self._model.predict(
            image, boxes, box_labels=list(box_labels), topk=topk
        )
        return self._normalize_results(raw_results, conf_threshold)

    @staticmethod
    def _normalize_results(
        raw_results: Any, conf_threshold: float
    ) -> list[RawTriplet]:
        """将 relsgg 输出的三元组列表转换为 RawTriplet。

        relsgg predict 返回格式预期为 (subject_idx, predicate, object_idx, score) 元组列表，
        具体格式以实际模型输出为准，此处做防御性解析。
        """
        triplets: list[RawTriplet] = []
        if not raw_results:
            return triplets
        for item in raw_results:
            try:
                if hasattr(item, "__len__") and len(item) >= 4:
                    subject_idx = int(item[0])
                    predicate = str(item[1])
                    object_idx = int(item[2])
                    confidence = float(item[3])
                elif isinstance(item, dict):
                    subject_idx = int(item["subject_index"])
                    predicate = str(item["predicate"])
                    object_idx = int(item["object_index"])
                    confidence = float(item["confidence"])
                else:
                    continue
                if confidence < conf_threshold:
                    continue
                triplets.append(
                    RawTriplet(
                        subject_index=subject_idx,
                        predicate=predicate,
                        object_index=object_idx,
                        confidence=confidence,
                    )
                )
            except (TypeError, ValueError, KeyError, IndexError):
                continue
        return triplets


class RelateAnythingReasoner:
    """满足 `RelationReasoner` 协议的 RelateAnything 关系推理器。

    参数：
    - `settings`：关系词白名单、阈值、设备与超时配置；
    - `backend`：已构造的后端实例（测试注入用），给出后不再惰性创建；
    - `backend_factory`：后端构造函数，默认 `RelSggBackend`，
      测试可注入以模拟初始化失败。
    """

    def __init__(
        self,
        settings: RelateAnythingSettings | None = None,
        *,
        backend: RelateAnythingBackend | None = None,
        backend_factory: Callable[[RelateAnythingSettings], RelateAnythingBackend] | None = None,
    ) -> None:
        self._settings = settings or RelateAnythingSettings()
        self._backend = backend
        self._backend_factory = backend_factory or RelSggBackend
        self._init_failed = False
        self._executor: ThreadPoolExecutor | None = None
        self.last_metrics: RelationMetrics | None = None
        self.stats = RelationStats()

    @property
    def settings(self) -> RelateAnythingSettings:
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
        backend: RelateAnythingBackend | None = None,
        backend_factory: Callable[[RelateAnythingSettings], RelateAnythingBackend] | None = None,
    ) -> RelateAnythingReasoner:
        return cls(
            RelateAnythingSettings.from_config(config),
            backend=backend,
            backend_factory=backend_factory,
        )

    def infer(self, frame: Frame, objects: list[TrackedObject]) -> list[Relation]:
        """对当前帧的跟踪目标执行关系推理，返回归一化的 Relation 列表。"""
        if not objects:
            return []

        image = frame.data
        if image is None:
            msg = (
                f"RelateAnything 需要真实像素数据，但帧 {frame.frame_id} 的 data 为空；"
                "请使用图片/视频输入源，或改用 Mock 关系推理器。"
            )
            raise PerceptionError(msg)

        backend = self._ensure_backend()
        boxes, box_labels = self._prepare_inputs(objects)

        start = time.perf_counter()
        try:
            raw = self._predict_with_timeout(backend, image, boxes, box_labels)
        finally:
            latency = time.perf_counter() - start

        relations = self._normalize(raw, objects, frame)
        metrics = self._record_metrics(frame, objects, relations, latency)
        logger.info(
            "relate-anything 推理 frame=%s model=%s 目标数=%s device=%s "
            "耗时=%.3fs 关系数=%s 谓词=%s",
            frame.frame_id,
            metrics.model_version,
            metrics.input_objects,
            metrics.device,
            metrics.latency_seconds,
            metrics.relation_count,
            metrics.predicate_counts,
        )
        return relations

    def close(self) -> None:
        """释放推理线程池；流水线不强制调用，进程退出时兜底回收。"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def _ensure_backend(self) -> RelateAnythingBackend:
        if self._backend is not None:
            return self._backend
        if self._init_failed:
            msg = "RelateAnything 后端此前初始化失败，已降级；跳过重复加载。"
            raise PerceptionError(msg)
        try:
            self._backend = self._backend_factory(self._settings)
        except PerceptionError:
            self._init_failed = True
            raise
        except Exception as exc:
            self._init_failed = True
            msg = f"RelateAnything 模型初始化失败，已降级：{exc}"
            raise PerceptionError(msg) from exc
        return self._backend

    def _prepare_inputs(
        self, objects: Sequence[TrackedObject]
    ) -> tuple[Any, list[str]]:
        """将 TrackedObject 列表转换为模型所需的 boxes 数组和标签列表。

        boxes 格式为 numpy float32 数组 [[x1, y1, x2, y2], ...]，像素坐标。
        """
        import numpy as np

        boxes_list: list[list[float]] = []
        box_labels: list[str] = []
        for obj in objects:
            x1 = obj.bbox.x
            y1 = obj.bbox.y
            x2 = obj.bbox.x + obj.bbox.width
            y2 = obj.bbox.y + obj.bbox.height
            boxes_list.append([x1, y1, x2, y2])
            box_labels.append(obj.label)
        return np.array(boxes_list, dtype=np.float32), box_labels

    def _predict_with_timeout(
        self,
        backend: RelateAnythingBackend,
        image: Any,
        boxes: Any,
        box_labels: Sequence[str],
    ) -> Sequence[RawTriplet]:
        settings = self._settings
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="relate-anything-infer"
            )
        future = self._executor.submit(
            backend.predict,
            image,
            boxes,
            box_labels,
            vocabulary=settings.vocabulary,
            conf_threshold=settings.conf_threshold,
            topk=settings.topk,
        )
        try:
            return future.result(timeout=settings.timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            msg = (
                f"RelateAnything 推理超过 {settings.timeout_seconds:.2f}s，已降级放弃该帧。"
            )
            raise PerceptionError(msg) from exc
        except PerceptionError:
            raise
        except Exception as exc:
            msg = f"RelateAnything 推理失败，已降级：{exc}"
            raise PerceptionError(msg) from exc

    def _normalize(
        self,
        raw: Sequence[RawTriplet],
        objects: Sequence[TrackedObject],
        frame: Frame,
    ) -> list[Relation]:
        """将原始三元组映射为领域层 Relation，按白名单过滤谓词。"""
        model_version = self.model_version
        vocabulary_set = set(self._settings.vocabulary)
        relations: list[Relation] = []

        for triplet in raw:
            # 索引越界防御
            if triplet.subject_index >= len(objects) or triplet.object_index >= len(objects):
                continue
            if triplet.subject_index < 0 or triplet.object_index < 0:
                continue
            # 白名单过滤（大小写不敏感匹配）
            predicate_normalized = triplet.predicate.lower().replace(" ", "_")
            if predicate_normalized not in vocabulary_set:
                # 尝试原始形式
                if triplet.predicate not in vocabulary_set:
                    continue
                predicate_normalized = triplet.predicate

            subject_obj = objects[triplet.subject_index]
            object_obj = objects[triplet.object_index]
            # 自引用跳过
            if subject_obj.track_id == object_obj.track_id:
                continue

            relations.append(
                Relation(
                    subject=TrackRef(
                        track_id=subject_obj.track_id, label=subject_obj.label
                    ),
                    predicate=predicate_normalized,
                    target=TrackRef(
                        track_id=object_obj.track_id, label=object_obj.label
                    ),
                    confidence=min(1.0, max(0.0, triplet.confidence)),
                    model_version=model_version,
                    observed_at=frame.captured_at,
                )
            )
        return relations

    def _record_metrics(
        self,
        frame: Frame,
        objects: Sequence[TrackedObject],
        relations: Sequence[Relation],
        latency: float,
    ) -> RelationMetrics:
        predicate_counts: dict[str, int] = {}
        for relation in relations:
            predicate_counts[relation.predicate] = (
                predicate_counts.get(relation.predicate, 0) + 1
            )
        metrics = RelationMetrics(
            frame_id=frame.frame_id,
            model_version=self.model_version,
            input_objects=len(objects),
            device=self._settings.device,
            latency_seconds=latency,
            relation_count=len(relations),
            predicate_counts=predicate_counts,
        )
        self.last_metrics = metrics
        self.stats.record(metrics)
        return metrics
