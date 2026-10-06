"""Local InsightFace adapter. Model files are never downloaded by this module."""

from __future__ import annotations

import importlib
import importlib.metadata
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Protocol, cast

from ..domain import BBox, FaceObservation, Frame
from ..errors import PerceptionError


class InsightFaceBackend(Protocol):
    def get(self, image: Any) -> list[Any]: ...


def _load_backend(model_dir: Path, model_name: str, device: str) -> InsightFaceBackend:
    weights = model_dir / "models" / model_name
    if not weights.is_dir() or not list(weights.glob("*.onnx")):
        raise PerceptionError(f"InsightFace 本地权重缺失：{model_dir / 'models' / model_name}")
    try:
        face_analysis = importlib.import_module("insightface.app").FaceAnalysis
    except ImportError as exc:
        raise PerceptionError("缺少 InsightFace；安装 face extra 后使用本地授权权重") from exc
    app = face_analysis(
        name=model_name,
        root=str(model_dir),
        providers=["CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"],
    )
    app.prepare(ctx_id=-1 if device == "cpu" else 0)
    return cast(InsightFaceBackend, app)


def _age_band(age: Any) -> str | None:
    if age is None or not 0 <= float(age) <= 110:
        return None
    return (
        f"{min(int(float(age)) // 10 * 10, 80)}+"
        if float(age) >= 80
        else f"{int(float(age)) // 10 * 10}-{int(float(age)) // 10 * 10 + 9}"
    )


class InsightFaceAnalyzer:
    """Single-flight inference with a per-frame timeout and local-only weights."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_name: str = "buffalo_l",
        device: str = "cpu",
        timeout_seconds: float = 2.0,
        min_quality: float = 0.5,
        backend_factory: Callable[[], InsightFaceBackend] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.model_name = model_name
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.min_quality = min_quality
        self._backend_factory = backend_factory or (
            lambda: _load_backend(model_dir, model_name, device)
        )
        self._backend: InsightFaceBackend | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avc-face")
        self._pending: Future[list[Any]] | None = None
        self.last_latency_ms = 0.0
        self.calls = 0
        self.failures = 0

    @property
    def model_version(self) -> str:
        try:
            sdk = importlib.metadata.version("insightface")
        except importlib.metadata.PackageNotFoundError:
            sdk = "uninstalled"
        return f"insightface-{sdk}/{self.model_name}"

    def analyze(self, frame: Frame) -> list[FaceObservation]:
        if frame.data is None:
            raise PerceptionError("人脸分析缺少像素数据")
        if self._pending is not None:
            if not self._pending.done():
                raise PerceptionError("上一帧人脸推理仍在运行")
            self._pending = None
        start = time.monotonic()
        self.calls += 1
        self._pending = self._executor.submit(self._infer, frame.data)
        try:
            raw = self._pending.result(timeout=self.timeout_seconds)
        except TimeoutError as exc:
            self.failures += 1
            raise PerceptionError(f"InsightFace 推理超过 {self.timeout_seconds:.2f}s") from exc
        except PerceptionError:
            self.failures += 1
            raise
        except Exception as exc:
            self.failures += 1
            raise PerceptionError(f"InsightFace 推理失败：{type(exc).__name__}") from exc
        finally:
            self.last_latency_ms = (time.monotonic() - start) * 1000
        self._pending = None
        faces: list[FaceObservation] = []
        for item in raw:
            box = item.bbox
            x0, y0, x1, y1 = (float(value) for value in box[:4])
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(frame.width), x1), min(float(frame.height), y1)
            if x1 <= x0 or y1 <= y0:
                continue
            quality = max(0.0, min(1.0, float(getattr(item, "det_score", 0.0))))
            usable = quality >= self.min_quality and min(x1 - x0, y1 - y0) >= 32
            vector = getattr(item, "normed_embedding", None) if usable else None
            gender = getattr(item, "gender", None) if usable else None
            landmarks = getattr(item, "kps", None)
            faces.append(
                FaceObservation(
                    bbox=BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                    landmarks=tuple((float(x), float(y)) for x, y in landmarks)
                    if landmarks is not None
                    else (),
                    quality=quality,
                    model_version=self.model_version,
                    embedding=tuple(float(value) for value in vector)
                    if vector is not None
                    else None,
                    age_band=_age_band(getattr(item, "age", None)) if usable else None,
                    apparent_gender=("female" if gender == 0 else "male" if gender == 1 else None),
                )
            )
        return faces

    def _infer(self, image: Any) -> list[Any]:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend.get(image)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
