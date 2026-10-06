"""Conservative session-local tracking using motion and box overlap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..domain import BBox, Detection, Frame, TrackedObject


@dataclass(slots=True)
class _Track:
    value: TrackedObject
    previous_center: tuple[float, float] | None = None
    previous_at: datetime | None = None


def _iou(a: BBox, b: BBox) -> float:
    intersection = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)) * max(
        0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    )
    return intersection / (a.width * a.height + b.width * b.height - intersection)


class ShortTermTracker:
    """Tracks continuous appearances; uncertain crossings receive new IDs."""

    model_version = "motion-iou-1"

    def __init__(self, *, max_gap_seconds: float = 1.0, max_motion: float = 1.5) -> None:
        self.max_gap_seconds = max_gap_seconds
        self.max_motion = max_motion
        self._tracks: dict[str, _Track] = {}
        self._sequence = 0

    def reset(self) -> None:
        self._tracks.clear()
        # IDs must remain unique within the tracker session, even after reset.

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedObject]:
        now = frame.captured_at
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if 0 <= (now - track.value.last_seen).total_seconds() <= self.max_gap_seconds
        }
        candidates: list[tuple[float, str, int]] = []
        for key, track in self._tracks.items():
            old = track.value
            gap = (now - old.last_seen).total_seconds()
            for index, detection in enumerate(detections):
                if detection.label != old.label:
                    continue
                center = old.bbox.center
                if track.previous_center is not None and track.previous_at is not None:
                    span = (old.last_seen - track.previous_at).total_seconds()
                    if span > 0:
                        center = (
                            center[0] + (center[0] - track.previous_center[0]) * gap / span,
                            center[1] + (center[1] - track.previous_center[1]) * gap / span,
                        )
                distance = (
                    sum((center[axis] - detection.bbox.center[axis]) ** 2 for axis in (0, 1)) ** 0.5
                )
                scale = max(
                    old.bbox.width, old.bbox.height, detection.bbox.width, detection.bbox.height
                )
                normalized = distance / scale
                overlap = _iou(old.bbox, detection.bbox)
                if normalized <= self.max_motion and (overlap >= 0.05 or normalized <= 0.7):
                    candidates.append((normalized - overlap * 0.3, key, index))

        assigned: dict[int, str] = {}
        used: set[str] = set()
        for _, key, index in sorted(candidates):
            if key not in used and index not in assigned:
                assigned[index] = key
                used.add(key)

        result: list[TrackedObject] = []
        for index, detection in enumerate(detections):
            assigned_key = assigned.get(index)
            if assigned_key is None:
                self._sequence += 1
                key = f"{detection.label}-{self._sequence:06d}"
                first_seen = now
                previous_center = None
                previous_at = None
            else:
                key = assigned_key
                previous = self._tracks[key].value
                first_seen = previous.first_seen
                previous_center = previous.bbox.center
                previous_at = previous.last_seen
            value = TrackedObject(
                track_id=key,
                label=detection.label,
                bbox=detection.bbox,
                confidence=detection.confidence,
                first_seen=first_seen,
                last_seen=now,
                model_version=self.model_version,
            )
            self._tracks[key] = _Track(value, previous_center, previous_at)
            result.append(value)
        return result
