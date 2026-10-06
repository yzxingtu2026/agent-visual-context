"""One-to-one geometric face-to-person association."""

from __future__ import annotations

from ..domain import FaceObservation, TrackedObject


def associate_faces(
    faces: list[FaceObservation], objects: list[TrackedObject], *, min_quality: float = 0.5
) -> list[FaceObservation]:
    people = [obj for obj in objects if obj.label == "person"]
    candidates: dict[int, list[tuple[float, str]]] = {}
    for index, face in enumerate(faces):
        if face.quality < min_quality:
            continue
        fx, fy = face.bbox.center
        for person in people:
            box = person.bbox
            if (
                box.x <= fx <= box.x + box.width
                and box.y <= fy <= box.y + box.height * 0.55
                and face.bbox.width <= box.width * 0.85
            ):
                distance = (
                    abs(fx - box.center[0]) / box.width
                    + abs(fy - (box.y + box.height * 0.2)) / box.height
                )
                candidates.setdefault(index, []).append((distance, person.track_id))
    # Ambiguous faces and competing faces remain unassigned.
    proposals: dict[int, str] = {}
    for index, choices in candidates.items():
        choices.sort()
        if len(choices) == 1 or choices[1][0] - choices[0][0] >= 0.15:
            proposals[index] = choices[0][1]
    counts: dict[str, int] = {}
    for key in proposals.values():
        counts[key] = counts.get(key, 0) + 1
    return [
        face.model_copy(update={"track_id": proposals[index]})
        if index in proposals and counts[proposals[index]] == 1
        else face
        for index, face in enumerate(faces)
    ]
