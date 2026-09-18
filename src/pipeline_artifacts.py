"""Small provenance helpers for resumable pipeline stages."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class ArtifactStatus(str, Enum):
    """Reusable-state classification for a pipeline stage."""

    MISSING = "missing"
    REUSABLE = "reusable"
    LEGACY_ADOPTABLE = "legacy_adoptable"
    STALE_DEPENDENCY = "stale_dependency"
    INCOMPATIBLE = "incompatible"


@dataclass
class StageCheck:
    """Result of checking whether a stage can be reused."""

    status: ArtifactStatus
    reason: str = ""
    dimensions: dict[str, int] = field(default_factory=dict)

    @property
    def reusable(self) -> bool:
        return self.status in {
            ArtifactStatus.REUSABLE,
            ArtifactStatus.LEGACY_ADOPTABLE,
        }


def provenance_path(primary: Path) -> Path:
    """Return the sidecar path used to describe a stage output."""
    return primary.with_name(primary.name + ".provenance.json")


def _normalise_outputs(outputs: Iterable[Path]) -> list[str]:
    return sorted(str(Path(path)) for path in outputs)


def classify_outputs(
    outputs: Iterable[Path],
    sidecar: Path,
    env_name: str,
    agent_name: str,
) -> StageCheck:
    """Classify a stage from its expected files and provenance sidecar."""
    output_paths = [Path(path) for path in outputs]
    missing = [str(path) for path in output_paths if not path.exists()]
    if missing:
        return StageCheck(ArtifactStatus.MISSING, "missing: " + ", ".join(missing))

    if not sidecar.exists():
        return StageCheck(
            ArtifactStatus.LEGACY_ADOPTABLE,
            "outputs exist without current provenance",
        )

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return StageCheck(ArtifactStatus.INCOMPATIBLE, f"invalid provenance: {exc}")

    if data.get("environment") != env_name:
        return StageCheck(
            ArtifactStatus.INCOMPATIBLE,
            f"provenance environment is {data.get('environment')!r}",
        )
    if data.get("agent") != agent_name:
        return StageCheck(
            ArtifactStatus.INCOMPATIBLE,
            f"provenance agent is {data.get('agent')!r}",
        )
    if data.get("status") == ArtifactStatus.STALE_DEPENDENCY.value:
        return StageCheck(
            ArtifactStatus.STALE_DEPENDENCY,
            data.get("reason", "stage was marked stale"),
        )

    recorded_outputs = data.get("outputs")
    if recorded_outputs and sorted(recorded_outputs) != _normalise_outputs(output_paths):
        return StageCheck(
            ArtifactStatus.LEGACY_ADOPTABLE,
            "outputs differ from provenance but are present",
        )

    dimensions = data.get("dimensions") or {}
    return StageCheck(ArtifactStatus.REUSABLE, dimensions=dimensions)


def write_provenance(
    path: Path,
    stage: str,
    env_name: str,
    agent_name: str,
    outputs: Iterable[Path],
    dimensions: dict[str, int],
    *,
    adopted: bool = False,
) -> None:
    """Write a compact JSON sidecar for a successful stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "environment": env_name,
        "agent": agent_name,
        "status": ArtifactStatus.REUSABLE.value,
        "adopted": bool(adopted),
        "outputs": _normalise_outputs(outputs),
        "dimensions": dimensions,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def mark_provenance_stale(
    path: Path,
    stage: str,
    env_name: str,
    agent_name: str,
    reason: str,
) -> None:
    """Record that a stage is intentionally being regenerated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "environment": env_name,
        "agent": agent_name,
        "status": ArtifactStatus.STALE_DEPENDENCY.value,
        "reason": reason,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
