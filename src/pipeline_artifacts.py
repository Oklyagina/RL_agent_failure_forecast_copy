"""Artifact status and provenance helpers for the resumable pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROVENANCE_SCHEMA_VERSION = 1


class ArtifactStatus(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    INCOMPATIBLE = "INCOMPATIBLE"
    LEGACY_ADOPTABLE = "LEGACY_ADOPTABLE"
    STALE_DEPENDENCY = "STALE_DEPENDENCY"


@dataclass
class StageCheck:
    status: ArtifactStatus
    reason: str = ""
    dimensions: Dict[str, Any] = field(default_factory=dict)

    @property
    def reusable(self) -> bool:
        return self.status in {
            ArtifactStatus.VALID,
            ArtifactStatus.LEGACY_ADOPTABLE,
        }


def provenance_path(primary_output: Path) -> Path:
    """Return a sidecar path without changing the artifact's own suffix."""
    return primary_output.with_name(primary_output.name + ".provenance.json")


def identity_matches(payload: dict, env_name: str, agent_name: str) -> bool:
    return (
        payload.get("schema_version") == PROVENANCE_SCHEMA_VERSION
        and payload.get("state") == "ready"
        and payload.get("environment") == env_name
        and payload.get("agent") == agent_name
    )


def classify_outputs(
    outputs: Iterable[Path],
    sidecar: Path,
    env_name: str,
    agent_name: str,
) -> StageCheck:
    missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return StageCheck(ArtifactStatus.MISSING, "missing: " + ", ".join(missing))

    if not sidecar.is_file():
        return StageCheck(
            ArtifactStatus.LEGACY_ADOPTABLE,
            "outputs exist but have no provenance sidecar",
        )

    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return StageCheck(ArtifactStatus.INCOMPATIBLE, f"invalid provenance: {exc}")

    if payload.get("state") != "ready":
        return StageCheck(
            ArtifactStatus.INCOMPATIBLE,
            f"provenance state is {payload.get('state')!r}, expected 'ready'",
        )
    if not identity_matches(payload, env_name, agent_name):
        actual = f"{payload.get('environment')}/{payload.get('agent')}"
        return StageCheck(
            ArtifactStatus.INCOMPATIBLE,
            f"provenance belongs to {actual}, expected {env_name}/{agent_name}",
        )
    return StageCheck(ArtifactStatus.VALID, "provenance matches current .env")


def write_provenance(
    sidecar: Path,
    stage: str,
    env_name: str,
    agent_name: str,
    outputs: Iterable[Path],
    dimensions: Optional[Dict[str, Any]] = None,
    adopted: bool = False,
) -> None:
    """Atomically write a portable provenance sidecar after validation."""
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "state": "ready",
        "stage": stage,
        "environment": env_name,
        "agent": agent_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adopted_legacy_artifact": adopted,
        "outputs": [path.name for path in outputs],
        "dimensions": dimensions or {},
        "diagnostic_paths": {
            "sidecar": str(sidecar.resolve()),
            "outputs": [str(path.resolve()) for path in outputs],
        },
    }
    temporary = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(sidecar)


def mark_provenance_stale(
    sidecar: Path,
    stage: str,
    env_name: str,
    agent_name: str,
    reason: str,
) -> None:
    """Atomically prevent an interrupted stage from looking reusable."""
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "state": "stale",
        "stage": stage,
        "environment": env_name,
        "agent": agent_name,
        "reason": reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(sidecar)
