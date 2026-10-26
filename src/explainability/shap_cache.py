"""Versioned, provenance-checked storage for per-line SHAP rankings."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ShapSettings:
    """Configuration that can affect the computed feature rankings."""

    top_k: int = 8
    max_rows_per_line: int = 200
    repeats: int = 5
    random_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("top_k", "max_rows_per_line", "repeats"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CacheValidation:
    """Result of checking a saved ranking cache against current artifacts."""

    compatible: bool
    reason: str
    rankings: dict[str, list[str]] | None = None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_classifier_contract(
    metadata_path: str | Path,
) -> tuple[list[str], dict[str, int]]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    features = metadata.get("features")
    line_map = metadata.get("line_map")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError("classifier metadata has no valid ordered feature list")
    if not isinstance(line_map, dict):
        raise ValueError("classifier metadata has no valid line mapping")
    try:
        normalized_line_map = {
            str(line_name): int(line_id) for line_name, line_id in line_map.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("classifier metadata has an invalid line mapping") from exc
    return list(features), normalized_line_map


def build_provenance(
    rows_path: str | Path,
    classifier_path: str | Path,
    metadata_path: str | Path,
    settings: ShapSettings,
) -> dict[str, Any]:
    """Describe every input that affects a saved ranking cache."""
    features, line_map = _load_classifier_contract(metadata_path)
    return {
        "classifier_sha256": _sha256(classifier_path),
        "classifier_metadata_sha256": _sha256(metadata_path),
        "failure_training_rows_sha256": _sha256(rows_path),
        "classifier_features": features,
        "line_map": line_map,
        "shap_settings": settings.as_dict(),
    }


def _validate_rankings_shape(
    rankings: Any,
    line_map: Mapping[str, int],
) -> dict[str, list[str]]:
    if not isinstance(rankings, dict):
        raise ValueError("rankings must be a JSON object")
    normalized: dict[str, list[str]] = {}
    for line_name, features in rankings.items():
        if not isinstance(line_name, str):
            raise ValueError("ranking line names must be strings")
        if line_name not in line_map:
            raise ValueError(f"rankings contain unknown line {line_name!r}")
        if not isinstance(features, list) or not all(
            isinstance(feature, str) for feature in features
        ):
            raise ValueError(f"ranking for {line_name!r} must be a list of features")
        normalized[line_name] = list(features)
    return normalized


def validate_rankings_cache(
    cache_path: str | Path,
    rows_path: str | Path,
    classifier_path: str | Path,
    metadata_path: str | Path,
    settings: ShapSettings = ShapSettings(),
) -> CacheValidation:
    """Return whether a ranking file is complete and matches current inputs."""
    path = Path(cache_path)
    if not path.is_file():
        return CacheValidation(False, "ranking cache is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return CacheValidation(False, f"ranking cache is malformed: {exc}")
    if not isinstance(payload, dict):
        return CacheValidation(False, "ranking cache is malformed: root must be an object")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return CacheValidation(False, "ranking cache schema is missing or unsupported")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return CacheValidation(False, "ranking cache provenance is missing or malformed")

    required = {
        "classifier_sha256",
        "classifier_metadata_sha256",
        "failure_training_rows_sha256",
        "classifier_features",
        "line_map",
        "shap_settings",
    }
    missing = sorted(required - set(provenance))
    if missing:
        return CacheValidation(
            False,
            "ranking cache provenance is incomplete; missing " + ", ".join(missing),
        )
    if "rankings" not in payload:
        return CacheValidation(False, "ranking cache is incomplete; rankings are missing")

    try:
        current = build_provenance(
            rows_path, classifier_path, metadata_path, settings
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return CacheValidation(False, f"current ranking inputs are invalid: {exc}")

    comparisons = (
        ("classifier_sha256", "classifier file hash changed"),
        ("failure_training_rows_sha256", "failure-training rows file hash changed"),
        ("classifier_features", "classifier feature order changed"),
        ("line_map", "line mapping changed"),
        ("shap_settings", "SHAP settings changed"),
        ("classifier_metadata_sha256", "classifier metadata file hash changed"),
    )
    for field, reason in comparisons:
        if provenance.get(field) != current[field]:
            return CacheValidation(False, reason)

    try:
        rankings = _validate_rankings_shape(payload["rankings"], current["line_map"])
    except ValueError as exc:
        return CacheValidation(False, f"ranking cache is malformed: {exc}")
    return CacheValidation(True, "all provenance fields match", rankings)


def write_rankings_cache(
    cache_path: str | Path,
    rankings: Mapping[str, list[str]],
    rows_path: str | Path,
    classifier_path: str | Path,
    metadata_path: str | Path,
    settings: ShapSettings = ShapSettings(),
    *,
    expected_provenance: Mapping[str, Any] | None = None,
) -> None:
    """Atomically write rankings and provenance as one cache document."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(
        rows_path, classifier_path, metadata_path, settings
    )
    if expected_provenance is not None and provenance != dict(expected_provenance):
        raise RuntimeError("SHAP ranking inputs changed while rankings were computed")
    normalized = _validate_rankings_shape(rankings, provenance["line_map"])
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "rankings": normalized,
        "provenance": provenance,
    }

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
