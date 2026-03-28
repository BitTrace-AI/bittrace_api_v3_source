"""Deterministic JSON artifact helpers for the additive V3 contract layer."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from bittrace.v3.contracts import (
    ArtifactContract,
    ArtifactRef,
    ContractValidationError,
    TOP_LEVEL_ARTIFACT_TYPES,
)


ARTIFACT_KIND_REGISTRY = MappingProxyType(
    {artifact_type.KIND: artifact_type for artifact_type in TOP_LEVEL_ARTIFACT_TYPES}
)


def artifact_kind_registry() -> Mapping[str, type[ArtifactContract]]:
    """Return the immutable top-level artifact registry keyed by `kind`."""

    return ARTIFACT_KIND_REGISTRY


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def compute_json_sha256(payload: ArtifactContract | Mapping[str, object]) -> str:
    """Compute the SHA-256 digest of the canonical JSON artifact bytes."""

    if isinstance(payload, ArtifactContract):
        raw_payload: Mapping[str, object] = payload.to_dict()
    else:
        raw_payload = payload
    return hashlib.sha256(_canonical_json_bytes(raw_payload)).hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest of a written artifact file."""

    digest = hashlib.sha256()
    artifact_path = Path(path)
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_artifact(path: str | Path, artifact: ArtifactContract) -> ArtifactRef:
    """Write a top-level contract artifact and return a resolved artifact ref."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.to_dict()
    artifact_path.write_bytes(_canonical_json_bytes(payload))
    return ArtifactRef(
        kind=artifact.kind,
        schema_version=artifact.schema_version,
        path=str(artifact_path.resolve()),
        sha256=compute_file_sha256(artifact_path),
    )


def load_json_artifact(path: str | Path) -> ArtifactContract:
    """Load a JSON artifact via the top-level kind registry."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError("Artifact payload must deserialize to a JSON object.")
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise ContractValidationError("Artifact payload must include string field `kind`.")
    artifact_type = ARTIFACT_KIND_REGISTRY.get(kind)
    if artifact_type is None:
        known = ", ".join(sorted(ARTIFACT_KIND_REGISTRY))
        raise ContractValidationError(
            f"Unknown V3 artifact kind `{kind}`. Known kinds: {known}."
        )
    return artifact_type.from_dict(payload)


def load_json_artifact_ref(ref: ArtifactRef, *, validate_sha256: bool = True) -> ArtifactContract:
    """Load an artifact through an `ArtifactRef`, optionally verifying the stored digest."""

    artifact = load_json_artifact(ref.path)
    if validate_sha256 and ref.sha256 is not None:
        digest = compute_file_sha256(ref.path)
        if digest != ref.sha256:
            raise ContractValidationError(
                f"Artifact SHA-256 mismatch for `{ref.path}`: expected `{ref.sha256}`, got `{digest}`."
            )
    return artifact


__all__ = [
    "ARTIFACT_KIND_REGISTRY",
    "artifact_kind_registry",
    "compute_file_sha256",
    "compute_json_sha256",
    "load_json_artifact",
    "load_json_artifact_ref",
    "write_json_artifact",
]
