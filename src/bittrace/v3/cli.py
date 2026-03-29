"""Thin CLI wrappers for the additive V3 campaign and verification flows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from bittrace.v3.artifacts import compute_file_sha256, load_json_artifact
from bittrace.v3.contracts import (
    ArtifactContract,
    ArtifactRef,
    CampaignManifest,
    CampaignRequest,
    ContractValidationError,
    GoldenVectorEntry,
    StageRequest,
)
from bittrace.v3.pipeline import run_canonical_campaign
from bittrace.v3.verify import ParityObservation, emit_canonical_verification_artifacts


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_object(path: str | Path, model_type: type[Any]) -> Any:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{path}` must deserialize to a JSON object.")
    return model_type.from_dict(payload)


def _load_list(path: str | Path, item_type: type[Any]) -> tuple[Any, ...]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ContractValidationError(f"`{path}` must deserialize to a JSON array.")
    items: list[Any] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ContractValidationError(
                f"`{path}[{index}]` must deserialize to a JSON object."
            )
        if hasattr(item_type, "from_dict"):
            items.append(item_type.from_dict(item))
            continue
        items.append(item_type(**dict(item)))
    return tuple(items)


def _load_artifact(path: str | Path, artifact_type: type[Any]) -> Any:
    artifact = load_json_artifact(path)
    if not isinstance(artifact, artifact_type):
        raise ContractValidationError(
            f"`{path}` must resolve to `{artifact_type.__name__}`."
        )
    return artifact


def _artifact_ref_from_path(path: str | Path) -> ArtifactRef:
    artifact = load_json_artifact(path)
    if not isinstance(artifact, ArtifactContract):
        raise ContractValidationError(f"`{path}` must resolve to a V3 artifact.")
    resolved_path = str(Path(path).resolve())
    return ArtifactRef(
        kind=artifact.kind,
        schema_version=artifact.schema_version,
        path=resolved_path,
        sha256=compute_file_sha256(resolved_path),
    )


def _load_factory(target: str) -> Any:
    module_name, separator, attr_name = target.partition(":")
    if separator == "" or attr_name == "":
        raise ContractValidationError(
            "`stage-config-factory` must use the form `module_path:function_name`."
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name, None)
    if factory is None or not callable(factory):
        raise ContractValidationError(
            f"`stage-config-factory` target `{target}` did not resolve to a callable."
        )
    return factory


def _run_campaign(args: argparse.Namespace) -> int:
    campaign_request = _load_object(args.campaign_request, CampaignRequest)
    campaign: CampaignRequest | CampaignManifest = campaign_request
    campaign_manifest_ref: ArtifactRef | None = None
    if args.campaign_manifest is not None:
        campaign = _load_artifact(args.campaign_manifest, CampaignManifest)
        campaign_manifest_ref = _artifact_ref_from_path(args.campaign_manifest)

    stage_config_factory = _load_factory(args.stage_config_factory)
    stage_configs = stage_config_factory(Path(campaign_request.output_dir))
    run_result = run_canonical_campaign(
        campaign,
        stage_configs=stage_configs,
        campaign_manifest_ref=campaign_manifest_ref,
        output_path=args.output_path,
    )
    if run_result.failed_stage_execution is not None:
        stage_result = run_result.failed_stage_execution.stage_result
        blocker = (
            f": {stage_result.exact_blocker}"
            if stage_result.exact_blocker is not None
            else ""
        )
        print(
            f"failed stage: {run_result.failed_stage_execution.stage_key.value}{blocker}",
            file=sys.stderr,
        )
        return 1
    print(run_result.campaign_result_ref.path)
    return 0


def _verify_campaign(args: argparse.Namespace) -> int:
    request = _load_artifact(args.stage_request, StageRequest)
    golden_vector_entries = _load_list(args.golden_vectors, GoldenVectorEntry)
    observations = _load_list(args.observations, ParityObservation)
    result = emit_canonical_verification_artifacts(
        request,
        source_freeze_export_manifest_ref=_artifact_ref_from_path(
            args.freeze_export_manifest
        ),
        deep_anchor_artifact_ref=_artifact_ref_from_path(args.deep_anchor_artifact),
        frontend_export_reference_ref=_artifact_ref_from_path(
            args.frontend_export_reference
        ),
        golden_vector_entries=golden_vector_entries,
        observations=observations,
    )
    if result.parity_report.pass_fail.value == "FAIL":
        print(
            f"verification failed: {result.parity_report.mismatch_count} mismatches",
            file=sys.stderr,
        )
        return 1
    print(result.parity_report_ref.path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal V3 CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bittrace-v3",
        description="Thin BitTrace V3 CLI wrappers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_campaign = subparsers.add_parser(
        "run-campaign",
        help="Execute a canonical V3 campaign.",
    )
    run_campaign.add_argument(
        "--campaign-request",
        required=True,
        help="Path to a JSON-serialized CampaignRequest payload.",
    )
    run_campaign.add_argument(
        "--stage-config-factory",
        required=True,
        help="Dotted callable target in the form module_path:function_name.",
    )
    run_campaign.add_argument(
        "--campaign-manifest",
        help="Optional path to a CampaignManifest artifact to replay.",
    )
    run_campaign.add_argument(
        "--output-path",
        help="Optional explicit path for the emitted campaign-result artifact.",
    )
    run_campaign.set_defaults(handler=_run_campaign)

    verify_campaign = subparsers.add_parser(
        "verify-campaign",
        help="Emit canonical V3 verification artifacts from frozen/exported outputs.",
    )
    verify_campaign.add_argument(
        "--stage-request",
        required=True,
        help="Path to the parity-verification StageRequest artifact.",
    )
    verify_campaign.add_argument(
        "--freeze-export-manifest",
        required=True,
        help="Path to the frozen/exported S6 FreezeExportManifest artifact.",
    )
    verify_campaign.add_argument(
        "--deep-anchor-artifact",
        required=True,
        help="Path to the frozen/exported S6 DeepAnchorArtifact artifact.",
    )
    verify_campaign.add_argument(
        "--frontend-export-reference",
        required=True,
        help="Path to the frozen/exported S6 FrontendExportReference artifact.",
    )
    verify_campaign.add_argument(
        "--golden-vectors",
        required=True,
        help="Path to a JSON array of GoldenVectorEntry payloads.",
    )
    verify_campaign.add_argument(
        "--observations",
        required=True,
        help="Path to a JSON array of ParityObservation payloads.",
    )
    verify_campaign.set_defaults(handler=_verify_campaign)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the minimal V3 CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ContractValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
