"""CLI for the frozen BitTrace V3 source lane."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from bittrace.v3 import (
    CAMPAIGN_RESULT_ARTIFACT_NAME,
    ContractValidationError,
    run_canonical_campaign,
)

from .full_binary_campaign import (
    DEFAULT_CONFIG_PATH as DEFAULT_SOURCE_PROFILE_CONFIG_PATH,
    DEFAULT_RUNS_ROOT,
    prepare_full_binary_campaign,
    write_campaign_request_json,
)
from .full_binary_verification import (
    DEFAULT_VERIFICATION_STAGE_DIRNAME,
    run_full_binary_verification,
)
from .leanlean_deployment_candidate import (
    DEFAULT_CONFIG_PATH as DEFAULT_DEPLOYMENT_CANDIDATE_CONFIG_PATH,
    prepare_leanlean_deployment_candidate,
    run_prepared_leanlean_deployment_candidate,
    write_leanlean_deployment_candidate_plan,
    write_persistence_prep_artifacts,
)
from .leanlean_persistence_tuning import (
    DEFAULT_CONFIG_PATH as DEFAULT_PERSISTENCE_CONFIG_PATH,
    load_leanlean_persistence_tuning_config,
    prepare_leanlean_persistence_tuning,
    run_prepared_leanlean_persistence_tuning,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bittrace",
        description=(
            "Supported BitTrace CLI for the "
            "temporal_threshold_36 + Lean-Lean shipping path."
        ),
        epilog=(
            "Supported surface: canonical campaign, verify/parity, canonical "
            "deployment-candidate, and the quiet/aggressive persistence profiles."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_supported_commands(subparsers)
    return parser


def register_supported_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    campaign = subparsers.add_parser(
        "campaign",
        help="Run the canonical source-lane freeze/export path.",
    )
    campaign.add_argument(
        "--config",
        default=str(DEFAULT_SOURCE_PROFILE_CONFIG_PATH),
        help="Path to the canonical source profile YAML.",
    )
    campaign.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier created under runs/<profile>/<run-id>.",
    )
    campaign.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Base directory that will contain the campaign run root.",
    )
    campaign.add_argument(
        "--campaign-seed",
        type=int,
        default=31,
        help="Deterministic campaign seed for the V3 CampaignRequest.",
    )
    campaign.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build request/stage configs without launching the canonical campaign.",
    )
    campaign.set_defaults(handler=_run_campaign)

    candidate = subparsers.add_parser(
        "deployment-candidate",
        help="Run the frozen temporal_threshold_36 + Lean-Lean deployment lane.",
    )
    candidate.add_argument(
        "--config",
        default=str(DEFAULT_DEPLOYMENT_CANDIDATE_CONFIG_PATH),
        help="Path to the canonical deployment-candidate YAML.",
    )
    candidate.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier created under runs/<config-stem>/<run-id>.",
    )
    candidate.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Base directory that will contain the deployment-candidate run root.",
    )
    candidate.add_argument(
        "--search-seed",
        type=int,
        default=None,
        help="Optional deterministic override for the Lean-Lean search seed.",
    )
    candidate.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config and emit plan/scaffold artifacts without launching the search.",
    )
    candidate.set_defaults(handler=_run_deployment_candidate)

    persistence = subparsers.add_parser(
        "persistence",
        help="Replay one of the two supported persistence profiles for a deployment run.",
    )
    persistence.add_argument(
        "--config",
        default=str(DEFAULT_PERSISTENCE_CONFIG_PATH),
        help="Path to a supported persistence profile YAML.",
    )
    persistence.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier created under <source-run-root>/persistence_tuning/<run-id>.",
    )
    persistence.add_argument(
        "--source-run-root",
        default=None,
        help="Optional override for the source deployment-candidate run root.",
    )
    persistence.add_argument(
        "--force-rematerialize-window-outputs",
        action="store_true",
        help="Rebuild per-record Lean-Lean outputs even if they already exist.",
    )
    persistence.set_defaults(handler=_run_persistence)

    verify = subparsers.add_parser(
        "verify",
        help="Emit canonical parity and golden-vector artifacts for a completed campaign run root.",
    )
    verify.add_argument("run_root", help="Completed run root containing freeze/export artifacts.")
    verify.add_argument(
        "--output-dir",
        help=(
            "Optional explicit verification output directory. "
            f"Defaults to <run_root>/{DEFAULT_VERIFICATION_STAGE_DIRNAME}."
        ),
    )
    verify.set_defaults(handler=_run_verify)


def _run_campaign(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = (Path(args.runs_root).resolve() / config_path.stem / args.run_id).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise ContractValidationError(f"Run root already exists and is not empty: {run_root}")

    prepared = prepare_full_binary_campaign(
        config_path,
        run_root,
        campaign_seed=args.campaign_seed,
    )
    request_json_path = write_campaign_request_json(
        prepared.campaign_request,
        prepared.run_root / "bt3.campaign_request.json",
    )
    campaign_result_path = prepared.run_root / CAMPAIGN_RESULT_ARTIFACT_NAME

    print(f"run_root={prepared.run_root}")
    print(f"campaign_request_json={request_json_path}")
    print(f"inventory_row_count={prepared.inventory_row_count}")
    print(f"lean_smoke_row_count={prepared.smoke_row_count}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0

    run_result = run_canonical_campaign(
        prepared.campaign_request,
        stage_configs=prepared.stage_configs,
        output_path=campaign_result_path,
    )
    if run_result.failed_stage_execution is not None:
        stage_result = run_result.failed_stage_execution.stage_result
        blocker = (
            f": {stage_result.exact_blocker}"
            if stage_result.exact_blocker is not None
            else ""
        )
        print(
            f"failed_stage={run_result.failed_stage_execution.stage_key.value}{blocker}",
            file=sys.stderr,
        )
        print(f"campaign_result_path={run_result.campaign_result_ref.path}")
        return 1

    print(f"campaign_result_path={run_result.campaign_result_ref.path}")
    return 0


def _run_deployment_candidate(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = (Path(args.runs_root).resolve() / config_path.stem / args.run_id).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise ContractValidationError(f"Run root already exists and is not empty: {run_root}")

    prepared = prepare_leanlean_deployment_candidate(
        config_path=config_path,
        run_root=run_root,
        search_seed=args.search_seed,
    )
    plan_path = write_leanlean_deployment_candidate_plan(prepared)
    persistence_refs = write_persistence_prep_artifacts(prepared)
    print(f"run_root={prepared.run_root}")
    print(f"plan_path={plan_path}")
    print(f"source_profile_path={prepared.source_profile_path}")
    print(f"frontend_regime={prepared.locked_frontend.regime_id}")
    print(f"semantic_bit_length={prepared.comparison_prepared.shared_bundle.semantic_bit_length}")
    print(f"comparison_bundle_bit_length={prepared.comparison_prepared.shared_bundle.packed_bit_length}")
    print(f"leanlean_search_seed={prepared.comparison_prepared.search_config.seed}")
    if persistence_refs is not None:
        print(f"persistence_scaffold_path={persistence_refs['scaffold_path']}")
        print(f"window_output_template_path={persistence_refs['window_output_template_path']}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0

    summary_json_path = run_prepared_leanlean_deployment_candidate(prepared)
    print(f"summary_json_path={summary_json_path}")
    print(f"summary_csv_path={prepared.run_root / 'summary.csv'}")
    print(f"summary_md_path={prepared.run_root / 'summary.md'}")
    return 0


def _run_persistence(args: argparse.Namespace) -> int:
    config = load_leanlean_persistence_tuning_config(Path(args.config).resolve())
    source_run_root = (
        Path(args.source_run_root).resolve()
        if args.source_run_root is not None
        else (
            config.source_deployment_run_root.resolve()
            if config.source_deployment_run_root is not None
            else None
        )
    )
    if source_run_root is None:
        raise ContractValidationError(
            "Persistence replay requires `--source-run-root` when the profile does not embed one."
        )
    prepared = prepare_leanlean_persistence_tuning(
        config_path=config.config_path,
        run_id=args.run_id,
        source_run_root=source_run_root,
    )
    summary_json_path = run_prepared_leanlean_persistence_tuning(
        prepared,
        force_rematerialize_window_outputs=args.force_rematerialize_window_outputs,
    )
    print(f"source_run_root={prepared.source_run_root}")
    print(f"run_root={prepared.run_root}")
    print(f"source_summary_path={prepared.source_summary_path}")
    print(f"materialized_window_outputs_path={prepared.source_materialized_window_outputs_path}")
    print(f"summary_json_path={summary_json_path}")
    print(f"summary_csv_path={prepared.run_root / prepared.config.summary_csv_name}")
    print(f"summary_md_path={prepared.run_root / prepared.config.summary_md_name}")
    print(f"selected_policy_path={prepared.run_root / prepared.config.selected_policy_json_name}")
    print(f"example_traces_path={prepared.run_root / prepared.config.example_traces_json_name}")
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    result = run_full_binary_verification(args.run_root, output_dir=args.output_dir)
    print(f"run_root={result.run_root}")
    print(f"verification_output_dir={result.output_dir}")
    print(f"stage_request_path={result.stage_request_ref.path}")
    print(f"verification_kit_manifest_path={result.verification_kit_manifest_ref.path}")
    print(f"golden_vector_manifest_path={result.golden_vector_manifest_ref.path}")
    print(f"parity_report_path={result.parity_report_ref.path}")
    print(f"golden_vector_count={result.vector_count}")
    print(f"parity_observation_count={result.observation_count}")
    return 0


def main(argv: list[str] | None = None) -> int:
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
