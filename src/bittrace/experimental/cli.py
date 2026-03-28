"""CLI plumbing for experimental BitTrace workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from bittrace.v3 import ContractValidationError

from .backend_architecture_comparison import (
    DEFAULT_CONFIG_PATH as DEFAULT_BACKEND_COMPARISON_CONFIG_PATH,
    DEFAULT_RUNS_ROOT as DEFAULT_EXPERIMENTAL_RUNS_ROOT,
    prepare_backend_architecture_comparison,
    run_prepared_backend_architecture_comparison,
    write_backend_architecture_plan,
)
from .frontend_capacity_check import (
    DEFAULT_CONFIG_PATH as DEFAULT_FRONTEND_CAPACITY_CONFIG_PATH,
    DEFAULT_RUNS_ROOT as DEFAULT_FRONTEND_CAPACITY_RUNS_ROOT,
    prepare_frontend_capacity_check,
    run_prepared_frontend_capacity_check,
    write_frontend_capacity_plan,
)
from .leandeep_max_search import (
    DEFAULT_CONFIG_PATH as DEFAULT_LEANDEEP_MAX_SEARCH_CONFIG_PATH,
    DEFAULT_RUNS_ROOT as DEFAULT_LEANDEEP_MAX_SEARCH_RUNS_ROOT,
    prepare_leandeep_max_search,
    run_prepared_leandeep_max_search,
    write_leandeep_max_search_plan,
)
from .leanlean_ceiling_search import (
    DEFAULT_CONFIG_PATH as DEFAULT_LEANLEAN_CEILING_SEARCH_CONFIG_PATH,
    DEFAULT_RUNS_ROOT as DEFAULT_LEANLEAN_CEILING_SEARCH_RUNS_ROOT,
    prepare_leanlean_ceiling_search,
    run_prepared_leanlean_ceiling_search,
    write_leanlean_ceiling_search_plan,
)
from .leanlean_max_search import (
    DEFAULT_CONFIG_PATH as DEFAULT_LEANLEAN_MAX_SEARCH_CONFIG_PATH,
    DEFAULT_RUNS_ROOT as DEFAULT_LEANLEAN_MAX_SEARCH_RUNS_ROOT,
    prepare_leanlean_max_search,
    run_prepared_leanlean_max_search,
    write_leanlean_max_search_plan,
)
from .leanlean_seed_sweep import (
    DEFAULT_CONFIG_PATH as DEFAULT_LEANLEAN_SEED_SWEEP_CONFIG_PATH,
    DEFAULT_RUNS_ROOT as DEFAULT_LEANLEAN_SEED_SWEEP_RUNS_ROOT,
    prepare_leanlean_seed_sweep,
    run_prepared_leanlean_seed_sweep,
    write_leanlean_seed_sweep_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEANLEAN_MAX_SEARCH_DEEP_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experimental" / "leanlean_max_search_deep.yaml"
)


def register_experimental_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the `bittrace experimental ...` command family."""

    experimental = subparsers.add_parser(
        "experimental",
        help="Run experimental or in-house research workflows.",
        description=(
            "Experimental and in-house BitTrace workflows. These commands are retained in "
            "the canonical repo/package but are not part of the supported commercial lane."
        ),
        epilog=(
            "No stability guarantees apply under `bittrace experimental ...`. Configs, "
            "artifacts, and command semantics may change."
        ),
    )
    experimental_subparsers = experimental.add_subparsers(
        dest="experimental_command",
        required=True,
    )

    _register_backend_comparison(experimental_subparsers)
    _register_frontend_capacity(experimental_subparsers)
    _register_seed_sweep(experimental_subparsers)
    _register_leanlean_max_search(
        experimental_subparsers,
        command="leanlean-max-search",
        help_text="Run the retained Lean-Lean max-search workflow.",
        default_config_path=DEFAULT_LEANLEAN_MAX_SEARCH_CONFIG_PATH,
    )
    _register_leanlean_max_search(
        experimental_subparsers,
        command="leanlean-deep-layer-max-search",
        help_text="Run the retained deep-layer Lean-Lean max-search workflow.",
        default_config_path=DEFAULT_LEANLEAN_MAX_SEARCH_DEEP_CONFIG_PATH,
    )
    _register_leanlean_ceiling_search(experimental_subparsers)
    _register_leandeep_max_search(experimental_subparsers)


def _register_backend_comparison(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "backend-comparison",
        help="Compare front-only, Lean-Deep, and Lean-Lean under the locked frontend.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_BACKEND_COMPARISON_CONFIG_PATH),
        help="Path to the experimental backend comparison YAML config.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier used under runs/<config-stem>/<run-id>.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_EXPERIMENTAL_RUNS_ROOT),
        help="Base directory that will contain the run root.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config and materialize the shared inputs without running the variants.",
    )
    parser.set_defaults(handler=_run_backend_comparison)


def _register_frontend_capacity(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "frontend-capacity-check",
        help="Run the retained bounded frontend-capacity comparison.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_FRONTEND_CAPACITY_CONFIG_PATH),
        help="Path to the experimental frontend-capacity YAML config.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier used under runs/<config-stem>/<run-id>.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_FRONTEND_CAPACITY_RUNS_ROOT),
        help="Base directory that will contain the run root.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config and materialize shared source bundles without running the sweep.",
    )
    parser.set_defaults(handler=_run_frontend_capacity)


def _register_seed_sweep(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "seed-sweep",
        help="Run the retained deterministic Lean-Lean deployment-candidate seed sweep.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_LEANLEAN_SEED_SWEEP_CONFIG_PATH),
        help="Path to the experimental seed-sweep YAML config.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier used under runs/<config-stem>/<run-id>.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_LEANLEAN_SEED_SWEEP_RUNS_ROOT),
        help="Base directory that will contain the sweep run root.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config and emit the sweep plan without running any seed candidate.",
    )
    parser.set_defaults(handler=_run_seed_sweep)


def _register_leanlean_max_search(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command: str,
    help_text: str,
    default_config_path: Path,
) -> None:
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument(
        "--config",
        default=str(default_config_path),
        help="Path to the experimental Lean-Lean max-search YAML config.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier used under runs/<config-stem>/<run-id>.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_LEANLEAN_MAX_SEARCH_RUNS_ROOT),
        help="Base directory that will contain the run root.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config, materialize the shared bundle, and emit the plan only.",
    )
    parser.set_defaults(handler=_run_leanlean_max_search)


def _register_leanlean_ceiling_search(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "leanlean-ceiling-search",
        help="Run the retained Lean-Lean ceiling-search workflow.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_LEANLEAN_CEILING_SEARCH_CONFIG_PATH),
        help="Path to the experimental Lean-Lean ceiling-search YAML config.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier used under runs/<config-stem>/<run-id>.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_LEANLEAN_CEILING_SEARCH_RUNS_ROOT),
        help="Base directory that will contain the run root.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config, materialize the shared bundle, and emit the plan only.",
    )
    parser.set_defaults(handler=_run_leanlean_ceiling_search)


def _register_leandeep_max_search(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "leandeep-max-search",
        help="Run the retained Lean-Deep max-search workflow.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_LEANDEEP_MAX_SEARCH_CONFIG_PATH),
        help="Path to the experimental Lean-Deep max-search YAML config.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier used under runs/<config-stem>/<run-id>.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_LEANDEEP_MAX_SEARCH_RUNS_ROOT),
        help="Base directory that will contain the run root.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config, materialize the shared bundle, and emit the plan only.",
    )
    parser.set_defaults(handler=_run_leandeep_max_search)


def _run_backend_comparison(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = _experimental_run_root(args, config_path)
    prepared = prepare_backend_architecture_comparison(config_path, run_root)
    plan_path = write_backend_architecture_plan(prepared)
    print(f"run_root={prepared.run_root}")
    print(f"plan_path={plan_path}")
    print(f"source_profile_path={prepared.source_profile_path}")
    print(f"shared_row_count={prepared.shared_row_count}")
    print(f"frontend_regime={prepared.locked_frontend.regime_id}")
    print(f"semantic_bit_length={prepared.shared_bundle.semantic_bit_length}")
    print(f"comparison_bundle_bit_length={prepared.shared_bundle.packed_bit_length}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0
    summary_json_path = run_prepared_backend_architecture_comparison(prepared)
    print(f"summary_json_path={summary_json_path}")
    print(f"summary_csv_path={prepared.run_root / 'summary.csv'}")
    print(f"summary_md_path={prepared.run_root / 'summary.md'}")
    return 0


def _run_frontend_capacity(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = _experimental_run_root(args, config_path)
    prepared = prepare_frontend_capacity_check(config_path, run_root)
    plan_path = write_frontend_capacity_plan(prepared)
    print(f"run_root={prepared.run_root}")
    print(f"plan_path={plan_path}")
    print(f"source_profile_path={prepared.source_profile_path}")
    print(f"shared_row_count={prepared.shared_row_count}")
    print(f"temporal_compatibility_drop_count={prepared.temporal_compatibility_drop_count}")
    print(f"regime_count={len(prepared.regimes)}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0
    summary_path = run_prepared_frontend_capacity_check(prepared)
    print(f"summary_path={summary_path}")
    return 0


def _run_seed_sweep(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = _experimental_run_root(args, config_path)
    prepared = prepare_leanlean_seed_sweep(config_path=config_path, run_root=run_root)
    plan_path = write_leanlean_seed_sweep_plan(prepared)
    print(f"run_root={prepared.run_root}")
    print(f"plan_path={plan_path}")
    print(f"deployment_candidate_config_path={prepared.deployment_candidate_config_path}")
    print(f"seeds={','.join(str(seed) for seed in prepared.seeds)}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0
    summary_json_path = run_prepared_leanlean_seed_sweep(prepared)
    print(f"summary_json_path={summary_json_path}")
    print(f"summary_csv_path={prepared.run_root / 'summary.csv'}")
    print(f"summary_md_path={prepared.run_root / 'summary.md'}")
    return 0


def _run_leanlean_max_search(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = _experimental_run_root(args, config_path)
    prepared = prepare_leanlean_max_search(config_path=config_path, run_root=run_root)
    plan_path = write_leanlean_max_search_plan(prepared)
    print(f"run_root={prepared.run_root}")
    print(f"plan_path={plan_path}")
    print(f"source_profile_path={prepared.source_profile_path}")
    print(f"frontend_regime={prepared.locked_frontend.regime_id}")
    print(f"semantic_bit_length={prepared.comparison_prepared.shared_bundle.semantic_bit_length}")
    print(f"comparison_bundle_bit_length={prepared.comparison_prepared.shared_bundle.packed_bit_length}")
    print(f"trials_per_candidate={prepared.max_search_spec.trials_per_candidate}")
    print(f"search_branches={prepared.max_search_spec.initial_search_branches}")
    print(f"bounded_random_fraction={prepared.max_search_spec.bounded_random_fraction:.2f}")
    print(f"winner_replay_branches={prepared.max_search_spec.winner_replay_branches}")
    print(f"winner_mutation_branches={prepared.max_search_spec.winner_mutation_branches}")
    print(f"deployment_candidate_summary_path={prepared.deployment_candidate_summary_path}")
    if prepared.current_max_search_summary_path is not None:
        print(f"current_max_search_summary_path={prepared.current_max_search_summary_path}")
    if prepared.ceiling_search_summary_path is not None:
        print(f"ceiling_search_summary_path={prepared.ceiling_search_summary_path}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0
    summary_json_path = run_prepared_leanlean_max_search(prepared)
    print(f"summary_json_path={summary_json_path}")
    print(f"summary_csv_path={prepared.run_root / 'summary.csv'}")
    print(f"summary_md_path={prepared.run_root / 'summary.md'}")
    return 0


def _run_leanlean_ceiling_search(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = _experimental_run_root(args, config_path)
    prepared = prepare_leanlean_ceiling_search(config_path=config_path, run_root=run_root)
    plan_path = write_leanlean_ceiling_search_plan(prepared)
    print(f"run_root={prepared.run_root}")
    print(f"plan_path={plan_path}")
    print(f"source_profile_path={prepared.source_profile_path}")
    print(f"frontend_regime={prepared.locked_frontend.regime_id}")
    print(f"semantic_bit_length={prepared.comparison_prepared.shared_bundle.semantic_bit_length}")
    print(f"comparison_bundle_bit_length={prepared.comparison_prepared.shared_bundle.packed_bit_length}")
    print(f"trials_per_candidate={prepared.ceiling_spec.trials_per_candidate}")
    print(f"search_branches={prepared.ceiling_spec.initial_search_branches}")
    print(f"bounded_random_fraction={prepared.ceiling_spec.bounded_random_fraction:.2f}")
    if prepared.baseline_summary_path is not None:
        print(f"baseline_summary_path={prepared.baseline_summary_path}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0
    summary_json_path = run_prepared_leanlean_ceiling_search(prepared)
    print(f"summary_json_path={summary_json_path}")
    print(f"summary_csv_path={prepared.run_root / 'summary.csv'}")
    print(f"summary_md_path={prepared.run_root / 'summary.md'}")
    return 0


def _run_leandeep_max_search(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    run_root = _experimental_run_root(args, config_path)
    prepared = prepare_leandeep_max_search(config_path=config_path, run_root=run_root)
    plan_path = write_leandeep_max_search_plan(prepared)
    print(f"run_root={prepared.run_root}")
    print(f"plan_path={plan_path}")
    print(f"source_profile_path={prepared.source_profile_path}")
    print(f"frontend_regime={prepared.locked_frontend.regime_id}")
    print(f"semantic_bit_length={prepared.comparison_prepared.shared_bundle.semantic_bit_length}")
    print(f"comparison_bundle_bit_length={prepared.comparison_prepared.shared_bundle.packed_bit_length}")
    print(f"trials_per_candidate={prepared.max_search_spec.trials_per_candidate}")
    print(f"search_branches={prepared.max_search_spec.initial_search_branches}")
    print(f"bounded_random_fraction={prepared.max_search_spec.bounded_random_fraction:.2f}")
    print(f"winner_replay_branches={prepared.max_search_spec.winner_replay_branches}")
    print(f"winner_mutation_branches={prepared.max_search_spec.winner_mutation_branches}")
    print(f"max_layers={prepared.max_search_spec.local_refine_evolution.max_layers}")
    print(f"deployment_candidate_summary_path={prepared.deployment_candidate_summary_path}")
    if prepared.current_max_search_summary_path is not None:
        print(f"current_max_search_summary_path={prepared.current_max_search_summary_path}")
    if prepared.deep_layer_max_search_summary_path is not None:
        print(f"deep_layer_max_search_summary_path={prepared.deep_layer_max_search_summary_path}")
    if args.prepare_only:
        print("prepare_only=true")
        return 0
    summary_json_path = run_prepared_leandeep_max_search(prepared)
    print(f"summary_json_path={summary_json_path}")
    print(f"summary_csv_path={prepared.run_root / 'summary.csv'}")
    print(f"summary_md_path={prepared.run_root / 'summary.md'}")
    return 0


def _experimental_run_root(args: argparse.Namespace, config_path: Path) -> Path:
    run_root = (Path(args.runs_root).resolve() / config_path.stem / args.run_id).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise ContractValidationError(f"Run root already exists and is not empty: {run_root}")
    return run_root


__all__ = ["register_experimental_commands"]
