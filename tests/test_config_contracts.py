from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from bittrace.source.full_binary_campaign import load_consumer_config
from bittrace.source.leanlean_deployment_candidate import (
    load_leanlean_deployment_candidate_config,
    load_persistence_prep_config,
)
from bittrace.source.leanlean_persistence_tuning import load_leanlean_persistence_tuning_config
from bittrace.v3 import ContractValidationError


def test_canonical_source_profile_loads(project_root: Path) -> None:
    profile = load_consumer_config(project_root / "configs" / "canonical_source_profile.yaml")
    assert profile["profile_name"] == "canonical_source_profile"
    assert profile["locked_frontend"]["regime_id"] == "temporal_threshold_36"
    assert profile["backend"]["lean"]["allow_backend_fallback"] is False
    assert profile["backend"]["deep"]["allow_backend_fallback"] is False


def test_deployment_candidate_config_loads(project_root: Path) -> None:
    config = load_leanlean_deployment_candidate_config(
        project_root / "configs" / "canonical_deployment_candidate.yaml"
    )
    assert config["profile_name"] == "canonical_deployment_candidate"
    assert config["source_profile"] == "configs/canonical_source_profile.yaml"
    assert config["leanlean_deployment_candidate"]["search"]["seed"] == 7100


def test_persistence_profiles_load(project_root: Path) -> None:
    quiet = load_leanlean_persistence_tuning_config(
        project_root / "configs" / "persistence_quiet_scout.yaml"
    )
    aggressive = load_leanlean_persistence_tuning_config(
        project_root / "configs" / "persistence_aggressive.yaml"
    )
    prep = load_persistence_prep_config(project_root / "configs" / "persistence_quiet_scout.yaml")

    assert quiet.profile_name == "persistence_quiet_scout"
    assert quiet.source_deployment_run_root is None
    assert quiet.split_scope == ("train", "val", "test")
    assert quiet.default_policy.policy_id == "fc_i1_d1_y6_r12_nolatch"

    assert aggressive.profile_name == "persistence_aggressive"
    assert aggressive.default_policy.policy_id == "fc_i1_d1_y3_r7_nolatch"

    assert prep.profile_name == "persistence_quiet_scout"
    assert prep.window_output_artifact_name == "leanlean_window_outputs_template.json"


def test_persistence_prep_rejects_invalid_split_scope(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_persistence.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            profile_name: invalid_persistence
            deployment_candidate_config: configs/canonical_deployment_candidate.yaml
            window_outputs:
              artifact_name: invalid_template.json
              fields: [source_record_id]
              split_scope: [train, holdout]
            fault_counter_policy:
              mode: fault_counter
              policy_id: invalid
              increment_on_unhealthy: 1
              decrement_on_healthy: 1
              yellow_threshold: 1
              red_threshold: 2
              optional_latch: false
            planned_outputs:
              scaffold_json_name: invalid_scaffold.json
              summary_csv_name: invalid_summary.csv
              summary_md_name: invalid_summary.md
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractValidationError, match="train/val/test"):
        load_persistence_prep_config(config_path)
