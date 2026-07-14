"""Tests for isolated DePPA-style DiffGui reward helpers."""

from __future__ import annotations

import torch

from ddpo.diffgui.deppa import (
    build_local_deppa_records,
    compute_deppa_advantages,
    compute_deppa_k3_kl,
    is_deppa_mode_enabled,
)
from ddpo.diffgui.deppa.advantage import (
    DEFAULT_DIVERSITY_BITS,
    DEFAULT_DIVERSITY_RADIUS,
    DEFAULT_WEIGHTS,
)


def _reconstruction_rows(count: int) -> list[dict[str, object]]:
    return [{"mol_index": index, "reconstruction_status": "ok"} for index in range(count)]


def test_deppa_switch_is_explicit() -> None:
    """DePPA mode should be opt-in so legacy configs stay unchanged."""
    assert not is_deppa_mode_enabled({})
    assert not is_deppa_mode_enabled({"diffgui_deppa_mode": {"enabled": False}})
    assert is_deppa_mode_enabled({"diffgui_deppa_mode": {"enabled": True}})


def test_deppa_default_weights_follow_pdf_reward_mix() -> None:
    """Default weights should match the DePPA PDF appendix reward formula."""
    assert DEFAULT_WEIGHTS == {
        "qed": 0.20,
        "sa": 0.20,
        "vina_score": 0.50,
        "distance": 0.10,
        "strain": 0.0,
    }


def test_deppa_diversity_fingerprint_follows_pdf_default() -> None:
    """Default diversity should use Morgan fingerprints as described in the PDF."""
    assert DEFAULT_DIVERSITY_RADIUS == 2
    assert DEFAULT_DIVERSITY_BITS == 2048


def test_deppa_vina_uses_negative_docking_raw_as_higher_better() -> None:
    """More negative docking_raw should rank higher after DePPA vina transform."""
    reward_rows = [
        {"source_index": 0, "rdkit_valid": 1, "smiles": "CCO", "docking_raw": -5.0},
        {"source_index": 1, "rdkit_valid": 1, "smiles": "CCN", "docking_raw": -7.0},
        {"source_index": 2, "rdkit_valid": 1, "smiles": "CCC", "docking_raw": -6.0},
    ]
    records = build_local_deppa_records(
        reconstruction_rows=_reconstruction_rows(3),
        reward_rows=reward_rows,
        rank=0,
        config={},
    )

    result = compute_deppa_advantages(
        local_records=records,
        gathered_records=[records],
        config={"enabled_metrics": ["vina_score"], "weights": {"vina_score": 1.0}},
    )

    assert result.rewards_by_graph[1] > result.rewards_by_graph[2] > result.rewards_by_graph[0]
    assert result.fields_by_graph[1]["deppa_vina_score_raw"] == 7.0
    assert result.summary["deppa_valid_count"] == 3.0


def test_deppa_rank_normal_matches_expected_percentiles() -> None:
    """A single enabled component should map ranks through normal percentiles."""
    reward_rows = [
        {"source_index": 0, "rdkit_valid": 1, "smiles": "CCO", "qed_raw": 0.1},
        {"source_index": 1, "rdkit_valid": 1, "smiles": "CCN", "qed_raw": 0.5},
        {"source_index": 2, "rdkit_valid": 1, "smiles": "CCC", "qed_raw": 0.9},
    ]
    records = build_local_deppa_records(
        reconstruction_rows=_reconstruction_rows(3),
        reward_rows=reward_rows,
        rank=0,
        config={},
    )

    result = compute_deppa_advantages(
        local_records=records,
        gathered_records=[records],
        config={"enabled_metrics": ["qed"], "weights": {"qed": 1.0}},
    )

    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    expected = [
        float(normal.icdf(torch.tensor(1.0 / 6.0)).item()),
        0.0,
        float(normal.icdf(torch.tensor(5.0 / 6.0)).item()),
    ]
    assert torch.allclose(torch.tensor(result.rewards_by_graph), torch.tensor(expected), atol=1.0e-6)
    assert torch.allclose(result.advantages, torch.tensor(expected), atol=1.0e-6)


def test_deppa_distance_uses_global_fingerprint_diversity() -> None:
    """Fingerprint distance should reward molecules unlike the rest of the batch."""
    reward_rows = [
        {"source_index": 0, "rdkit_valid": 1, "smiles": "CCO"},
        {"source_index": 1, "rdkit_valid": 1, "smiles": "CCO"},
        {"source_index": 2, "rdkit_valid": 1, "smiles": "c1ccccc1"},
    ]
    records = build_local_deppa_records(
        reconstruction_rows=_reconstruction_rows(3),
        reward_rows=reward_rows,
        rank=0,
        config={},
    )

    result = compute_deppa_advantages(
        local_records=records,
        gathered_records=[records],
        config={"enabled_metrics": ["distance"], "weights": {"distance": 1.0}},
    )

    assert result.fields_by_graph[2]["deppa_distance_raw"] > result.fields_by_graph[0]["deppa_distance_raw"]
    assert result.rewards_by_graph[2] > result.rewards_by_graph[0]


def test_deppa_invalid_reward_uses_valid_mean_minus_std_penalty() -> None:
    """Invalid molecules get a finite penalty derived from valid DePPA rewards."""
    reconstruction_rows = _reconstruction_rows(3)
    reconstruction_rows[2]["reconstruction_status"] = "failed"
    reward_rows = [
        {"source_index": 0, "rdkit_valid": 1, "smiles": "CCO", "qed_raw": 0.1},
        {"source_index": 1, "rdkit_valid": 1, "smiles": "CCC", "qed_raw": 0.9},
    ]
    records = build_local_deppa_records(
        reconstruction_rows=reconstruction_rows,
        reward_rows=reward_rows,
        rank=0,
        config={},
    )

    result = compute_deppa_advantages(
        local_records=records,
        gathered_records=[records],
        config={"enabled_metrics": ["qed"], "weights": {"qed": 1.0}, "invalid_penalty_std": 3.0},
    )

    valid = torch.tensor(result.rewards_by_graph[:2])
    expected_invalid = float(valid.mean().item() - 3.0 * valid.std(unbiased=False).item())
    assert abs(result.rewards_by_graph[2] - expected_invalid) < 1.0e-6
    assert result.fields_by_graph[2]["deppa_valid"] == 0
    assert result.summary["deppa_invalid_count"] == 1.0


def test_deppa_no_valid_records_use_configured_fallback() -> None:
    """All-failed batches should still produce finite trainable advantages."""
    reconstruction_rows = [{"mol_index": 0, "reconstruction_status": "failed"}]
    records = build_local_deppa_records(
        reconstruction_rows=reconstruction_rows,
        reward_rows=[],
        rank=0,
        config={},
    )

    result = compute_deppa_advantages(
        local_records=records,
        gathered_records=[records],
        config={"enabled_metrics": ["qed"], "weights": {"qed": 1.0}, "invalid_fallback_reward": -2.5},
    )

    assert result.rewards_by_graph == [-2.5]
    assert result.advantages.tolist() == [-2.5]


def test_deppa_pb_invalid_and_reconstruction_invalid_share_penalty() -> None:
    """A PB-filtered reward miss should equal the ordinary illegal penalty."""
    reconstruction_rows = _reconstruction_rows(4)
    reconstruction_rows[3]["reconstruction_status"] = "failed"
    reward_rows = [
        {"source_graph_index": 0, "source_index": 0, "rdkit_valid": 1, "smiles": "CCO", "qed_raw": 0.1},
        {"source_graph_index": 1, "source_index": 1, "rdkit_valid": 1, "smiles": "CCN", "qed_raw": 0.9},
    ]
    records = build_local_deppa_records(
        reconstruction_rows=reconstruction_rows,
        reward_rows=reward_rows,
        rank=0,
        config={},
    )

    result = compute_deppa_advantages(
        local_records=records,
        gathered_records=[records],
        config={"enabled_metrics": ["qed"], "weights": {"qed": 1.0}, "invalid_penalty_std": 3.0},
    )

    assert records[2]["reconstruction_success"] == 1
    assert records[2]["reward_found"] == 0
    assert records[3]["reconstruction_success"] == 0
    assert result.rewards_by_graph[2] == result.rewards_by_graph[3]
    assert result.fields_by_graph[2]["deppa_valid"] == 0
    assert result.fields_by_graph[3]["deppa_valid"] == 0


def test_deppa_k3_kl_formula_and_clip() -> None:
    """K3 KL is exp(ref-current)-1-(ref-current), with optional clipping."""
    reference = torch.tensor([0.0, 4.0, -100.0])
    current = torch.tensor([0.0, 1.0, 100.0])

    kl = compute_deppa_k3_kl(reference, current, clip=2.0)

    expected_logratio = torch.tensor([0.0, 2.0, -2.0])
    expected = torch.exp(expected_logratio) - 1.0 - expected_logratio
    assert torch.allclose(kl, expected)
