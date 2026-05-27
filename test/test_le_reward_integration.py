"""Full-pipeline integration test for LigandEfficiencyReward.

Covers the complete RL reward chain:
    factory → CompositeReward → StabilityAwareReward →
    compute_batch_stats → build_epoch_facts → EpochFacts

Avoids real I/O (docking backend / RDKit molecule build) by using
a stub DockingReward that directly injects the expected batch tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

# 确保能导入项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "DiffSBDD"))

# ddpo.runtime.__init__ → build.py → integration → DiffSBDD.lightning_modules
# 这是一条不必要导入链，预先 mock 以阻断。
class _FakeLightningModule:
    LigandPocketDDPM = None
    EnVariationalDiffusion = None

sys.modules.setdefault("wandb", type(sys)("wandb"))
sys.modules.setdefault("diffusion", type(sys)("diffusion"))
sys.modules["DiffSBDD.lightning_modules"] = sys.modules.get(
    "DiffSBDD.lightning_modules"
) or type("DiffSBDD.lightning_modules", (), {"LigandPocketDDPM": None})

# 直接导入 train_helpers 绕过 runtime.__init__ 的级联链
import importlib
_train_helpers = importlib.import_module("ddpo.runtime.train_helpers")
compute_batch_stats = _train_helpers.compute_batch_stats
build_batch_facts = _train_helpers.build_batch_facts
build_epoch_facts = _train_helpers.build_epoch_facts
PreparedSamples = _train_helpers.PreparedSamples
BatchStats = _train_helpers.BatchStats

from ddpo.rewards.interfaces import BaseReward
from ddpo.rewards.ligand_efficiency_reward import LigandEfficiencyReward
from ddpo.rewards.composite import CompositeReward
from ddpo.rewards.stability import StabilityAwareReward
from ddpo.field_registry import collect_field_specs, FieldSpec

# ---------------------------------------------------------------------------
# Stub DockingReward — injects docking_raw, docking_reward, ligand_efficiency_raw
# without touching filesystem or docking backend.
# ---------------------------------------------------------------------------

class _StubDockingReward(BaseReward):
    """Writes docking fields directly, bypassing all I/O."""

    def __init__(
        self,
        docking_raw: list[float],
        docking_reward: list[float],
        ligand_efficiency_raw: list[float],
    ) -> None:
        self._docking_raw = docking_raw
        self._docking_reward = docking_reward
        self._ligand_efficiency_raw = ligand_efficiency_raw

    def calculate_rewards(self, data):
        batch = data.batch
        device = batch["x"].device
        batch["docking_raw"] = torch.tensor(self._docking_raw, device=device)
        batch["docking_reward"] = torch.tensor(self._docking_reward, device=device)
        batch["ligand_efficiency_raw"] = torch.tensor(
            self._ligand_efficiency_raw, device=device
        )
        return data

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(
                name="ligand_efficiency_raw",
                scope="molecule",
                source="batch",
                source_key="ligand_efficiency_raw",
                reduce="none",
                order=240,
            ),
            FieldSpec(
                name="avg_ligand_efficiency",
                scope="epoch",
                source="batch",
                source_key="ligand_efficiency_raw",
                reduce="valid_mean",
                order=240,
            ),
        ]


# ---------------------------------------------------------------------------
# Stub QEDReward — injects qed_raw / qed_reward directly.
# ---------------------------------------------------------------------------

class _StubQEDReward(BaseReward):
    """Writes QED fields directly."""

    def __init__(self, qed_raw: list[float]) -> None:
        self._qed_raw = qed_raw

    def calculate_rewards(self, data):
        device = data.batch["x"].device
        qed = torch.tensor(self._qed_raw, device=device)
        data.batch["qed_raw"] = qed
        data.batch["qed_reward"] = qed
        data.batch["qed_success"] = torch.ones(len(qed), device=device)
        return data


# ---------------------------------------------------------------------------
# Helper: build a realistic DataProto mock with non_tensor_batch.
# ---------------------------------------------------------------------------

def _make_data(
    batch_size: int = 4,
    num_nodes: int = 10,
    num_atom_types: int = 7,
) -> SimpleNamespace:
    ntb = {
        "rdkit_valid": [1, 1, 1, 1][:batch_size],
        "rdmols": [None] * batch_size,
        "failure_type": ["success"] * batch_size,
        "smiles": ["CCO", "CCN", "CCCl", "CCF"][:batch_size],
    }
    batch: dict = {
        "x": torch.randn(batch_size, num_nodes, 3),
        "categorical": torch.randn(batch_size, num_nodes, num_atom_types),
        "nodesxsample": torch.full((batch_size,), num_nodes, dtype=torch.long),
    }
    return SimpleNamespace(batch=batch, non_tensor_batch=ntb)


# ---------------------------------------------------------------------------
# Stub runtime for build_epoch_facts.
# ---------------------------------------------------------------------------

class _StubRuntime:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.max_epochs = 200
        self.checkpoint_interval = 5


# ===========================================================================
# Tests
# ===========================================================================

def test_composite_reward_chain():
    """CompositeReward(LigandEfficiencyReward + StubDocking + StubQED).

    Verifies:
        - reward_total_raw 包含 LE 加权贡献
        - ligand_efficiency_reward 正确 clamp
        - stability 字段注入（rdkit_valid → stability）
    """
    docking_stub = _StubDockingReward(
        docking_raw=[-8.0, -10.0, 0.0, -6.0],
        docking_reward=[0.8, 1.0, 0.0, 0.6],
        ligand_efficiency_raw=[1.0, 1.5, 0.0, 3.0],  # 3.0 会被 clamp 到 2.0
    )
    le_reward = LigandEfficiencyReward(
        failure_penalty=-1.0, min_le=0.0, max_le=2.0, scale=1.0
    )
    qed_stub = _StubQEDReward(qed_raw=[0.8, 0.6, 0.0, 0.5])

    composite = CompositeReward(
        rewards={
            "docking": docking_stub,
            "ligand_efficiency": le_reward,
            "qed": qed_stub,
        },
        weights={"docking": 0.5, "ligand_efficiency": 0.3, "qed": 0.2},
        normalize_weights=True,
    )
    rewarder = StabilityAwareReward(composite, {})

    data = _make_data(batch_size=4)
    result = rewarder.calculate_rewards(data)

    batch = result.batch

    # 1. Docking stub wrote fields
    assert torch.allclose(
        batch["docking_raw"], torch.tensor([-8.0, -10.0, 0.0, -6.0])
    ), f"docking_raw: {batch['docking_raw']}"

    # 2. LE reward clamped molecule 3 (3.0 → 2.0)
    expected_le_reward = torch.tensor([1.0, 1.5, 0.0, 2.0])
    assert torch.allclose(
        batch["ligand_efficiency_reward"], expected_le_reward
    ), f"ligand_efficiency_reward: {batch['ligand_efficiency_reward']}"

    # 3. reward_total_raw = Σ(normalized_weight * component_reward)
    weight_sum = 0.5 + 0.3 + 0.2  # = 1.0 before normalization → normalize to 1.0
    # After normalization:
    w_dock = 0.5 / weight_sum
    w_le = 0.3 / weight_sum
    w_qed = 0.2 / weight_sum
    expected_total = (
        w_dock * torch.tensor([0.8, 1.0, 0.0, 0.6])
        + w_le * expected_le_reward
        + w_qed * torch.tensor([0.8, 0.6, 0.0, 0.5])
    )
    assert torch.allclose(
        batch["reward_total_raw"], expected_total
    ), f"reward_total_raw: {batch['reward_total_raw']} vs {expected_total}"

    # 4. stability derived from rdkit_valid (all valid)
    assert torch.allclose(
        batch["stability"], torch.ones(4)
    ), f"stability: {batch['stability']}"

    print("[PASS] test_composite_reward_chain")


def test_field_specs_propagation():
    """StabilityAwareReward → CompositeReward → DockingStub 的 field_specs 链。

    验证 collect_field_specs 能递归收集到 avg_ligand_efficiency。
    """
    docking_stub = _StubDockingReward(
        docking_raw=[-8.0] * 4,
        docking_reward=[0.8] * 4,
        ligand_efficiency_raw=[1.0] * 4,
    )
    le_reward = LigandEfficiencyReward()
    composite = CompositeReward(
        rewards={"docking": docking_stub, "ligand_efficiency": le_reward},
        weights={"docking": 0.5, "ligand_efficiency": 0.5},
    )
    rewarder = StabilityAwareReward(composite, {})

    specs = collect_field_specs(rewarder)
    spec_names = {s.name for s in specs}

    assert "ligand_efficiency_raw" in spec_names
    assert "avg_ligand_efficiency" in spec_names
    print("[PASS] test_field_specs_propagation — spec names:", sorted(spec_names))


def test_batch_stats_and_epoch_facts():
    """compute_batch_stats → build_epoch_facts 验证 avg_ligand_efficiency 流入。

    模拟完整的 rank-0 后处理流程：从 reward 后的 DataProto 到 EpochFacts。
    """
    docking_stub = _StubDockingReward(
        docking_raw=[-8.0, -10.0, 0.0, -6.0],
        docking_reward=[0.8, 1.0, 0.0, 0.6],
        ligand_efficiency_raw=[1.0, 1.5, 0.0, 2.0],
    )
    le_reward = LigandEfficiencyReward()
    qed_stub = _StubQEDReward(qed_raw=[0.8, 0.6, 0.0, 0.5])

    composite = CompositeReward(
        rewards={"docking": docking_stub, "ligand_efficiency": le_reward, "qed": qed_stub},
        weights={"docking": 0.5, "ligand_efficiency": 0.3, "qed": 0.2},
        normalize_weights=True,
    )
    rewarder = StabilityAwareReward(composite, {})

    data = _make_data(batch_size=4)
    result = rewarder.calculate_rewards(data)

    # Collect field_specs from the rewarder tree
    field_specs = collect_field_specs(rewarder)

    prepared = PreparedSamples(
        samples=result,
        filtering_ratio=0.0,
        novelty_penalty_ratio=0.0,
    )

    config = {
        "reward": {
            "weights": {"docking": 0.5, "ligand_efficiency": 0.3, "qed": 0.2}
        }
    }

    local_stats = compute_batch_stats(prepared, config=config, field_specs=field_specs)

    # avg_ligand_efficiency 通过 registered_epoch_aggregates → extra_epoch_metrics
    reg = local_stats.registered_epoch_aggregates or {}
    assert "avg_ligand_efficiency" in reg, (
        f"avg_ligand_efficiency missing from registered aggregates. Keys: {list(reg.keys())}"
    )
    # valid_mean over 4 molecules: mean([1.0, 1.5, 0.0, 2.0]) = 1.125
    # But wait, valid_mean only counts valid molecules (rdkit_valid=1, all 4 are valid)
    # value=sum, count=4
    agg = reg["avg_ligand_efficiency"]
    assert agg.count == 4, f"Expected count=4, got {agg.count}"
    assert abs(agg.value - 4.5) < 0.01, f"Expected sum=4.5, got {agg.value}"

    # Simulate global reduce (single rank → just divide by count)
    reduced_le = agg.value / agg.count  # 4.5 / 4 = 1.125
    extra_epoch_metrics = {"avg_ligand_efficiency": reduced_le}
    local_stats.extra_epoch_metrics = extra_epoch_metrics

    # build epoch facts
    runtime = _StubRuntime(config)
    epoch_facts = build_epoch_facts(
        runtime=runtime,
        epoch=5,
        metrics={"Loss": 0.5},
        batch_stats=local_stats,
        prepared=prepared,
    )

    assert abs(epoch_facts.avg_ligand_efficiency - 1.125) < 0.01, (
        f"EpochFacts.avg_ligand_efficiency = {epoch_facts.avg_ligand_efficiency}, expected ≈1.125"
    )
    assert "avg_ligand_efficiency" in epoch_facts.extra_metrics, (
        "avg_ligand_efficiency should also be in extra_metrics"
    )
    print(
        "[PASS] test_batch_stats_and_epoch_facts — "
        f"avg_ligand_efficiency = {epoch_facts.avg_ligand_efficiency:.4f}"
    )


def test_docking_disabled_le_still_monitored():
    """验证：docking 启用但 LE 不启用时，avg_ligand_efficiency 仍然被监控。

    因为 DockingStub 的 field_specs 总是声明 avg_ligand_efficiency，
    即使 CompositeReward 不包含 LigandEfficiencyReward。
    """
    docking_stub = _StubDockingReward(
        docking_raw=[-8.0] * 4,
        docking_reward=[0.8] * 4,
        ligand_efficiency_raw=[1.0, 1.5, 0.0, 2.0],
    )
    # No LigandEfficiencyReward — only docking + qed
    composite = CompositeReward(
        rewards={"docking": docking_stub, "qed": _StubQEDReward([0.8] * 4)},
        weights={"docking": 0.7, "qed": 0.3},
    )
    rewarder = StabilityAwareReward(composite, {})

    data = _make_data(batch_size=4)
    result = rewarder.calculate_rewards(data)

    field_specs = collect_field_specs(rewarder)
    prepared = PreparedSamples(samples=result, filtering_ratio=0.0, novelty_penalty_ratio=0.0)

    local_stats = compute_batch_stats(
        prepared,
        config={"reward": {"weights": {"docking": 0.7, "qed": 0.3}}},
        field_specs=field_specs,
    )

    reg = local_stats.registered_epoch_aggregates or {}
    assert "avg_ligand_efficiency" in reg, (
        "avg_ligand_efficiency should be monitored even when LE reward is disabled"
    )
    print("[PASS] test_docking_disabled_le_still_monitored")


def test_ligand_efficiency_enabled_without_docking():
    """验证：启用 LE 但不启用 docking 时，LE reward 收到缺失警告并返回 penalty。

    这模拟了用户配置错误的情况。
    """
    le_reward = LigandEfficiencyReward(failure_penalty=-2.0)
    composite = CompositeReward(
        rewards={"ligand_efficiency": le_reward},
        weights={"ligand_efficiency": 1.0},
    )
    rewarder = StabilityAwareReward(composite, {})

    data = _make_data(batch_size=4)
    # No docking stub → no ligand_efficiency_raw in batch
    result = rewarder.calculate_rewards(data)

    expected = torch.full((4,), -2.0)
    assert torch.allclose(
        result.batch["ligand_efficiency_reward"], expected
    ), f"All should get failure_penalty: {result.batch['ligand_efficiency_reward']}"
    print("[PASS] test_ligand_efficiency_enabled_without_docking — graceful failure")


def test_factory_get_enabled_rewards():
    """验证 get_enabled_rewards 正确识别 LE 启用/禁用，以及 docking 自动启用。"""
    from ddpo.rewards.factory import get_enabled_rewards

    # LE weight=0 → not enabled
    cfg = {"reward": {"weights": {"docking": 0.5, "ligand_efficiency": 0}}}
    assert "ligand_efficiency" not in get_enabled_rewards(cfg)

    # LE weight=0.2 → enabled (docking already present)
    cfg2 = {"reward": {"weights": {"docking": 0.5, "ligand_efficiency": 0.2}}}
    assert "ligand_efficiency" in get_enabled_rewards(cfg2)

    # LE enabled, docking NOT in weights → docking auto-enabled as provider
    cfg3 = {"reward": {"weights": {"ligand_efficiency": 0.3, "qed": 0.5}}}
    enabled3 = get_enabled_rewards(cfg3)
    assert "ligand_efficiency" in enabled3
    assert "docking" in enabled3, "docking should be auto-enabled when LE is on"
    assert "qed" in enabled3

    # LE disabled → docking not auto-enabled
    cfg4 = {"reward": {"weights": {"ligand_efficiency": 0, "qed": 0.5}}}
    assert "docking" not in get_enabled_rewards(cfg4)

    print("[PASS] test_factory_get_enabled_rewards")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LigandEfficiencyReward 集成测试")
    print("=" * 60)

    test_composite_reward_chain()
    test_field_specs_propagation()
    test_batch_stats_and_epoch_facts()
    test_docking_disabled_le_still_monitored()
    test_ligand_efficiency_enabled_without_docking()
    test_factory_get_enabled_rewards()

    print("\n" + "=" * 60)
    print("全部 6 个集成测试通过 ✅")
    print("=" * 60)
