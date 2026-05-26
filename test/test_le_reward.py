"""Quick smoke test for LigandEfficiencyReward."""
import torch
from types import SimpleNamespace
from ddpo.rewards.ligand_efficiency_reward import LigandEfficiencyReward

r = LigandEfficiencyReward(failure_penalty=-1.0, min_le=0.0, max_le=2.0, scale=1.0)
print(repr(r))

# Test 1: no ligand_efficiency_raw → failure_penalty
batch = {"x": torch.randn(3, 10, 3), "nodesxsample": torch.tensor([10, 10, 10])}
data = SimpleNamespace(batch=batch)
result = r.calculate_rewards(data)
print("No LE raw (all penalty):", result.batch["ligand_efficiency_reward"])

# Test 2: with ligand_efficiency_raw → clamping
batch2 = {
    "x": torch.randn(3, 10, 3),
    "nodesxsample": torch.tensor([10, 10, 10]),
    "ligand_efficiency_raw": torch.tensor([0.5, 1.5, 3.0]),
}
data2 = SimpleNamespace(batch=batch2)
result2 = r.calculate_rewards(data2)
print("With LE raw [0.5, 1.5, 3.0] clamped to [0, 2]:",
      result2.batch["ligand_efficiency_reward"])

expected = torch.tensor([0.5, 1.5, 2.0])
assert torch.allclose(result2.batch["ligand_efficiency_reward"], expected), (
    f"Mismatch: {result2.batch['ligand_efficiency_reward']} vs {expected}"
)

# Test 3: factory integration (weight=0, should not instantiate)
from ddpo.rewards.factory import get_enabled_rewards
config = {"reward": {"weights": {"ligand_efficiency": 0}}}
enabled = get_enabled_rewards(config)
assert "ligand_efficiency" not in enabled, f"Should not be enabled: {enabled}"

# Test 4: factory integration (weight=0.1, should instantiate)
config2 = {"reward": {"weights": {"docking": 0.5, "ligand_efficiency": 0.1}}}
enabled2 = get_enabled_rewards(config2)
assert "ligand_efficiency" in enabled2, f"Should be enabled: {enabled2}"

print("ALL TESTS PASSED")
