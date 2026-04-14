#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete DiffSBDDModel Verification Script
验证所有已实现的功能并记录详细信息

包括:
1. 所有方法的功能测试
2. 边界情况测试
3. 数值稳定性测试
4. SE(3)等变性测试
5. 格式转换测试
"""

import argparse
import sys
import os
import traceback
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

# ============================================================================
# Verification Result Tracking with Markdown Logging
# ============================================================================

results = []
md_log = []  # Store markdown content

def add_md_header(title: str, level: int = 1):
    """Add markdown header"""
    md_log.append(f"\n{'#' * level} {title}\n")

def add_md_text(text: str):
    """Add markdown text"""
    md_log.append(f"{text}\n")

def add_md_code_block(code: str, language: str = ""):
    """Add markdown code block"""
    md_log.append(f"```{language}\n{code}\n```\n")

def add_md_table_row(cols: list):
    """Add markdown table row"""
    md_log.append("| " + " | ".join(str(c) for c in cols) + " |\n")

def record(name: str, success: bool, msg: str = ""):
    results.append((name, success, msg))
    status = "[PASS]" if success else "[FAIL]"
    print(f"  {status} {name}")
    if msg:
        if not success:
            print(f"     -> {msg}")
        else:
            print(f"     {msg}")

def run_check(name: str, fn):
    try:
        success = fn()
        record(name, success)
    except Exception as e:
        record(name, False, str(e))
        traceback.print_exc()


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    add_md_header(title, level=2)

def log_test_data(test_name: str, data: dict):
    """Log test data to markdown"""
    add_md_header(test_name, level=4)
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            add_md_text(f"- **{key}:** `{value}`")
        elif isinstance(value, dict):
            add_md_text(f"- **{key}:**")
            for k, v in value.items():
                add_md_text(f"  - {k}: `{v}`")
        else:
            add_md_text(f"- **{key}:** `{value}`")
    add_md_text("")  # Empty line

def save_md_report(output_path: str):
    """Save markdown report to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(md_log)
    print(f"\n  📄 Report saved to: {output_path}")


# ============================================================================
# Part 1: Method Completeness Tests
# ============================================================================

def test_method_completeness():
    """Test all DiffSBDDModel methods exist and are callable"""
    print_section("Part 1: Method Completeness")
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    required_methods = [
        '__init__', '_load_model', '_load_pocket_residues',
        'load', 'to', 'eval', 'train',
        'get_mask', 'get_pocket_info',
        '_rlpf_to_diffsbdd_format', '_diffsbdd_to_rlpf_format',
        '_replicate_pocket',
        'compute_log_p_zs_given_zt',
        'sample_p_xh_given_z0',
        'sample_p_zs_given_zt',
        'sample',
        '_assert_mean_zero_with_mask',
        '_remove_mean_with_mask',
    ]
    
    def check_methods():
        missing = [m for m in required_methods if not hasattr(DiffSBDDModel, m)]
        if missing:
            return False
        return True
    
    run_check("All required methods exist", check_methods)


# ============================================================================
# Part 2: Model Loading and Initialization
# ============================================================================

def test_model_initialization():
    """Test model loading and initialization"""
    print_section("Part 2: Model Initialization")
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"     Skipping initialization tests")
        return False
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    # Test 1: Basic initialization
    def test_basic_init():
        model = DiffSBDDModel(
            checkpoint_path=checkpoint_path,
            pocket_pdb_path="data/8GJJ-C-Cip.pdb",
            ref_ligand_path="C:1101",
            device="cpu",
            dataset_name="crossdock"
        )
        
        assert hasattr(model, 'model'), "Missing model attribute"
        assert hasattr(model, 'ddpm'), "Missing ddpm attribute"
        assert hasattr(model, 'residues'), "Missing residues attribute"
        assert model.T > 0, f"Invalid T: {model.T}"
        assert model.n_dims == 3, f"Invalid n_dims: {model.n_dims}"
        assert model.atom_nf > 0, f"Invalid atom_nf: {model.atom_nf}"
        
        print(f"     Model params: T={model.T}, n_dims={model.n_dims}, atom_nf={model.atom_nf}")
        print(f"     Pocket residues: {len(model.residues)}")
        
        # Log to markdown
        log_test_data("Basic Initialization", {
            "Timesteps (T)": model.T,
            "Spatial dimensions (n_dims)": model.n_dims,
            "Atom features (atom_nf)": model.atom_nf,
            "Amino acid features (aa_nf)": model.aa_nf,
            "Pocket residues": len(model.residues),
            "Device": model.device,
            "Dataset": model.dataset_name
        })
        
        return True
    
    run_check("Basic initialization", test_basic_init)

    
    # Test 2: get_pocket_info
    def test_pocket_info():
        model = DiffSBDDModel(
            checkpoint_path=checkpoint_path,
            pocket_pdb_path="data/8GJJ-C-Cip.pdb",
            ref_ligand_path="C:1101",
            device="cpu",
            dataset_name="crossdock"
        )
        
        info = model.get_pocket_info()
        
        assert 'num_residues' in info
        assert 'pocket_size' in info
        assert 'pocket_x_shape' in info
        assert 'pocket_one_hot_shape' in info
        
        print(f"     Pocket info: {info}")
        
        # Log to markdown
        log_test_data("Pocket Information", info)
        
        return True
    
    run_check("get_pocket_info method", test_pocket_info)
    
    # Test 3: Mode switching
    def test_mode_switching():
        model = DiffSBDDModel(
            checkpoint_path=checkpoint_path,
            pocket_pdb_path="data/8GJJ-C-Cip.pdb",
            ref_ligand_path="C:1101",
            device="cpu",
            dataset_name="crossdock"
        )
        
        # Test eval mode
        model.eval()
        assert not model.model.training, "Model should be in eval mode"
        
        # Test train mode
        model.train()
        assert model.model.training, "Model should be in train mode"
        
        # Switch back to eval
        model.eval()
        assert not model.model.training, "Model should be in eval mode"
        
        print(f"     Mode switching works correctly")
        
        return True
    
    run_check("Mode switching (eval/train)", test_mode_switching)
    
    return True


# ============================================================================
# Part 3: Mask Operations
# ============================================================================

def test_mask_operations():
    """Test mask generation and conversion"""
    print_section("Part 3: Mask Operations")
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping mask tests (no checkpoint)")
        return False
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )

    
    # Test 1: Single sample
    def test_single_sample_mask():
        nodesxsample = torch.tensor([5])
        batch_size = 1
        max_n_nodes = 5
        
        node_mask, edge_mask = model.get_mask(nodesxsample, batch_size, max_n_nodes)
        
        assert node_mask.shape == (1, 5, 1)
        assert edge_mask.shape == (25, 1)
        assert node_mask.sum() == 5
        
        print(f"     Single sample: node_mask sum={node_mask.sum().item()}")
        return True
    
    run_check("get_mask: single sample", test_single_sample_mask)
    
    # Test 2: Multiple samples with varying nodes
    def test_varying_nodes_mask():
        nodesxsample = torch.tensor([3, 5, 4, 6])
        batch_size = 4
        max_n_nodes = 6
        
        node_mask, edge_mask = model.get_mask(nodesxsample, batch_size, max_n_nodes)
        
        assert node_mask.shape == (4, 6, 1)
        assert edge_mask.shape == (4 * 6 * 6, 1)
        
        # Check each sample
        for i, n in enumerate(nodesxsample):
            sample_mask = node_mask[i, :, 0]
            assert sample_mask[:n].sum() == n, f"Sample {i} mask incorrect"
            assert sample_mask[n:].sum() == 0, f"Sample {i} has non-zero padding"
        
        print(f"     Varying nodes: {nodesxsample.tolist()}")
        print(f"     Mask sums: {[node_mask[i].sum().item() for i in range(4)]}")
        return True
    
    run_check("get_mask: varying nodes", test_varying_nodes_mask)
    
    # Test 3: Large batch
    def test_large_batch_mask():
        batch_size = 32
        max_n_nodes = 20
        nodesxsample = torch.randint(10, 20, (batch_size,))
        
        node_mask, edge_mask = model.get_mask(nodesxsample, batch_size, max_n_nodes)
        
        assert node_mask.shape == (batch_size, max_n_nodes, 1)
        assert edge_mask.shape == (batch_size * max_n_nodes * max_n_nodes, 1)
        
        print(f"     Large batch: {batch_size} samples")
        print(f"     Node counts range: [{nodesxsample.min().item()}, {nodesxsample.max().item()}]")
        return True
    
    run_check("get_mask: large batch", test_large_batch_mask)
    
    return True



# ============================================================================
# Part 4: Format Conversion Tests
# ============================================================================

def test_format_conversion():
    """Test RLPF <-> DiffSBDD format conversion"""
    print_section("Part 4: Format Conversion")
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping format conversion tests (no checkpoint)")
        return False
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )
    
    # Test 1: Round-trip conversion
    def test_round_trip():
        B, N, D = 4, 10, 3
        x_padded = torch.randn(B, N, D)
        node_mask = torch.ones(B, N, 1)
        node_mask[0, 7:] = 0  # 7 nodes
        node_mask[1, 8:] = 0  # 8 nodes
        node_mask[2, 9:] = 0  # 9 nodes
        node_mask[3, 10:] = 0  # 10 nodes
        
        # Convert to DiffSBDD format
        x_flat, batch_mask = model._rlpf_to_diffsbdd_format(x_padded, node_mask)
        
        # Convert back to RLPF format
        x_recovered = model._diffsbdd_to_rlpf_format(x_flat, batch_mask, N)
        
        # Check valid positions match
        valid_mask = node_mask.squeeze(-1).bool()
        diff = (x_padded[valid_mask] - x_recovered[valid_mask]).abs().max().item()
        
        print(f"     Round-trip max diff: {diff:.6e}")
        assert diff < 1e-6, f"Round-trip conversion failed: diff={diff}"
        
        return True
    
    run_check("Format conversion: round-trip", test_round_trip)
    
    # Test 2: Single sample conversion
    def test_single_sample_conversion():
        B, N, D = 1, 5, 3
        x_padded = torch.randn(B, N, D)
        node_mask = torch.ones(B, N, 1)
        
        x_flat, batch_mask = model._rlpf_to_diffsbdd_format(x_padded, node_mask)
        
        assert x_flat.shape[0] == 5, f"Flat shape incorrect: {x_flat.shape}"
        assert batch_mask.shape[0] == 5, f"Batch mask shape incorrect: {batch_mask.shape}"
        assert torch.all(batch_mask == 0), "Batch mask should be all zeros"
        
        print(f"     Single sample: x_flat shape={x_flat.shape}, batch_mask={batch_mask.tolist()}")
        return True
    
    run_check("Format conversion: single sample", test_single_sample_conversion)

    
    # Test 3: Padding handling
    def test_padding_handling():
        B, N, D = 3, 8, 3
        x_padded = torch.randn(B, N, D)
        node_mask = torch.ones(B, N, 1)
        node_mask[0, 3:] = 0  # Only 3 valid nodes
        node_mask[1, 5:] = 0  # Only 5 valid nodes
        node_mask[2, 8:] = 0  # All 8 valid
        
        x_flat, batch_mask = model._rlpf_to_diffsbdd_format(x_padded, node_mask)
        
        # Total valid nodes should be 3 + 5 + 8 = 16
        assert x_flat.shape[0] == 16, f"Expected 16 nodes, got {x_flat.shape[0]}"
        
        # Check batch_mask values
        expected_batch_mask = torch.tensor([0]*3 + [1]*5 + [2]*8)
        assert torch.equal(batch_mask, expected_batch_mask), "Batch mask incorrect"
        
        print(f"     Padding test: 3+5+8 nodes -> {x_flat.shape[0]} total")
        print(f"     Batch mask: {batch_mask.tolist()}")
        return True
    
    run_check("Format conversion: padding handling", test_padding_handling)
    
    return True


# ============================================================================
# Part 5: Log Probability Computation
# ============================================================================

def test_log_prob_computation():
    """Test compute_log_p_zs_given_zt with various inputs"""
    print_section("Part 5: Log Probability Computation")
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping log_prob tests (no checkpoint)")
        return False
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )
    model.eval()
    
    # Test 1: Basic computation
    def test_basic_log_prob():
        B, N, D = 2, 5, 13
        zs = torch.randn(B, N, D)
        mu = torch.randn(B, N, D)
        sigma = torch.tensor(0.5)
        node_mask = torch.ones(B, N, 1)
        
        log_prob = model.compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)
        
        assert log_prob.shape == (B,), f"Shape incorrect: {log_prob.shape}"
        assert torch.all(torch.isfinite(log_prob)), "Contains NaN or Inf"
        assert torch.all(log_prob < 0), "Log prob should be negative"
        
        print(f"     Basic: log_prob={log_prob.tolist()}")
        return True
    
    run_check("log_prob: basic computation", test_basic_log_prob)

    
    # Test 2: Different sigma formats
    def test_sigma_formats():
        B, N, D = 2, 5, 13
        zs = torch.randn(B, N, D)
        mu = torch.randn(B, N, D)
        node_mask = torch.ones(B, N, 1)
        
        # Scalar sigma
        sigma_scalar = torch.tensor(0.5)
        lp1 = model.compute_log_p_zs_given_zt(zs, mu, sigma_scalar, node_mask)
        
        # Batch sigma
        sigma_batch = torch.ones(B) * 0.5
        lp2 = model.compute_log_p_zs_given_zt(zs, mu, sigma_batch, node_mask)
        
        # Full sigma
        sigma_full = torch.ones(B, N, D) * 0.5
        lp3 = model.compute_log_p_zs_given_zt(zs, mu, sigma_full, node_mask)
        
        # All should be equal
        assert torch.allclose(lp1, lp2, atol=1e-5), "Scalar vs batch sigma mismatch"
        assert torch.allclose(lp1, lp3, atol=1e-5), "Scalar vs full sigma mismatch"
        
        print(f"     Sigma formats: scalar={lp1[0]:.4f}, batch={lp2[0]:.4f}, full={lp3[0]:.4f}")
        return True
    
    run_check("log_prob: different sigma formats", test_sigma_formats)
    
    # Test 3: Numerical stability with small sigma
    def test_small_sigma_stability():
        B, N, D = 2, 5, 13
        zs = torch.randn(B, N, D)
        mu = torch.randn(B, N, D)
        node_mask = torch.ones(B, N, 1)
        
        # Test with very small sigma
        sigma_small = torch.tensor(1e-6)
        log_prob = model.compute_log_p_zs_given_zt(zs, mu, sigma_small, node_mask)
        
        assert torch.all(torch.isfinite(log_prob)), "Numerical instability with small sigma"
        
        print(f"     Small sigma (1e-6): log_prob={log_prob.tolist()}")
        return True
    
    run_check("log_prob: numerical stability (small sigma)", test_small_sigma_stability)
    
    # Test 4: With padding
    def test_log_prob_with_padding():
        B, N, D = 3, 8, 13
        zs = torch.randn(B, N, D)
        mu = torch.randn(B, N, D)
        sigma = torch.tensor(0.5)
        node_mask = torch.ones(B, N, 1)
        node_mask[0, 5:] = 0  # 5 nodes
        node_mask[1, 6:] = 0  # 6 nodes
        node_mask[2, 8:] = 0  # 8 nodes
        
        log_prob = model.compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)
        
        # Log prob should be different for different number of nodes
        assert log_prob[0] != log_prob[2], "Log prob should differ with different node counts"
        
        print(f"     With padding (5,6,8 nodes): log_prob={log_prob.tolist()}")
        return True
    
    run_check("log_prob: with padding", test_log_prob_with_padding)
    
    return True



# ============================================================================
# Part 6: Zero-Center Operations
# ============================================================================

def test_zero_center_operations():
    """Test _assert_mean_zero_with_mask and _remove_mean_with_mask"""
    print_section("Part 6: Zero-Center Operations")
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping zero-center tests (no checkpoint)")
        return False
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )
    
    # Test 1: Remove mean
    def test_remove_mean():
        B, N, D = 2, 10, 3
        x = torch.randn(B, N, D) + 5.0  # Add offset
        node_mask = torch.ones(B, N, 1)
        node_mask[0, 7:] = 0
        node_mask[1, 8:] = 0
        
        # Calculate initial CoG
        x_masked = x * node_mask
        x_sum = x_masked.sum(dim=1)
        n_valid = node_mask.sum(dim=1)
        cog_before = (x_sum / n_valid).abs().max().item()
        
        # Remove mean
        x_centered = model._remove_mean_with_mask(x, node_mask)
        
        # Calculate new CoG
        x_masked_after = x_centered * node_mask
        x_sum_after = x_masked_after.sum(dim=1)
        cog_after = (x_sum_after / n_valid).abs().max().item()
        
        print(f"     CoG before: {cog_before:.6f}")
        print(f"     CoG after: {cog_after:.6e}")
        
        assert cog_after < 1e-6, f"CoG not zero: {cog_after}"
        
        return True
    
    run_check("Zero-center: remove mean", test_remove_mean)
    
    # Test 2: Assert mean zero (should pass)
    def test_assert_zero_pass():
        B, N, D = 2, 10, 3
        x = torch.randn(B, N, D)
        node_mask = torch.ones(B, N, 1)
        
        # Center it first
        x_centered = model._remove_mean_with_mask(x, node_mask)
        
        # This should not raise
        try:
            model._assert_mean_zero_with_mask(x_centered, node_mask)
            print(f"     Assert passed for centered data")
            return True
        except AssertionError:
            print(f"     Assert failed unexpectedly")
            return False
    
    run_check("Zero-center: assert zero (pass)", test_assert_zero_pass)
    
    return True



# ============================================================================
# Part 7: Sampling Tests with Detailed Logging
# ============================================================================

def test_sampling_operations():
    """Test all sampling methods with detailed logging"""
    print_section("Part 7: Sampling Operations (Detailed)")
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping sampling tests (no checkpoint)")
        return False
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )
    model.eval()
    
    # Test 1: sample_p_zs_given_zt with fix_noise
    def test_deterministic_sampling():
        B, N, D = 2, 10, 13
        nodesxsample = torch.tensor([8, 10])
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        zt = torch.randn(B, N, D)
        s = torch.tensor([[0.5], [0.5]])
        t = torch.tensor([[0.6], [0.6]])
        
        with torch.no_grad():
            # Sample twice with fix_noise=True
            zs1, lp1, mu1, sigma1 = model.sample_p_zs_given_zt(s, t, zt, node_mask, edge_mask, fix_noise=True)
            zs2, lp2, mu2, sigma2 = model.sample_p_zs_given_zt(s, t, zt, node_mask, edge_mask, fix_noise=True)
        
        # Should be identical
        assert torch.allclose(zs1, zs2, atol=1e-6), "Deterministic sampling not reproducible"
        assert torch.allclose(lp1, lp2, atol=1e-6), "Log prob not reproducible"
        
        print(f"     Deterministic sampling: zs diff={((zs1-zs2).abs().max().item()):.6e}")
        return True
    
    run_check("Sampling: deterministic (fix_noise=True)", test_deterministic_sampling)
    
    # Test 2: sample_p_xh_given_z0
    def test_final_step_sampling():
        B, N, D = 2, 10, 13
        nodesxsample = torch.tensor([8, 10])
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        z0 = torch.randn(B, N, D)
        
        with torch.no_grad():
            x, h, mu, sigma, log_prob, zeros, xh = model.sample_p_xh_given_z0(
                z0, node_mask, edge_mask, fix_noise=False
            )
        
        # Verify outputs
        assert x.shape == (B, N, 3), f"x shape: {x.shape}"
        assert 'categorical' in h, "Missing categorical"
        assert 'integer' in h, "Missing integer"
        assert h['categorical'].shape == (B, N, model.atom_nf)
        
        # Check one-hot
        h_sum = h['categorical'].sum(dim=-1)
        valid_sum = h_sum[node_mask.squeeze(-1).bool()]
        assert torch.allclose(valid_sum, torch.ones_like(valid_sum), atol=1e-5)
        
        print(f"     Final step: x range=[{x.min():.3f}, {x.max():.3f}]")
        print(f"     h categorical sum check: {valid_sum[:5].tolist()}")
        
        return True
    
    run_check("Sampling: final step (sample_p_xh_given_z0)", test_final_step_sampling)

    
    # Test 3: Full sampling with detailed trajectory logging
    def test_full_sampling_detailed():
        print("\n     === Full Sampling Test (20 steps) ===")
        
        B = 2
        N = 12
        nodesxsample = torch.tensor([10, 12])
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        print(f"     Config: batch={B}, max_nodes={N}, nodes={nodesxsample.tolist()}")
        
        # Log sampling configuration
        add_md_header("Full Sampling Test (20 steps)", level=4)
        log_test_data("Sampling Configuration", {
            "Batch size": B,
            "Max nodes": N,
            "Nodes per sample": nodesxsample.tolist(),
            "Timesteps": 20
        })
        
        with torch.no_grad():
            x, h, latents, logps, timesteps_list, mus, sigmas = model.sample(
                n_samples=B,
                n_nodes=nodesxsample,
                node_mask=node_mask,
                edge_mask=edge_mask,
                fix_noise=False,
                timestep=20  # Use 20 steps for faster testing
            )
        
        print(f"\n     Trajectory info:")
        print(f"       latents: {len(latents)} entries")
        print(f"       logps: {len(logps)} entries")
        print(f"       timesteps: {len(timesteps_list)} entries")
        
        # Log trajectory statistics
        log_test_data("Trajectory Statistics", {
            "Latents length": len(latents),
            "Log probs length": len(logps),
            "Timesteps length": len(timesteps_list)
        })
        
        # Log every 5 steps
        print(f"\n     Timestep details (every 5 steps):")
        add_md_header("Timestep Details", level=5)
        add_md_table_row(["Step", "Time (t)", "Latent Min", "Latent Max", "Latent Mean", "Latent Std", "Log Prob Mean", "Log Prob Std"])
        add_md_table_row(["---"] * 8)
        
        for i in range(0, len(logps), 5):
            lp = logps[i]
            lat = latents[i+1]
            ts = timesteps_list[i]
            
            t_val = ts.item() if torch.is_tensor(ts) else ts
            lat_min = lat.min().item()
            lat_max = lat.max().item()
            lat_mean = lat.mean().item()
            lat_std = lat.std().item()
            lp_mean = lp.mean().item()
            lp_std = lp.std().item()
            
            print(f"       Step {i:2d} (t={t_val:.1f}): "
                  f"latent range=[{lat_min:.3f}, {lat_max:.3f}], "
                  f"log_prob={lp_mean:.4f}±{lp_std:.4f}")
            
            add_md_table_row([
                i, f"{t_val:.1f}", 
                f"{lat_min:.3f}", f"{lat_max:.3f}",
                f"{lat_mean:.3f}", f"{lat_std:.3f}",
                f"{lp_mean:.4f}", f"{lp_std:.4f}"
            ])
        
        # Final output analysis
        print(f"\n     Final output:")
        print(f"       x shape: {x.shape}")
        print(f"       x range: [{x.min():.3f}, {x.max():.3f}]")
        
        # Check zero-center
        x_masked = x * node_mask
        x_sum = x_masked.sum(dim=1)
        n_valid = node_mask.sum(dim=1)
        cog = x_sum / (n_valid + 1e-8)
        max_cog = cog.abs().max().item()
        print(f"       Max CoG: {max_cog:.6e}")
        
        # Atom types
        h_argmax = torch.argmax(h['categorical'], dim=-1)
        print(f"\n       Atom type distribution:")
        
        atom_dist = {}
        for sample_idx in range(B):
            n_nodes = nodesxsample[sample_idx].item()
            atom_types = h_argmax[sample_idx, :n_nodes]
            unique, counts = torch.unique(atom_types, return_counts=True)
            print(f"         Sample {sample_idx}: ", end="")
            sample_dist = {}
            for at, ct in zip(unique.tolist(), counts.tolist()):
                print(f"type{at}:{ct} ", end="")
                sample_dist[f"type{at}"] = ct
            print()
            atom_dist[f"Sample {sample_idx}"] = sample_dist
        
        # Log final output
        x_valid = x[node_mask.squeeze(-1).bool()]
        log_test_data("Final Output Statistics", {
            "Coordinate shape": str(x.shape),
            "Coordinate range": f"[{x_valid.min():.3f}, {x_valid.max():.3f}]",
            "Coordinate mean": f"{x_valid.mean():.6f}",
            "Coordinate std": f"{x_valid.std():.3f}",
            "Max CoG offset": f"{max_cog:.6e}",
            "Atom type distribution": atom_dist,
            "Total log prob (sample 0)": f"{sum(lp[0].item() for lp in logps):.4f}",
            "Total log prob (sample 1)": f"{sum(lp[1].item() for lp in logps):.4f}"
        })
        
        # Verify
        assert len(latents) == 22, f"Expected 22 latents, got {len(latents)}"
        assert len(logps) == 21, f"Expected 21 logps, got {len(logps)}"
        assert max_cog < 0.1, f"CoG too large: {max_cog}"
        
        return True
    
    run_check("Sampling: full trajectory (detailed)", test_full_sampling_detailed)
    
    return True



# ============================================================================
# Part 8: SE(3) Equivariance Tests
# ============================================================================

def test_se3_equivariance():
    """Test SE(3) equivariance properties"""
    print_section("Part 8: SE(3) Equivariance")
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping SE(3) tests (no checkpoint)")
        return False
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )
    model.eval()
    
    # Test 1: Translation invariance (output is always zero-centered)
    def test_translation_invariance():
        B, N = 2, 10
        nodesxsample = torch.tensor([8, 10])
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        with torch.no_grad():
            x, h, latents, logps, timesteps_list, mus, sigmas = model.sample(
                n_samples=B,
                n_nodes=nodesxsample,
                node_mask=node_mask,
                edge_mask=edge_mask,
                fix_noise=False,
                timestep=10
            )
        
        # Check zero-center
        x_masked = x * node_mask
        x_sum = x_masked.sum(dim=1)
        n_valid = node_mask.sum(dim=1)
        cog = x_sum / (n_valid + 1e-8)
        max_cog = cog.abs().max().item()
        
        print(f"     Translation test: max CoG = {max_cog:.6e}")
        assert max_cog < 1e-5, f"Output not zero-centered: {max_cog}"
        
        return True
    
    run_check("SE(3): translation invariance", test_translation_invariance)
    
    # Test 2: Coordinate range check
    def test_coordinate_range():
        B, N = 2, 10
        nodesxsample = torch.tensor([8, 10])
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        with torch.no_grad():
            x, h, latents, logps, timesteps_list, mus, sigmas = model.sample(
                n_samples=B,
                n_nodes=nodesxsample,
                node_mask=node_mask,
                edge_mask=edge_mask,
                fix_noise=False,
                timestep=10
            )
        
        x_valid = x[node_mask.squeeze(-1).bool()]
        x_min, x_max = x_valid.min().item(), x_valid.max().item()
        x_std = x_valid.std().item()
        
        print(f"     Coordinate range: [{x_min:.3f}, {x_max:.3f}], std={x_std:.3f}")
        
        # Reasonable range check (should be within ~[-5, 5] for typical molecules)
        assert -10 < x_min < 10, f"Coordinate range unusual: min={x_min}"
        assert -10 < x_max < 10, f"Coordinate range unusual: max={x_max}"
        
        return True
    
    run_check("SE(3): coordinate range check", test_coordinate_range)
    
    return True



# ============================================================================
# Part 9: Edge Cases and Boundary Conditions
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print_section("Part 9: Edge Cases")
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping edge case tests (no checkpoint)")
        return False
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )
    model.eval()
    
    # Test 1: Minimum nodes (single atom)
    def test_single_atom():
        B, N = 1, 1
        nodesxsample = torch.tensor([1])
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        with torch.no_grad():
            x, h, latents, logps, timesteps_list, mus, sigmas = model.sample(
                n_samples=B,
                n_nodes=nodesxsample,
                node_mask=node_mask,
                edge_mask=edge_mask,
                fix_noise=False,
                timestep=5
            )
        
        assert x.shape == (1, 1, 3)
        assert h['categorical'].shape == (1, 1, model.atom_nf)
        
        print(f"     Single atom: x={x[0,0].tolist()}")
        print(f"     Atom type: {torch.argmax(h['categorical'][0,0]).item()}")
        
        return True
    
    run_check("Edge case: single atom", test_single_atom)
    
    # Test 2: Maximum practical nodes
    def test_large_molecule():
        B, N = 1, 50
        nodesxsample = torch.tensor([50])
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        with torch.no_grad():
            x, h, latents, logps, timesteps_list, mus, sigmas = model.sample(
                n_samples=B,
                n_nodes=nodesxsample,
                node_mask=node_mask,
                edge_mask=edge_mask,
                fix_noise=False,
                timestep=5
            )
        
        assert x.shape == (1, 50, 3)
        print(f"     Large molecule (50 atoms): x range=[{x.min():.3f}, {x.max():.3f}]")
        
        return True
    
    run_check("Edge case: large molecule (50 atoms)", test_large_molecule)
    
    # Test 3: Extreme batch size
    def test_large_batch():
        B, N = 16, 15
        nodesxsample = torch.randint(10, 15, (B,))
        node_mask, edge_mask = model.get_mask(nodesxsample, B, N)
        
        with torch.no_grad():
            x, h, latents, logps, timesteps_list, mus, sigmas = model.sample(
                n_samples=B,
                n_nodes=nodesxsample,
                node_mask=node_mask,
                edge_mask=edge_mask,
                fix_noise=False,
                timestep=5
            )
        
        assert x.shape == (B, N, 3)
        print(f"     Large batch (16 samples): successful")
        
        return True
    
    run_check("Edge case: large batch (16 samples)", test_large_batch)
    
    return True



# ============================================================================
# Part 10: Pocket Replication Tests
# ============================================================================

def test_pocket_replication():
    """Test _replicate_pocket method"""
    print_section("Part 10: Pocket Replication")
    
    checkpoint_path = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"  ⚠️  Skipping pocket tests (no checkpoint)")
        return False
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    
    model = DiffSBDDModel(
        checkpoint_path=checkpoint_path,
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",
        device="cpu",
        dataset_name="crossdock"
    )
    
    # Test 1: Single batch
    def test_single_batch_pocket():
        pocket = model._replicate_pocket(1)
        
        assert 'x' in pocket
        assert 'one_hot' in pocket
        assert 'size' in pocket
        assert 'mask' in pocket
        
        pocket_size = pocket['size'][0].item()
        assert pocket['x'].shape[0] == pocket_size
        assert pocket['one_hot'].shape[0] == pocket_size
        
        print(f"     Single batch: pocket_size={pocket_size}")
        print(f"     x shape: {pocket['x'].shape}")
        print(f"     one_hot shape: {pocket['one_hot'].shape}")
        
        return True
    
    run_check("Pocket: single batch replication", test_single_batch_pocket)
    
    # Test 2: Multiple batches
    def test_multi_batch_pocket():
        batch_size = 8
        pocket = model._replicate_pocket(batch_size)
        
        pocket_size = pocket['size'][0].item()
        expected_total = pocket_size * batch_size
        
        assert pocket['x'].shape[0] == expected_total
        assert pocket['one_hot'].shape[0] == expected_total
        assert pocket['size'].shape[0] == batch_size
        assert pocket['mask'].shape[0] == expected_total
        
        # Check all sizes are the same
        assert torch.all(pocket['size'] == pocket_size)
        
        print(f"     Multi-batch (8): pocket_size={pocket_size}, total={expected_total}")
        
        return True
    
    run_check("Pocket: multi-batch replication", test_multi_batch_pocket)
    
    # Test 3: Pocket data consistency
    def test_pocket_consistency():
        pocket1 = model._replicate_pocket(1)
        pocket2 = model._replicate_pocket(1)
        
        # Should be identical
        assert torch.equal(pocket1['x'], pocket2['x'])
        assert torch.equal(pocket1['one_hot'], pocket2['one_hot'])
        
        print(f"     Pocket data is consistent across calls")
        
        return True
    
    run_check("Pocket: data consistency", test_pocket_consistency)
    
    return True


# ============================================================================
# Summary and Main
# ============================================================================

def print_summary():
    """Print verification summary"""
    print_section("Verification Summary")
    
    total = len(results)
    passed = sum(1 for _, success, _ in results if success)
    failed = total - passed
    
    print(f"\n  Total tests: {total}")
    print(f"  [PASS] Passed: {passed}")
    print(f"  [FAIL] Failed: {failed}")
    
    # Add to markdown
    add_md_text(f"**Total Tests:** {total}")
    add_md_text(f"**Passed:** ✅ {passed}")
    add_md_text(f"**Failed:** ❌ {failed}")
    add_md_text(f"**Success Rate:** {(passed / total * 100) if total > 0 else 0:.1f}%\n")
    
    # Detailed results table
    add_md_header("Detailed Test Results", level=3)
    add_md_table_row(["#", "Test Name", "Status", "Details"])
    add_md_table_row(["---", "---", "---", "---"])
    
    for idx, (name, success, msg) in enumerate(results, 1):
        status = "✅ PASS" if success else "❌ FAIL"
        details = msg[:100] + "..." if len(msg) > 100 else msg
        details = details.replace("|", "\\|").replace("\n", " ")
        add_md_table_row([idx, name, status, details])
    
    if failed > 0:
        print(f"\n  Failed tests:")
        add_md_header("Failed Tests Details", level=3)
        for name, success, msg in results:
            if not success:
                print(f"    [FAIL] {name}")
                if msg:
                    print(f"       -> {msg}")
                add_md_text(f"**{name}**")
                add_md_code_block(msg, "")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n  Success rate: {success_rate:.1f}%")
    
    # Final verdict
    add_md_header("Final Verdict", level=3)
    if success_rate == 100:
        print(f"\n  ✓ All tests passed! DiffSBDDModel is ready for RLPF integration.")
        add_md_text("✅ **All tests passed!** DiffSBDDModel is ready for RLPF integration.")
    elif success_rate >= 80:
        print(f"\n  ⚠ Most tests passed, but some issues need attention.")
        add_md_text("⚠️ **Most tests passed**, but some issues need attention.")
    else:
        print(f"\n  ✗ Multiple failures detected. Please review and fix.")
        add_md_text("❌ **Multiple failures detected.** Please review and fix.")
    
    return failed == 0



def main():
    parser = argparse.ArgumentParser(
        description="Complete DiffSBDDModel Verification"
    )
    parser.add_argument(
        "--skip-sampling", 
        action="store_true",
        help="Skip time-consuming sampling tests"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (no sampling, no edge cases)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/COMPLETE_MODEL_VERIFICATION_REPORT.md",
        help="Output markdown report path"
    )
    args = parser.parse_args()
    
    # Initialize markdown report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_md_header("Complete DiffSBDDModel Verification Report", level=1)
    add_md_text(f"**Generated:** {timestamp}\n")
    add_md_text("This report contains comprehensive verification results for the DiffSBDDModel implementation.\n")
    
    print("="*70)
    print("  Complete DiffSBDDModel Verification")
    print("="*70)
    
    try:
        # Part 1: Method completeness
        test_method_completeness()
        
        # Part 2: Model initialization
        test_model_initialization()
        
        # Part 3: Mask operations
        test_mask_operations()
        
        # Part 4: Format conversion
        test_format_conversion()
        
        # Part 5: Log probability
        test_log_prob_computation()
        
        # Part 6: Zero-center operations
        test_zero_center_operations()
        
        if not args.skip_sampling and not args.quick:
            # Part 7: Sampling operations
            test_sampling_operations()
            
            # Part 8: SE(3) equivariance
            test_se3_equivariance()
        
        if not args.quick:
            # Part 9: Edge cases
            test_edge_cases()
        
        # Part 10: Pocket replication
        test_pocket_replication()
        
        # Print summary
        all_passed = print_summary()
        
        # Save markdown report
        save_md_report(args.output)
        
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\n  ⚠ Verification interrupted by user")
        add_md_text("\n⚠️ **Verification interrupted by user**\n")
        save_md_report(args.output)
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ✗ Verification failed with error: {e}")
        traceback.print_exc()
        add_md_text(f"\n❌ **Verification failed with error:** `{e}`\n")
        add_md_code_block(traceback.format_exc(), "python")
        save_md_report(args.output)
        sys.exit(1)


if __name__ == "__main__":
    main()
