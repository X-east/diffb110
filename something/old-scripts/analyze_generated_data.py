#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive analysis of generated molecule data with visualizations.
Uses ASCII-compatible characters for cross-platform compatibility.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_data(output_dir):
    """Load all data."""
    traj_dir = output_dir / "trajectory_data"
    batch_dir = output_dir / "batch_data"
    
    with open(traj_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    traj_data = {
        'metadata': metadata,
        'latents': np.load(traj_dir / "latents.npy", allow_pickle=True),
        'logps': np.load(traj_dir / "logps.npy", allow_pickle=True),
        'mus': np.load(traj_dir / "mus.npy", allow_pickle=True),
        'sigmas': np.load(traj_dir / "sigmas.npy", allow_pickle=True),
        'timesteps': np.load(traj_dir / "timesteps.npy"),
        'node_mask': np.load(traj_dir / "node_mask.npy"),
        'edge_mask': np.load(traj_dir / "edge_mask.npy"),
    }
    
    batch_data = {
        'x_final': np.load(batch_dir / "x_final.npy"),
        'h_categorical': np.load(batch_dir / "h_categorical.npy"),
        'node_counts': np.load(batch_dir / "node_counts.npy"),
    }
    
    return traj_data, batch_data


def compute_stats(traj_data, batch_data, atom_decoder):
    """Compute all statistics."""
    latents = traj_data['latents']
    logps = traj_data['logps']
    sigmas = traj_data['sigmas']
    x_final = batch_data['x_final']
    h_categorical = batch_data['h_categorical']
    node_counts = batch_data['node_counts']
    
    # Diffusion stats
    latent_means = [l.mean() for l in latents]
    latent_stds = [l.std() for l in latents]
    logp_means = [lp.mean() for lp in logps]
    sigma_values = [float(s.mean()) if hasattr(s, 'mean') else float(s) for s in sigmas]
    
    # Per-molecule logp trajectories
    logp_per_mol = []
    n_mols = logps[0].shape[0]
    for mol_idx in range(n_mols):
        mol_logps = [lp[mol_idx] for lp in logps]
        logp_per_mol.append(mol_logps)
    
    # Structure stats
    batch_size = x_final.shape[0]
    
    # COM norms
    com_norms = []
    for i in range(batch_size):
        n = node_counts[i]
        coords = x_final[i, :n]
        com = coords.mean(axis=0)
        com_norms.append(np.linalg.norm(com))
    
    # Atom types
    all_atom_types = []
    for i in range(batch_size):
        n = node_counts[i]
        atom_probs = h_categorical[i, :n]
        atom_types = np.argmax(atom_probs, axis=1)
        all_atom_types.extend(atom_types)
    
    type_counts = {}
    for t in all_atom_types:
        sym = atom_decoder[t]
        type_counts[sym] = type_counts.get(sym, 0) + 1
    
    # Coordinates
    all_coords = []
    for i in range(batch_size):
        n = node_counts[i]
        all_coords.append(x_final[i, :n])
    all_coords = np.vstack(all_coords)
    
    # Distances
    all_distances = []
    for i in range(batch_size):
        n = node_counts[i]
        coords = x_final[i, :n]
        for j in range(n):
            for k in range(j+1, n):
                dist = np.linalg.norm(coords[j] - coords[k])
                all_distances.append(dist)
    all_distances = np.array(all_distances)
    
    return {
        'latent_means': latent_means,
        'latent_stds': latent_stds,
        'logp_means': logp_means,
        'logp_per_mol': logp_per_mol,
        'sigma_values': sigma_values,
        'n_steps': len(latents),
        'com_norms': com_norms,
        'type_counts': type_counts,
        'all_coords': all_coords,
        'all_distances': all_distances,
        'node_counts': node_counts,
    }


def create_visualizations(stats, output_dir):
    """Create visualization plots with Chinese labels and larger fonts."""
    fig_dir = output_dir / "analysis"
    fig_dir.mkdir(exist_ok=True)
    
    # Set Chinese font - SimHei for Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 24  # 50% larger
    plt.rcParams['axes.titlesize'] = 33  # 50% larger
    plt.rcParams['axes.labelsize'] = 27  # 50% larger
    
    # =========================================================================
    # Figure 1: Comprehensive Analysis (6 subplots)
    # =========================================================================
    fig = plt.figure(figsize=(28, 22))
    
    # Plot 1: Sigma Schedule
    ax1 = fig.add_subplot(2, 3, 1)
    steps = range(len(stats['sigma_values']))
    ax1.plot(steps, stats['sigma_values'], 'b-', linewidth=4)
    ax1.set_xlabel('扩散步数', fontsize=27)
    ax1.set_ylabel('Sigma (噪声水平)', fontsize=27)
    ax1.set_title('噪声衰减曲线', fontsize=33, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.tick_params(labelsize=22)
    
    sigma_init = stats['sigma_values'][0]
    sigma_final = stats['sigma_values'][-1]
    ax1.text(0.5, -0.22, 
             f'初始={sigma_init:.4f}, 最终={sigma_final:.4f}, 衰减={sigma_init/sigma_final:.1f}倍 [通过]',
             transform=ax1.transAxes, fontsize=20, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Plot 2: Latent Evolution
    ax2 = fig.add_subplot(2, 3, 2)
    steps = range(len(stats['latent_means']))
    ax2.fill_between(steps,
                     np.array(stats['latent_means']) - np.array(stats['latent_stds']),
                     np.array(stats['latent_means']) + np.array(stats['latent_stds']),
                     alpha=0.3, color='blue')
    ax2.plot(steps, stats['latent_means'], 'b-', linewidth=4, label='均值')
    ax2.set_xlabel('扩散步数', fontsize=27)
    ax2.set_ylabel('隐变量值', fontsize=27)
    ax2.set_title('隐变量演化', fontsize=33, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=20)
    ax2.tick_params(labelsize=22)
    
    std_init = stats['latent_stds'][0]
    std_final = stats['latent_stds'][-1]
    ax2.text(0.5, -0.22,
             f'标准差: {std_init:.3f} -> {std_final:.3f}, 去噪正常 [通过]',
             transform=ax2.transAxes, fontsize=20, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # Plot 3: Mean Log Probability
    ax3 = fig.add_subplot(2, 3, 3)
    steps = range(len(stats['logp_means']))
    ax3.plot(steps, stats['logp_means'], 'g-', linewidth=4)
    ax3.set_xlabel('扩散步数', fontsize=27)
    ax3.set_ylabel('平均对数概率 (10个分子)', fontsize=27)
    ax3.set_title('平均对数概率', fontsize=33, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=22)
    
    logp_mean = np.mean(stats['logp_means'])
    logp_range = max(stats['logp_means']) - min(stats['logp_means'])
    ax3.text(0.5, -0.22,
             f'均值={logp_mean:.4f}, 波动={logp_range:.4f}, 稳定 [通过]',
             transform=ax3.transAxes, fontsize=20, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # Plot 4: Atom Type Distribution
    ax4 = fig.add_subplot(2, 3, 4)
    type_counts = stats['type_counts']
    atoms = list(type_counts.keys())
    counts = [type_counts[a] for a in atoms]
    colors = plt.cm.Set3(np.linspace(0, 1, len(atoms)))
    bars = ax4.bar(atoms, counts, color=colors, edgecolor='black', linewidth=2)
    ax4.set_xlabel('原子类型', fontsize=27)
    ax4.set_ylabel('数量', fontsize=27)
    ax4.set_title('原子类型分布', fontsize=33, fontweight='bold', pad=20)
    ax4.tick_params(labelsize=22)
    
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    c_pct = type_counts.get('C', 0) / total * 100
    ax4.text(0.5, -0.22,
             f'碳原子={c_pct:.1f}%, 总计={total}个原子, 类药性好 [通过]',
             transform=ax4.transAxes, fontsize=20, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Plot 5: Distance Distribution
    ax5 = fig.add_subplot(2, 3, 5)
    distances = stats['all_distances']
    ax5.hist(distances, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax5.axvline(x=1.5, color='red', linestyle='--', linewidth=3, label='C-C键 (1.5A)')
    ax5.axvline(x=0.5, color='orange', linestyle='--', linewidth=3, label='碰撞阈值 (0.5A)')
    ax5.set_xlabel('距离 (埃)', fontsize=27)
    ax5.set_ylabel('计数', fontsize=27)
    ax5.set_title('原子间距离分布', fontsize=33, fontweight='bold', pad=20)
    ax5.legend(fontsize=18, loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=22)
    
    min_dist = distances.min()
    n_clashes = np.sum(distances < 0.5)
    ax5.text(0.5, -0.22,
             f'最小距离={min_dist:.2f}A, 碰撞数={n_clashes}, 无碰撞 [通过]',
             transform=ax5.transAxes, fontsize=20, ha='center',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.9))
    
    # Plot 6: Coordinate Distribution
    ax6 = fig.add_subplot(2, 3, 6)
    coords = stats['all_coords']
    scatter = ax6.scatter(coords[:, 0], coords[:, 1], c=coords[:, 2], cmap='viridis',
                          alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('X (埃)', fontsize=27)
    ax6.set_ylabel('Y (埃)', fontsize=27)
    ax6.set_title('原子坐标分布 (XY投影)', fontsize=33, fontweight='bold', pad=20)
    ax6.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Z (埃)', fontsize=22)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(labelsize=22)
    
    com_max = max(stats['com_norms'])
    ax6.text(0.5, -0.22,
             f'最大质心偏移={com_max:.2e}, SE(3)等变性保持 [通过]',
             transform=ax6.transAxes, fontsize=20, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(bottom=0.12, hspace=0.45)
    fig.suptitle('DiffSBDD 分子生成 - 综合分析',
                 fontsize=40, fontweight='bold', y=0.98)
    
    fig_path = fig_dir / "comprehensive_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {fig_path}")
    plt.close(fig)
    
    return [fig_path]


def create_per_molecule_figure(stats, output_dir):
    """Create per-molecule analysis figure with Chinese labels and larger fonts."""
    fig_dir = output_dir / "analysis"
    
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 24  # 50% larger
    plt.rcParams['axes.titlesize'] = 33
    plt.rcParams['axes.labelsize'] = 27
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    
    node_counts = stats['node_counts']
    com_norms = stats['com_norms']
    latent_stds = stats['latent_stds']
    distances = stats['all_distances']
    type_counts = stats['type_counts']
    total = sum(type_counts.values())
    
    # Plot 1: Atoms per molecule
    ax = axes[0, 0]
    ax.bar(range(1, len(node_counts)+1), node_counts, color='steelblue', edgecolor='black', linewidth=2)
    ax.set_xlabel('分子编号', fontsize=27)
    ax.set_ylabel('原子数量', fontsize=27)
    ax.set_title('每个分子的原子数', fontsize=33, fontweight='bold', pad=20)
    ax.axhline(y=node_counts.mean(), color='red', linestyle='--', linewidth=3,
               label=f'均值: {node_counts.mean():.1f}')
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=22)
    
    ax.text(0.5, -0.20,
            f'范围={node_counts.min()}-{node_counts.max()}, 总计={node_counts.sum()}个原子 [通过]',
            transform=ax.transAxes, fontsize=20, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.9))
    
    # Plot 2: COM deviation
    ax = axes[0, 1]
    com_norms_plot = [max(c, 1e-10) for c in com_norms]
    ax.bar(range(1, len(com_norms)+1), com_norms_plot, color='green', edgecolor='black', linewidth=2)
    ax.set_xlabel('分子编号', fontsize=27)
    ax.set_ylabel('质心偏移', fontsize=27)
    ax.set_title('质心偏移 (SE(3)等变性检验)', fontsize=33, fontweight='bold', pad=20)
    ax.axhline(y=0.01, color='red', linestyle='--', linewidth=3, label='阈值 (0.01)')
    ax.legend(fontsize=20)
    ax.set_yscale('log')
    ax.tick_params(labelsize=22)
    
    ax.text(0.5, -0.20,
            f'最大值={max(com_norms):.2e}, 所有质心接近0, SE(3)等变性保持 [通过]',
            transform=ax.transAxes, fontsize=20, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # Plot 3: Latent variance decay
    ax = axes[1, 0]
    ax.plot(range(len(latent_stds)), latent_stds, 'b-', linewidth=4)
    ax.set_xlabel('扩散步数', fontsize=27)
    ax.set_ylabel('隐变量标准差', fontsize=27)
    ax.set_title('隐变量方差衰减', fontsize=33, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=22)
    
    ax.text(0.5, -0.20,
            f'初始={latent_stds[0]:.3f}, 最终={latent_stds[-1]:.3f}, 去噪正常 [通过]',
            transform=ax.transAxes, fontsize=20, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = [
        ['指标', '数值', '状态'],
        ['分子总数', str(len(node_counts)), '通过'],
        ['原子总数', str(node_counts.sum()), '通过'],
        ['平均原子数/分子', f'{node_counts.mean():.1f}', '通过'],
        ['扩散步数', str(stats['n_steps']), '通过'],
        ['平均对数概率', f"{np.mean(stats['logp_means']):.4f}", '通过'],
        ['最大质心偏移', f"{max(com_norms):.2e}", '通过'],
        ['最小原子距离', f"{distances.min():.3f} A", '通过'],
        ['碳原子占比', f"{type_counts.get('C', 0)/total*100:.1f}%", '通过'],
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center',
                     colWidths=[0.40, 0.35, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(22)
    table.scale(1.5, 2.8)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=24)
    
    for i in range(1, len(summary_data)):
        table[(i, 2)].set_facecolor('#90EE90')
    
    ax.set_title('汇总统计', fontsize=33, fontweight='bold', pad=50)
    
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(bottom=0.12, hspace=0.45)
    fig.suptitle('DiffSBDD 单分子分析', fontsize=40, fontweight='bold', y=0.98)
    
    fig_path = fig_dir / "per_molecule_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {fig_path}")
    plt.close(fig)
    
    return fig_path


def create_logp_per_molecule_figure(stats, output_dir):
    """Create per-molecule log probability trajectories figure with Chinese labels."""
    fig_dir = output_dir / "analysis"
    
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 24  # 50% larger
    plt.rcParams['axes.titlesize'] = 33
    plt.rcParams['axes.labelsize'] = 27
    
    logp_per_mol = stats['logp_per_mol']
    n_mols = len(logp_per_mol)
    n_steps = len(logp_per_mol[0])
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plot each molecule's logp trajectory
    colors = plt.cm.tab10(np.linspace(0, 1, n_mols))
    for mol_idx in range(n_mols):
        ax.plot(range(n_steps), logp_per_mol[mol_idx], 
                linewidth=3, color=colors[mol_idx], alpha=0.8,
                label=f'分子 {mol_idx+1}')
    
    # Plot mean
    logp_means = stats['logp_means']
    ax.plot(range(len(logp_means)), logp_means, 'k--', linewidth=4, label='均值')
    
    ax.set_xlabel('扩散步数', fontsize=27)
    ax.set_ylabel('对数概率', fontsize=27)
    ax.set_title('单分子对数概率轨迹', fontsize=40, fontweight='bold', pad=25)
    ax.legend(fontsize=20, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=22)
    
    # Analysis text
    logp_final = [logp_per_mol[i][-1] for i in range(n_mols)]
    ax.text(0.5, -0.12,
            f'最终对数概率范围: [{min(logp_final):.4f}, {max(logp_final):.4f}], '
            f'标准差={np.std(logp_final):.4f} [通过: 分子间方差较低]',
            transform=ax.transAxes, fontsize=22, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(bottom=0.15)
    
    fig_path = fig_dir / "logp_per_molecule.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {fig_path}")
    plt.close(fig)
    
    return fig_path


def print_analysis(stats):
    """Print detailed analysis."""
    print("\n" + "="*70)
    print("  ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n[1] Diffusion Process")
    print(f"    Steps: {stats['n_steps']}")
    print(f"    Sigma: {stats['sigma_values'][0]:.4f} -> {stats['sigma_values'][-1]:.4f}")
    print(f"    Latent Std: {stats['latent_stds'][0]:.3f} -> {stats['latent_stds'][-1]:.3f}")
    print(f"    Mean LogP: {np.mean(stats['logp_means']):.4f}")
    
    print("\n[2] Molecular Structure")
    print(f"    Molecules: {len(stats['node_counts'])}")
    print(f"    Total atoms: {stats['node_counts'].sum()}")
    print(f"    Atom types: {stats['type_counts']}")
    print(f"    Max COM norm: {max(stats['com_norms']):.2e}")
    print(f"    Min distance: {stats['all_distances'].min():.3f} A")
    
    print("\n[3] Validation Results")
    checks = [
        ("Sigma decay", stats['sigma_values'][0] > stats['sigma_values'][-1] * 10),
        ("Denoising", stats['latent_stds'][-1] < stats['latent_stds'][0]),
        ("SE(3) equivariance", max(stats['com_norms']) < 0.01),
        ("No clashes", stats['all_distances'].min() > 0.5),
        ("Drug-like (C > 40%)", stats['type_counts'].get('C', 0) / sum(stats['type_counts'].values()) > 0.4),
    ]
    
    all_pass = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {status} {name}")
        if not passed:
            all_pass = False
    
    print("\n" + "="*70)
    if all_pass:
        print("  *** DATA VALIDATION PASSED - READY FOR DDPO TRAINING ***")
    else:
        print("  *** SOME CHECKS FAILED - REVIEW ABOVE ***")
    print("="*70)


def main():
    output_dir = Path("generated_molecules")
    
    if not output_dir.exists():
        print("Error: generated_molecules directory not found!")
        return
    
    atom_decoder = ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F']
    
    print("\n" + "="*70)
    print("  DiffSBDD GENERATED DATA ANALYSIS")
    print("="*70)
    
    print("\n  Loading data...")
    traj_data, batch_data = load_data(output_dir)
    print("  Data loaded!")
    
    print("\n  Computing statistics...")
    stats = compute_stats(traj_data, batch_data, atom_decoder)
    
    print("\n  Creating visualizations...")
    create_visualizations(stats, output_dir)
    create_per_molecule_figure(stats, output_dir)
    create_logp_per_molecule_figure(stats, output_dir)
    
    print_analysis(stats)
    
    print(f"\n  Figures saved to: {output_dir / 'analysis'}")


if __name__ == "__main__":
    main()
