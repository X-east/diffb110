"""
Molecule Recorder: 记录训练中生成分子的全量明细与epoch汇总。

.. deprecated::
    MoleculeRecorder 已废弃，将在 P4b recording 子系统接入主线完成后删除。
    功能已由 ddpo.runtime.recording.RunRecorder（及 writers/ 子模块）完整覆盖。

    迁移路径：
        - 分子记录：RunRecorder.on_batch_end(BatchFacts)
        - epoch 合并：RunRecorder.on_epoch_end(EpochFacts)
        - 全局汇总：RunRecorder.on_run_finished(RunLifecycleFacts)

    P4b 接入验证通过后：
        1. lightning_ddpo_module.py 中的 MoleculeRecorder 调用全部换成 RunRecorder
        2. 本文件改为空 shim（只保留 import 兼容）
        3. 最终从代码库删除

设计目标（历史）：
1. DDP 多卡下每个 rank 都记录本地分子（不丢样本）。
2. rank0 在每个 epoch 结束后合并所有 rank 的当轮数据，生成全量分子表。
3. rank0 生成每个 epoch 的全局汇总表，与终端摘要口径一致。
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

log = logging.getLogger("DiffSBDD-DDPO")


class MoleculeRecorder:
    """记录生成分子明细，并在 rank0 汇总全局 CSV。"""

    def __init__(self, output_dir: str, rank: int = 0, world_size: int = 1):
        self.output_dir = output_dir
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.molecules_dir = os.path.join(output_dir, "molecules")
        os.makedirs(self.molecules_dir, exist_ok=True)

        # 每个 rank 自己的明细文件
        self.rank_csv_path = os.path.join(self.molecules_dir, f"molecules_rank{self.rank}.csv")
        self.rank_csv_file = open(self.rank_csv_path, "w", newline="", encoding="utf-8")
        self.rank_csv_writer = csv.DictWriter(
            self.rank_csv_file,
            fieldnames=[
                "epoch",
                "batch",
                "rank",
                "molecule_id",
                "smiles",
                "rdkit_valid",  # Bug4 Fix-6: canonical field name
                "qed",
                "sa_raw",
                "reward_total",
                "reward_druglikeness",
                "reward_docking",
            ],
        )
        self.rank_csv_writer.writeheader()
        self.rank_csv_file.flush()

        # rank0 全量汇总文件
        self.all_csv_path = os.path.join(self.molecules_dir, "molecules_all.csv")
        self.epoch_summary_csv_path = os.path.join(self.molecules_dir, "epoch_summary_global.csv")

        if self.rank == 0:
            self.all_csv_file = open(self.all_csv_path, "w", newline="", encoding="utf-8")
            self.all_csv_writer = csv.DictWriter(
                self.all_csv_file,
                fieldnames=[
                    "epoch",
                    "batch",
                    "rank",
                    "molecule_id",
                    "smiles",
                    "rdkit_valid",  # Bug4 Fix-6: canonical field name
                    "qed",
                    "sa_raw",
                    "reward_total",
                    "reward_druglikeness",
                    "reward_docking",
                ],
            )
            self.all_csv_writer.writeheader()
            self.all_csv_file.flush()

            self.epoch_summary_csv_file = open(self.epoch_summary_csv_path, "w", newline="", encoding="utf-8")
            self.epoch_summary_csv_writer = csv.DictWriter(
                self.epoch_summary_csv_file,
                fieldnames=[
                    "epoch",
                    "generated_total",
                    "rdkit_valid",
                    "rdkit_valid_ratio",
                    "loss",
                    "reward_mean",
                    "avg_qed",
                    "avg_sa",
                ],
            )
            self.epoch_summary_csv_writer.writeheader()
            self.epoch_summary_csv_file.flush()
        else:
            self.all_csv_file = None
            self.all_csv_writer = None
            self.epoch_summary_csv_file = None
            self.epoch_summary_csv_writer = None

        # 总体统计（rank0在merge时更新为全局统计）
        self.total_molecules = 0
        self.total_valid = 0

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    @staticmethod
    def _get_item(values: Any, idx: int, default: Any) -> Any:
        if values is None:
            return default
        if isinstance(values, torch.Tensor):
            if idx < values.shape[0]:
                return values[idx]
            return default
        if isinstance(values, list):
            if idx < len(values):
                return values[idx]
            return default
        return default

    def record_batch(self, epoch: int, batch_idx: int, samples: Dict[str, Any]) -> None:
        """记录当前 rank 本批次分子明细。"""
        smiles_list = samples.batch.get("smiles", [])
        qed_list = samples.batch.get("qed_scores", [])
        sa_list = samples.batch.get("sa_scores", [])
        rewards_list = samples.batch.get("rewards", [])
        reward_druglikeness_list = samples.batch.get("druglikeness_reward", None)
        reward_docking_list = samples.batch.get("docking_reward", None)

        valid_list = samples.batch.get("rdkit_valid")
        if valid_list is None:
            log.error("[ERROR] Missing required field 'rdkit_valid' in samples.batch during CSV recording")
            raise ValueError("Missing required field 'rdkit_valid' in samples.batch")

        n_rows = len(valid_list)

        for mol_id in range(n_rows):
            smiles = self._get_item(smiles_list, mol_id, None)
            qed = self._get_item(qed_list, mol_id, 0.0)
            sa = self._get_item(sa_list, mol_id, 0.0)
            reward_total = self._get_item(rewards_list, mol_id, 0.0)
            reward_druglikeness = self._get_item(reward_druglikeness_list, mol_id, None)
            reward_docking = self._get_item(reward_docking_list, mol_id, None)
            valid = self._get_item(valid_list, mol_id, 0)

            row = {
                "epoch": int(epoch),
                "batch": int(batch_idx),
                "rank": self.rank,
                "molecule_id": int(mol_id),
                "smiles": str(smiles),
                "rdkit_valid": self._to_int(valid, 0),
                "qed": f"{self._to_float(qed, 0.0):.6f}",
                "sa_raw": f"{self._to_float(sa, 0.0):.6f}",
                "reward_total": f"{self._to_float(reward_total, 0.0):.6f}",
                "reward_druglikeness": (
                    f"{self._to_float(reward_druglikeness, 0.0):.6f}"
                    if reward_druglikeness is not None
                    else ""
                ),
                "reward_docking": (
                    f"{self._to_float(reward_docking, 0.0):.6f}"
                    if reward_docking is not None
                    else ""
                ),
            }

            self.rank_csv_writer.writerow(row)

        self.rank_csv_file.flush()

    def flush(self) -> None:
        if self.rank_csv_file and not self.rank_csv_file.closed:
            self.rank_csv_file.flush()
        if self.all_csv_file and not self.all_csv_file.closed:
            self.all_csv_file.flush()
        if self.epoch_summary_csv_file and not self.epoch_summary_csv_file.closed:
            self.epoch_summary_csv_file.flush()

    def _read_rank_epoch_rows(self, rank: int, epoch: int) -> List[Dict[str, str]]:
        rank_path = os.path.join(self.molecules_dir, f"molecules_rank{rank}.csv")
        if not os.path.exists(rank_path):
            return []

        rows: List[Dict[str, str]] = []
        with open(rank_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("epoch", -1)) == int(epoch):
                    rows.append(row)
        return rows

    def merge_epoch_from_ranks(
        self,
        epoch: int,
        loss: float,
        reward_mean: float,
        avg_qed: float,
        avg_sa: float,
    ) -> Dict[str, float]:
        """仅 rank0 调用：合并指定 epoch 的所有 rank 明细并写全局汇总。"""
        if self.rank != 0:
            return {}

        merged_rows: List[Dict[str, str]] = []
        for r in range(self.world_size):
            merged_rows.extend(self._read_rank_epoch_rows(r, epoch))

        for row in merged_rows:
            self.all_csv_writer.writerow(row)
        self.all_csv_file.flush()

        generated_total = len(merged_rows)
        rdkit_valid = sum(int(row.get("rdkit_valid", 0)) for row in merged_rows)
        rdkit_valid_ratio = rdkit_valid / generated_total if generated_total > 0 else 0.0

        summary_row = {
            "epoch": int(epoch),
            "generated_total": int(generated_total),
            "rdkit_valid": int(rdkit_valid),
            "rdkit_valid_ratio": f"{rdkit_valid_ratio:.6f}",
            "loss": f"{float(loss):.6f}",
            "reward_mean": f"{float(reward_mean):.6f}",
            "avg_qed": f"{float(avg_qed):.6f}",
            "avg_sa": f"{float(avg_sa):.6f}",
        }
        self.epoch_summary_csv_writer.writerow(summary_row)
        self.epoch_summary_csv_file.flush()

        # rank0 维护全局累计统计
        self.total_molecules += generated_total
        self.total_valid += rdkit_valid

        return {
            "generated_total": float(generated_total),
            "rdkit_valid": float(rdkit_valid),
            "rdkit_valid_ratio": float(rdkit_valid_ratio),
            "loss": float(loss),
            "reward_mean": float(reward_mean),
            "avg_qed": float(avg_qed),
            "avg_sa": float(avg_sa),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_molecules": int(self.total_molecules),
            "total_valid": int(self.total_valid),
            "valid_ratio": self.total_valid / max(self.total_molecules, 1),
        }

    def close(self) -> None:
        for fh in [self.rank_csv_file, self.all_csv_file, self.epoch_summary_csv_file]:
            if fh is not None and not fh.closed:
                fh.flush()
                fh.close()

    def __del__(self):
        self.close()
