"""Tests for the DiffGui PoseBusters training gate."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from ddpo.diffgui.runtime.posebusters_filter import run_posebusters_filter
from ddpo.diffgui.runtime.reward_only import load_sdf_records


class _FakePoseBusters:
    """Return one passing and one failing result in call order."""

    def __init__(self) -> None:
        self.calls = 0

    def bust(self, *, mol_pred: str, mol_cond: str) -> pd.DataFrame:
        """Return deterministic boolean checks."""
        del mol_pred, mol_cond
        self.calls += 1
        passes = self.calls == 1
        return pd.DataFrame([{"loaded": True, "geometry_ok": passes}])


class _FakePoseBustersWithMissingCheck:
    """Return a missing required binary check."""

    def bust(self, *, mol_pred: str, mol_cond: str) -> pd.DataFrame:
        """Return one passing check and one missing check."""
        del mol_pred, mol_cond
        return pd.DataFrame([{"loaded": True, "geometry_ok": pd.NA}])


def _write_input_sdf(path: Path) -> None:
    """Write two sanitizable 3D molecules with non-consecutive graph ids."""
    writer = Chem.SDWriter(str(path))
    for graph_index, smiles in [(2, "CCO"), (5, "CCN")]:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=graph_index + 1) == 0
        mol.SetProp("_Name", f"graph_{graph_index}")
        mol.SetProp("mol_index", str(graph_index))
        writer.write(mol)
    writer.close()


def test_posebusters_filter_preserves_graph_indices_and_denominators(tmp_path: Path) -> None:
    """PB output should preserve graph ids and report conditional/attempt rates."""
    input_sdf = tmp_path / "input.sdf"
    receptor = tmp_path / "receptor.pdb"
    output_sdf = tmp_path / "pb_valid.sdf"
    output_csv = tmp_path / "posebusters.csv"
    summary_json = tmp_path / "summary.json"
    _write_input_sdf(input_sdf)
    receptor.write_text("HEADER TEST\nEND\n", encoding="utf-8")

    summary = run_posebusters_filter(
        input_sdf=input_sdf,
        receptor_pdb=receptor,
        output_sdf=output_sdf,
        output_csv=output_csv,
        summary_json=summary_json,
        total_attempts=8,
        buster=_FakePoseBusters(),
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["source_graph_index"]) for row in rows] == [2, 5]
    assert [int(row["pb_valid"]) for row in rows] == [1, 0]
    assert summary["posebusters_sanitize_valid"] == 2
    assert summary["posebusters_pb_valid"] == 1
    assert summary["posebusters_pb_valid_rate_sanitize"] == 0.5
    assert summary["posebusters_pb_valid_rate_attempts"] == 0.125

    records = load_sdf_records(
        output_sdf,
        mol_id_prefix="pb",
        remove_hs=False,
        strict_connectivity=True,
    )
    assert len(records) == 1
    assert records[0].source_graph_index == 2


def test_posebusters_filter_fails_closed_on_missing_binary_check(tmp_path: Path) -> None:
    """A missing or NaN PoseBusters check must never be treated as PB-valid."""
    input_sdf = tmp_path / "input.sdf"
    receptor = tmp_path / "receptor.pdb"
    output_sdf = tmp_path / "pb_valid.sdf"
    output_csv = tmp_path / "posebusters.csv"
    summary_json = tmp_path / "summary.json"
    _write_input_sdf(input_sdf)
    receptor.write_text("HEADER TEST\nEND\n", encoding="utf-8")

    summary = run_posebusters_filter(
        input_sdf=input_sdf,
        receptor_pdb=receptor,
        output_sdf=output_sdf,
        output_csv=output_csv,
        summary_json=summary_json,
        total_attempts=2,
        buster=_FakePoseBustersWithMissingCheck(),
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["posebusters_config"] == "dock_fast"
    assert summary["posebusters_pb_valid"] == 0
    assert [row["pb_valid"] for row in rows] == ["0", "0"]
    assert [row["posebusters_missing_checks"] for row in rows] == ["geometry_ok", "geometry_ok"]
    assert [row["pb_check_geometry_ok"] for row in rows] == ["", ""]
