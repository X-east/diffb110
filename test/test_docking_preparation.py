"""Regression coverage for docking input preparation helpers."""

from __future__ import annotations

import os

from ddpo.utils.molecule_semantics import FAILURE_DOCKING_INPUT
import ddpo.rewards.docking_preparation as prep


class _FakeMol:
    pass


def test_prepare_batch_sdf_inputs_tracks_success_and_failures(monkeypatch, tmp_path):
    def fake_build_sample_to_sdf(x, categorical, nodesxsample, sample_idx, sdf_path, dataset_info, **kwargs):
        if sample_idx == 1:
            return False, "sanitize_failed"
        with open(sdf_path, "w", encoding="utf-8") as f:
            f.write(f"mol-{sample_idx}\n")
        return True, None

    monkeypatch.setattr(prep, "build_sample_to_sdf", fake_build_sample_to_sdf)

    def _fake_mkdtemp(prefix):
        d = str(tmp_path / f"{prefix}tmp")
        os.makedirs(d, exist_ok=True)
        return d

    monkeypatch.setattr(prep.tempfile, "mkdtemp", _fake_mkdtemp)

    prepared = prep.prepare_batch_sdf_inputs(
        x=None,
        categorical=None,
        nodesxsample=None,
        batch_size=3,
        dataset_info={},
        prefix="dock_",
    )

    assert prepared.sample_indices == [0, 2]
    assert len(prepared.ligand_files) == 2
    assert prepared.failures == {1: "sanitize_failed"}
    assert all(os.path.exists(path) for path in prepared.ligand_files)

    prepared.cleanup()
    assert not os.path.exists(prepared.tmpdir)



def test_prepare_single_sdf_input_removes_failed_temp_file(monkeypatch, tmp_path):
    temp_file = tmp_path / "single_fail.sdf"

    class _FakeTmp:
        def __enter__(self):
            temp_file.write_text("", encoding="utf-8")
            return type("Tmp", (), {"name": str(temp_file)})()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(prep.tempfile, "NamedTemporaryFile", lambda suffix, delete: _FakeTmp())
    monkeypatch.setattr(
        prep,
        "build_sample_to_sdf",
        lambda **kwargs: (False, FAILURE_DOCKING_INPUT),
    )

    prepared = prep.prepare_single_sdf_input(
        x=None,
        categorical=None,
        nodesxsample=None,
        sample_idx=0,
        dataset_info={},
    )

    assert prepared.path is None
    assert prepared.failure_type == FAILURE_DOCKING_INPUT
    assert not temp_file.exists()



def test_prepare_single_sdf_input_returns_valid_file(monkeypatch, tmp_path):
    temp_file = tmp_path / "single_ok.sdf"

    class _FakeTmp:
        def __enter__(self):
            return type("Tmp", (), {"name": str(temp_file)})()

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_build_sample_to_sdf(x, categorical, nodesxsample, sample_idx, sdf_path, dataset_info, **kwargs):
        with open(sdf_path, "w", encoding="utf-8") as f:
            f.write("ok\n")
        return True, None

    monkeypatch.setattr(prep.tempfile, "NamedTemporaryFile", lambda suffix, delete: _FakeTmp())
    monkeypatch.setattr(prep, "build_sample_to_sdf", fake_build_sample_to_sdf)

    prepared = prep.prepare_single_sdf_input(
        x=None,
        categorical=None,
        nodesxsample=None,
        sample_idx=0,
        dataset_info={},
    )

    assert prepared.path == str(temp_file)
    assert prepared.failure_type is None
    assert temp_file.exists()

    prepared.cleanup()
    assert not temp_file.exists()
