"""Regression coverage for docking backend factory policy.

These tests avoid running real docking binaries. They only validate that
mainline reward/backend assembly stays explicit and that legacy/debug backends
remain clearly downgraded in factory policy.
"""

from __future__ import annotations

import logging

import ddpo.rewards.factory as factory


class _DummyBackend:
    def __init__(self, support_level: str):
        self.support_level = support_level


def _make_config(backend: str) -> dict:
    return {
        "reward": {
            "docking": {
                "backend": backend,
                "fld_file": "/tmp/fake.maps.fld",
                "gpu_id": 0,
                "nrun": 20,
                "heuristics": True,
                "search_mode": "fast",
                "num_modes": 1,
                "exhaustiveness": 8,
            }
        },
        "pocket": {
            "pdb_path": "/tmp/fake_receptor.pdbqt",
            "pdbqt_path": "/tmp/fake_receptor.pdbqt",
        },
    }


def test_mainline_backend_registry_marks_only_autodock_gpu_supported():
    assert factory.DOCKING_BACKEND_REGISTRY["autodock_gpu"].support_level == "supported"
    assert factory.DOCKING_BACKEND_REGISTRY["unidock"].support_level == "legacy"
    assert factory.DOCKING_BACKEND_REGISTRY["smina"].support_level == "debug"
    assert factory.MAINLINE_DOCKING_BACKENDS == {"autodock_gpu"}


def test_build_docking_backend_uses_registry_spec(monkeypatch):
    monkeypatch.setitem(
        factory.DOCKING_BACKEND_REGISTRY,
        "autodock_gpu",
        factory.DockingBackendSpec(
            name="autodock_gpu",
            support_level="supported",
            execution_mode="batch",
            builder=lambda cfg: _DummyBackend("supported"),
        ),
    )
    backend_impl, backend_name, execution_mode, support_level = factory._build_docking_backend(
        _make_config("autodock_gpu"),
        logging.getLogger("test"),
    )
    assert isinstance(backend_impl, _DummyBackend)
    assert backend_name == "autodock_gpu"
    assert execution_mode == "batch"
    assert support_level == "supported"


def test_build_docking_backend_returns_legacy_backend_policy(monkeypatch):
    monkeypatch.setitem(
        factory.DOCKING_BACKEND_REGISTRY,
        "unidock",
        factory.DockingBackendSpec(
            name="unidock",
            support_level="legacy",
            execution_mode="batch",
            builder=lambda cfg: _DummyBackend("legacy"),
        ),
    )
    backend_impl, backend_name, execution_mode, support_level = factory._build_docking_backend(
        _make_config("unidock"),
        logging.getLogger("test"),
    )
    assert isinstance(backend_impl, _DummyBackend)
    assert backend_name == "unidock"
    assert execution_mode == "batch"
    assert support_level == "legacy"


def test_invalid_backend_name_raises_clear_error():
    logger = logging.getLogger("test")
    try:
        factory._build_docking_backend(_make_config("bad_backend"), logger)
    except ValueError as exc:
        message = str(exc)
        assert "Invalid docking backend" in message
        assert "autodock_gpu" in message
        assert "unidock" in message
        assert "smina" in message
    else:
        raise AssertionError("Expected ValueError for invalid docking backend")
