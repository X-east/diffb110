"""Tests for reference DiffGui dependency preflight checks."""

from __future__ import annotations

from ddpo.diffgui.integration.reference_preflight import check_reference_diffgui_preflight


def test_reference_diffgui_preflight_reports_status() -> None:
    """Preflight should report module/file readiness without importing DiffGui."""
    status = check_reference_diffgui_preflight()

    assert "models/model.py" in status.file_available
    assert "easydict" in status.module_available
    assert isinstance(status.ready, bool)
    assert isinstance(status.error_message(), str)
