"""Tests for molecular descriptor monitoring helpers."""

from __future__ import annotations

from rdkit import Chem

from ddpo.utils.molecular_descriptors import compute_fingerprint_pairwise_similarities


def test_fingerprint_pairwise_similarity_uses_standalone_rdkit_path() -> None:
    """Fingerprint monitoring should not import the reward package."""
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]

    ecfp4 = compute_fingerprint_pairwise_similarities(mols, use_features=False)
    fcfp4 = compute_fingerprint_pairwise_similarities(mols, use_features=True)

    assert len(ecfp4) == 2
    assert len(fcfp4) == 2
    assert all(0.0 <= value <= 1.0 for value in ecfp4)
    assert all(0.0 <= value <= 1.0 for value in fcfp4)
