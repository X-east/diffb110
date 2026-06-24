"""Tests for dual-sidechain phenolic substitution audit helpers."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from ddpo.constraints.dual_sidechain import (
    audit_dual_sidechain,
    build_core_mapping_query,
    build_dual_substitution_spec_from_phenolic_sites,
    extract_sidechain_branch,
    find_phenolic_oh_sites,
    map_core_to_target,
    substitute_dual_r_groups,
)


def test_find_phenolic_oh_sites_on_honokiol_like_molecule() -> None:
    """Honokiol-like phenol has two aromatic-carbon OH substitution sites."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CCc1ccc(O)c(-c2ccc(O)c(CC=C)c2)c1"))
    sites = find_phenolic_oh_sites(mol)

    assert len(sites) == 2
    for site in sites:
        core = mol.GetAtomWithIdx(site.core_atom_idx)
        oxygen = mol.GetAtomWithIdx(site.oxygen_atom_idx)
        assert core.GetAtomicNum() == 6
        assert core.GetIsAromatic()
        assert oxygen.GetAtomicNum() == 8
        assert mol.GetBondBetweenAtoms(site.core_atom_idx, site.oxygen_atom_idx) is not None


def test_audit_reference_honokiol_oh_branches_match() -> None:
    """The two phenolic OH leaving groups match as one-atom substituent branches."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CCc1ccc(O)c(-c2ccc(O)c(CC=C)c2)c1"))
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)

    result = audit_dual_sidechain(mol, spec)

    assert result.dual_success
    assert result.left.atom_symbols == ("O", "H")
    assert result.right.atom_symbols == ("O", "H")
    assert result.attachment_match
    assert result.graph_signature_match


def test_ignore_reference_oh_marks_unsubstituted_sites_missing() -> None:
    """When OH is treated as leaving group, the reference phenol has no replacement R."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CCc1ccc(O)c(-c2ccc(O)c(CC=C)c2)c1"))
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)

    result = audit_dual_sidechain(mol, spec, ignore_leaving_atoms=True)

    assert not result.dual_success
    assert result.failure_type.startswith("left_") or result.failure_type.startswith("right_")
    assert "attachment_missing" in result.failure_type


def test_extract_replacement_branch_directly_from_aromatic_core() -> None:
    """A non-O replacement branch attached to the aromatic core is extracted as R."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccc(CC)c(-c2ccc(CC)c(OCC)c2)c1"))
    sites = find_phenolic_oh_sites(Chem.AddHs(Chem.MolFromSmiles("Oc1ccc(-c2ccc(O)cc2)cc1")))
    # Manual core indices are easier for this synthetic molecule; use the two O-ethyl attachments.
    aromatic_o_sites = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        aromatic_neighbors = [n for n in atom.GetNeighbors() if n.GetIsAromatic()]
        if aromatic_neighbors:
            aromatic_o_sites.append((aromatic_neighbors[0].GetIdx(), atom.GetIdx()))
    assert len(aromatic_o_sites) == 2

    left_core, left_root = aromatic_o_sites[0]
    branch = extract_sidechain_branch(mol, core_atom_idx=left_core, label="R1")

    assert branch.extractable
    assert branch.root_atom_idx == left_root
    assert "O" in branch.atom_symbols
    assert "C" in branch.atom_symbols
    assert branch.attachment_bond_type == "single"
    assert sites  # Keep the phenolic auto-detection path exercised in this test file.


def test_substitute_same_dual_r_groups_passes_replacement_audit() -> None:
    """The same replacement R graph on both sides passes the dual audit."""
    mol = _embedded_honokiol_like_molecule()
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)

    result = substitute_dual_r_groups(mol, spec, left_r_group="ethyl", right_r_group="ethyl")
    audit = audit_dual_sidechain(result.mol, result.adjusted_spec)

    assert audit.dual_success
    assert audit.left.signature == audit.right.signature
    assert audit.left.attachment_bond_type == "single"
    assert result.removed_atom_indices


def test_substitute_different_dual_r_groups_fails_replacement_audit() -> None:
    """Different replacement R graphs fail the dual audit."""
    mol = _embedded_honokiol_like_molecule()
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)

    result = substitute_dual_r_groups(mol, spec, left_r_group="methyl", right_r_group="ethyl")
    audit = audit_dual_sidechain(result.mol, result.adjusted_spec)

    assert not audit.dual_success
    assert audit.failure_type in {"atom_symbols_mismatch", "graph_signature_mismatch"}


def test_substitute_one_sided_r_group_fails_replacement_audit() -> None:
    """A one-sided replacement is rejected as a missing attachment."""
    mol = _embedded_honokiol_like_molecule()
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)

    result = substitute_dual_r_groups(
        mol,
        spec,
        left_r_group="methyl",
        right_r_group=None,
        attach_left=True,
        attach_right=False,
    )
    audit = audit_dual_sidechain(result.mol, result.adjusted_spec)

    assert not audit.dual_success
    assert audit.failure_type == "right_attachment_missing"


def test_fixed_core_without_oh_keeps_honokiol_core_for_inpainting() -> None:
    """Removing OH leaving groups leaves a heavy-atom core suitable for inpainting."""
    mol = _embedded_honokiol_like_molecule()
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)

    result = substitute_dual_r_groups(
        mol,
        spec,
        left_r_group=None,
        right_r_group=None,
        attach_left=False,
        attach_right=False,
        add_hs=False,
    )
    fixed_core = Chem.RemoveHs(result.mol, sanitize=False)
    Chem.SanitizeMol(fixed_core)
    query = build_core_mapping_query(mol, spec)
    mapping = map_core_to_target(fixed_core, query)

    assert fixed_core.GetNumHeavyAtoms() == mol.GetNumHeavyAtoms() - 2
    assert mapping.core_found
    assert mapping.target_spec is not None


def test_core_mapping_finds_adjusted_dual_r_sites_after_substitution() -> None:
    """Core mapping locates the two R attachment cores in a replacement target."""
    mol = _embedded_honokiol_like_molecule()
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)
    query = build_core_mapping_query(mol, spec)
    substituted = substitute_dual_r_groups(mol, spec, left_r_group="ethyl", right_r_group="ethyl")

    mapping = map_core_to_target(substituted.mol, query)
    assert mapping.core_found
    assert mapping.target_spec is not None

    audit = audit_dual_sidechain(substituted.mol, mapping.target_spec)
    assert audit.dual_success
    assert mapping.target_left_core_idx == substituted.adjusted_spec.left_core_idx
    assert mapping.target_right_core_idx == substituted.adjusted_spec.right_core_idx


def test_core_mapping_reports_missing_core() -> None:
    """A molecule without the honokiol core fails before dual-R branch extraction."""
    mol = _embedded_honokiol_like_molecule()
    sites = find_phenolic_oh_sites(mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(sites)
    query = build_core_mapping_query(mol, spec)
    target = Chem.AddHs(Chem.MolFromSmiles("CCO"))

    mapping = map_core_to_target(target, query, compute_mcs=True)

    assert not mapping.core_found
    assert mapping.failure_type == "core_no_match"
    assert mapping.mcs_num_atoms is not None


def _embedded_honokiol_like_molecule() -> Chem.Mol:
    """Return a 3D honokiol-like molecule with explicit hydrogens."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CCc1ccc(O)c(-c2ccc(O)c(CC=C)c2)c1"))
    status = AllChem.EmbedMolecule(mol, randomSeed=17)
    assert status == 0
    Chem.SanitizeMol(mol)
    return mol
