"""Structural pathology rules R1-R5 for E2PO good/bad molecule pools.

This module implements the user-confirmed definition of "unreasonable
molecule" used to build structural preference pairs. It is deliberately
isolated from the existing reward classes (``ChemistryGuardReward`` and
``RingPathologyReward``) because the rules below intentionally **conflict**
with those classes -- most notably the existing guard treats every
three-membered ring as a violation, while the rules here treat an isolated
cyclopropane as reasonable.

Rules
-----
R1  Large ring          : any ring with >= LARGE_RING_MIN_SIZE atoms.
R2  Ring system size    : a connected ring system (rings sharing
                          >= MIN_SHARED_ATOMS_FOR_SAME_SYSTEM atoms) that
                          contains more than MAX_RING_SYSTEM_SIZE rings.
                          Spiro and fused connections both merge rings into
                          the same system (shared >= 1 atom).
R3  Four-membered ring  : any ring with exactly FOUR_RING_SIZE atoms.
R4a Three-ring hetero   : a three-membered ring containing a non-carbon atom.
R4b Three-ring over-attached
                        : a three-membered ring where more than
                          MAX_THREE_RING_EXO_ATTACHED_ATOMS ring atoms carry
                          an exocyclic heavy-atom neighbour.
R4c Three-ring spiro or fused
                        : a three-membered ring sharing at least one atom
                          with another ring.
R5  Carbon fraction     : carbon_fraction > MAX_CARBON_FRACTION. This is a
    composition-drift guard and is reported separately from structural
    pathology when preference pairs are built.

Only heavy atoms (atomic number >= 2) are counted for exocyclic attachment
and for the carbon fraction, matching the project-wide heavy-atom convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule thresholds. Kept as module constants so they can be overridden by a
# config layer later without editing the判据 logic.
# ---------------------------------------------------------------------------
LARGE_RING_MIN_SIZE = 7
FOUR_RING_SIZE = 4
THREE_RING_SIZE = 3
MAX_RING_SYSTEM_SIZE = 3
MIN_SHARED_ATOMS_FOR_SAME_SYSTEM = 1
MAX_THREE_RING_EXO_ATTACHED_ATOMS = 1
MAX_CARBON_FRACTION = 0.85
HEAVY_ATOM_MIN_ATOMIC_NUM = 2

# Column order used when the facts are written to a sidecar CSV.
PATHOLOGY_COLUMNS = (
    "ring_count",
    "max_ring_system_size",
    "large_ring_count",
    "four_ring_count",
    "three_ring_count",
    "bad_three_ring_count",
    "hetero_three_ring_count",
    "over_attached_three_ring_count",
    "spiro_or_fused_three_ring_count",
    "max_three_ring_exo_attached",
    "carbon_fraction",
    "r1_large_ring_violation",
    "r2_ring_system_violation",
    "r3_four_ring_violation",
    "r4a_three_ring_hetero_violation",
    "r4b_three_ring_over_attached_violation",
    "r4c_three_ring_spiro_or_fused_violation",
    "r5_carbon_fraction_violation",
    "r4_any_three_ring_violation",
    "is_pathological",
    "violation_count",
    "is_structurally_pathological",
    "structural_violation_count",
)


# Per-rule flag fields in reporting order. Shared by every consumer so that
# summary tables stay consistent between the DiffSBDD sidecar and the ChEMBL
# control measurement.
RULE_FIELDS: tuple[str, ...] = (
    "r1_large_ring_violation",
    "r2_ring_system_violation",
    "r3_four_ring_violation",
    "r4a_three_ring_hetero_violation",
    "r4b_three_ring_over_attached_violation",
    "r4c_three_ring_spiro_or_fused_violation",
    "r5_carbon_fraction_violation",
)

# R5 is intentionally excluded from the structural preference label.
STRUCTURAL_RULE_FIELDS: tuple[str, ...] = RULE_FIELDS[:-1]


@dataclass(frozen=True)
class StructurePathologyFacts:
    """Per-molecule structural pathology judgement for rules R1-R5."""

    # --- raw counts -------------------------------------------------------
    ring_count: int
    max_ring_system_size: int
    large_ring_count: int
    four_ring_count: int
    three_ring_count: int
    bad_three_ring_count: int
    hetero_three_ring_count: int
    over_attached_three_ring_count: int
    spiro_or_fused_three_ring_count: int
    max_three_ring_exo_attached: int
    carbon_fraction: float

    # --- per-rule flags ---------------------------------------------------
    r1_large_ring_violation: bool
    r2_ring_system_violation: bool
    r3_four_ring_violation: bool
    r4a_three_ring_hetero_violation: bool
    r4b_three_ring_over_attached_violation: bool
    r4c_three_ring_spiro_or_fused_violation: bool
    r5_carbon_fraction_violation: bool

    # --- aggregates -------------------------------------------------------
    r4_any_three_ring_violation: bool
    is_pathological: bool
    violation_count: int
    is_structurally_pathological: bool
    structural_violation_count: int

    def as_row(self) -> dict[str, object]:
        """Return the facts as a CSV-ready dict following PATHOLOGY_COLUMNS."""
        return {
            "ring_count": self.ring_count,
            "max_ring_system_size": self.max_ring_system_size,
            "large_ring_count": self.large_ring_count,
            "four_ring_count": self.four_ring_count,
            "three_ring_count": self.three_ring_count,
            "bad_three_ring_count": self.bad_three_ring_count,
            "hetero_three_ring_count": self.hetero_three_ring_count,
            "over_attached_three_ring_count": self.over_attached_three_ring_count,
            "spiro_or_fused_three_ring_count": self.spiro_or_fused_three_ring_count,
            "max_three_ring_exo_attached": self.max_three_ring_exo_attached,
            "carbon_fraction": self.carbon_fraction,
            "r1_large_ring_violation": int(self.r1_large_ring_violation),
            "r2_ring_system_violation": int(self.r2_ring_system_violation),
            "r3_four_ring_violation": int(self.r3_four_ring_violation),
            "r4a_three_ring_hetero_violation": int(self.r4a_three_ring_hetero_violation),
            "r4b_three_ring_over_attached_violation": int(
                self.r4b_three_ring_over_attached_violation
            ),
            "r4c_three_ring_spiro_or_fused_violation": int(
                self.r4c_three_ring_spiro_or_fused_violation
            ),
            "r5_carbon_fraction_violation": int(self.r5_carbon_fraction_violation),
            "r4_any_three_ring_violation": int(self.r4_any_three_ring_violation),
            "is_pathological": int(self.is_pathological),
            "violation_count": self.violation_count,
            "is_structurally_pathological": int(self.is_structurally_pathological),
            "structural_violation_count": self.structural_violation_count,
        }


def _ring_system_sizes(
    ring_sets: list[set[int]],
    min_shared_atoms: int = MIN_SHARED_ATOMS_FOR_SAME_SYSTEM,
) -> list[int]:
    """Return the ring count of every connected ring system.

    Two rings belong to the same system when they share at least
    ``min_shared_atoms`` atoms. ``min_shared_atoms=1`` merges spiro and fused
    rings; ``2`` merges fused rings only.
    """
    count = len(ring_sets)
    if count == 0:
        return []
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(count):
        for j in range(i + 1, count):
            if len(ring_sets[i] & ring_sets[j]) >= min_shared_atoms:
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    sizes: dict[int, int] = {}
    for i in range(count):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return list(sizes.values())


def _analyze_three_ring(
    mol: Chem.Mol,
    ring: tuple[int, ...],
    ring_sets: list[set[int]],
    ring_index: int,
) -> tuple[bool, bool, int]:
    """Analyse one three-membered ring.

    Returns ``(has_hetero, shares_with_other_ring, exo_attached_atom_count)``.
    ``exo_attached_atom_count`` counts how many ring atoms carry at least one
    exocyclic heavy-atom neighbour.
    """
    ring_set = set(ring)
    has_hetero = any(
        mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in ring
    )
    shares_with_other_ring = any(
        other_index != ring_index and (ring_set & other_set)
        for other_index, other_set in enumerate(ring_sets)
    )
    exo_attached = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for neighbour in atom.GetNeighbors():
            if (
                neighbour.GetIdx() not in ring_set
                and neighbour.GetAtomicNum() >= HEAVY_ATOM_MIN_ATOMIC_NUM
            ):
                exo_attached += 1
                break
    return has_hetero, shares_with_other_ring, exo_attached


def _carbon_fraction(mol: Chem.Mol) -> tuple[float, bool]:
    """Return ``(carbon_fraction, has_no_heavy_atom)``."""
    heavy_atoms = [
        atom for atom in mol.GetAtoms() if atom.GetAtomicNum() >= HEAVY_ATOM_MIN_ATOMIC_NUM
    ]
    if not heavy_atoms:
        return 0.0, True
    carbon_count = sum(1 for atom in heavy_atoms if atom.GetSymbol() == "C")
    return carbon_count / len(heavy_atoms), False


def compute_structure_pathology(mol: Chem.Mol | None) -> StructurePathologyFacts:
    """Evaluate rules R1-R5 for one RDKit molecule.

    A ``None`` molecule is treated as fully pathological (it cannot be judged).
    """
    if mol is None:
        return StructurePathologyFacts(
            ring_count=0,
            max_ring_system_size=0,
            large_ring_count=0,
            four_ring_count=0,
            three_ring_count=0,
            bad_three_ring_count=0,
            hetero_three_ring_count=0,
            over_attached_three_ring_count=0,
            spiro_or_fused_three_ring_count=0,
            max_three_ring_exo_attached=0,
            carbon_fraction=0.0,
            r1_large_ring_violation=True,
            r2_ring_system_violation=False,
            r3_four_ring_violation=False,
            r4a_three_ring_hetero_violation=False,
            r4b_three_ring_over_attached_violation=False,
            r4c_three_ring_spiro_or_fused_violation=False,
            r5_carbon_fraction_violation=True,
            r4_any_three_ring_violation=False,
            is_pathological=True,
            violation_count=2,
            is_structurally_pathological=True,
            structural_violation_count=1,
        )

    atom_rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()]
    ring_sets = [set(ring) for ring in atom_rings]

    large_ring_count = sum(1 for ring in atom_rings if len(ring) >= LARGE_RING_MIN_SIZE)
    four_ring_count = sum(1 for ring in atom_rings if len(ring) == FOUR_RING_SIZE)
    three_ring_indices = [
        index for index, ring in enumerate(atom_rings) if len(ring) == THREE_RING_SIZE
    ]

    system_sizes = _ring_system_sizes(ring_sets)
    max_ring_system_size = max(system_sizes) if system_sizes else 0

    hetero_three = 0
    over_attached_three = 0
    spiro_or_fused_three = 0
    max_exo_attached = 0
    bad_three_ring_count = 0

    for ring_index in three_ring_indices:
        has_hetero, shares, exo_attached = _analyze_three_ring(
            mol, atom_rings[ring_index], ring_sets, ring_index
        )
        max_exo_attached = max(max_exo_attached, exo_attached)
        hetero_hit = has_hetero
        over_attached_hit = exo_attached > MAX_THREE_RING_EXO_ATTACHED_ATOMS
        spiro_or_fused_hit = shares
        if hetero_hit:
            hetero_three += 1
        if over_attached_hit:
            over_attached_three += 1
        if spiro_or_fused_hit:
            spiro_or_fused_three += 1
        if hetero_hit or over_attached_hit or spiro_or_fused_hit:
            bad_three_ring_count += 1

    carbon_fraction, no_heavy_atom = _carbon_fraction(mol)

    r1 = large_ring_count > 0
    r2 = max_ring_system_size > MAX_RING_SYSTEM_SIZE
    r3 = four_ring_count > 0
    r4a = hetero_three > 0
    r4b = over_attached_three > 0
    r4c = spiro_or_fused_three > 0
    r5 = no_heavy_atom or carbon_fraction > MAX_CARBON_FRACTION
    r4_any = r4a or r4b or r4c

    rule_flags = (r1, r2, r3, r4a, r4b, r4c, r5)
    structural_rule_flags = rule_flags[:-1]

    return StructurePathologyFacts(
        ring_count=len(atom_rings),
        max_ring_system_size=max_ring_system_size,
        large_ring_count=large_ring_count,
        four_ring_count=four_ring_count,
        three_ring_count=len(three_ring_indices),
        bad_three_ring_count=bad_three_ring_count,
        hetero_three_ring_count=hetero_three,
        over_attached_three_ring_count=over_attached_three,
        spiro_or_fused_three_ring_count=spiro_or_fused_three,
        max_three_ring_exo_attached=max_exo_attached,
        carbon_fraction=carbon_fraction,
        r1_large_ring_violation=r1,
        r2_ring_system_violation=r2,
        r3_four_ring_violation=r3,
        r4a_three_ring_hetero_violation=r4a,
        r4b_three_ring_over_attached_violation=r4b,
        r4c_three_ring_spiro_or_fused_violation=r4c,
        r5_carbon_fraction_violation=r5,
        r4_any_three_ring_violation=r4_any,
        is_pathological=any(rule_flags),
        violation_count=sum(1 for flag in rule_flags if flag),
        is_structurally_pathological=any(structural_rule_flags),
        structural_violation_count=sum(1 for flag in structural_rule_flags if flag),
    )


def compute_structure_pathology_from_smiles(
    smiles: str,
) -> StructurePathologyFacts | None:
    """Parse a SMILES string and evaluate R1-R5; ``None`` if unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return compute_structure_pathology(mol)
