"""Enforce single-attachment (one-place) constraint on native DiffSBDD inpaint output.

Native DiffSBDD fragment growing expands "in and around" the whole fixed core, so
the grown atoms often attach to several core atoms (bridges/rings). This
post-processor keeps ONLY the grown branch that attaches to C0 (atom 0):

1. Keep all fixed core atoms (0..N_FIXED-1) and only the generated atoms that are
   connected to C0.
2. Drop any bond between a kept generated atom and a fixed core atom other than C0,
   so the branch attaches at exactly one place.
3. Rebuild the molecule (version-independent, no RemoveBond/RemoveAtom), sanitize,
   and verify the only core attachment site is C0.

This realizes "extend from one place (C0), not two" as a pure post-processing
constraint on native DiffSBDD output, without modifying the diffusion model.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from rdkit import Chem

INP = Path(
    "analysis/dual_sidechain_honokiol_9pc_20260617/native_inpaint/native_inpaint_raw.sdf"
)
OUT = Path(
    "analysis/dual_sidechain_honokiol_9pc_20260617/native_inpaint/native_single_place.sdf"
)
N_FIXED = 19


def _kept_generated_branch(mol: Chem.Mol) -> set[int] | None:
    """Return generated-atom indices forming the branch connected to C0, or None."""
    n = mol.GetNumAtoms()
    if n < N_FIXED:
        return None
    c0_neigh = [
        nb.GetIdx()
        for nb in mol.GetAtomWithIdx(0).GetNeighbors()
        if nb.GetIdx() >= N_FIXED
    ]
    if not c0_neigh:
        return None
    seen = set(c0_neigh)
    dq = deque(c0_neigh)
    while dq:
        a = dq.popleft()
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = nb.GetIdx()
            if ni >= N_FIXED and ni not in seen:
                seen.add(ni)
                dq.append(ni)
    return seen


def enforce(mol: Chem.Mol) -> Chem.Mol | None:
    """Return a rebuilt molecule with grown atoms attached only at C0, or None."""
    branch = _kept_generated_branch(mol)
    if not branch:
        return None
    n = mol.GetNumAtoms()
    keep = set(range(N_FIXED)) | branch  # fixed core + C0-connected branch

    rw = Chem.RWMol()
    old2new: dict[int, int] = {}
    new2old: list[int] = []
    conf = mol.GetConformer()
    conf_new = Chem.Conformer(len(keep))
    for old in sorted(keep):
        atom = mol.GetAtomWithIdx(old)
        new_a = Chem.Atom(atom.GetAtomicNum())
        new_a.SetFormalCharge(atom.GetFormalCharge())
        new_idx = rw.AddAtom(new_a)
        old2new[old] = new_idx
        new2old.append(old)
        conf_new.SetAtomPosition(new_idx, conf.GetAtomPosition(old))
    rw.AddConformer(conf_new, assignId=True)

    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if i not in keep or j not in keep:
            continue
        # drop cross-links from the branch to any core atom other than C0
        if (i >= N_FIXED and j < N_FIXED and j != 0) or (
            j >= N_FIXED and i < N_FIXED and i != 0
        ):
            continue
        rw.AddBond(old2new[i], old2new[j], b.GetBondType())

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None

    # verify single attachment at C0 only
    sites = set()
    for a in range(rw.GetNumAtoms()):
        old_a = new2old[a]
        if old_a >= N_FIXED:
            for nb in rw.GetAtomWithIdx(a).GetNeighbors():
                if new2old[nb.GetIdx()] < N_FIXED:
                    sites.add(new2old[nb.GetIdx()])
    if sites == {0}:
        return rw
    return None


def main() -> None:
    sup = Chem.SDMolSupplier(str(INP), removeHs=False, sanitize=False)
    writer = Chem.SDWriter(str(OUT))
    src = [m for m in sup if m is not None]
    kept = 0
    for i, m in enumerate(src):
        res = enforce(m)
        if res is None:
            continue
        res.SetProp("_Name", f"single_place_{i}")
        writer.write(res)
        kept += 1
        try:
            smi = Chem.MolToSmiles(res)
        except Exception:
            smi = "(unsan)"
        n_gen = res.GetNumAtoms() - N_FIXED
        print(f"idx {i}: gen_atoms={n_gen} {smi}")
    writer.close()
    print(f"\nkept single-place molecules: {kept}/{len(src)}")


if __name__ == "__main__":
    main()
