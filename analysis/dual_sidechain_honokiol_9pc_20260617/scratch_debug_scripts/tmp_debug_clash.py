import sys
sys.path.insert(0, ".")
from rdkit import Chem
from rdkit import Geometry
from ddpo.constraints.dual_sidechain import copy_sidechain_branch_to_anchor

core = Chem.SDMolSupplier(
    "analysis/dual_sidechain_honokiol_9pc_20260617/inpaint_fixed_core_inputs/fixed_core_no_oh.sdf",
    sanitize=False,
)[0]
Chem.SanitizeMol(core)
conf = core.GetConformer()
p = conf.GetAtomPosition(0)
poss = [conf.GetAtomPosition(i) for i in range(core.GetNumAtoms()) if i != 0]
cx = sum(q.x for q in poss) / len(poss)
cy = sum(q.y for q in poss) / len(poss)
cz = sum(q.z for q in poss) / len(poss)
dx, dy, dz = p.x - cx, p.y - cy, p.z - cz
n = (dx * dx + dy * dy + dz * dz) ** 0.5
dx, dy, dz = dx / n, dy / n, dz / n
rw = Chem.RWMol(core)
root = rw.AddAtom(Chem.Atom("C"))
child = rw.AddAtom(Chem.Atom("C"))
rw.AddBond(0, root, Chem.BondType.SINGLE)
rw.AddBond(root, child, Chem.BondType.SINGLE)
c2 = rw.GetConformer()
c2.SetAtomPosition(root, Geometry.Point3D(p.x + dx * 1.5, p.y + dy * 1.5, p.z + dz * 1.5))
c2.SetAtomPosition(child, Geometry.Point3D(p.x + dx * 3.0, p.y + dy * 3.0, p.z + dz * 3.0))
mol = rw.GetMol()
Chem.SanitizeMol(mol)
res = copy_sidechain_branch_to_anchor(mol, anchor1_idx=0, anchor2_idx=11)
print("status", res.status, "clash", res.clash, "min", round(res.min_clash_distance, 3), "pair", res.closest_pair)
if res.closest_pair:
    a, b = res.closest_pair
    print("atoms", a, res.mol.GetAtomWithIdx(a).GetSymbol(), b, res.mol.GetAtomWithIdx(b).GetSymbol())
    cc = res.mol.GetConformer()
    pa = cc.GetAtomPosition(a)
    pb = cc.GetAtomPosition(b)
    print("posA", round(pa.x, 2), round(pa.y, 2), round(pa.z, 2))
    print("posB", round(pb.x, 2), round(pb.y, 2), round(pb.z, 2))
