import sys
sys.path.insert(0, ".")
from rdkit import Chem
from rdkit import Geometry


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def unit(v, fb):
    n = dot(v, v) ** 0.5
    if n < 1e-8:
        return fb
    return (v[0] / n, v[1] / n, v[2] / n)


def basis(d):
    x = unit(d, (1, 0, 0))
    ref = (0, 0, 1)
    if abs(dot(x, ref)) > 0.95:
        ref = (0, 1, 0)
    y = unit((x[1] * ref[2] - x[2] * ref[1], x[2] * ref[0] - x[0] * ref[2], x[0] * ref[1] - x[1] * ref[0]), (0, 1, 0))
    z = unit((x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]), (0, 0, 1))
    return x, y, z


core = Chem.SDMolSupplier(
    "analysis/dual_sidechain_honokiol_9pc_20260617/inpaint_fixed_core_inputs/fixed_core_no_oh.sdf",
    sanitize=False,
)[0]
Chem.SanitizeMol(core)
conf = core.GetConformer()
p0 = conf.GetAtomPosition(0)
poss = [conf.GetAtomPosition(i) for i in range(core.GetNumAtoms()) if i != 0]
cx = sum(q.x for q in poss) / len(poss)
cy = sum(q.y for q in poss) / len(poss)
cz = sum(q.z for q in poss) / len(poss)
dx, dy, dz = p0.x - cx, p0.y - cy, p0.z - cz
n = (dx * dx + dy * dy + dz * dz) ** 0.5
dx, dy, dz = dx / n, dy / n, dz / n
rw = Chem.RWMol(core)
root = rw.AddAtom(Chem.Atom("C"))
child = rw.AddAtom(Chem.Atom("C"))
rw.AddBond(0, root, Chem.BondType.SINGLE)
rw.AddBond(root, child, Chem.BondType.SINGLE)
c2 = rw.GetConformer()
c2.SetAtomPosition(root, Geometry.Point3D(p0.x + dx * 1.5, p0.y + dy * 1.5, p0.z + dz * 1.5))
c2.SetAtomPosition(child, Geometry.Point3D(p0.x + dx * 3.0, p0.y + dy * 3.0, p0.z + dz * 3.0))
mol = rw.GetMol()
Chem.SanitizeMol(mol)

# local-frame copy anchor1=0 -> anchor2=11
conf = mol.GetConformer()
a1 = conf.GetAtomPosition(0)
a2 = conf.GetAtomPosition(11)
# outward dirs
def outward(m, anchor):
    c = m.GetConformer()
    pa = c.GetAtomPosition(anchor)
    ps = [c.GetAtomPosition(i) for i in range(m.GetNumAtoms()) if i != anchor]
    ccx = sum(q.x for q in ps) / len(ps)
    ccy = sum(q.y for q in ps) / len(ps)
    ccz = sum(q.z for q in ps) / len(ps)
    return unit((pa.x - ccx, pa.y - ccy, pa.z - ccz), (1, 0, 0))

e1, f1, g1 = basis(outward(mol, 0))
e2, f2, g2 = basis(outward(mol, 11))
print("outward1", outward(mol, 0))
print("outward2", outward(mol, 11))
# R1 atoms: root=18? find atoms bonded to 0
r1_atoms = [a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 6]
print("r1 atoms", r1_atoms)
for ai in r1_atoms + [a.GetIdx() for a in mol.GetAtomWithIdx(r1_atoms[0]).GetNeighbors() if a.GetAtomicNum() == 6 and a.GetIdx() != 0]:
    p = conf.GetAtomPosition(ai)
    rel = (p.x - a1.x, p.y - a1.y, p.z - a1.z)
    a = dot(rel, e1); b = dot(rel, f1); c = dot(rel, g1)
    nx = a2.x + a * e2[0] + b * f2[0] + c * g2[0]
    ny = a2.y + a * e2[1] + b * f2[1] + c * g2[1]
    nz = a2.z + a * e2[2] + b * f2[2] + c * g2[2]
    print(f"atom {ai}: rel={tuple(round(x,2) for x in rel)} -> copied=({round(nx,2)},{round(ny,2)},{round(nz,2)})")
# print anchor2 neighbors (core) positions to see clash
print("anchor2 neighbors:")
for nb in mol.GetAtomWithIdx(11).GetNeighbors():
    pp = conf.GetAtomPosition(nb.GetIdx())
    print(f"  core atom {nb.GetIdx()} ({nb.GetSymbol()}) at ({round(pp.x,2)},{round(pp.y,2)},{round(pp.z,2)})")
