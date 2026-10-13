import rdkit
from rdkit.Chem import rdMolDraw2D as d2d

print("rdkit", rdkit.__version__)
o = d2d.MolDrawOptions()
print(sorted(f for f in dir(o) if not f.startswith("_")))
