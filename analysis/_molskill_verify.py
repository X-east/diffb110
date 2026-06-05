"""验证 MolSkill 评分方向：已知好药 vs 全碳/坏分子 vs ChEMBL."""
import numpy as np, torch, pandas as pd
import sys; sys.path.insert(0, "reference project/molskill")
from rdkit import Chem
from molskill.data.featurizers import get_featurizer
from molskill.models.ranknet import LitRankNet

feat = get_featurizer("morgan_count_rdkit_2d")
model = LitRankNet.load_from_checkpoint(
    checkpoint_path="reference project/molskill/models/default/checkpoints/last.ckpt",
    input_size=feat.dim(), map_location="cpu")
model.eval()

def score_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.nan
    x = torch.from_numpy(feat.get_feat(mol).astype(np.float32)).unsqueeze(0)
    with torch.inference_mode():
        return model.net.score(x).item()

# ── 控制组 ──
tests = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Atorvastatin": "CC(C)C1=C(C(=C(N1CC[C@H](C[C@@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "Sildenafil": "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "Morphine": "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@@H](C=C4)O",
    "全碳_C16": "C1=CC=C2C(=C1)C3=CC=CC=C3C4=CC=CC=C42",
    "全碳_C20": "C1=CC=C2C(=C1)C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C52",
    "全碳_C24": "C1=CC=C2C(=C1)C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C5C6=CC=CC=C62",
    "含汞": "C[Hg]C",
    "全硫_S8": "S1SSSSSSS1",
    "生成分子样例": "CONCCN1CCC(N2CCCC(C(CO)C(N)=O)C2)CC1",
}

print(f"{'Molecule':<22} {'MolSkill':>10}")
print("-" * 35)
for name, smi in tests.items():
    s = score_one(smi)
    print(f"{name:<22} {s:>10.4f}")

# ── ChEMBL 重新统计 ──
print("\n--- ChEMBL 5000 sample ---")
df = pd.read_csv("data/chembl/chembl_36_properties.csv.gz",
                 usecols=["canonical_smiles"], dtype=str, nrows=5000).dropna()
scores = np.array([score_one(s) for s in df["canonical_smiles"].tolist()])
clean = scores[~np.isnan(scores)]
print(f"N={len(clean)}, Mean={np.mean(clean):.4f}, Median={np.median(clean):.4f}, Std={np.std(clean):.4f}")
print(f"P5={np.percentile(clean,5):.4f}, P95={np.percentile(clean,95):.4f}")
print(f"Min={np.min(clean):.4f}, Max={np.max(clean):.4f}")

# ── 生成分子重新统计 ──
print("\n--- Generated molecules ---")
import csv, os
gen_smiles = []
base = "OUT/7gpu-dock-ecfp4-fcfp4_20260526_115341/epochs"
for ep in sorted(os.listdir(base)):
    for f in sorted(os.listdir(f"{base}/{ep}")):
        if f.startswith("molecules_rank") and f.endswith(".csv"):
            with open(f"{base}/{ep}/{f}") as fh:
                for row in csv.DictReader(fh):
                    if row["rdkit_valid"] == "1":
                        gen_smiles.append(row["smiles"])
gen_scores = np.array([score_one(s) for s in gen_smiles])
clean_g = gen_scores[~np.isnan(gen_scores)]
print(f"N={len(clean_g)}, Mean={np.mean(clean_g):.4f}, Median={np.median(clean_g):.4f}, Std={np.std(clean_g):.4f}")
print(f"P5={np.percentile(clean_g,5):.4f}, P95={np.percentile(clean_g,95):.4f}")
print(f"Min={np.min(clean_g):.4f}, Max={np.max(clean_g):.4f}")
