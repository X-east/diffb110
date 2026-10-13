"""Freeze reference structures and completed epoch80 batch-two docking inputs."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    """Export four annotated references and every saved batch-two molecule."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RDLogger.DisableLog("rdApp.*")
    source = ROOT / "data/rnd-epi/epi_53_molecules_3d.sdf"
    library = Chem.SDMolSupplier(str(source), removeHs=False)
    box = json.loads((ROOT / "analysis/diffgui_epoch80_independent_eval_261010/docking/run_summary.json").read_text())["box"]
    center = np.asarray([box[f"center_{axis}"] for axis in "xyz"])
    manifest = {"reference_source": str(source), "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "references": []}
    with Chem.SDWriter(str(OUT / "references_input.sdf")) as writer:
        for index, name in [(0, "PABN"), (52, "NMP"), (35, "MBX2319"), (34, "D13-9001")]:
            mol = library[index]
            mol.SetProp("reference_id", name)
            mol.SetProp("_Name", name)
            writer.write(mol)
            manifest["references"].append({"id": name, "source_record_0based": index, "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)), "properties": mol.GetPropsAsDict()})
    batch = ROOT / "OUT/diffgui_epoch80_independent_500steps_6gpu_1000attempts_batch2_20261010_101241"
    files = sorted(batch.glob("outputs/**/home_SDF/*.sdf"), key=lambda p: (str(p.parent), int(p.stem)))
    good, failed = 0, []
    with Chem.SDWriter(str(OUT / "diffgui_batch2_input.sdf")) as writer:
        for path in files:
            mol = Chem.SDMolSupplier(str(path), removeHs=False)[0]
            if mol is None:
                failed.append(str(path))
                continue
            gpu = next(x for x in path.parts if x.startswith("gpu_"))
            record_id = f"b2_g{gpu.split('_')[1]}_{path.stem}"
            mol.SetProp("record_id", record_id)
            mol.SetProp("source_sdf", str(path))
            mol.SetProp("_Name", record_id)
            writer.write(mol)
            good += 1
    manifest["batch2"] = {"directory": str(batch), "attempts": 1000, "saved_files": len(files), "readable": good, "failed": failed, "finished_log": str(batch / "logs/generation.log")}
    with Chem.SDWriter(str(OUT / "references_receptor_frame_input.sdf")) as writer:
        for mol in Chem.SDMolSupplier(str(OUT / "references_input.sdf"), removeHs=False):
            xyz = np.asarray(mol.GetConformer().GetPositions())
            shift = center - xyz.mean(axis=0)
            for atom_index, point in enumerate(xyz + shift):
                mol.GetConformer().SetAtomPosition(atom_index, point)
            mol.SetProp("input_coordinate_frame", "receptor_frame")
            mol.SetProp("reference_input_translation", json.dumps(shift.tolist()))
            writer.write(mol)
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logging.info("Prepared 4 references and %d readable batch-two molecules (%d failures)", good, len(failed))


if __name__ == "__main__":
    main()
