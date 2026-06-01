"""Minimal DiffMeta molecular generation smoke test.

The official test-only path hard-codes 5000 generated molecules. This helper
keeps the same DiffMeta model/data construction path, loads a pretrained
checkpoint, and samples a tiny batch for timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from analysis.rdkit_functions import gen_smile_list
from analysis.visualization import MolecularVisualization
from dataset import moses_dataset as ds
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from metrics.molecular_metrics import SamplingMolecularMetrics, TrainMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from model.diffusion_discrete import DiscreteDenoisingDiffusion


def _load_cfg(project_root: pathlib.Path, datadir: pathlib.Path) -> OmegaConf:
    cfg = OmegaConf.create(
        {
            "general": OmegaConf.load(project_root / "configs/general/general_default.yaml"),
            "model": OmegaConf.load(project_root / "configs/model/discrete.yaml"),
            "train": OmegaConf.load(project_root / "configs/train/train_default.yaml"),
            "dataset": OmegaConf.load(project_root / "configs/dataset/moses.yaml"),
        }
    )
    experiment_cfg = project_root / "configs/experiment/moses_ppo.yaml"
    if experiment_cfg.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(experiment_cfg))
    cfg.dataset.datadir = str(datadir)
    cfg.dataset.filter = False
    cfg.general.name = "diffmeta_min_generate"
    cfg.general.wandb = "disabled"
    cfg.general.gpus = 1
    cfg.general.seed = 666
    cfg.train.batch_size = 1
    cfg.train.num_workers = 0
    cfg.train.save_model = False
    return cfg


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _build_model(cfg: OmegaConf, train_smiles: list[str]) -> DiscreteDenoisingDiffusion:
    datamodule = ds.MosesDataModule(cfg)
    dataset_infos = ds.MOSESinfos(datamodule, cfg)

    if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    dataset_infos.train_smiles = train_smiles

    if cfg.model.type == "discrete":
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    else:
        train_metrics = TrainMolecularMetrics(cfg.dataset.remove_h)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    model.train_smiles = train_smiles
    return model


def _load_checkpoint(model: DiscreteDenoisingDiffusion, checkpoint: pathlib.Path) -> None:
    state_dict = torch.load(checkpoint, map_location="cpu")
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=pathlib.Path, required=True)
    parser.add_argument("--datadir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(args.project_root, args.datadir)
    _seed(cfg.general.seed)

    train_path = args.datadir / "raw/train_moses.csv"
    train_smiles = pd.read_csv(train_path)["SMILES"].astype(str).tolist()

    t0 = time.perf_counter()
    model = _build_model(cfg, train_smiles)
    _load_checkpoint(model, args.checkpoint)
    model.eval()
    model.cuda()
    t_loaded = time.perf_counter()

    with torch.no_grad():
        molecules = model.sample_batch(
            batch_id=0,
            batch_size=args.num_samples,
            keep_chain=0,
            number_chain_steps=cfg.general.number_chain_steps,
            save_final=0,
            num_nodes=args.num_nodes if args.num_nodes > 0 else None,
        )
    torch.cuda.synchronize()
    t_sampled = time.perf_counter()

    smiles, valid_ratio, unique, unique_ratio, freq_list = gen_smile_list(molecules, model.dataset_info)

    smiles_path = args.out_dir / "generated_smiles.csv"
    with smiles_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "smiles", "valid"])
        for idx, smi in enumerate(smiles):
            writer.writerow([idx, smi if smi is not None else "", smi is not None])

    torch.save({"molecules": molecules, "smiles": smiles}, args.out_dir / "generated_graphs.pt")

    timing = {
        "num_samples": args.num_samples,
        "num_nodes": args.num_nodes if args.num_nodes > 0 else None,
        "diffusion_steps": int(model.T),
        "setup_load_seconds": t_loaded - t0,
        "sample_seconds": t_sampled - t_loaded,
        "total_seconds": t_sampled - t0,
        "valid_ratio": float(valid_ratio),
        "unique_ratio": float(unique_ratio) if unique_ratio is not None else None,
        "num_unique": len(unique),
        "freq_list": freq_list,
        "smiles": smiles,
    }
    with (args.out_dir / "timing.json").open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
