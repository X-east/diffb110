import logging
import torch
from verl_diffusion.protocol import DataProto, TensorDict

from ddpo.integration import DiffSBDDModel


class DiffSBDDRollout:
    """
    Rollout worker for DiffSBDD model.

    Generates molecular samples using DiffSBDDModel and records
    trajectories for DDPO training.
    """

    def _log_cuda_memory(self, stage: str, device: torch.device | None = None) -> None:
        if not torch.cuda.is_available():
            return
        try:
            current_device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
            allocated_mb = torch.cuda.memory_allocated(current_device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(current_device) / (1024 ** 2)
            max_allocated_mb = torch.cuda.max_memory_allocated(current_device) / (1024 ** 2)
            logging.getLogger("DiffSBDD-DDPO").info(
                "[Memory][rollout] %s | device=%s allocated=%.1fMB reserved=%.1fMB max_allocated=%.1fMB",
                stage,
                current_device,
                allocated_mb,
                reserved_mb,
                max_allocated_mb,
            )
        except Exception:
            logging.getLogger("DiffSBDD-DDPO").exception("Failed to log CUDA memory at stage %s", stage)

    def __init__(self, model: DiffSBDDModel, config: dict):
        self.model = model
        self.config = config
        self.timestep = config["model"].get("time_step", 500)

    def generate_minibatch(self, prompts: DataProto) -> DataProto:
        """Generate a minibatch of molecular samples."""
        batch_size = prompts.batch.batch_size[0]
        max_n_nodes = prompts.meta_info["max_n_nodes"]
        nodesxsample = prompts.batch["nodesxsample"]
        device = nodesxsample.device

        self._log_cuda_memory("generate_minibatch:start", device=device)

        node_mask, edge_mask = self.model.get_mask(nodesxsample, batch_size, max_n_nodes)
        node_mask = node_mask.to(device)
        edge_mask = edge_mask.to(device)

        x, h, latents, logps, timesteps = self.model.sample(
            n_samples=batch_size,
            n_nodes=max_n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            timestep=self.timestep,
        )

        self._log_cuda_memory("generate_minibatch:after_sample", device=device)
        latents_stacked = torch.stack(latents, dim=1)
        logps_stacked = torch.stack(logps, dim=1)
        self._log_cuda_memory("generate_minibatch:after_stack", device=device)

        timesteps_tensor = torch.tensor(timesteps, dtype=torch.long, device=device)
        expanded_timesteps = timesteps_tensor.unsqueeze(0).expand(batch_size, -1).clone()

        assert logps_stacked.shape[1] == expanded_timesteps.shape[1], (
            f"logps length {logps_stacked.shape[1]} != "
            f"timesteps length {expanded_timesteps.shape[1]}"
        )

        h_categorical = h["categorical"] if isinstance(h, dict) else h

        batch_data = {
            "x": x,
            "categorical": h_categorical,
            "latents": latents_stacked,
            "logps": logps_stacked,
            "nodesxsample": nodesxsample,
            "timesteps": expanded_timesteps,
        }

        self._log_cuda_memory("generate_minibatch:before_return", device=device)
        return DataProto(
            batch=TensorDict(batch_data, batch_size=batch_size),
            meta_info=prompts.meta_info.copy(),
        )
