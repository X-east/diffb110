import torch
from verl_diffusion.protocol import DataProto, TensorDict

from ddpo.integration import DiffSBDDModel
from ddpo.runtime.logging_utils import log_cuda_memory


class DiffSBDDRollout:
    """
    Rollout worker for DiffSBDD model.

    Generates molecular samples using DiffSBDDModel and records
    trajectories for DDPO training.
    """

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

        log_cuda_memory("generate_minibatch:start", component="rollout", device=device)

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

        log_cuda_memory("generate_minibatch:after_sample", component="rollout", device=device)
        latents_stacked = torch.stack(latents, dim=1)
        logps_stacked = torch.stack(logps, dim=1)
        log_cuda_memory("generate_minibatch:after_stack", component="rollout", device=device)

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

        log_cuda_memory("generate_minibatch:before_return", component="rollout", device=device)
        return DataProto(
            batch=TensorDict(batch_data, batch_size=batch_size),
            meta_info=prompts.meta_info.copy(),
        )
