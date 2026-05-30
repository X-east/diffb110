import torch

from ddpo.runtime.protocol import DataProto, TensorDict

from ddpo.integration import DiffSBDDModel
from ddpo.runtime.logging_utils import log_cuda_memory


class DiffSBDDRollout:
    """
    Rollout worker for DiffSBDD model.

    Generates molecular samples using DiffSBDDModel and records
    trajectories for DDPO training. Supports splitting a large batch
    into smaller micro-batches during generation to reduce peak GPU
    memory, then recombining the results.

    Micro-batch size is controlled by ``config["dataloader"]["rollout_micro_batch_size"]``;
    defaults to the full ``per_rank_batch_size`` (no splitting).
    """

    def __init__(self, model: DiffSBDDModel, config: dict):
        self.model = model
        self.config = config
        self.timestep = config["model"].get("time_step", 500)
        self._micro_batch_size = config.get("dataloader", {}).get("rollout_micro_batch_size")

    def generate_samples(self, prompts: DataProto) -> DataProto:
        """Generate molecular samples, optionally splitting into micro-batches.

        When ``rollout_micro_batch_size`` is configured and smaller than the
        prompt batch size, the batch is split into chunks of that size, each
        chunk is generated via :meth:`generate_minibatch`, and the results are
        concatenated back together.

        This reduces peak GPU memory during the diffusion rollout at the cost
        of a small overhead for chunk/merge bookkeeping.

        Args:
            prompts: DataProto with dim-0 batch-size ``B`` and ``nodesxsample``.

        Returns:
            DataProto containing generated molecules for the full batch.
        """
        batch_size = prompts.batch.batch_size[0]

        # No chunking if micro_batch_size is not configured or equals batch_size ------
        if self._micro_batch_size is None or self._micro_batch_size >= batch_size:
            return self.generate_minibatch(prompts)

        # Split prompts by index ranges and generate each chunk -----------------------
        results: list[DataProto] = []
        for start in range(0, batch_size, self._micro_batch_size):
            end = min(start + self._micro_batch_size, batch_size)
            indices = list(range(start, end))
            chunk = prompts.select_idxs(indices)
            result = self.generate_minibatch(chunk)
            results.append(result)

        # Concatenate results ---------------------------------------------------------
        combined = DataProto.concat(results)
        return combined

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

        # Rollout trajectories are stored for PPO old-policy replay.  They do not
        # require gradients after sampling, so keep the exchange object on CPU to
        # make large effective rollout batches feasible.
        batch_data = {
            "x": x.detach().to("cpu"),
            "categorical": h_categorical.detach().to("cpu"),
            "latents": latents_stacked.detach().to("cpu"),
            "logps": logps_stacked.detach().to("cpu"),
            "nodesxsample": nodesxsample.detach().to("cpu"),
            "timesteps": expanded_timesteps.detach().to("cpu"),
        }

        result = DataProto(
            batch=TensorDict(batch_data, batch_size=batch_size),
            meta_info=prompts.meta_info.copy(),
        )
        del x, h, h_categorical, latents, logps, latents_stacked, logps_stacked
        del timesteps_tensor, expanded_timesteps, node_mask, edge_mask, batch_data
        log_cuda_memory("generate_minibatch:before_return", component="rollout", device=device)
        return result
