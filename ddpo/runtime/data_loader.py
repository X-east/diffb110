from verl_diffusion.protocol import DataProto, TensorDict
import torch
import logging

logger = logging.getLogger(__name__)


class DiffSBDDDataLoader:
    """
    DataLoader for DiffSBDD DDPO training.

    Generates batches of node counts for molecular generation.
    Supports two node sampling strategies controlled by config:
      - "uniform":            torch.randint(min_nodes, max_nodes+1)
      - "size_distribution":  P(N_lig | N_pocket) from learned conditional distribution
    """

    def __init__(
        self,
        config: dict,
        dataset_info: dict,
        device: str = "cuda",
        num_batches: int = None,
        size_distribution=None,
        pocket_size_tensor=None,
    ):
        self.config = config
        self.dataset_info = dataset_info
        self.device = device
        self.num_batches = num_batches or config["dataloader"].get("epoches", config["dataloader"].get("epochs", 100))

        self.group_sample = config["dataloader"].get("sample_group_size", 4)
        self.each_prompt_sample = config["dataloader"].get("each_prompt_sample", 32)
        self.batch_size = self.group_sample * self.each_prompt_sample

        self.min_nodes = config["dataloader"].get("min_nodes", 10)
        self.max_nodes = config["dataloader"].get("max_nodes", 50)
        self.max_n_nodes = config["dataloader"].get(
            "max_n_nodes",
            config["dataloader"].get("max_nodes", dataset_info.get("max_n_nodes", 50)),
        )

        # Node sampling strategy
        self.node_sampling = config["dataloader"].get("node_sampling", "uniform")
        if self.node_sampling not in ("uniform", "size_distribution"):
            raise ValueError(
                f"Invalid node_sampling={self.node_sampling!r}. "
                f"Must be 'uniform' or 'size_distribution'."
            )

        # Conditional size distribution (optional, from DiffSBDD model)
        self.size_distribution = size_distribution
        self.pocket_size_tensor = pocket_size_tensor

        # Determine which strategy is actually active at runtime
        self._use_conditional = (
            self.node_sampling == "size_distribution"
            and self.size_distribution is not None
            and self.pocket_size_tensor is not None
        )
        if self.node_sampling == "size_distribution" and not self._use_conditional:
            logger.warning(
                "[DiffSBDDDataLoader] node_sampling='size_distribution' requested but "
                "size_distribution or pocket_size_tensor is missing. Falling back to uniform."
            )
        if self._use_conditional:
            # size_distribution can produce values well beyond the uniform range;
            # ensure max_n_nodes is large enough for padding allocation.
            dist_max = getattr(self.size_distribution, 'prob', None)
            if dist_max is not None:
                # prob shape[0] - 1 == max ligand atom count in the histogram
                hist_max_lig = dist_max.shape[0] - 1
                if self.max_n_nodes < hist_max_lig:
                    logger.warning(
                        "[DiffSBDDDataLoader] max_n_nodes=%d < distribution max=%d. "
                        "Padding will be insufficient for large samples from "
                        "size_distribution. Consider raising max_n_nodes to >= %d.",
                        self.max_n_nodes, hist_max_lig, hist_max_lig,
                    )

        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __len__(self):
        return 1

    def __next__(self):
        if self.current_batch >= 1:
            raise StopIteration
        self.current_batch += 1
        return self.generate_batch()

    def generate_batch(self):
        """Generate a batch of prompts (node counts).

        Two strategies:
          1. uniform:            Uniform integer sampling in [min_nodes, max_nodes].
                                 Samples ``group_sample`` values then repeats each
                                 ``each_prompt_sample`` times for PPO advantage
                                 estimation over shared prompts.
          2. size_distribution:  Conditional sampling P(N_lig | N_pocket) from the
                                 learned joint distribution over training data.
                                 Each of the ``batch_size`` samples draws an
                                 independent node count so the full diversity of
                                 the conditional distribution is realised.
        """
        if self._use_conditional:
            # --- Strategy 2: size_distribution (conditional) ---
            # Every sample gets its own independent draw from P(N_lig | N_pocket).
            # The learned distribution already encodes the full training-data
            # statistics (CrossDocked ligand atom counts ~5–55).  No artificial
            # clamping is applied here — this matches the original DiffSBDD
            # generate_ligands() behaviour which only does clamp(min=0).
            pocket_sizes = self.pocket_size_tensor.repeat(self.batch_size)
            nodesxsample = self.size_distribution.sample_conditional(
                n1=None, n2=pocket_sizes,
            )
            nodesxsample = nodesxsample.long()

        else:
            # --- Strategy 1: uniform (default, original behavior) ---
            nodesxsample = torch.randint(
                self.min_nodes,
                self.max_nodes + 1,
                (self.group_sample,),
                device=self.device,
            )

            nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample)

        tensors = {
            "nodesxsample": nodesxsample,
        }

        meta_info = {
            "max_n_nodes": self.max_n_nodes,
            "condition": False,
        }

        return DataProto(
            batch=TensorDict(tensors, batch_size=[self.batch_size]),
            meta_info=meta_info,
        )

    def reset(self):
        self.current_batch = 0
