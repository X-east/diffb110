from verl_diffusion.protocol import DataProto, TensorDict
import torch


class DiffSBDDDataLoader:
    """
    DataLoader for DiffSBDD DDPO training.

    Generates batches of node counts for molecular generation.
    Unlike EDMDataLoader which samples from a learned distribution,
    this uses fixed or configurable node counts for drug-like molecules.
    """

    def __init__(
        self,
        config: dict,
        dataset_info: dict,
        device: str = "cuda",
        num_batches: int = None,
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
        """Generate a batch of prompts (node counts)."""
        nodesxsample = torch.randint(
            self.min_nodes,
            self.max_nodes + 1,
            (self.group_sample,),
            device=self.device,
        )

        group_index = torch.arange(self.group_sample, device=self.device)
        group_index = group_index.repeat_interleave(self.each_prompt_sample)
        nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample)

        tensors = {
            "nodesxsample": nodesxsample,
            "group_index": group_index,
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
