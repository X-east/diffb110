import torch
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.protocol import DataProto, TensorDict

class SequentialDataLoader(EDMDataLoader):
    def __init__(self, config, dataset_info, nodes_dist, prop_dist, device, condition=False, num_batches=None):
        """
        Initialize the Sequential dataloader that iterates through nodes 14-21.
        
        Args:
            config: Configuration object containing sampling parameters
            dataset_info: Dictionary containing dataset information
            nodes_dist: Distribution for sampling nodes (not used in this implementation)
            prop_dist: Distribution for sampling properties
            device: Device to put tensors on
            condition: Whether to generate conditional samples
            num_batches: Number of batches per epoch (if None, will run indefinitely)
        """
        # Calculate number of batches needed to iterate through nodes 14-21
        total_nodes = 21 - 14 + 1  # Total number of nodes to iterate through
        _dl = config["dataloader"]
        batch_size = int(_dl.get("batch_size", int(_dl.get("sample_group_size", 4)) * int(_dl.get("each_prompt_sample", 128))))
        num_batches = (total_nodes + batch_size - 1) // batch_size  # Ceiling division
        
        super().__init__(config, dataset_info, nodes_dist, prop_dist, device, condition, num_batches)
        self.current_node = 14  # Start from 14 nodes
        self.max_node = 21     # End at 21 nodes
        
    def generate_batch(self):
        """
        Generate a batch of data with sequential node numbers.

        Each sample receives an independent node count.
        No group_index is emitted — advantage normalization is global z-score.
        
        Returns:
            DataProto: DataProto containing the generated batch data
        """
        tensors = {}
        max_n_nodes = self.dataset_info['max_n_nodes']
        
        # Create sequential nodesxsample for each sample
        remaining_nodes = self.max_node - self.current_node + 1
        n_sequential = min(self.batch_size, remaining_nodes)
        sequential_nodes = torch.arange(
            self.current_node, self.current_node + n_sequential, device=self.device
        )
        # Update current_node for next batch
        self.current_node += n_sequential
        if self.current_node > self.max_node:
            self.current_node = 14  # Reset when reaching max

        # If batch_size > available sequential nodes, fill the rest with random
        if n_sequential < self.batch_size:
            extra = self.batch_size - n_sequential
            extra_nodes = torch.randint(14, self.max_node + 1, (extra,), device=self.device)
            nodesxsample = torch.cat([sequential_nodes, extra_nodes])
        else:
            nodesxsample = sequential_nodes
        
        if self.condition:
            context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
            tensors.update({
                'nodesxsample': nodesxsample.to(self.device),
                'context': context,
            })
        else:
            tensors.update({
                'nodesxsample': nodesxsample.to(self.device),
            })
        
        # Create meta info
        meta_info = {
            'max_n_nodes': max_n_nodes,
            'condition': self.condition
        }
            
        # Create DataProto
        return DataProto(batch=TensorDict(tensors, batch_size=[self.batch_size]), meta_info=meta_info)
        
    def reset(self):
        """Reset the iterator and node counter"""
        super().reset()
        self.current_node = 14 