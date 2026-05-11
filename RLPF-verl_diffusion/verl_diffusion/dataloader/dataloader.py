import torch
from verl_diffusion.protocol import DataProto, TensorDict

class EDMDataLoader:
    def __init__(self, config, dataset_info, nodes_dist, prop_dist, device, condition=False, num_batches=None):
        """
        Initialize the EDM dataloader
        
        Args:
            config: Configuration object containing sampling parameters
            dataset_info: Dictionary containing dataset information
            nodes_dist: Distribution for sampling nodes
            prop_dist: Distribution for sampling properties
            device: Device to put tensors on
            condition: Whether to generate conditional samples
            num_batches: Number of batches per epoch (if None, will run indefinitely)
        """
        self.config = config
        self.dataset_info = dataset_info
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.device = device
        self.condition = condition
        self.num_batches = num_batches
        self.current_batch = 0

        # batch_size: prefer direct config; fallback to legacy product
        _dl = config["dataloader"]
        if "batch_size" in _dl and _dl["batch_size"] is not None:
            self.batch_size = int(_dl["batch_size"])
        else:
            _gs = int(_dl.get("sample_group_size", 4))
            _ep = int(_dl.get("each_prompt_sample", 128))
            self.batch_size = _gs * _ep
        
    def __iter__(self):
        """Make the class iterable"""
        self.current_batch = 0
        return self
        
    def __len__(self):
        """Return the number of batches in an epoch"""
        return self.num_batches if self.num_batches is not None else float('inf')
    
    def __next__(self):
        """Get the next batch of data"""
        if self.num_batches is not None and self.current_batch >= self.num_batches:
            raise StopIteration
            
        self.current_batch += 1
        return self.generate_batch()
        
    def generate_batch(self):
        """
        Generate a batch of data.

        Each sample receives an independent node count.
        No group_index is emitted — advantage normalization is global z-score.

        Returns:
            DataProto: DataProto containing the generated batch data
        """
        tensors = {}
        max_n_nodes = self.dataset_info['max_n_nodes']

        # Every sample gets an independent draw from the node distribution
        nodesxsample = self.nodes_dist.sample(self.batch_size)

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
        """Reset the iterator"""
        self.current_batch = 0
   