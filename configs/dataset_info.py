"""
DiffSBDD dataset_info configuration.

Provides dataset information for DiffSBDD crossdock dataset.
"""

# DiffSBDD crossdock dataset info
DIFFSBDD_DATASET_INFO = {
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
    'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
    'max_n_nodes': 50,
    'n_nodes': 50,
}


def get_crossdock_info():
    """Get DiffSBDD crossdock dataset info."""
    return DIFFSBDD_DATASET_INFO.copy()
