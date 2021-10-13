import torch
import numpy as np
from uuid import UUID
from uuid import uuid4
from torch.utils.data._utils.collate import default_collate

def id_collate_fn(batch):
    """Collate data, targets and IDs  into batches
    This custom function is necessary as default collate
    functions cannot handle UUID objects.
    Args:
        batch (tuple of (data, target, id) tuples) : tuple of data returns from each index call
            to the dataset in a batch. To be turned into batched data
    Returns:
        list : List of batched data objects:
            data (torch.Tensor), targets (torch.Tensor), IDs (tuple of strings)
    """
    results = []

    for samples in zip(*batch):
        if isinstance(samples[0], UUID):
            # Turn into a tuple of strings
            samples = (*map(str, samples),)

        # Batch data
        results.append(default_collate(samples))
    return results
