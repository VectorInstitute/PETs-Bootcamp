import numpy as np
from torch.utils.data import Dataset

class VerticalDataset(Dataset):
    """Dataset for Vertical Federated Learning"""

    def __init__(self, ids, data, labels=None):
        """
        Args:
            ids (Numpy Array) : Numpy Array with UUIDS
            data (Numpy Array) : Numpy Array with Features
            targets (Numpy Array) : Numpy Array with Labels. None if not available.
        """
        self.ids = ids
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Return record single record"""
        feature = self.data[index].astype(np.float32)

        if self.labels is None:
            label = None
        else:
            label = int(self.labels[index]) if self.labels is not None else None

        id = self.ids[index]

        # Return a tuple of non-None elements
        return (*filter(lambda x: x is not None, (feature, label, id)),)

    def get_ids(self):
        """Return a list of the ids of this dataset."""
        return [str(id_) for id_ in self.ids]

    def sort_by_ids(self):
        """
        Sort the dataset by IDs in ascending order
        """
        ids = self.get_ids()
        sorted_idxs = np.argsort(ids)

        self.data = self.data[sorted_idxs]

        if self.labels is not None:
            self.labels = self.labels[sorted_idxs]

        self.ids = self.ids[sorted_idxs]