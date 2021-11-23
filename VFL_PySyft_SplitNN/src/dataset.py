from torch.utils.data import Dataset as torch_Dataset
import numpy as np


class VerticalDataset(torch_Dataset):
    """Dataset for VFL"""

    def __init__(self, ids, data, label=None):
        """
        Args:
            ids      - Numpy array of UUIDS
            data     - Dataframe of all the data
            label    - String or index of column for data
        """

        self.ids = ids
        if label is not None:
            self.labels = data.pop(label)
        else:
            self.labels = None

        self.features = data
        self.data = data

    def __getitem__(self, index):
        """Return single record"""
        feature = self.features[index].astype(np.float32)

        if self.labels is None:
            label = None
        else:
            label = int(self.labels[index]) if self.labels is not None else None

        id = self.ids[index]
        return (*filter(lambda x: x is not None, (feature, label, id)), )

    def get_ids(self):
        """Return a list of the ids of this dataset"""
        return [str(id_) for id_ in self.ids]

    def sort_by_ids(self):
        """"Sort the dataset by IDs in ascending order"""
        ids = self.get_ids()
        sorted_idxs = np.argsort(ids)

        self.features = self.features[sorted_idxs]

        if self.labels is not None:
            self.labels = self.labels[sorted_idxs]

        self.ids = self.ids[sorted_idxs]
