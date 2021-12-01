from torch.utils.data import DataLoader, Dataset
from .psi.util import Client, Server
from .utils.data_utils import id_collate_fn
import numpy as np


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
        

class SinglePartitionDataLoader(DataLoader):
    """DataLoader for a single vertically-partitioned dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = id_collate_fn
        
        
class VerticalDataLoader:
    """Dataloader which batches data from a complete
    set of vertically-partitioned datasets
    i.e. the images dataset AND the labels dataset
    """

    def __init__(self, data, *args, **kwargs):
        
        #self.data = data
        for key in data:
            exec(f'self.dataloader{key} =  SinglePartitionDataLoader(data[{key}], *args, **kwargs)')

    def __iter__(self):
        """
        Zip Dataloaders
        """
        lst = [dataloader for key, dataloader in self.__dict__.items()]
        return zip(*lst)

    
    def __len__(self): #not sure about purpose
        """
        Return length of dataset
        """
        return (len(self.dataloader1) + len(self.dataloader2)) // 2

    def drop_non_intersecting(self, intersection):
        """Remove elements and ids in the datasets that are not in the intersection."""
        
        for key, dataloader in self.__dict__.items():
            dataloader.dataset.data = dataloader.dataset.data[intersection]
            dataloader.dataset.ids = dataloader.dataset.ids[intersection]
            try:
                dataloader.dataset.labels = dataloader.dataset.labels[intersection]
            except:
                print (f'dataloader{key} does not have labels.')
            dataloader.dataset.ids = dataloader.dataset.ids[intersection]

    def sort_by_ids(self) -> None:
        """
        Sort each dataset by ids
        """
        for key, dataloader in self.__dict__.items(): 
            dataloader.dataset.sort_by_ids()
