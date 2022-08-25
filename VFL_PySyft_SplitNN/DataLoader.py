from torch.utils.data import DataLoader
from src.utils.data_utils import id_collate_fn

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

    def __init__(self, hc_data, cb_data, *args, **kwargs):

        self.dataloader1 = SinglePartitionDataLoader(hc_data, *args, **kwargs)
        self.dataloader2 = SinglePartitionDataLoader(cb_data, *args, **kwargs)

    def __iter__(self):
        """
        Zip Dataloaders
        """
        return zip(self.dataloader1, self.dataloader2)

    def __len__(self):
        """
        Return length of dataset
        """
        return (len(self.dataloader1) + len(self.dataloader2)) // 2

    def drop_non_intersecting(self, intersection):
        """Remove elements and ids in the datasets that are not in the intersection."""
        self.dataloader1.dataset.data = self.dataloader1.dataset.data[intersection]
        self.dataloader1.dataset.ids = self.dataloader1.dataset.ids[intersection]

        self.dataloader1.dataset.labels = self.dataloader1.dataset.labels[intersection]
        self.dataloader2.dataset.ids = self.dataloader2.dataset.ids[intersection]

    def sort_by_ids(self) -> None:
        """
        Sort each dataset by ids
        """
        self.dataloader1.dataset.sort_by_ids()
        self.dataloader2.dataset.sort_by_ids()