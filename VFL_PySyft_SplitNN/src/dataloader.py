from torch.utils.data import DataLoader
from .psi.util import Client, Server
from .utils.data_utils import id_collate_fn


class SinglePartitionDataLoader(DataLoader):
    """DataLoader for a single vertically-partitioned dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = id_collate_fn


class VerticalDataLoader:
    """DataLoader which creates batches using complete set
    of vertically partitioned datasets.

    Args:
        hc_data(PyTorch DataSet) - Home Credit dataset
        cb_data(PyTorch DataSet) - Central Bureau dataset
    """

    def __init__(self, hc_data, cb_data, *args, **kwargs):
        self.dataloader1 = SinglePartitionDataLoader(
            hc_data, *args, **kwargs
        )
        self.dataloader2 = SinglePartitionDataLoader(
            cb_data, *args, **kwargs
        )

    def __iter__(self):
        """ Zip Dataloaders """
        return zip(self.dataloader1, self.dataloader2)

    def __len__(self):
        """ Return length of dataset """
        return (len(self.dataloader1) + len(self.dataloader2)) // 2

    def drop_non_intersecting(self, intersection) -> None:
        """Remove elements and ids that are not in both datasets"""
        self.dataloader1.dataset.features = self.dataloader1.dataset.features[intersection]
        self.dataloader1.dataset.ids = self.dataloader1.dataset.ids[intersection]
        self.dataloader1.dataset.labels = self.dataloader1.dataset.labels[intersection]

        self.dataloader2.dataset.ids = self.dataloader2.dataset.ids[intersection]

    def sort_by_ids(self) -> None:
        """
        Sort each dataset by ids
        """
        self.dataloader1.dataset.sort_by_ids()
        self.dataloader2.dataset.sort_by_ids()

    def clipNsort(self) -> None:
        """
        Removes non-intersecting rows between the two dataloaders and sorts them.
        """
        client = Client(self.dataloader1.dataset.get_ids())
        server = Server(self.dataloader2.dataset.get_ids())
         
        setup, response = server.process_request(
            client.request, len(self.dataloader1.dataset.get_ids())
        )
        intersection = client.compute_intersection(setup, response)
        print("INTER", intersection)

        # Order data
        self.drop_non_intersecting(intersection)
        self.sort_by_ids()
