from uuid import uuid4
import numpy as np
import pandas as pd
from Dataset import VerticalDataset
from DataLoader import VerticalDataLoader


def get_data_loaders():
    # Load Home Credit Data
    HC_DATA_PATH = "home_credit_train.csv"
    hc_df = pd.read_csv(HC_DATA_PATH)
    hc_df.head()

    # Load Credit Bureau Data
    CB_DATA_PATH = "credit_bureau_train.csv"
    cb_df = pd.read_csv(CB_DATA_PATH)
    cb_df.head()

    # Carve out validation set
    assert cb_df.shape[0] == hc_df.shape[0]
    val_size = 20000
    val_ind = np.random.choice(range(cb_df.shape[0]), val_size, replace=False)
    hc_df_val = hc_df.iloc[val_ind]
    cb_df_val = cb_df.iloc[val_ind]
    hc_df.drop(val_ind, axis=0)
    cb_df.drop(val_ind, axis=0)

    # Get UID Column
    uuids = np.array([uuid4() for _ in range(len(hc_df))])
    uuids_val = np.array([uuid4() for _ in range(len(hc_df_val))])

    # Home Credit Dataset

    # Training
    hc_labels = np.array(hc_df.pop("target"))
    hc_data = np.array(hc_df)
    print("train", uuids.shape, hc_data.shape, hc_labels.shape)
    hc_dim = hc_data.shape[1]
    hc_dataset = VerticalDataset(ids=uuids, data=hc_data, labels=hc_labels)

    # Validation
    hc_labels_val = np.array(hc_df_val.pop("target"))
    hc_data_val = np.array(hc_df_val)
    print("val", uuids_val.shape, hc_data_val.shape, hc_labels_val.shape)
    hc_dataset_val = VerticalDataset(ids=uuids_val, data=hc_data_val, labels=hc_labels_val)

    # Credit Bureau Dataset

    # Training
    cb_data = np.array(cb_df)
    print(cb_data.shape)
    cb_dim = cb_data.shape[1]
    cb_feat_dim = 4
    cb_dataset = VerticalDataset(ids=uuids, data=cb_data, labels=None)

    # Validation
    cb_dataset_val = np.array(cb_df_val)
    cb_dataset_val = VerticalDataset(ids=uuids_val, data=cb_dataset_val, labels=None)

    ## Initialize Train Dataloader
    dataloader = VerticalDataLoader(hc_dataset, cb_dataset, batch_size=2048)

    dataloader.sort_by_ids()

    ## Initialize Train Dataloader
    val_dataloader = VerticalDataLoader(hc_dataset_val, cb_dataset_val, batch_size=2048)

    val_dataloader.sort_by_ids()
    return dataloader, val_dataloader



