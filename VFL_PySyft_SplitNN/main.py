#!/pkgs/jupyterhub/bin/python

import pandas as pd
import numpy as np
import src
from uuid import UUID, uuid4


def main():
    # Load Raw Data
    datadir = "/ssd003/projects/pets/datasets/home_credit"
    hc_data = pd.read_csv(datadir + "/home_credit_train.csv")
    cb_data = pd.read_csv(datadir + "/credit_bureau_train.csv")

    # Get Train Test Split
    train, test = split_test([hc_data, cb_data], 0.4)
    hc_train, hc_test = train[0], test[0]
    cb_train, cb_test = train[1], test[1]

    # Get Joint Ids
    uuids = np.array([uuid4() for _ in range(len(hc_train))])
    uuids_test = np.array([uuid4() for _ in range(len(hc_test))])

    # Organize Datasets 
    hc_train_ds = src.VerticalDataset(ids=uuids, data=hc_train, label="target")
    hc_test_ds = src.VerticalDataset(ids=uuids_test, data=hc_test, label="target")
    cb_train_ds = src.VerticalDataset(ids=uuids, data=cb_train)
    cb_test_ds = src.VerticalDataset(ids=uuids, data=cb_test)

    # Initialize DataLoaders
    print("\nInitializing dataloader for \n\thc_train_ds: {}\n\tcb_train_ds: {}".format(
        hc_train_ds.data.shape, cb_train_ds.data.shape
    ))
    train_loader = src.VerticalDataLoader(hc_train_ds, cb_train_ds, batch_size=2048)
    train_loader.clipNsort()

    print("Initializing dataloader for \n\thc_test_ds: {}\n\tcb_test_ds: {}".format(
        hc_test_ds.data.shape, cb_test_ds.data.shape
    ))
    test_loader = src.VerticalDataLoader(hc_test_ds, cb_test_ds, batch_size=2048)
    test_loader.clipNsort()


def split_labels(df, label_column):
    """
    Takes a dataframe and splits off the labels.
    """
    labels = np.array(df.pop(label_column))
    features = np.array(df)
    return features, labels


def split_test(datasets, test_frac=0.4):
    """
    Creates a test set with test_frac entries. 
    Returns (1) train and (2) test sets
    
    Args:
        test_frac: 0 < float < 1       - Fraction of data to be used in test set
        datasets : list[np array-like] - List of datasets
    """
    assert np.var([i.shape[0] for i in datasets]) == 0
        
    test_size = int(datasets[0].shape[0])
    test_idx = np.random.choice(range(datasets[0].shape[0]), test_size, replace=False)

    test_sets = [datasets[0].iloc[test_idx], datasets[1].iloc[test_idx]]
    datasets[0].drop(test_idx, axis=0)
    datasets[1].drop(test_idx, axis=0)
    train_sets = [datasets[0], datasets[1]]

    return train_sets, test_sets


if __name__ == "__main__":
    main()
