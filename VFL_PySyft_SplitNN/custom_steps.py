import torch
import numpy as np
from sklearn.metrics import roc_auc_score
# from torch.nn.utils import clip_grad_norm_


def train_step(dataloader, splitNN, criterion, device):
    """
    Train Step for Split Neural Network

    :param dataloader Train Dataloader
    :param splitNN Split Neural Network that contains Home Credit Model and Credit Bureau Model

    :return: Train Loss
    """
    running_loss = 0
    for (hc_data, labels, id1), (cb_data, id2) in dataloader:
        hc_data = hc_data.to(device)
        labels = labels.to(device)
        cb_data = cb_data.to(device)

        splitNN.zero_grads()

        pred = splitNN.forward(hc_data, cb_data).squeeze()
        loss = criterion(pred, labels.float().to(device))
        loss.backward()
        running_loss += loss.data
        splitNN.step()
    return running_loss


def val_step(val_dataloader, splitNN, criterion, device):
    """
    Val Step for Split Neural Network

    :param dataloader Val Dataloader
    :param splitNN Split Neural Network that contains Home Credit Model and Credit Bureau Model

    :return: (auc, accuracy, running_loss)
    """
    running_loss = 0
    exs = 0
    correct = 0
    aucs = []
    for (hc_data_val, labels_val, id1), (cb_data_val, id2) in val_dataloader:
        hc_data_val = hc_data_val.to(device)
        labels_val = labels_val.to(device)
        cb_data_val = cb_data_val.to(device)
        labels_val = labels_val.float()

        # Make a prediction
        with torch.no_grad():
            pred = splitNN.forward(hc_data_val, cb_data_val).squeeze()

        # Calcualte Loss
        loss = criterion(pred, labels_val)

        # Calculate AUC
        auc = roc_auc_score(labels_val.cpu(), pred.data.cpu().numpy())

        # Calculate Accuracy Components
        thresh_pred = (pred > 0.5).float()
        num_exs = hc_data_val.shape[0]
        num_correct = torch.sum(thresh_pred == labels_val).item()

        # Accumulate loss, accuracy and auc
        exs += num_exs
        correct += num_correct
        running_loss += loss.data
        aucs.append(auc)

    auc = np.mean(np.array(aucs))
    accuracy = correct / exs

    return auc, accuracy, running_loss