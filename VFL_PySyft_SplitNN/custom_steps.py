import torch
from models import HCModel, CBModel, SplitNN
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
hc_dim = 98
cb_feat_dim = 4
cb_dim = 6

lr = 0.001
noise_multiplier = 1.1
max_grad_norm = 1.0

# Training globals
epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Iniatialize Home Credit Model and Optimizer
hc_model = HCModel(hc_dim, cb_feat_dim)
hc_opt = torch.optim.Adam(hc_model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Iniatialize Credit Bureau Model and Optmizer
cb_model = CBModel(cb_dim)
cb_opt = torch.optim.Adam(cb_model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Define Split Neural Network
splitNN = SplitNN(hc_model, cb_model, hc_opt, cb_opt)
criterion = torch.nn.BCELoss()

hc_model.to(device)
cb_model.to(device)

def train_step(dataloader, splitNN):
    """
    Train Step for Split Neural Network

    :param dataloader Train Dataloader
    :param splitNN Split Neural Network that contains Home Credit Model and Credit Bureau Model

    :return: Train Loss
    """
    running_loss = 0
    for (hc_data, labels, id1), (cb_data, id2) in dataloader:
         # Send data and labels to machine model is on
        hc_data = hc_data.to(device)
        labels = labels.to(device)
        # id1 = id1.to(device)
        cb_data = cb_data.to(device)
        # id2 = id2.to(device)
        labels = labels.float()

        for param in splitNN.cb_model.parameters():
            param.accumulated_grads = []

        for param in splitNN.hc_model.parameters():
            param.accumulated_grads = []

        for sample in zip(hc_data, labels, cb_data):
            pred = splitNN.forward(sample[0].unsqueeze(0), sample[2].unsqueeze(0)).squeeze()
            loss = criterion(pred, sample[1])
            loss.backward()
            running_loss += loss.data


            # Clip each parameter's per-sample gradient
            for param in splitNN.hc_model.parameters():
                per_sample_grad = param.grad.detach().clone()
                clip_grad_norm_(per_sample_grad, max_norm=max_grad_norm)  # in-place
                param.accumulated_grads.append(per_sample_grad)
            for param in splitNN.cb_model.parameters():
                per_sample_grad = param.grad.detach().clone()
                clip_grad_norm_(per_sample_grad, max_norm=max_grad_norm)  # in-place
                param.accumulated_grads.append(per_sample_grad)

        # Aggregate back
        for param in splitNN.hc_model.parameters():
            param.grad = torch.stack(param.accumulated_grads, dim=0)

        # Now we are ready to update and add noise!
        for param in splitNN.hc_model.parameters():
            param = param - lr * param.grad
            param += torch.normal(mean=0, std=noise_multiplier * max_grad_norm)
        param.grad = 0

        for param in splitNN.cb_model.parameters():
            param.grad = torch.stack(param.accumulated_grads, dim=0)

        for param in splitNN.cb_model.parameters():
            param = param - lr * param.grad
            param += torch.normal(mean=0, std=noise_multiplier * max_grad_norm)
        param.grad = 0

        # hc_data = hc_data.send(hc_model.location)
        # labels = labels.send(hc_model.location)
        # cb_data = cb_data.send(cb_model.location)

        # # Zero our grads
        # splitNN.zero_grads()
        #
        # # Make a prediction
        # pred = splitNN.forward(hc_data, cb_data).squeeze()
        #
        # # Figure out how much we missed by
        # loss = criterion(pred, labels)
        #
        # # Backprop the loss on the end layer
        # loss.backward()
        # # splitNN.backward()
        #
        # # Change the weights
        # splitNN.step()
        #
        # # Accumulate Loss
        # # running_loss += loss.get()
        # running_loss += loss.data

    return running_loss

def val_step(val_dataloader, splitNN):
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
        # Send data and labels to machine model is on
        hc_data_val = hc_data_val.to(device)
        labels_val = labels_val.to(device)
        cb_data_val = cb_data_val.to(device)
        labels_val = labels_val.float()
        # hc_data_val = hc_data_val.send(hc_model.location)
        # labels_val = labels_val.send(hc_model.location)
        # cb_data_val = cb_data_val.send(cb_model.location)

        # Make a prediction
        with torch.no_grad():
            pred = splitNN.forward(hc_data_val, cb_data_val).squeeze()

        # Calcualte Loss
        criterion = torch.nn.BCELoss()
        loss = criterion(pred, labels_val)

        # Calculate AUC
        thresh_pred = (pred > 0.5).float()
        # thresh_pred = thresh_pred.get().int()
        # labels_val = labels_val.get().int()

        # Fix Me: Undefined for batches with all-same labels...
        # auc = roc_auc_score(labels_val, pred.get().numpy())
        auc = roc_auc_score(labels_val.cpu(), pred.data.cpu().numpy())
        # auc = roc_auc_score(labels_val, pred.data.numpy())

        # Calculate Accuracy Components
        num_exs = hc_data_val.shape[0]
        num_correct = torch.sum(thresh_pred == labels_val).item()

        # Accumulate loss, accuracy and auc
        exs += num_exs
        correct += num_correct
        # running_loss += loss.get()
        running_loss += loss.data
        aucs.append(auc)

    auc = np.mean(np.array(aucs))
    accuracy = correct / exs

    return auc, accuracy, running_loss