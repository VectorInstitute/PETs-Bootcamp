import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import numpy as np


class SplitNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

        self.data = []
        self.remote_tensors = []

    def forward(self, x):
        data = []
        remote_tensors = []

        #forward pass through first model
        data.append(self.models[0](x[0]))

        #if location of data is the same as location of the subsequent model
        if data[-1].location == self.models[1].location:
            #store computation in remote tensor array
            #gradient will be only computed backward upto the point of detachment
            remote_tensors.append(data[-1].detach().requires_grad_())
        else:
            # else move data to location of subsequent model and store computation in remote tensor array
            # Gradients will be only computed backward upto the point of detachment
            remote_tensors.append(
                data[-1].detach().move(self.models[1].location).requires_grad_()
            )

        i = 1
        while i < (len(self.models) - 1):
#             print (i)
#             print (x[i].shape)
#             print (remote_tensors[-1].shape)
            feat = torch.cat([x[i], remote_tensors[-1]], dim=1)
            data.append(self.models[i](feat))

            if data[-1].location == self.models[i + 1].location:
                remote_tensors.append(data[-1].detach().requires_grad_())
            else:
                remote_tensors.append(
                    data[-1].detach().move(self.models[i + 1].location).requires_grad_()
                )

            i += 1
            
        # Get and return final output of model
        feat = torch.cat([x[i], remote_tensors[-1]], dim=1)
        data.append(self.models[i](feat))

        self.data = data
        self.remote_tensors = remote_tensors

        return data[-1]

    def backward(self):
        for i in range(len(self.models) - 2, -1, -1):
            # if location of data is the same as detatched data
            if self.remote_tensors[i].location == self.data[i].location:
                grads = self.remote_tensors[i].grad.copy()
            else:
                #Move gradients to location of stored grad
                grads = self.remote_tensors[i].grad.copy().move(self.data[i].location)
    
            self.data[i].backward(grads)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()
            

def train_step(dataloader, splitNN):
    running_loss = 0

    for item in dataloader:
        x=[]
        for i in range(len(item)):
            if i==0:
                target = item[i][1].float()
                target = target.send(splitNN.models[-1].location)
            x.append(item[i][0].send(splitNN.models[i].location))

        #1) Zero our grads
        splitNN.zero_grads()

        #2) Make a prediction
        pred = splitNN.forward(x).squeeze()

        #3) Figure out how much we missed by
        criterion = nn.BCELoss()
#         print ('pred', pred)
#         print ('target', target)
        loss = criterion(pred, target)

        #4) Backprop the loss on the end layer
        loss.backward()

        #5) Feed Gradients backward through the nework
        splitNN.backward()

        #6) Change the weights
        splitNN.step()
    
        # Accumulate Loss
        running_loss += loss.get()
        
    return running_loss


def val_step(dataloader, splitNN):
    running_loss = 0
    exs = 0
    correct = 0
    aucs = []
    all_preds = []
    all_labels = []

    for item in dataloader:
        x=[]
        for i in range(len(item)):
            if i==0:
                target = item[i][1].float()
                target = target.send(splitNN.models[-1].location)
            x.append(item[i][0].send(splitNN.models[i].location))

        # Make a prediction
        with torch.no_grad():
            pred = splitNN.forward(x).squeeze()

        # Calcualte Loss
        criterion = torch.nn.BCELoss()
        loss = criterion(pred, target)
        

        # Calculate AUC
        #print (pred[:10])
        thresh_pred = (pred > 0.5).float()
        thresh_pred = thresh_pred.get().int()
        target = target.get().int()
        pred = pred.get().numpy()
        all_preds.extend(pred)
        all_labels.extend(target.numpy())

        # Fix Me: Undefined for batches with all-same labels...
        auc = roc_auc_score(target, pred)

        # Calculate Accuracy Components
        num_exs = x[0].shape[0]
        num_correct = torch.sum(thresh_pred == target).item()

        # Accumulate loss, accuracy and auc
        exs += num_exs
        correct += num_correct
        running_loss += loss.get()
        aucs.append(auc)

    auc = np.mean(np.array(aucs))
    accuracy = correct / exs

    return auc, accuracy, running_loss, all_preds, all_labels


def model_complexity(input_sizes, partition):
    
    model = {}
    
    output_size = 0
    model['naive'] = []
    for i in range(partition):        
        naive =  nn.Sequential()
        naive.add_module(f'linear{i}', nn.Linear(input_sizes[i] + output_size, int(input_sizes[i]/8.0) if i!=(partition-1) else 1)),
        naive.add_module(f'sigmoid{i}',nn.Sigmoid())
        output_size = int(input_sizes[i]/8.0)
        model['naive'].append(naive)
        
    output_size = 0
    model['medium'] = []
    for i in range(partition):
        med =  nn.Sequential()
        med.add_module(f'linear{i}_0', nn.Linear(input_sizes[i] + output_size, int(input_sizes[i]/8.0))),
        med.add_module(f'ReLu{i}', nn.ReLU())
        med.add_module(f'linear{i}_1', nn.Linear(int(input_sizes[i]/8.0), int(input_sizes[i]/4.0) if i!=(partition-1) else 1)),
        med.add_module(f'sigmoid{i}',nn.Sigmoid())
        output_size = int(input_sizes[i]/4.0)
        model['medium'].append(med)
        
    output_size= 0
    model['complex'] = []
    for i in range(partition):        
        cpl =  nn.Sequential()
        cpl.add_module(f'linear{i}_0', nn.Linear(input_sizes[i] + output_size, int(input_sizes[i]/8.0))),
        cpl.add_module(f'ReLu{i}_0', nn.ReLU())
        cpl.add_module(f'linear{i}_1', nn.Linear(int(input_sizes[i]/8.0), int(input_sizes[i]/4.0))),
        cpl.add_module(f'ReLu{i}_1', nn.ReLU())
        cpl.add_module(f'linear{i}_2', nn.Linear(int(input_sizes[i]/4.0), int(input_sizes[i]/2.0) if i!=(partition-1) else 1)),
        cpl.add_module(f'sigmoid{i}',nn.Sigmoid())
        output_size = int(input_sizes[i]/2.0)
        model['complex'].append(cpl)
        
    return model

        
#     cpl = {'naive': nn.Sequential(
#                         ),
#            'moderate': nn.Sequential(),
#            'complex': nn.Sequential()}
