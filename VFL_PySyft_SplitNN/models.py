import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

noise_multiplier = 1.1
max_grad_norm = 1.0

class HCModel(torch.nn.Module):
    """
    Model for Credit Bureau

    Attributes
    ----------
    cb_dim:
        Dimensionality of Credit Bureau Data
    Methods
    -------
    forward(x):
        Performs a forward pass through the Credit Bureau Model
    """

    def __init__(self, hc_dim, cb_dim):
        super(HCModel, self).__init__()
        self.fused_input_dim = hc_dim + cb_dim
        self.layers = nn.Sequential(
            nn.Linear(self.fused_input_dim, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, hc_feat, cb_feat):
        feat = torch.cat([hc_feat, cb_feat], dim=1)
        pred = self.layers(feat)
        return pred

class CBModel(torch.nn.Module):
    """
    Model for Credit Bureau

    Attributes
    ----------
    cb_dim:
        Dimensionality of Credit Bureau Data
    Methods
    -------
    forward(x):
        Performs a forward pass through the Credit Bureau Model
    """

    def __init__(self, cb_dim):
        super(CBModel, self).__init__()
        self.cb_dim = cb_dim
        self.layers = torch.nn.Sequential(
            nn.Linear(self.cb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
        )

    def forward(self, cb_feat):
        pred = self.layers(cb_feat)
        return pred

class SplitNN:
    """
    A class representing SplitNN

    Attributes
    ----------
    hc_model:
        Home Credit Neural Network Module

    cb_model:
        Credit Bureau Neural Network Module

    hc_opt:
        Optimizer for the Home Credit Neural Network Module

    cb_model:
        Optimizer for the Credit Bureau Neural Network Module

    data:
        A list storing intermediate computations at each index

    remote_tensors:
        A list storing intermediate computations at each index (Computation from each model detached from global computation graph)

    Methods
    -------
    forward(x):
        Performs a forward pass through the SplitNN

    backward():
        Performs a backward pass through the SplitNN

    zero_grads():
        Zeros the gradients of all networks in SplitNN

    step():
        Updates the parameters of all networks in SplitNN
    """

    def __init__(self, hc_model, cb_model, hc_opt, cb_opt, device='cpu'):
        self.hc_model = hc_model.to(device)
        self.cb_model = cb_model.to(device)
        self.hc_opt = hc_opt
        self.cb_opt = cb_opt
        self.data = []
        self.remote_tensors = []
        self.device = device

    def forward(self, hc_x, cb_x):
        """
        Parameters
        ----------
        x:
            Input Sample
        """

        data = []
        # remote_tensors = []

        cb_output = self.cb_model(cb_x)

        # Forward pass through first model
        data.append(self.cb_model(cb_x))

        # if location of data is the same as location of the subsequent model
        # if data[-1].location == self.hc_model.location:
        #     # store computation in remote tensor array
        #     # Gradients will be only computed backward upto the point of detachment
        #     remote_tensors.append(data[-1].detach().requires_grad_())
        # else:
        #     # else move data to location of subsequent model and store computation in remote tensor array
        #     # Gradients will be only computed backward upto the point of detachment
        #     remote_tensors.append(
        #         data[-1].detach().move(self.hc_model.location).requires_grad_()
        #     )

        data.append(self.hc_model(hc_x, cb_output))

        # Get and return final output of model
        # data.append(self.hc_model(hc_x, remote_tensors[-1]))
        #
        self.data = data
        # self.remote_tensors = remote_tensors
        return data[-1]

    # def backward(self):
    #     # if location of data is the same as detatched data
    #     # if self.remote_tensors[0].location == self.data[0].location:
    #     #     # Store gradients from remote_tensor
    #     #     grads = self.remote_tensors[0].grad.copy()
    #     # else:
    #     #     # Move gradients to lovation of Store grad
    #     #     grads = self.remote_tensors[0].grad.copy().move(self.data[0].location)
    #     grads = self.data[0].grad.copy()
    #     self.data[0].backward(grads)

    def zero_grads(self):
        """
        Parameters
        ----------
        """
        self.cb_opt.zero_grad()
        self.hc_opt.zero_grad()

    def step(self):
        """
        Parameters
        ----------
        """
        # update params for hc_model
        self.hc_opt.step()
        # Adding noise
        state_hc = self.hc_model.state_dict()
        for name, param in self.hc_model.named_parameters():
            # state_hc[name] = state_hc[name] - self.hc_opt.param_groups[0]['lr'] * param.grad
            state_hc[name] += torch.normal(0, 0.001, size=param.shape).to(self.device)
        self.hc_model.load_state_dict(state_hc)

        # update params for cb_model
        self.cb_opt.step()
        state_cb = self.cb_model.state_dict()
        for name, param in self.cb_model.named_parameters():
            # state_cb[name] = state_cb[name] - self.cb_opt.param_groups[0]['lr'] * param.grad
            state_cb[name] += torch.normal(0, 0.001, size=param.shape).to(self.device)
        self.cb_model.load_state_dict(state_cb)
