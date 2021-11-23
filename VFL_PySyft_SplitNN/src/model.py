import torch
from torch import nn


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
    _architecture = [
        nn.Sequential(
            nn.Linear(self.fused_input_dim, 1),
            nn.Sigmoid()
        ),
        nn.Sequential(
            nn.Linear(self.fused_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ),
        nn.Sequential(
            nn.Linear(self.fused_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ),
        nn.Sequential(
            nn.Linear(self.fused_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ),
        nn.Sequential(
            nn.Linear(self.fused_input_dim, self.fused_input_dim),
            nn.ReLU(),
            nn.Linear(self.fused_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    ]

    def __init__(self, hc_dim, cb_dim, complexity): 
        super(HCModel, self).__init__()
        self.fused_input_dim = hc_dim + cb_dim
        self.layers = self._architecture[complexity]
    
    def forward(self, hc_feat, cb_feat):
        feat = torch.cat([hc_feat, cb_feat], dim=1)
        pred = self.layers(feat)
        return pred


class CBModel(torch.nb.Model):
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
    _complexity = [
        self.layers = torch.nn.Sequential(
            nn.Linear(self.cb_dim, 14),
            nn.Sigmoid()
        ),
        self.layers = torch.nn.Sequential(
            nn.Linear(self.cb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid()
        ),
        self.layers = torch.nn.Sequential(
            nn.Linear(self.cb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid()
        ),
        self.layers = torch.nn.Sequential(
            nn.Linear(self.cb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid()
        ),
        self.layers = torch.nn.Sequential(
            nn.Linear(self.cb_dim, self.cb_dim),
            nn.ReLU(),
            nn.Linear(self.cb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid()
        )
    ]

    def __init__(self, cb_dim):
        super(CBModel, self).__init__()
        self.cb_dim = cb_dim
        self.layers = torch.nn.Sequential(
            nn.Linear(self.cb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid()
        )

    def forward(self, cb_feat):
        pred = self.layers(cb_feat)
        return pred
