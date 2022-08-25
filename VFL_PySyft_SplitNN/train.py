import torch
from custom_steps import train_step, val_step
from prepare_dataloader import get_data_loaders
from models import HCModel, CBModel, SplitNN
from opacus.privacy_engine import PrivacyEngine
dataloader, val_dataloader = get_data_loaders()

hc_dim = 98
cb_feat_dim = 4
cb_dim = 6

# training part
metric_names = ["Train Loss", "Validation Loss", "Accuracy", "AUC"]
metrics = {metric: [] for metric in metric_names}

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

hc_model.to(device)
cb_model.to(device)
# Define Split Neural Network
splitNN = SplitNN(hc_model, cb_model, hc_opt, cb_opt)
criterion = torch.nn.BCELoss()

# dataloader.to(device)
# val_dataloader.to(device)
# Train Loop
for i in range(epochs):

    # Train Step

    train_loss = train_step(dataloader, splitNN)

    # Train Step
    auc, accuracy, val_loss = val_step(val_dataloader, splitNN)

    # Log metrics
    print(f"Epoch: {i} \t AUC: {auc}")
    metrics["Train Loss"].append(train_loss.item())
    metrics["Validation Loss"].append(val_loss.item())
    metrics["Accuracy"].append(accuracy)
    metrics["AUC"].append(auc)

# model = splitNN
# optimizer = SGD(model.parameters(), lr=0.05)
# # enter PrivacyEngine
# privacy_engine = PrivacyEngine()
# model, optimizer, data_loader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=data_loader,
#     noise_multiplier=1.1,
#     max_grad_norm=1.0,
# )