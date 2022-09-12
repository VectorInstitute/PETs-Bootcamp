import torch
from custom_steps import train_step, val_step
from prepare_dataloader import get_data_loaders
from models import HCModel, CBModel, SplitNN

# Set up dataloaders
dataloader, val_dataloader = get_data_loaders()

# Set up sub models
hc_dim = 98
cb_feat_dim = 4
cb_dim = 6

# training part
metric_names = ["Train Loss", "Validation Loss", "Accuracy", "AUC"]
metrics = {metric: [] for metric in metric_names}

# Training globals
epochs = 10

# Determine device to use
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built():  # For M1 mac
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Iniatialize Home Credit Model and Optimizer
hc_model = HCModel(hc_dim, cb_feat_dim)
hc_opt = torch.optim.Adam(hc_model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Iniatialize Credit Bureau Model and Optmizer
cb_model = CBModel(cb_dim)
cb_opt = torch.optim.Adam(cb_model.parameters(), lr=0.001, betas=(0.9, 0.999))

hc_model.to(device)
cb_model.to(device)

# Define Split Neural Network
splitNN = SplitNN(hc_model, cb_model, hc_opt, cb_opt, device=device)
criterion = torch.nn.BCELoss().to(device)

# Train Loop
for i in range(epochs):

    # Train Step

    train_loss = train_step(dataloader, splitNN, criterion, device)

    # Val Step
    auc, accuracy, val_loss = val_step(val_dataloader, splitNN, criterion, device)

    # Log metrics
    print(f"Epoch: {i} \t Accuracy: {accuracy} \tAUC: {auc}")
    metrics["Train Loss"].append(train_loss.item())
    metrics["Validation Loss"].append(val_loss.item())
    metrics["Accuracy"].append(accuracy)
    metrics["AUC"].append(auc)