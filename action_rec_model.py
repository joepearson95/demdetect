import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import math

# GPU if applicable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
         
        out = self.fc(out)
        return out


class DemDataset(Dataset):
    def __init__(self, csv_file):
        xy = np.loadtxt(csv_file, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, :-4])
        self.y = torch.from_numpy(xy[:, -4:])
        self.no_samples = xy.shape[0]
    def __len__(self):
        return self.no_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
# Get the dataset ready
csv_file = '#'
demdataset = DemDataset(csv_file)

# Get the data split
train_size = int(0.8 * len(demdataset))
test_size = len(demdataset) - train_size
trainset, testset = random_split(demdataset, [train_size, test_size])

# Hyperparams
batch_size = 4
no_workers = 2
no_epochs = 2
total_num_samp = len(demdataset)
no_iters = math.ceil(total_num_samp / batch_size)

# Create the dataloader
training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=no_workers)
# testing_loader = DataLoader(testset, batch_size=100, shuffle=True)

for epochs in range(no_epochs):
    for i, (inputs, labels) in enumerate(training_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = action_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs == labels).float().sum()
    print(100 * correct / len(trainset))
    if epoch%25  == 1:
        print(f'epoch: {i:3} loss: {loss.item():10.8f}')