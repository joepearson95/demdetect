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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
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

input_size = 52
sequence_length = 1
batch_size = 4
hidden_size = 200
num_layers = 2
num_classes = 4
no_epochs = 6
lr_rate = 0.0001

# Create the dataloader
training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(testset, batch_size=100, shuffle=True)
demdetect = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
# batch of 4
#torch.Size([4, 51]) torch.Size([4, 4])
# inputs, classes = next(iter(training_loader))  
# inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
# print(inputs.shape)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(demdetect.parameters(), lr=lr_rate)

total_steps = len(training_loader)
for epochs in range(no_epochs):
    correct = 0
    for i, (points, labels) in enumerate(training_loader):
        points = points.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = demdetect(points)
        loss = criterion(outputs, torch.max(labels,1)[1])
        loss.backward()
        optimizer.step()    
        if (i+1) % 100 == 0:
            outputs = (outputs>0.5).float()
            correct = (outputs == labels).float().sum()
            print(f'Epoch [{epochs+1}/{no_epochs}], step [{i+1}/{total_steps}], Loss: {loss.item():.4f}, a: {correct/outputs.shape[0]:.3f}')