import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import normalise
import numpy as np

# GPU if applicable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.first_layer = torch.nn.Conv1d(45,3,1)

        # Upsampling
        self.up_samp = torch.nn.ConvTranspose1d(
            in_channels=3,
            out_channels=3,
            kernel_size=3
        )
        self.uplayer1 = torch.nn.Conv1d(3,3,3)

    def forward(self, x):
        encode1 = F.relu(self.first_layer(x))
        pooled = F.max_pool1d(encode1,1)
        # decoder
        x = self.up_samp(pooled)
        x = F.softmax(F.relu(self.uplayer1(x)), dim=1)
        return x

class DemDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.x = torch.Tensor(self.data.iloc[idx, :-3])
        self.y = torch.Tensor(self.data.iloc[idx, -3:])

        return self.x, self.y


# def hindawi_norm()

# Get the dataset ready
csv_file = '../demcare1_ingest_dataset.csv'
normaliser = normalise.Normalise(csv_file)
normaliserAlg = normaliser.hal()
demdataset = DemDataset(normaliserAlg)
trainset, testset = train_test_split(demdataset, test_size=0.2, shuffle=False)

training_loader = DataLoader(trainset, batch_size=3, shuffle=False)
testing_loader = DataLoader(testset, batch_size=3, shuffle=False)
demdetect = TCN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(demdetect.parameters(), lr=0.001)

no_epochs = 30
num_correct = 0
num_samples = 0
t_num_correct = 0
t_num_samples = 0
train_err = []
test_err = []

# Begin training
for epochs in range(no_epochs):
    running_loss = 0.0
    demdetect.train()
    for i, (points, labels) in enumerate(training_loader):
        points = points.to(device)
        points = points.unsqueeze_(2)
        labels = labels.to(device)
        outputs = demdetect(points)
        labels = torch.max(labels, 1)[1] 
        labels = labels.view(labels.size(0), -1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = outputs.max(1)
        num_correct += (predictions ==  labels).sum()
        num_samples += predictions.size(0)
        running_loss += loss.item()
    train_acc = float(num_correct) / float(num_samples) * 100
    train_err.append((train_acc - 100) * -1)

    # begin training
    demdetect.eval()
    for (points, labels) in testing_loader:
        points = points.to(device)
        points = points.unsqueeze_(2)
        labels = labels.to(device)
        outputs = demdetect(points)

        labels = torch.max(labels, 1)[1]
        labels = labels.view(labels.size(0), -1) 
        _, predictions = outputs.max(1)
        t_num_correct += (predictions == labels).sum()
        t_num_samples += predictions.size(0)
    test_acc = float(t_num_correct) / float(t_num_samples) * 100
    test_err.append((test_acc - 100) * -1)
    print(f'[Epoch {epochs + 1}/{no_epochs}] Epoch Loss: {running_loss / len(training_loader):.3f} | Got {num_correct} of {num_samples} with accuracy {train_acc:.2f}%')
    print(f'[Epoch {epochs + 1}/{no_epochs}] | Test: Got {t_num_correct} of {t_num_samples} with accuracy {test_acc:.2f}%')
print('Finished Training')
