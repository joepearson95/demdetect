import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import normalise
import numpy as np

# GPU if applicable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hindawi = 1,80,90 -> 80,20 -> 20,3
# hal = 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 80, 90)
        self.drop = torch.nn.Dropout(0.1)
        self.conv2 = torch.nn.Conv1d(80,50,1)
        self.hiddden = torch.nn.Linear(50*1*1, 20)
        self.out = torch.nn.Linear(20,3)
        self.activ1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.activ1(self.conv1(x))
        x = self.drop(x)
        x = self.activ1(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.activ1(self.hiddden(x))
        x = self.out(x)
        
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
demdetect = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(demdetect.parameters(), lr=0.01)

no_epochs = 25
num_correct = 0
num_samples = 0
t_num_correct = 0
t_num_samples = 0

train_err = []
test_err = []
for epochs in range(no_epochs):
    running_loss = 0.0
    demdetect.train()
    for i, (points, labels) in enumerate(training_loader):
        points = points.to(device)
        points = points.unsqueeze_(1)
        labels = labels.to(device)
        outputs = demdetect(points)
        # if labels.shape == (3,3):
        # labels = np.reshape(torch.max(labels,1)[1], (3,1))
        loss = criterion(outputs, torch.max(labels,1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = outputs.max(1)
        num_correct += (predictions == labels).sum()
        num_samples += predictions.size(0)
        running_loss += loss.item()
    train_acc = float(num_correct) / float(num_samples) * 100
    train_err.append((train_acc - 100) * -1)

    demdetect.eval()
    for (points, labels) in testing_loader:
        points = points.to(device)
        points = points.unsqueeze_(1)
        labels = labels.to(device)
        outputs = demdetect(points)
        # if labels.shape == (3,3):
        # labels = np.reshape(torch.max(labels, 1)[1], (3,1))
        _, predictions = outputs.max(1)
        t_num_correct += (predictions == labels).sum()
        t_num_samples += predictions.size(0)
    test_acc = float(t_num_correct) / float(t_num_samples) * 100
    test_err.append((test_acc - 100) * -1)
    print(f'[Epoch {epochs + 1}/{no_epochs}] Epoch Loss: {running_loss / len(training_loader):.3f} | Got {num_correct} of {num_samples} with accuracy {train_acc:.2f}%')
    print(f'[Epoch {epochs + 1}/{no_epochs}] | Test: Got {t_num_correct} of {t_num_samples} with accuracy {test_acc:.2f}%')
print('Finished Training')