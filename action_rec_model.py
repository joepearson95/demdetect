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
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        del self.df['file_name']
        self.transform = transform
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.x = torch.Tensor(self.df.iloc[idx,:-3].astype('float64'))
        self.y = torch.Tensor(self.df.iloc[idx, -3:].astype('float64'))

        return self.x,self.y
        
# Get the dataset ready
csv_file = '#'

demdataset = DemDataset(csv_file)

from sklearn.model_selection import train_test_split    
trainset, testset = train_test_split(demdataset, test_size=0.2, shuffle=True)

loader = DataLoader(trainset, batch_size=len(trainset), num_workers=1)
data = next(iter(loader))
mean, std = data[0].mean(), data[0].std()
print(mean, std)

# Hyper Param
input_size = 51
sequence_length = 1 
batch_size = 3
hidden_size = 37
num_layers = 1
num_classes = 3
no_epochs = 50
lr_rate = 0.001

# # Create the dataloader
training_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)
testing_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
demdetect = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(demdetect.parameters(), lr=lr_rate)
demdetect.train()

train_correct = []
epoch_loss = []
count_loss = []
num_correct = 0
num_samples = 0
for epochs in range(no_epochs):
    ep_loss = 0.0
    run_loss = 0.0
    for i, (points, labels) in enumerate(training_loader):
        points = points.reshape(-1, sequence_length, input_size).to(device)
        point_norm = transforms.Normalize(mean,std)
        points = point_norm(points)
        labels = labels.to(device)
        
        outputs = demdetect(points)

        loss = criterion(outputs, torch.max(labels,1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = outputs.max(1)
        num_correct += (predictions == labels).sum()
        num_samples += predictions.size(0)

        run_loss += loss.item()
        ep_loss += loss.item()
        count_loss.append(run_loss)
    train_correct.append(float(num_correct)/float(num_samples)*100)
    print(f'[Epoch {epochs+1}/{no_epochs}] Epoch Loss: {ep_loss/len(training_loader):.3f} | Got {num_correct} of {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%') 
print('Finished Training.')
demdetect.eval()

test_correct = []
t_num_correct = 0
t_num_samples = 0
with torch.no_grad():
    test_loss = 0
    accuracy = 0
    accuracy_running = []
    for i, (points, labels) in enumerate(testing_loader):
        points = points.reshape(-1, sequence_length, input_size).to(device)
        point_norm = transforms.Normalize(mean, std)
        points = point_norm(points)
        labels = labels.to(device)
        outputs = demdetect.forward(points)
        _, predictions = outputs.max(1)
        t_num_correct += (predictions == labels).sum()
        t_num_samples += predictions.size(0)
    test_correct.append(float(t_num_correct)/float(t_num_samples)*100)
    print(f'Got {t_num_correct} of {t_num_samples} with accuracy {float(t_num_correct)/float(t_num_samples)*100:.2f}%')
demdetect.train()