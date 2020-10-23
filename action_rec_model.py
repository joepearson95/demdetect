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
        self.df = pd.read_csv(csv_file)
        del self.df['file_name']
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        self.x = torch.Tensor(self.df.iloc[idx,:-4].astype('float64'))
        self.y = torch.Tensor(self.df.iloc[idx, -4:].astype('float64'))

        return self.x,self.y
        
# Get the dataset ready
csv_file = '#'
demdataset = DemDataset(csv_file)

# Get the data split
train_size = int(0.8 * len(demdataset))
test_size = len(demdataset) - train_size
trainset, testset = random_split(demdataset, [train_size, test_size])

# Hyper Param
input_size = 51
sequence_length = 1
batch_size = 4
hidden_size = 200
num_layers = 2
num_classes = 4
no_epochs = 4
lr_rate = 0.0001

# # Create the dataloader
training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
demdetect = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(demdetect.parameters(), lr=lr_rate)
demdetect.train()
for epochs in range(no_epochs):
    ep_loss = 0.0
    run_loss = 0.0
    for i, (points, labels) in enumerate(training_loader):
        points = points.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = demdetect(points)
        loss = criterion(outputs, torch.max(labels,1)[1])
        loss.backward()
        optimizer.step()
        ep_loss += outputs.shape[0]*loss.item()
        run_loss += loss.item()
        if (i+1) % 100 == 0:
            print('[Epoch %d of %d] Running Loss: %.3f' % (epochs+1, no_epochs, run_loss/100))
            run_loss = 0.0
    print('[Epoch %d of %d] Epoch Loss: %.3f' % (epochs+1, no_epochs, ep_loss/len(trainset)))
print('Finished Training.')

correct = 0
total = 0
demdetect.eval()
with torch.no_grad():
    for i, (points, labels) in enumerate(testing_loader):
        points = points.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = demdetect(points)
        _, predicted = outputs.max(1)
        correct +=  (predicted == labels).sum()
        total += predicted.size(0)
    print(f'Got {correct}/{total} with accuracy {float(correct)/float(total)*100:.2f}')