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
from numpy import load, save
import os

# GPU if applicable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TCN(nn.Module):
    def __init__(self, seq_len, batch_size):
        super(TCN, self).__init__()
        self.first_layer = torch.nn.Conv1d(seq_len*45,batch_size,1)

        # Upsampling
        self.up_samp = torch.nn.ConvTranspose1d(
            in_channels=batch_size,
            out_channels=1,
            kernel_size=1
        )
        self.uplayer1 = torch.nn.Conv1d(1,3,1)

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
        self.x = self.data[idx]
        # return self.x
        frames = []
        labels = []
        for i in self.x:
            frames.append(i[:-3])
            labels.append(i[-3:])
        return np.stack(frames), np.stack(labels)

# Get the dataset ready
csv_file = '../dataset-edited.csv'#'../demcare1_ingest_dataset.csv'
normaliser = normalise.Normalise(csv_file)
normaliserAlg = normaliser.hal()

# Params
keypoints = 45
window = 3
batch_size = 2
learning_rate = 0.001
no_epochs = 200
num_correct = 0
num_samples = 0
t_num_correct = 0
t_num_samples = 0
train_err = []
test_err = []

# Dynamically create the .npy file
if os.path.exists("data.npy"):
    os.remove("data.npy")
    frames = []
    for i in range(len(normaliserAlg) - window):
        if i == 0:
            frames.append(normaliserAlg.iloc[i:i+window].to_numpy())
        else:
            frames.append(normaliserAlg.iloc[i+1:(i+1)+window].to_numpy())
    save('data.npy', frames)
data = load('data.npy', allow_pickle=True)
print("Data remove before saved and loaded.")

demdataset = DemDataset(data)
trainset, testset = train_test_split(demdataset, test_size=0.2, shuffle=False)
training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
testing_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

demdetect = TCN(seq_len=window, batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(demdetect.parameters(), lr=learning_rate)

# Begin training
for epochs in range(no_epochs):
    running_loss = 0.0
    demdetect.train()
    for i, (frames, labels) in enumerate(training_loader):
        # pass frames/labels to device and gather the max from labels
        frames = frames.float()
        points = frames.to(device)
        if points.shape == (batch_size,window,45):
            # print(points.shape)
            points = points.view(2,-1)
            points = points.unsqueeze_(2)

            labels = labels.to(device)
            
            labels = torch.max(labels, 2)[1]
            labels_max = []
            for j in labels:
                labels_max.append(torch.max(j))
            labels = torch.Tensor(labels_max)
            # print(labels)
            # break
            # labels = labels.unsqueeze_(0)
            
            # print(points.shape)
            # print(labels, labels.shape)
            
            # alter the size due to problems caused otherwise
            # break
            outputs = demdetect(points)
            outputs = outputs.squeeze(2)
            
            # outputs = outputs.view(outputs.size(0), window*45)
            # print(labels, labels.shape)
            # print(outputs, outputs.shape)
            
            loss = criterion(outputs, labels.long())
            # break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predictions = outputs.max(1)        
            num_correct += (predictions ==  labels).sum()
            num_samples += predictions.size(0)
            running_loss += loss.item()
    train_acc = float(num_correct) / float(num_samples) * 100
    train_err.append((train_acc - 100) * -1)
    # begin testing
    demdetect.eval()
    for (points, labels) in testing_loader:
        frames = frames.float()
        points = frames.to(device)

        if points.shape == (batch_size,window,45):
            print(1, points.shape)
            points = points.view(2,-1)
            points = points.unsqueeze_(2)
        else:
            points = points.squeeze(2)
            print(2, points.shape)
            labels = labels.to(device)
            
            labels = torch.max(labels, 2)[1]
            labels_max = []

            for j in labels:
                labels_max.append(torch.max(j))
            labels = torch.Tensor(labels_max)

            outputs = demdetect(points)
            outputs = outputs.squeeze(2)

            _, predictions = outputs.max(1)
            
            t_num_correct += (predictions == labels).sum()
            t_num_samples += predictions.size(0)
    test_acc = float(t_num_correct) / float(t_num_samples) * 100
    test_err.append((test_acc - 100) * -1)
    print(f'[Epoch {epochs + 1}/{no_epochs}] Epoch Loss: {running_loss / len(training_loader):.3f} | Got {num_correct} of {num_samples} with accuracy {train_acc:.2f}%')
    print(f'[Epoch {epochs + 1}/{no_epochs}] | Test: Got {t_num_correct} of {t_num_samples} with accuracy {test_acc:.2f}%')
print('Finished Training')