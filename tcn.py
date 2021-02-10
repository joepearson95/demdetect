import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os

def train_test_split(data,labels,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(int(len(data)) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    train_data=data[train_indices]
    train_labels=labels[train_indices]
    test_data=data[test_indices]
    test_labels=labels[test_indices]
    return train_data,train_labels,test_data,test_labels

class TCN(nn.Module):
    def __init__(self, sequence_length,batch_size,input_size):
        super(TCN, self).__init__()
        self.first_layer = torch.nn.Conv1d(sequence_length,batch_size,input_size)
        self.second_layer = torch.nn.Conv1d(batch_size, 3, 1)
        # Upsampling
        self.up_samp = torch.nn.ConvTranspose1d(
            in_channels=3,
            out_channels=12,
            kernel_size=1
        )
        self.uplayer1 = torch.nn.Conv1d(12,12,1)

        self.up_samp2 = torch.nn.ConvTranspose1d(
            in_channels=12,
            out_channels=20,
            kernel_size=1
        )

        self.out = torch.nn.Conv1d(20,3,1)

    def forward(self, x):
        encode1 = F.relu(self.first_layer(x))
        pooled = F.max_pool1d(encode1,1)
        encode2 = F.relu(self.second_layer(pooled))
        pooled2 = F.max_pool1d(encode2, 1)

        # decoder
        x = self.up_samp(pooled2)
        x = F.relu(self.uplayer1(x))
        x = self.up_samp2(x)
        x = F.softmax(F.relu(self.out(x)), dim=1)
        return x

data=np.load('normalized_data.npy')
data=data.reshape((data.shape[0],data.shape[2],data.shape[1]))
labels1=np.load('labels.npy')
labels=[]
for i in range(0,len(labels1)):
    if labels1[i][0]==1:
        label=0
    if labels1[i][1]==1:
        label=1
    if labels1[i][2]==1:
        label=2
    labels.append(label)
labels=np.array(labels)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 20
np.random.seed(1)
test_ratio=0.3

train_data,train_labels,test_data,test_labels=train_test_split(data,labels,test_ratio)



train_data=torch.Tensor(train_data)
train_labels=torch.Tensor(train_labels)
torch_train_dataset = Data.TensorDataset(train_data,train_labels)

test_data=torch.Tensor(test_data)
test_labels=torch.Tensor(test_labels)
torch_test_dataset = Data.TensorDataset(test_data,test_labels)


train_loader = Data.DataLoader(
    dataset=torch_train_dataset,     
    batch_size=BATCH_SIZE,      
    shuffle=True,               
    num_workers=1,              
)

test_loader = Data.DataLoader(
    dataset=torch_test_dataset,     
    batch_size=BATCH_SIZE,      
    shuffle=True,               
    num_workers=1,              
)

input_size = 102
sequence_length = 6
batch_size = 20
no_epochs = 100
lr_rate = 0.001

# Create the dataloader
demdetect = TCN(sequence_length,batch_size,input_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(demdetect.parameters(), lr=lr_rate)


t_num_correct = 0
t_num_samples = 0

train_err = []
test_err = []

for epochs in range(no_epochs):
    ep_loss = 0.0
    num_correct = 0
    num_samples = 0
    t_num_correct = 0
    t_num_samples = 0
    demdetect.train()
    for i, (points, labels) in enumerate(train_loader):
        points = points.to(device)
        labels = labels.to(device)
        labels=labels.long()
        outputs = demdetect(points)
        outputs = outputs.squeeze(2)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predictions = outputs.max(1)
        num_correct += (predictions == labels).sum()
        num_samples += predictions.size(0)
        ep_loss += loss.item()
    demdetect.eval()
    for (points, labels) in test_loader:
        points = points.to(device)
        labels = labels.to(device)
        labels=labels.long()
        outputs = demdetect(points)
        outputs = outputs.squeeze(2)

        _, predictions = outputs.max(1)
        t_num_correct += (predictions == labels).sum()
        t_num_samples += predictions.size(0)
    test_acc = float(t_num_correct) / float(t_num_samples) * 100
    train_acc = float(num_correct) / float(num_samples) * 100
    print(f'[Epoch {epochs + 1}/{no_epochs}] Epoch Loss: {ep_loss / len(train_loader):.3f} | Got {num_correct} of {num_samples} with accuracy {train_acc:.2f}%')
    print(f'[Epoch {epochs + 1}/{no_epochs}] | Test: Got {t_num_correct} of {t_num_samples} with accuracy {test_acc:.2f}%')

