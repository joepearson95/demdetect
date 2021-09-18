import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

class Model(nn.Module):
    def __init__(self,sequence_length, batch_size, input_size):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(sequence_length, batch_size, input_size)
        self.conv2 = nn.Conv1d(batch_size, 30, 1)
        self.lstm1 = nn.LSTM(1,30, 2, batch_first=True)
        self.fc1 = nn.Linear(30, 10)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


data=np.load('normalized_data_v2.npy')
data=data.reshape((data.shape[0],data.shape[2],data.shape[1]))
labels1=np.load('labels_v2.npy')
labels=[]
for i in range(0,len(labels1)):
    if labels1[i][0]==1:
        label=0
    if labels1[i][1]==1:
        label=1
    if labels1[i][2]==1:
        label=2
    if labels1[i][3]==1:
        label=3
    if labels1[i][4]==1:
        label=4
    if labels1[i][5]==1:
        label=5
    if labels1[i][6]==1:
        label=6
    if labels1[i][7]==1:
        label=7
    if labels1[i][8]==1:
        label=8
    if labels1[i][9]==1:
        label=9
    labels.append(label)
labels=np.array(labels)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 10
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
batch_size = 5
no_epochs = 100
lr_rate = 0.001

# Create the dataloader
demdetect = Model(sequence_length, batch_size, input_size).to(device)
print(demdetect)
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
        # print(labels)
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

        _, predictions = outputs.max(1)
        t_num_correct += (predictions == labels).sum()
        t_num_samples += predictions.size(0)
    test_acc = float(t_num_correct) / float(t_num_samples) * 100
    train_acc = float(num_correct) / float(num_samples) * 100
    print(f'[Epoch {epochs + 1}/{no_epochs}] Epoch Loss: {ep_loss / len(train_loader):.3f} | Got {num_correct} of {num_samples} with accuracy {train_acc:.2f}%')
    print(f'[Epoch {epochs + 1}/{no_epochs}] | Test: Got {t_num_correct} of {t_num_samples} with accuracy {test_acc:.2f}%')

