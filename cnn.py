import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import normalise
import numpy

# GPU if applicable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 3, 90)
        self.activ = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(3, 3, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)

        log = torch.nn.functional.softmax(x, dim=1)

        return log

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
normaliserAlg = normaliser.hal() #hindawi()
demdataset = DemDataset(normaliserAlg)
trainset, testset = train_test_split(demdataset, test_size=0.2, shuffle=False)

training_loader = DataLoader(trainset, batch_size=3, shuffle=False)
testing_loader = DataLoader(testset, batch_size=3, shuffle=False)
demdetect = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(demdetect.parameters(), lr=0.001)

for epochs in range(1):
    ep_loss = 0.0
    demdetect.train()
    for i, (points, labels) in enumerate(training_loader):
        points = points.to(device)
        points = points.unsqueeze_(1)
        labels = labels.to(device)
        outputs = demdetect(points)
        labels = torch.max(labels,1)[1]
        print(labels)
        break
        # loss = criterion(outputs, torch.max(labels, 1)[1])
        #optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         runnin_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')