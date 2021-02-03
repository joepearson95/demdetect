import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import normalise

# GPU if applicable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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

# Hyper Param
input_size = 45
sequence_length = 1
batch_size = 1
hidden_size = 3
num_layers = 1
num_classes = 3
no_epochs = 10
lr_rate = 0.001

# Create the dataloader
training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
testing_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
demdetect = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(demdetect.parameters(), lr=lr_rate)

num_correct = 0
num_samples = 0
t_num_correct = 0
t_num_samples = 0

train_err = []
test_err = []
for epochs in range(no_epochs):
    ep_loss = 0.0
    demdetect.train()
    for i, (points, labels) in enumerate(training_loader):
        points = points.to(device)
        points = points.unsqueeze_(1)
        labels = labels.to(device)
        outputs = demdetect(points)
        labels = torch.max(labels, 1)[1]
        print(outputs, outputs.shape)
        print(labels, labels.shape)
        break
        loss = criterion(outputs, torch.max(labels, 1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = outputs.max(1)
        num_correct += (predictions == torch.max(labels, 1)[1]).sum()
        num_samples += predictions.size(0)
        ep_loss += loss.item()
    train_acc = float(num_correct) / float(num_samples) * 100
    train_err.append((train_acc - 100) * -1)
    # End train loop
    demdetect.eval()
    for (points, labels) in testing_loader:
        points = points.to(device)
        points = points.unsqueeze_(1)
        labels = labels.to(device)
        outputs = demdetect(points)

        _, predictions = outputs.max(1)
        t_num_correct += (predictions == torch.max(labels, 1)[1]).sum()
        t_num_samples += predictions.size(0)
    test_acc = float(t_num_correct) / float(t_num_samples) * 100
    test_err.append((test_acc - 100) * -1)
    print(f'[Epoch {epochs + 1}/{no_epochs}] Epoch Loss: {ep_loss / len(training_loader):.3f} | Got {num_correct} of {num_samples} with accuracy {train_acc:.2f}%')
    print(f'[Epoch {epochs + 1}/{no_epochs}] | Test: Got {num_correct} of {t_num_samples} with accuracy {test_acc:.2f}%')
# print('Finished Training.')

# # print(len(train_acc_list), len(test_acc_list))
# # plt.plot(train_err, 'r', test_err, 'g')
# # plt.show()
