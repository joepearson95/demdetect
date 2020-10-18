import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os

# Dataset creation of the above csv file
class DemDataset(Dataset):
    # read the csv in but not the images for efficiency
    def __init__(self, csv_file,transform=None):
        self.dem_dataset = pd.read_csv(csv_file)
        self.dem_dataset = self.dem_dataset.drop(list(self.dem_dataset)[53:153], axis=1)
        #self.dem_dataset = self.dem_dataset[self.dem_dataset.columns.drop(list(self.dem_dataset.filter(regex='score')))]
        self.transform = transform
        
    def __len__(self):
        return len(self.dem_dataset)

    # resolve pandas issue and display the image name/keypoint x & y with score
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.dem_dataset.iloc[:, -2]
        keypoints =  self.dem_dataset.iloc[idx, :-2]
        keypoints = np.array([keypoints])
        keypoints = keypoints.reshape(-1,3)
        sample = {'image': img_name, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return [img_name, keypoints]

#dem_dataset = DemDataset(csv_name)

# LSTM model to classify the actions
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
         
        out = self.fc(out)
        return out

#action_model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Hyper-parameters 
csv_name = '#'
sequence_length = 28


def train_model(csv_file, num_epochs):
    # Load in the required dataset
    demdataset = DemDataset(csv_name)

    # split into the right sets
    # MAYBE CHANGE THIS TO A NICER LOOKING ONE?
    train_size = int(0.8 * len(demdataset))
    test_size = len(demdataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Create the dataloader
    training_loader = DataLoader(trainset, batch_size=100, shuffle=True)
    testing_loader = DataLoader(testset, batch_size=100, shuffle=True)

    # GPU if applicable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyperparam
    input_size = 28
    hidden_size = 15
    num_layers = 2
    num_classes = 4
    learning_rate = 0.0001
    num_epochs = 2

    # Model
    action_model = RNN(input_size, hidden_size, num_layers, num_classes)

    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(action_model.parameters(), lr=learning_rate)

    # Begin training
    # loss_iter = []
    # loss_batch = []
    iter = 0
    for epoch in range(num_epochs):
        correct = 0
        for i, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = action_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs == labels).float().sum()
        print(100 * correct / len(trainset))
        if epoch%25  == 1:
            print(f'epoch: {i:3} loss: {loss.item():10.8f}')
            
# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# # Train the model
# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):  
#         # origin shape: [N, 1, 28, 28]
#         # resized: [N, 28, 28]
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')