#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
from PIL import Image

criterion = nn.BCEWithLogitsLoss()
batch_size = 50
shuffle_dataset = True
random_seed = 42
num_workers = 3
validation_split = 0.2
input_size = 224


class TransferNnet(nn.Module):
    def __init__(self):
        super(TransferNnet, self).__init__()
        self.main = models.alexnet(pretrained=True).eval()
        self.fc = nn.Linear(1000, 1)

    def forward(self, input):
        x=self.main.forward(input)
        x=x.view(-1, self.num_flat_features(x))
        return self.fc(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

class loader:
    """Load in the dataset from the csv file."""
    def __init__(self, csv_file, transform=None):
        """Initialize the dataframe to hold the data."""
        self.transform = transform
        self.frame = pd.read_csv(csv_file)

    def __len__(self):
        """Get the length of the dataframe."""
        return len(self.frame)

    def __getitem__(self, idx):
        """Get image and label for a certain index."""
        img_name = self.frame.iloc[idx, 0]
        img = Image.open(img_name)
        label = self.frame.iloc[idx, 1]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
def model_init(computing_device):
    """Initialize model."""
    net = TransferNnet()
    net = nn.DataParallel(net).to(computing_device)
    print(net)
    return net

def train_loaders():
    """Get the train and validation loaders."""
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
    dataset = loader('train.csv', transform=transform)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    split = int(np.floor(dataset_size * validation_split))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = utils.SubsetRandomSampler(train_indices)
    valid_sampler = utils.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=num_workers)
    return train_loader, validation_loader

def test_loaders():
    """Get the test loader."""
    transform = transforms.Compose([
                transforms.Resize(input_size), 
                transforms.CenterCrop(input_size), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
    test_set = loader(test_dataset,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    return test_loader

def sigmoid_output(x, step=True):
    s = nn.Sigmoid()
    if step:
        return torch.round(s(x))
    else:
        return s(x)
    
def get_padding(image):
    w, h = image.shape[0:2]
    max_dim = max([w,h])
    horz = (max_dim - w) / 2
    vert = (max_dim - h) / 2
    l_pad = int(np.ceil(horz))
    r_pad = int(np.floor(horz))
    t_pad = int(np.ceil(vert))
    b_pad = int(np.floor(vert))
    return (t_pad, r_pad, b_pad, l_pad)

def image_output(model, image):
    imarray = np.array(image)
    transform = transforms.Compose([
        transforms.Pad(get_padding(imarray)),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    with torch.no_grad():
        transformed = transform(image).reshape([1, 3, input_size, input_size])
        output = sigmoid_output(model(transformed), step=False)
        print(f'Output: {float(output)}')

def train(model, computing_device):
    """Train the model."""

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    loc = "saved_models"
    if not os.path.exists(loc):
        os.makedirs(loc)
    train_loader, validation_loader = train_loaders()

    for epoch in range(50):
        # learning_rates = [1e-3 for x in range(20)] + [0.5e-3 for x in range(10)] + [0.25e-3 for x in range(10)] + [0.12e-3 for x in range(10)]
        learning_rates = [1e-5 for x in range(50)]
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[epoch], weight_decay=0.001)
        for (images, labels) in train_loader:
            
            optimizer.zero_grad()
            images, labels = images.to(computing_device), labels.to(computing_device)
            outputs = model(images)
            labels = labels.reshape((len(labels), 1)).type_as(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    

            train_loss = float(loss)
            num_correct = int(sum(sigmoid_output(outputs) == labels))
            num_examples = len(labels)

        train_losses.append(train_loss)
        train_accuracies.append(num_correct / num_examples)
        print("Finished", epoch + 1, "epochs of training (lr = " + str(learning_rates[epoch]) + ")")
        print("Average training loss: " + str(train_loss))

        # Validation
        
        num_correct = 0
        num_examples = 0
        val_loss = 0
        
        with torch.no_grad():
            for minibatch_count, (images, labels) in enumerate(validation_loader, 0):
                images, labels = images.to(computing_device), labels.to(computing_device)
                outputs = model(images)
                labels = labels.reshape((len(labels), 1)).type_as(outputs)
                
                num_correct += int(torch.sum(sigmoid_output(outputs) == labels))
                num_examples += len(labels)
                val_loss += criterion(outputs, labels).item()
            
            val_loss /= (minibatch_count + 1)
            print("Validation loss: " + str(val_loss))
            val_accuracies.append(num_correct / num_examples)
            val_losses.append(val_loss)

        # Save statistics
        model.module.save(f'saved_models/test/{epoch}.pth')
        print("Train accuracy: %f, Validation accuracy: %f" % (train_accuracies[epoch], val_accuracies[epoch]))
        print()
        """
        training_statistics = pd.DataFrame({
           "train_losses":train_losses, 
           "val_losses":val_losses, 
           "train_accuracies":train_accuracies, 
           "val_accuracies":val_accuracies,
           }) 
        training_statistics.to_csv(loc + "/training_statistics.csv")
        """
        
def main():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("CUDA is supported")
        print(torch.cuda.device_count())
    else:
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    net = model_init(computing_device)
    net.module.load("saved_models/test/24.pth")
    #train(net, computing_device)
    #image_output(net, Image.open('martin_test.jpg'))

if __name__ == '__main__':
    main()




