import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os

from dataset import ImageDataset

dir_train = 'images/data/train'

criterion = nn.CrossEntropyLoss()
epochs = 50
batch_size = 10
validation_split = 0.2
learning_rate = 1e-5

shuffle_dataset = True
num_workers = 8
input_size = 224

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class TransferNnet(nn.Module):
    def __init__(self):
        super(TransferNnet, self).__init__()
        self.main = models.resnet18(pretrained=True).train()
        self.fc = nn.Linear(1000, 9)

    def forward(self, input):
        x=self.main.forward(input)
        x=x.view(-1, 1000)
        return self.fc(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))

def model_init(computing_device):
    """Initialize model."""
    net = TransferNnet()
    net = nn.DataParallel(net).to(computing_device)
    return net

def train_loaders():
    """Get the train and validation loaders."""
    dataset = ImageDataset(transform, dir_train)
    n_val = int(len(dataset) * validation_split)
    n_train = len(dataset) - n_val
    train, val = utils.random_split(dataset, [n_train, n_val])
    train_loader = utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=num_workers, pin_memory=True)
    validation_loader = utils.DataLoader(val, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=num_workers, pin_memory=True)
    return train_loader, validation_loader

def softmax(x, step=True):
    s = nn.Softmax(dim=1)
    values, indices = torch.max(s(x), dim=1)
    return [values, indices]

def image_output(model, images):
    with torch.no_grad():
        dataset = ImageDataset(transform)
        for image in images:
            dataset.append([image, 0, ''])
        loader = utils.DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)
        for item in loader:
            image = item[0]
            output = softmax(model(image))
            print(f'Output: {output[1].item()}')

def train(model, computing_device):
    """Train the model."""

    print('\nStarting training')
    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    train_loader, validation_loader = train_loaders()

    for epoch in range(epochs):

        # Train
        train_loss = 0
        num_correct = 0
        num_examples = 0

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=2)
        for minibatch_count, item in enumerate(train_loader, 1):

            images, labels = item[0].to(computing_device), item[1].to(computing_device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()    

            with torch.no_grad():
                train_loss += float(loss)
                num_correct += int(sum(softmax(outputs)[1] == labels))
                num_examples += len(labels)
            
        train_loss /= minibatch_count

        train_losses.append(train_loss)
        train_accuracies.append(num_correct / num_examples)
        print(f'\nFinished {epoch + 1} epochs of training')
        print(f'Average training loss: {train_loss}')

        # Validation
        val_loss = 0
        num_correct = 0
        num_examples = 0

        with torch.no_grad():
            for minibatch_count, item in enumerate(validation_loader, 1):
                images, labels = item[0].to(computing_device), item[1].to(computing_device)
                outputs = model(images)
                
                num_correct += int(sum(softmax(outputs)[1] == labels))
                num_examples += len(labels)
                val_loss += criterion(outputs, labels).item()

            val_loss /=  minibatch_count
            print(f'Validation loss: {val_loss}')
            val_accuracies.append(num_correct / num_examples)
            val_losses.append(val_loss)

        scheduler.step(val_loss)
        if not os.path.isdir('saved_models'):
            os.makedirs('saved_models')
        model.module.save(os.path.join('saved_models', f'{epoch + 1}.pth'))
        print(f'Train accuracy: {train_accuracies[epoch]}, Validation accuracy: {val_accuracies[epoch]}')

def main():
    use_cuda = torch.cuda.is_available()
    print(f'Using CUDA: {use_cuda}')
    if use_cuda:
        computing_device = torch.device("cuda")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    else:
        computing_device = torch.device("cpu")

    print(f'Batch size: {batch_size}')
    print(f'Validation split: {validation_split}')
    print(f'Initial learning rate: {learning_rate}')

    net = model_init(computing_device)
    train(net, computing_device)
    net.module.load("saved_models/23.pth")
    image_output(net, ['images/data/train/mitchell/0IMG_1807.jpg'])

if __name__ == '__main__':
    main()




