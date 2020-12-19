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
batch_size = 50
validation_split = 0.2
learning_rate = 1e-5

shuffle_dataset = True
num_workers = 3
input_size = 224

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

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
    print(net)
    return net

def train_loaders():
    """Get the train and validation loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ImageDataset(dir_train, transform)
    n_val = int(len(dataset) * validation_split)
    n_train = len(dataset) - n_val
    train, val = utils.random_split(dataset, [n_train, n_val])
    train_loader = utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=num_workers, pin_memory=True)
    validation_loader = utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, validation_loader

def softmax(x, step=True):
    s = nn.Softmax(dim=1)
    values, indices = torch.max(s(x), dim=1)
    return [values, indices]

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

def image_output(model, images):
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Pad(get_padding(imarray)),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        for image in images:
            transformed = transform(image).reshape([1, 3, input_size, input_size])
            output = softmax(model(transformed))
            print(f'Output: {output}')

def train(model, computing_device):
    """Train the model."""

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    train_loader, validation_loader = train_loaders()

    for epoch in range(50):

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
        print(f'Finished {epoch + 1} epochs of training')
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
        model.module.save(f'saved_models/test/{epoch}.pth')
        print(f'Train accuracy: {train_accuracies[epoch]}, Validation accuracy: {val_accuracies[epoch]}')
        print()

def main():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        print("CUDA is supported")
        print(torch.cuda.device_count())
    else:
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    net = model_init(computing_device)
    train(net, computing_device)
    net.module.load("saved_models/test/15.pth")
    image_output(net, ['images/data/train/mitchell/0IMG_1807.jpg'])

if __name__ == '__main__':
    main()




