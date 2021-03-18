import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as utils

from dataset import ImageDataset

dir_train = 'images/data/train'
dir_test = 'images/data/test'

criterion = nn.CrossEntropyLoss()
epochs = 50
batch_size = 50
validation_split = 0.2
learning_rate = 1e-5

shuffle_dataset = True
num_workers = 8
input_size = 224

def train_loaders(transform):
    """Get the train and validation loaders."""

    dataset = ImageDataset(transform, dir_train)
    n_val = int(len(dataset) * validation_split)
    n_train = len(dataset) - n_val
    train, val = utils.random_split(dataset, [n_train, n_val])
    train_loader = utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=num_workers, pin_memory=True)
    validation_loader = utils.DataLoader(val, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=num_workers, pin_memory=True)
    return train_loader, validation_loader

def train_loop(model, device, optimizer, scheduler, train_loader):
    """Execute one epoch of training"""

    train_loss = 0
    num_correct = 0
    num_examples = 0

    for minibatch, item in enumerate(train_loader, 1):
        images, labels = item[0].to(device), item[1].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        with torch.no_grad():
            train_loss += float(loss)
            num_correct += int(sum(torch.argmax(F.softmax(outputs, dim=1),dim=1) == labels))
            num_examples += len(labels)

    train_loss /= minibatch
    accuracy = num_correct / num_examples
    return train_loss, accuracy

def val_loop(model, device, scheduler, validation_loader):
    """Execute one epoch of validation"""

    val_loss = 0
    num_correct = 0
    num_examples = 0

    with torch.no_grad():
        for minibatch, item in enumerate(validation_loader, 1):
            images, labels = item[0].to(device), item[1].to(device)
            outputs = model(images)

            val_loss += criterion(outputs, labels).item()
            num_correct += int(sum(torch.argmax(F.softmax(outputs, dim=1),dim=1) == labels))
            num_examples += len(labels)

        val_loss /= minibatch
        val_accuracy = num_correct / num_examples

    scheduler.step(val_loss)
    return val_loss, val_accuracy


def train(model, device):
    """Train the model."""

    print('\nStarting training')
    print(f'Batch size: {batch_size}')
    print(f'Validation split: {validation_split}')
    print(f'Initial learning rate: {learning_rate}')

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_loader, validation_loader = train_loaders(transform)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(epochs):

        # Train
        train_loss, train_accuracy = train_loop(model, device, optimizer, scheduler, train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f'\nFinished {epoch + 1} epochs of training')
        print(f'Average training loss: {train_loss}')

        # Validation
        val_loss, val_accuracy = val_loop(model, device, scheduler, validation_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Validation loss: {val_loss}')

        print(f'Train accuracy: {train_accuracies[epoch]}, Validation accuracy: {val_accuracies[epoch]}')

        # Save model
        if not os.path.isdir('saved_models'):
            os.makedirs('saved_models')
        model.module.save(os.path.join('saved_models', f'{epoch + 1}.pth'))
