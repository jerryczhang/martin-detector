import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from dataset import ImageDataset
from model import MNet

dir_train = 'images/data/train'
dir_val = 'images/data/val'

epochs = 100
batch_size = 125
learning_rate = 1e-5

shuffle_dataset = True
input_size = 224

def train_loaders():
    """Get the train and validation loaders"""

    train_augment = transforms.Compose([
        transforms.ColorJitter(0.9, 0.9, 0.9, 0.2),
        transforms.RandomAffine(degrees=10, shear=10),
        transforms.GaussianBlur(5),
        transforms.RandomPerspective()
    ])
    train_dataset = ImageDataset(img_dir=dir_train, augment=train_augment)
    val_dataset = ImageDataset(img_dir=dir_val)
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=1, pin_memory=True)
    validation_loader = utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=1, pin_memory=True)
    return train_loader, validation_loader

def train_loop(model, device, criterion, optimizer, scheduler, train_loader):
    """Execute one epoch of training"""

    train_loss = 0
    num_correct = 0
    num_examples = 0

    model.train()
    for minibatch, item in enumerate(train_loader, 1):
        images, labels = item['image'].to(device), item['label'].to(device)

        outputs = model(images)
        loss = criterion(outputs['logits'], labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        with torch.no_grad():
            train_loss += float(loss)
            num_correct += int(sum(outputs['argmax'] == labels))
            num_examples += len(labels)

    train_loss /= minibatch
    accuracy = num_correct / num_examples
    return train_loss, accuracy

def val_loop(model, device, criterion, scheduler, validation_loader):
    """Execute one epoch of validation"""

    val_loss = 0
    num_correct = 0
    num_examples = 0

    model.eval()
    with torch.no_grad():
        for minibatch, item in enumerate(validation_loader, 1):
            images, labels = item['image'].to(device), item['label'].to(device)
            outputs = model(images)

            val_loss += criterion(outputs['logits'], labels).item()
            num_correct += int(sum(outputs['argmax'] == labels))
            num_examples += len(labels)

        val_loss /= minibatch
        val_accuracy = num_correct / num_examples

    scheduler.step(val_loss)
    return val_loss, val_accuracy


def train(model, device):
    """Train the model"""

    print('\nStarting training')
    print(f'Batch size: {batch_size}')
    print(f'Initial learning rate: {learning_rate}')

    train_loader, validation_loader = train_loaders()

    train_weights = torch.Tensor([0.05,0.078,0.58,0.059,0.116,0.362,0.161,0.034,0.725]).to(device)
    criterion = nn.CrossEntropyLoss(weight=train_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    writer = SummaryWriter()

    for epoch in tqdm(range(epochs)):

        # Train
        train_loss, train_accuracy = train_loop(model, device, criterion, optimizer, scheduler, train_loader)

        # Validation
        val_loss, val_accuracy = val_loop(model, device, criterion, scheduler, validation_loader)

        # Summary
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch+1)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch+1)

        # Save model
        if not os.path.isdir('saved_models'):
            os.makedirs('saved_models')
        model.module.save(os.path.join('saved_models', f'{epoch + 1}.pth'))
    writer.close()

def main():
    use_cuda = torch.cuda.is_available()
    assert use_cuda, "CUDA not available"
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    net = MNet()
    net = nn.DataParallel(net).to(device)
    train(net, device)

if __name__ == '__main__':
    main()
