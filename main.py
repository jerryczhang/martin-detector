import torch
import torch.nn as nn
import numpy as np
import os

from dataset import ImageDataset
from train import train
from model import MNet

def model_init(device):
    """Initialize model."""

    net = MNet()
    net = nn.DataParallel(net).to(device)
    return net

def image_output(model, images):
    with torch.no_grad():
        model.eval()
        dataset = ImageDataset(transform)
        for image in images:
            dataset.append([image, 0, ''])
        loader = utils.DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)
        for item in loader:
            image = item[0]
            output = softmax(model(image))
            print(f'Output: {output[1].item()}')

def test_eval(model, device):
    print('\nStarting evaluation on test set')
    model.eval()

    dataset = ImageDataset(transform, dir_test)
    test_loader = utils.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    num_correct = 0
    for item in test_loader:
        images, labels = item[0].to(device), item[1].to(device)

        outputs = model(images)
        num_correct += int(sum(softmax(outputs)[1] == labels))

    accuracy = num_correct / len(dataset)
    print(f'Accuracy on test set: {accuracy * 100}')

def main():
    use_cuda = torch.cuda.is_available()
    print(f'Using CUDA: {use_cuda}')
    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    net = model_init(device)
    #net.module.load("saved_models/50.pth")
    train(net, device)
    #test_eval(net, computing_device)

if __name__ == '__main__':
    main()




