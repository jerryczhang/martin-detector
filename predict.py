import torch
import torch.nn as nn
import torch.utils.data as utils

from dataset import ImageDataset
from model import MNet

dir_test = 'images/data/test'
batch_size = 50

def image_output(model, images):
    """Get predicted output on a single image"""

    with torch.no_grad():
        model.eval()
        dataset = ImageDataset()
        for image in images:
            dataset.append([image, 0, ''])
        loader = utils.DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)
        for item in loader:
            image = item[0]
            output = model(image, output='argmax')
            print(f'Output: {output}')

def test_eval(model, device):
    """Evaluate performance on the test set"""

    print('\nStarting evaluation on test set')
    model.eval()

    dataset = ImageDataset(dir_test)
    test_loader = utils.DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    num_correct = 0
    for item in test_loader:
        images, labels = item[0].to(device), item[1].to(device)

        outputs = model(images, output='argmax')
        num_correct += int(sum(outputs == labels))

    accuracy = num_correct / len(dataset)
    print(f'Accuracy on test set: {accuracy * 100}')

def main():
    use_cuda = torch.cuda.is_available()
    assert use_cuda, "CUDA not available"
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    net = MNet()
    net = nn.DataParallel(net).to(device)
    net.module.load('saved_models/49.pth')
    test_eval(net, device)

if __name__ == '__main__':
    main()




