import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.main = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 9)

    def forward(self, input):
        x = self.main.forward(input)
        x = x.view(-1, 1000)
        logits = self.fc(x)
        softmax = F.softmax(logits, dim=1)
        argmax = torch.argmax(softmax, dim=1)
        return {'logits':logits, 'softmax':softmax, 'argmax':argmax}

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
