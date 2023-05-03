from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import warnings

warnings.filterwarnings('ignore')

# sigNet
class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 128)

    def forward(self, x1):
        x1 = F.relu(self.conv1(x1))
        x1 = F.local_response_norm(x1, size=2)
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)

        x1 = F.relu(self.conv2(x1))
        x1 = F.local_response_norm(x1, size=2)
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)

        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        x1 = x1.view(-1, 256 * 4 * 4)
        x1 = F.relu(self.fc1(x1))
        x1 = F.dropout(x1, training=self.training)
        x1 = F.relu(self.fc2(x1))
        x1 = F.dropout(x1, training=self.training)
        out = self.fc3(x1)

        return out

# Resnet10 Pretrained
resnet = models.resnet18(pretrained=True)
# input size is 1x96x96
# resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(512, 128)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = resnet

    def forward(self, x1, x2):
        out1 = self.model(x1)
        out2 = self.model(x2)
        return out1, out2

    def get_embedding(self, x):
        return self.model(x)

    def get_distance(self, x1, x2):
        out1, out2 = self.forward(x1, x2)
        return F.pairwise_distance(out1, out2)

    def get_similarity(self, x1, x2):
        return 1 - self.get_distance(x1, x2)

    def get_loss(self, x1, x2, y):
        out1, out2 = self.forward(x1, x2)
        loss = F.cross_entropy(torch.cat([out1, out2], dim=0), torch.cat([y, y], dim=0))
        return loss

    def get_accuracy(self, x1, x2, y):
        out1, out2 = self.forward(x1, x2)
        pred = torch.argmax(torch.cat([out1, out2], dim=0), dim=1)
        return torch.sum(pred == torch.cat([y, y], dim=0)).item() / (2 * len(y))

    # cosine similarity
    def get_cosine_similarity(self, x1, x2):
        out1, out2 = self.forward(x1, x2)
        return F.cosine_similarity(out1, out2)
