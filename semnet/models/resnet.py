import torch
import torch.nn as nn
from torchvision.models import resnet

class SEMNet(nn.Module):
    def __init__(self, n_feature=4, n_class=2, backbone='resnet18'):
        super(SEMNet, self).__init__()
        self.n_feature = n_feature
        self.n_class = n_class
        self.backbnone = backbone
        if self.backbnone == 'resnet18':
            base = resnet.resnet18(pretrained=True)
        self.in_block = nn.Sequential(
            nn.Conv2d(self.n_feature, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            base.bn1,
            base.relu,
            base.maxpool)
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        self.avgpool = base.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 1 , bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.in_block(x)
        h = self.encoder1(h)
        h = self.encoder2(h)
        h = self.encoder3(h)
        h = self.encoder4(h)
        y = self.sigmoid(self.fc(self.flatten(self.avgpool(h))))
        return y

if __name__ == "__main__":
    model = SEMNet().cuda()
    x = torch.rand(2, 4, 512, 512).cuda()
    y = model(x)
    print(y)
    # t = torch.tensor([1.0, 0.0, 1.0])
    # y = torch.tensor([0.02, 0.05, 0.99])
    # bce = nn.BCELoss()
    # l = torch.log(torch.tensor(0.02))+ torch.log(torch.tensor(1-0.05)) + torch.log(torch.tensor(0.99))
    # print(bce(y, t))
    # print(l)