import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc


def G_p(ob, p):
    temp = ob.detach()

    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    return temp


def load_resnet(dataset):
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(in_planes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        def forward(self, x):
            t = self.conv1(x)
            out = F.relu(self.bn1(t))
            torch_model.record(t)
            torch_model.record(out)
            t = self.conv2(out)
            out = self.bn2(self.conv2(out))
            torch_model.record(t)
            torch_model.record(out)
            t = self.shortcut(x)
            out += t
            torch_model.record(t)
            out = F.relu(out)
            torch_model.record(out)

            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = conv3x3(3, 64)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)

            self.collecting = False

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y

        def record(self, t):
            if self.collecting:
                self.gram_feats.append(t)

        def gram_feature_list(self, x):
            self.collecting = True
            self.gram_feats = []
            self.forward(x)
            self.collecting = False
            temp = self.gram_feats
            self.gram_feats = []
            return temp

        def load(self, path="resnet_cifar10.pth"):
            tm = torch.load(path, map_location="cpu")
            self.load_state_dict(tm)

        def get_min_max(self, data, power):
            mins = []
            maxs = []

            for i in range(0, len(data), 128):
                batch = data[i:i + 128].cuda()
                feat_list = self.gram_feature_list(batch)
                for L, feat_L in enumerate(feat_list):
                    if L == len(mins):
                        mins.append([None] * len(power))
                        maxs.append([None] * len(power))

                    for p, P in enumerate(power):
                        g_p = G_p(feat_L, P)

                        current_min = g_p.min(dim=0, keepdim=True)[0]
                        current_max = g_p.max(dim=0, keepdim=True)[0]

                        if mins[L][p] is None:
                            mins[L][p] = current_min
                            maxs[L][p] = current_max
                        else:
                            mins[L][p] = torch.min(current_min, mins[L][p])
                            maxs[L][p] = torch.max(current_max, maxs[L][p])

            return mins, maxs

        def get_deviations(self, data, power, mins, maxs):
            deviations = []

            for i in range(0, len(data), 128):
                batch = data[i:i + 128].cuda()
                feat_list = self.gram_feature_list(batch)
                batch_deviations = []
                for L, feat_L in enumerate(feat_list):
                    dev = 0
                    for p, P in enumerate(power):
                        g_p = G_p(feat_L, P)

                        dev += (F.relu(mins[L][p] - g_p) / torch.abs(mins[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                        dev += (F.relu(g_p - maxs[L][p]) / torch.abs(maxs[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                    batch_deviations.append(dev.cpu().detach().numpy())
                batch_deviations = np.concatenate(batch_deviations, axis=1)
                deviations.append(batch_deviations)
            deviations = np.concatenate(deviations, axis=0)

            return deviations

    torch_model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=100 if dataset == "cifar100" else 10)
    torch_model.load(f"../models/{dataset}/resnet/state_dict.pt")
    torch_model.cuda()
    torch_model.params = list(torch_model.parameters())
    torch_model.eval()
    print("Done")
    return torch_model
