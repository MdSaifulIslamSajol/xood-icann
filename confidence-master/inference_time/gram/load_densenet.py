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
import math
import sys
sys.path.insert(0, '..')
from models.load import load_model


def load_densenet(dataset):
    def G_p(ob, p):
        temp = ob.detach()
        temp = temp ** p
        temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
        temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
        temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)
        return temp

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class BottleneckBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(BottleneckBlock, self).__init__()
            inter_planes = out_planes * 4
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(inter_planes)
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate

        def forward(self, x):

            out = self.conv1(self.relu(self.bn1(x)))

            torch_model.record(out)

            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

            out = self.conv2(self.relu(self.bn2(out)))
            torch_model.record(out)

            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return torch.cat([x, out], 1)

    class TransitionBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(TransitionBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.droprate = dropRate

        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            torch_model.record(out)

            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return F.avg_pool2d(out, 2)

    class DenseBlock(nn.Module):
        def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
            super(DenseBlock, self).__init__()
            self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

        def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
            layers = []
            for i in range(int(nb_layers)):
                layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
            return nn.Sequential(*layers)

        def forward(self, x):
            t = self.layer(x)
            torch_model.record(t)
            return t

    class DenseNet3(nn.Module):
        def __init__(self, depth, num_classes, growth_rate=12,
                     reduction=0.5, bottleneck=True, dropRate=0.0):
            super(DenseNet3, self).__init__()

            self.collecting = False

            in_planes = 2 * growth_rate
            n = (depth - 4) / 3
            if bottleneck == True:
                n = n / 2
                block = BottleneckBlock
            else:
                block = BasicBlock
            # 1st conv before any dense block
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            # 1st block
            self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes + n * growth_rate)
            self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes * reduction))
            # 2nd block
            self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes + n * growth_rate)
            self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes * reduction))
            # 3rd block
            self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes + n * growth_rate)
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(in_planes, num_classes)
            self.in_planes = in_planes

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

        def forward(self, x):
            out = self.conv1(x)
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.in_planes)
            return self.fc(out)

        def load(self, path="densenet_cifar100.pth"):
            state_dict = torch.load(path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

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

        def get_min_max(self, data, power):
            mins = []
            maxs = []

            for i in range(0, len(data), 64):
                batch = data[i:i + 64].cuda()
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

            for i in range(0, len(data), 64):
                batch = data[i:i + 64].cuda()
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

    torch_model = DenseNet3(100, num_classes=100 if dataset == "cifar100" else 10)
    torch_model.load(f"../models/{dataset}/densenet/state_dict.pt")
    torch_model.cuda()
    torch_model.params = list(torch_model.parameters())
    torch_model.eval()
    print("Done")
    return torch_model
