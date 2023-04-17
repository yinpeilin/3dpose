import torch
import torch.nn as nn
import torch.optim as optim


class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Basic3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
class Residual3DCNN(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(Residual3DCNN, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 512)
        self.fc1 = nn.Linear(100, 1)
        self.fc2 = nn.Linear(50, 1)
        self.fc3 = nn.Linear(362, 100)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        
        
        out = self.fc(out)
        if out.shape[0] ==1:
            y1 = self.fc1(out[0][0:100])
            y2 = self.fc2(out[0][100:150])
            y3 = self.fc3(out[0][150:512])
        else:
            y1 = self.fc1(out[:][0:100])
            y2 = self.fc2(out[:][100:150])
            y3 = self.fc3(out[:][150:512])


        return y1,y2,y3




def Residual3DCNN18(num_classes=1):
    return Residual3DCNN(Basic3DBlock, [2, 3, 3, 2], num_classes)
# model = Residual3DCNN18()
# criterion = nn.MSELoss()
# optimizer = optim

# class Bottleneck3DBlock(nn.Module):
#     expansion = 4

#     def __init__(self, in_channels, out_channels, stride):
#         super(Bottleneck3DBlock, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.conv3 = nn.Conv3d(
#             out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)

#         self.relu = nn.ReLU(inplace=True)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels * self.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm3d(out_channels * self.expansion)
#             )

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out