import torch
import torch.nn as nn
torch.manual_seed(1234)

def extras():
    layers = []
    in_channels = 1024 # output of vgg 

    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]

    # Block 1
    conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cfgs[0], kernel_size=1) # 19x19
    conv2 = nn.Conv2d(in_channels=cfgs[0], out_channels=cfgs[1], kernel_size=3, stride=2, padding=1) # 10x10

    # Block 2
    conv3 = nn.Conv2d(in_channels=cfgs[1], out_channels=cfgs[2], kernel_size=1) # 10x10
    conv4 = nn.Conv2d(in_channels=cfgs[2], out_channels=cfgs[3], kernel_size=3, stride=2, padding=1) # 5x5

    # Block 3
    conv5 = nn.Conv2d(in_channels=cfgs[3], out_channels=cfgs[4], kernel_size=1) # 5x5
    conv6 = nn.Conv2d(in_channels=cfgs[4], out_channels=cfgs[5], kernel_size=3) # 3x3

    # Block 4
    conv7 = nn.Conv2d(in_channels=cfgs[5], out_channels=cfgs[6], kernel_size=1) # 3x3
    conv8 = nn.Conv2d(in_channels=cfgs[6], out_channels=cfgs[7], kernel_size=3) # 1x1

    layers += [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8]

    return nn.ModuleList(layers)

if __name__ == '__main__':
    base_extras = extras()
    print(base_extras)