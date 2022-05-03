import torch.nn as nn
"""
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
    padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
"""

def vgg():
    layers = []
    in_channels = 3

    cfgs = [64, 64, 'M', # block 1
            128, 128, 'M', # block 2
            256, 256, 256, 'MC', # block 3
            512, 512, 512, 'M', # block 4
            512, 512, 512 # block 5
           ]
    
    for cfg in cfgs:
        if cfg == 'M': # floor 3.5 -> 3
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            layers += [pool]
        elif cfg == 'MC': # ceil 3.5 -> 4
            pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            layers += [pool]
        else: 
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=cfg, kernel_size=3,padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg
            
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)   
    layers += [pool5,conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

if __name__ == '__main__':
    vgg_model = vgg()
    print(vgg_model)
    
