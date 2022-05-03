import torch
import torch.nn as nn
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=10):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.eps = 1e-10
        self.reset_parameters()
    
    def reset_parameters(self):
        init.constant_(self.weight, self.scale)
        
    def forward(self, x):
        # x.size(): (batch_size, 512, h, w)
        norm = x.pow(2).sum(dim=1, keepdims=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        # self.weight.size(): 512 -> (1,512,1,1) -> (1,512,h,w)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)*x
        return out