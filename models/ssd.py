from models.extras import extras
from models.loc_conf import loc_conf
from models.vgg import vgg
from models.modules.l2norm import L2Norm
from models.modules.default_boxes import DefaultBoxes
from models.detection import Detect
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.num_classes = cfg['num_classes']
        self.phase = phase
        
        # main layers
        self.vgg = vgg()
        self.extras = extras()  
        self.loc, self.conf = loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])        

        # modules
        self.L2Norm = L2Norm()
        dbox_instance = DefaultBoxes(cfg)
        self.default_boxes = dbox_instance.create_boxes()
        
        if phase == 'inference':
            self.detect = Detect()

    # input x: image
    def forward(self, x):
        sources = [] # chứa 6 sources
        loc = []
        conf = []
        
        # Đi qua mạng VGG
        for k in range(23):
            x = self.vgg[k](x)
        # x = output của conv4_3
        source1 = self.L2Norm(x)
        sources.append(source1)
        
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        
        # x = source2
        sources.append(x)

        # Đi qua mạng Extras, tạo ra source 3,4,5,6
        for k,v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
                
        # Đi qua mạng loc và conf
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # (batch_size, 4*bbox_aspect_num, featuremap_height, featuremap_width)
            # -> (batch_size, featuremap_height, featuremap_width, 4*bbox_aspect_num)
            loc.append(l(x).permute(0,2,3,1).contiguous()) # đảm bảo bộ nhớ liên tiếp để sử dụng view
            conf.append(c(x).permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0),-1) for o in loc], dim=1) # (batch_size, 4*8732)
        conf = torch.cat([o.view(o.size(0),-1) for o in conf], dim=1) # (batch_size, 21*8732)

        loc = loc.view(loc.size(0), -1, 4) # (batch_size, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes) #(batch_size, 8732, 21)
        
        output = (loc, conf, self.default_boxes)
        
        if self.phase == 'inference':
            with torch.no_grad():
                return self.detect(output[0], output[1], output[2])
        else:
            return output
        
        
if __name__ == '__main__':
    cfg = {
        'num_classes': 21, # number of classes (contained back_ground)
        'input_size': 300, # SSD300
        'bbox_aspect_num': [4,6,6,6,4,4], # number of bboxes for each cell (pixel) in each feature map
        'feature_maps': [38, 19, 10, 5, 3, 1], # feature maps spatial
        'steps' : [8, 16, 32, 64, 100, 300], # Size of default box
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    }
    ssd = SSD('train', cfg)
    print(ssd)