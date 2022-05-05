from models.extras import extras
from models.loc_conf import loc_conf
from models.vgg import vgg
from models.modules.l2norm import L2Norm
from models.modules.default_boxes import DefaultBoxes
from models.modules.cfg import *
import torch
import torch.nn as nn
from models.detection import Detect

class SSD(nn.Module):
    def __init__(self, phase, cfg=cfg):
        super(SSD, self).__init__()
        self.num_classes = cfg['num_classes']
        self.phase = phase
        
        # main layers
        self.vgg = vgg()
        self.extras = extras()
        loc_conf_layers = loc_conf(self.num_classes, cfg['bbox_aspect_num'])    
        self.loc = loc_conf_layers[0]
        self.conf = loc_conf_layers[1]        
        
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
        # x ~ output của conv4_3
        source1 = self.L2Norm(x)
        sources.append(source1)
        
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        
        # x ~ source2
        sources.append(x)
        
        # Đi qua mạng Extras, tạo ra source 3,4,5,6
        for k,v in enumerate(self.extras):
            x = nn.ReLU(self.extras[k](x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        
        # Đi qua mạng loc và conf
        for (s, l, c) in zip(sources, self.loc, self.conf):
            # (batch_size, 4*bbox_aspect_num, featuremap_height, featuremap_width, featuremap_width)
            # -> (batch_size, featuremap_height, featuremap_width, featuremap_width, 4*bbox_aspect_num)
            loc.append(l(s).permute(0,2,3,1).contiguous()) # đảm bảo bộ nhớ liên tiếp để sử dụng view
            conf.append(c(s).permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0),-1) for o in loc], dim=1) # (batch_size, 4*8732)
        conf = torch.cat([o.view(o.size(0),-1) for o in conf], dim=1) # (batch_size, 21*8732)

        loc = loc.view(loc.size(0), -1, 4) # (batch_size, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes) #(batch_size, 8732, 21)
        
        output = (loc, conf, self.default_boxes)
        
        if self.phase == 'inference':
            return self.detect(output[0], output[1], output[2])
        else:
            return output
        
        
if __name__ == '__main__':
    ssd = SSD('train', cfg)
    print(ssd)