import torch
import torch.nn as nn
from math import sqrt
import itertools
import pandas as pd
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
class DefaultBoxes(object):
    def __init__(self, cfg):
        self.input_size = cfg['input_size']
        self.feature_maps = cfg['feature_maps']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        
    def create_boxes(self):
        default_boxes = []
        
        for k, v in enumerate(self.feature_maps):
            for i, j in itertools.product(range(v), repeat=2):
                f_k = self.input_size/self.steps[k]
                
                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k
                
                # Aspect ratio = 1, small default box
                s_k = self.min_sizes[k]/self.input_size
                default_boxes += [cx, cy, s_k, s_k]
                
                # Aspect ratio = 1, big default box
                s_k_prime = sqrt(s_k*(self.max_sizes[k]/self.input_size))
                default_boxes += [cx, cy, s_k_prime, s_k_prime]
                
                # Aspect ratio = 2
                for ar in self.aspect_ratios[k]:
                    # width > height
                    default_boxes += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    # height > width
                    default_boxes += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        
        out = torch.Tensor(default_boxes).view(-1, 4)
        out.clamp_(max=1, min=0)
            
        return out
        
if __name__ == "__main__":
    df = DefaultBoxes(cfg=cfg) 
    defaults = df.create_boxes()
    print(pd.DataFrame(defaults.numpy()))
               
                
                
                