import torch
import torch.nn as nn
from modules.box_utils import *

class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_threshold=0.5, negpos_ratio=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.negpos_ratio = negpos_ratio
        self.device = device
    
    def forward(self, predictions, targets):
        """MultiBoxLoss
        Args:
            predictions (tuple): chứa lần lượt output của loc model - loc_data, conf model - conf_data
                                và default_boxes sinh ra từ mạng SSD
                        loc_data (tensor): torch.size(batch_size, 8732, 4)
                        conf_data (tensor): torch.size(batch_size, 8732, 21)
                        default_boxes (tensor): torch.size(8732, 4)
            targets (tensor): ground truth bboxes và labels của từng batch
                        torch.size(batch_size,num_objs,5) (last index is label)
        """
        loc_data, conf_data, default_boxes = predictions
        batch_size = loc_data.size(0)
        num_dfboxes = loc_data.size(1) # 8732
        num_classes = conf_data.size(2) # 21
        
        # Match default boxes and ground truth boxes
        conf_t = torch.LongTensor(batch_size, num_dfboxes).to(self.device) # batch_size, 8732   
        loc_t = torch.Tensor(batch_size, num_dfboxes, 4).to(self.device) # batch_size, 8732, 4
        
        for idx in range(batch_size):
            # ground truth data
            truths = targets[idx][:,:-1].to(self.device) # xmin, ymin, xmax, ymax
            labels = targets[idx][:,-1].to(self.device) # label
            
            dfboxes = default_boxes.to(self.device) 
            variances = [0.1, 0.2]
            match(self.jaccard_threshold, truths, dfboxes, variances, labels, loc_t, conf_t, idx)
            