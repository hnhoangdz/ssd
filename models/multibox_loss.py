import torch
import torch.nn as nn
from models.modules.box_utils import *
import torch.nn.functional as F

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
        
        #SmoothL1Loss
        pos_mask = conf_t > 0
        # loc_data(batch_size, 8732, 4)
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # positive dbox, loc_data
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

        #loss_conf
        #CrossEntropy
        batch_conf = conf_data.view(-1, num_classes) #(batch_size*num_box, num_classes)
        loss_conf = F.cross_entropy(batch_conf, conf_t.view(-1), reduction="none")

        # hard negative mining
        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_conf = loss_conf.view(batch_size, -1) # torch.size([batch_size, 8732])

        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # idx_rank chính là thông số để biết được độ lớn loss nằm ở vị trí bao nhiêu

        num_neg = torch.clamp(num_pos*self.neg_pos, max=num_dfboxes)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        #(batch_size, 8732) -> (batch_size, 8732, 21)
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)
        conf_t_pre = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)
        conf_t_ = conf_t[(pos_mask+neg_mask).gt(0)]
        loss_conf = F.cross_entropy(conf_t_pre, conf_t_, reduction="sum")

        # total loss = loss_loc + loss_conf
        N = num_pos.sum()
        loss_loc = loss_loc/N
        loss_conf = loss_conf/N

        return loss_loc, loss_conf

            