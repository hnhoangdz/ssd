import torch
import torch.nn as nn
from torch.autograd import Function

# Decode boxes
def decode(loc, dboxes):
    """ đối với 1 bức ảnh,
        kết hợp default boxes với offset boxes từ loc model để lấy ra decode boxes (boxes dự đoán) 
    Args:
        loc (8732, 4): output của loc model
        dboxes (8732, 4): default boxes
    Return:
        boxes (8732, 4): (xmin, ymin, xmax, ymax)
    Note:boxes[:,:2] -> cx,cy, boxes[:,2:] -> w,h
        dim=1: concatenate bằng cột (e.g: [1],[1] -> [[1,1]]
    """
    # công thức: ../images/decode.png 
    boxes = torch.cat(( dboxes[:, :2]*(1 + 0.1*loc[:,:2]),
                        dboxes[:, 2:]*torch.exp*(0.2*loc[:,2:])), dim=1)
    # biến đổi cx,cy,w,h -> xmin,ymin,xmax,ymax 
    boxes[:, :2] -= boxes[:, 2:]/2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

# Với mỗi class, tìm ra những boxes tốt nhất thuộc class đó
# Confidence cao nhất và bounding boxes có độ overlap < 0.45
def nms(boxes, scores, overlap=0.45, top_k=200):
    """ đối với 1 bức ảnh, đi từng class
        sử dụng non-max suppression để xóa bớt các bounding boxes trùng quá lớn của 1 object
    Args:
        boxes (8732, 4): decode boxes
        scores (8732,): confidence score của class i-th
        overlap (float): ngưỡng để loại bỏ hay giữ lại (>0.45 thì loại bỏ)
        top_k (int): số lượng scores và boxes tối đa giữ lại
    Return:
        keep: index của các boxes được lưu trữ thông qua default boxes
        count: số lượng xuất hiện của một class trong ảnh
    """
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    
    # tọa độ các boxes
    x1 = boxes[:,0] # xmin (8732,)
    y1 = boxes[:,1] # ymin (8732,)
    x2 = boxes[:,2] # xmax (8732,)
    y2 = boxes[:,3] # ymax (8732,)
    
    # diện tích của tất cả các boxes
    areas = torch.mul(x2-x1, y2-y1) # (8732,)
    
    # sắp xếp tăng dần (dim = 0: theo hàng)
    v,idx = scores.sort(dim=0)
    
    # tạo các temporary tensor để lưu trữ lúc sau
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    ww = boxes.new()
    hh = boxes.new()
    
    # chọn ra top_k scores lớn nhất
    best_idxes = idx[-top_k:]
    
    while best_idxes.numel() > 0:
        # lấy ra id của score lớn nhất
        id = best_idxes[-1] 
        
        # giữ lại id của score lớn nhất
        keep[count] = id
        # tăng số lượng xuất hiện của một class trong ảnh
        count += 1
        
        if best_idxes.size(0) == 1:
            print('there is only one box here !!!')
            break
        
        # bỏ qua id lớn nhất đang xét
        best_idxes = best_idxes[:-1] 
        
        # Lấy ra Information Boxes
        # gán giá trị của x1 cho xx1 thông qua best_idxes
        torch.index_select(x1, dim=0, index=best_idxes, out=xx1)
        # gán giá trị của y1 cho yy1 thông qua best_idxes
        torch.index_select(y1, dim=0, index=best_idxes, out=yy1)
        # gán giá trị của x2 cho xx2 thông qua best_idxes
        torch.index_select(x2, dim=0, index=best_idxes, out=xx2)
        # gán giá trị của y2 cho yy2 thông qua best_idxes
        torch.index_select(y2, dim=0, index=best_idxes, out=yy2)
        
        # Tính toán tọa độ của intersection theo id để tính intersection
        # if xx1 < x1[id] -> xx1 = x1[id] (e.g: xx1 = [2,3], x1[id]=3 -> xx1=[3,3])
        xx1 = torch.clamp(xx1, min=x1[id]) 
        # if yy1 < y1[id] -> yy1 = y1[id]
        yy1 = torch.clamp(yy1, min=y1[id])
        # if xx2 > x2[id] -> xx2 = x2[id]
        xx2 = torch.clamp(xx2, max=x2[id])
        # if yy2 > y2[id] -> yy2 = y2[id]
        yy2 = torch.clamp(yy2, max=y2[id])
        
        # Resize ww, hh to the same xx2 and yy2
        ww.resize_as_(xx2) # (<200, )
        hh.resize_as_(yy2) # (<200, )
        
        # Calculate width, height of intersection
        ww = xx2 - xx1 # (<200, )
        hh = yy2 - yy1 # (<200, )
        
        # Keep range
        ww = torch.clamp(ww, min=0.0) # (<200, )
        hh = torch.clamp(hh, min=0.0) # (<200, )
        
        # Intersection
        inter = ww*hh # (<200, )
        
        # Lấy ra diện tích của < 200 boxes có score cao nhất ngoại trừ id đang xét
        other_areas = torch.index_select(areas, 0, best_idxes) # (<200, )
        
        # Union areas
        union = areas[id] + other_areas - inter # (<200, )
        
        # IoU
        iou = inter/union # (<200, )
        
        # Giữ lại indexes mà có overlap < 0.45
        best_idxes = best_idxes[iou.le(overlap)] # (<200, )
        
    return keep, count

class Detect(Function):
    def __init__(self, conf_threshold=0.01, top_k=200, nms_threshold=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
    
    def forward(self, loc_data, conf_data, dboxes):
        """Khi inference

        Args:
            loc_data (batch_size, 8732, 4): output của loc model
            conf_data (batch_size, 8732, 21): output của conf model
            dboxes (8732, 4): default boxes
        Return:
        
        """
        # Thông tin size
        batch_size = loc_data.size(0)
        num_dboxes = loc_data.size(1) # 8732
        num_classes = conf_data.size(2) # 21
        
        # Confidence dự đoán khi sử dụng softmax function
        conf_preds = self.softmax(conf_data) # (batch_size, 8732, 21)
        conf_preds = conf_preds.transpose(2,1) # (batch_size, 21, 8732)
        
        # Output: số lượng batch_size (batch_size)
        # trong mỗi batch sẽ có bao nhiêu num_classes (num_classes)
        # trong mỗi class sẽ có bao nhiêu boxes(self.top_k)
        # tọa độ mỗi boxes và conf (5)
        output = torch.zeros(batch_size, num_classes, self.top_k, 5)
        
        for i in range(batch_size):
            # decode boxes của image thứ i
            # loc_data[i] là output của image thứ i từ loc model
            decode_boxes = decode(loc_data[i], dboxes) # (8732, 4)
            # clone conf_preds của image thứ i
            conf_scores = conf_preds[i].clone() # (21, 8732)
            
            # Bỏ qua background class (0)
            # Với mỗi class, đầu tiên lấy ra những boxes có confidence > 0.01
            # tức box nào confidence <0.01 thì bỏ
            for c in range(1, num_classes):
                # bỏ qua boxes có confidence < 0.01
                # conf_scores[c]: (8732)
                c_mask = conf_scores[c].gt(self.conf_threshold) # (8732) [True, False,...,True]
                # scores: (số scores có confidence > 0.01)
                scores = conf_scores[c][c_mask] # (<8732,)
                
                # nếu toàn bộ scores < 0.01 -> hủy
                if scores.size(0) == 0:
                    continue
                
                """
                    expand_as (3,4): [[1],[2],[3]]
                    => [[ 1,  1,  1,  1],
                        [ 2,  2,  2,  2],
                        [ 3,  3,  3,  3]]
                    l_mask: 8732 -> 8732,1 -> 8732,4
                    e.g:[[true],[true],[false],[true]] 
                    -> [[true,true,true,true],
                    [true,true,true,true],
                    [false,false,false,false],
                    [true,true,true,true]]
                """
                # l_mask: (8732,) -> (8732,1) -> (8732,4)
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes)
                boxes = decode_boxes[l_mask].view(-1,4) # (số boxes có confidence > 0.01, 4)
                ids, count = nms(boxes, scores, self.nms_threshold, self.top_k)
                # lấy ra scores và boxes cho class c
                # vd trong ảnh với class=person thì có só lượng là count
                # ids là vị trí của họ
                output[i, c, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),boxes[ids[:count]]),dim=1)
        return output
            
            
        
        
        
        
        
        
        
        
        
        
        
    
    
    