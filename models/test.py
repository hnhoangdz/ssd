import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
# a = torch.Tensor([[[[1, 2],
#           [3, 4]],

#          [[5, 6],
#           [7, 8]],

#          [[9, 10],
#           [11, 12]]],


#         [[[13, 14],
#           [15, 16]],

#          [[17, 18],
#           [19, 20]],

#          [[21, 22],
#           [23, 24]]]])
# print(a.shape)
# x = nn.Parameter(torch.Tensor(3))
# print(x)
# out = x.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(a)
# norm = a.sum(dim=1)
# print(a)
# print('----------------------------')
# print(out)
# import itertools
# cfg = {
#     "num_classes": 21, #VOC data include 20 class + 1 background class
#     "input_size": 300, #SSD300
#     "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
#     "feature_maps": [38, 19, 10, 5, 3, 1],
#     "steps": [8, 16, 32, 64, 100, 300], # Size of default box
#     "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
#     "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
#     "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
# }
# count = 0
# for k, f in enumerate(cfg['feature_maps']):
#     for i,j in itertools.product(range(f), repeat=2):
#         count += 1
#         # print(i,j)
#     break

# print(count)

# for a in [2]:
#     print(a)

# x = torch.randn(200)
# v,i = x.sort(0)
# best_idxes = i[-10:]
# a = x.new()

# while len(best_idxes) > 0:
#     id = best_idxes[-1]
#     print(id)
#     if best_idxes.size(0) == 1:
#         print('there is only one box here !!!')
#         break
#     best_idxes = best_idxes[:-1] 
#     torch.index_select(x, dim=0, index=best_idxes, out=a)
#     print(a)
#     a = torch.clamp(a, min=x[id])
#     print(a.shape)
#     print('---------------')
#     break
# w = x.new()
# i = torch.tensor([153])
# a = torch.tensor([1.6701, 1.7361, 1.7664, 1.7860, 1.8149, 1.8927, 1.9435, 2.0867, 2.4640])
# a = torch.clamp(a, min=x[i])
# print(x[i])
# print(a)
# print(w)
# w.resize_as_(a)
# print(w)
# print(torch.__version__)
# m = nn.Softmax(dim=0)

# x = torch.randint(1,5,size=(5,), dtype=torch.float32)
# print(x.dtype)
# print(m(x))
# decode_boxes = torch.randn((8732,4))
# print(decode_boxes)
# x = torch.rand((4, 21, 8732), dtype=torch.float16)
# c_mask = x[1][1].gt(0.01)
# a = c_mask.unsqueeze(1).expand_as(decode_boxes)
# print(a)
# print(c_mask.shape, a.shape)

x = torch.Tensor(([[1]]))
y = torch.Tensor(([[1]]))
z = torch.cat((x,y),dim=0)
t = torch.cat((x,y),dim=1)
print(x.shape)
print(z)
print(t)