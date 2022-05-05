class_names = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
                "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor"]
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
input_size = 300
color_mean = (104, 117, 123)
batch_size = 32  
jaccard_threshold=0.5
negpos_ratio=3
num_epochs = 100