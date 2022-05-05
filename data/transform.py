from data.agumentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans
    
class DataTransform(object):
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(), # convert to float32
                ToAbsoluteCoords(), # convert annotations back to normal size
                PhotometricDistort(), # change value of image randomly to learn easiser
                Expand(color_mean), # expand size of image
                RandomSampleCrop(), # random crop image
                RandomMirror(), # Rotate vertical of image
                ToPercentCoords(), # convert annotations back to normalize size
                Resize(input_size),
                SubtractMeans(color_mean) # Subtract mean của BGR
            ]),
            'val': Compose([
                ConvertFromInts(), # convert to float32
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
    
    def __call__(self, image, phase, boxes, labels):
        return self.data_transform[phase](image, boxes, labels)

if __name__ == '__main__':
    from make_datapath import make_datapath_list
    from extract_xml import Anno_xml
    import cv2
    import matplotlib.pyplot as plt
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    root_path = '../dataset/VOCdevkit/VOC2012/'

    train_img_paths, train_anno_paths, val_img_paths, val_anno_paths = make_datapath_list(root_path)
    img_path_train = train_img_paths[0]
    anno_path_train = train_anno_paths[0]
    print(img_path_train)
    print(anno_path_train)
    img = cv2.imread(img_path_train)
    h,w,c = img.shape

    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    trans_anno = Anno_xml(classes)
    anno_info_train = trans_anno(anno_path_train, w, h)
    print(anno_info_train)
    boxes = anno_info_train[:, :4]
    labels = anno_info_train[:, 4:]

    transform = DataTransform(input_size, color_mean)
    results = transform(img, 'train', boxes, labels)
    cv2.rectangle(results[0], boxes, end_point, color, thickness)
    print(results[0].shape)
    plt.imshow(results[0])
    plt.show()


