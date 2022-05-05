import xml.etree.ElementTree as ET 
import numpy as np
np.random.seed(1234)

"""
    This file is to generate annotions of images 
    E.g: [[xmin, ymin, xmax, ymax, class_id]]
"""

class Anno_xml(object):
    def __init__(self, class_names):
        self.class_names = class_names
    
    def __call__(self, xml_path, width, height):
        annotations = []
        xml = ET.parse(xml_path).getroot()

        # find class_name, bounding box in each object
        for obj in xml.iter('object'):

            # if this image is too hard to learn -> skip
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            
            bndbox = []
            # class_name of object
            name = obj.find('name').text.lower().strip()
            # bounding box of object
            bbox_obj = obj.find('bndbox')

            coordinates = ['xmin','ymin','xmax','ymax']
            # get all coordinates value of object
            for coord_name in coordinates:
                # find coordinates of object
                # orginal coordinates (1,1) -> (0,0)
                coord_value = int(bbox_obj.find(coord_name).text.strip()) - 1
                if coord_name == 'xmin' or coord_name == 'xmax':
                    coord_value /= width
                else:
                    coord_value /= height
                bndbox.append(coord_value)

            class_id = self.class_names.index(name)
            bndbox.append(class_id)
            annotations.append(bndbox) # [[xmin,ymin,xmax,ymax,class_id], [...], ...]

        return np.array(annotations)

if __name__ == '__main__':
    from make_datapath import make_datapath_list
    import cv2
    class_names = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
                    "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant",
                    "sheep", "sofa", "train", "tvmonitor"]

    root_path = '../dataset/VOCdevkit/VOC2012/'
    train_img_paths, train_anno_paths, val_img_paths, val_anno_paths = make_datapath_list(root_path)
    test_img = train_img_paths[0]
    test_anno = train_anno_paths[0]
    print('test img path: ', test_img)
    print('test anno path: ', test_anno)
    img = cv2.imread(test_img)
    h,w,c = img.shape

    xml_obj = Anno_xml(class_names)

    annotations = xml_obj(test_anno,w,h)
    print(annotations)