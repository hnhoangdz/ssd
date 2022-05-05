import os

"""
    This file is to create paths of training and validation dataset.
    Each of them includes: paths of images and corresponding annotations
"""


def make_datapath_list(root_path):

    # All file paths: images (.jpg), annotations(.xml)
    image_paths = os.path.join(root_path,'JPEGImages','%s.jpg')
    annotation_paths = os.path.join(root_path,'Annotations','%s.xml')
    
    # File names of trainning
    train_name_paths = os.path.join(root_path, 'ImageSets/Main/train.txt')
    train_img_paths = []
    train_anno_paths = []

    for name in open(train_name_paths):
        img_name = name.strip()
        img_path = image_paths % img_name
        anno_path = annotation_paths % img_name
        
        train_img_paths.append(img_path)
        train_anno_paths.append(anno_path)

    # File names of validation
    val_name_paths = os.path.join(root_path, 'ImageSets/Main/val.txt')
    val_img_paths = []
    val_anno_paths = []

    for name in open(val_name_paths):
        img_name = name.strip()
        img_path = image_paths % img_name
        anno_path = annotation_paths % img_name

        val_img_paths.append(img_path)
        val_anno_paths.append(anno_path)

    return train_img_paths, train_anno_paths, val_img_paths, val_anno_paths
    

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    root_path = '../dataset/VOCdevkit/VOC2012/'
    train_img_paths, train_anno_paths, val_img_paths, val_anno_paths = make_datapath_list(root_path)
    print(train_img_paths[:2])
    img = cv2.imread(train_img_paths[0])
    image = img[:, :, (2,1,0)]
    print(img.shape)
    print(image.shape)
    plt.imshow(image)
    plt.show()
    plt.imshow(img)
    plt.show()
    # image1 = image[:, ::-1, :]
    # plt.imshow(image1)
    # plt.show()
    # height, width, depth = image.shape
    # a = np.array([[[1,2,3],
    #                 [4,5,6]]])
    # print(a.shape)
    # print(height, width)
    # ratio = random.uniform(1, 4)
    # print(ratio)
    # left = random.uniform(0, width*ratio - width)
    # print(left)
    # top = random.uniform(0, height*ratio - height)
    # print(top)
    # expand_image = np.zeros(
    #     (int(height*ratio), int(width*ratio), depth),
    #     dtype=image.dtype)
    # expand_image[:, :, :] = (104, 117, 123)
    # expand_image[int(top):int(top + height),
    #                 int(left):int(left + width)] = image
    # image = expand_image