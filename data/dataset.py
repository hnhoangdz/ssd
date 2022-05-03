from random import shuffle
from lib import *
from transform import DataTransform
from extract_xml import Anno_xml
from make_datapath import make_datapath_list

class MyDataset(data.Dataset):
    def __init__(self, img_list_path, anno_list_path, phase, transform, anno_xml):
        self.img_list_path = img_list_path
        self.anno_list_path = anno_list_path
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml
    
    def __len__(self):
        return len(self.img_list_path)

    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)
        return img, gt

    def pull_item(self, index):
        # get image information
        img_file_path = self.img_list_path[index]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        # get anno information
        anno_file_path = self.anno_list_path[index]
        anno_info = self.anno_xml(anno_file_path, width, height)

        # transform
        img, boxes, labels = self.transform(img, self.phase, anno_info[:,:4],anno_info[:,4:])

        # convert BGR -> RGB 
        # h,w,c -> c,h,w
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        # ground truth
        gt = np.hstack((boxes, labels))

        return img, gt, height, width

def collate_fn(batch):
    imgs = []
    targets = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets
if __name__ == '__main__':
    class_names = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
                    "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant",
                    "sheep", "sofa", "train", "tvmonitor"]
    root_path = '../dataset/VOCdevkit/VOC2012/'
    train_img_paths, train_anno_paths, val_img_paths, val_anno_paths = make_datapath_list(root_path)

    input_size = 300
    color_mean = (104, 117, 123)
    batch_size = 4
    transform = DataTransform(input_size,color_mean)
    anno_xml = Anno_xml(class_names)

    train_dataset = MyDataset(train_img_paths,train_anno_paths,'train',transform,anno_xml)
    val_dataset = MyDataset(val_img_paths, val_anno_paths,'val',transform,anno_xml)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    dataloader_dict = {
        'train':train_dataloader,
        'val':val_dataloader
    }
    batch_iter = iter(dataloader_dict['val'])
    images, targets = next(batch_iter)
    print(images.size())
    print(len(targets))
    print(targets[0].size())


        