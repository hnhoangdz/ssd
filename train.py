# Data-processing
from data.dataset import MyDataset,collate_fn
from data.make_datapath import make_datapath_list
from data.transform import *
from data.extract_xml import Anno_xml

# Base models
from models.ssd import SSD
from models.multibox_loss import MultiBoxLoss

import cv2
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

torch.manual_seed(1234)
from cfg import *
from tqdm import tqdm
import time 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device using: ', device)
torch.backends.cudnn.benchmark = True

# 1. Create dataloader
# 2. Create ssd300
# 3. Create loss function
# 4. Create optimizer
# 5. Training, eval 

# 1. Create dataloader
# Load paths images and annotations for training, validation dataset
root_path = 'dataset/VOCdevkit/VOC2012'
train_img_paths, train_anno_paths, val_img_paths, val_anno_paths = make_datapath_list(root_path=root_path)

transform = DataTransform(input_size, color_mean)
anno = Anno_xml(class_names)

# Read file and transform data
train_dataset = MyDataset(train_img_paths, train_anno_paths, phase='train', transform=transform, anno_xml=anno)
val_dataset = MyDataset(val_img_paths, val_anno_paths, phase='val', transform=transform, anno_xml=anno) 

# data loader
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
dataloader_dict = {'train': train_dataloader, 
                    'val': val_dataloader}

# 2. Create ssd300
model = SSD(phase='train', cfg=cfg) # load ssd300 model
vgg_weights = torch.load('dataset/vgg16-weights-imagenet/vgg16_reducedfc.pth') # load weights vgg16 trained on imagenet
model.vgg.load_state_dict(vgg_weights) # embedd vgg weights into main model

# He-Initialization
def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight.data)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0.0)

# Apply initialization to main layers
model.extras.apply(init_weights)
model.loc.apply(init_weights)
model.conf.apply(init_weights)

# 3. Create loss function
# Loss function
criterion = MultiBoxLoss(jaccard_threshold=jaccard_threshold, negpos_ratio=negpos_ratio, device=device)

# Optimizer (baseline)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# 4. Training, evaluation
def train_model(model, criterion, optimizer, num_epochs):
    # Move model to CPU/GPU
    model.to(device)
    
    iterations = 1
    train_loss_epoch = 0.0
    val_loss_epoch = 0.0
    logs = []
    
    for epoch in tqdm(range(num_epochs+1)):
        t_start_epoch = time.time()
        t_start_iter = time.time()
        
        print('---'*20)
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
                print('Start training')
            else:
                if (epoch+1)%10 == 0:
                    model.eval()
                    print('---'*20)
                    print('Start evaluating')
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                # Move data to GPU/CPU
                images = images.to(device)
                targets = [anno.to(device) for anno in targets]
                
                # init optimizer
                optimizer.zero_grad()
                
                # forward
                with torch.set_grad_enabled(phase=='train'):
                    # predictions of training
                    outputs = model(images)
                    
                    # calculate loss value
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c
                    
                    if phase=='train':
                        loss.backward() # backpropagation
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.5) # clip value of parameters
                        optimizer.step() # update parameters
                    
                        if iterations%10==0:
                            t_iter_end = time.time()
                            iter_duration = t_iter_end-t_start_iter
                            print('iteration: {} || train_loss: {} || duration 10iters: {} seconds'.format(iterations, round(loss.item(),4), round(iter_duration,4)))
                            t_start_iter = time.time()
                        train_loss_epoch += loss.item()
                        iterations += 1
                    else:
                        print('val_loss here: ',loss.item())
                        val_loss_epoch += loss.item()
        
        t_end_epoch = time.time()
        
        print("---"*20)
        print('epoch: {} || train_loss: {}'.format(epoch+1, round(train_loss_epoch,4)))
        if (epoch+1)%10 == 0:
          print('epoch: {} || val_loss: {}'.format(epoch+1, round(val_loss_epoch,4)))
        print('duration: {} seconds'.format(round(t_end_epoch-t_start_epoch,4)))                
        t_start_epoch = time.time()
        
        log_epoch = {'epoch': epoch+1, 'train_loss': train_loss_epoch, 'val_loss': val_loss_epoch}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv('weights/ssd_logs.cvs')
        train_loss_epoch = 0.0
        val_loss_epoch = 0.0
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(),'weights/ssd300_' + str(epoch+1) + '.pth')

train_model(model, criterion, optimizer, num_epochs)