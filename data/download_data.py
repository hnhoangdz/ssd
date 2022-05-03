import os
import urllib.request
import zipfile
import tarfile

data_dir = '../dataset'
if not os.path.exists(data_dir):
    print('Creating data folder')
    os.mkdir(data_dir)

url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
target_path = os.path.join(data_dir, 'VOCtrainval_11-May-2012.tar')

if not os.path.exists(target_path):
    print('Downloading data')
    # download via url to target path   
    urllib.request.urlretrieve(url, target_path)

    # read data file
    tar = tarfile.TarFile(target_path)
    # extract data
    tar.extractall(data_dir)
    print('Done download data')
    tar.close()