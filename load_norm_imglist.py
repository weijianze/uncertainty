import torch.utils.data as data
from PIL import Image
from PIL import ImageFilter
import os
import os.path
import pdb

def default_loader(path):
    img = Image.open(path).convert('L')
    # img = img.filter(ImageFilter.MedianFilter(size=3))  # the normalized iris images from CASIA website have been filtered using median filter
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


def multi_list_reader(root, fileList):
    imgList = []
    clsList = []
    if len(root)==len(fileList):
        for i in range(len(root)):
            cls_max = 0
            with open(fileList[i], 'r') as file:
                for line in file.readlines():
                    imgPath, label = line.strip().split(' ')
                    label = int(label)
                    if label>cls_max:
                        cls_max = label
                    imgList.append((root[i], imgPath, sum(clsList)+label))
                    # imgList.append((os.path.join(root[i], imgPath), int(label)))
            clsList.append(cls_max+1)
    else:
        print('!!!!!')
    return imgList


class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=multi_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(root,fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        img_root, imgPath, target = self.imgList[index]
        # load image and filter it using median filter (local step)
        img = self.loader(os.path.join(img_root, imgPath))
    
        if self.transform is not None:
            img = self.transform(img)
        
        # global step
        img_mean = img.mean()
        img_std = img.std()+1e-8
        img = (img-img_mean)/img_std
        
        return img, target

    def __len__(self):
        return len(self.imgList)
