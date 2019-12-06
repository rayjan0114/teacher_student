import glob
import math
import os
import copy
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

# from utils import image_files_in_folder


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
# from utils.utils import xyxy2xywh, xywh2xyxy
class LoadImagesAndLabels(Dataset):
    
    def __init__(self, path, img_size=112, batch_size=16, augment=False, hyp=None, rect=True, image_weights=False,
                    cache_labels=False, cache_images=False):
        path = str(Path(path))
        
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        with open(path, 'r') as f:
            # self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
            #                     if os.path.splitext(x)[-1].lower() in img_formats]
            self.img_files = [x for x in f.read().splitlines()  # os-agnostic
                                if os.path.splitext(x)[-1].lower() in img_formats]
        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % path

        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
        self.n = n

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        label_path = label_path.replace('\\\\','\\')
        hyp = self.hyp
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'Image Not Found ' + img_path

        flip_aug = True
        if flip_aug and random.random() > 0.5:
            img = cv2.flip(img,1)
              

        hsv_aug = True
        if hsv_aug:
            img = img_random(img)

        # r = self.img_size / max(img.shape)  # size ratio
        # if self.augment and r < 1:  # if training (NOT testing), downsize to inference shape
        #     h, w, _ = img.shape
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR fastest
        
        img = img[...,::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img = (img - 125)/255
        try:
            with open(label_path, 'r') as f:
                labels_out = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        except Exception as e:
            labels_out = []
            # print('label not exist')  TODO   
        labels_out = torch.tensor(labels_out).reshape(-1)
        return torch.from_numpy(img), labels_out
    
    @staticmethod
    def collate_fn(batch):
        batch = [(img, label) for (img, label) in batch 
                                if img is not None and label.shape[0] != 0]
        if len(batch) == 0:
            return None,None
        # img, label = list(zip(*batch))
        # print('='*100)
        # return torch.stack(img, 0), torch.stack(label, 0)
        return default_collate(batch)

def img_random(img_bgr):
    img = copy.deepcopy(img_bgr)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
    S = img_hsv[:, :, 1].astype(np.float32)  # saturation
    V = img_hsv[:, :, 2].astype(np.float32)  # value
    hsv_s = 0.5703 / 2
    hsv_v = 0.3174 / 2
    a = random.uniform(-1, 1) * hsv_s + 1
    b = random.uniform(-1, 1) * hsv_v + 1

    S *= a
    V *= b

    img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
    img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    new_size = (int(b * img.shape[1])), (int(b * img.shape[0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    return img


if __name__ == "__main__":
    # dataloader = LoadImagesAndLabels('./data/test3_dir.txt')
    # print(next(iter(dataloader))[1]) 
    hyp = {}
    
    dataset = LoadImagesAndLabels('./data/test3_dir.txt',
                                  img_size=112,
                                  batch_size=8,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  image_weights=False,
                                  cache_labels= False,
                                  cache_images= False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=8,
                                             num_workers=min([os.cpu_count(), 2, 8]),
                                             shuffle=True,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

 