from glob import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *

class ImageDataset(Dataset):
    def __init__(self, augment=None, img_dir=None):
        self.img_dir = img_dir
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.augment = augment
        self.items = []

        if img_dir == None:
            return
        for label,name in enumerate(os.listdir(img_dir)):
            subdir = os.path.join(img_dir, name)
            for image in glob(os.path.join(subdir, '**', '*'), recursive=True):
                if os.path.isfile(image):
                    self.items.append({'image':image, 'label':label, 'name':name})

        print(f'Created dataset with {len(self)} examples')

    def preprocess(self, pil_image, size=(224,224)):
        pil_image = pil_image.resize(size).convert('RGB')
        if self.augment:
            image_array = self.augment(pil_image)
        image_array = self.normalize(pil_image)
        if image_array.max() > 1:
            image_array = image_array / 255
        return image_array

    def __getitem__(self, i):
        item = self.items[i]
        pil_image = pil_open(item['image'])
        image = self.preprocess(pil_image)
        item['image'] = image
        return item

    def __len__(self):
        return len(self.items)

    def append(self, item):
        self.items.append([item[0], item[1], item[2]])
