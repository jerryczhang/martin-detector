from glob import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import *

class ImageDataset(Dataset):
    def __init__(self, transform, img_dir=None):
        self.img_dir = img_dir
        self.transform = transform
        self.items = []

        if img_dir == None:
            return
        for i,label in enumerate(os.listdir(img_dir)):
            subdir = os.path.join(img_dir, label)
            for image in glob(os.path.join(subdir, '**', '*'), recursive=True):
                if os.path.isfile(image):
                    self.items.append([image, i, label])


        print(f'Created dataset with {len(self)} examples')

    def preprocess(self, pil_image, size=(224,224)):
        pil_image = pil_image.resize(size).convert('RGB')
        image_array = np.array(pil_image)
        image_array = self.transform(image_array)
        if image_array.max() > 1:
            image_array = image_array / 255
        return image_array

    def __getitem__(self, i):
        item = self.items[i]
        pil_image = pil_open(item[0])
        image = self.preprocess(pil_image)
        return [image, item[1], item[2]]

    def __len__(self):
        return len(self.items)

    def append(self, item):
        self.items.append([item[0], item[1], item[2]])
