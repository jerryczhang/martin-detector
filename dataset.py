from glob import glob
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.items = []

        for i,label in enumerate(os.listdir(img_dir)):
            subdir = f'{img_dir}/{label}'
            for image in os.listdir(subdir):
                self.items.append([f'{subdir}/{image}', i, label])

    def preprocess(self, pil_image, size=(224,224)):
        pil_image = pil_image.resize(size)
        image_array = np.array(pil_image)
        image_array = self.transform(image_array.transpose((2,0,1)))
        if image_array.max() > 1:
            image_array = image_array / 255
        return image_array

    def __getitem__(self, i):
        item = self.items[i]
        pil_image = Image.open(item[0])
        image = self.preprocess(pil_image)
        return [image, item[1], item[2]]

    def __len__(self):
        return len(self.items)
