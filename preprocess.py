import argparse
from glob import glob
import numpy as np
import os
from PIL import Image
import random
import shutil
import torch
from torchvision import transforms
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Perform various pre-processing tasks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--augment', type=int, default=0, help='Perform augments on the data', dest='augment')
    parser.add_argument('-c', '--count', action='store_const', const=count, default=None, help='Count files in directories', dest='count')
    parser.add_argument('-v', '--visualize', action='store_const', const=augment_vis, default=None, help='Visualize results of augmentation', dest='augment_vis')
    parser.add_argument('-d', '--directory', type=str, help='The directory to perform the operation', dest='dir')
    return parser.parse_args()

def augment(size, dir):
    assert dir != None, 'Augment requires directory argument'
    labels = [label for label in os.listdir(dir)]

    train_augment = transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.RandomAffine(degrees=10),
        transforms.GaussianBlur(5),
        transforms.RandomPerspective(distortion_scale=0.2)
    ])

    for label in labels:
        subdir = os.path.join(dir, label)
        output = os.path.join(subdir, 'augmented')
        if os.path.isdir(output):
            shutil.rmtree(output)

        images = glob(os.path.join(subdir, '*'))
        num_images = len(images)
        if num_images >= size:
            print(f'\'{subdir}\' already has enough images ({num_images})')
            continue

        os.makedirs(output)
        for i in tqdm(range(size - num_images)):
            img_name = random.choice(images)
            image = Image.open(img_name)
            transformed = train_augment(image)
            transformed = transformed.resize((224, 224))
            transformed.save(os.path.join(output, f'{i}_{img_name.split(os.sep)[-1]}'))

def augment_vis(dir):
    assert dir != None, 'Augmentation visualization requires directory argument'
    output = os.path.join(dir, 'test')
    if not os.path.isdir(output):
        os.makedirs(output)
    train_augment = transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.RandomAffine(degrees=10),
        transforms.GaussianBlur(5),
        transforms.RandomPerspective(distortion_scale=0.2)
    ])
    images = glob(os.path.join(dir, '*'))
    for i in range(20):
        img_name = random.choice(images)
        image = Image.open(img_name)
        transformed = train_augment(image)
        transformed.save(os.path.join(output, img_name.split(os.sep)[-1]))


def count(dir):
    assert dir != None, 'Count requires directory argument'
    counts = []
    labels = [label for label in os.listdir(dir)]
    for label in labels:
        subdir = os.path.join(dir, label)
        count = len([file for file in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, file))])
        counts.append(count)
    weights = np.array(counts) / sum(counts) * 100
    weights = 1 / weights
    weights = np.round(weights, decimals=3)

    print(f'Counts: {counts}')
    print(f'Weights: {weights}')

if __name__ == '__main__':
    args = get_args()
    if args.augment > 0:
        augment(args.augment, args.dir)
    if args.count:
        args.count(args.dir)
    if args.augment_vis:
        args.augment_vis(args.dir)
