import argparse
import Augmentor
import numpy as np
import os
import shutil
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Perform various pre-processing tasks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--augment', type=int, default=0, help='Perform augments on the data', dest='augment')
    parser.add_argument('-c', '--count', action='store_const', const=count, default=None, help='Count files in directories', dest='count')
    parser.add_argument('-d', '--directory', type=str, help='The directory to perform the operation', dest='dir')
    return parser.parse_args()

def augment(size, dir):
    assert dir != None, 'Augment requires directory argument'
    labels = [label for label in os.listdir(dir)]
    for label in labels:
        current_dir = os.path.join(dir, label)
        output = os.path.join(current_dir, 'output')
        if os.path.isdir(output):
            shutil.rmtree(output)

        num_images = len([file for file in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, file))])
        if num_images >= size:
            continue

        p = Augmentor.Pipeline(current_dir)
        p.rotate(probability=0.8, max_left_rotation=20, max_right_rotation=20)
        p.random_brightness(probability=0.8, min_factor=0.4, max_factor=1.6)
        p.random_contrast(probability=0.8, min_factor=0.4, max_factor=1.6)
        p.random_color(probability=0.8, min_factor=0.4, max_factor=1.6)

        p.sample(size - num_images)

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
