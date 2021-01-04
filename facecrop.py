import cv2
from PIL import Image
from glob import glob
import os
import numpy as np

from utils import *

def get_face_images(srcdir, dest, min_neighbors):
    if not os.path.exists(dest):
        os.makedirs(dest)
    images = glob(f'{srcdir}/*.*')
    for image in images:
        pil_image = pil_open(image)
        face_images = pull_faces(pil_image, min_neighbors)
        for i,face_image in enumerate(face_images):
            name = image.split(os.sep)[-1]
            face_image.save(f'{dest}/{i}{name}')

get_face_images('images/raw', 'images/faces', min_neighbors=1)

