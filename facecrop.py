import cv2
from PIL import Image
from glob import glob
import os

def pull_faces(pil_image, cascade):
    return [pil_image]

def get_face_images(srcdir, dest, cascade):
    if not os.path.exists(dest):
        os.makedirs(dest)
    images = glob(f'{srcdir}/*.*')
    for image in images:
        pil_image = Image.open(image)
        face_images = pull_faces(pil_image, cascade)
        for face_image in face_images:
            name = image.split(os.sep)[-1]
            face_image.save(f'{dest}/{name}')

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
get_face_images('images/raw/alex', 'images/cropped', cascade)

