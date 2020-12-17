import cv2
from PIL import Image
from glob import glob
import os

def get_face(pil_image, cascade):
    return pil_image

def gen_faces(srcdir, dest, cascade):
    if not os.path.exists(dest):
        os.makedirs(dest)
    images = glob(f'{srcdir}/*.*', recursive=True)
    for image in images:
        pil_image = Image.open(image)
        face_image = get_face(pil_image)
        face_image.save(f'{dest}/name)')

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gen_faces('images/raw', 'images/cropped', cascade)

