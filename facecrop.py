import cv2
from PIL import Image
from glob import glob
import os
import numpy as np

def pull_faces(pil_image, cascade):
    array = np.array(pil_image)
    cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    faces = cascade.detectMultiScale(cv_image, minNeighbors=6, scaleFactor=1.1, minSize=(224,224))
    pil_faces = []
    for (x1,y1,w,h) in faces:
        s = max(w,h)
        x2 = x1 + s
        y2 = y1 + s
        pil_faces.append(Image.fromarray(array[y1:y2,x1:x2,:]))
    return pil_faces

def get_face_images(srcdir, dest, cascade):
    if not os.path.exists(dest):
        os.makedirs(dest)
    images = glob(f'{srcdir}/*.*')
    for image in images:
        pil_image = Image.open(image)
        pil_image = autorotate(pil_image)
        face_images = pull_faces(pil_image, cascade)
        for i,face_image in enumerate(face_images):
            name = image.split(os.sep)[-1]
            face_image.save(f'{dest}/{i}{name}')

def autorotate(pil_image):
    transforms = [
        [],
        [],
        [Image.FLIP_LEFT_RIGHT],
        [Image.ROTATE_180],
        [Image.FLIP_TOP_BOTTOM],
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
        [Image.ROTATE_270],
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],
        [Image.ROTATE_90],
    ]
    try:
        transform = transforms[pil_image._getexif()[0x0112]]
    except Exception:
        return pil_image
    else:
        for t in transform:
            pil_image.transpose(method=t)
        return pil_image

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
get_face_images('images/raw', 'images/faces', cascade)

