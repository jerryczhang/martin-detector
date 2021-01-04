import cv2
from PIL import Image
import numpy as np

def pil_open(image_str):
    pil_image = Image.open(image_str)
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
    exif = pil_image._getexif()
    if exif:
        transform = transforms[exif[0x0112]]
        for t in transform:
            pil_image.transpose(method=t)
        return pil_image
    return pil_image

def pull_faces(pil_image, min_neighbors):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    array = np.array(pil_image)
    cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    faces = cascade.detectMultiScale(cv_image, minNeighbors=min_neighbors, scaleFactor=1.1, minSize=(224,224))
    pil_faces = []
    for (x1,y1,w,h) in faces:
        s = max(w,h)
        x2 = x1 + s
        y2 = y1 + s
        pil_faces.append(Image.fromarray(array[y1:y2,x1:x2,:]))
    return pil_faces
