import numpy as np
from PIL import Image
import os

def load_data(image_dir, size):

    gray_dir  = f'{image_dir}/gray'
    color_dir = f'{image_dir}/color'

    img_size  = (size, size)

    gray  = []
    color = []

    for name in sorted(os.listdir(gray_dir)):
        file_path = os.path.join(gray_dir, name)
        img = Image.open(file_path)
        #img = img.convert('RGB')
        img = img.resize(img_size)
        img = np.array(img)/255
        gray.append(img)

    for name in sorted(os.listdir(color_dir)):
        file_path = os.path.join(color_dir, name)
        img = Image.open(file_path)
        img = img.convert('RGB')
        img = img.resize(img_size)
        img = np.array(img)/255
        color.append(img)

    return gray, color