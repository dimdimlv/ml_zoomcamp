from io import BytesIO
from urllib import request
import numpy as np
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    # ToTensor: [0, 255] -> [0, 1] and HWC -> CHW
    x = np.array(x, dtype='float32') / 255.0
    x = np.transpose(x, (2, 0, 1))
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    
    # (3, H, W) - (3,) -> broadcasting? 
    # numpy broadcasting matches last dimensions. 
    # x is (3, 200, 200). mean is (3,).
    # We need to reshape mean/std to (3, 1, 1) to broadcast correctly over H, W.
    
    x = (x - mean[:, None, None]) / std[:, None, None]
    return x

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
target_size = (200, 200)

img = download_image(url)
img = prepare_image(img, target_size)
x = preprocess_input(img)

print(f"R channel first pixel: {x[0, 0, 0]}")
