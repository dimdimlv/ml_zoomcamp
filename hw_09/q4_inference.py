import onnxruntime as ort
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
    x = np.array(x, dtype='float32') / 255.0
    x = np.transpose(x, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    x = (x - mean[:, None, None]) / std[:, None, None]
    return x

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
target_size = (200, 200)

img = download_image(url)
img = prepare_image(img, target_size)
x = preprocess_input(img)

# Add batch dim
x = x[None, ...] # (1, 3, 200, 200)

model_path = 'hair_classifier_v1.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

outputs = session.run([output_name], {input_name: x})
print(f"Model output: {outputs[0][0][0]}")
