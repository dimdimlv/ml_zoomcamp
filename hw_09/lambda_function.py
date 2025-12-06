import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO
from urllib import request

# Initialize model
model_path = 'hair_classifier_empty.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

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

def predict(url):
    img = download_image(url)
    target_size = (200, 200)
    img = prepare_image(img, target_size)
    x = preprocess_input(img)
    x = x[None, ...] # Batch dimension
    
    outputs = session.run([output_name], {input_name: x})
    return float(outputs[0][0][0])

def lambda_handler(event, context):
    url = event.get('url')
    result = predict(url)
    return result
