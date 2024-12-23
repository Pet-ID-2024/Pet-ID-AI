""" 
@Author: LYS
@Date: 24. 7. 11.
"""
import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image_path, input_size):
    image = Image.open(image_path)
    image = image.resize(input_size)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def load_labels(label_path):
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return [label.strip() for label in labels]

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/home/leeyongseong/Downloads/dogs2_float32_metadata.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the input image
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])
image = preprocess_image("/home/leeyongseong/Downloads/a32226d73f7d42890291fd0e06b96f44.jpg", input_size)

print(image.shape)

# Run the inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Process the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

# labels = load_labels("labels.txt")
# top_k = 5

# top_k_indices = np.argsort(output_data[0])[-top_k:][::-1]

# for i in top_k_indices:
#     print(f"Label: {labels[i]}, Probability: {output_data[0][i]}")

# onnx2tf -i dogs2.onnx - output_dir     <- onnx to tflite