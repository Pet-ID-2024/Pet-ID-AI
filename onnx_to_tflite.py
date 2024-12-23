# """
# @Author: LYS
# @Date: 24. 7. 7.
# """
#
# import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf
#
# import numpy as np
# from PIL import Image
# import glob
#
# # ONNX 모델 로드
# onnx_model = onnx.load('dogs2.onnx')
#
# # TensorFlow 모델로 변환
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph('model.pb')
#
# def representative_data_gen():
#     num_samples = 10
#     for _ in range(num_samples):
#         input_data = np.random.normal(size=(1, 224, 224, 3))
#         input_data = input_data / 255.0
#         yield [input_data.astype(np.float32)]
#
# converter = tf.lite.TFLiteConverter.from_saved_model('model_tf')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
#
# tflite_model = converter.convert()
#
# with open('dogs2_float32.tflite', 'wb') as f:
#     f.write(tflite_model)

import tensorflow as tf
import numpy as np

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def representative_data_gen():
    num_samples = 10
    for _ in range(num_samples):
        input_data = np.random.normal(size=(1, 224, 224, 3))
        input_data = input_data / 255.0
        yield [input_data.astype(np.float32)]

onnx_model = onnx.load('dogs2.onnx')

# TensorFlow 모델로 변환
tf_rep = prepare(onnx_model)
tf_rep.export_graph('model_tf')


# TensorFlow 모델을 로드
converter = tf.lite.TFLiteConverter.from_saved_model('model_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# float32 양자화를 사용하여 TFLite 모델로 변환
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

try:
    tflite_model = converter.convert()
    with open('dogs4_float32.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted successfully.")
except Exception as e:
    print("Error during conversion:", e)

# onnx2tf -i dogs14.onnx - output_dir
"""
pip install nvidia-pyindex
pip install onnx-graphsurgeon
pip install -U onnx2tf
"""