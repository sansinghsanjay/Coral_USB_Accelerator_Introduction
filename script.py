# libraries
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
import platform
import numpy as np
import pandas as pd

# use this function as it is, otherwise it will throw "Segmentation Fault" error
def output_tensor(interpreter):
	output_details = interpreter.get_output_details()[0]
	output_data = np.squeeze(interpreter.tensor(output_details['index'])())
	scale, zero_point = output_details['quantization']
	return scale * (output_data - zero_point)

# paths
model_path = "/home/pi/coral/my_script/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
image_path = "/home/pi/coral/my_script/input_images/parrot.jpg"

# shared library for EdgeTPU
EDGE_TPU_SHARED_LIB = {
	'Linux': 'libedgetpu.so.1',
	'Windows': 'edgetpu.dll'
}[platform.system()]

# load tflite model
interpreter = tflite.Interpreter(model_path = model_path, experimental_delegates = [tflite.load_delegate(EDGE_TPU_SHARED_LIB)])
interpreter.allocate_tensors()

# get input size
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Input Height: ", height)
print("Input Width: ", width)

# cv2 image read is providing less performance - only 0.70 confidence in inference
'''# read input image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("Actual size of input image: ", img.shape)

# resize image
img = cv2.resize(img, (height, width))
print("Modified size of input image: ", img.shape)'''

# PIL image read is providing high performance - around 0.77 confidence in inference
# make image suitable for input in model - size, color model, etc.
size = (height, width)
img = Image.open(image_path).convert('RGB').resize(size, Image.ANTIALIAS)

# putting input in model
tensor_index = interpreter.get_input_details()[0]['index']
interpreter.tensor(tensor_index)()[0][:, :] = img

# invoke interpreter
interpreter.invoke()

scores = output_tensor(interpreter)

# reading labels
labels = pd.read_csv("./models/inat_bird_labels.txt", header=None)

# showing prediction
print("Index of highest score: ", np.argmax(scores))
print("Highest score: ", np.max(scores))
print("Prediction: ", labels.iloc[np.argmax(scores)])
