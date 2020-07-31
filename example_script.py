# libraries
from PIL import Image
import tflite_runtime.interpreter as tflite
import platform
import numpy as np
import collections
import operator

# use this function as it is, otherwise it will throw "Segmentation Fault" error
def output_tensor(interpreter):
	output_details = interpreter.get_output_details()[0]
	output_data = np.squeeze(interpreter.tensor(output_details['index'])())
	scale, zero_point = output_details['quantization']
	return scale * (output_data - zero_point)

# paths
model_path = "/home/pi/coral/my_script/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
image_path = "/home/pi/coral/my_script/input_images/parrot.jpg"
labels_path = "/home/pi/coral/my_script/models/inat_bird_labels.txt"

# driver for Coral Accelerator
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

# load labels
labels = {}
with open(labels_path, 'r', encoding='utf-8') as f:
	lines = f.readlines()
	if not lines:
		labels = {}
	if lines[0].split(' ', maxsplit=1)[0].isdigit():
		pairs = [line.split(' ', maxsplit=1) for line in lines]
		labels = {int(index): label.strip() for index, label in pairs}
	else:
		labels = {index: line.strip() for index, line in enumerate(lines)}

# get tflite model
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])

# initiate interpreter
interpreter.allocate_tensors()

# get input size
_, height, width, _ = interpreter.get_input_details()[0]['shape']
size = width, height

# make image suitable for input in model - size, color model, etc.
image = Image.open(image_path).convert('RGB').resize(size, Image.ANTIALIAS)

# set input
tensor_index = interpreter.get_input_details()[0]['index']
interpreter.tensor(tensor_index)()[0][:, :] = image

# invoke model
interpreter.invoke()

# make prediction
top_k = 1
score_threshold = 0.0
Class = collections.namedtuple('Class', ['id', 'score'])
scores = output_tensor(interpreter)
classes = [
	Class(i, scores[i])
	for i in np.argpartition(scores, -top_k)[-top_k:]
		if scores[i] >= score_threshold
]
classes = sorted(classes, key=operator.itemgetter(1), reverse=True)
print('-------RESULTS--------')
for klass in classes:
	print('%s: %.5f' % (labels.get(klass.id), klass.score))
