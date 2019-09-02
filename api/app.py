from flask import Flask
import torch
import json
from flask import request
import io
from PIL import Image

from assets.utils import *

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_for_inference():
	net = yolonet()
	checkpoint = torch.load('assets/yolov1_model_sgd_stage2.pth') 
	net.load_state_dict(checkpoint['model_state_dict'])
	del checkpoint
	
	#datadict = json.loads(request.data.decode('utf-8'))
	input_image= Image.open(io.BytesIO(request.files.get('image').read())).convert('RGB')
	
	obj_threshold = 0.3#float(datadict['obj_threshold'])
	iou_threshold = 0.6#float(datadict['iou_threshold'])
	#input_image = Image.open(io.BytesIO(datadict['image'])).convert('RGB')
	#preprocessing code 
	preprocessed_image = transform_input(input_image)

	#prediction
	net.eval()
	with torch.no_grad():
		y_pred = net(preprocessed_image.unsqueeze(0))	

	#postprocessing 
	y_suppressed = non_max_suppression(y_pred, obj_threshold, iou_threshold)

	return str([str(y.numpy()) for y in y_suppressed])
 
@app.route('/hello')
def hello():
	return 'Hello World!'

if __name__ == '__main__':
	app.run(debug=True)
