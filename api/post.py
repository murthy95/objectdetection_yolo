import json
import requests
import io
from PIL import Image
import sys

url = 'http://127.0.0.1:5000/upload'


#sending image directly from the local as wrekzeug storage oabject
files = {'image': open(sys.argv[1], 'rb')}
r = requests.post(url, files=files)
print(r.content)

#pil_image = Image.open(sys.argv[1]).convert('RGB')
#imgByteArr = io.BytesIO()
#pil_image.save(imgByteArr, format='PNG')
#imgByteArr = imgByteArr.getvalue()
#
#data = {'image': imgByteArr,'obj_threshold':0.3, 'iou_threshold':0.6}
#headers = {'content-type': 'application/json'}
#r = requests.post(url, data=data, headers=headers)
#print (r.content)
