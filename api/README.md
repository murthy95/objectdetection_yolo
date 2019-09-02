##How to call the api

`<import requests
url = 'http://127.0.0.1:5000'
files = {'image': open('test.jpg', 'rb'), 'obj_threshold':0.3, 'iou_threshold':0.6}
requests.post(url, files=files)>`
