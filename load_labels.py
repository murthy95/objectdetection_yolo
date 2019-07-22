import torch
import os 
import torchvision

#Download script
def VOCDataset(train_batch_size, val_batch_size, rank=None, size=None):
	#code to download the dataset
	download = os.path.exists('./data/VOCtrainval_06-Nov-2007.tar')
	transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((448, 448), interpolation=2), 
		torchvision.transforms.ToTensor()
		#torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
		#										 std=[0.229, 0.224, 0.225])
])

	trainset = torchvision.datasets.VOCDetection(root='./data', year='2007', 
                                             image_set='train',
                                        download=download, transforms=transform_labels(448, transform))
	valset = torchvision.datasets.VOCDetection(root='./data', year='2007', 
                                             image_set='val',
                                       download=download, transforms=transform_labels(448, transform))
	if not size==None:
		trainset.images = trainset.images[rank * len(trainset) // size : (rank + 1)*len(trainset) // size]
		trainset.annotations = trainset.annotations[rank * len(trainset) // size : (rank + 1)*len(trainset) // size]
		
		valset.images = valset.images[rank * len(valset) // size : (rank + 1)*len(valset) // size]
		valset.annotations = valset.annotations[rank * len(valset) // size : (rank + 1)*len(valset) // size]

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size,
                                         shuffle=False, num_workers=2)
	return trainloader, valloader

#make a transform class to transform the voc json labels 
#to	bounding box tensors 
class transform_labels(object):
	'''reshapes the boudning box coordinates in coordination with the input image
  Args:
      image_size(int, tuple) accepts int or tuple of ints: image shape and width
   
	'''
	def __init__(self, image_shape, transform):
		self.resize = image_shape
		self.factor = (1, 1)
		self.transform = transform
		self.class_dict = {'aeroplane' : 0,
                    'bicycle' : 1,
                    'bird' : 2,
                    'boat' : 3,
                    'bottle' : 4,
                    'bus' : 5,
                    'car' : 6,
                    'cat' : 7,
                    'chair' : 8,
                    'cow' : 9,
                    'diningtable' : 10,
                    'dog' : 11,
                    'horse' : 12,
                    'motorbike' : 13,
                    'person' : 14,
                    'pottedplant' : 15,
                    'sheep' : 16,
                    'sofa' : 17,
                    'train' : 18,
                    'tvmonitor' : 19}
	def __call__(self, img, target, grid_size=7):
 		
		label = target
		shape_x = int(label['annotation']['size']['width'])
		shape_y = int(label['annotation']['size']['height'])
		self.factor = (1/shape_x, 1/shape_y)
		objects = label['annotation']['object']
		transformed_boxes = torch.zeros((7,7,6))
		try:
			for obj in list(objects):
				transformed_box =  self.return_transformed_box(obj)
				transformed_boxes[int(transformed_box[0]*7), int(transformed_box[1]*7)] = transformed_box
		except:
			transformed_box = self.return_transformed_box(objects)
			transformed_boxes[int(transformed_box[0]*7), int(transformed_box[1]*7)] = transformed_box
		return self.transform(img), transformed_boxes.permute(2,0,1)
    
	def return_transformed_box(self, obj):
		class_id = self.class_dict[obj['name']]
		bndbox = [int(obj['bndbox']['xmin']),
             int(obj['bndbox']['xmax']),
             int(obj['bndbox']['ymin']),
             int(obj['bndbox']['ymax'])]
		return torch.tensor([(bndbox[0] + bndbox[1])/2 * self.factor[0],\
            (bndbox[2] + bndbox[3])/2 * self.factor[1],\
            (bndbox[1] - bndbox[0]) * self.factor[0],\
            (bndbox[3] - bndbox[2]) * self.factor[1], 1, class_id])

	
