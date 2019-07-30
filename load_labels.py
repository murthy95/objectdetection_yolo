import torch
import os 
import torchvision
from PIL import Image

from augmentation import *
from utils import to_center_dims_format

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

	trainset_07 = torchvision.datasets.VOCDetection(root='./data', year='2007', 
                                             image_set='train',
                                        download=download, transforms=transform_labels(448, transform))
	trainset_12 = torchvision.datasets.VOCDetection(root='./data', year='2012', 
                                             image_set='train',
                                        download=False, transforms=transform_labels(448, transform))

	valset = torchvision.datasets.VOCDetection(root='./data', year='2007', 
                                             image_set='val',
                                       download=download, transforms=transform_labels(448, transform))
	if not size==None:
		trainset_07.images = trainset_07.images[rank * len(trainset_07) // size : (rank + 1)*len(trainset_07) // size]
		trainset_07.annotations = trainset_07.annotations[rank * len(trainset_07) // size : (rank + 1)*len(trainset_07) // size]
		
		trainset_12.images = trainset_12.images[rank * len(trainset_12) // size : (rank + 1)*len(trainset_12) // size]
		trainset_12.annotations = trainset_12.annotations[rank * len(trainset_12) // size : (rank + 1)*len(trainset_12) // size]

	
		valset.images = valset.images[rank * len(valset) // size : (rank + 1)*len(valset) // size]
		valset.annotations = valset.annotations[rank * len(valset) // size : (rank + 1)*len(valset) // size]

	trainset = torch.utils.data.ConcatDataset([trainset_07, trainset_12])
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size,
                                         shuffle=True, num_workers=2)
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
		self.augmentations = [hsv_augmentation, flip, translation, random_cropping]
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
		if not isinstance(objects, list) :
			objects = list([objects])
		
		numpy_objects = []
		for obj in list(objects):
			bbox = [int(obj['bndbox']['xmin']),
							int(obj['bndbox']['ymin']),
							int(obj['bndbox']['xmax']),
							int(obj['bndbox']['ymax']), 1, 
							self.class_dict[obj['name']]]
			numpy_objects.append(bbox)
		objects = np.array(numpy_objects)
		augment = np.random.choice(self.augmentations)
		if torch.randn(1) < 0.8:
			img, bboxes = augment(np.asarray(img), np.array(numpy_objects))
		else:
			img, bboxes  = np.asarray(img), np.array(numpy_objects)
		bboxes = to_center_dims_format(bboxes)
		bboxes[:,:4] = bboxes[:,:4] / np.tile([shape_x, shape_y], 2)		

		for transformed_box in bboxes:
			#transformed_box = to_center_dims_format(obj) #self.return_transformed_box(obj)
			coordx, coordy = int(transformed_box[0]*7), int(transformed_box[1]*7)
			transformed_box[0] = transformed_box[0]*7 - int(transformed_box[0]*7)
			transformed_box[1] = transformed_box[1]*7 - int(transformed_box[1]*7)
			transformed_boxes[coordy, coordx] = torch.tensor(transformed_box)

#		except:
#			transformed_box = self.return_transformed_box(objects)
#			coordx, coordy = int(transformed_box[0]*7), int(transformed_box[1]*7)
#			transformed_box[0] = transformed_box[0]*7 - int(transformed_box[0]*7)
#			transformed_box[1] = transformed_box[1]*7 - int(transformed_box[1]*7)
#			transformed_boxes[coordx, coordy] = transformed_box
#
		return self.transform(Image.fromarray(img)), transformed_boxes.permute(2,0,1)
    
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
	
	def get_objects(self, objects):
		pass 
	
