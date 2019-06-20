from load_labels import *
from utils import *

train, test = VOCDataset()
trainiter = iter(test)
x, y = next(trainiter)

visualize_bbox(x[0], y[0], save_path='test.png')
