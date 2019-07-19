import torch
from torchsummary import summary
import numpy as np
from tqdm import tqdm 

from yolov1 import yolonet
from load_labels import VOCDataset
from yolov1loss import yolov1loss

from absl import flags 
FLAGS = flags.FLAGS

flags.DEFINE_integer('n_epochs', 1,'Number of training epochs',
											lower_bound=0)
flags.DEFINE_boolean('train', False,'Network set to train')
flags.DEFINE_boolean('resume_train', False,'--resume training operation')
flags.DEFINE_integer('val_frequency', 1,'--val results printing frequency',
											lower_bound=1)
flags.DEFINE_boolean('gpu', True,'use GPU for training')
flags.DEFINE_integer('train_batch', 16,'training batch size',
											lower_bound=1)
flags.DEFINE_integer('val_batch', 16,'validation batch size',
											lower_bound=1)
flags.DEFINE_string('model_path', '', 'path to model_chkpt file')
flags.DEFINE_string('test_path', '', 'path to test file')
flags.DEFINE_string('save_pred', '', 'path to save image with predictions')

def train(argv):
	#make a network object
	device = 'cpu'
	if FLAGS.gpu:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = yolonet()
	net = net.to(device)
	#print(summary(net, (3,448,448)))
	
	loss = yolov1loss(lambda_coord=0.005, lambda_noobj=0.005, device=device)
	optimizer = torch.optim.Adam(net.parameters())
	
	if FLAGS.resume_train:
		checkpoint = torch.load(FLAGS.model_path)
		net.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		
	
	if FLAGS.train :
		traindata, valdata = VOCDataset(FLAGS.train_batch, FLAGS.val_batch)
		train_loss, val_loss= [], []
		for i in range(FLAGS.n_epochs):
			trainiter = iter(traindata)
			net.train()
			for _ in tqdm(range(len(trainiter)), ncols=50):
				x, y = trainiter.next()
				x, y = x.to(device), y.to(device)
				y_hat = net(x)
				loss_ = loss(y_hat, y)
				train_loss.append(loss_.item())
				optimizer.zero_grad()
				loss_.backward()
				optimizer.step()
 			  
			print('training loss after {}/{} epochs is {}'.format(i+1, FLAGS.n_epochs, np.mean(train_loss[-len(trainiter):])))	
			
			if i % FLAGS.val_frequency == 0:
				valiter = iter(valdata)
				net.eval()
				with torch.no_grad():
					for _ in tqdm(range(len(valiter)), ncols=50):
						x, y = valiter.next()
						x, y = x.to(device), y.to(device)
						y_hat = net(x)
						loss_ = loss(y_hat, y)
						val_loss.append(loss_.item())
						
					print('validation loss after {}/{} epochs is {}'.format(i+1, FLAGS.n_epochs, np.mean(val_loss[-len(valiter):])))
	
	else:
		
	
							
if__name__ == '__main__':
	from absl import app
	app.run(train)
			 
			
		
	
		

