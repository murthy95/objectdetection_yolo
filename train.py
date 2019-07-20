import torch
from torchsummary import summary
import numpy as np
from tqdm import tqdm 
from absl import flags 

from yolov1 import yolonet
from load_labels import VOCDataset
from yolov1loss import yolov1loss
from utils import *

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_epochs', 1,'Number of training epochs',
											lower_bound=0)
flags.DEFINE_boolean('train', False,'Network set to train')
flags.DEFINE_boolean('resume_train', False,'--resume training operation')
flags.DEFINE_integer('val_frequency', 1,'--val results printing frequency',
											lower_bound=1)
flags.DEFINE_integer('result_frequency', 1,'--result image saving frequency',
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
	
	loss = yolov1loss(lambda_coord=0.5, lambda_noobj=0.5, device=device)
	optimizer = torch.optim.Adam(net.parameters())
	
	if FLAGS.resume_train:
		checkpoint = torch.load(FLAGS.model_path)
		net.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
	
	if FLAGS.train :
		traindata, valdata = VOCDataset(FLAGS.train_batch, FLAGS.val_batch)
		train_loss, val_loss= [], []
		for i in range(FLAGS.n_epochs):
			trainiter = iter(traindata)
			net.train()
			for _ in tqdm(range(len(trainiter)), ncols=100):
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
					for _ in tqdm(range(len(valiter)), ncols=100):
						x, y = valiter.next()
						x, y = x.to(device), y.to(device)
						y_hat = net(x)
						loss_ = loss(y_hat, y)
						val_loss.append(loss_.item())
						
					print('validation loss after {}/{} epochs is {}'.format(i+1, FLAGS.n_epochs, np.mean(val_loss[-len(valiter):])))

			if i % FLAGS.result_frequency == 0:
				with torch.no_grad():
					suppressed_preds = non_max_suppression(y_hat[0].unsqueeze(0))
					if not len(suppressed_preds[0]) == 0:
						save_path = 'results/result_preds_e{}.png'.format(i+1)
						visualize_bbox(x[0].to('cpu'), suppressed_preds[0], save_path)
			#saving model to checkpoint 
			torch.save({
  	          'epoch': i+1,
  	          'model_state_dict': net.state_dict(),
  	          'optimizer_state_dict': optimizer.state_dict()
  	          }, 'chkpt_files/yolov1_model.pth')	
		
			import pickle
			with open('chkpt_files/training_loss.txt', 'wb') as f:
				pickle.dump(train_loss, f)
			with open('chkpt_files/validation_loss.txt', 'wb') as f:
				pickle.dump(val_loss, f)
	
	else:
		raise 'NotImplementedError'	
							
if __name__ == '__main__':
	from absl import app
	app.run(train)
			 
			
		
	
		

