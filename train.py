import os
import torch
from torchsummary import summary
import numpy as np
from tqdm import tqdm 
from absl import flags 
import pickle
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as multiprocessing

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
flags.DEFINE_boolean('distributed', False, 'use all available resources for training')

def train(argv):
	#make a network object
	device = 'cpu'
	if FLAGS.gpu:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = yolonet()
	net = net.to(device)
	#print(summary(net, (3,448,448)))
	
	loss = yolov1loss(lambda_coord=0.5, lambda_noobj=0.5, device=device)
	optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
	epoch = 0

	if FLAGS.resume_train or not FLAGS.train:
		checkpoint = torch.load(FLAGS.model_path)
		net.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		del checkpoint	
		torch.cuda.empty_cache()
		print('----successfully loaded network and training parameters---')
	
	if FLAGS.train :				
		traindata, valdata = VOCDataset(FLAGS.train_batch, FLAGS.val_batch)
		train_loss, val_loss= [], []
		for i in range(epoch, epoch + FLAGS.n_epochs, 1):
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
 			  
			print('training loss after {}/{} epochs is {}'.format(i + 1, epoch + FLAGS.n_epochs, np.mean(train_loss[-len(trainiter):])))	
			
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
						
					print('validation loss after {}/{} epochs is {}'.format(i + 1, epoch + FLAGS.n_epochs, np.mean(val_loss[-len(valiter):])))

			if i % FLAGS.result_frequency == 0:
				with torch.no_grad():
					suppressed_preds = non_max_suppression(y_hat[0].unsqueeze(0))
					if not len(suppressed_preds[0]) == 0:
						save_path = 'results/result_preds_e{}.png'.format(i + 1)
						visualize_bbox(x[0].to('cpu'), suppressed_preds[0], save_path)
			#saving model to checkpoint 
			torch.save({
		          'epoch': i + 1,
		          'model_state_dict': net.state_dict(),
		          'optimizer_state_dict': optimizer.state_dict()
		          }, 'chkpt_files/yolov1_model_sgd.pth')	
		
			with open('chkpt_files/training_loss.txt', 'wb') as f:
				pickle.dump(train_loss, f)
			with open('chkpt_files/validation_loss.txt', 'wb') as f:
				pickle.dump(val_loss, f)
	
	else:
		plot_random_n_predictions(10, net, device)

def plot_random_n_predictions(num, net, device):
	traindata, valdata = VOCDataset(FLAGS.train_batch, FLAGS.val_batch)
	valiter = iter(valdata)
	net.eval()
	with torch.no_grad():
		x, y = valiter.next()
		x, y = x.to(device), y.to(device)
		y_hat = net(x)
		suppressed_preds = non_max_suppression(y_hat)
		for i in range(len(suppressed_preds)):
			print(suppressed_preds[i])
			save_path = 'results/test_preds_{}.png'.format(i+1)
			visualize_bbox(x[i].to('cpu'), suppressed_preds[i], save_path)

			#preds = visualize_bbox(x[i].to('cpu'), suppressed_preds[i])
			#trues = visualize_bbox(x[i].to('cpu'), y_hat[i].to('cpu'))
			#print(preds.shape, trues.shape, x[i].shape)
			#image = np.concatenate([x[i].to('cpu').permute(1,2,0).numpy(), trues, preds], axis=1 )
			#import matplotlib.pyplot as plt 
			#plt.imsave(save_path, image)

def run(rank, size):
	import time
	time.sleep(1)
	torch.manual_seed(1234)
	traindata, valdata = VOCDataset(FLAGS.train_batch, FLAGS.val_batch, rank, size)
	device = torch.device("cuda:{}".format(rank))
	net = yolonet()
	net = net.to(device)
	loss = yolov1loss(lambda_coord=0.5, lambda_noobj=0.5, device=device)
	optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

	for i in range(FLAGS.n_epochs):
		trainiter = iter(traindata)
		net.train()
		epoch_loss = 0.0
		for _ in range(len(trainiter)):
			x, y = trainiter.next()
			x, y = x.to(device), y.to(device)
			y_hat = net(x)
			loss_ = loss(y_hat, y)
			epoch_loss += loss_.item()
			optimizer.zero_grad()
			loss_.backward()
			average_gradients(net)
			optimizer.step()
	
		print('Rank ', dist.get_rank(), ', epoch ',  i, ': ', epoch_loss / len(trainiter))
	
def init_processes(rank, size, fn, backend='gloo'):
	""" Initialize the distributed environment. """
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	dist.init_process_group(backend, rank=rank, world_size=size)
	fn(rank, size)

def average_gradients(model):
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
		param.grad.data /= size
	
def main(argv):
	if FLAGS.distributed :
		ngpus = torch.cuda.device_count()
		processes = []
		for rank in range(ngpus):
			p = Process(target=init_processes, args=(rank, ngpus, run))
			p.start()
			processes.append(p)
		for p in processes:
			p.join()	
					
if __name__ == '__main__':
	from absl import app
#	app.run(train)
	multiprocessing.set_start_method('spawn')
	app.run(main)			 
			
		
	
		

