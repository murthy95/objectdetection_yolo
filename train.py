import torch
from torchsummary import summary
from yolov1 import yolonet
from load_labels import VOCDataset
from yolov1loss import yolov1loss

def train(**kwargs):
	#make a network object
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = yolonet()
	net = net.to(device)
	print(summary(net, (3,448,448))
	
	loss = yolov1loss()
	optimizer = torch.optim.Adam(net.parameters())
	traindata, valdata = VOCDataset()
	train_loss, val_loss= [], []
	for i in range(kwargs[n_epochs]):
		trianiter = iter(traindata)
		net.train()
		try:
			x, y = trainiter.next()
			x = x.to(device)
			y = y.to(device)
			y_hat = net(x)
			loss_ = loss(y_hat, y)
			train_loss.append(loss_.data[0])
			optimizer.zero_grad()
			loss_.backward()
			optimizer.step()
		except:
			print('training loss after {}/{} epochs is {}'.format(i+1, kwargs[n_epochs], torch.mean(train_loss[-len(trainiter):])))	

		if i % kwargs[val_frequency] == 0:
			valiter = iter(valdata)
			net.eval()
			with torch.no_grad():
				try:
					x, y = valiter.next()
					x = x.to(device)
					y = y.to(device)
					y_hat = net(x)
					loss_ = loss(y_hat, y)
					val_loss.append(loss_.data[0])
				except:
					print('validation loss after {}/{} epochs is {}'.format(i+1, kwargs[n_epochs], torch.mean(val_loss[-len(valiter):])))	
			
if __name__ == '__main__':
	train(**kwargs) 
			 
			
		
	
		

