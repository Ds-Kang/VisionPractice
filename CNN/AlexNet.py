import torch.nn as nn
import torch.nn.fucntional as F

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75):
        super(LRN, self).__init__()
        self.average=nn.AvgPool2d(kernel_size=local_size,
            stride=1,
            padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        div = x.pow(2)
        div = self.average(div)
        div = div.mul(self.alpha).add(1.0).pow(self.be	ta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):
	def __init__(self):
	super(AlexNet, self).__init__()
	self.conv1=nn.Conv2d(3,96,kernel_size=11,stride=4,padding=0)
	self.conv2=nn.Conv2d(96,256,kernel_size=5,stride=2,padding=2)
	self.conv3=nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1)
	self.conv4=nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1)
	self.conv5=nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1)
	self.fc6=nn.linear(256*6*6,4096)
	self.fc7=nn.linear(4096,4096)
	self.fc8=nn.linear(4096,1000)

	def forward(self,x):
		out=F.relu(self.conv1(x))
		out=F.max_pool2d(out,3,stride=2)

