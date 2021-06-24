#PNN in prediction
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class classifier(object):
	def __init__(self,input_field,continus_nums,learning_rate,K,dense_M,h1,h2,h3):
		self.input_field=input_field
		self.feats_N=len(input_field)
		self.feats_sum=sum(input_field)
		self.continus_nums=continus_nums
		self.learning_rate=learning_rate
		self.dense_M=dense_M
		self.h_c1=16
		self.h1=h1
		self.h2=h2
		self.h3=h3
		self.K=K
		self.W0=Variable(nn.init.uniform(torch.randn(self.feats_sum,dense_M),-np.sqrt(6. / (self.feats_sum+dense_M)) ,np.sqrt(6. / (self.feats_sum+dense_M) )),requires_grad=True)
		self.W_z=Variable(nn.init.uniform(torch.randn(self.feats_N*dense_M,h1),-np.sqrt(6. / (self.feats_N*dense_M+h1)),np.sqrt(6. / (self.feats_N*dense_M+h1))),requires_grad=True)

		self.W_c1=Variable(nn.init.uniform(torch.randn(continus_nums,self.h_c1),-np.sqrt(6. / (continus_nums+self.h_c1)),np.sqrt(6. / (continus_nums+self.h_c1))),requires_grad=True)
		self.W_c2=Variable(nn.init.uniform(torch.randn(self.h_c1,self.h1),-np.sqrt(6. / (self.h_c1+self.h1)),np.sqrt(6. / (self.h_c1+self.h1))),requires_grad=True)

		self.W_fm=Variable(nn.init.uniform(torch.randn(self.feats_N,self.K),-np.sqrt(6. / (self.feats_N+self.K)),np.sqrt(6. / (self.feats_N+self.K))),requires_grad=True)
		self.W_p=Variable(nn.init.uniform(torch.randn(self.feats_N*self.feats_N,h1),-np.sqrt(6. / (h1+self.feats_N*self.feats_N)),np.sqrt(6. / (h1+self.feats_N*self.feats_N))),requires_grad=True)
		self.b_c1=Variable(torch.randn(1,self.h_c1),requires_grad=True)
		self.b1=Variable(torch.randn(1,h1),requires_grad=True)

		self.Wr1=Variable(nn.init.uniform(torch.randn(self.h1,self.h2),-np.sqrt(6. / (self.h1+self.h2)),np.sqrt(6. / (self.h1+self.h2))),requires_grad=True)
		self.Wr21=Variable(nn.init.uniform(torch.randn(self.h1,self.h1),-np.sqrt(6. / (self.h1+self.h1)),np.sqrt(6. / (self.h1+self.h1))),requires_grad=True) 
		self.Wr22=Variable(nn.init.uniform(torch.randn(self.h1*self.h1,self.h2),-np.sqrt(6. / (self.h1*self.h1+self.h2)),np.sqrt(6. / (self.h1*self.h1+self.h2))),requires_grad=True)
		self.br=Variable(torch.randn(1,h2),requires_grad=True) 

		self.W2=Variable(nn.init.uniform(torch.randn(h2,h3),-np.sqrt(6. / (h2+h3)),np.sqrt(6. / (h2+h3))),requires_grad=True) 
		self.b2=Variable(torch.randn(1,h3),requires_grad=True) 
		self.W3=Variable(nn.init.uniform(torch.randn(h3,h3),-np.sqrt(6. / (h3+h3)),np.sqrt(6. / (h3+h3))),requires_grad=True) 
		self.b3=Variable(torch.randn(1,h3),requires_grad=True)
		self.W4=Variable(nn.init.uniform(torch.randn(h3,h3),-np.sqrt(6. / (h3+h3)),np.sqrt(6. / (h3+h3))),requires_grad=True) 
		self.b4=Variable(torch.randn(1,h3),requires_grad=True)
		self.W5=Variable(nn.init.uniform(torch.randn(h3,1),-np.sqrt(6. / (h3+1)),np.sqrt(6. / (h3+1))),requires_grad=True) 
		self.b5=Variable(torch.randn(1,1),requires_grad=True)


		self.weights=[self.W0,self.W_z,self.W_c1,self.W_c2,self.W_fm,self.W_p,self.Wr1,self.Wr21,self.Wr22,self.W2,self.W3,self.W4,self.W5,self.br,self.b_c1,self.b1,self.b2,self.b3,self.b4,self.b5]
		self.cost=0.0
	def train_model(self,input,y):
		x=input[0]
		continus_x=input[1]
		#dense layer 从每一个field的(ends-starts+1)映射到M,一共有N*M个神经元
		feats_input=x[0,:self.input_field[0]].unsqueeze(0).mm(self.W0[:self.input_field[0]])
		for i in range(1,self.feats_N):
			start=sum(self.input_field[:i])
			temp=x[0,start:start+self.input_field[i]].unsqueeze(0).mm(self.W0[start:start+self.input_field[i]])
			feats_input=torch.cat((feats_input,temp),1)

		#FM层
		#一阶关系 
		l_z=feats_input.mm(self.W_z)

		#二阶关系
		#特征之间二阶相乘
		input_p=feats_input.view(self.feats_N,self.dense_M)
		input_p=input_p.mm(input_p.t())
		#参数矩阵
		fm_parameter=self.W_fm.mm(self.W_fm.t())
		inter_p=(input_p*fm_parameter).view(1,self.feats_N*self.feats_N)
		#to hidden1 layer
		l_p=inter_p.mm(self.W_p)	

		#continus到hidden1
		c1=continus_x.mm(self.W_c1)+self.b_c1
		o1=c1.clamp(min=0)
		l_c=o1.mm(self.W_c2)
		#hidden层
		a1=(l_c+l_z+l_p+self.b1).clamp(min=0)

		#一阶关系
		r1=a1.mm(self.Wr1)
		#二阶关系
		r21=(torch.transpose(a1,0,1).mm(a1))*self.Wr21
		r22=r21.view(1,self.h1*self.h1).mm(self.Wr22)

		ar=(r1+r22+self.br).clamp(min=0)
		#加入残差a1
		a2=(ar.mm(self.W2)+a1+self.b2).clamp(min=0)
		a3=(a2.mm(self.W3)+self.b3).clamp(min=0)
		#残差a2
		a4=(a3.mm(self.W4)+a2+self.b4).clamp(min=0)
		z5=a4.mm(self.W5)+self.b5
		##二分类问题吗
		out=torch.sigmoid(z5)
		cost=-(y*torch.log(out)+(1-y)*torch.log(1-out))
		##在反向传播
		cost.backward()
		
		self.W0.data-=self.learning_rate*self.W0.grad.data
		self.W_z.data-=self.learning_rate*self.W_z.grad.data
		self.W_c1.data-=self.learning_rate*self.W_c1.grad.data
		self.W_c2.data-=self.learning_rate*self.W_c2.grad.data
		self.W_fm.data-=self.learning_rate*self.W_fm.grad.data
		self.W_p.data-=self.learning_rate*self.W_p.grad.data
		self.Wr1.data-=self.learning_rate*self.Wr1.grad.data
		self.Wr21.data-=self.learning_rate*self.Wr21.grad.data
		self.Wr22.data-=self.learning_rate*self.Wr22.grad.data
		self.W2.data-=self.learning_rate*self.W2.grad.data
		self.W3.data-=self.learning_rate*self.W3.grad.data
		self.W4.data-=self.learning_rate*self.W4.grad.data
		self.W5.data-=self.learning_rate*self.W5.grad.data
		self.b_c1.data-=self.learning_rate*self.b_c1.grad.data
		self.br.data-=self.learning_rate*self.br.grad.data
		self.b1.data-=self.learning_rate*self.b1.grad.data
		self.b2.data-=self.learning_rate*self.b2.grad.data
		self.b3.data-=self.learning_rate*self.b3.grad.data
		self.b4.data-=self.learning_rate*self.b4.grad.data
		self.b5.data-=self.learning_rate*self.b5.grad.data


		self.W0.grad.data.zero_()
		self.W_z.grad.data.zero_()
		self.W_c1.grad.data.zero_()
		self.W_c2.grad.data.zero_()
		self.W_fm.grad.data.zero_()
		self.W_p.grad.data.zero_()
		self.Wr1.grad.data.zero_()
		self.Wr21.grad.data.zero_()
		self.Wr22.grad.data.zero_()
		self.W2.grad.data.zero_()
		self.W3.grad.data.zero_()
		self.W4.grad.data.zero_()
		self.W5.grad.data.zero_()
		self.b_c1.grad.data.zero_()
		self.br.grad.data.zero_()
		self.b1.grad.data.zero_()
		self.b2.grad.data.zero_()
		self.b3.grad.data.zero_()
		self.b4.grad.data.zero_()
		self.b5.grad.data.zero_()

		return float(cost.data[0])

	def predict(self,input,y):
		x=input[0]
		continus_x=input[1]
		#dense layer 从每一个field的(ends-starts+1)映射到M,一共有N*M个神经元
		feats_input=x[0,:self.input_field[0]].unsqueeze(0).mm(self.W0[:self.input_field[0]])
		for i in range(1,self.feats_N):
			start=sum(self.input_field[:i])
			temp=x[0,start:start+self.input_field[i]].unsqueeze(0).mm(self.W0[start:start+self.input_field[i]])
			feats_input=torch.cat((feats_input,temp),1)

		#FM层
		#一阶关系 
		l_z=feats_input.mm(self.W_z)

		#二阶关系
		#特征之间二阶相乘
		input_p=feats_input.view(self.feats_N,self.dense_M)
		input_p=input_p.mm(input_p.t())
		#参数矩阵
		fm_parameter=self.W_fm.mm(self.W_fm.t())
		inter_p=(input_p*fm_parameter).view(1,self.feats_N*self.feats_N)
		#to hidden1 layer
		l_p=inter_p.mm(self.W_p)

		#continus到hidden1
		c1=continus_x.mm(self.W_c1)+self.b_c1
		o1=c1.clamp(min=0)
		l_c=o1.mm(self.W_c2)
		#hidden层
		a1=(l_c+l_z+l_p+self.b1).clamp(min=0)

		#一阶关系
		r1=a1.mm(self.Wr1)
		#二阶关系
		r21=(torch.transpose(a1,0,1).mm(a1))*self.Wr21
		r22=r21.view(1,self.h1*self.h1).mm(self.Wr22)

		ar=(r1+r22+self.br).clamp(min=0)
		#加入残差a1
		a2=(ar.mm(self.W2)+a1+self.b2).clamp(min=0)
		a3=(a2.mm(self.W3)+self.b3).clamp(min=0)
		#残差a2
		a4=(a3.mm(self.W4)+a2+self.b4).clamp(min=0)
		z5=a4.mm(self.W5)+self.b5
		out=torch.sigmoid(z5)
		cost=-(y*torch.log(out)+(1-y)*torch.log(1-out))
		#return float(out.data[0])
		return float(out.data[0]),float(cost.data[0])

	def setMax(self,cost):
		self.weights=[self.W0,self.W_z,self.W_c1,self.W_c2,self.W_fm,self.W_p,self.Wr1,self.Wr21,self.Wr22,self.W2,self.W3,self.W4,self.W5,self.br,self.b_c1,self.b1,self.b2,self.b3,self.b4,self.b5]
		self.cost=cost

	def getMax(self):
		self.W0,self.W_z,self.W_c1,self.W_c2,self.W_fm,self.W_p,self.Wr1,self.Wr21,self.Wr22,self.W2,self.W3,self.W4,self.W5,self.br,self.b_c1,self.b1,self.b2,self.b3,self.b4,self.b5=self.weights

	def minLoss(self):
		return self.cost

	def reduceLR(self):
		self.learning_rate=self.learning_rate/2

	def setInitial(self):
		self.learning_rate=self.learning_rate/2

		#参数重新设置
		self.W0=Variable(nn.init.uniform(torch.randn(self.feats_sum,self.dense_M),-np.sqrt(6. / (self.feats_sum+self.dense_M)) ,np.sqrt(6. / (self.feats_sum+self.dense_M) )),requires_grad=True)
		self.W_z=Variable(nn.init.uniform(torch.randn(self.feats_N*self.dense_M,self.h1),-np.sqrt(6. / (self.feats_N*self.dense_M+self.h1)),np.sqrt(6. / (self.feats_N*self.dense_M+self.h1))),requires_grad=True)

		self.W_c1=Variable(nn.init.uniform(torch.randn(self.continus_nums,self.h_c1),-np.sqrt(6. / (self.continus_nums+self.h_c1)),np.sqrt(6. / (self.continus_nums+self.h_c1))),requires_grad=True)
		self.W_c2=Variable(nn.init.uniform(torch.randn(self.h_c1,self.h1),-np.sqrt(6. / (self.h_c1+self.h1)),np.sqrt(6. / (self.h_c1+self.h1))),requires_grad=True)
		#self.W_c3=Variable(nn.init.uniform(torch.randn(self.h_c2,self.h1),-np.sqrt(6. / (self.h_c2+self.h1)),np.sqrt(6. / (self.h_c2+self.h1))),requires_grad=True)

		self.W_fm=Variable(nn.init.uniform(torch.randn(self.feats_N,self.K),-np.sqrt(6. / (self.feats_N+self.K)),np.sqrt(6. / (self.feats_N+self.K))),requires_grad=True)
		self.W_p=Variable(nn.init.uniform(torch.randn(self.feats_N*self.feats_N,self.h1),-np.sqrt(6. / (self.h1+self.feats_N*self.feats_N)),np.sqrt(6. / (self.h1+self.feats_N*self.feats_N))),requires_grad=True)

		self.Wr1=Variable(nn.init.uniform(torch.randn(self.h1,self.h2),-np.sqrt(6. / (self.h1+self.h2)),np.sqrt(6. / (self.h1+self.h2))),requires_grad=True)
		self.Wr21=Variable(nn.init.uniform(torch.randn(self.h1,self.h1),-np.sqrt(6. / (self.h1+self.h1)),np.sqrt(6. / (self.h1+self.h1))),requires_grad=True) 
		self.Wr22=Variable(nn.init.uniform(torch.randn(self.h1*self.h1,self.h2),-np.sqrt(6. / (self.h1*self.h1+self.h2)),np.sqrt(6. / (self.h1*self.h1+self.h2))),requires_grad=True)
		self.br=Variable(torch.randn(1,self.h2),requires_grad=True) 

		self.b_c1=Variable(torch.randn(1,self.h_c1),requires_grad=True)
		#self.b_c2=Variable(torch.randn(1,self.h_c2),requires_grad=True)
		self.b1=Variable(torch.randn(1,self.h1),requires_grad=True)
		self.W2=Variable(nn.init.uniform(torch.randn(self.h2,self.h3),-np.sqrt(6. / (self.h2+self.h3)),np.sqrt(6. / (self.h2+self.h3))),requires_grad=True) 
		self.b2=Variable(torch.randn(1,self.h3),requires_grad=True) 
		self.W3=Variable(nn.init.uniform(torch.randn(self.h3,self.h3),-np.sqrt(6. / (self.h3+self.h3)),np.sqrt(6. / (self.h3+self.h3))),requires_grad=True) 
		self.b3=Variable(torch.randn(1,self.h3),requires_grad=True)
		self.W4=Variable(nn.init.uniform(torch.randn(self.h3,self.h3),-np.sqrt(6. / (self.h3+self.h3)),np.sqrt(6. / (self.h3+self.h3))),requires_grad=True) 
		self.b4=Variable(torch.randn(1,self.h3),requires_grad=True)
		self.W5=Variable(nn.init.uniform(torch.randn(self.h3,1),-np.sqrt(6. / (self.h3+1)),np.sqrt(6. / (self.h3+1))),requires_grad=True) 
		self.b5=Variable(torch.randn(1,1),requires_grad=True)
