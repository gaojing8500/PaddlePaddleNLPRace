import numpy as np
import torch
from torch_nn3 import classifier
import pandas as pd
from sklearn import metrics
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#新加2行
import warnings 
warnings.filterwarnings('ignore')
df=pd.read_csv(r'/home/gaojing/PTM/datasets/time_data/brightkite_NY_train.csv')
df_test=pd.read_csv(r'/home/gaojing/PTM/datasets/time_data/brightkite_NY_test.csv')
x_test,y_test=[],[]

for row in df_test.itertuples():
	#把1~11列装到X里面
	x_test.append(list(row)[1:11])
	#把第11列装到Y里面
	y_test.append(np.float(row[11]))

def Create(x,y):
	discrete_field,continus_field=[],[]
	lens=len(y)
	for j in range(lens):
		discrete_field.append([0]*74)
		continus_field.append([0]*5)
		for i in range(len(input_field)):
			start=sum(input_field[:i])
			discrete_field[len(discrete_field)-1][start+int(x[j][6+i])-1]=1
		for i in range(5):
			continus_field[len(continus_field)-1][i]=np.float(x[j][1+i])
	
	x=Variable(torch.Tensor(discrete_field),requires_grad=False)
	continus_x=Variable(torch.Tensor(continus_field),requires_grad=False)
	y=torch.Tensor(y)
	
	return x,continus_x,y

def train():
	max,iter=-1,0
	flag=0
	for k in range(120):
		if (k+1)%10==0:
			model.reduceLR()
		
		cost=0.0
		for i in range(train_nums):
			input=[train_x[i].unsqueeze(0),train_continus_x[i].unsqueeze(0)]
			cost+=model.train_model(input,train_y[i])

		if np.isnan(cost)==True:
			print('get1!!')
			model.setInitial()
			flag=1
			return flag
		cost/=train_nums
		#print('train loss:%f' %(cost))				


		predict=0
		for i in range(vali_nums):
			input=[vali_x[i].unsqueeze(0),vali_continus_x[i].unsqueeze(0)]
			out,vali_loss=model.predict(input,vali_y[i])
			predict+=int((out>0.2)==vali_y[i])
	
		vali_acc=predict/vali_nums
		#print('vali accu:%f' %(vali_acc))
		#保存validation最大的accuracy
		if vali_acc>max:
			max=vali_acc
			model.setMax(cost)
			iter=0
		else:
			iter+=1
			if iter>10 and k>40:
				model.getMax()
				break
		#pre=np.array(predict)
		#np_y=test_y.numpy()
		#auc=metrics.roc_auc_score(np_y,pre)
		#print(auc)
	return flag
	

#initial
#初始学习率
learning_rate=0.04   
input_field=[12,31,24,7]
continus_nums=5
dense_M=20
K=20
#三层隐藏层神经网络结构，神经元数量设置32-64-32
'''
可以测试 4 中不同的网络结构：恒等、递增、递减、菱形。
'''
h1,h2,h3=32,64,32


id,hit_nums,ave_loss=0,0,0
data_X,data_Y,test_predict=[],[],[]
for row in df.itertuples():
	if id!=row[1]:
		#分割数据集
		seed0,seed1=int(np.random.uniform(0,1000)),int(np.random.uniform(0,1000))
		data_X,data_Y=shuffle(data_X,data_Y,random_state=seed0)
		x_train, x_vali, y_train, y_vali = train_test_split(data_X, data_Y, test_size=0.2, random_state=seed1)
		train_x,train_continus_x,train_y=Create(x_train,y_train)
		vali_x,vali_continus_x,vali_y=Create(x_vali,y_vali)
		test_x,test_continus_x,test_y=Create(x_test,y_test)
		train_nums,vali_nums,test_nums=train_x.shape[0],vali_x.shape[0],test_x.shape[0]

		#建立模型
		model=classifier(input_field,continus_nums,learning_rate,K,dense_M,h1,h2,h3)
		flag=train()
		while(flag==1):
			flag=train()

		#测试集预测
		input=[test_x[id].unsqueeze(0),test_continus_x[id].unsqueeze(0)]
		out,loss=(model.predict(input,test_y[id]))
		test_predict.append(out)
		hit_nums+=int((out>0.2)==test_y[id])
		print(id,hit_nums)


		data_X,data_Y=[],[]
		id=row[1]
		


	data_X.append(list(row)[1:11])
	data_Y.append(np.float(row[11]))

print(hit_nums)
#AUC(准确率，roc曲线下的面积，用于评价二值分类器的好坏，取值0.5~1，越大越好)
'''
AUC 是以 FPR 为横坐标，TPR为纵坐标所绘制的 ROC曲线下的面积。关于准确率和 AUC 有以下相关定义：
（1）TP：如果用户在给定位置上执行了签到，并且预测结果也签到了。
（2）FN：如果用户在给定位置上执行了签到，但是预测结果没有签到。
（3）FP：如果用户在给定位置上没有执行签到，但是预测结果签到了。
（4）TN：如果用户在给定位置上没有执行签到，并且预测结果没有签到。
那么，准确率，FPR 和 TPR 公式如下所示：
准确率=正确预测的签到条目数/总条目数=(TP+TN)/(TP+FN+FP+TN)
FPR=FP/(FP+TN)
TPR=TP/(TP+FN)
'''
pre=np.array(test_predict)
np_y=test_y.numpy()
auc=metrics.roc_auc_score(np_y,pre)
print(auc)