# 题目
考虑目标函数f(x) = sin(wx)和大小为N=2的数据集，并假设该数据集是无噪声的。我们在[-1,1]中均匀采样x，生成一个数据集(x1,y1)，(x2,y2);然后用两种模型中的一种来拟合数据:

a) H0:形式为h(x) = b的所有直线的集合;

b) H1: h(x) = ax+b的所有直线的集合。

对于H0，我们选择最适合数据的常数假设(中点的水平线，b = (y1+y2)/2)。对于H1，我们选择经过两个数据点(x1, y1)和(x2, y2)的直线。对许多数据集重复这个过程来估计偏差和方差。从实验结果回答哪个模型更好?

# 生成数据
生成1000组随机采样的（-1，1）样本，feature的每行（x1,x2)  labels每行(y1,y2) 维度1000*2
x1 x2分别都是[-1,1]的1000个数，之后用stack按列堆叠成features 

``` python
import torch  
from torch import  nn  
from torch.utils import data  
import numpy as np  
from d2l import torch as d2l  
%matplotlib inline
w=1   # 数据集大小
N = 1000  # 数据集大小
x1 = torch.rand(N)   # 生成从 0 到 1 的随机数  
x1 = 2 * (x1 - 0.5)  # 将随机数映射到 [-1, 1] 范围  
x2 = torch.rand(N)   
x2= 2 * (x2 - 0.5)  
features = torch.stack((x1, x2), dim=1)  # 计算 sine 函数的值作为标签数据集 (1000, 2)
labels = torch.sin(w*features)
```
观察sinx在[-1,1]图像
```
x=torch.from_numpy(np.linspace(-1, 1, 1000))  
d2l.plot(x ,torch.sin(x))
```
![输入图片说明](/imgs/2023-10-24/LhEgi95ivVujCzdN.png)

```
def load_array(data_arrays, batch_size, is_train=True):  #@save  
    """构造一个PyTorch数据迭代器"""  
    dataset = data.TensorDataset(*data_arrays)  
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
 ```
 ```
batch_size = 10#每次迭代器中取出批量为10的10组数据   
data_iter=load_array((features,labels),batch_size)
```
```
for X,y in data_iter:  
    print(X,y)  
    break
 ```
 打印X，y 得到了一个批量的数据输出 X 10×2， Y 10×2
![输入图片说明](/imgs/2023-10-24/0TFdVSu6uyjVdvnc.png)
# 构造模型 
h0模型得到的是这一批量数据的平均的y 
第一列和第二列相加除以2,每行代表这组数据y的平均值 
``` 
def h0(X,y):  
    return (y[:,0]+y[:,1])/2
```
h1模型计算了每组数据的斜率和截距 
m，b是10×1向量,代表这批数据的斜率和截距
```
def h1(X,y):  
    m = (y[:,1] - y[:,0]) / (X[:,1] - X[:,0])  
    # 计算截距  
    b = y[:,1]- m * X[:,1]  
    return m, b
```
h1pred_y作用：传入测试数据X，用h1模型的斜率和截距输出这批测试数据的预测Y
```
def h1pred_y(m,b,X):  
    return m*X+b
```
# 评估模型
先在样本点得到h0模型的预测值b0  预测结果y=b0
在样本点中获得h1模型的斜率a1 和截距b1  预测结果y=a1*x+b1
测试集 从[-1,1]均匀选取10个样本 
分别计算h0，h1模型在测试集合上的损失  累加到两个模型的偏差bias0，bias1 
最终输出两模型的偏差
```
def evaluate(data_iter):  
    bias0=0.0   
	bias1=0.0  
    for X,y in data_iter:  
        b0=h0(X,y)#得到h0模型的b b0是两个点的平均值   
		a1,b1=h1(X,y)#得到H1模型的斜率和截距a，b  
        test_x= torch.from_numpy(np.linspace(-1, 1, 10))  # 示例多个点作为测试，评估模型性能 计算模型的偏差  
        loss = nn.MSELoss()  
        l0=loss(b0,torch.sin(test_x))  
        l1=loss(h1pred_y(a1,b1,test_x),torch.sin(test_x))  
        bias0+=l0  
        bias1+=l1  
    return bias0 , bias1
```
# 结论
可以看出在1000个数据中模型
h0偏差44.4780
h1偏差0.5731 
h1的性能好于h0
![输入图片说明](/imgs/2023-10-24/daLXTy5qoqc3eTzD.png)
