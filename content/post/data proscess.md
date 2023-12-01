## fillna 缺失值处理函数
`fillna` 是一个用于在数据处理中填充缺失值（NaN或None）的方法，通常用于处理数据框架（如Pandas中的DataFrame）或类似的数据结构。这个方法可以帮助你将缺失值替换为指定的数值、字符串或其他数据，以便后续分析或计算不会受到缺失值的干扰
```
data
```
![输入图片说明](/imgs/2023-09-25/oWgbDC8zamyRfiZt.png)
```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  
inputs = inputs.fillna(inputs.mean())  
print(inputs)
```
![输入图片说明](/imgs/2023-09-25/3sOEvWhlfYqRzdeV.png)

##  pd.get_dummies（）
>是 Pandas 中的一个函数，通常用于进行**独热编码（One-Hot Encoding）**，将分类数据（categorical data）转换为机器学习模型可以处理的格式。
```
inputs=pd.get_dummies(inputs, dummy_na=True)
或者inputs = pd.get_dummies(inputs, columns=['Alley'],dummy_na=True)注意中括号
inputs
```
![输入图片说明](/imgs/2023-09-25/0KBqh4QkVtWCGT6o.png)
对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别。**]  
由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN” 
`pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。  
巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。
缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
 
 ### 例子2
 ![输入图片说明](/imgs/2023-09-25/g4tJWrGIiBYAgg9W.png)
 可用column指定要对那一列进行操作，注意[]
## touch.normal()
```
 torch.normal(means, std, out=None)
```
**返回一个张量**，包含从给定参数`means`,`std`的离散正态分布中抽取随机数。 均值`means`是一个张量，包含每个输出元素相关的正态分布的均值。 `std`是一个张量，包含每个输出元素相关的正态分布的标准差。 均值和标准差的形状不须匹配，但每个张量的元素个数须相同
 -   means (Tensor) – 均值
 -   std (Tensor) – 标准差
 -   out (Tensor) – 可选的输出张量
```
X = torch.normal(0, 1, (num_examples, len(w)))
```
## len()函数用于返回一个对象的长度或元素个数
```
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
```
len(matrix) 来获取其行数，结果将是 3，因为矩阵中有三个子向量（每个子向量代表一行）。
len()对于二维向量来说返回的是行数，而不是列数。如果想要获取列数，你可以通过检查第一行的长度来实现，例如 len(matrix[0]).在上面的示例中，len(matrix[0])返回3，因为第一行有三个元素。

##  d2l.synthetic_data(true_w, true_b, num) 
> 生成具有线性关系的合成数据集
- true_w 表示真实权重
- true_b 表示真实偏差
- num 数据个数
```
true_w = torch.tensor([2, -3.4])  
true_b = 4.2  
features, labels = d2l.synthetic_data(true_w, true_b, 1000) #生成具有线性关系的合成数据集
```
## data.DataLoader 
> from torch.utils import data
> 创建一个数据加载器对象，用于批量加载训练数据和标签，通常在训练深度学习模型时使用
-  `dataset`：一个包含训练数据和标签的数据集对象，通常是 `torch.utils.data.Dataset` 类的实例，例如 `torchvision.datasets.ImageFolder`。
    
-   `batch_size`：一个整数，表示每个批次的大小。即每次从数据集中加载多少个样本。
    
-   `shuffle`：一个布尔值，表示是否在每个 epoch（训练周期）之前随机打乱数据集。通常，在训练时希望数据集的顺序是随机的，以确保模型能够更好地泛化。对于训练集，通常设置为 `True`，对于验证集和测试集通常设置为 `False`。
```
def load_array(data_arrays, batch_size, is_train=True):  #@save  
    """构造一个PyTorch数据迭代器"""  
    dataset = data.TensorDataset(*data_arrays)  
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  
```
```
batch_size = 10  
data_iter = load_array((features, labels), batch_size)
```
