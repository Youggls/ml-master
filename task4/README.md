# Task3
> 线性回归

## 如何运行

### Requirements

- python package:

  ```
  numpy
  ```

- python version: python > 3.5

### Shell

在源代码目录下运行 `python main.py` 即可

## 判别式最小二乘算法

最小二乘分类决策通过公式 $y(x)=W^Tx$ 给出，通过 $c_k=\argmax_ky_k(x)$ 给出类别标签

对于输入 $X\in \mathbb R^{n\times f}, t\in\{0,1\}^{n,c}$，进行 $k$ 分类参数学习，其估计公式为：$W=(X^TX)^{-1}X^Tt$

## Probit 回归算法

### Probit 函数

广义的线性回归模型可以表示为：$p(t=1|x)=f(wx)$，其中 $f(\cdot)$ 为激活函数，而 probit 即为一种激活函数。

首先定义 erf 函数（高斯误差函数），erf 函数为：$\text{erf}(x)=\frac{2}{\sqrt{\pi}} \int_{0}^{x}exp(-\theta/2)d\theta$

probit 函数定义为：$\text{probit}(x)=\frac{1}{2}(1 + \frac{1}{\sqrt{2}}\text{erf}(x))$


### 学习算法

Porbit 回归学习使用梯度下降法进行学习，损失函数采用交叉熵。

损失函数对参数 $W$ 的梯度为：$\nabla E(x)=(y-t)x$

根据梯度下降法进行参数优化：$W^{k+1}=W^{k}-\lambda \nabla E(x)$


## 概率生成模型

概率生成模型根据对输入进行建模来进行参数估计。

首先对第 $k$ 类输入进行均值和方差估计：$\mu_k=\text{mean}(x_i),i\in C_k$

然后分别计算每类的方差：$S_k$

最后，$S=\frac{n_1}{n}S_1+\frac{n_2}{n}S_2+...+\frac{n_k}{n}S_k$

模型参数为：$W_k=\Sigma^{-1}\mu_k,w_0=-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+\ln p(c_k)$


## 实验结果

|模型|测试集错误率|训练集错误率|
|:-:|:-:|:-:|
|最小二乘|9.27%|12.5%|
|Probit|8.61%|16%|
|概率生成|9.27%|12.5%|

可以看到，概率生成模型相比 Probit 函数结果更差。
