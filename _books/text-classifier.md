---
layout:     post
title:      "文本分类算法"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 自然语言处理
    - 分类器
    - 朴素贝叶斯分类器
    - 逻辑回归
    - 最大熵模型
    - 线性对数模型
---

本文介绍经典的文本分类算法，包括朴素贝叶斯分类器、逻辑回归模型、最大熵模型。本文并不会介绍深度学习的算法，但是通过了解深度学习之前主流的算法，能够知道传统算法的问题在哪以及深度学习算法在什么时候会比传统的算法更有优势，从而可以不同的问题选择合适的算法。本文数学公式比较多，不感兴趣的读者可以忽略推导细节。更多文章请点击[深度学习理论与实战：提高篇]({{ site.baseurl }}{% post_url 2019-03-14-dl-book %})。
<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>
 <!--more-->
 
**目录**
* TOC
{:toc}

文本分类是一个很常见的任务，比如在语音助手的应用中，用户用一个句子表达自己的意图。我们通常会定义很多意图，比如"查天气"，"订机票"。这是一个短文本的分类任务。下面先介绍传统的机器学习方法，包括朴素贝叶斯分类器、逻辑回归/最大熵模型，然后分析它们的缺点，最后介绍深度学习的方法为什么能(一定程度)解决这些问题。

## 朴素贝叶斯分类器

### 简介

假定输入的特征是$x_1, x_2, ..., x_n$，输出是分类y，朴素贝叶斯模型使用贝叶斯公式计算后验概率：

$$
P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)}{P(x_1, \dots, x_n)}
$$

朴素贝叶斯模型假设给定分类y的条件下，输入特征是相互独立的，因此对于任意的i都有：

$$
P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y)
$$

把它代入前面的式子得到：

$$
P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, \dots, x_n)}
$$

因为输入是固定的，$P(x_1, \dots, x_n)$是一个常量，所以我们可以用下式来分类：

$$
\begin{split}
P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y) \\
\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y)
\end{split}
$$

当然，在实际使用中，特征会很多，概率$p(x_i \vert y)$通常会比较小，直接相乘会下溢，因此我们实际会计算/存储log的概率，把概率的乘法变成log域的加法。p(y)可以简单的用最大似然的方法从训练数据中估计出来。而$p(x_i \vert y)$可以有很多种不同的分布，从而得到不同的朴素贝叶斯模型的具体实现。我们下面介绍几种常见的朴素贝叶斯模型，主要是关注文本分类这个具体应用。


### 高斯朴素贝叶斯(Gaussian Naive Bayes)模型


$$
P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)
$$

因此它适合特征是集中分布于某个中心点的连续特征。sklearn(本书偶尔会简单的使用，但不会详细的介绍，有兴趣的读者可以找一些资料了解一下)里有一个GaussianNB类，可以用于高斯朴素贝叶斯模型，下面是代码示例：
```
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
	 % (iris.data.shape[0],(iris.target != y_pred).sum()))
输出：Number of mislabeled points out of a total 150 points : 6
```

### Bernoulli朴素贝叶斯模型

Bernoulli朴素贝叶斯模型假设似然概率服从二项分布，也就是特征$x_i$是二值特征。为了使用Bernoulli朴素贝叶斯模型，我们需要对输入进行二值化(binarize)。比如用朴素贝叶斯模型来进行垃圾邮件过滤，这是一个二分类问题。假设词典的大小是1000，那么每一个邮件都可以表示成一个1000维的向量。如果某个词在邮件里出现了(出现多次和一次没有区别)，那么对于的位置的值就是1，否则就是0。

因此$p(x_i \vert y=1)$是一个Bernoulli分布，它有一个参数$p_1$，表示$p(x_i=1 \vert y=1)=p_1$，而$p(x_i=0 \vert y=1)=1-p_1$。类似的$p(x_i \vert y=0)$也是一个Bernoulli分布，它的参数是$p_0$。注意$p_1$和$p_0$没有任何关系。

估计参数$p_1$或者$p_0$也很简单。比如我们要估计$p_1$，首先统计一下训练数据里有多少个邮件分类是1(垃圾)，记为$N_1$，假设为100；接着统计这100个邮件里有多少封出现了词1，记为$N_{11}$，假设为10。那么$\hat{p}_1=10/100$。

和语言模型一样，很多词没有在训练数据中，或者只在正/负样本里出现，从而出现概率为零的情况，我们可以采取平滑的方法。

### 多项式朴素贝叶斯模型

Bernoulli朴素贝叶斯模型的问题是不考虑一个词在文档中出现的频率，这是有一些问题的。而多项式朴素贝叶斯模型可以利用词频信息。多项式朴素贝叶斯模型也是把一篇文档变成长度为N(词典大小)的向量，某个词在文档中出现的次数就是对应位置的值。和Bernoulli相比，Bernoulli模型的特征是二值的(binary)而多项式模型的特征不是。


我们假设所有的类别集合是C，词典大小是N，第i个文档是$t_i$。那么分类的时候需要计算后验概率：

$$
Pr(c|t_i)=\frac{Pr(t_i|c)Pr(c)}{Pr(t_i)}
$$

分母与c无关，因此可以忽略，Pr(c)是分类c的先验概率，而$Pr(t_i \vert c)$是分类c"生成"文档$t_i$的概率。这个概率我们假设它服从多项式分布：

$$
Pr(t_i|c)=(\sum_nf_{ni})! \prod_n \frac{Pr(w_n|c)^{f_{ni}}}{f_{ni}!}
$$

上式中n是词典里的第n个词，$f_{ni}$表示第n个词$w_n$在文档t中出现的次数，$Pr(w_n) \vert c$是类别c的文档产生词$w_n$的概率。

上面的式子看起来很复杂，可能需要一些组合计数的知识，但是核心是$\prod_n Pr(w_n \vert c)^{f_{ni}}$，其余的阶乘都是为了归一化。我们来考虑这样一个问题：假设文档的次数包含的词已知，那么多少种可能的文档呢？

假设一篇文档有5个词，2个hello，2个world，1个bye。我们首先看5个词的所有排列数，总共有5!种不同排列组合：
```
w1 w2 w3 w4 w5
w1 w2 w3 w5 w4
w1 w2 w4 w3 w5
w1 w2 w4 w5 w3
....
w5 w4 w3 w2 w1
```
为什么是5!呢？我们复习一下高中的排列组合知识。因为w1可以任意选择5个中的一个位置，w1选定之后w2就只能从剩下的4个中选择，。。。，最后w5只有一种选择，所以共有5x4x3x2x1=5!种排列。

如果w1,...,w5都不相同，那么文档数量就是5!。但是如果有相同的词，那么我们要去掉重复的计数。假设w1=w3="hello",w2=w4="world", w5="bye"
```
w1 w2 w3 w4 w5 = hello world hello world bye
w3 w2 w1 w4 w5 = hello world hello world bye
w1 w4 w3 w2 w5 = hello world hello world bye
w3 w4 w1 w2 w5 = hello world hello world bye
```
显然"hello world hello world bye"被重复计算了4次。4是怎么算出来的呢？比如对于"hello1 world1 hello2 world2 bye1"，我们可以把相同的词随意的排列，比如这里有两个hello(hello1和hello2)和两个world，那么hello的排列为2!种，world的排列也是2!种，bye只有1!种，乘起来就是2!2!1!=4。类似的"hello hello world world bye"也重复了4次，所以两个hello、两个world和一个bye的不重复文档共有$\frac{5!}{2!2!1!}$。

如果推广一下，假设文档的词频向量是$f_1, f_2,...,f_n$，$f_i$表示第i个词出现的频次(可以为0)。那么不重复的文档数为$\frac{(\sum_{i=1}^n f_i)!}{\prod_{i=1}^{n}f_i!}$。


而我们的多项式分布假设文档的表示与顺序无关，也就是词袋(Bag of Word)模型。因此文档$Pr(t_i \vert c)$的概率可以这样来理解：如果不考虑重复，那么$Pr(t_i \vert c)=\prod_{i=1}^{n} Pr(w_n \vert c)^{f_{ni}}$，但是那些重复的都应该算到这篇文档里，因此乘起来就得到了$Pr(t_i \vert c)=(\sum_nf_{ni})! \prod_n \frac{Pr(w_n \vert c)^{f_{ni}}}{f_{ni}!}$。


在分类的时候，因为$(\sum_nf_{ni})!, \prod_n \frac{1}{f_{ni}!}$与c无关，因此可以看出归一化常量，所以只需要计算$\prod_n Pr(w_n \vert c)^{f_{ni}}$。问题的关键就是估计参数$Pr(w_n \vert c)$，我们可以用最大似然方法估计，为了避免数据稀疏带来的零概率问题，我们加入了Laplace平滑，计算公式如下：

$$
\widehat{Pr}(w_n|c)=\frac{1+F_{nc}}{N+\sum_{x=1}^NF_{xc}}
$$

举个例子，假设共有3个词，假设分类c有两篇文档
```
w1 w2 w1 
w2 w1 
```
我们可以统计出$F_{1c}=3, F_{2c}=2, F_{3c}=0$，如果不加平滑，$Pr(w_1 \vert c)=3/5, Pr(w_2 \vert c)=2/5, Pr(w_3 \vert c)=0$。如果加1平滑(Laplace平滑)，我们可以认为新的计数为$F_{1c}=4, F_{2c}=3, F_{3c}=1$，因此$Pr(w_1 \vert c)=4/8, Pr(w_2 \vert c)=3/8, Pr(w_3 \vert c)=1/8$。

因此进行参数估计需要得到一个文档-词频矩阵Count，每一行表示一个文档，每一列表示一个词，Count(ij)表示文档i中出现词j的次数。在文本分类中，我们经常用tf-idf替代原始的Count矩阵，这样做的好处是降低那些高频词的计数。这是信息检索和文本处理常见的一种方法，有兴趣的读者可以找相关资料深入了解，本书不详细介绍了。


## 逻辑回归

我们这里介绍的逻辑回归可以处理多分类的问题，而二分类的“经典”逻辑回归只是它的特例。我们假设输入是一个向量$x^T=[x_1, x_2, ..., x_n]$，假设要分为3类。逻辑回归对于每一类都有一个权重和bias。类别1的权重是$w_1^T=[w_{11}, w_{12}, ..., w_{1n}]$，类别2的权重是$w_2^T=[w_{21}, w_{22}, ..., w_{2n}]$，...。我们把每一个类的权重w和输入向量x求内积(加权求和)：

$$
\begin{split}
w_1^Tx=w_{11}x_1+w_{12}x_2+...+w_{1n}x_n + b_1 \\
w_2^Tx=w_{21}x_1+w_{22}x_2+...+w_{2n}x_n + b_2\\
w_3^Tx=w_{31}x_1+w_{32}x_2+...+w_{3n}x_n + b_3\\
\end{split}
$$

这样我们就可以得到3个数，那么怎么用这3个数分类呢？最简单的就是哪个数大就分到哪个类，这就是所谓的感知机(Perceptron)算法。但是很多时候我们需要得到分类的一个概率值，而两个向量的内积可以是从$-\infty$到$+\infty$的任意数，显然不满足我们的要求。我们可以用softmax函数把这3个数变成一个概率：

$$
\begin{split}
p(y=1|x) & =\frac{e^{w_1^Tx+b_1}}{\sum_{i=1}^{e}e^{w_i^Tx+b_i}} \\
p(y=2|x) & =\frac{e^{w_2^Tx+b_2}}{\sum_{i=1}^{e}e^{w_i^Tx+b_i}} \\
p(y=3|x) & =\frac{e^{w_3^Tx+b_3}}{\sum_{i=1}^{e}e^{w_i^Tx+b_i}}
\end{split}
$$
 
有了输出的概率，我们就可以计算预测的概率分布和真实的分布的交叉熵损失函数，然后用梯度下降学习了。如果类别数是2，我们来看看是什么情况。为了和通常的记号一致，我们把这两类的标签记为1和0。

$$
\begin{split}
p(y=1|x) & =\frac{e^{w_1^Tx+b_1}}{\sum_{i=1}^{2}e^{w_i^Tx+b_i}} \\
p(y=0|x) & =\frac{e^{w_2^Tx+b_2}}{\sum_{i=1}^{2}e^{w_i^Tx+b_i}}
\end{split}
$$

因为$p(y=0 \vert x)=1-p(y=1 \vert x)$，因此我们只看y=1的情况，我们把分子分母都除以$e^{w_1^Tx+b_1}$，可以得到：

$$
\begin{split}
p(y=1|x) & = \frac{e^{w_1^Tx+b_1}}{\sum_{i=1}^{2}e^{w_i^Tx+b_i}} \\ 
         & = \frac{1}{1+e^{-((w_1-w_2)^Tx+(b_1-b_2))}} \\
         & = \frac{1}{1+e^{-w^Tx-b}}
\end{split}
$$

我们把$w_1-w_2$用一个权重向量w来表示，而$b_1-b_2$用b来表示，这就得到了我们常见的(两类)逻辑回归模型。

## 最大熵模型

### 最大熵原理

本节的数学公式较多，不感兴趣的读者可以跳过。我们首先来复习一下熵的定义，假设X是一个离散的随机变量，P(X)是它的概率分布函数，则它的熵为：

$$
H(P)=-\sum_xP(X=x)logP(X=x)
$$

H(P)满足如下不等式：

$$
0 \le H(P) \le log|X|
$$

在上式中，$\vert X \vert$是X的取值个数。上式取等号的条件是X是均匀分布，也就是每个可能的取值概率相同的情况下熵最大。

最大熵原理的基本思想是：我们的模型首先需要满足已有的事实，在满足这些事实的约束下可能有很多概率分布(模型)，我们需要选择其中熵最大的那一个模型。我们下面通过一个例子来介绍最大熵原理。我们现在掷一枚骰子，请问出现1点到6点的概率分别是多少？显然需要满足概率的定义：

$$
P(X=1)+P(X=2)+P(X=3)+P(X=4)+P(X=5)+P(X=6)=1
$$

满足这个约束的概率分布有无穷多个，但是我们很可能会选择这个：

$$
P(X=1)=P(X=2)=P(X=3)=P(X=4)=P(X=5)=P(X=6)=\frac{1}{6}
$$

为什么会选择这个呢？有很多解释，其中之一就是它满足了上面的最大熵原理，在所有满足约束的概率分布中它的熵是最大的。

假设现在通过实验发现$P(X=1)+p(X=2)=\frac{1}{2}$，那么现在约束条件变成了2个，满足这个两个条件下熵最大的概率分布是：

$$
P(X=1)=P(X=2)=1/4, P(X=3)=P(X=4)=P(X=5)=P(X=6)=1/8
$$

把最大熵原理用到分类问题上就得到最大熵模型。最大熵模型是一个概率分类模型$P(Y \vert X)$，其中$X \in \mathcal{X} \subseteq R^n$，$Y \in \mathcal{Y}$是个有限的集合($\mathcal{X}$不要求是有限集合)。同时会给定一个训练集合$T=\{(x_1,y_1),(x_2,y_2),...(x_N,y_N) \}$。

在介绍最大熵模型的推导前，我们首先介绍拉格朗日对偶性。

### 拉格朗日对偶性

本节内容主要参考了李航的《统计机器学习》的附录部分。



#### 原始问题

假设$f(x)$, $c_i(x)$和$h_j(x)$是定义在$\mathbb{R}^n$上的连续可微函数。我们定义约束优化问题：

<a name='eq_prime'></a>

$$
\begin{split}
\underset{x \in R^n}{min} \text{  } & f(x) \\
s.t.\text{  } & c_i(x) \le 0, i=1,2,...,k \\
              & h_j(x) = 0, j=1,2,...,l
\end{split}
$$

所谓的约束优化就是最小化一个函数f(x)，但是x不能取所有的值，而只能在满足约束条件的x中寻找使得f(x)最小的x。这里的约束又分为不等式约束($c_i(x) \le 0$)和等式约束。这里只有小于等于，那大于等于怎么办？把约束函数加个负号就行了！如果是小于的约束呢？作者没有证明过，但似乎可以直接改成小于等于的约束，因为f(x)和c(x)都是连续可微函数，如果最小值在c(x)=0的点，那么离它足够小的邻域内应该有个x'满足c(x')<0而且f(x')无限接近f(x)。

我们把上面的约束优化问题叫作原始问题(Prime)。我们接着定义广义拉格朗日函数(generalized Lagrange function)：

$$
L(x,\alpha,\beta)=f(x)+\sum_{i=1}^k\alpha_ic_i(x) + \sum_{j=1}^{l}\beta_jh_j(x)
$$

这里$x \in R^n$，$\alpha_i,\beta_j$是拉格朗日乘子，并且要求$\alpha_i \ge 0$。因此L是变量$x,\alpha,\beta$的函数。接着我们考虑下面这个函数：

$$
\theta_P(x)=\underset{\alpha,\beta:\alpha_i \ge 0}{max}L(x,\alpha,\beta)
$$

注意这个$\theta_P(x)$函数是关于x的函数：给定任意x，我们在所有满足$\alpha_i \ge 0$的所有$\alpha,\beta$中寻找$L(x,\alpha,\beta)$的最大值。如果x违反原始问题的约束条件，也就是说存在$c_i(x)>0$或者$h_j(x) \ne 0$，那么就有：

$$
\theta_P(x)=\underset{\alpha,\beta:\alpha_i \ge 0}{max}L(x,\alpha,\beta)=+\infty
$$

因为如果某个$c_i(x)>0$，那么我们可以让$\alpha_i \to +\infty$；而如果某个$h_j(x) \ne 0$，那么我们也可以根据$h_j(x)$是正数还是负数取正无穷或者负无穷，从而使得$\theta_P(x) \to +\infty$

而如果x满足约束条件，为了使$\theta_P(x)$最小，$h_j(x)=0$，所有跟$\beta_j$无关，而因为$c_i(x) \le 0$，因此当$\beta_i=0$时$\theta_P(x)$最小。这时有$\theta_P(x)=f(x)$，因此，

$$
\theta_P(x)=\begin{cases}
f(x), & x\text{满足原始问题的约束} \\
+\infty & \text{其它}
\end{cases}
$$

所以无约束的优化问题：

$$
\underset{x}{\theta_P(x)}=\underset{x}{min} \underset{\alpha,\beta:\alpha_i \ge 0}{max} L(x,\alpha, \beta)
$$

与<a href='#eq_prime'>原始问题</a>是等价的，也就是说它们的解是相同的。问题$\underset{x}{min} \underset{\alpha,\beta:\alpha_i \ge 0}{max} L(x,\alpha, \beta)$成为广义拉格朗日函数的极小极大问题。这样就把原始的约束优化问题转换成广义拉格朗日函数的(无约束的)极小极大问题。为了方便，把原始问题的最优值记为$$p^*$$：

$$
p^*=\underset{x}{min}\theta_P(x)
$$


#### 对偶问题

我们先定义函数$\theta_D$：

$$
\theta_D(\alpha, \beta) = \underset{x}{min}L(x,\alpha, \beta)
$$

$\theta_D$是$\alpha, \beta$的函数，给定一组$\alpha, \beta$，它会找到L的最小值。现在$\theta_D$是$\alpha, \beta$的函数了，我们又可以求它的最大值(从所有$\alpha, \beta$的组合里寻找)：

$$
\underset{\alpha, \beta :\alpha_i \ge 0}{max}\theta_D(\alpha, \beta)=\underset{\alpha, \beta :\alpha_i \ge 0}{max} \underset{x}{min}L(x,\alpha, \beta)
$$

问题$\underset{\alpha, \beta :\alpha_i \ge 0}{max} \underset{x}{min}L(x,\alpha, \beta)$成为广义拉格朗日函数的极大极小问题。因为要求$\alpha_i \ge 0$，所以它也是一个约束优化问题：

<a name='eq_dual'></a>

$$
\begin{split}
\underset{\alpha, \beta}{max}\theta_D(\alpha, \beta)=\underset{\alpha, \beta}{max} \underset{x}{min}L(x,\alpha, \beta) \\
s.t. \alpha_i \ge 0, i=1,2,...,k
\end{split}
$$

我们把这个约束优化问题叫作对偶问题。定义对偶问题的最优值为：

$$
d^*=\underset{\alpha, \beta :\alpha_i \ge 0}{max}\theta_D(\alpha, \beta)
$$

#### 原始问题和对偶问题的关系

**定理1**：若<a href='#eq_prime'>原始问题</a>和<a href='#eq_dual'>对偶问题</a>都有最优值，那么它们满足如下关系，

$$
d^* = \underset{\alpha,\beta:\alpha_i \ge 0}{max} \underset{x}{min}L(x,\alpha, \beta) \le \underset{x}{min}\underset{\alpha,\beta:\alpha_i \ge 0}{max}L(x,\alpha,\beta)L(x,\alpha, \beta)
$$

证明：对于任意的$\alpha(\alpha_i \ge 0)$，$\beta$和$x$，有

$$
\theta_D(\alpha,\beta)=\underset{x}{min}L(x,\alpha,\beta) \le L(x,\alpha,\beta) \le \underset{\alpha,\beta:\alpha_i \ge 0}{max}L(x,\alpha,\beta)L(x,\alpha, \beta)=\theta_P(x)
$$

因此

$$
\theta_D(\alpha,\beta) \le \theta_P(x)
$$

由于原始问题与对偶问题都有最优值，所以

$$
\underset{\alpha,\beta:\alpha_i \ge 0}{max}\theta_D(\alpha,\beta) \le  \underset{x}{min} \theta_P(x)
$$

即：

$$
d^* = \underset{\alpha,\beta:\alpha_i \ge 0}{max} \underset{x}{min}L(x,\alpha, \beta) \le \underset{x}{min}\underset{\alpha,\beta:\alpha_i \ge 0}{max}L(x,\alpha,\beta)L(x,\alpha, \beta)
$$

**推论** 假设$$x^*$$和$$\alpha^*, \beta^*$$分别是<a href='#eq_prime'>原始问题</a>和<a href='#eq_dual'>对偶问题</a>的可行解(满足约束)，并且这两个可行解对应的值满足$$d^*=p^*$$。则$$x^*$$和$$\alpha^*, \beta^*$$分别是原始问题和对偶问题的最优解。用通俗的话来说：如使得$d=p$的点就是最优解，而非最优解只能严格$d<p$。

**定理2**：对于<a href='#eq_prime'>原始问题</a>和<a href='#eq_dual'>对偶问题</a>，如果函数$f(x)$和$c_i(x)$是凸函数，$h_j(x)$是仿射函数；并且不等式约束$c_i(x)$是严格可行的(也就是只是存在一个x使得所有$c_i(x)<0$)。那么一定存在$$x^*, \alpha^*,\beta^*$$，其中$$x^*$$是原始问题的最优解，而$$\alpha^*, \beta^*$$是对偶问题的最优解，并且有

$$
p^*=d^*=L(x^*,\alpha^*, \beta^*)
$$


好奇的读者可能会有这样的问题：我们通常并不是求最优值(最小的函数值)，而是要求最优解(最小的函数值对于的x)，那即使我们求出对偶问题的最优值和最优解，根据上面的定理，我们也只能求出原始问题的最优值，而不能求出最优解。因此我们需要下面的定理3。

**定理3**：如果<a href='#eq_prime'>原始问题</a>和<a href='#eq_dual'>对偶问题</a>满足上面定理2的条件，也就是函数$f(x)$和$c_i(x)$是凸函数，$h_j(x)$是仿射函数；并且不等式约束$c_i(x)$是严格可行的(也就是只是存在一个x使得所有$c_i(x)<0$)。则$$x^*$$和$$\alpha^*, \beta^*$$分别是原始问题和对偶问题的最优解的充要条件是如下的Karush-Kuhn-Tucker(KKT)条件：

$$
\label{eq:KKT}
\begin{split}
\Delta_xL(x^*,\alpha^*,\beta^*)=0 \\
\Delta_{\alpha}L(x^*,\alpha^*,\beta^*)=0 \\
\Delta_{\beta}L(x^*,\alpha^*,\beta^*)=0 \\
\alpha_i^*c_i(x^*)=0, i=1,2,...,k \\
c_i(x^*) \le 0, i=1,2,...,k \\
\alpha_i \ge 0, i=1,2,...,k \\
h_j(x^*)=0, j=1,2,...,l
\end{split}
$$

通过上面的充要条件，我们通常可以求解出$$x^*$$和$$\alpha^*,\beta^*$$的函数关系来，有了$$\alpha^*$$和$$\beta^*$$，我们就可以直接解出$$x^*$$。比如在SVM中，我们可以得到$w=\sum_{i=1}^{N}\alpha_iy_ix_i$，这里的$w$就是定理中的$x$。

### 最大熵模型的定义

有了上面的预备知识后，我们就可以来看最大熵模型是怎么推导出来的了。假设分类模型是一个条件概率分布$P(Y \vert X)$，$X \in \mathcal{X} \subseteq R^n$表示输入，$Y \in \mathcal{Y}$表示输出，我们要求$\mathcal{Y}$是个有限的集合，比如在分类中，$\mathcal{Y}$是分类的集合。

给定一个训练集合$T=\{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}$，我们的目标是用最大熵原理找到"最好"的模型。根据前面的例子，在没有任何约束的情况下，熵最大的模型就是均匀分布的概率分布。但是有了训练数据后，我们的模型需要满足训练数据的约束。数据怎么表示呢？在最大熵模型里，数据由特征向量来表示，而特征向量的每一维都是有一个特征函数来表示。在NLP里，常见的特征函数通常都是取值范围是$\{0,1\}$的二值函数：

$$
f(x,y)=\begin{cases}
1, & \text{x与y满足某种条件} \\
0, & \text{否则}
\end{cases}
$$

比如我们可以用这个模型来作为语言模型——根据之前的历史预测下一个词，那么可能会有如下的特征函数：

$$
\begin{split}
& f_1(x,y)=\begin{cases}1, & \text{如果y=model} \\ 0,  & \text{其它}\end{cases} \\
& f_2(x,y)=\begin{cases}1, & \text{如果y=model，}w_{i-1}=statistical \\ 0,  & \text{其它}\end{cases} \\
\end{split}
$$

它可以这样解读：$f_1$是如果当前要预测成单词"model"，而$f_2$是前一个词是"statistical"并且要预测成单词"model"。对于不同的特征，会对最终的预测产生影响。比较直觉的判断是——$f_2$应该会对$p(Y="model" \vert X)$有正面的影响，可以对应N-Gram的bigram $p(model \vert statistical)$。而$f_1$也会对$p(Y="model" \vert X)$有影响，类似与uni-gram p(model)。但是我们的特征可以比n-gram更加复杂，比如：

$$
\begin{split} 
& f_3(x,y)=\begin{cases}1, & \text{如果y=model，}w_{i-1}\text{是一个形容词} \\ 0,  & \text{其它}\end{cases} \\
& f_4(x,y)=\begin{cases}1, & \text{如果y=model，}w_{i-2}=any \\ 0,  & \text{其它}\end{cases} \\
\end{split}
$$

特征$f_3$利用了前一个词的词性，而$f_4$可以跳过前一个词，向前看两个词。

注意：上面的特征是关于x和y的具体取值。通常我们会定义特征“模板”，比如类似bi-gram的特征，假设有n个词，那么bi-gram的特征可能会有$n^2$个。当然如果某种组合在训练数据中没有出现过，那么就没有必要定义，因为也学习不到它的参数。此外即使在训练数据中出现了，如果频率非常低，我们也可以认为它是非常不稳定的特征，可以把它去掉。

最大熵模型的约束条件就是特征函数$f(x,y)$关于经验分布$\tilde{P}(X,Y)$的期望值等于特征函数关于模型$P(Y \vert X)$的期望值。我们首先来看经验联合分布($\tilde{P}(X,Y)$)和经验边缘分布($\tilde{P}(X)$)：

$$
\label{eq:emp-dist}
\begin{split}
& \tilde{P}(X,Y)=\frac{count(X=x,Y=y)}{N} \\
& \tilde{P}(X)=\frac{count(X=x)}{N}
\end{split}
$$

上式中$count(X=x,Y=y)$表示训练数据中(x,y)出现的频次，而$count(X=x)$表示训练数据中x出现的频次，N表示训练数据的总数。有了经验分布之后，我们就可以定义特征函数关于经验分布的期望值：

$$
E_{\tilde{P}}(f)=\sum_{x,y}\tilde{P}(x,y)f(x,y)
$$

类似的我们可以定义特征函数关于模型(也是一个分布)的期望值：

$$
E_P(f)=\sum_{x,y}P(x,y)f(x,y)=\sum_{x,y} \tilde{P}(x)P(y|x)f(x,y)
$$

上式中由于我们的模型只是建模条件概率，因此模型的联合概率分布用经验边缘分布$\tilde{P}(x)$来替代模型的边缘概率分布。最大熵模型要求特征函数关于这两个分布的期望值是相同的：

$$
\sum_{x,y}\tilde{P}(x,y)f(x,y)=\sum_{x,y} \tilde{P}(x)P(y|x)f(x,y)
$$

上面是一个特征f的约束，如果有n个特征，那么就有n个约束条件。

假设满足约束条件的模型集合为$\mathcal{C}$：

$$
\mathcal{C} \equiv \{P \in \mathcal{P}|E_P(f_i)=E_{\tilde{P}}(f_i), i=1,2,...,n \} 
$$

也就是我们把所有满足约束的模型放到集合$\mathcal{C}$里。条件概率分布$P(Y \vert X)$的条件熵为：

$$
H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)
$$

在集合$\mathcal{C}$中，条件熵最大的那个模型就叫最大熵模型。

定义了最大熵模型之后的问题就是：这样的模型是否存在；如果存在的话它是什么样的函数形式；如果它是某种函数形式的话怎么求解参数。

### 最大熵模型的函数形式

根据上面的定义，最大熵模型的求解就是一个约束优化问题：

$$
\begin{split}
 \underset{P \in \mathcal{C}}{max} \text{          } & H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x) \\
s.t. \text{          } & E_P(f_i)=E_{\tilde{P}}(f_i), i=1,2,...,n \\
& \sum_{y}P(y|x)=1
\end{split}
$$

按照优化问题的习惯，我们把它变成最小化的问题：

$$
\begin{split}
\underset{P \in \mathcal{C}}{min} \text{          } & -H(P)=\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x) \\
s.t. \text{          } & E_P(f_i)-E_{\tilde{P}}(f_i)=0, i=1,2,...,n \\
& \sum_{y}P(y|x)=1, P(y|x) \le 0
\end{split}
$$

对比前面的原始问题的定义，我们这里只有等式约束。另外可能有一点让读者困惑的是x是什么？我们这里是一个概率模型$P(y \vert x)$，给定(x,y)，则$P(y \vert x)$就是一个概率值(未知)，因此我们可以把$P(y \vert x)$看成一个变量。

这个约束优化问题只有等式约束，而没有不等式约束。为了把原问题转换成对偶问题，我们首先引入拉格朗日函数$L(P,w)$，这里的$P(y \vert x)$就是原始问题的变量(x)，而w就是对偶问题的$\beta$。

<a name='eq_me_prime'></a>

$$

\begin{split}
L(P,w) & \equiv -H(P)+w_0(1-\sum_yP(y|x)) +\sum_{i=1}^{n}w_i(E_{\tilde{P}}(f_i)-E_P(f_i)) \\
       & =\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)+w_0(1-\sum_yP(y|x)) \\
       & \text{  }+\sum_{i=1}^{n}w_i(\sum_{x,y}\tilde{P}(x,y)f_i(x,y)-\sum_{x,y}\tilde{P}(x)P(y|x)f_i(x,y))
\end{split}
$$

原始问题是：

$$
\underset{P\in \mathcal{C}}{min} \underset{w}{max} L(P,w)
$$

对偶问题是：

$$
\underset{w}{max} \underset{P\in \mathcal{C}}{min} L(P,w)
$$

由于拉格朗日函数$L(P,w)$是凸函数，因此原始问题和对偶问题的解是等价的。我们可以通过求解对偶问题来求解原始问题。根据KKT条件：

$$
\begin{split}
\frac{\partial L(P,w)}{\partial P(y|x)} & = \sum_{x,y}\tilde{P}(x)(logP(y|x)+1)-w_0-\sum_{x,y}(\tilde{P}(x)\sum_{i=1}^{n}w_if_i(x,y)) \\
& - \sum_{x,y}\tilde{P}(x)(logP(y)+1-w_0-\sum_{i=1}^{n}w_if_i(x,y))
\end{split}
$$

求偏导数把$P(y \vert x)$看成一个变量，稍微难理解的是$w_0(1-\sum_yP(y \vert x))$对$P(y \vert x)$的导数是$w_0$，我们可以把求和符号里的y改成y'，$w_0(1-\sum_yP(y \vert x))$，这里我们把x看成常量，因为这个循环中只有一个y'的值正好等于y，而其余的y'与y无关，所以导数就是$w_0$。

另外$\sum_{i=1}^{n}w_i(\sum_{x,y}\tilde{P}(x,y)f_i(x,y)-\sum_{x,y}\tilde{P}(x)P(y \vert x)f_i(x,y))$第一项与$P(y \vert x)$无关，偏导数是0，第二项很容易，我们在上面用到了两重求和的换序，对于这个不太熟悉的读者可能要熟悉一下，这是很常用的一个数学技巧，下面我们用一个简单的例子来验证一下它的正确性。

$$
\begin{split}
& \sum_{x=1}^{3}\sum_{y=1}^{3}f(x,y) \\
& =\sum_{x=1}^{3}(f(x,1)+f(x,2)+f(x,3)) \\
& =(f(1,1)+f(2,1)+f(3,1)) + (f(1,2)+f(2,2)+f(3,2)) + (f(1,3)+f(2,3)+f(3,3)) \\
& =\sum_{y=1}^{3}\sum_{x=1}^{3}f(x,y)
\end{split}
$$

其实$\sum_x\sum_y$就是先遍历y的所有可能，然后再遍历x的所有可能，在上面的例子中x有3中可能，y也是，组合起来就是9种可能。那么先遍历y再遍历x也是这9种可能。

令上面的偏导数等于0，因为$\tilde{P}(x)>0$(我们可以这样假设，但是没在训练数据中出现的x其实是不一定满足的)，所以对于任意的x,y，都有

$$
P(y|x)=e^{\sum_{i=1}^{n}w_if_i(x,y)+w_0-1}=\frac{e^{\sum_{i=1}^{n}w_if_i(x,y)}}{e^{1-w_0}}
$$

注意上式中y和w是变量，而x是常量。分母$e^{1-w_0}$与y无关，我们可以把它记作Z(x),注意不同的x(虽然是常量)对于不同的Z(x)，所有Z(x)是x的函数。
由于约束条件$\sum_yP(y \vert x)=1$，我们得到：

$$
\begin{split}
\frac{\sum_y{e^{\sum_iw_if_i(x,y)}}}{Z(x)}=1 \\
\text{因此} Z(x)=\sum_y{e^{\sum_iw_if_i(x,y)}}
\end{split}
$$

所以我们最终得到$P(y \vert x)$是如下的函数形式：

<a name='eq_me'></a>

$$

P_w(y|x)=\frac{1}{Z_w(x)}e^{\sum_{i=1}^{n}w_if_i(x,y)}
$$

其中

$$
Z_w(x)=\sum_{y}
$$

因此我们可以把$P(y \vert x)$表示成w的函数，我们把$P(y \vert x)$代入对偶问题，就可以求对偶问题的最优解$$w^*$$，然后代回来得到$P_w(y \vert x)$。我们可以发现最大熵模型的函数形式和之前的逻辑回归是很类似的，我们后面会介绍它们的等价性。

### 最大熵模型的最大似然估计

上面解决了最大熵模型的函数形式的问题，但是我们在遇到一个问题时首先定义原问题，然后要把它变成对偶问题，然后求解对偶问题。通过这一节的学习，我们会发现不用这么复杂。我们只需要定义特征函数，然后就可以得到参数形式的最大熵模型<a href='#eq_me'>公式</a>，然后我们直接对这个模型进行最大似然估计求出$$w^*$$。那然后呢？我们就可以使用梯度上升(下降)算法来找到最优的参数。后面在介绍对数线性模型(最大熵是一种常见的对数线性模型)时会介绍怎么计算梯度。

本节我们来证明对最大熵模型<a href='#eq_me'>公式</a>的最大似然估计等价于对偶问题的最优化(最大化)，也就等价于原问题的最优化(最小化)。已知训练数据的经验概率分布$\tilde{P}(X,Y)$，条件概率分布的对数似然函数为：

$$
L\_{\tilde{P}}(P_w)=\frac{1}{N} log\prod_{x,y}P(y|x)^{\tilde{P}(x,y)}=\frac{1}{N} \sum_{x,y}\tilde{P}(x,y)logP(y|x)
$$

这个公式如果难以理解，我们可以和交叉熵对比，会发现很像，它也是计算两个概率分布的差别。另外我们可以把$\tilde{P}(x,y)$的定义代入：

$$
L\_{\tilde{P}}(P_w)=\sum_{x,y} \frac{count(x,y)}{N} logP(y|x)=\frac{1}{N} \sum_{i=1}^{N}logP(y_i|x_i)
$$

最后一个式子读者肯定能够理解，因为一个数据$x_i,y_i$的似然是$P(y_i \vert x_i)$，N个数据的似然是它们的乘积$\prod_{i=1}^NP(y_i \vert x_i)$，那么对数似然就是$\sum_{i=1}^{N}logP(y_i \vert x_i)$。那怎么从倒数第二个式子推导出最后一个式子呢？其实这是下标表示的一种trick，这个技巧在多项式分布(二项式分布)里经常用到。我们可以通过一个例子来熟悉这个trick。

假设y的取值范围是$\{1,2,3\}$，x的取值范围是$\{4,5\}$，同时我们有3个训练数据$(1,4),(1,4),(2,5)$。首先看我们容易理解的表示方法：

$$
L=1/3\sum_{i=1}^{3}logP(y_i|x_i)=1/3logP(4|1)+1/3logP(4|1)+1/3logP(5|2)
$$

另外我们可以估计出经验分布：

$$
\tilde{P}(4,1)=2/3 \\
\tilde{P}(5,2)=1/3 \\
\tilde{P}(x,y)=0, \text{其它的x和y的组合}
$$

我们用倒数第二个式子来表示：

$$
L=\sum_{x,y}\frac{count(x,y)}{N} logP(y|x)=2/3logP(4|1)+1/3logP(5|2) 
$$

这下就很容易理解了。我们把<a href='#eq_me'>公式</a>代入就可以得到：

$$
\begin{split}
L\_{\tilde{P}}(P_w) & = \frac{1}{N} \sum_{x,y}\tilde{P}(x,y)logP(y|x) \\
& = \frac{1}{N} \sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y) - \frac{1}{N} \sum_{x,y}\tilde{P}(x,y)logZ_w(x) \\
& = \frac{1}{N} \sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y) - \frac{1}{N} \sum_{x}\tilde{P}(x)logZ_w(x)
\end{split}
$$

上面的最后一步推导成立是因为$Z_w(x)$是与y无关的，因此：

$$
\begin{split}
\frac{1}{N} \sum_{x,y}\tilde{P}(x,y)logZ_w(x) & =\frac{1}{N} \sum_x(sum_y(\tilde{P}(x,y))logZ_w(x)) \\
& = \frac{1}{N} \sum_x(\tilde{P}(x)logZ_w(x))
\end{split}
$$

我们再看对偶函数的最大化问题。我们把<a href='#eq_me'>公式</a>代入<a href='#eq_me_prime'>公式</a>可以得到需要最大化的函数是关于参数w的函数$\Psi$：

$$
\begin{split}
\Psi(w) & = \sum_{x,y}\tilde{P}(x)P_w(y|x)logP_w(y|x) \\
        & \text{  }+\sum_{i=1}^{n}w_i(\sum_{x,y}\tilde{P}(x,y)f_i(x,y)-\sum_{x,y}\tilde{P}(x)P_w(y|x)f_i(x,y)) \\
        & = \sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y)+\sum_{x,y}\tilde{P}(x)P_w(y|x)(logP_w(y|x)-\sum_{i=1}^{n}w_if_i(x,y)) \\
        & = \sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y) -\sum_{x,y}\tilde{P}(x)P_w(y|x)logZ_w(x) \\
        & = \sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y) - \sum_{x}\tilde{P}(x)logZ_w(x)
\end{split}
$$

推导的第一步直接使用了<a href='#eq_me_prime'>公式</a>，只不过把约束条件$\sum_yP(y \vert x)=1$用上从而减少了一项。

第二步就是前面提到的求和的换序的技巧。

第三步就是把$P(y \vert x)$代入$logP(y \vert x)$。

第四步也用到了$\sum_yP(y \vert x)=1$，和上面类似的技巧，因为$Z_w(x)$与y无关。

我们会发现除了差一个$\frac{1}{N}$，两者的优化目标都是一样的，因此直接对最大熵模型进行最大似然估计就等价于对对偶问题的最大化。

## 逻辑回归和最大熵模型的等价性

从上面两节可以看出，逻辑回归和最大熵模型的函数形式非常类似。本节我们会通过一个短文本分类的例子来说明它们两者是等价的。假设我们的任务是短文本的分类，我们先用逻辑回归来实现。假设词典的大小是2，分类的类别数是3，我们用Bag of Words(Bow)来把文本变成向量。

比如一个训练数据x可能是$x^T=[1,0,0,0,1]$，假设词典为$\{\text{足球、经济、体育、股市、娱乐}\}$，那么x代表的文本是"足球 娱乐"，因为Bow是不在意词的顺序的，所以它和"娱乐 足球"对应的向量是一样的。假设模型参数已知，那么我们可以得到给定x输出不同类别y的概率：

$$
\begin{split}
P(y=1|x)=\frac{e^{w_{11}x_1+...+w_{15}x_5}}{\sum_{i=1}^{2}e^{w_{i1}x_1+...+w_{i5}x_5}} \\
P(y=2|x)=\frac{e^{w_{21}x_1+...+w_{25}x_5}}{\sum_{i=1}^{2}e^{w_{i1}x_1+...+w_{i5}x_5}} 
\end{split}
$$

下面我们构造一个和它等价的最大熵模型。我们首先来定义特征：

$$
\begin{split}
f_1(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"足球"} \\
0 & \text{否则}
\end{cases} \\
f_2(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"经济"} \\
0 & \text{否则}
\end{cases} \\
f_3(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"体育"} \\
0 & \text{否则}
\end{cases} \\
f_4(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"股市"} \\
0 & \text{否则}
\end{cases} \\
f_5(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"娱乐"} \\
0 & \text{否则}
\end{cases} \\
f_6(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"足球"} \\
0 & \text{否则}
\end{cases} \\
f_7(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"经济"} \\
0 & \text{否则}
\end{cases} \\
f_8(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"体育"} \\
0 & \text{否则}
\end{cases} \\
f_9(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"股市"} \\
0 & \text{否则}
\end{cases} \\
f_10(x,y)=\begin{cases}
1 & \text{如果y=1并且x包含"娱乐"} \\
0 & \text{否则}
\end{cases} \\
\end{split}
$$

根据这些特征函数，我们可以构造最大熵模型：

$$
P(y=1|x)=\frac{e^{w_1f_1(x,1)+...+w_{10}f_{10}(x,1)}}{\sum_{i=1}^{2}e^{w_1f_1f(x,i)+...+w_{10}f_{10}(x,i)}} \\
P(y=2|x)=\frac{e^{w_1f_1(x,2)+...+w_{10}f_{10}(x,2)}}{\sum_{i=1}^{2}e^{w_1f_1f(x,i)+...+w_{10}f_{10}(x,i)}}
$$

看起来这两个模型不太一样？我们用把$x^T=[1,0,0,0,1]$分别代入逻辑回归模型和最大熵模型。代入逻辑回归模型后：

$$
\begin{split}
P(y=1|x)=\frac{e^{w_{11}+w_{15}}}{e^{w_{11}+w_{15}} + e^{w_{21}+w_{25}}} \\
P(y=2|x)=\frac{e^{w_{21}+w_{25}}}{e^{w_{11}+w_{15}} + e^{w_{21}+w_{25}}} 
\end{split}
$$

代入最大熵模型之后：

$$
P(y=1|x)=\frac{e^{w_1+w_5}}{e^{w_1+w_5}+e^{w_6+w_{10}}} \\
P(y=2|x)=\frac{e^{w_6+w_{10}}}{e^{w_1+w_5}+e^{w_6+w_{10}}}
$$

我们可以让$w_1=w_{11},...,w_5=w_{15},\text{   },w_6=w_{21},...,w_{10}=w_{25}$，那么这两个模型就完全相同了。

只不过逻辑回归不关注特征函数，把x表示成向量的过程就要考虑怎么提取特征，比如我们使用的Bow方法；而最大熵模型第一考虑的就是特征函数。逻辑回归考虑的特征提取一般不考虑y，但是最大熵模型的特征函数都是包含y的。我们可以任务逻辑回归的特征是一种模板，它会应用到所有的y上。但是对于上面的例子，在定义特征函数的时候，我们可以去掉$f_{10}$，这仍然是最大熵模型，这个模型在计算$P(y=2 \vert x)$的时候忽略了"娱乐"这个词；这相当于强制要求逻辑回归模型的参数$w_{25}=0$。

另外一个问题是特征函数的定义中为什么一定要有y，没有行不行？比如我们可不可以定义与分类y无关的特征函数：

$$
f_{11}(x,y)=\begin{cases}
1 & \text{x包含"经济"} \\
0 & \text{否则}
\end{cases}
$$

理论上没有什么不可以，但是我们会发现把它加进去之后对分类没有任何影响：

$$
\begin{split}
P(y=1|x)=\frac{e^{w_1f_1(x,1)+...+w_{10}f_{10}(x,1)+w_{11}f_{11}(x,1)}}{\sum_{i=1}^{2}e^{w_1f_1f(x,i)+...+w_{10}f_{10}(x,i)+w_{11}f_{11}(x,y)}} \\
P(y=2|x)=\frac{e^{w_1f_1(x,2)+...+w_{10}f_{10}(x,2)+w_{11}f_{11}(x,2)}}{\sum_{i=1}^{2}e^{w_1f_1f(x,i)+...+w_{10}f_{10}(x,i)+w_{11}f_{11}(x,y)}}
\end{split}
$$

因为$f_{11}(x,y)$和y无关，因此对于任意的x都有$f_{11}(x,1)=f_{11}(x,2)=c$，比如原来没有特征函数$f_{11}$之前：

$$
\begin{split}
P(y=1|x)=\frac{e^{0.5}}{e^{0.5}+e^{1.4}} \\
P(y=2|x)=\frac{e^{1.4}}{e^{0.5}+e^{1.4}}
\end{split}
$$
 
现在变成：

$$
\begin{split}
P(y=1|x)=\frac{e^{0.5+c}}{e^{0.5+c}+e^{1.4+c}} \\
P(y=2|x)=\frac{e^{1.4+c}}{e^{0.5+c}+e^{1.4+c}}
\end{split}
$$
 
分子分母都除以$e^c$，结果是不变的。因此这样的特征并没有任何意义！

## 对数线性模型

我们通过对最大熵模型的形式可以推广到更为一般的对数线性模型(log linear models)，这样做的好处是可以更加容易的理解CRF模型，它也是一种对数线性模型，只不过它的输出是一个序列而不是一个类别。之前我们介绍最大熵模型更多的是从模型的形式和推导角度出发的，本节我们会从实用的角度介绍为什么要实使用对数线性模型(最大熵模型就是一种对数线性模型)。

### 为什么要用线性对数模型

考虑我们之前介绍过的语音模型的问题，我们需要估计概率$P(W_i=w_i \vert W_1=w_1,...,W_{i-1}=w_{i-1})=p(w_i \vert w_1,...,w_{i-1})$。在Trigram模型里，我们假设：

$$
p(w_i|w_1,...,w_{i-1})=q(w_i|w_{i-2},w_{i-1})
$$

其中$q(w \vert u,v)$是模型的参数，我们可以用最大似然方法来估计它，并且可以使用各种平滑方法解决数据稀疏的问题。

但是Trigram语言模型有很多问题，比如下面的文本：

>Third, the notion “grammatical in English” cannot be identified in any
>way with the notion “high order of statistical approximation to En-
>glish”. It is fair to assume that neither sentence (1) nor (2) (nor indeed
>any part of these sentences) has ever occurred in an English discourse.
>Hence, in any statistical __

我们需要预测statistical后面的词是"model"的概率，假设它的下标是i，我们需要计算$P(W_i=model \vert w_1,...,w_{i-1})$。我们(人)在预测第i个词的时候除了考虑前两个词"any statistical"之外，也会考虑更早之前的词；此外我们也会考虑更多样的特征。

比如我们可能会考虑$P(W_i = model \vert W_{i-2} = any)$，它表示往前两个词是any的条件下出现model的概率。除了词之外，我们可能还会考虑词性，比如$P(W_i = model \vert pos(W_{i−1}) = adjective)$，它表示前一个词的词性是形容词的条件下出现model的概率，这样的特征使得模型的泛化能力更强，因为词性通常代表一类词，这些词可以出现在相似的上下文中。比如"good model"和"bad model"可能在训练语料中出现的次数不同，比如前者出现了比较多次，但是后者从没出现。但是因为good和bad都是形容词，那么这个特征也能帮助它学到"bad model"是有一定(甚至不低的)概率出现的。当然这里假设词的词性是知道的，后面我们会介绍怎么进行词性标注。

我们也可以使用前后缀的信息，比如$P (W_i = model \vert \text{suff4}(W_{i−1}) = ical)$，它表示前一个词是以"ical"结尾的条件下出现model的概率，根据我们的经验，这个特征通常会增加模型输出model的概率。

我们也可以写更加复杂的特征，使用"任取"($\forall$)和"不存在"($!\exists$)这样的谓词。比如$P(W_i = model \vert W_j \ne model \text{ for j }\in \{1 . . . (i − 1)
\})$表示在之前没有出现model这个词的条件下第i个词是model的概率。$P (W_i = model \vert W_j = grammatical, j \in \{1 . . . (i − 1)\})$表示在这之前出现过grammatical的条件下出现model的概率。如果要使用之前的语言模型的方法，我们可以会这样组合这些概率：

$$
\begin{split}
p(model|w_1 , . . . w_{i−1}) & = \lambda_1 \times q_1(model|w_{i-2}=any, w_{i-1}=statistical) \\
& + \lambda_2 \times q_2(model|w_{i−1} = statistical) \\
& + \lambda_3 \times q_3(model) \\
& + \lambda_4 × q_4(model|w_{i−2} = any) \\
& + \lambda_5 × q_5(model|w_{i−1} \text{是一个形容词}) \\
& + \lambda_6 × q_6(model|w_{i−1} \text{以"ical"结尾})  \\
& + \lambda_7 × q_7(model|"model" \text{没有在}w_1 , . . . w_{i−1} \text{中出现过}) \\
& + \lambda_8 × q_8(model|"grammatical" \text{在}w_1 , . . . w_{i−1} \text{中出现过}) 
\end{split}
$$

我们需要估计很多概率$q_1,...,q_8,...$，而需要学习组合的参数。但这通常是很难组合的，因为这些特征并不是完全独立的，它们是有关联的。而使用下面我们介绍的对数线性模型，就可以很容易的加入这些特征。

### 对数线性模型的定义

对数线性模型包含如下内容：

* 一个输入集合$\mathcal{X}$，可以是无限的集合
* 一个输出集合$\mathcal{Y}$，必须是有限集合
* 一个正整数d，表示特征和参数的维度
* 一个函数$f:\mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^d$，把任意的(x,y)映射成d维的特征向量
* 一个参数向量$w \in \mathbb{R}^d$

有了如上的条件，我们可以定义对数线性模型为：

<a name='eq_loglinear'></a>

$$
p(y|x;w)=\frac{e^{w \dot f(x,y)}}{\sum_{y' \in \mathcal{Y}}e^{w \dot f(x,y')}}
$$

其中分母是与y无关(但于x有关)的一个归一化函数$Z_w(x)$。为什么要叫对数线性模型呢？如果我们不看分母只看分子，那么我们对分子求对数，得到的就是一个线性模型，所以就叫对数线性模型。比如前面语言模型的问题，对应的$\mathcal{X}$是$x_1,...,x_{i-1}$，$\mathcal{Y}$是所有的词。

### 特征

对数线性模型最大的特点之一就是可以加入大量的特征，这些特征可以是相关的。对于任意输入(x,y)，特征函数f把它映射到d维的特征空间：$f(x,y)\in R^d$。在实际应用中，d通常很大，几十万到几百万都是很常见的情况。每一维$f_k(x,y)$都叫做一个特征(feature)，它表示输入x和输出y的不同属性。每个特征都有一个权值$w_k$，权值是一个大的正数，说明这个特征倾向于把x分类为y；权值是一个绝对值很大的负数，说明倾向于不分为y类；权值的绝对值很小，说明它的影响力也很小。

#### 语言模型里使用的特征

在语言模型中，输入x是$w_1w_2...w_{i-1}$，y是词典里的一个词。下面是一些特征的例子(实际比这多得多):

$$
\begin{split}
f_1(x,y) & =\begin{cases}
1 \text{  if  } y=model \\
0 \text{其它}
\end{cases} \\
f_2(x,y) & =\begin{cases}
1 \text{  if  } y = model and w_{i−1} = statistical \\
0 \text{其它}
\end{cases} \\
f_3(x,y) & =\begin{cases}
1 \text{  if  } y = model, w_{i−2} = any, w_{i−1} = statistical \\
0 \text{其它}
\end{cases} \\
f_4(x,y) & =\begin{cases}
1 \text{  if  } y = model, w_{i−2} = any \\
0 \text{其它}
\end{cases} \\
f_5(x,y) & =\begin{cases}
1 \text{  if  } y = model, w_{i−1}\text{是一个形容词} \\
0 \text{其它}
\end{cases} \\
f_6(x,y) & =\begin{cases}
1 \text{  if  } y = model, w_{i-1} \text{以"ical"结尾} \\
0 \text{其它}
\end{cases} \\
f_7(x,y) & =\begin{cases}
1 \text{  if  } if y = model, \text{"model"没有在}w_1,...,w_{i-1}\text{中出现过} \\
0 \text{其它}
\end{cases} \\
f_8(x,y) & =\begin{cases}
1 \text{  if  } if y = model, \text{"grammatical"在}w_1,...,w_{i-1}\text{中出现过} \\
0 \text{其它}
\end{cases} \\
\end{split}
$$

上面只是列举了其中的8个(类)特征，实际的特征远远大于8个。这里的8个特征可以把y换成任何一个其它词，比如总共有10万个词，那么就可以得到80万个特征。

另外比如$f_2$，我们可以把y换成80万个词中的任意一个，也可以把$w_{i-1}$换成80万中的任意一个，那么就得到6400亿个特征！当然它们大部分组合在训练数据中从来不会出现，因此即使加进去也学不到任何合适的参数，所有通常只会保留训练数据中出现过(甚至频次超过一定阈值的)组合。第一个(类)特征类似与Unigram；第二个特征类似与Bigram的特征；第三个特征类似于Trigram的特征；而第四个到第八个特征已经超出N-gram的范畴，因此它的表示能力更强，比如$f_4$跳过了前一个词，而直接看前两个词；$f_5$考虑前一个词的词性。这些特征使得对数线性模型的泛化能力更强。其实这也是判别模型（discriminative model）和生成模型（generative model）的重要区别。我们之前学习过的NB、N-gram LM、HMM和GMM都是生成模型，而最大熵等对数线性模型(包括后面介绍的CRF模型)都是判别模型。

#### 特征模板

前面我们例子中的8个特征其实是8类特征，比如类似Trigram的特征。我们没有必要一一列举所有的词，而是可以把它做出一个模板(template)：

$$
f_{N(u,v,w)}=\begin{cases}
1 \text{  if } y = w, w_{i−2} = u, w_{i−1} = v \\
0 \text{  其它}
\end{cases}
$$

我们可以把u,v,w换成任意的词的组合(前提是uvw在训练数据中出现过)。这样一个模板可能展开成几百甚至上千万个特征。
除了类似n-gram的特征，我们也可以定义前一个词的后缀的特征模板：

$$
f_{N_{suff4}(v,w)}=\begin{cases}
1 \text{  if } y=w \text{并且}w_{i-1}\text{的后缀是v} \\
0 \text{  其它}
\end{cases}
$$

#### 特征的稀疏性

上面模板展开的特征非常多，可能上千万，但是实际一个(x,y)对应的特征向量是非常稀疏的，其中大部分是0，只少部分不为零。比如拿bigram特征为例，训练数据中出现的bigram组合可能上百万，但是对于某一个(x,y)：
$$
(x,y)=(w_1,...,w_{i-1},y)
$$
这上百万个特征里只有$w_{i-1}y$对应的bigram是1，其余都是0。因此我们计算出来的特征向量通常用稀疏向量的方式存储，这样计算特征权值和特征的内积的时候速度会快很多。比如特征的维度D=10,000,000，如果用普通的向量表示，那么$w \cdot f(x,y)$需要10,000,000次乘法和9,000,000次加法。而如果用稀疏的方法表示，假设不为零的特征只有100，那么内积只需要100次乘法和99次加法。

### 对数线性模型的形式

下面我们来分析一些对数线性模型的具体形式。对于任何的输入(x,y)，其中$x \in \mathcal{X}, y \in \mathcal{Y}$，给定x的条件下模型输出y的概率为：

$$
p(y|x;w)=\frac{exp(w \cdot f(x,y))}{\sum_{y' \in \mathcal{Y}} exp(w \cdot f(x,y'))}
$$

内积$w \cdot f(x,y)$是上式的核心部分。比如我们前面的语言模型预测第i个词的概率的问题：

>Third, the notion “grammatical in English” cannot be identified in any
>way with the notion “high order of statistical approximation to En-
>glish”. It is fair to assume that neither sentence (1) nor (2) (nor indeed
>any part of these sentences) has ever occurred in an English discourse.
>Hence, in any statistical __


我们怎么用对数线性模型来预测第i个词是model的概率呢？分子我们需要计算$w \cdot f(x,model)$，而分母我们需要计算$w \cdot f(x,model), w \cdot f(x,the), w \cdot f(x,is), ...$。分母需要计算所有可能的y。假设模型参数w已知(下节会介绍怎么用数据训练参数w)。可能的计算结果为：

$$
w \cdot f(x,model) = 5.6 \\
w \cdot f(x,the) = -3.2 \\
w \cdot f(x,is) = 1.5 ...
$$

注意：内积可能取$-\infty \text{到} \infty$间的任何数。根据直觉，内积越大，那么模型输出它的概率就越大。因此上面的模型预测$p(model \vert x)$的概率会大于$p(of \vert the)$。

我们怎么想到要有这样的函数形式呢？因为我们是期望模型输出一个概率，因此需要满足$p(y \vert x)\le 0\text{ 并且} \sum_yp(y \vert x)=1$。我们希望$p(y \vert x) \le 0$，因此我们用指数函数把所有的数都变成整数，而且内积越大，指数函数后也越大。另外为了实现概率的第二个要求，我们对它们归一化。这其实就是softmax函数。

### 对数线性模型的参数估计

下面我们来讨论对数线性模型的参数估计问题。我们假设有一个训练集$(x^i,y^i), i=1,2,...,n$，其中$x^i \in \mathcal{X}, y^i \in \mathcal{Y}$。如果参数w已知，我们可以计算第i个训练数据的似然概率：

$$
p(y^i|x^i;w)=\frac{exp(w \cdot f(x^i,y^i))}{\sum_{y' \in \mathcal{Y}} exp(w \cdot f(x^i,y'))}
$$

假设每个训练数据是独立同分布的(i.i.d.)，那么我们可以计算整个训练集的似然概率：

$$
Likelihood(w)=\prod_{i=1}^{n}p(y^i|x^i;w)
$$

我们的目的是找到最近的参数w，使得L最大，为了方便计算，我们可以对它取对数，我们把对数似然记为L：

$$
L(w)=log \prod_{i=1}^{n}p(y^i|x^i;w) = \sum_{i=1}^{n}logp(y^i|x^i;w)
$$

因为对数函数是单调递增函数，因此：

$$
\underset{w}{argmax}Likelihood(w)=\underset{w}{argmax}L(w)
$$

另外为了防止过拟合，我们通常加入正则项，比如使用L2正则：

$$
L'(w)=\sum_{i=1}^{n}logp(y^i|x^i;w) - \frac{1}{2}\lambda \left\Vert w \right\Vert^2 
$$

我们最终的目标是找到$$w^*$$，使得：

$$
w^*=\underset{w}{argmax}L'(w)
$$

L'(w)是一个凸函数，因此我们可以使用(随机)梯度下降来求(全局)最优解，但梯度下降通常收敛的比较慢。对于对数线性模型，我们通常可以使用L-BFGS算法。当然前提条件是我们可以计算梯度$\frac{dL'(w)}{dw_k}$。下面我们来推导这个梯度。我们首先来计算$\frac{dL(w)}{dw_k}$，其中$L(w)=\sum_{i=1}^{n}logp(y^i \vert x^i;w)$。我们先只看求和中的第i项$(x^i,y^i)$：

$$
p(y^i|x^i;w)=\frac{exp(w \cdot f(x^i,y^i))}{\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))}
$$

因此有：

$$
log p(y^i|x^i;w)=w \cdot f(x^i,y^i) -log \sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))
$$

上式第一项对$w_k$的导数很简单：

$$
\frac{d}{dw_k}(w \cdot f(x^i,y^i))=\frac{d}{dw_k}(\sum_{j}w_jf_j(x^i,y^i))=f_k(x^i,y^i)
$$

接着求第二项：

$$
\begin{split}
& \frac{d}{dw_k} log \sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y')) \\
& = \frac{\frac{d}{dw_k}\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))}{\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))} \\
& = \frac{\sum_{y' \in \mathcal{Y}}\frac{d}{dw_k}exp(w \cdot f(x^i,y'))}{\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))}\\
& = \frac{\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))\frac{d}{dw_k}(w \cdot f(x^i,y'))}{\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))}\\
& =\frac{\sum_{y' \in \mathcal{Y}}f_k(x^i,y')exp(w \cdot f(x^i,y'))}{\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))} \\
& =\sum_{y' \in \mathcal{Y}}(f_k(x^i,y') \times \frac{exp(w \cdot f(x^i,y'))}{\sum_{y' \in \mathcal{Y}}exp(w \cdot f(x^i,y'))}) \\
& =\sum_{y' \in \mathcal{Y}}	f_k(x^i,y') p(y'|x;w)
\end{split}
$$

每一步推导都比较直观，第5步往第6步的推导需要注意的是分母因为对y'求和了，因此分母是和y'无关的一个常量。把这两项放到一起，我们就可以得到$\frac{dL(w)}{dw_k}$：

$$
\frac{dL(w)}{dw_k}=\sum_{i=1}^{n}f_k(x^i,y^i) -\sum_{i=1}^{n}\sum_{y \in \mathcal{Y}} p(y|x;w)f_k(x^i,y)
$$

我们来分析一些上式，第一项是特征$f_k$在训练数据中值为1出现的次数(假设$f_k$都是非零即一的函数)；而第二项是特征$f_k$的值为1的期望次数(关于条件概率$p(y \vert x)$的期望)。如果这个两个值相同，则梯度为零，从而参数$w_k$不需要变动。因此模型学到的w就是使得这两个值相等。接下来我们加入正则项：

$$
\label{eq:loglinear-derivative}
\frac{dL'(w)}{dw_k}=\sum_{i=1}^{n}f_k(x^i,y^i) -\sum_{i=1}^{n}\sum_{y \in \mathcal{Y}} p(y|x;w)f_k(x^i,y) - \lambda w_k
$$

有了梯度的计算公式之后，我们就可以用梯度下降或者L-BFGS算法来求解对数线性模型的最优解了。最大熵模型(逻辑回归模型)作为它的一个特例也自然的可以求解了。


## 传统机器学习算法的缺点

传统机器学习算法最大的问题就是特征的稀疏导致的泛化能力差。比如训练数据中有"北京的温度"，它可能学到"温度"与意图查天气有关系，但是如果训练数据没有出现过"气温"，那么它很可能无法正确的识别"上海的气温怎么样"这个句子。因为我们提取的特征都是以词为基本单位的，两个语义相似的词在模型看来完全就是两个无关的词。当然我们可以人工提取一些同义词的特征，但是词的语义关系不只是同义词，它还包括反义、上下位等等复杂的语义关系。另外一些任务依赖词的顺序或者句子的结构，比如情感分类任务，基于BoW等特征的方法无法很好的解决这些问题。

## 深度学习的优点

深度学习通过Embedding一定程度能够解决泛化能力差的问题，尤其是通过无监督的Pretraining可以低成本的学习出词之间的语义关系。另外RNN等模型可以解决词的顺序和句子结果的问题，从而可以获得更好的效果。因此对于这类文本分类问题，标准的算法如下：首先可以使用通过Embedding把one-hot表示变成稠密低维的表示，然后再使用多层的CNN/LSTM等模型来学习上下文相关的语言表示。这样的例子在基础篇的循环神经网络部分我们已经介绍过了，这里就不再赘述了。另外随着BERT等Contextual Word Embedding的出现，作者更建议直接使用BERT模型来进行文本分类，在训练数据不是特别大的情况下，使用BERT的模型的效果要比传统的LSTM等好很多。但是BERT模型的缺点是模型太复杂，参数太多，因此预测的时间也会比较慢。针对具体的数据集和任务，读者通常可以用LR/最大熵模型作为一个Baseline，然后再尝试一下CNN/LSTM等深度学习模型。如果计算资源足够，那么就直接使用BERT模型。






