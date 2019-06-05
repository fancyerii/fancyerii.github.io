---
layout:     post
title:      "机器翻译"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 自然语言处理
    - 机器翻译
---

本文介绍机器翻译，包括经典的IBM翻译模型、基于短语的机器翻译和GNMT。更多文章请点击[深度学习理论与实战：提高篇]({{ site.baseurl }}{% post_url 2019-03-14-dl-book %})。
<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>
 <!--more-->
 
**目录**
* TOC
{:toc}


## IBM模型

### 统计机器翻译简介

IBM翻译模型是1980年代晚期到1990年代提出的，它是现代很多统计翻译模型的基础。根据习惯，我们假设翻译任务是把法语的句子翻译成英语的句子。法语是源(Source)语言，而英语是目标(Target)语言。我们用字母f表示法语的句子，$f_1,...,f_m$是m个法语单词；e表示英语的句子，$e_1,...,e_l$表示l个英语单词。

统计机器翻译需要双语的句对$(f^k,e^k), k=1...n$，表示n个训练句对。比如$f^k$是第k个法语句子，而$e^k$是它的翻译。$f^k=f_1^k,f_2^k,...,f_{m_k}^k$，其中$m_k$是第k个法语句子包含的词的个数；而$e^k=e_1^k,e_2^k,...,e_{l_k}^k$，其中$l_k$是第k个英语句子包含的词的个数。

### Noisy-Channel方法

IBM models用Noisy-Channel方法来建模翻译过程。法语句子f经过一个有噪音的信道被变成e，我们需要寻找$$\underset{e}{argmax}p(e \vert f)$$，根据贝叶斯公式：

$$
e^*=\underset{e}{argmax}p(e|f)=\underset{e}{argmax}\frac{p(f|e)p(e)}{p(f)}=\underset{e}{argmax} p(f|e)p(e)
$$

它包含两个模型：

* p(e) 英语的语言模型，它的训练数据不限定于双语的句对，可以是大量未标注的文本
* p(f\|e) 英语到法语的翻译模型，需要根据双语句对来学习模型的参数。我们需要计算英语句子$e_1,...,e_{l}$翻译成法语句子$f_1,...,f_m$的概率，IBM模型把翻译分成两步：首先估计法语句子的长度m，然后再生成m个词。这是一个产生式模型。


注意：虽然我们需要的是法语到英语的翻译系统，我们这里的翻译模型却是英语到法语的概率！语言模型之前我们已经介绍过了，这里只关注翻译模型。它包括两个问题：

* 怎么定义翻译模型$p(f \vert e)$？
* 怎么根据训练句对估计模型的参数？


下面我们介绍IBM模型是怎么解决这两个问题的。虽然IBM模型是很老的模型，但是它仍然是现代很多统计翻译的基础。因为它：

* 它使用了对齐(align)的技术，它可以学习出对齐模型，这是很多统计翻译模型的基础
* 它使用EM算法来自动实现对齐，而不需要人工标注法语的某个单词和英语的某个单词对齐


### 对齐
现在我们来解决$p(f \vert e)=p(f_1...f_m \vert e_1...e_l)$，我们把这个概率分成两个部分：

$$
p(f|e)=p(f_1...f_m|e_1...e_l)=p(f_1,...,f_m|m,e_1...e_l)p(m|e_1...e_l)\\
=p(f_1,...,f_m|m,e_1...e_l)p(m|l)
$$

我们假设翻译后的法语句子的长度m只与英语句子长度l有关系，也就是$p(m \vert l)$，因此问题的关键就是给定英语句子和法语句子的长度m，计算$p(f_1,...,f_m \vert m,e_1...e_l)$。这个概率也很难直接计算，为了便于计算，我们引入m个"对齐"变量$a_1,...,a_m$。其中$0 \le a_i \le l$，说明法语单词$f_i$对齐的英语单词的下标。如果$a_i=0$则说明没有任何英语单词与之对应，为了简单，我们假设它对齐到特殊的单词"NULL"。

我们会先计算$p(f_1,...,f_m,a_1,...,a_m \vert e_1,...,e_l,m)$，这是f和a的联合概率分布，然后对所有的$a_i$求和就可以得到$p(f_1,...,f_m \vert m,e_1...e_l)$：

$$
p(f_1,...,f_m|e_1...e_l,m)=\sum_{a_1=0}^{l}\sum_{a_2=0}^{l}...\sum_{a_m=0}^{l}p(f_1,...,f_m,a_1,...,a_m|e_1,...,e_l,m)
$$

$p(f_1,...,f_m \vert e_1...e_l,m)$也可以写成$p(f_1,...,f_m \vert e_1...e_l)$，因为$f_1...f_m$已经隐含了法语句子的单词数是m个。

我们下面来通过例子来熟悉对齐的概率。简单来说，$a_i$说明了$f_i$”对齐“到$e_{a_i}$。也可以认为我们把英语单词$e_{a_i}$翻译成法语单词$f_i$。比如：
```
e = And the programme has been implemented
f = Le programme a ete mis en application
```

这里英语句子的长度l=6，法语的长度m=7。假设对齐变量为：$a_1 , a_2 , . . . , a_7 = <2, 3, 4, 5, 6, 6, 6>$，那么实际的对齐为：

$$
\begin{split}
Le & \Rightarrow the \\
Programme & \Rightarrow program \\
a & \Rightarrow has \\
ete & \Rightarrow been \\
mis & \Rightarrow implemented \\
en & \Rightarrow implemented \\
application & \Rightarrow implemented
\end{split}
$$

注意这里每一个法语单词只会对齐到一个英语单词(或者0表示没有对齐)，这是一种多对一的对齐关系——可以多个法语单词对齐到一个英语单词，比如这个例子3个法语单词都对齐到一个英语单词application；但是不能多个英语单词对齐到一个法语单词。这显然是有问题的。

前面的对齐是比较合理的对齐，模型应该计算出比较大的概率。我们也可以这样对齐：$a_1 , a_2 , . . . , a_7 = <1, 1, 1, 1, 1, 1, 1>$，它是把所有的法语单词都对齐到英语单词"and"，这显然是不合理的，因此模型应该给予很低的概率。


### 对齐模型——IBM模型2

下面我们介绍$p(f_1,...,f_m \vert e_1...e_l,m)$，这里介绍的模型被叫作IBM模型2。

假设英语单词集合是一个有限的集合$\mathcal{E}$，法语词典集合是一个有限的集合$\mathcal{F}$，L和M分别表示英语和法语句子的最大长度。模型的参数为：

* $t(f \vert e), f \in \mathcal{F}, e \in {NULL} \cup \mathcal{E}, $。因此$t(f \vert e)$可以认为把e翻译成f的概率。
* $q(j \vert i, l, m), l \in \{1 . . . L\}, m \in \{1 . . . M \}, i \in \{1 . . . m\}, j \in \{0 . . . l\}$。参数$q(j \vert i, l, m)$可以看成给定句子长度l和m的条件下，第i个法语词对齐到第j个英语词的概率。这个概率统计的是位置的信息，比如法语的第一个词更可能对齐英语的前几个词，而与词本身无关。
 
给定上述参数的条件下：

$$
p(f_1 . . . f_m , a_1 . . . a_m |e_1 . . . e_l , m) =\prod_{i=1}^{m}q(a_i |i, l, m)t(f_i |e_{a_i} )
$$

我们下面通过一个例子来看IBM模型2是怎么计算这个概率的。这里的句子和对齐还是之前的例子：

$$
\begin{split}
Le & \Rightarrow the \\
Programme & \Rightarrow program \\
a & \Rightarrow has \\
ete & \Rightarrow been \\
mis & \Rightarrow implemented \\
en & \Rightarrow implemented \\
application & \Rightarrow implemented
\end{split}
$$

我们来看概率的计算：

$$
\begin{split}
& p(f_1 . . . f_m , a_1 . . . a_m |e_1 . . . e_l , m) \\
& = q(2|1, 6, 7) \times t(Le|the) \\
& \times q(3|2, 6, 7) \times t(Programme|program) \\
& \times q(4|3, 6, 7) \times t(a|has) \\
& \times q(5|4, 6, 7) \times t(ete|been) \\
& \times q(6|5, 6, 7) \times t(mis|implemented) \\
& \times q(6|6, 6, 7) \times t(en|implemented) \\
& \times q(6|7, 6, 7) \times t(application|implemented)
\end{split}
$$


因此每个法语单词对于两个概率，一个是它对齐的英文单词翻译成它的概率；另一个是它所在位置和对齐的英语单词的下标的概率。比如$f_5 = mis$，它的对齐$a_5 = 6$，因此有一个概率q(6\|5, 6, 7)表示法语的第5个词对齐到英语第6个词的概率(给定条件l和m)；此外第6个英语单词是$e_6 = implemented$，因此还有一个概率t(mis\|implemented)，它表示把implemented翻译成mis的概率。

### IBM模型2的独立性假设

通过上面的介绍，我们可以发现IBM模型2是一个非常简单的模型，但是它确实建模了翻译的一些重要信息，下面我们来简单的分析一下这个模型都用到了哪些假设来简化翻译模型。假设英语句子长度L是一个随机变量，法语句子长度M也是一个随机变量。法语的单词是$F_1...F_M$，英文的单词是$E_1...E_L$。对齐变量是$A_1...A_M$。我们的目标是建模概率分布：

$$
P (F_1 = f_1 . . . F_m = f_m , A_1 = a_1 . . . A_m = a_m |E_1 = e_1 . . . E_l = e_l , L = l, M = m)
$$

我们首先使用链式法则把它分解成两个概率：

$$
\begin{split}
& P (F_1 = f_1 . . . F_m = f_m , A_1 = a_1 . . . A_m = a_m |E_1 = e_1 . . . E_l = e_l , L = l, M = m) \\
= & P (A_1 = a_1 . . . A_m = a_m |E_1 = e_1 . . . E_l = e_l , L = l, M = m) \\
& \times P (F_1 = f_1 . . . F_m = f_m |A_1 = a_1 . . . A_m = a_m , E_1 = e_1 . . . E_l = e_l , L = l, M = m)
\end{split}
$$

我们首先来看第一项：

$$
\begin{split}
& P (A_1 = a_1 . . . A_m = a_m |E_1 = e_1 . . . E_l = e_l , L = l, M = m) \\
= & \prod_{i=1}^{m}P (A_i = a_i |A_1 = a_1 . . . A_{i−1} = a_{i−1}, E_1 = e_1 . . . E_l = e_ l , L = l, M = m) \\
= & \prod_{i=1}^{m}P (A_i = a_i |L = l, M = m)
\end{split}
$$

第一个等式使用了链式法则，第二个等式是一个非常强的独立性假设——对齐变量$A_i$与英文单词无关，至于句子长度L和M有关。因此它统计(建模)的是这样的概率：在给定英语句子和法语句子长度的条件下，法语的第i个词对齐到英语词的概率分布。这里没有任何词的信息，只是句子长度的统计。

因此$ \prod_{i=1}^{m}P (A_i = a_i \vert L = l, M = m)=q(a_i \vert i, l, m)$。再看第二项：

$$
\begin{split}
& P (F_1 = f_1 . . . F_m = f_m |A_1 = a_1 . . . A_m = a_m , E_1 = e_1 . . . E_l = e_l , L = l, M = m) \\
= & \prod_{i=1}^{m}P (F_i = f_i |F_1 = f_1 . . . F_{i-1} = f_{i-1} , A_1 = a_1 . . . A_m = a_m , E_1 = e_1 . . . E_l = e_l , L = l, M = m) \\
= & \prod_{i=1}^{m}P (F_i = f_i |E_{a_i} = e_{a_i})
\end{split}
$$

第一个等式还是使用链式法则，而第二个等式假设法语的第i个词之于它对齐的那个英语词有关系。因此$P (F_i = f_i \vert E_{a_i} = e_{a_i})=t(f_i \vert e_{a_i})$。


### IBM模型2的应用

有了模型之后，假设参数也已知(后面我们会介绍怎么估计参数)，我们就可以用它来进行翻译了。根据上面的定义，我们有了概率q(j\|i, l, m)和t(f\|e)，我们就可以计算p(f, a\|e)，从而可以计算：

$$
p(f|e)=\sum_ap(f,a|e)
$$

然后我们可以遍历所有可能的e，寻找出概率最大的$$e^*$$作为f的翻译：

$$
e^*=\underset{e}{argmax}p(e|f)=\underset{e}{argmax}p(f|e)p(e)
$$

当然这只是理论上的，实际上搜索的空间会非常大需要近似算法。实际的统计机器翻译系统已经很少直接使用IBM模型2了，但是它有两个特点：

* 概率t(f\|e)在很多实际的统计翻译系统中被使用
* 它找到的翻译对应的对齐方式被很多实际系统用到


第一点涉及到参数估计，我们后面会讲到。这里我们详细来看IBM模型2怎么实现对齐的。也就是给定英语和法语句子，找出最可能的对齐方式：

$$
\underset{a_1,...,a_m}{argmax}p(a_1,...,a_m|f_1,...,f_m,e_1,...,e_l,m)
$$

根据模型的假设，每个词的翻译是与其它词无关的，因此我们可以简单的找每一个法语词的最优对齐方式：

$$
a_i=\underset{j \in \{0...l\}}{argmax}(q(j|i,l,m) \times t(f_i|e_j))
$$

用自然语言处理来描述就是：遍历0到l这l+1中对齐方式j，选择$q(j \vert i,l,m) \times t(f_i \vert e_j)$最大的哪个。它会考虑两个因素：一个是$e_j$翻译成$f_i$的概率，这个越大越好；另外一个就是q(j \| i,l,m)，它是法语的第i个词对齐到英语的第j个词的概率。

### IBM模型2的参数估计

接下来我们看怎么估计模型参数t(f\|e)和q(j\|i, l, m)，这里要用到EM算法。和HMM一样，我们不会形式化的介绍EM算法，而是用更直觉(非正式)的方式介绍怎么用EM算法估计。这里的隐变量是对齐变量$a_i$，如果它是已知的，那么这两个参数就可以很容易的最大似然估计：

$$
\begin{split}
t_{ML}(f|e)=\frac{c(e,f)}{c(e)} \\
q_{ML}(j|i,l,m)=\frac{c(j|i, l, m)}{c(i, l, m)}
\end{split}
$$

其中$c(e)$表示训练数据中英语单词e被对齐的次数，$c(e,f)$是训练数据中f对齐到e的次数；c(j\|i, l, m)是长度为l的英文句子、长度为m的法语句子，法语句子的第i个词对齐到英语第j个词的次数；c(j\|i, l, m)是长度为l的英文句子、长度为m的法语句子对出现的次数。

具体是算法如下图所示。注意算法是3重循环：下标k表示第k个句对；i表示法语单词的下标；j表示英语单词的下标。在这里，$f_i$只能对齐到某一个$e_j$，因此第3重循环只有一个$\delta(k,i,j)=1$，其余都是零。因此这么写更加高效一些：


<a name='1'>![](/img/mt/1.png)</a>
*图：对齐变量已知的情况下IBM模型2的参数估计算法*

但是下图的写法更容易扩展到EM算法里。
 

<a name='ibm2-1'>![](/img/mt/ibm2-1.png)</a>
*图：对齐变量已知的情况下IBM模型2的参数估计算法*


和HMM类似，问题的关键是我们并不知道法语单词是怎么对齐到英语单词的。如果我们一种模型的参数t(f\|e)和q(j\|i, l, m)，我们可以最优的对齐方式：

$$
P (A_i = j|e_1 . . . e_l , f_1 . . . f_m , m)=\frac{q(j|i, l, m)t(f_i |e_j )}{\sum_{j=0}^{l_k}q(j|i, l, m)t(f_i |e_j ) }
$$

这个公式可以这样解读：分子是把法语的第i个词对齐到英语的第j个词”概率“；而分母是一个归一化项，它是法语第i个词对齐到英语所有词的”概率“和。

有了这个$P (A_i = j \vert e_1 . . . e_l , f_1 . . . f_m , m)$，我们就可以用它来替换$\delta(k,i,j)$，这是一种soft的对齐，原来的$\delta(k,i,j)$只有一个j是1，其余的是0；而现在的$P (A_i = j \vert e_1 . . . e_l , f_1 . . . f_m , m)$是一个概率分布，可以加起来是1但是非零项很多。

这样，我们就可以得到如下图所示的算法。这里需要经过多次(S)次迭代，或者直到模型收敛。每次迭代用老的模型参数计算新的统计量，然后再用新的统计量更新模型参数。
 

<a name='ibm2-2'>![](/img/mt/ibm2-2.png)</a>
*图：IBM模型2的EM算法*

### IBM模型1

IBM模型1是一个更加简化的模型，因此效果比模型2更差(模型2已经就不怎么样了)。那介绍它除了回顾历史还有什么实际作用呢？因为IBM模型2的EM算法会收敛到局部最优解，因此模型的初始化参数比较重要。所以我们可以使用模型1的参数作为模型2的初始化参数。

IBM模型1也有两个概率q(j\|i, l, m)和t(f\|e)，t(f\|e)和IBM模型2一样，但是它的q(j\|i, l, m)更加简单：

$$
q(j|i, l, m)=\frac{1}{1+l}
$$

也就是说第i个法语词对齐到第j个英语词是均匀分布的，比如英语单词有3个，在加上NULL，那么q(j\|i,l,m)=1/4。这个概率与i，j，m都没有关系，只依赖于英语句子的长度l。因此这个概率是不需要用数据估计的，它是直接人为定义的。我们可以先用EM算法估计IBM模型1的参数t(f\|e)，然后把它作为IBM模型2的初始化值。


## 基于短语的(phrase based)统计机器翻译

基于短语的翻译模型是目前统计机器翻译state of the art的技术，下面我们来介绍一些基于短语的翻译系统。前面介绍的IBM翻译模型的基本单元只能是词，而基于短语的机器翻译系统的基本单元除了词也可以是短语。

### 基于短语的词典(lexicon)

我们首先定义基于短语的词典。一个基于短语的词典$\mathcal{L}$是一个集合，它的每一个元素是一个三元组(f, e, g)，其中：
* f是一个或者多个法语词
* e是一个或者多个英语词
* g是一个实数值的得分，我们可以理解为把e翻译成f的"概率"。
 

注意这里并不要求f和e包含的词个数相同，比如：
```
(au, to the, 0.5)
(au banque, to the bank, 0.01)
(allez au banque, go to the bank, −2.5)
```
我们也可以发现g并不是严格的"概率"，它可以是负数。我们之后会介绍怎么用这样的词典来进行机器翻译，但在这之前，我们先看看怎么构建这个词典。


### 通过双语句对学习短语词典

和之前一样，我们的训练数据是一些双语句对，包括法语句子$\{f^k\}, k=1,2,...,n$和对应的英语句子$\{e^k\}$，其中每个$e^k=e^k_1...e^k_{l_k}$，每个$f^k=f^k_1...f^k_{m_k}$。这里$l_k$和$m_k$分别表示第k个英语句子和法语句子的长度(单词个数)。

除此之外，我们还假设对于每一个句对$e^k$和$f^k$，我们有一个对齐矩阵(alignment matrix)$A^k$。$A^k$是一个$l_k \times m_k$的矩阵，它的定义是：

$$
A^k_{ij}=\begin{cases}
1 \text{ if 第i个英语词和第j个法语词是对齐的} \\
0 \text{ 否则}
\end{cases}
$$

注意这个对齐矩阵比IBM模型的对齐变量更加通用，IBM模型的一个法语词只能对齐到一个英语词；而这里没有这样的限制，一个法语词可以对齐到多个英语词，当然一个英语词也可以对齐到多个法语词。

我们这里假设这个对齐矩阵是已知的，比如我们可以用IBM模型来得到这个对齐矩阵（实际会用更复杂的方法来得到对齐矩阵，IBM模型得到的对齐矩阵一个法语词只能对应到一个英语词）。

假设我们有了这个对齐矩阵，那么怎么获得短语词典呢？我们可以用一些启发式的规则来抽取短语词典，比如下图所示的算法。这个算法会遍历所有可能的短语对，它会有两个下标对$(s,t)$和$(s',t')$。比如下面的句对：

$$
f^k = \text{ wir müssen auch diese kritik ernst nehmen} \\
e^k = \text{ we must also take these criticisms seriously}
$$


<a name='pbmt-1'>![](/img/mt/pbmt-1.png)</a>
*图：计算短语词典的算法*

如果(s, t) = (1, 2), (s', t') = (2, 5)，那么对于的短语对就是"wir müssen, must also take these"。算法会遍历所有可能的短语对，然后用一个consistent函数判断它们是否"一致"，如果一致，那么就把它们加到$\mathcal{L}$中，并且更新计数器$c(e),c(e,f)$。遍历结束后用$log\frac{c(e,f)}{c(e)}$作为短语对的"概率"。

consistent函数的实现如下图所示。它的思路其实很简单：比如短语对"wir müssen, must also take these"，它会确保法语的每一个词(wir和müssen)，与它对齐的英语词都在"must also take these"里，只要有一个不在，就不一致；类似的也会反过来确保每一个英语词对齐的法语词都是在"wir müssen"。另外为了防止英语和法语短语都对齐到NULL的情况(空集合不会不一致，因为它根本没有对齐的词)，这个函数要求法语短语中至少有一个词适合英语短语中的一个词对齐的。




<a name='pbmt-2'>![](/img/mt/pbmt-2.png)</a>
*图：consistent函数*
 

### 基于短语的翻译系统

我们首先介绍短语(phrase)和推导(derivation)两个概念。这里用德语到英语的翻译为例子，德语句子是f(french -> foregin language，还是可以用缩写f)，英语句子是e。比如德语句子：

```
wir müssen auch diese kritik ernst nehmen
```

假设根据之前的方法我们以及抽取了很多短语对，比如：

```
(wir müssen, we must)
(wir müssen auch, we must also)
(ernst, seriously)
```

以及得分$log\frac{c(e,f)}{c(e)}$，这可以任务是英语句子e翻译成德语句子f的log概率。假设要翻译的德语句子是$x_1...x_n$，我们把三元组p=(s,t,e)的定义为短语(phrase)，它表示把德语短语$x_s...x_t$翻译成英语e。比如短语(1, 2, we must)表示把德语句子的一部分"wir müssen"翻译成英语"we must"。对于一个短语p，我们用s(p),t(p),e(p)来"提取"它的s,t和e。对于输入的德语句子，我们可以获得所有可能的短语p，把它们放到一个集合$\mathcal{P}$里。

注意集合$\mathcal{P}$里会有重叠的部分，比如：

```
(1,2, we must)
(1,3, we must also)
```

第一个是被第二个包含的。接下来我们定义推导(derivation)为短语的序列。推导$y=p_1,p_2,...,p_L$，其中$p_j \in \mathcal{P}$。

比如y = (1, 3, we must also), (7, 7, take), (4, 5, this criticism), (6, 6, seriously)，它是一个推导。我们用e(y)表示把每个短语p的英语字符串拼接起来。比如上面的y，其e(y)="we must also take this criticism seriously"。

上面的推导其实就是一种翻译结果。但是并不是所有推导都是合理的(valid)，下面我们来定义什么推导是合理的。对于输入的德语句子$x=x_1x_2...x_n$，一个推导(短语序列)$p_1p_2...p_L$是合理的推导需要满足如下条件：

* 对于每一个$p_k, k \in \{1,2,...,L\}$，一定有$p_k \in \mathcal{P}$，这里$\mathcal{P}$是根据输入x得到的短语集合
* 每一个德语词被且仅被翻译过一次。我们可以用数学语言更加准确的描述。对于一个推导$y=p_1p_2...p_L$，我们可以定义
	
	$$
		y(i)=\sum_{k=1}^{L}[[ s(p_k) \le i \le t(p_k) ]]
	$$

	其中函数$[[\pi]]$是indicator函数，如果条件$\pi$是true，那么它的值是1，否则是0。
	
	因此y(i)就是德语的第i个词被短语包含的次数，也就是被翻译过的次数。我们要求对于任意的$i=1...n, \; y(i)=1$。
	
* 对于所有的$k \in \{1...L-1\}$，有$\vert t(p_k)+1-t(p_{k+1}) \vert \le d$，这里d是一个超参数，此外我们还要求$\vert 1-s(p_1) \vert \le d$。


前两个条件很直观，第3个条件要求前后两个短语不能隔得太远，因为大部分情况下我们是按照顺序来翻译的(其实也不一定？)。我们通过一个例子来看第3个条件。比如下面的一个推导：

```
y = (1, 3, we must also), (7, 7, take), (4, 5, this criticism), (6, 6, seriously)
```

我们把y记作$y=p_1p_2p_3p_4$，假设d=4，我们逐个来检查：

$$
\begin{split}
|t(p_1 ) + 1 − s(p_2 )| = |3 + 1 − 7| = 3  \\
|t(p_2 ) + 1 − s(p_3 )| = |7 + 1 − 4| = 4 \\
|t(p 3 ) + 1 − s(p 4 )| = |5 + 1 − 6| = 0 \\
|1 − s(p_1 )| = |1 - 1| = 0
\end{split}
$$

显然它们都小于等于4，因此这个推导是合理的。而下面的推导：

```
y = (1, 2, we must), (7, 7, take), (3, 3, also), (4, 5, this criticism), (6, 6, seriously)
```

我们能发现$\vert t(p_2 ) + 1 − s(p_3 ) \vert = \vert 7 + 1 − 3 \vert = 5$，它不满足小于等于4的条件，因此不是合理的推导。

合理推导的最后一个条件有两个作用：

* 减少搜索空间 不合理的推导就不要作为候选的翻译方案了
* 提供翻译质量 实践证明增加这个条件可以提高翻译质量


但是超参数d怎么选择其实是很tricky的，不同的语言(对)可以有不同的最优值。

接下来我们需要有一个函数f来给一个合理的推导y打分。如果有了这个打分函数，我们的翻译实现了——从所有合理的推导中选择得分最高的推导作为翻译：$\underset{y \in \mathcal{Y}(x)}{argmax}f(y)$。

打分函数的定义如下：

$$
f(y)=h(e(y))+\sum_{k=1}^Lg(p_k)+\sum_{k=1}^{L-1}\eta \times |t(p_k ) + 1 − s(p_{k+1})|
$$

这个函数共分为3个部分，我们逐个来看：

* e(y)是翻译后的英文，h(e(y))是log域的语言模型得分。
* $g(p_k)$是短语的得分，比如g((1,2, we must))就是把"wir müssen"翻译成"we must"的log概率。
* $\eta$是个负数，而$\vert t(p_k ) + 1 − s(p_{k+1}) \vert $表示第k个短语的结束下标和第k+1个短语的开始下标的距离，我们期望它们越小越好。因此距离越小，乘以一个负数就越大。


有了上面的介绍，我们就可以定义基于短语的翻译模型了。基于短语的翻译模型是一个四元组$(\mathcal{L}, h, d, η)$，其中：

* 其中$\mathcal{L}$是前面我们定义的短语词典，它的每一个元素是一个三元组(f,e,g)。f是源语言(法语/德语)的短语；e是目标语言的短语；g是log概率
* h是一个n-gram语言模型，比如trigram。对于英语句子$e_1e_2...e_m$，$h(e_1...e_m)=\sum_{i=1}^{m}logq(e_i \vert e_{i-2} , e_{i-1})$。
* d是一个非负的整数，表示合法推导的最大的错位(distortion)
* $\eta \in R$是错位的乘法参数


对于输入源语言句子$x=x_1...x_n$，假设$\mathcal{Y}(x)$是模型$(\mathcal{L}, h, d, η)$下所有合法的推导的集合。基于短语的翻译就是寻找$$\underset{y \in \mathcal{Y}(x)}{argmax}f(y)$$。假设$y=p_1...p_L$，$f(y)=h(e(y))+\sum_{k=1}^Lg(p_k)+\sum_{k=1}^{L-1}\eta \times \vert t(p_k ) + 1 − s(p_{k+1})\vert $。


### 短语模型的解码

最后的问题就是怎么解码，也就是寻找$\underset{y \in \mathcal{Y}(x)}{argmax}f(y)$。和语音识别的解码一样，这是个NP问题，不存在多项式时间复杂度的算法，因此在实际应用中我们必须使用近似的解码算法。

我们首先定义一个关键的数据结构状态(state)，这里状态是一个五元组$(e_1 , e_2 , b, r, \alpha)$。这里$e_1$和$e_2$是英语单词，b是长度为n(和输入x一样的长度)的位图，r是一个整数表示最后一个短语的结束下标，$\alpha$是当前累加得分。

任何一个短语序列(也就是一个推导)都可以对应到一个状态。比如短语序列：

```
y = (1, 3, we must also), (7, 7, take), (4, 5, this criticism)
```

可以对应为状态：

```
(this, criticism, 1111101, 5, α)
```

状态会记录最后两个词(用于trigram语言模型)；位图用来记录哪些词已经翻译过了；最后一个短语的结束下标，这里是5；得分$\alpha$。假设当前短语序列的长度为L，则得分$\alpha$的计算公式为：

$$
\alpha=h(e(y)) + \sum_{k=1}^{L}g(p_k) + \sum_{k=1}^{L-1} \eta \times |t(p_k ) + 1 − s(p_{k+1})|
$$

我们定义初始状态为$q_0 = (∗, ∗, 0^n , 0, 0)$，其中*表示特殊的开始词，$0^n$表示所有的词都还没有翻译过，第4个0表示最后一个短语的开始位置是0（还没有短语），最后一个0表示当前得分0。

接下来我们定义函数ph(q)，它的输入是状态q，输出是所有可以跟在当前状态之后的合法短语。假设状态$q=(e_1 , e_2 , b, r, \alpha)$，一个属于ph(q)的短语p必须满足如下条件：

* p不能包含以及翻译过的词，也就是$b_i=0, i=\{s(p)...t(p)\}$。
* 不能错位太多，也就是满足$\vert r + 1 − s(p) \vert \le d$


对于状态q和它的后续短语集合ph(q)中的短语p，我们定义next(q,p)为状态q对应的推导再假设短语p之后的新状态。如果$q = (e_1 , e_2 , b, r, \alpha)$，$p = (s, t, \epsilon_1 . . . \epsilon_M )$，那么next(q, p)是状态$q' = (e'_1 , e'_2 , b' , r' , \alpha')$，其中：

* 为了记号方便，我们定义$\epsilon_{-1}=e_1, \epsilon_0=e_2$

* 这里 $$e'_1=\epsilon_{M-1},  e'_2=\epsilon_M$$ 

* b'先从b复制一份，然后把$b_s...b_t$都设置为1（已翻译）
* r'=t
* 并且 $$\alpha' = \alpha + g(p) + \sum_{i=1}^M log q(\epsilon_i \vert \epsilon_{i-2}, \epsilon_{i−1}) +\eta \times \vert r + 1 − s \vert $$


因为$$e'_1$$和$$e'_2$$分别表示新状态最后两个英语单词，通常它是p的最后两个单词$\epsilon_{M-1},\epsilon_M$，但是万一p的单词个数不足，则要用前面的$e_1$或$e_2$，为了记号简单，我们定义$$\epsilon_{-1}=e_1, \epsilon_0=e_2$$。

位图b'比较容易理解，就是把新的词对应的下标设置成1，表示它已经被翻译过来。r'=t也容易理解。而新的得分是原来的得分加上短语得分g(p)在加上语言模型得分再加上错误的惩罚($\eta<0$)。

在介绍解码算法之前还需要再定义一个简单的函数eq(q,q')来判断两个状态是否"相等"，状态相等的条件是除了得分之外其它的部分都必须相等，也就是如果$q = (e_1 , e_2 , b, r, \alpha)$，$q' = (e'_1 , e'_2 , b' , r' , \alpha')$，那么eq(q,q')是True当且仅当$e_1=e'_1, e_2=e'_2, b=b', r=r'$。

解码算法如<a href='#bp-decoding-algo'>下图</a>所示。这个算法用n+1个集合$Q_i, i=\{0,1,...,n\}$，$Q_i$存储已翻译词等于i的状态，一开始$Q_0=\{q_0\}$，而其它的$Q_i$是空集合。接着遍历n次，每次从$Q_i$选择已翻译长度为i的状态，为了效率，只选择得分最高的几个状态(通过后面介绍的Beam函数)。对于这些长度为i的状态，遍历它们所有合法的后续状态，然后得到新的状态，然后根据这些新状态的长度加到不同的$Q_i$里，新状态的长度至少比i要大一，也可以大很多，因此一个状态可以多次被加进去。所以需要一个Add函数选择得分最高的那个。

Add函数<a href='#bp-decoding-func-add'>下图</a>所示。如果q'不存在，那么直接加到Q里，如果存在那么要和原来的比，如果比原来的分高，那么替掉原来的得分和回溯指针。beam函数如<a href='#bp-decoding-func-beam'>下图</a>所示，它返回最高得分的状态以及那些和最高得分差距小于阈值$\beta$的状态。
 

<a name='bp-decoding-algo'>![](/img/mt/bp-decoding-algo.png)</a>
*图：基于短语的翻译系统的解码算法*


<a name='bp-decoding-func-add'>![](/img/mt/bp-decoding-func-add.png)</a>
*图：Add(q,q')函数*
 

<a name='bp-decoding-func-beam'>![](/img/mt/bp-decoding-func-beam.png)</a>
*图：beam函数的定义*


## Google NMT

前面我们介绍过怎么使用Encoder-Decoder以及Attention来实现End-to-End的神经网络机器翻译系统，相比于SMT，神经网络机器翻译系统要简单的多。

本节我们介绍Google的论文[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)，我们通过这篇论文来了解在实际的系统中怎么使用神经网络来实现翻译系统。


### 模型结构

Google NMT还是采用经典的Encoder-Decoder加Attention的框架。它包括3个网络——Encoder、Decoder和Attention。Encoder把输入的每一个词都编码成一个向量，然后把这些向量都输入给Decoder，接着Decoder每个时刻都生成一个词，直到遇到EOS完成翻译。Decoder会使用Attention来决定每个时刻应该重点关注哪些时刻的输入。假设(X,Y)是源语言的句子和目标语言的句子对，其中$X=x_1,...,x_M$，$Y=y_1,...,y_N$。Encoder可以看成如下的函数：

$$
\boldsymbol{x_1}, ... ,\boldsymbol{x_M}=EncoderRNN(x_1,...,x_M)
$$

我们用粗体的$x_t$来表示Encoder在t时刻的输出。接下来我们看$P(Y \vert X)$：

$$
\begin{split}
P(Y|X) & =P(Y|\boldsymbol{x_1}, ... ,\boldsymbol{x_M}) \\
& = \prod_{i=1}^{N}P(y_i|y_0,...,y_{i-1};\boldsymbol{x_1}, ... ,\boldsymbol{x_M})
\end{split}
$$


其中$y_0$是特殊的开始符号，通常记为BOS(Beginning of Sentence)。在Decode的时候，我们使用RNN+Attention来根据之前的翻译和输入的编码来计算当前时刻的输出：

$$
P(y_i|y_0,...,y_{i-1};\boldsymbol{x_1}, ... ,\boldsymbol{x_M})
$$


每个时刻t，Decoder都会生成这个时刻的隐状态$\boldsymbol{y}_t$，然后使用softmax变成输出不同词的概率。本文作者认为，为了达到比较好的效果，Encoder和Decoder都需要很多层。每增加一层，PPL能减小10%。这里使用的Attention计算公式如下：

$$
\begin{split}
s_t & =AttentionFunction(\boldsymbol{y}_{i−1} , \boldsymbol{x}_t) \; 1 \le t \le M \\
p_t & = exp(s_t)/\sum_{t=1}^{M}exp(s_t) \\
\boldsymbol{a}_t & =\sum_{t=1}^{M}p_t\boldsymbol{x}_t
\end{split}
$$


模型结构如<a href='#gnmt-1'>下图</a>所示，Encoder一共8层，第一层的LSTM是双向的，其余各层的LSTM是单向的，为了提高并发效率，第二层到第8层分别放到GPU2-GPU8上，而第一层的LSTM的正向放到GPU1上，逆向计算放到GPU2上。当LSTM层次变多之后很难训练，作者发现使用残差连接能够让模型更快收敛，因此从第二层开始都引入了残差连接。

对于Decoder，为了提高效率，我们只让第一层使用Attention，而之上的层不用Attention。

<a name='gnmt-1'>![](/img/mt/gnmt-1.png)</a>
*图：Google NMT的结构图*

### 分词

机器翻译很大的一个问题就是OOV(Out of Vacabulary)的问题，对于训练时没有见过的词是很难翻译好的。一些新词虽然没见过，但是我们也是可能翻译出来的。例如一些需要音译的人名，比如Jim，即使我们不认识，但是英语人名翻译成汉语其实是有一定规则的，一般是寻找与它发音类似的汉字，比如吉姆，当然也可能翻译成吉穆，但是不太可能翻译成"积木"或者"鸡姆"，因为前者是已经存在的一个词，而后者太难听了（某些词是有贬义色彩的）。另外一些合成词，比如left-handed，即使我们没见过，也可能根据left和hand猜测出来它的含义。

因此我们是可能让模型自己学习出这样的知识来的。但是有一个前提就是分词的粒度要细。如果我们按照普通的按空格来分词，那么left-handed就会被分成一个词，那模型肯定无法翻译，但是如果我们把left-handed分开成 left handed，虽然训练数据中没有left-handed，但是训练数据中可能有right-handed，它也是有可能学到怎么翻译left-handed的。类似的，我们希望把Jim分成"Ji m"，这样模型可能学到在人名里，Ji经常被翻译成吉，而m则被翻译成姆。

那细粒度的分词就很有必要，一种最简单的切分方法就是切分成字符(character)，但是在英语等语言里(汉语可以切分成单个汉字)，字符太少，因此每个字符承载的语义太多。比如left-handed，我们可能希望切分成[left,hand,ed]，前两个都是词，而ed表示过去式。当然我们也可以切分成[lef, t, han, ded]，这似乎不太好。为什么不好呢？我们可能会说left是词，hand也是词，ed是一种语法现象。但是god能不能切分成[go,d]能？unlike为什么要切分成[un, like]呢？

套用一句话“世界上本来没有路，走的人多了也就成了路”，对于词我们也可以这样说“世上本没有词，用的多的字符串就是词”。这其实就是Byte Pair Encoding(BPE)的思路，这是来自论文[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)。

BPE的实现很简单，给定一个语料库(一堆词)，刚开始的词只包括所有的字符，因此每个词都是切分成一个一个的字符。然后把连续出现的频率最高的词组合成一个新的词加到词典里，这样就多了一个词，然后把语料库里词用新的词典重新切分。然后不断的重复这个过程直到词典的大小是我们期望的值。实现的代码非常简单：
```
import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
          pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w' : 5, 'l o w e r' : 2,
			'n e w e s t':6, 'w i d e s t':3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
```

比如例子的语料库只有4个单词["low", "lowest", "newer", "widest"]，那么初始的词典包括[l,o,w,e,s,t,n,r,i,d]等"词"。用这些词切分后得到：
```
l o w
l o w e s t
n e w e r
w i d e s t
```

然后我们统计发现连续出现最多的是es(2次)，因此我们把es也加到词典里重新分词(最大正向匹配)：
```
l o w
l o w es t
n e w e r
w i d es t
```
然后再统计，发现es和t连续出现最多(2次)，因此把est加到词典后再分词得到：
```
l o w
l o w est
n e w e r
w i d est
```
每次我们都能往词典里添加一个新词。Google的NMT也是使用类似的思路，使用的wordpiece model，最终也是会构造出一个细粒度的分词词典，这里就不详细介绍了。

来了一个句子之后首先是用空格分词，接着使用细粒度的词典使用最大正向匹配算法再次分词，比如句子：
```
Jet makers feud over seat width with big orders at stake
```
使用WordPiece模型之后可能被切分成：
```
_J et _makers _fe ud _over _seat _width _with _big _orders _at _stake
```
为了能够根据WordPiece恢复出原始句子，这里在词的第一个WordPiece前增加一个特殊的下划线。因此对于Google NMT来说，它的输入是WordPiece。


### 训练目标函数

给定训练集合$\mathcal{D}=\\{X^i, Y_*^i\\}_{i=1}^N$最常见的目标函数就是训练数据的似然，我们需要调整参数使得下面的似然概率尽可能的大：

$$
\mathcal{O}_{ML}(\theta)=\sum_{i=1}^{N}P_\theta(Y_*^i|X^i)
$$

但是上面的目标函数和我们最终优化的指标比如BLEU并不完全一致，而且它只关系怎么把"完全"正确的翻译$$P(Y_*^i \vert X^i)$$的概率调大，但是对于"不完全"正确的翻译，比如Y1和Y2，它就完全不关心了。但是如果Y1比Y2好，其实我们是希望P(Y1\|X)的概率是要大于P(Y1\|X)的，但是在上面的目标函数是不关心这个的。比如X的完全正确的翻译是Y(训练句对)，而Y1和Y2是不完全正确的翻译，那么参数$\theta_1$可能的输出为：

$$
\begin{split}
P(Y|X) & =0.7 \\
P(Y1|X) & =0.1 \\
P(Y2|X) &= 0.2
\end{split}
$$

而参数$\theta_2$的输出为：

$$
\begin{split}
P(Y|X) & =0.7 \\
P(Y1|X) & =0.2 \\
P(Y2|X) &= 0.1
\end{split}
$$

因为这两种情况正确的概率都是0.7，因此在这个目标函数下它认为参数$\theta_1$和$\theta_2$是一样好的，但是我们知道Y1比Y2更好，因此我们更希望参数是$\theta_2$。为此本文引入了另外一个目标函数$\mathcal{O}_{RL}(\theta)$：

$$
\mathcal{O}_{RL}(\theta)=\sum_{i=1}^{N}\sum_{Y \in \mathcal{Y}}P_\theta(Y|X^i)r(Y,Y_*^i)
$$


上式中$$r(Y,Y_*^i)$$表示翻译Y和"正确"翻译$$Y_*^i$$的相似程度。因此上面的目标函数除了要把正确的翻译的概率调大(因为$$r(Y_*^i,Y_*^i)=1$$)，而且也需要使得$P(Y1 \vert X^i)>P(Y2 \vert X^i)$，原因是$$r(Y1,Y_*^i) > r(Y2, Y_*^i)$$。在实际的Google NMT系统中是把这两个目标函数都考虑进来的：

$$
\mathcal{O}_{Mixed}(\theta)=\alpha \times \mathcal{O}_{ML}(\theta) + \mathcal{O}_{RL}(\theta)
$$

论文中的超参数$\alpha=0.017$。

### Decoder

Google NMT的Decoder使用Beam Search算法，也就是每个时刻都保留得分最高的N条路径。然后再t时刻展开这些路径扩展出更多路径，最终又选择得分最高的N条。因为概率都是小于1的，翻译的句子越多概率越高，所以模型是倾向于短的翻译，为了避免这个情况，本文引入了Length Normalization。

神经网络翻译系统的另外一个问题就是它经常会出现某些词不翻译的情况，原因可能是某些很难翻译的词不管怎么翻译都是错的，干脆就放弃了，这样可能语言模型的得分还会高一些。但这对于一个实际的系统来说是很糟糕的，因为机器翻译的结果一般是给人来看的，如果每个词系统都尝试翻译了，即使不通顺，人也可能猜测出它的含义来。而如果某些词(尤其是关键词)不翻译，即使看起来很通顺，人以为理解了，但是实际理解错了，这后果更严重。

为了解决这个问题，我们会引入翻译的覆盖率(coverage)的概念，源语言被翻译(覆盖)过的词越多越好。那怎么定义一个词是否被翻译过呢？我们可以使用Attention。比如t时刻我们Pay了0.8的Attention在word1上，那么我们就认为word1被覆盖了0.8。因此在Beam Search的得分定义如下：

$$
\begin{split}
s(Y,X) & =log(P(Y|X))/lp(Y)+cp(X,Y) \\
lp(Y) & =\frac{(5+|Y|)^\alpha}{(5+1)^\alpha} \\
cp(X,Y) & =\beta \times \sum_{i=1}^{|X|}log(min(\sum_{j=1}^{|Y|}p_{i,j}, 1.0))
\end{split}
$$

Y越长，lp(Y)越大，而logP(Y\|X)是一个负数，因此logP(Y\|X)/lp(Y)越大，这样在相同的P(Y\|X)条件下更倾向于长的Y，当然这个公式也没有什么理论依据，就是个经验公式。

接下来看cp(X,Y)，$p_{ij}$表示$Y_j$对$X_i$的Attention。第一个求和符号是遍历所有X中的词$X_i$，然后对于某个$X_i$，我们遍历所有的$Y_j$，把$Y_j$对$X_i$的Attention加起来，就认为是$X_i$被翻译(cover)的概率，当然这个值可能大于1，因此需要用min把它限制在1之内。如果$X_i$被翻译的概率等于1，那么这个词就完全被覆盖了，因此log1是0，这是一个不错的值。如果$X_i$被翻译的概率很小，比如0.1，那么log0.1是一个负数，就相当于惩罚。

