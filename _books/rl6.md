---
layout:     post
title:      "强化学习简介(六)"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 强化学习
    - 强化学习简介系列文章
    - 《深度学习理论与实战》拾遗
    - Policy Gradient
---

 本文介绍Policy Gradient，这是这个系列的最后一篇文章。

 更多本系列文章请点击<a href='/tags/#强化学习简介系列文章'>强化学习简介系列文章</a>。更多内容请点击[深度学习理论与实战：提高篇]({{ site.baseurl }}{% post_url 2019-03-14-dl-book %})。
<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>
 <!--more-->
 
**目录**
* TOC
{:toc}
 
值函数的方法里的策略是隐式的，比如$\pi(a\|s)=\underset{a}{argmax}Q(s, a)$。而Policy Gradient不同，它直接有一个参数化的策略(比如是一个神经网络)，Policy Gradient通过直接求Reward对策略函数的参数的梯度来不断的找到更好的策略(参数)使得期望的Reward越来越大。这是一种梯度上升(Gradient Ascent)算法，和梯度下降类似，只不过一个是求最大值，一个是求最小值，如果加一个负号，那么就是一样的了。

## Reward
假设策略函数(可以是很复杂的神经网络)的参数是$\theta$，我们把策略函数记作$\pi_{\theta}(a|s)$，它表示在状态s时采取策略a的概率。Reward函数的定义如下：

$$
J(\theta) = \sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s)  = \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \vert s) Q^\pi(s, a)
$$

上式中，$d^\pi(s)$是以$\pi_\theta$为转移概率的马尔科夫链的稳态分布(stationary distribution)。马尔科夫链有一个很好的性质，当跳转次数趋于无穷大的时候，最终它处于某个状态的概率只取决于跳转概率，而与初始状态无关。为了记号的简单，我们把$d^{\pi_\theta}$简记为$d^{\pi}$，$Q^{\pi_\theta}$简记作$Q^{\pi}$。稳态概率的形式化定义为：$d^\pi(s) = \lim_{t \to \infty} P(s_t = s \vert s_0, \pi_\theta)$。当t趋于无穷大的时候，概率$P(s_t = s \vert s_0, \pi_\theta)$与$s_0$无关，因此可以记作$d^\pi(s)$。

我们可以这样来解读$J(\theta)$：要计算一个策略$\pi$的Reward，我们可以一直运行(run)这个策略无穷多次，那么最终停在状态s的概率是稳态分布$d^\pi(s)$，而状态s的价值是$V^\pi(s)$，因此我们认为最终的Reward就是$\sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s) $。而后面那个等式就是简单的把$V^{\pi(s)}$展开成$Q^{\pi(s)}$，这个技巧我们在前面已经见过很多次了。

## Policy Gradient定理

计算Reward对参数$\theta$的梯度$\nabla_\theta J(\theta)$比较Tricky。因为$J(\theta)$中的三项$d^{\pi(s)}$、$\pi_\theta(a \vert s)$和$Q^\pi(s, a)$都与参数$\theta$有关，而且$d^{\pi(s)}$和$Q^\pi(s, a)$都是非常间接的受$\theta$的影响——$\theta$影响策略$\pi_\theta(a \vert s)$，而策略(跳转概率)影响稳态分布$d^{pi(s)}$和值函数$Q^\pi(s,a)$。

Policy Gradient定理帮我们理清上面复杂的函数依赖关系，给出了简洁的Policy Gradient的计算公式：

$$
\begin{split}
\nabla_\theta J(\theta) 
& = \nabla_\theta \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \pi_\theta(a \vert s) \\
& \propto \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \vert s) 
\end{split}
$$

上面的公式非常简洁好记，直接把梯度符号$\nabla$越过各种求和符合直接放到$\pi_\theta(a \vert s)$前就行。

## Policy Gradient定理的推导

推导数学公式有点多，跳过也不影响理解后续的内容(但是Policy Gradient定理得记住)，但是作者强烈建议读者能拿出纸笔详细的抄写一遍，这会对后续的算法的理解很有帮助。虽然推导过程有些繁琐，但并不复杂，如果有一两步确实不能理解，读者也可以忽略其推导过程暂时"假设"它是对的，也许等读完整个过程之后就能理解它了。

我们先看$V^\pi(s)$的梯度：


$$
\begin{aligned}
& \nabla_\theta V^\pi(s) \\
=& \nabla_\theta \Big(\sum_{a \in \mathcal{A}} \pi_\theta(a \vert s)Q^\pi(s, a) \Big) & \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\nabla_\theta Q^\pi(s, a)} \Big) & \scriptstyle{\text{; 乘法的导数}} \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\nabla_\theta \sum_{s', r} P(s',r \vert s,a)(r + V^\pi(s'))} \Big) & \scriptstyle{\text{; 用未来的} Q^\pi \text{ 扩展 }} \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\sum_{s', r} P(s',r \vert s,a) \nabla_\theta V^\pi(s')} \Big) & \scriptstyle{P(s',r \vert s,a) \text{ 不是 }\theta \text{的函数}}\\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\sum_{s'} P(s' \vert s,a) \nabla_\theta V^\pi(s')} \Big) & \scriptstyle{\text{; 因为 }  P(s' \vert s, a) = \sum_r P(s', r \vert s, a)}
\end{aligned}
$$

因此我们有：

$$
\color{red}{\nabla_\theta V^\pi(s)} 
= \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \sum_{s'} P(s' \vert s,a) \color{red}{\nabla_\theta V^\pi(s')} \Big)
$$

上面的公式是递归定义的，右边的$\nabla_\theta V^\pi(s')$又可以用相同的方法展开，后面我们会用到。

我们下面考虑如下的访问序列：

$$
s \xrightarrow[]{a \sim \pi_\theta(.\vert s)} s' \xrightarrow[]{a \sim \pi_\theta(.\vert s')} s'' \xrightarrow[]{a \sim \pi_\theta(.\vert s'')} \dots
$$

定义从状态s经过k步跳转到状态x的概率为$\rho^\pi(s \to x, k)$。这个概率的计算需要递归进行：

当k=0时，$\rho^\pi(s \to s, k=0) = 1$，除了跳转到自己之外其余的概率都是0

 k=1时，$\rho^\pi(s \to s’, k=1) = \sum_a \pi_\theta(a \vert s) P(s’ \vert s, a)$。

k>1时，$\rho^\pi(s \to x, k+1) = \sum_{s'} \rho^\pi(s \to s', k) \rho^\pi(s' \to x, 1)$。

当k=1时，也就是从状态s调整到s'的概率，我们需要遍历每一个action a，在策略$\pi$下，我们采取a的概率是$\pi(a \vert s)$，而我们在状态s下采取a跳到s'的概率是$P(s’ \vert s, a)$，因此就得到k=1时的计算公式。

而从s通过k+1步跳转到x的概率计算，我们分为两步：第一步是s通过k步跳转到s'；第二步从s'跳转到x。前者的概率是
$\rho^\pi(s \to s', k)$，后者的概率是$\rho^\pi(s' \to x, 1)$，因此就得到k>1的情况。

接下来我们递归的展开$\nabla_\theta V^\pi(s)$，为了简单，我们定义$\phi(s) = \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a)$，因为对a求和了，所有右边是只与s有关而与a无关的函数。

下面的推导就是通过不断的展开$\nabla_\theta V^\pi(s)$：

$$
\begin{aligned}
& \color{red}{\nabla_\theta V^\pi(s)} \\
=& \phi(s) + \sum_a \pi_\theta(a \vert s) \sum_{s'} P(s' \vert s,a) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \sum_a \pi_\theta(a \vert s) P(s' \vert s,a) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \color{red}{[ \phi(s') + \sum_{s''} \rho^\pi(s' \to s'', 1) \nabla_\theta V^\pi(s'')]} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \phi(s') + \sum_{s''} \rho^\pi(s \to s'', 2)\color{red}{\nabla_\theta V^\pi(s'')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \phi(s') + \sum_{s''} \rho^\pi(s \to s'', 2)\phi(s'') + \sum_{s'''} \rho^\pi(s \to s''', 3)\color{red}{\nabla_\theta V^\pi(s''')} \\
=& \dots \scriptstyle{\text{; 重复不断的展开 }\nabla_\theta V^\pi(.)} \\
=& \sum_{x\in\mathcal{S}}\sum_{k=0}^\infty \rho^\pi(s \to x, k) \phi(x)
\end{aligned}
$$

上面的推导把$\nabla_\theta Q^\pi(s, a)$去掉了，有了$\nabla_\theta V^\pi(s)$之后，我们就可以计算$\nabla_\theta J(\theta)$：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta V^\pi(s_0) & \scriptstyle{\text{; 稳态分布与初始状态无关，可能随机选择初始状态} s_0} \\
&= \sum_{s}\color{blue}{\sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)} \phi(s) &\scriptstyle{\text{; 令 }\color{blue}{\eta(s) = \sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)}} \\
&= \sum_{s}\eta(s) \phi(s) & \\
&= \Big( {\sum_s \eta(s)} \Big)\sum_{s}\frac{\eta(s)}{\sum_s \eta(s)} \phi(s) & \scriptstyle{\text{; 把 } \eta(s), s\in\mathcal{S} \text{ 归一化成概率分布}}\\
&\propto \sum_s \frac{\eta(s)}{\sum_s \eta(s)} \phi(s) & \scriptstyle{\sum_s \eta(s)\text{  是一个常量}} \\
&= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) & \scriptstyle{d^\pi(s) = \frac{\eta(s)}{\sum_s \eta(s)}\text{ 是稳态分布}}
\end{aligned}
$$

我们可以这样来解读$\eta(s) = \sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)$：$\eta(s)$表示这个policy从$s_0$开始重复不断的执行，"经过"状态s的概率。显然我们可以从$s_0$零步跳转到s(只能是跳到自己)；$s_0$一步跳转到s；...。因此把这些概率加起来就是"经过"状态s的概率。

因为马尔科夫链的极限是趋近于稳态分布，用通俗的话说就是时间足够大之后处于状态s的概率与初始状态无关。因此存在某个T，当时刻t>T时，p(s)=$d^\pi(s)$。因此$\sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)$可以分为两部分，第一部分是$\sum_{k=0}^T$，另一部分是$sum_{k=T+1}^\infty$。前一部分总是一个有限的值，而后一部分是无穷大，因此可以忽略前一部分，而$sum_{k=T+1}^\infty \rho^\pi(s_0 \to s, k)$的平均值等于$d^\pi(s)$，而且$\sum_sd^\pi(s)=1$，因此有$d^\pi(s) = \frac{\eta(s)}{\sum_s \eta(s)}$。
	
对于连续的情况$\sum_s\eta(s)=1$，而对于Episode的情况$\sum_s\eta(s)$等于Episode的平均长度。上面的梯度可以继续简化：

$$
\begin{aligned}
\nabla_\theta J(\theta) 
&\propto \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \vert s)  &\\
&= \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \vert s) Q^\pi(s, a) \frac{\nabla_\theta \pi_\theta(a \vert s)}{\pi_\theta(a \vert s)} &\\
&= \mathbb{E}_\pi [Q^\pi(s, a) \nabla_\theta \ln \pi_\theta(a \vert s)] & \scriptstyle{\text{; 因为 } (\ln x)' = 1/x}
\end{aligned}
$$

上式中$$\mathbb{E}_\pi$$指的是$$\mathbb{E}_{s \sim d_\pi, a \sim \pi_\theta}$$。这里有一个公式需要大家熟悉：

$$
\begin{split}
\mathbb{E}_{x \sim p(x)} [f(x)]=\sum_x p(x)f(x)  & \text{离散情况p(x)是概率分布函数} \\
\mathbb{E}_{x \sim p(x)} [f(x)]=\int p(x)f(x) dx  & \text{连续情况p(x)是概率密度函数} 
\end{split}
$$

对照上面的公式，最后一步就比较容易理解了。把Policy Gradient定理写成期望的形式在实现的时候更加方便，因为实现时我们通常会使用采样的方法(不过是MC的全采样还是TD的只采样一个时刻)，期望等价于采样的求和$Ef(X) \approx \frac{1}{N} \sum_i f(x_i)$。

这个式子是各种Policy Gradient算法的基础，所有的Policy Gradient算法的目的都是为了使得估计的$\mathbb{E}_\pi$的均值接近真实值同时又尽量保证方差较少。

也就是说，Policy Gradient的目的是为了计算梯度$g:=\nabla_\theta\mathbb{E}[\sum_{t=0}^{\infty}r_t]$，最终又都可以写出统一的形式：$g=\mathbb{E}[\sum_{t=0}^\infty \Psi_t \nabla_\theta log \pi_\theta (a_t\|s_t)]$。其中$log \pi_\theta (a_t\|s_t)$可以类比$\nabla_\theta \ln \pi_\theta(a \vert s)$，而$\Psi_t $可以有很多种近似方法，比如：

*  $\sum_{t=0}^{\infty}r_t$，这是整个trajectory的reward
*  $\sum_{t'=t}^{\infty}r_{t'}$，这是$a_t$之后的reward，我们通常假设"因果"关系——$a_t$不影响t时刻之前的reward。
*  $\sum_{t'=t}^{\infty}r_{t'} - b(s_t)$，减去baseline，使得方差变小但是均值不变
*  $Q^\pi(s_t,a_t)$，这就是上面我们推导的形式
*  $A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$，使用Advantage。
*  $r_t +\gamma V^\pi(s_{t+1}) -V^\pi(s_t)$，TD的$\delta$。


因为Policy Gradient通常和深度学习结合，因此本章不介绍具体的代码，后面深度强化学习的部分会有Policy Gradient代码介绍。
