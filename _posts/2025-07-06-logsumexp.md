---
layout:     post
title:      "翻译：The Log-Sum-Exp Trick" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - pytorch
---

本文翻译[The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)。

 
对数概率向量的归一化是统计建模中的常见任务，但当对大数值进行指数运算时，这可能导致下溢或上溢。本文将讨论用于解决此问题的对数-和-指数技巧（log-sum-exp trick）。

<!--more-->
 
 
 

在统计建模和机器学习中，我们经常在**对数尺度（logarithmic scale）下工作。这样做有很多很好的理由。例如，当 $x$ 和 $y$ 都是很小的数字时，计算 $x$ 乘以 $y$ 可能会导致下溢（underflow）**。然而，我们可以在对数尺度下工作，将乘法转换为加法，因为：

$$
\log(xy) = \log x + \log y \quad (1)
$$

此外，如果我们处理的是诸如 $f(x)$ 和 $g(x)$ 这样的函数，那么根据微分的线性性质，我们有：

$$
\frac{\partial}{\partial x} \log[f(x)g(x)] = \frac{\partial}{\partial x} \log f(x) + \frac{\partial}{\partial x} \log g(x) \quad (2)
$$

通常，分别对每个函数求导比应用乘法法则更容易。

这些只是我们通常更喜欢处理**对数似然（log likelihoods）**和**对数概率（log probabilities）**的两个原因。而且，在对数尺度下工作在数值上要稳定得多，以至于许多标准库简单地通过对数 PDF 进行指数化来计算概率密度函数（PDFs）。有关示例，请参阅我之前关于 SciPy 多元正态 PDF 的文章。

然而，有时我们需要**还原**这些可能很大的对数尺度值。例如，如果我们需要归一化一个 $N$ 维的对数概率向量 $x\_i = \\log p\_i$，我们可能会天真地计算：

$$
p_i = \frac{\exp(x_i)}{\sum_{n=1}^{N} \exp(x_n)}, \quad \sum_{n=1}^{N} p_n = 1 \quad (3)
$$

由于每个 $x\_n$ 都是一个对数概率，它可能非常大，并且可以是负数或正数，那么进行指数运算可能分别导致**下溢或上溢**。（如果一个值 $x$ 在 $(0,1)$ 之间，那么 $\\log(x)$ 必须是负数。然而，我们通常希望即使数量超出此范围，也能将其解释为概率。）想想对数似然如何根据似然函数和数据点的数量具有任意尺度。

考虑到这些思想，我们来看**对数-和-指数（log-sum-exp）操作**：

$$
\text{LSE}(x_1, \ldots, x_N) = \log\left(\sum_{n=1}^{N} \exp(x_n)\right) \quad (4)
$$

对数-和-指数操作可以帮助**防止这些数值问题**，但其工作原理可能并不立即显而易见。首先，让我们将 (3) 式改写为：

$$
\log(p_i) = \log\left(\frac{\exp(x_i)}{\sum_{n=1}^{N} \exp(x_n)}\right)
$$

$$
= \log(\exp(x_i)) - \log\left(\sum_{n=1}^{N} \exp(x_n)\right)
$$

$$
= x_i - \text{LSE}(x_1, \ldots, x_N)
$$

$$
p_i = \exp\left(x_i - \text{LSE}(x_1, \ldots, x_N)\right) \quad (5)
$$

换句话说，我们可以使用 (4) 中的对数-和-指数操作来执行 (3) 中的归一化。但这真的有帮助吗？我们仍然在对每个 $x\_n$ 进行指数运算。要看到该操作的实用性的最后一步是考虑这个推导：

$$
y = \log\left(\sum_{n=1}^{N} \exp(x_n)\right)
$$

$$
e^y = \sum_{n=1}^{N} \exp(x_n)
$$

$$
e^y = e^c \sum_{n=1}^{N} \exp(x_n - c)
$$

$$
y = c + \log\left(\sum_{n=1}^{N} \exp(x_n - c)\right) \quad (6)
$$

换句话说，(4) 中的对数-和-指数运算符之所以好用，是因为我们可以通过**任意常数 $c$ 来平移指数中的值**，同时仍然计算出相同的最终值。如果我们将：

$$c = \max\{x_1, \ldots, x_N\} \quad (7)$$

我们确保最大的正指数项是 $\\exp(0) = 1$。

### 代码示例

让我们在代码中看看这个技巧。首先，请注意，即使不是很大的 $x\_n$ 值也可能导致上溢：

```python
>>> x = np.array([1000, 1000, 1000])
>>> np.exp(x)
array([inf, inf, inf])
```

你的结果可能会因机器的精度而异。显然，像 (3) 中那样对 $x$ 进行归一化在当前值下是不可能的。如果你在一个更大的管道中运行此输入和代码，最终某些函数会因为 `inf` 值而崩溃或将它们转换为 `nan` 值。然而，让我们实现一个 `logsumexp` 函数——如果你在 Python 中工作，可能应该使用 SciPy 的实现——：

```python
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))
```

然后应用 (5) 中的归一化技巧：

```python
>>> logsumexp(x)
1001.0986122886682
>>> np.exp(x - logsumexp(x))
array([0.33333333, 0.33333333, 0.33333333])
```

这个技巧也可以帮助处理下溢，当机器精度将小值四舍五入为零时：

```python
>>> x = np.array([-1000, -1000, -1000])
>>> np.exp(x)
array([0., 0., 0.])
>>> np.exp(x - logsumexp(x))
array([0.33333333, 0.33333333, 0.33333333])
```

同样，我们天真地计算的 (3) 中的归一化会崩溃，可能是由于除以零错误。这个技巧仍然适用于大范围的值：

```python
>>> x = np.array([-1000, -1000, 1000])
>>> np.exp(x - logsumexp(x))
array([0., 0., 1.])
```

虽然第一个和第二个分量的概率并非真正为零，但这合理地近似了这些对数概率所代表的含义。此外，我们实现了数值稳定性。我们可以可靠地计算 (3)，而不会引入 `inf` 或 `nan` 值或除以零。

## **致谢** 感谢 Eugen Hotaj 纠正了这篇帖子早期版本中的一个错误。
 
