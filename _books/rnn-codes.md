---
layout:     post
title:      "用Numpy实现vanilla RNN"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - RNN
    - 《深度学习理论与实战：基础篇》补充材料
---

本文是《深度学习理论与实战：基础篇》第4章的补充知识。介绍怎么不适用深度学习框架而只用Numpy实现最简单的RNN，这可以帮助读者理解RNN的计算过程。

 <!--more-->
 
**目录**
* TOC
{:toc}

下面我们会用RNN来实现一个简单的Char RNN语言模型。为了让读者了解RNN的一些细节，本示例会使用numpy来实现forward和backprop的计算。RNN的反向传播算法一般采用BPTT，如果读者不太明白也不要紧，但是forward的计算一定要清楚。这个RNN代码参考了karpathy的blog文章[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)附带的[代码](https://gist.github.com/karpathy/d4dee566867f8291f086)。代码总共一百来行，我们下面逐段来阅读。


## 数据预处理

```
data = open('../data/tiny-shakespeare.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
```


上面的代码读取莎士比亚的文字到字符串data里，通过set()得到所有不重复的字符并放到chars这个list里。然后得到char_to_ix和ix_to_char两个dict，分别表示字符到id的映射和id到字符的映射，id从零开始。

## 模型超参数和参数定义

```
# 超参数
hidden_size = 100  # 隐藏层神经元的个数
seq_length = 25  # BPTT时最多的unroll的步数
learning_rate = 1e-1

# 模型参数
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # 输入-隐藏层参数
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层-隐藏层参数
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # 隐藏层-输出层参数
bh = np.zeros((hidden_size, 1))  # 隐藏层bias
by = np.zeros((vocab_size, 1))  # 输出层bias
```

上面的代码定义超参数hidden_size，seq_length和learning_rate，以及模型的参数Wxh, Whh和Why。

## lossFun

```

def lossFun(inputs, targets, hprev):
	"""
	inputs,targets都是整数的list
	hprev是Hx1的数组，是隐状态的初始值
	返回loss，梯度和最后一个时刻的隐状态
	"""
	xs, hs, ys, ps = {}, {}, {}, {}
	hs[-1] = np.copy(hprev)
	loss = 0
	# forward pass
	for t in xrange(len(inputs)):
		xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
		xs[t][inputs[t]] = 1
		hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
		ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
		loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
	# backward pass: compute gradients going backwards
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])
	for t in reversed(xrange(len(inputs))):
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
		dWhy += np.dot(dy, hs[t].T)
		dby += dy
		dh = np.dot(Why.T, dy) + dhnext  # backprop into h
		dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
		dbh += dhraw
		dWxh += np.dot(dhraw, xs[t].T)
		dWhh += np.dot(dhraw, hs[t - 1].T)
		dhnext = np.dot(Whh.T, dhraw)
	for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
	return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]
```

我们这里只阅读一下forward的代码，对backward代码感兴趣的读者请参考：https://github.com/pangolulu/rnn-from-scratch。
```
	# forward pass
	for t in xrange(len(inputs)):
		xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
		xs[t][inputs[t]] = 1
		hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
		ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
		loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
```

上面的代码遍历每一个时刻t，首先把字母的id变成one-hot的表示，然后计算hs[t]。计算方法是：hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)。也就是根据当前输入xs[t]和上一个状态hs[t-1]计算当前新的状态hs[t]，注意如果t=0，则hs[t-1] = hs[-1] = np.copy(hprev)，也就是函数参数传入的隐状态的初始值hprev。接着计算ys[t] = np.dot(Why, hs[t]) + by。然后用softmax把它变成概率：ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))。最后计算交叉熵的损失：loss += -np.log(ps[t][targets[t], 0])。注意：ps[t]的shape是[vocab_size,1]

## sample函数

这个函数随机的生成一个句子（字符串）。
```
def sample(h, seed_ix, n):
	"""
	使用rnn模型生成一个长度为n的字符串
	h是初始隐状态，seed_ix是第一个字符
	"""
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = []
	for t in xrange(n):
		h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
		y = np.dot(Why, h) + by
		p = np.exp(y) / np.sum(np.exp(y))
		ix = np.random.choice(range(vocab_size), p=p.ravel())
		x = np.zeros((vocab_size, 1))
		x[ix] = 1
		ixes.append(ix)
	return ixes
```
sample函数会生成长度为n的字符串。一开始x设置为seed_idx：x[seed_idx]=1(这是one-hot表示)，然后和forward类似计算输出下一个字符的概率分布p。然后根据这个分布随机采样一个字符(id) ix，把ix加到结果ixes里，最后用这个ix作为下一个时刻的输入：x[ix]=1

## 训练
```
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
	# prepare inputs (we're sweeping from left to right in steps seq_length long)
	if p + seq_length + 1 >= len(data) or n == 0:
		hprev = np.zeros((hidden_size, 1))  # reset RNN memory
		p = 0  # go from start of data
	inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
	targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
	
	# sample from the model now and then
	if n % 1000 == 0:
		sample_ix = sample(hprev, inputs[0], 200)
		txt = ''.join(ix_to_char[ix] for ix in sample_ix)
		print('----\n %s \n----' % (txt,))
	
	# forward seq_length characters through the net and fetch gradient
	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001
	if n % 1000 == 0:
		print('iter %d, loss: %f' % (n, smooth_loss))  # print progress
	
	# perform parameter update with Adagrad
	for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
	[dWxh, dWhh, dWhy, dbh, dby],
	[mWxh, mWhh, mWhy, mbh, mby]):
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
	
	p += seq_length  # move data pointer
	n += 1  # iteration counter
```

上面是训练的代码，首先初始化mWxh, mWhh, mWhy。因为这里实现的是Adgrad，所以需要这些变量来记录每个变量的“delta”，有兴趣的读者可以参考[cs231n的notes](http://cs231n.github.io/neural-networks-3/#ada)。

接下来是一个无限循环来不断的训练，首先是得到一个训练数据，输入是data[p:p + seq_length]，而输出是data[p+1:p + seq_length+1]。然后是lossFun计算这个样本的loss、梯度和最后一个时刻的隐状态（用于下一个时刻的隐状态的初始值），然后用梯度更新参数。每1000次训练之后会用sample函数生成一下句子，可以通过它来了解目前模型生成的效果。

完整代码在[这里](https://github.com/fancyerii/deep_learning_theory_and_practice/tree/master/src/ch4#%E4%BD%BF%E7%94%A8numpy%E5%AE%9E%E7%8E%B0char-rnn%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)。



