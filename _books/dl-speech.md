---
layout:     post
title:      "深度学习在语音识别中的应用"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 语音识别
    - 深度学习
    - HMM-DNN
    - End-to-End
    - 《深度学习理论与实战：提高篇》
---

前面介绍了经典的基于HMM模型的语音识别系统，接下来我们介绍深度学习在语音识别中的应用。更多文章请点击[深度学习理论与实战：提高篇]({{ site.baseurl }}{% post_url 2019-03-14-dl-book %})。
<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>
 <!--more-->
 
**目录**
* TOC
{:toc}

提起深度学习的再次兴起，大家首先可能会想到2012年AlexNet在图像分类上的突破，但是最早深度学习的大规模应用发生在语音识别领域。自从2006年Geoffrey Hinton提出逐层的Pretraining之后，神经网络再次进入大家的视野。2009年Geoffrey Hinton和Deng Li把DNN用于声学模型建模，用于替代GMM，同时大家发现在训练数据足够的情况下Pretraining是不必要的。使用了DNN后，语音识别的词错误率相对降低了30%。这里的深度学习还只是用于替代HMM-GMM里的GMM，再到后来，End-to-End的语音识别系统的出现，从根本上抛弃了复杂的HMM(包括WFST这样复杂的解码算法)。

## 深度学习和HMM的结合
前面介绍了经典的HMM-GMM模型，这是在深度学习流行前最主流的方法。使用深度神经网络DNN来替代GMM是深度学习在语音识别的重要进展，它使得语音识别效果有了极大的提高。

我们回顾一下，在HMM-GMM模型里，我们使用GMM来建模状态的发射概率$P(X \vert q)$，也就是状态q下观察是X的概率，这里X通常是当前帧的MFCC特征。我们不能直接用DNN来建模这个发射概率，因为DNN是区分性(discriminative)模型而不是生成(generative)模型，它只能得到概率$P(q \vert X)$，也就是给定观察，输出不同状态的概率。根据公式：

$$
P(X|q) = \frac{P(q|X)P(X)}{P(q)}
$$

因为X是已知的，P(X)是个常量，所以我们可以计算：

$$
\frac{P(X|q)}{P(X)} = \frac{P(q|X)}{P(q)}
$$

为了训练DNN，我们需要更细粒度的标注，比如q是triphone，那么我们需要知道每一帧特征X对应的triphone标签。让人来标注是不可能的，我们通常先训练一个HMM-GMM模型，然后通过Force-Alignment得到triphone级别的标签用于训练DNN。

用DNN来替代GMM得到的模型通常叫做HMM-DNN混合(hybrid)模型。除了用DNN替代GMM，还可以用DNN来实现特征提取，把MFCC特征再加上DNN的特征作为HMM-GMM的特征，这种特征叫做Tandem特征。DNN相当于GMM有如下优点：


* GMM的输入要求各个维度是不相关的，因为为了简化，通常加上GMM的协方差矩阵是对角阵。
* DNN可以学习深层次的特征，这是深度学习相对于传统机器学习最大的优势

## End-to-End语音识别系统

前面介绍的HMM-DNN模型还是需要使用HMM来建模状态的时序信息，整个系统还是非常复杂。因此现在也有很多研究放到了End-to-End的语音识别系统，也就是完全抛弃HMM模型。目前End-to-End的系统的效果达到以前最好的系统的水平，比如Google声称最新的End-to-End模型，词错率降至5.6%，相比传统的商用方法实现了16\%的相对(不是绝对)词错误率下降。

有两大类的End-to-End系统，一种是使用seq2seq模型，这是非常自然的想法，因为语音识别的输入是一个语音波形时序信号，而输出是词的序列。这和用于机器翻译的seq2seq模型基本是类似的，有兴趣的读者可以参考相关论文，如[Listen, Attend and Spell, LAS](https://arxiv.org/abs/1508.01211)，[State-of-the-art Speech Recognition With Sequence-to-Sequence Models](https://arxiv.org/abs/1712.01769)，[Wav2letter: an end-to-end convnet-based speech recognition system](https://arxiv.org/abs/1609.03193)。也可以参考[ESPNet](https://github.com/espnet/espnet)、[OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq)和[Wav2letter++](https://github.com/facebookresearch/wav2letter)等开源实现。后文我们主要介绍基于CTC模型(损失函数)的End-to-End系统。


