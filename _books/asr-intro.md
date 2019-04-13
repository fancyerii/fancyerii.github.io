---
layout:     post
title:      "语音识别简介"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 语音识别
    - MFCC
    - 《深度学习理论与实战：提高篇》
---

本文介绍语音识别的基本概念。更多文章请点击[深度学习理论与实战：提高篇]({{ site.baseurl }}{% post_url 2019-03-14-dl-book %})。
<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>
 <!--more-->
 
**目录**
* TOC
{:toc}


语音识别(Speech Recognition)的目标是把语音转换成文字，因此语音识别系统也叫做STT(Specch to Text)系统。语音识别是实现人机自然语言交互非常重要的第一个步骤，把语音转换成文字之后就由自然语言理解系统来进行语义的计算。

有的学者把语音识别和自然语言理解都放到一起叫做Speech and Language Processing，比如Dan Jurafsky等人的书[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)，讨论的内容包括语音识别和自然语言处理。在语音识别时会使用语言模型，这也是自然语言处理的研究对象，在很多其它自然语言处理系统比如机器翻译等都会使用到语言模型。

更多的时候这两个方向的研究并不会有太多重叠的地方，语音识别除了语言模型之外也不会考虑太多的"语义"。而自然语言处理假设的研究对象都是文本，他们并不关心文本是语音识别的结果还是用户从键盘的输入亦或是OCR(图像处理)扫描的结果。但是从人类的语言发展来说，我们都是首先有语言而后才有文字，即使到今天，仍然有一些语言只有声音而没有文字。虽然研究的时候需要有一个更具体的方向，但是也不能把Speech和Language完全割裂开来。

## 任务分类

语音识别的任务可以根据如下的一些维度来分类：

* 词汇量(vocabulary)大小

    分为小词汇量(small vocabulary)和大词汇量(large vocabulary)的语音识别。
* 说话人(Speaker)

    分为说话人相关(Speaker dependent)和说话人无关(Speaker independent)语音识别系统。
* 声学(Acoustic)环境 

    录音室 vs 不同程度的噪音环境。
* 说话方式(style) 

   连续(continously)说话还是一个词一个词(isolated words)的说话；计划(plan)好的还是spontaneous的——"呃，这个东西，不，那个是啥？"


这些维度的组合就决定了不同任务的难度，比如最早的语音识别系统只能识别孤立词(词之间有停顿，因此很容易切分)，而且词汇量很小(比如只能识别0-9之间的数字)。而现在的语音识别系统能够在噪声环境识别大词汇量的任务，而且说话人的方式是连续的，它可以处理不同说话人的差异甚至可以处理非标准的发音(比如带口音的普通话)。

## 常见概念

下面是一些常见的概念，因为本书的目的更多关注工程实现而不是研究语言学/语音学，所以只介绍会用到的一些基本概念。

语言(Language)是用于沟通的符号系统。语音(Speech)是由语言产生的声音，唱歌或者汽车的刹车声都不是语音。音素(Phoneme)是语言学的概念，比如/a/就是一个音素，英语有四五十个音素。因子(Phone)是一个声学(Acoustic)概念，表示不同的发音。一个音素可能对于多个不同的发音，比如/t/在"cat"和"stop"的发音是不同的，我们把不同的发音叫做allophone。


## 语音识别效果评测

语音识别的效果通常使用词错误率(Word Error Rate/WER)来评测。每段语音都会有一个正确的文本，语音识别系统也会输出一段文字，我们可以使用编辑距离的算法来计算三种错误：替换错误S , 删除错误D 和插入错误I，然后WER的计算公式为：$WER=\frac{S+D+I}{N}$。

编辑距离算法也是一种动态规划算法，不熟悉的读者可以参考[wiki](https://en.wikipedia.org/wiki/Edit_distance)。


