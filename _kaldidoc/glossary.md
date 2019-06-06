---
layout:     post
title:      "术语表"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Glossary of terms](http://kaldi-asr.org/doc/glossary.html)。
 <!--more-->
 
**目录**
* TOC
{:toc}


本文包括Kaldi的用户需要知道的一些术语。

## acoustic scale

acoustic scale用于decoding，在C++程序里的参数名为–acoustic-scale，在命令行程序里为-acwt。这是对声学log概率的一个缩放因子，在HMM-GMM和HMM-DNN系统中都可以使用。它的常见值为0.1。而语言模型的scale看成1，那么声学模型的权重是比较低的。在scroing的脚本里我们会搜索不同的语言模型的权重(通常是7到15之间)。语言模型的scale可以看成是acoustic scale的倒数。比如acoustic scale为0.1并且lm的scale为1等价于acoustic scale为1并且lm scale为10。

## alignment

它表示使用Viterbi算法得到的一个句子最佳路径的HMM状态序列。在Kaldi里alignment是一个transition-id序列。很多时候alignment是参考一个句子(utterance)的标注词序列的，这种情况叫做forced alignment。因为使用Viterbi算法得到的最优状态序列不一定能对于正确的词序列，比如使用Viterbi算法输出的是"/b/a/t"，但是实际句子可能是bad。forced alignment可以认为在所有正确词序列能生成的状态序列中的最佳状态。比如假设正确的句子只有bad这个词，语音信号是5帧(不考虑silence)，我们这里假设是monophone，那么bad能生成的状态序列只可能是：
```
b b b a d
b b a d d
b b a a d
b a d d d
b a a d d
b a a a d
```
因此force alignment是从这些可能的状态序列里选择最佳的一个。

lattices also contain alignment information as sequences of transition-ids for each word sequence in the lattice
lattice的每一个词序列都包含了对应的alignment，也就是一个transition-id的序列。

show-alginments命令行工具可以输出人类可读的对齐信息。(前面的[Kaldi教程(二)]({{ site.baseurl }}/kaldidoc/tutorial2)有介绍)

## cost

WFST算法中的权值，不管是声学模型得分还是语言模型得分，都叫做cost。更多细节可以参考[Lattices in Kaldi](http://kaldi-asr.org/doc/lattices.html)。cost通常是-log的概率，但是可以是scaling过的。

## forced alignment

参考前面的alignment。

## lattice
 
lattice是一个句子可能性比较大的词序列(n-best)的一种紧凑表示，它包含对齐(时间)和cost的信息。更多细节请参考[Lattices in Kaldi](http://kaldi-asr.org/doc/lattices.html)。


## likelihood

一个数学概念，表示给定模型的条件下看到某个数据的概率。在Kaldi里通常表示声学模型的概率$P(X \vert q)$，其中X是观察(比如MFCC特征的)，而q是HMM状态。因为这个概率通常很小，为了避免乘法下溢，我们通常取log把它变换到log域。对于DNN这样的分类模型来说，通常只能计算$P(q \vert X)$，我们可以除以状态的先验概率$P(q)$来得到pseudo-likelihoods，详细内容读者可以参考[PyTorch-Kaldi简介之原理回顾]({{ site.baseurl }}/books/pytorch-kaldi/#%E5%8E%9F%E7%90%86%E5%9B%9E%E9%A1%BE)。

 
## posterior

posterior是后验概率(posterior probability)的简称，这是一个非常通用的数学概率，一般表示给定相关数据后某个随机变量的概率。在Kaldi里，如果看到"posterior"(简称post)，那么它通常表示给定观察向量(MFCC)的条件下HMM状态(更加准确的说是transition-id，这里我们暂时不区分它们)的概率，也就是$P(q \vert X)$。因为是概率，所以有$\sum_q P(q \vert X)=1$。如果posterior来自与alignment或者lattice那么通常它会非常的peaky(也就是概率大部分集中在某个状态上，而其它的状态概率几乎为零)。Alignment和lattice可以转成transtion-id上的posterior(参考lattice-to-post.cc)，也可以转成lattice边上的posterior(参考ali-to-post.cc和lattice-arc-post.cc)。而transtion-id上的posterior又可能这成pdf-id或者phone上的posterior，具体参考ali-to-post.cc, post-to-pdf-post.cc 和 post-to-phone-post.cc。


## pdf-id

聚类后的CD HMM状态的下标，从零开始。通过这个下标就可以得到的pdf(GMM的概率密度函数)。更多信息参考[Integer identifiers used by TransitionModel](http://kaldi-asr.org/doc/hmm.html#transition_model_identifiers)。

## transition-id

1开始的下标，把包含了pdf-id、phone的id以及是自跳转还是非自跳转。它会出现在lattices、decoding graphs和alignments中。我们可以认为它是一种比HMM状态更细粒度的东西，一般理解成HMM的状态(id)就行。

## transition model

TransitionModel对象编码了HMM模型的状态跳转概率。更多信息参考[ Transition models](http://kaldi-asr.org/doc/hmm.html#transition_model)。命令行工具show-transitions可以显示这些信息，前面的[Kaldi教程(二)]({{ site.baseurl }}/kaldidoc/tutorial2/#monophone%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)有这个命令的用法，读者可以参考。


## G.fst

语言模型的FST通常在data/lang目录下(参考[Data preparation](http://kaldi-asr.org/doc/data_prep.html#data_prep_lang))。它通常是N-Gram语言模型的FST表示。它大部分情况下每条边的输入后和输出符号相同(也就是WFSA)，但是如果是backoff的表示方法，则backoff的边的输入会有#0。如果像去掉这些消歧符号，可以使用命令"fstproject –project_output=true"。消歧符号在FST的复合的时候必须保留，否则复合的结果就不(一定)是可以确定化的(determinizable)。但是在某些场合，比如语言模型的重打分，这是不需要的。



