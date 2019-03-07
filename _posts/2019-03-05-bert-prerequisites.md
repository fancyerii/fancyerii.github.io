---
layout:     post
title:      "BERT课程预备知识"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 深度学习
    - BERT
---

 本文是作者即将在CSDN作直播的课程的预备知识，对课程感兴趣但是没有相关背景知识的同学可以提前学习这些内容。
 <!--more-->
 
**目录**
* TOC
{:toc}
 
## 背景知识
为了理解课程的内容，读者需要以下背景知识。
* 深度学习基础知识
* Word Embedding
* Language Model
* RNN/LSTM/GRU
* Seq2Seq模型
* Attention机制
* Tensorflow基础知识
* PyTorch基础知识

## 深度学习基础知识
 

* [从Image Caption Generation理解深度学习（part I）](https://www.easemob.com/news/739)
 
    介绍机器学习和深度学习的基本概念。
    
* [从Image Caption Generation理解深度学习（part II）](https://www.easemob.com/news/740)
  
    介绍多层神经网络(DNN)的基本概念

* [从Image Caption Generation理解深度学习（part III）](https://www.easemob.com/news/1445)

    介绍反向传播算法，不感兴趣的读者可以跳过细节。

* [自动梯度求解——反向传播算法的另外一种视角](https://www.easemob.com/news/742)

    建议了解一下就行，自动梯度是深度学习框架的基础。我们通常不需要实现反向算法，因为框架通常帮我们做了，但是了解一下它的原理是有用的。

* [自动梯度求解——cs231n的notes](https://www.easemob.com/news/752)

    内容主要参考CS231N课程的Notes，还是介绍自动梯度。
   
* [自动梯度求解——使用自动求导实现多层神经网络](https://blog.csdn.net/qunnie_yi/article/details/80126965) 

    使用自动梯度实现多层神经网络，对怎么自己实现自动梯度感兴趣的读者可以参考，不感兴趣的可以跳过。
   
 * [详解卷积神经网络](https://www.easemob.com/news/754)
 
    卷积神经网络的介绍。
    
* [Theano tutorial和卷积神经网络的Theano实现 Part1](https://blog.csdn.net/qunnie_yi/article/details/80127692) 

    卷积神经网络的实现，代码实现是用theano的，建议了解一下就行。现在更建议使用Tensorflow或者PyTorch。
     
* [Theano tutorial和卷积神经网络的Theano实现 Part2](https://blog.csdn.net/weixin_33695082/article/details/87289237) 

    同上。
    
* [卷积神经网络之Batch Normalization的原理及实现](https://www.easemob.com/news/758) 
    
    介绍Batch Normalization。读者了解一下原理即可，现在的框架都有现成的。
      
* [卷积神经网络之Dropout](https://www.easemob.com/news/759)

    Dropout是最常见也最有用的防止过拟合的技巧之一。
      
* [三层卷积网络和vgg的实现](https://www.easemob.com/news/760)
    
    自动动手实现CNN，并且实现常见的CNN架构——VGG。比较关注实现细节，不感兴趣的读者可以跳过。
      
* [Caffe训练ImageNet简介及深度卷积网络最新技术](https://www.easemob.com/news/761) 

    训练ImageNet的过程可以跳过，因为现在很少需要自己从零开始训练ImageNet了，大部分框架都有Pretraining好的模型。读者可以了解一下ResNet和Inception，后面的BERT也使用到了残差连接，在如今(2019年)这是非常常用的技巧了。


* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) 

    Michael Nielsen的免费书籍，作者前面的文章参考了很多里面的内容。
    
* [CS231N课程](http://cs231n.stanford.edu/) 

    斯坦福的课程，作者的文章也参考了一些内容。
    

## Word Embedding

* [Word Embedding教程]({{ site.baseurl }}{% post_url 2019-03-08-word-embedding %})

    理解Word Embedding的概念即可，跳过Word2Vec的推导、Softmax和Negative Sample并不影响后续阅读。


