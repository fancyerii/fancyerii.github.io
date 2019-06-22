---
layout:     post
title:      "关于Kaldi"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[About the Kaldi project](http://kaldi-asr.org/doc/about.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

## Kaldi是什么

Kaldi是一个C++实现的语音识别工具，它使用Apache v2.0开源协议。它的主要目标用户是语音识别的研究者(而不是普通的用户)。

## 名字由来

Kaldi是埃塞俄比亚人的牧羊人，他发现了咖啡。

## Kaldi和其它工具的对比

Kaldi的目标和HTK类似，它需要提供现代和灵活的代码，使用C++实现，容易修改和扩展。它包括如下重要特性：

* 代码级别集成WFST
    * Kaldi是把OpenFST作为一个库编译进来。(而不是脚本的方式集成)。
* 广泛的线性代数支持
    * Kaldi包括封装了标准BLAS和LAPACK库的[矩阵库](http://kaldi-asr.org/doc/matrix.html)。
* 可扩展的设计
    * 如果可能的话，我们提供的算法会尽量的通用。比如，我们的decoder是基于模板，模板的对象根据(frame, fst-input-symbol)来计算score。这就意味着decoder可以很容易的用神经网络梯度GMM模型。
* 开源协议
    * Apache 2.0开源协议，最自由的开源协议。
* 完整的recipe
    * 对于很多常见语音数据集(主要是LDC的数据，当然也有一些其它开源数据集)都提供完整的recipe，从而可以完整的复现整个过程。

包含完整的reciple是Kaldi的重要特性。这样其他人就可以轻松的复现整个实验过程。

我们尽量多的提供文档，但是短期内我们不太可能提高像HTK那样透彻的文档。大部分Kaldi的文档都是给领域的专家使用的，因为Kaldi的目标用户是这个行业的研究者。总体来说，Kaldi不是一个可以"for dummy"的工具。

## Kaldi的特色

* 强调通用算法和recipe
    * 通用算法指的是像线性变换这样的算法，而不是那种只能针对某种特定语音的算法。当然我们也不那么绝对的教条，如果这些特定的算法很有效的话也可以使用。
    * 我们更喜欢能够用于不同数据集的recipe，而不是那些只能用于某个特定数据集的recipe
* 我们喜欢能被证明正确的算法
    * recipe应该尽量避免可能失败。比如WFST的weight pushing虽然大部分情况下会成功，但是偶尔也会失败，我们就尽量不使用它。
* Kaldi的代码都是通过彻底的测试的
    * 几乎所有代码都有测试
* 尽量避免把简单问题复杂化
    * 当构建一个大型的语音识别系统的时候很容易出现大量很少使用的代码。为了避免这点，每一个命令行工具都只针对有限的情况(而不是为了追求通用性搞得非常复杂，比如某个decoder就只考虑GMM作为声学模型，而不是设计某个通用的decoder兼容各种不同的声学模型)。
* Kaldi是容易理解的
    * 虽然Kaldi是一个庞大的系统。但是对于每一个工具，我们希望它容易被理解。如果能够增加可读性，我们不在乎某些代码的冗余。
* Kaldi是容易复用和重构的
    * 各个模块尽量松耦合。这就意味着一个头文件需要include的头文件尽可能少。比如矩阵库，它只依赖于下面的子目录而完全不依赖其它部分，因此它可以独立于Kaldi的其它部分被使用(比如把它当成一个普通的和Kaldi完全没有关系的矩阵库使用)。

## 项目状态


目前，我们的代码和脚本实现了大部分标准技术，包括标准的线性变换、MMI、boosted MMI和MCE区分式训练，也包括特征空间的区分性训练(类似与fMPE，但是基于boosted MMI)。我们有WSJ和RM数据集的recipe，也有Switchboard的reciple。因为我们不使用外部的数据来训练语言模型，所以我们在Switchboard上的效果并不特别好。

因为维护多个版本代价很高，因此我们只维护最新的版本，所以用户应该定期更新到最新的master分支。

## 在论文中引用Kaldi

在论文中引用Kaldi的方法为：
```
@INPROCEEDINGS{
         Povey_ASRU2011,
         author = {Povey, Daniel and Ghoshal, Arnab and Boulianne, Gilles and Burget, Lukas and Glembek, Ondrej and Goel, Nagendra and Hannemann, Mirko and Motlicek, Petr and Qian, Yanmin and Schwarz, Petr and Silovsky, Jan and Stemmer, Georg and Vesely, Karel},
       keywords = {ASR, Automatic Speech Recognition, GMM, HTK, SGMM},
          month = dec,
          title = {The Kaldi Speech Recognition Toolkit},
      booktitle = {IEEE 2011 Workshop on Automatic Speech Recognition and Understanding},
           year = {2011},
      publisher = {IEEE Signal Processing Society},
       location = {Hilton Waikoloa Village, Big Island, Hawaii, US},
           note = {IEEE Catalog No.: CFP11SRW-USB},
}
```

