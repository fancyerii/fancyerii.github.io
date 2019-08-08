---
layout:     post
title:      "基于WFST的语音识别解码器"
author:     "lili"
mathjax: false
sticky: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - 解码器
    - WFST
---

本系列文章介绍基于WFST的语音识别解码器的理论知识。前面的[微软Edx语音识别课程](/2019/05/25/dev287x/)的[最后一个实验](/dev287x/decoder/)因为没有足够的理论知识，很难读懂其中的代码，因此本系列文章介绍这些缺失的内容。

* 8/8更新[语音识别系统概述](/wfst/overview/)，<span class='zz'>本文回顾WFST之前的解码器基础知识，便于没有基础的读者了解最基本的Viterbi算法和Beam搜索算法、Word Lattice等基本概念。</span>

 <!--more-->
 
 


#### [语音识别系统概述](/wfst/overview/)



