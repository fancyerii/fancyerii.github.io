---
layout:     post
title:      "Huggingface Transformer教程" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - Transformer
    - BERT
    - PyTorch
    - Tensorflow
    - Huggingface
---

本文介绍Huggingface Transformer这个好用的工具的用法。

<!--more-->

**目录**
* TOC
{:toc}

## 简介

目前各种Pretraining的Transformer模型层出不穷，虽然这些模型都有开源代码，但是它们的实现各不相同，我们在对比不同模型时也会很麻烦。[Huggingface Transformer](https://huggingface.co/transformers/)能够帮我们跟踪流行的新模型，并且提供统一的代码风格来使用BERT、XLNet和GPT等等各种不同的模型。而且它有一个[模型仓库](https://huggingface.co/models)，所有常见的预训练模型和不同任务上fine-tuning的模型都可以在这里方便的下载。截止目前，最新的版本是3.0.2，本文的介绍基于3.0.2。

## 安装

Huggingface Transformer需要安装Tensorflow 2.0+ **或者** PyTorch 1.0+，它自己的安装非常简单：

```
pip install transformers
```

## 用法


