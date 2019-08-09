---
layout:     post
title:      "其它工具"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Other Kaldi utilities](http://kaldi-asr.org/doc/util.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}


本文介绍Kaldi代码里的常见工具函数的概览。

这不包括自成体系的一些工具，比如矩阵库、I/O、logging与错误报告和命令行parsing等。

## Text相关的工具

In text-utils.h are various functions for manipulating strings, mostly used in parsing. Important ones include the templated function ConvertStringToInteger(), and the overloaded ConvertStringToReal() functions which are defined for float and double. There is also the SplitStringToIntegers() template whose output is a vector of integers, and SplitStringToVector() which splits a string into a vector of strings.

[text-utils.h](http://kaldi-asr.org/doc/text-utils_8h.html)包含了一些操纵字符串的函数，注意是用于parsing。比较重要的函数包括模板函数[ConvertStringToInteger](http://kaldi-asr.org/doc/namespacekaldi.html#a39eb3f7ce5fab6dbbcb396b3f8e196f4)、用于float和double的重载函数[ConvertStringToInteger](http://kaldi-asr.org/doc/namespacekaldi.html#aa624cc5f189fcc01469558a0865c1f3c)。模板函数[SplitStringToIntegers](http://kaldi-asr.org/doc/namespacekaldi.html#ae95740f87fc2fb79419753e21396c647)会把字符串表示的整数序列变成vector，[SplitStringToVector](http://kaldi-asr.org/doc/namespacekaldi.html#ac332ce23fcf74c011cfa83276cd08477)把字符串split成字符串的vector。

## STL工具
In stl-utils.h are templated functions for manipulating STL types. A commonly used one is SortAndUniq(), which sorts and removes duplicates from a vector (of an arbitrary type). The function CopySetToVector() copies the elements of a set into a vector, and is part of a larger category of similar functions that move data between sets, vectors and maps (see a list in stl-utils.h). There are also the hashing-function types VectorHasher (for vectors of integers) and StringHasher (for strings); these are for use with the STL unordered_map and unordered_set templates. Another commonly used function is DeletePointers(), which deletes pointers in a std::vector of pointers, and sets them to NULL.

[stl-utils.h](http://kaldi-asr.org/doc/const-integer-set_8h.html)包含用于处理STL类型的模板函数。最常用的一个是[SortAndUniq](http://kaldi-asr.org/doc/namespacekaldi.html#a3dd7a9cc33032bfb870b4b6c053822db)，它的作用是对一个vector进行排序和去重。函数[CopySetToVector](http://kaldi-asr.org/doc/namespacekaldi.html#afaa2d6854352e931e19778f6f8608f2a)用于把一个集合复制到一个vector，和它类似的还有很多在set、vector和map直接互相转换的函数。同时也包含函数函数类型[VectorHasher](http://kaldi-asr.org/doc/structkaldi_1_1VectorHasher.html)和[StringHasher](http://kaldi-asr.org/doc/structkaldi_1_1StringHasher.html)，它们用于STL的unordered_map和unordered_set模板类。另一个常用的函数是[DeletePointers](http://kaldi-asr.org/doc/namespacekaldi.html#a34158c6c567bc6c51c90af0251395d6a)，它的作用是删除std::vector里的指针，然后设置为NULL。

## 数学工具


在kaldi-math.h里，除了用于补充标准math.h里缺失的函数，还包含如下函数：

* 生成随机数的函数：[RandInt](http://kaldi-asr.org/doc/namespacekaldi.html#a0361eab3c5ebb78e6be060e6fa78e1a1), [RandGauss](http://kaldi-asr.org/doc/namespacekaldi.html#a3871a9c94a69d8807ea4c1d94e2d4ab9), [RandPoisson](http://kaldi-asr.org/doc/namespacekaldi.html#a913cf12460baf36ca015809e2b35e700)
* [LogAdd](http://kaldi-asr.org/doc/namespacekaldi.html#a4ae8be9c3451f1dfa8cf05ffa48085ae)和[LogSub](http://kaldi-asr.org/doc/namespacekaldi.html#ad0ddf41b480c417d53ed81a3e59c5f12)
* 用于测试和assert近似相等的函数，包括[ApproxEqual](http://kaldi-asr.org/doc/namespacekaldi.html#a99dcbb9fdaa6c980ba67941d45adf017), [AssertEqual](http://kaldi-asr.org/doc/namespacekaldi.html#aeb6ba6085493e3f15358e49751342177), [AssertGeq]和[AssertLeq]

## 其它工具

[const-integer-set.h](http://kaldi-asr.org/doc/const-integer-set_8h.html)包含一个[ConstIntegerSet]()类用于存储整数集合，它可以实现高效的方式存储和查询。不过它要求这个集合构造完成后就不能修改了。这用于决策树的代码里。根据这些整数值的大小，内部可能存储为vector\<bool>或者排过序的整数的vector。

[timer.h](http://kaldi-asr.org/doc/timer_8h.html)里包括一个用于跨平台的实现计时[Timer](http://kaldi-asr.org/doc/classkaldi_1_1Timer.html)类。
 
其它的工具函数和类在[simple-io-funcs.h](http://kaldi-asr.org/doc/simple-io-funcs_8h.html)和[hash-list.h](http://kaldi-asr.org/doc/hash-list_8h.html)，它们用于特殊场景。其它一些矩阵代码依赖的函数和宏在[kaldi-utils.h](http://kaldi-asr.org/doc/kaldi-utils_8h.html),这包括字节的swapping、内存对齐和编译时的assertion。
