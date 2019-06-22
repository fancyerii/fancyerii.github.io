---
layout:     post
title:      "Kaldi的Coding Style"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[The Kaldi coding style](http://kaldi-asr.org/doc/style.html)，介绍Kaldi的Coding Style。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

 
 
为了和OpenFst一致，我们采样了和它一样的Coding Style。通过流量Kaldi的代码很快可以发现其代码风格。重要的风格包括：

* Token的命名规则，比如MyTypeName, MyFunction, my_class_member_var_, my_struct_member, KALDI_MY_DEFINE, kGlobalConstant, kEnumMember, g_my_global_variable, my_namespace, my-file.h, my-file.cc, my-implementation-inl.h

* 函数参数，所有的引用都是const的，输入在输出前面。

* 每行最多80个字符(少数例外)，{和函数在同一行。 

* I/O：使用C++风格的I/O，对象的I/O使用特定的约定(参考[Kaldi I/O mechanisms](http://kaldi-asr.org/doc/io.html))。 

* 函数参数：我们不允许非const的引用(如果要修改请使用指针而不是非const的引用)，但是iostream可以使用const的引用。函数的输入参数必须在输出参数之后。 

* 错误状态通常使用异常来表示(参考[Kaldi logging and error-reporting](http://kaldi-asr.org/doc/error.html))。

* 对于"普通"的整数，我们尽量用int32。这是因为Kaldi的二进制的I/O(Kaldi I/O mechanisms](http://kaldi-asr.org/doc/io.html))使用定长的数据类型会更加方便。

* 对于"普通"的浮点数，我们使用BaseFloat，这是一个typedef，如果我们编译时设置了KALDI_DOUBLEPRECISION=1，那么就是double，否则就是float。但是我们的累加器都使用double。

* 所有的#define都以KALDI_开头，以便和其它的代码发生冲突。所有的Kaldi代码都放到kaldi这个namespace下，OpenFst的扩展除外，它们还是放到fst这个namespace下。

* 只有一个参数的类的构造函数必须加上"explicit"，这是为了避免编译器偷偷的类型转换带来的问题。

* 避免拷贝和赋值构造函数(通过KALDI_DISALLOW_COPY_AND_ASSIGN来禁止编译器自动生成)

* 处理STL算法需要的，我们尽量避免运算符重载

* 尽量避免使用函数重载，而使用不同的函数名字

* 使用C++风格的static_cast，而不是C风格的强制类型转换

* 尽可能的使用const

和Google C++更高冲突的包括：

* 我们是iostream，并且把它的非const的引用作为函数的参数，这是违背之前的规定的

* 对于get/set方法，假定类成员变量是x_，Google的风格是x()和set_x()。但是因为OpenFst的风格是X()和SetX()，比如Mean()和SetMean()。这个规则是新加的，因此老的代码可能还是Google的风格，以后我们会全部使用OpenFst风格。



