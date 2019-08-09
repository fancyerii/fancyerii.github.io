---
layout:     post
title:      "Kaldi的logging和错误报告"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Kaldi logging and error-reporting](http://kaldi-asr.org/doc/error.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

 
##  Overview


Kaldi程序的所有输出包括logging信息、警告和错误都会重定向到标注错误输出。这样的好处是我们的程序可以用于管道而不用担心这些信息和程序的输出混在一起。产生logging、警告和错误的最常见用法是通过KALDI_LOG、KALDI_WARN和KALDI_ERR这3个宏来实现。调用KALDI_ERR通常会终止程序(除非异常被捕获)。下面是展示这3个宏用法的代码片段：

```
KALDI_LOG << "On iteration " << iter
          << ", objective function change was " << delta;
if (delta < 0.0) {
  KALDI_WARN << "Negative objf change " << delta;
  if (delta < -0.1) 
    KALDI_ERR << "Convergence failure in EM";
}
```


注意上面的例子中输出的字符串没有包括换行(宏会自己在后面加换行)。下面是产生的message的典型示例输出：

```
  WARNING (copy-feats:Next():util/kaldi-table-inl.h:381) Invalid archive file format
```

对于不那么重要的log信息(或者太啰嗦)，你可以使用KALDI_VLOG，比如：
```
KALDI_VLOG(2) << "This message is not important enough to use KALDI_LOG for.";
```

这个信息只有在开启了\-\-verbose选项的基本大于等于括号里的值才会输出，比如上面代码示例里只有\-\-verbose选项大于等于2(比如指定\-\-verbose=2)才会输出。关于这个的更多信息请参考[Implicit command-line arguments](http://kaldi-asr.org/doc/parse_options.html#parse_options_implicit)。


某些(不好的)代码会直接把logging信息输出到标准错误输出(而不是使用宏)，这是不推荐的做法。

## Kaldi里的Assertion


Assertion应该使用KALDI_ASSERT宏。这比普通的assert()函数会打印出更多有用的信息，因为KALDI_ASSERT会打印出stack trace。此外KALDI_ASSERT更容易重新配置(reconfigurable)。

典型的assertion为：

```
KALDI_ASSERT(i < M.NumRows());
```
在assert失败的时候输出更多有用信息的一个trick是在assert条件后面加上"&& [some string]"，比如：
```
KALDI_ASSERT(ApproxEqual(delta, objf_change) && "Probable coding error in optimization");
```
如果assert成功，后面逻辑与一个非空字符串，也是成功的。如果assert失败，则会打印stack trace，也就会看到这个字符串了。


如果正常编译，assert会被执行，如果使用宏NDEBUG编译，则不会执行。对于循环里的可能消耗大量CPU的assert，我们建议使用这种模式：
```
#ifdef KALDI_PARANOID
  KALDI_ASSERT(i>=0);
#endif
```

宏KALDI_PARANOID在build的时候默认是on的。

## KALDI_ERR抛出的异常


当KALDI_ERR被调用的时候，它会把错误信息打印到标准输出，然后抛出类型为std::runtime_error的异常。这个异常包含错误信息的字符串以及stack trace信息。目前Kaldi程序的通常做法是在mian函数里使用try...catch捕获这个异常，打印错误信息到标准错误输出然后终止程序。这通常会导致错误信息会被打印两次。
 

在某些情况下，Kaldi代码会捕获的异常并且不会重新抛出(也就是说它会吞掉这个异常，这通常不是好的编程实践)。在Table类(参考[Table的概念](/kaldidoc/io/#table%E7%9A%84%E6%A6%82%E5%BF%B5))使用的Holder类的代码里(参考[Table的辅助类Holder](/kaldidoc/io/#table%E7%9A%84%E8%BE%85%E5%8A%A9%E7%B1%BBholder))会出现这种情况。在这里Read函数抛出第一场会把Table类代码捕获，然后返回一个boolean值来说明是否有异常(参考[KaldiObjectHolder::Read()](http://kaldi-asr.org/doc/classkaldi_1_1KaldiObjectHolder.html#a6c0ca0e40fa9d1fe27d1adfbc6e90e32))。根据不同的选项，比如"p"(permissive)，以及Table代码的调用方式，这可能抛出或者不抛出另一个新的异常。


除了std::runtime_error之外Kaldi程序唯一会抛出的其它异常可能就是std::alloc_error。


某些Kaldi的代码会直接抛出std::runtime_error异常，或者直接调用assert()函数，以后这些代码会被改成更标准的KALDI_ERR和KALDI_ASSERT宏。

## Kaldi编译时的Assertion
 
也可以在编译的时候做一些assertion(如果失败会导致编译错误)。这是通过kaldi-utils.h里的一些宏来实现的。这对于检查模板类被实例化成正确的类型非常有用。编译是的assertion示例为：


```
KALDI_COMPILE_TIME_ASSERT(kSomeConstant < 0);
...
template<class T> class foo {
   foo() { KALDI_ASSERT_IS_INTEGER_TYPE(T);
};
...
template<class T> class bar {
   bar() { KALDI_ASSERT_IS_FLOATING_TYPE(T);
}
```
