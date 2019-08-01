---
layout:     post
title:      "Kaldi的I/O机制的命令行用法"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Kaldi I/O from a command-line perspective](http://kaldi-asr.org/doc/io_tut.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

 
本文从命令行工具使用者的角度来介绍Kaldi的I/O机制。关于代码级别的介绍请参考[Kaldi的I/O机制](/kaldidoc/io/)。
 
## 非Table的I/O

我们首先介绍非Table的I/O。它指的是只包含一两个对象(比如声学模型文件；变换矩阵)的文件或者流。而不是Table：敢上是类似map的数据结构，其中key是字符串。

* Kaldi的默认输出是二进制的，我们可以使用\-\-binary=false来改变默认行为
* 许多对象有对应的"copy"程序，比如copy-matrix和gmm-copy，可以使用这些程序来把二进制的格式转成文本格式，比如"copy-matrix --binary=false foo.mat -"，它会读取二进制的foo.mat这个矩阵，然后输出文本格式到标准输出("-")
* 通常磁盘上的文件和内存里的C++对象存在一对一的关系，当然也有时候一个文件里包含多个对象(声学模型文件里先是一个[TransitionModel](http://kaldi-asr.org/doc/classkaldi_1_1TransitionModel.html)对象，然后是声学模型本身)
* Kaldi程序通常知道它要读取的对象的类型，而不需要根据流里的内容来判断到底读取的是那种类型的对象
* 和perl类似，文件名可以被替换为"-"(代表标准输入或者输出)或者"|gzip -c >foo.gz"、"gunzip -c foo.gz|"，前者表示把流(内容)通过管道"\|"重定向给gzip从而压缩到foo.gz；后者表示从foo.gz读取后重定向到标准输入作为程序的输入。
* 对于读取操作，我们也支持类似"foo:1045"这样的"文件"，它表示从foo文件的1045(offset)开始读取
* 为了表示扩展文件名，我们通常使用"rxfilename"和"wxfilename"这个两个术语，扩展文件名的详细介绍请参考[这里](/kaldidoc/io/#%E6%89%A9%E5%B1%95%E6%96%87%E4%BB%B6%E5%90%8D)。

为了展示上面的概念，请把$KALDI_ROOT/src/bin加到PATH环境变量里，其中$KALDI_ROOT是Kaldi的顶层目录，然后输入：

```
$ echo '[ 0 1 ]' | copy-matrix - -
copy-matrix - - 
BFM �?LOG (copy-matrix[5.5.388~1-777f8]:main():copy-matrix.cc:112) Copied matrix to -

```

上面会输出log信息，同时我们看到一些乱码一样的东西，这是二进制的矩阵。接下来：

```
$ echo '[ 0 1 ]' | copy-matrix --binary=false - -
copy-matrix --binary=false - - 
 [
  0 1 ]
LOG (copy-matrix[5.5.388~1-777f8]:main():copy-matrix.cc:112) Copied matrix to -
```

我们看到的是人更加可读的文本格式的矩阵。上面的输出看起来把矩阵的内容和log信息混在一起了，但是实际上内容是输出到标准输出，而log是输出到标准错误，只不过默认的是都显示在terminal里了，我们可以把标准错误重定向到别的地方，比如"2>/dev/null"。

Kaldi程序可以使用观点连接起来，下面是一个例子：


```
$ echo '[ 0 1 ]' | copy-matrix - - | copy-matrix --binary=false - -
copy-matrix - - 
copy-matrix --binary=false - - 
LOG (copy-matrix[5.5.388~1-777f8]:main():copy-matrix.cc:112) Copied matrix to -
 [
  0 1 ]
LOG (copy-matrix[5.5.388~1-777f8]:main():copy-matrix.cc:112) Copied matrix to -
```

上面首先使用echo把文本格式的矩阵重定向给copy-matrix，这个copy-matrix从标准输入读取这个矩阵，然后用二进制(默认)的格式输出到标准输出，然后重定向给第二个copy-matrix，最终以文本的格式输出。当然用两个copy-matrix是没有意义的，这里只是为了演示管道的用法，它等价于：
```
copy-matrix 'echo [ 0 1 ]\|' '\|copy-matrix --binary=false - -'
```

上面的命令只有一个copy-matrix，然后把'echo [0 1]|'作为其输入(等价于执行这个echo重定向给copy-matrix)，然后把copy-matrix的结果输出到'\|copy-matrix --binary=false - -'，因为这个字符串是以\|开头，因此有等价于把第一个copy-matrix的输出作为它的输入。


## Table I/O
 

对于Table(用字符串作为key的对象集合)，Kaldi有专门的I/O机制。比如通过utterance-id索引的特征矩阵、通过speaker-id索引的说话人自适应变换矩阵。用于索引的字符串必须是非空并且不包含whitespace。更多细节请参考[Kaldi的I/O机制](/kaldidoc/io/)。


Table有两种形式：archive和script文件。它们的区别是archive包含实际的数据，而script文件只是指向包含数据的位置。


读取Table对象的程序需要一个叫做rspecifier的字符串来指明怎么读取数据，类似的在输出的时候也需要一个wspecifier。这两个字符串指明读写的文件是archive还是script文件、文件的位置和其它一些选项。rspecifier的典型例子是"ark:-"，它表示把标准输入当成一个archive文件；或者"scp:foo.scp"，说明从foo.scp读取，foo.scp说明了怎么读取具体的内容。下面是一些要点：

* 冒号之后的内容是wxfilename和rxfilename(这和非Table的I/O一样)，它可以是具体的文件，也可以是类似管道的命令和标准输入/输出。
* Table只能包含一种对象(比如浮点类型的矩阵)
* rspecifier和rspecifier可能会有选项，常见的为：
    * 在rspecifier里，"ark,s,cs:-"表示我们读取的时候(这里是从标准输入读取)，我们期望key是排过序的(s)，而且读取它的程序是按照顺序(cs)来访问key的(也就是访问了key "3"之后肯定不会访问比它小的key)。这样Kaldi程序可以使用少量的内存来随机的读取这个流。
    * 如果数据不太大并且我们很难保证顺序(比如说话人自适应的变换矩阵)，去掉",s,cs"也不会有什么问题。
    * 如果一个程序的参数有多个rspecifier，那么它通常会顺序的遍历第一个rspecifier，然后随机的访问后面的rspecifier。因此第一个rspecifier通常不需要",s,cs"。
    * "scp,p:foo.scp"，p指的是如果foo.scp里的指向的某些文件不存在，那么可以静悄悄的忽略这个错误。如果没有p选项，那么它就会crash。
    * 对于输出，选项",t"表示输出文本格式，比如"ark,t:-"。命令行选项\-\-binary对于archive文件不起作用，只能使用",t"选项。
 * script文件的每一行都是"<key> <rspecifier\|wspecifier>"，比如"utt1 /foo/bar/utt1.mat"。rspecifier和wspecifier可以包含空格，比如"utt1 gunzip -c /foo/bar/utt1.mat.gz\|"。它只是用第一个空格来区分key和value。
* archive格式为："<key1> <object1> <newline> <key2> <object2> <newline> ..."
* 多个archive拼接起来还是一个合法的archive，但是注意拼接的顺序，如果你想保证key是排序的，应该尽量避免：

```
    "cat a/b/*.ark"
```
* 虽然比较罕见，但是script文件也可以作为输出，比如输出到"scp:foo.scp"。假设程序需要输出key "utt1"，那么它会在foo.scp里找key为"utt1"的行，如果找到了，那么它的value就作为真正的输出位置，但是如果找不到，则会crash。
* 可以同时输出一个archive和scri文件，比如"ark,scp:foo.ark,foo.scp"。archive是实际的输出的文件，而script文件是索引，比如script文件里可能包含"utt1 foo.ark:1016"，它的意思是key "utt1"在foo.ark文件的offset是1016。这种用法的场景是：你需要随机的根据key来访问value。

* 很tricky的使用一行的script文件的技巧：
```
         echo '[ 0 1 ]' | copy-matrix 'scp:echo foo -|' 'scp,t:echo foo -|'
```

    copy-matrix有两个参数，分别是rspecifier和wspecifier，这里是"scp:echo foo -\|"和"scp,t:echo foo -\|"。"scp:echo foo -\|"等价与一个bar.scp——这个scp只有一行"foo -"，因此copy-matrix会从标准输入(也就是echo的输出)读取key为"foo"的矩阵([0,1])，然后输出到一个script文件，这个script文件也只包含一行"foo -"，因此它会把key为foo的矩阵([0,1])输出到标准输出。这个trick不应该被滥用，如果你需要经常使用这个trick，则很可能你的程序有问题。

* 如果你像从一个archive里提取一个key的value，你可以使用",p"选项然后使用"scp:"作为输出，这样的话其它的key因为找不到而被忽略，只有这个key的value被写出去。比如你想在一个archive里提取key为"foo_bar"的对象，那么可以使用：

```
         copy-matrix 'ark:some_archive.ark' 'scp,t,p:echo foo_bar -|'
```
它的意思是读取some_archive.ark，然后输出到某个scp文件，但是这个scp文件只有一行——"foo_bar -\|"，因为有",t"，所以除了foo_bar之外的key都被忽略，只有key为"foo_bar"的对象被写到"-"(标准输出上)。

* 某些情况下读取archive的代码允许一定程度的类型转换，比如在float和double之间、Lattice和CompactLattice之间。

## Table I/O(带范围)

在scp文件里可以指定文件的offset。比如dump一个特征文件，我们可以把它表示为scp文件：
```
 utt-00001  /some/dir/feats.scp:0
 utt-00002  /some/dir/feats.scp:16402
 ...
```
它表示第一个utt-00001在feats.scp(似乎应该是feats.ark?)的offset为0的位置(文件的开头)；而utt-00002在feats.scp的offset为16402的位置。我们也可以用slice提取这个特征矩阵的某些行，比如：
```
 utt-00001  /some/dir/feats.scp:0[0:9]
 utt-00001  /some/dir/feats.scp:0[10:19]
 ...
```

它表示第一个utt-00001是特征矩阵的第0到第9行，第二个utt-00001是第10到第19行。注意：如果Table不是随机访问的话，是可以有重复的key的。类似的我们可以提前某些列：
```
 utt-00001  /some/dir/feats.scp:0[:,0:12]
 utt-00001  /some/dir/feats.scp:0[:,13:25]
 ...
```
它提取特征矩阵的0-12列以及13-25列。我们还可以同时指定行和列的范围，从一个大矩阵里提取一个小的矩阵：
```
 utt-00001 /some/dir/feats.scp:0[10:19,0:12]
```

## Utterance-to-speaker和speaker-to-utterance的映射

Many Kaldi programs take utterance-to-speaker and speaker-to-utterances maps– files called "utt2spk" or "spk2utt". These are generally specified by command-line options –utt2spk and –spk2utt. The utt2spk map has the format

许多Kaldi程序可以接受一个从utterance-to-speaker或者speaker-to-utterances的映射文件的选项，这些文件简称为"utt2spk"和"spk2utt"。这可以通过\-\-utt2spk和\-\-spk2utt命令行选项来知道。utt2spk映射的格式为：

```
utt1 spk_of_utt1
utt2 spk_of_utt2
...
```

而spk2utt的格式为：
```
spk1 utt1_of_spk1 utt2_of_spk1 utt3_of_spk1
spk2 utt1_of_spk2 utt2_of_spk2
...
```

这些文件用于说话人自适应，比如找到某个utterance的说话人是谁、或者找到某个说话人说过的所有utterance。注意utterance-to-speaker必须要排序(参考[数据准备](/kaldidoc/dataprep/)。不过Kaldi只是把它当成一个archive，因此我们看到的命令行选项是"\-\-utt2spk=ark:data/train/utt2spk"。因此utt2spk的文件就是通用的archive格式："<key1> <data> <newline> <key2> <data> <newline>"。在代码级别，utt2spk文件被看成Table，key是utterance id，value是speaker id；而spk2utt也是Table，key是spaker id，value是utterance id的list。


