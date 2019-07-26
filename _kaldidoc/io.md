---
layout:     post
title:      "Kaldi的I/O机制"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Kaldi的I/O机制](http://kaldi-asr.org/doc/io.html)，介绍Kaldi的I/O机制。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

本文介绍Kaldi的输入-输出机制的概述。这里是介绍代码基本的I/O机制；关于命令行相关的用法，请参考[Kaldi I/O from a command-line perspective](http://kaldi-asr.org/doc/io_tut.html)。


## Kaldi类的输入/输出风格
Kaldi的类有统一的I/O接口。标准的接口如下：
```
class SomeKaldiClass {
 public:
   void Read(std::istream &is, bool binary);
   void Write(std::ostream &os, bool binary) const;
};
```

注意返回值为空；如果发生错误会抛出异常(参考[Kaldi logging and error-reporting](http://kaldi-asr.org/doc/error.html))。boolean的binary参数说明对象是作为二进制的格式还是文本个数来读写。调用的代码必须知道它读写的对象是二进制还是文本格式(请参考[Kaldi对象是如何保存到文件里的](#kaldi%E5%AF%B9%E8%B1%A1%E6%98%AF%E5%A6%82%E4%BD%95%E4%BF%9D%E5%AD%98%E5%88%B0%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84)。注意：这里的binary和(Windows下)文件打开模式是二进制还是文本模式并非同一个东西。请参考[Kaldi对象是如何保存到文件里的](#kaldi%E5%AF%B9%E8%B1%A1%E6%98%AF%E5%A6%82%E4%BD%95%E4%BF%9D%E5%AD%98%E5%88%B0%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84)来了解更多文件打开模式与Kaldi的二进制/文本模式的关系。

读写函数可以有其它的可选参数。比如读函数通常的常见形式为：
```
class SomeKaldiClass {
 public:
  void Read(std::istream &is, bool binary, bool add = false);
};
```

如果add为True，假设类的内容非空，则Read函数会把磁盘上的内容(比如统计信息)加到类的当前内容里面。


## 基本类型和STL类型的I/O机制

如果像查看相关的函数列表，可以参考[Low-level I/O functions](http://kaldi-asr.org/doc/group__io__funcs__basic.html)。我们提供了这些函数来方便的读写基本类型；它们通常是被Kaldi类的Read和Write函数调用。当然Kaldi的类并没有强制要求一定要使用这些函数来实现读写功能，它完全可以自己造一个轮子用完全不同的方式存储和读取数据。

这些函数里最重要的是[ReadBasicType](http://kaldi-asr.org/doc/group__io__funcs__basic.html#ga1bd1af0ef712b76f2332febcec8d095a)和[WriteBasicType](http://kaldi-asr.org/doc/group__io__funcs__basic.html#ga108506a9aabafd9006926aa1b47617da)；它是一个模板函数，参数可以是bool、float、double和整数类型。下面是在Read和Write函数里使用它们的例子：
```
// 我们假设class_member_的类型是int32
void SomeKaldiClass::Read(std::istream &is, bool binary) {
  ReadBasicType(is, binary, &class_member_);
}
void SomeKaldiClass::Write(std::ostream &os, bool binary) const {
  WriteBasicType(os, binary, class_member_);
}
```
我们上面假设class_member_是int32，这是一种定长的数据类型。使用int是不安全的(因为在不同的平台它的长度是不同的)。在二进制模式下，它会用一个char来编码整数类型的长度和是否有符号，如果不匹配读取会识别。下面是WriteBasicType的代码，我们可以看"if (binary)"分支：

```
// Template that covers integers.
template<class T>  void WriteBasicType(std::ostream &os,
                                       bool binary, T t) {
  // Compile time assertion that this is not called with a wrong type.
  KALDI_ASSERT_IS_INTEGER_TYPE(T);
  if (binary) {
    char len_c = (std::numeric_limits<T>::is_signed ? 1 :  -1)
        * static_cast<char>(sizeof(t));
    os.put(len_c);
    os.write(reinterpret_cast<const char *>(&t), sizeof(t));
  } else {
    if (sizeof(t) == 1)
      os << static_cast<int16>(t) << " ";
    else
      os << t << " ";
  }
  if (os.fail()) {
    KALDI_ERR << "Write failure in WriteBasicType.";
  }
}
```

我们也可以在实现的时候尝试自动的进行类型转换，但是我们并没有这么做；目前在I/O时你只能使用长度固定整数类型(通常推荐使用int32)。而浮点数是自动进行转换的。这样便于调试，因此如果你使用-DKALDI_DOUBLE_PRECISION编译，你仍然可以读取单精度的二进制格式的文件。我们的I/O程序不出来byte swapping(Big Indian和Little Indian)；如果有问题(比如需要用其它程序读取Kaldi的输出)，你可以使用文本格式的。

此外也有[WriteIntegerVector](http://kaldi-asr.org/doc/group__io__funcs__basic.html#ga99e4557f845a9c1e3202ab136226a980)和[ReadIntegerVector](http://kaldi-asr.org/doc/group__io__funcs__basic.html#ga0e0391048fe585c1e97b4d65d59d32bc)模板函数。它和前面的WriteBasicType/ReadBasicType函数类似，但是它们的参数是std::vector<I>，这里I是某种整数类型(和前面一样，它要求是定长的，比如int32)。

另外重要的底层I/O函数包括：
```
void ReadToken(std::istream &is, bool binary, std::string *token);
void WriteToken(std::ostream &os, bool binary, const std::string & token);
```

Token是一个非空的不包含空格的字符串，通常像XML的字符串(前后加尖括号)，比如"<SomeKaldiClass>"、"<SomeClassMemberName>"或者"</SomeKaldiClass>"。为了方便，我们也提供函数ExpectToken()，它很像ReadToken()，只不过如果它读取的和期望的不同就会抛出异常，典型的用法如下：
```
// 写文件的代码
WriteToken(os, binary, "<MyClassName>");
// 读取的代码
ExpectToken(is, binary, "<MyClassName>");
// 或者一个类有多种形式： 
std::string token;
ReadToken(is, binary, &token);
if(token == "<OptionA>") { ... }
else if(token == "<OptionB>") { ... }
...
```
 
此外还有WritePretty和ExpectPretty函数。它们很少被使用，它们的作用和WriteToken与ExpectToken类似，但是它们只能用于文本模式，并且它们可以处理任何字符串(包括空格)。Kaldi的Read函数不会检查文件的结束，但是应该会读取到Write函数写出的结束位置(在文本模式，没有读取一些空格不会有什么问题)。这样多个Kaldi对象可以放到同一个文件，并且可以支持archive(参考[The Kaldi archive format](#io_sec_archive))。

## Kaldi对象是如何保存到文件里的
 
在上面我们发现，Kaldi的的读取代码需要知道它读的文件是文本格式还是二进制格式，但是我们又不想让用户自己记住一个文件到底是哪种格式。因此，包含Kaldi对象的文件需要能够区分这两种格式。二进制的Kaldi文件会以"\0B"开头；因为正常的文本文件不会包含"\0"。如果我们使用标准的C++来读取(当然通常不需要自己实现读写Kaldi对象的代码)，那么你需要根据这个规则来判断。你可以使用函数[InitKaldiOutputStream](http://kaldi-asr.org/doc/namespacekaldi.html#a527dbcafdb9efac657f18943cdf1a217)和[InitKaldiInputStream](http://kaldi-asr.org/doc/namespacekaldi.html#a73789daaacd32961040ff917e9c5bc59)来帮你读取或者写入这个特殊的字符。


## Kaldi是怎么打开文件的

假设你像加载或者保存Kaldi对象，比如它是一个语言模型(但不是很多个的那种，比如语音特征；参考[The Table concept](http://kaldi-asr.org/doc/io.html#io_sec_tables))。你通常会使用[Input](http://kaldi-asr.org/doc/classkaldi_1_1Input.html)和[Output](http://kaldi-asr.org/doc/classkaldi_1_1Output.html)，下面是一个例子：

```
{ // 输入
  bool binary_in;
  // Kaldi根据开头是否"\0B"来判断是二进制还是文本格式，同时准备好一个流
  Input ki(some_rxfilename, &binary_in);
  // 从流里读取
  my_object.Read(ki.Stream(), binary_in);
  // 一个文件可能包含多个对象 
  my_other_object.Read(ki.Stream(), binary_in);
}
// 输出。注意，"binary"可能是命令行的选项
{
  Output ko(some_wxfilename, binary);
  my_object.Write(ko.Stream(), binary);
}
```


上面代码的括号的目的是让Input和Output对象尽快被回收，从而文件可以尽快关闭。这看起来有一点奇怪(为什么不适用标准的C++流？)。这么做的目的是为了支持各种扩展类型(extended type)的文件名。同时它让错误处理也稍微简单一点(Input和Output类会对于有信息量的错误并且抛出异常)。注意文件名是rxfilename和wxfilename。我们在很多地方都使用这个名字，用于提醒编码者它们是扩展类型的文件名，下一节我们会介绍扩展文件名。


Input和Output类比上面演示的功能还要稍微多一点。你可以使用Open函数打开它们，也可以使用Close()在对象被回收前关闭它们。这些函数通过返回布尔类型的状态码来表示调用是否成功，这和使用构造、析构函数不同——它们是通过抛出异常来表示识别(构造函数没法返回值)。Open函数也可以调用的时候不出来Kaldi的二进制头("\0B")。当然通常我们不需要这个功能。


参考[Classes for opening streams](http://kaldi-asr.org/doc/group__io__group.html)来获得更多与Input和Output相关的类与函数。

## 扩展文件名

"rxfilename"和"wxfilename"不是类；它们通常是可变的文件名的描述，比如：
* rxfilename是一个字符串，Input类在读入的时候会把它当成特殊的扩展文件名。
* wxfilename是一个字符串，Output类在写出的时候会把它当成特殊的扩展文件名。

下面是rxfilename的所有类型：

* "-"或者""，表示标准输入
* "some command \|"，表示一个命令的输出，它在处理的时候会把"\|"去掉然后作为popen的参数
* "/some/filename:12345"，表示打开某个文件，然后seek到12345
* "/some/filename"，如果不是上面的情况，就认为是一个普通的文件(但是在打开之前会做一些基本检查从而发现一些很基本的错误)

你可以使用函数ClassifyRxfilename来区分一个rxfilename是上面4种的哪一种，当然这通常没有必要。

下面是wxfilename的类型：


* "-"或者""，表示标准输出
* "\| some command"，表示一个命令的输出流
* "/some/filename"，如果不是上面的情况，就认为就是一个普通文件

类似的我们可以使用ClassifyWxfilename来判断一个wxfilename是上面的哪种。


## Table的概念

Table是一个概念而不是一个实际的C++类。它可以(近似)看成一个Python的dict(Java的Map或者C++的std::map\<string, T>)，key是字符串，而value是某种类型的一个对象。key的字符串必须是Token(没有空格的非空字符串)。常见的Table的例子为：

* 一个特征文件的集合(特征表示为Matrix\<float>)，key是utterance id
* 一个录音文本(transcription)的集合(录音文本表示为std::vector\<int32>)，key是utterance id
* 一个Constrained MLLR变换(变换矩阵表示为Matrix\<float>)，key是说话人id

我们会在[Types of data that we write as tables](http://kaldi-asr.org/doc/table_examples.html)里详细介绍上面这些Table类型；这里我们只是解释通用的原则和内部实现机制。Table可以以两种可能的格式保存在磁盘上(也可以在pipe里)：script文件或者archive文件，下面会详细介绍这两种文件。关于Table相关的类和类型，请参考[Table types and related functions](http://kaldi-asr.org/doc/group__table__group.html)。


Table可以用三种方法访问：[TableWriter](http://kaldi-asr.org/doc/classkaldi_1_1TableWriter.html)、[SequentialTableReader](http://kaldi-asr.org/doc/classkaldi_1_1SequentialTableReader.html)和[RandomAccessTableReader](http://kaldi-asr.org/doc/classkaldi_1_1RandomAccessTableReader.html)(也有一个[RandomAccessTableReaderMapped](http://kaldi-asr.org/doc/classkaldi_1_1RandomAccessTableReaderMapped.html))。它们都是模板类；它们的模板不是Value的对象，而是一个Holder类型(参考下面的[](http://kaldi-asr.org/doc/io.html#io_sec_holders)这一节)，这个Holder类型告诉Table代码怎么读取和写入对应的对象。为了打开某个Table对象，你需要提供一个wspecifiers和rspecifiers告诉Table是怎么存储的。下面是一些示例代码。这段代码读取特征，对它做线性变换，然后再写入到磁盘。

```
std::string feature_rspecifier = "scp:/tmp/my_orig_features.scp",
   transform_rspecifier = "ark:/tmp/transforms.ark",
   feature_wspecifier = "ark,t:/tmp/new_features.ark";
// 下面的类型其实有更加简洁的typedef，但是为了便于理解，我们这里没有使用它们
// 实际可以使用BaseFloatMatrixWriter, SequentialBaseFloatMatrixReader等等
// 这样代码更加简短
TableWriter<BaseFloatMatrixHolder> feature_writer(feature_wspecifier);
SequentialTableReader<BaseFloatMatrixHolder> feature_reader(feature_rspecifier);
RandomAccessTableReader<BaseFloatMatrixHolder> transform_reader(transform_rspecifier);
for(; !feature_reader.Done(); feature_reader.Next()) {
   std::string utt = feature_reader.Key();
   if(transform_reader.HasKey(utt)) {
      Matrix<BaseFloat> new_feats(feature_reader.Value());
      ApplyFmllrTransform(new_feats, transform_reader.Value(utt));
      feature_writer.Write(utt, new_feats);
   }
}
```

上面的代码使用了SequentialTableReader来顺序的读取特征文件，核心的代码是循环遍历所有的特征：
```
for(; !feature_reader.Done(); feature_reader.Next()) {
    std::string utt = feature_reader.Key();
    if(...){
        Matrix<BaseFloat> new_feats(feature_reader.Value());
```

而使用RandomAccessTableReader来根据utterance id随机的找到对应的变换，然后对特征进行变换。最后使用TableWriter把变换后的结果写到磁盘上。



使用这种方式的好处是我们可以把Table当成一种通用的map或者list。数据的格式和读取的方式(比如容错)可以通过rspecifiers和wspecifiers里的选项传入而被SequentialTableReader等统一处理；比如上面的例子中，选项",t"说明TableWriter的输出是文本格式的。

Table可能会被当成一个map。但是如果我们不做随机访问，那么即使存在重复的key，代码也不会报错。

对于常见的Table类型的typedef，请参考[Specific Table types](http://kaldi-asr.org/doc/group__table__types.html)。


## Kaldi script文件

 
一个script文件(这个名字有些误导，如果直译的话就是脚本文件这很容易连续到shell或者perl脚本，所以这里不翻译)是一个文本文件，它的每一行包含的内容类似如下：
```
some_string_identifier /some/filename
```

另外也可以把一个命令的输出当成一个文件(Linux的概念里文件只是一个流)：
```
utt_id_01002 gunzip -c /usr/data/file_010001.wav.gz |
```
注意上面最后的管道符号"\|"。

它的通用形式是：
```
<key> <rxfilename>
```

### script文件的范围(比如取某个矩阵的一部分)

We also allow an optional 'range-specifier' to appear after the rxfilename; this is useful for representing parts of matrices, such as row ranges. Ranges are currently not supported for any data types other than matrices. For example, we can express a row range of a matrix as follows:

我们在rxfilename之后也可以有可选"范围描述"；这在表示矩阵的一部分是很有用。目前只有矩阵类型才能使用范围。比如，我们可以取一个矩阵的部分行：
```
  utt_id_01002 foo.ark:89142[0:51]
```
它的意思是从foo.ark文件的89142位置读取矩阵的第0行-第50行(51是不包含在内的)。除了行的范围之外，我们也可以取列的范围，比如：
```
  utt_id_01002 foo.ark:89142[0:51,89:100]
```

如果你只想取列的范围，那么行可以省略，只留下一个都会，这样Kaldi会自动的选取所有行，比如：
```
  utt_id_01002 foo.ark:89142[,89:100]
```

### Kaldi是怎么处理script文件的每一行


当读取一个script文件的一行时，Kaldi首先会trim掉开头和结尾的whitespace，然后用第一个whitespace把它切分成两部分(因此第二个及其以后的whitespace都会作为第二部分的内容)。第一部分是Table的key(比如utterance id "utt_id_01001")，第二部分(处理完可选的范围描述之后)就是xfilename(我们把wxfilename和rxfilename统称为xfilename，比如"gunzip -c /usr/data/file_010001.wav.gz \|")。空行或者空的xfilename是不允许的。一个script文件可能可以用于读取或者写入或者同时读写，这依赖与xfilename是合法的rxfilename或者wxfilename或者同时是rxfilename与wxfilename。



注意：可选的范围描述被去掉之后，script文件每一行xfilename剩下的部分可以一般可以当作一个文件名传给任何Kaldi程序。即使xfilename包含offset信息，比如foo.ark:8432。Kaldi的程序会处理特殊的"\0B"。

## Kaldi的archive文件

The Kaldi archive format is quite simple. First recall that a token is defined as a whitespace-free string. The archive format could be described as:

Kaldi的archive文件非常简单。回忆一下token是不包含空格的字符串。archive文件格式为如下：
```
     token1 [something]token2 [something]token3 [something] ....
```

它是如下内容的重复：一个token；然后一个空格；然后是Holder调用Write函数的输出。Holder是用于告诉Table代码怎么读写对象的。

当写一个Kaldi对象的时候，Holder写出的[something]会包含二进制的头(如果是二进制格式)，接着是Write函数的输出。当写非Kaldi对象的时候(比如int32或者vector\<int32>)，Holder类在文本格式输出是会保证[something]是以换行结尾。这样得到的archive文件就会每行都是一个key加一个对象，类似于：
```
    utt_id_1 5
    utt_id_2 7
    ...
```

上面的内容是用文本格式存储整数的例子。



archive格式可以保证把两个archive文件拼接(concatenate)起来仍然是一个合法的archive文件(当然要假设它们存储的值是同一种类型)。这种格式是很时候pipe的，也就是你可以把一个archive文件读取到pipe里，而从pipe里读取的程序能够不用到pipe的尾部才开始处理(也就是说按行就可以处理)。为了提高随机访问archive文件的效率，我们可以同时写出一个archive文件以及与之对应的script文件。下面会详细介绍。

## 指定Table的格式：wspecifier和rspecifier

Table类需要在构造函数或者Open的时候传入一个字符串。对于TableWriter这个字符串叫做wspecifier，而对于RandomAccessTableReader或者SequentialTableReader它叫做rspecifier。下面是rspecifier和wspecifier的例子：
```
std::string rspecifier1 = "scp:data/train.scp"; // script文件
std::string rspecifier2 = "ark:-"; // 从stdin读取的archive文件
// 以文本格式输出archive文件然后再用gzip压缩 
std::string wspecifier1 = "ark,t:| gzip -c > /some/dir/foo.ark.gz";
// 同时包含archive和script文件，其中script是这个archive文件的"索引"
std::string wspecifier2 = "ark,scp:data/my.ark,data/my.scp";
```

通常，rspecifier和wspecifier包含一个或多个字符串的list(无顺序)，如果有多个则用都会分开。此外在最前面会有ark/scp开头的字符串，用冒号分开它和list。

### 同时写archive和script文件

wspecifier有一种特殊的写法：冒号前是"ark,scp"，而冒号后是一个archive的xwfilename加一个逗号再加一个script文件的wxfilename。比如：
```
  "ark,scp:/some/dir/foo.ark,/some/dir/foo.scp"
```

这会输出一个archive文件，同时还会输出一个script文件，这个script文件的行类似于"utt_id /somedir/foo.ark:1234"，我们可以把它当作索引，可以利用它快速定位某个utt_id在ark文件中的offset。和普通的script文件一样，你可以把它切分成Segment。虽然一般情况下wspecifier里的冒号前面的ark/scp可以交换顺序，但是在这里却必须要保持"ark,scp"的顺序。

### wspecifier的选项

下面是wspecifier可以使用的选项：

* "b" (binary) 表示以二进制格式输出(目前不需要，因为默认就是二进制)
* "t" (text) 文本格式
* "f" (flush) 每写一个对象都flush一下 
* "nf" (no-flush) 没写一次不调用flush(可能没意义，因为代码可以自己会flush)
* "p" permissive模式，如果script文件确实某些项，它不会报错，只是默默的忽略


下面是使用这些选项的例子：
```
       "ark,t,f:data/my.ark"
       "ark,scp,t,f:data/my.ark,|gzip -c > data/my.scp.gz"
```

第一行表示输出archive文件，文本格式，每写一次都flush。第二个同时输出ark和scp文件，也是文本格式，每写一次都flush，并且scp文件会用gzip压缩。


### rspecifier的选项

当阅读下面的选项时，请记住如果archive是一个pipe(通常都是)是，读取archive文件的代码不会seek。如果RandomAccessTableReader在读取一个archive，如果它需要回过头来重复读取，则需要在内存中缓存。

重要的rspecifier选项包括：

* "o" (once)，用户告诉RandomAccessTableReader某个对象只会读取一次，因此不需要在内存中保存已经读取的对象。
* "p" (permissive) 开启这个选项时Kaldi代码在遇到数据错误时会忽略。对于scp文件，HasKey()函数会强制加载整个文件，如果文件有问题则会返回false。对于ark文件，如果文件坏了则不会抛出异常。
* "s" (sorted) 说明读取的ark文件是(按照key)排过序的。因此对于RandomAccessTableReader，HasKey()可以在遇到它大的key时提前结束，而不用一种读完整个文件。 
* "cs" (called-sorted) 告诉Kaldi对于HasKey()和Value()的调用是按照key的顺序的。因此如果读取了某个key之后，比它小的key就可以丢弃(假设缓存了的话)。这可以节约内存。

如果用户错误的提供了选项的话，比如提供了"s"选项但是事实上文件并没有按照key排序，则RandomAccessTableReader会尽量检测错误然后crash。

下面的选项是为了对称和方便，但是它们目前并没有太大作用。

* "no" (not-once) "o"的相反(在目前的代码里不起任何作用)
* "np" (not-permissive) "p"的相反(在目前的代码里不起任何作用)
* "ns" (not-sorted) "s"的相反(在目前的代码里不起任何作用)
* "ncs" (not-called-sorted)  "cs"的相反(在目前的代码里不起任何作用) 
* "b" (binary) 不起作用 
* "t" (text) 不起作用

下面是rspecifier的一些典型例子：
```
     "ark:o,s,cs:-"
     "scp,p:data/my.scp"
```

## Table的辅助类Holder


前面提到过，Table类包括TableWriter、RandomAccessTableReader和SequentialTableReader都是基于Holder类的模板类。Holder不是一个具体的类而是一个抽象基类，它有很多子类比如TokenHolder和KaldiObjectHolder，这些类从名字大多可以猜测其用途。KaldiObjectHolder是一个通用的Holder，任何满足前面介绍的Kaldi I/O风格的对象都可以使用这个Holder。我们也实现了一个GenericHolder，它不应该被使用，它的目的是提供文档。


被Holder类"held"的类类型是Holder::T的typedef。用到的类型请参考[这个文档](http://kaldi-asr.org/doc/group__holders.html)。


## 文本/二进制格式与文件打开模式的关系

本节只与Windows平台有关。基本的规则是当写的时候，文件的打开模式与Write函数的binary参数一致；当读取二进制数据是，文件的打开方式总是二进制，但是读取文本格式的时候，文件的打开方式可能是文本也可能是二进制。

## 随机读取archive文件时避免内存消耗过大


当使用随机的方式读取大的archive文件是，很容易消耗大量内存。这通常发生在RandomAccessTableReader\<SomeHolder>读取某个archive的时候。Table代码首先要保证的是正确性，因此当在随机模式读取archive文件时，除非提供额外的信息(后面会讨论)，它总是把读取过的对象都保存到内存以便下次使用。读者可能会问：为什么不能记住每个key对于的对象在文件的offset，然后需要的时候fseek过去？我们没有这么实现，原因如下：只有读取的是实际的文件时才能fseek(因此pipe的输出或者标准输入是不行的)。如果archive文件在磁盘上，你可以同时输出ark和scp文件，然后使用scp文件快速定位。这和直接读取archive一样高效，因为读取scp文件的代码会避免重新打开文件和不必要的fseek。因此把基于文件的archive当成特例然后cache它的位置并不能解决任何问题。

当你使用随机访问模式读取一个archive时不知道任何附加的选项，则会有两个问题：
* 如果你寻找一个archive里不存在的key，则读取的代码需要一直读到文件结尾才能判断它确实不存在。
* 美读取一个对象，都会保存在内存里一般下次访问。

对于第一个问题，我们可以对archive按照key排序来解决(字符串的排序必须安装"C"的方式，因此你需要"export LC_ALL=C")。如果是排过序的archive文件，你可以在rspecifier里增加"s"选项：比如，"ark,s:-"告诉代码把标准输入当成一个archive，并且它是按照key排好序的。Table代码会检查输入是否排过序，如果发现逆序则会crash。

对于第二个问题，有两种解决方法：

* 第一种方法是非常脆弱的方法，也就是提供"once"选项；比如：rspecifier "ark,o:-"从标准输入读取archive，然后假定你每个对象只会读取一次。为了保证这一点，你必须知道某个程序(或者这个程序就是你写的)确实不会重复访问某个key(对于顺序访问模式下的Table是可以有重复的key的)。

如果提供了"o"的选项，那么对象被访问过后都会被释放。但是这只有在你的archive文件是一致的(含义参考下面)并且没有缺失的元素才能工作，比如下面的命令：

```
     some-program ark:somedir/some.ark "ark,o:some command|"
```

假设"some-program"会顺序的遍历ark:somedir/some.ark，对于遇到的每一个key，在第二个archive里通过key随机的寻找。注意：上面的命令行参数不是随便排序的：我们通常把顺序访问的rspecifier放在前面，而把随机访问的防止后面。


假设这两个archive文件是一致的(synchronized)但是有一些缺失(比如某些key值出现在一个archive里)。比如第一个文件的key是"3 1 5"，第二个是"5 2 3 1 4"，则在读取3的时候，我们必须cache第二个archive文件的key "2"和"5"。因为我们不知道第一个文件会不会读取它们，只有第一个文件读取过的我们可以回收内存。如果第二个archive有缺失，那么问题更加严重，我们必须要读到文件介绍后才能确保这个key不存在。

* 第二种解决办法更加鲁棒，它要使用"called-sorted" (cs)选项。它的意思是调用者确保访问key是按照顺序进行的。"cs"选项和"s"一起使用会更加有用。比如下面的命令：

```
     some-program ark:somedir/some.ark "ark,s,cs:some command|"
```

我们假设两个archive都是排过序的，并且程序顺序的访问第一个archive然后随机访问第二个archive。那么它对于key的缺失是鲁棒的。比如第一个archive的key是有缺失的(001, 002, 003, 081, 082, ...)。那么在搜索"003"之后再搜索"081"的时候，它可以不缓存中间读取的"004", "005", ……。如果第二个archive有缺失，它也不需要读取到文件结尾，而只需要找到比它大的key就可以判断它不存在了(因为第二个archive说明了它是排序过的"s")。


## RandomAccessTableReaderMapped

为了精简代码，我们引入了RandomAccessTableReaderMapped这个模板类来抽象一种经常出现的代码模式。和RandomAccessTableReader不同，它的构造函数有两个参数：
```
   std::string rspecifier, utt2spk_map_rspecifier; // 从某个地方获得的参数
   RandomAccessTableReaderMapped<BaseFloatMatrixHolder> transform_reader(rspecifier,
                                                                      utt2spk_map_rspecifier);
```

如果utt2spk_map_rspecifier是空字符串，则它和普通的RandomAccessTableReader一样。如果非空，比如是"ark:data/train/utt2spk"，它会把这个文件读取到一个utternace-speaker的map里。当查询某个key，比如utt1的时候，它会用这个map把utt1变成对于的说话人id，(比如spk1)然后使用这个key去rspecifier查找。这个map在实现的时候也是一个archive，这样的实现在Table代码里更加方便。











