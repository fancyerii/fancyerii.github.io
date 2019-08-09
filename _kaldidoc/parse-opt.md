---
layout:     post
title:      "Parsing命令行选项"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Parsing command-line options](http://kaldi-asr.org/doc/parse_options.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

 
## Introduction



类[ParseOptions](http://kaldi-asr.org/doc/classkaldi_1_1ParseOptions.html)用于从来自main()函数的argc和argv的参数里提取命令行的选项。我们首先给出一个典型的Kaldi命令行程序使用它的例子：
```
   gmm-align --transition-scale=10.0 --beam=75 \
       exp/mono/tree exp/mono/30.mdl data/L.fst \
       'ark:add-deltas --print-args=false scp:data/train.scp ark:- |' \
        ark:data/train.tra ark:exp/tri/0.ali
```



命令行选项只有长的形式(\-\-beam这样的，没有常见Linux命令的单个字符的短形式(比如ls -l))，并且它必须出现在所有位置参数的前面。在这个例子里有6个位置参数，"exp/mono/tree"是第一个位置参数。注意："ark:add-deltas"开头的参数是一个包含空格的字符串。


## parsing命令行选项的示例
 
我们通过[gmm-align.cc](http://kaldi-asr.org/doc/gmm-align_8cc.html)的代码(为了简洁做了一些修改)来展示在C++代码基本怎么处理命令行选项：

```
int main(int argc, char *argv[])
{
  try { // try-catch block is standard and relates to handling of errors.
    using namespace kaldi;
    const char *usage =
        "Align features given [GMM-based] models.\n" 
        "Usage: align-gmm [options] tree-in model-in lexicon-fst-in feature-rspecifier "
        "transcriptions-rspecifier alignments-wspecifier\n";
    // Initialize the ParseOptions object with the usage string.
    ParseOptions po(usage);
    // Declare options and set default values.
    bool binary = false;
    BaseFloat beam = 200.0;
    // Below is a structure containing options; its initializer sets defaults.
    TrainingGraphCompilerOptions gopts;
    // Register the options with the ParseOptions object.
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("beam", &beam, "Decoding beam");
    gopts.Register(&po);
    // The command-line options get parsed here.
    po.Read(argc, argv);
    // Check that there are a valid number of positional arguments.
    if(po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }
    // The positional arguments get read here (they can only be obtained
    // from ParseOptions as strings).
    std::string tree_in_filename = po.GetArg(1);
    ...
    std::string alignment_wspecifier = po.GetArg(6);
    ...   
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
```

The code above is mostly self-explanatory. In a normal Kaldi program, the sequence is as follows:
上面的代码大部分是自解释的(self-explanatory)。在一个典型的Kaldi程序里，处理流程为：

* 构造一个ParseOptions对象，参数为介绍其用法的字符串。
* 定义和设置选项(可选参数)的默认值(以及定义option的结构体)
* 对ParseOptions对象注册命令行选项(option的结构体有它自己注册的函数)
* 调用"po.Read(argc, argv);"
* 使用"po.NumArgs()"检查位置参数的个数是否正确
* 使用"po.GetArg(1)"获取位置参数，而选项的值直接保存在对应的变量里。

比如上面的代码里binary是一个boolean类型的选项，可以通过"po.Register("binary", &binary, "Write output in binary mode");"来注册它，这样命令行的用户可以使用"\-\-binary=true"来设置。而gopts是一个结构体(TrainingGraphCompilerOptions)，它通过"gopts.Register(&po);"来自己处理命令行选项：

```
struct TrainingGraphCompilerOptions {

  BaseFloat transition_scale;
  BaseFloat self_loop_scale;
  bool rm_eps;
  bool reorder;  // (Dan-style graphs)

  explicit TrainingGraphCompilerOptions(BaseFloat transition_scale = 1.0,
                                        BaseFloat self_loop_scale = 1.0,
                                        bool b = true) :
      transition_scale(transition_scale),
      self_loop_scale(self_loop_scale),
      rm_eps(false),
      reorder(b) { }

  void Register(OptionsItf *opts) {
    opts->Register("transition-scale", &transition_scale, "Scale of transition "
                   "probabilities (excluding self-loops)");
    opts->Register("self-loop-scale", &self_loop_scale, "Scale of self-loop vs. "
                   "non-self-loop probability mass ");
    opts->Register("reorder", &reorder, "Reorder transition ids for greater decoding efficiency.");
    opts->Register("rm-eps", &rm_eps,  "Remove [most] epsilons before minimization (only applicable "
                   "if disambig symbols present)");
  }
};
```

通常如果要编写一个新的Kaldi命令行程序，可以复制一个然后照着修改。

## 隐式的命令选项

Certain command-line options are automatically registered by the ParseOptions object itself. These include the following:

ParseOptions对象会自动的注册一些命令行选项，因此所有的Kaldi命令行程序都可以使用这些选项。这包括：

* \-\-config 这个选项从一个配置文件加载命令行选项。比如我们指定\-\-config=configs/my.conf，文件my.conf可能包含：

```
          --first-option=15  # This is the first option
          --second-option=false # This is the second option
```
* \-\0print-args 这是一个boolean类型的选项，控制程序是否打印命令行参数到标准错误输出(默认是true)，可以用\-\-print-args=false关掉这个行为

* \-\-help 打印ParseOptions对象的用法字符串。不过你通常可以使用没有任何参数的命令，一般的命令都需要参数，它发现参数不对时也会打印用法字符串。
* \-\-verbose 控制verbose级别，这样决定KALDI_VLOG是否输出。值越大级别越高(\-\-verbose=2是比较典型的用法)



