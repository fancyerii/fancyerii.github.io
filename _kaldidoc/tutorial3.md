---
layout:     post
title:      "Kaldi教程(三)"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Kaldi tutorial](http://kaldi-asr.org/doc/tutorial.html)。
 <!--more-->
 
**目录**
* TOC
{:toc}

## 阅读和修改代码

在triphone模型正在训练的过程中，我们来看看部分源代码。

在这一部分教程里，读者将会学习Kaldi的代码是怎么组织的以及依赖结构是怎么样的，也会学习怎么修改和调试代码。如果读者想更加深入的理解代码，可以参考[main documentation page](http://kaldi-asr.org/doc/index.html)，那里会根据主题有更加详细的介绍(这些我们都会翻译)。


### 常用工具


进入src目录，看一下base/kaldi-common.h。除了对于标准库的依赖，其它的那些依赖几乎是所有Kaldi代码都需要的。

```
#include "base/kaldi-utils.h"
#include "base/kaldi-error.h"
#include "base/kaldi-types.h"
#include "base/io-funcs.h"
#include "base/kaldi-math.h"
```

从名字我们也大致可以猜测出它们的用途。比如kaldi-utils.h是一些常用的工具类，比如判断机器是Little Endian还是Big Endian。kaldi-error.h定义了error logging的一些宏。kaldi-types.h定义了跨平台的基本类型，比如int32等。io-funcs.h定义一些保存基本数据类型的IO函数。而kaldi-math.h则是一些数学工具。


但这里面的都是简化过的工具，util/common-utils.h是一个更完整的工具，它包括命令行的解析工具，能够处理扩展文件名的I/O函数。下面是common-utils.h包含的头文件：

```
#include "base/kaldi-common.h"
#include "util/parse-options.h"
#include "util/kaldi-io.h"
#include "util/simple-io-funcs.h"
#include "util/kaldi-holder.h"
#include "util/kaldi-table.h"
#include "util/table-types.h"
#include "util/text-utils.h"
```

它首先包含了kaldi-common.h，然后增加了其它一些常见的util下的头文件。

这么做的目的是为了减少matrix库对它们的依赖，这样matrix库可以更加方便的被第三方工具使用(而不用引入过多的依赖)。我们看一下matrix/Makefile：
```
LIBNAME = kaldi-matrix

ADDLIBS = ../base/kaldi-base.a 

```
我们发现它值依赖base/kaldi-base.a，也就是只依赖base/这个目录。

### Matrix库(修改和调试代码)

现在看一下matrix/matrix-lib.h。看看它包括了哪些头文件。
```
#include "base/kaldi-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/sp-matrix.h"
#include "matrix/tp-matrix.h"
#include "matrix/matrix-functions.h"
#include "matrix/srfft.h"
#include "matrix/compressed-matrix.h"
#include "matrix/sparse-matrix.h"
#include "matrix/optimization.h"
```

这个头文件是matrix对外的所有接口，如果包含这个头文件，那么所有matrix的函数声明都包括了。当然我们也可以只使用其中的某一个，比如\#include "matrix/kaldi-matrix.h"。这个库是对BLAS和LAPACK的C++封装，读者如果每听说过这些东西也不要紧。sp-matrix.h和tp-matrix.h是关于对称(symmetric)矩阵和上/下三角(triangular)矩阵更加紧凑的表示。快速的浏览一下matrix/kaldi-matrix.h，通过它你会了解矩阵代码长什么样子。如果你对矩阵的细节感兴趣，可以参考[The Kaldi Matrix library ](http://kaldi-asr.org/doc/matrix.html)。

这里我们会看到"///"或者"/** **/"这样的注释，这是给Doxygen用的，这样它可以生成格式化(HTML)的文档。

现在我们可以阅读和修改代码。我们会在matrix/matrix-lib-test.cc里增加一个单元测试，这个文件是用来对matrix库做单元测试的。前面提到过，如果单元测试失败(Assert失败)，则程序会异常退出。

我们会编写一个函数来测试Vector::AddVec。这个函数先用一个常数乘以一个向量，然后把计算结果加到另外一个向量里。请仔细阅读下面的代码，尽量理解。为了演示调试，我们故意在代码里加入了一些错误。如果你不熟悉template，那么阅读可能会有一些困难(也没那么困难，我们一般不要编写和调试新的代码，只是阅读和使用而已)。请读者把下面的代码加到matrix/matrix-lib-test.cc

```
template<typename Real>
void UnitTestAddVec() {
  // note: Real will be float or double when instantiated.
  int32 dim = 1 + Rand() % 10;
  Vector<Real> v(dim); w(dim); // two vectors the same size.
  v.SetRandn();
  w.SetRandn();
  Vector<Real> w2(w); // w2 is a copy of w.
  Real f = RandGauss();
  w.AddVec(f, v); // w <-- w + f v
  for (int32 i = 0; i < dim; i++) {
    Real a = w(i), b = f * w2(i) + v(i);
    AssertEqual(a, b); // will crash if not equal to within
    // a tolerance.
  }
}
```
注：在template里现在大家似乎更倾向于使用typename而不是class，我们会发现Kaldi的代码现在都改成template\<typename xxx>了，不过使用template\<class Real>也没有什么问题，这只是习惯而已，读者可以参考[这个so问题](https://stackoverflow.com/questions/213121/use-class-or-typename-for-template-parameters)。


把上面的代码加到函数MatrixUnitTest()上面，然后在这个函数里增加测试：
```
UnitTestAddVec<Real>();
```

我们是"make test"来编译和测试，这会产生如下的编译错误：
```
lili@lili-Precision-7720:~/codes/kaldi/src/matrix$ make test
g++ -std=c++11 -I.. -I/home/lili/codes/kaldi/tools/openfst/include  -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I/home/lili/codes/kaldi/tools/ATLAS_headers/include -msse -msse2 -pthread -g  -fPIC -DHAVE_CUDA -I/usr/local/cuda/include   -c -o matrix-lib-test.o matrix-lib-test.cc
matrix-lib-test.cc: In function ‘void kaldi::UnitTestAddVec()’:
matrix-lib-test.cc:38:29: error: there are no arguments to ‘w’ that depend on a template parameter, so a declaration of ‘w’ must be available [-fpermissive]
   Vector<Real> v(dim); w(dim); // two vectors the same size.

```

去读者仔细看一下出错的行，看看能不能发现问题。

把错误翻译一下，大意是：w没有哪个参数依赖template参数，因此w必须要先声明。这似乎有些奇怪(template的报错通常都很难排除)，

我们仔细一点会发现定义w时把前面的逗号错误的拼写成了分号。如果普通的这种错误，比如：
```
int x;y;
```
编译器的报错是：
```
error: ‘y’ was not declared in this scope
```
这是很直接的，它以为我们是在使用变量y，但是还没有声明。但是在上面的错误里，它只是推测可能w没有声明，读者可以对比一下。

把分号改成逗号后就可以正确编译了，但是它仍然会在运行matrix-lib-test时出错：
```
Running matrix-lib-test .../bin/bash: 行 1:  6221 已放弃               (核心已转储) ./$x > $x.testlog 2>&1
 0s... FAIL matrix-lib-test
Running kaldi-gpsr-test ... 0s... SUCCESS kaldi-gpsr-test
Running sparse-matrix-test ... 0s... SUCCESS sparse-matrix-test
../makefiles/default_rules.mk:80: recipe for target 'test' failed
make: *** [test] Error 1

```

我们可以自己运行一下获得更多信息：
```
lili@lili-Precision-7720:~/codes/kaldi/src/matrix$ ./matrix-lib-test 
ASSERTION_FAILED ([5.4.232-532f3]:AssertEqual():base/kaldi-math.h:278) : 'ApproxEqual(a, b, relative_tolerance)' 

[ Stack-Trace: ]

kaldi::MessageLogger::HandleMessage(kaldi::LogMessageEnvelope const&, char const*)
kaldi::MessageLogger::~MessageLogger()
kaldi::KaldiAssertFailure_(char const*, char const*, int, char const*)
./matrix-lib-test() [0x427073]
void kaldi::UnitTestAddVec<float>()
./matrix-lib-test() [0x4276ae]
main
__libc_start_main
_start

已放弃 (核心已转储)

```

从stacktrace可以看出是"AssertEqual(a, b);"失败，也就是程序的结果并不是我们预期的。

下面我们来调试程序以找出bug：
```
gdb ./matrix-lib-test
```

输入"r"让它运行，然后就会crash然后调用abort()，这会被gdb捕获。然后我们使用命令"bt"让它打印出stack：
```
(gdb) bt
#0  0x00007ffff6a44428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
#1  0x00007ffff6a4602a in __GI_abort () at abort.c:89
#2  0x00007ffff78a8e66 in kaldi::MessageLogger::HandleMessage (envelope=..., 
    message=0x6a6de0 ": 'ApproxEqual(a, b, relative_tolerance)' ") at kaldi-error.cc:209
#3  0x00007ffff78a8aa5 in kaldi::MessageLogger::~MessageLogger (this=0x7fffffffcb60, __in_chrg=<optimized out>)
    at kaldi-error.cc:157
#4  0x00007ffff78a8fe3 in kaldi::KaldiAssertFailure_ (
    func=0x47fb20 <kaldi::AssertEqual(float, float, float)::__func__> "AssertEqual", 
    file=0x47df9e "../base/kaldi-math.h", line=278, cond_str=0x47df78 "ApproxEqual(a, b, relative_tolerance)")
    at kaldi-error.cc:232
#5  0x0000000000427073 in kaldi::AssertEqual (a=-1.39748704, b=-0.473557711, relative_tolerance=0.00100000005)
    at ../base/kaldi-math.h:278
#6  0x00000000004712da in kaldi::UnitTestAddVec<float> () at matrix-lib-test.cc:46
#7  0x00000000004276ae in kaldi::MatrixUnitTest<float> (full_test=false) at matrix-lib-test.cc:4605
#8  0x00000000004275e1 in main () at matrix-lib-test.cc:4763

```

我们可以发现我们的代码出错的是"\#6 ... kaldi::UnitTestAddVec\<float>"，因此我们通过"up"命令走到stack的这一层(如果走多轮可以输入"down"往下走)，然后使用"p a"和"p b"查看变量的值：
```
(gdb) p a
$1 = -1.39748704
(gdb) p b
$2 = -0.473557711

```

注意：因为是随机的参数，所以读者运行时的结果可能不同。但是我们发现a和b的差别很大，这不是浮点数舍入的误差。所以我们可以判断是代码的bug。

现在的问题是a和b不相等，其中a是AddVec计算的结果，而b是我们用另外一种方法计算的结果，显然这两个中至少有一个是有问题的。那哪个是错误的呢？很简单，我们自己来计算一下，当然首先我们需要知道计算的变量：
```
(gdb) p a
$3 = -1.39748704
(gdb) p b
$4 = -0.473557711
(gdb) p v.data_[0]
$5 = 0.479047596
(gdb) p w.data_[0]
$6 = -1.39748704
(gdb) p w2.data_[0]
$7 = -1.67064154
(gdb) p f
$8 = 0.570203304

```

回顾一下，我们要计算的是$w=w+fv$，其中f是0.57，v是0.479，w原来的值是-1.67，我们简单的计算一下就能发现a是对的而b是错的。因此我们可以判断AddVec没有问题，那么问题出在用于验证的"b = f * w2(i) + v(i);"上，通过分析，我们发现正确的应该是"b = w2(i)+ f * v(i)"。

修改之后再次make test，我们应该可以通过测试(至少是通过这个测试)。为了验证(学习gdb)，我们让gdb停在计算b的这一行：
```
(gdb)gdb ./matrix-lib-test
(gdb)b matrix-lib-test.cc:45

(gdb)r
Starting program: /home/lili/codes/kaldi/src/matrix/matrix-lib-test 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, kaldi::UnitTestAddVec<float> () at matrix-lib-test.cc:45
45	    Real a = w(i), b = w2(i) + f * v(i);

```
注意45是作者的代码行号，读者需要改成自己的。

如果运行程序需要参数，那么可以在gdb后面增加参数：
```
 gdb --args kaldi-program arg1 arg2 ...
 (gdb) r
 ...
```

如果我们发现了Kaldi的bug并且修复了它，那么可以通过Github的Pull Request来提交给Kaldi，这样也能帮到别人，关于怎么提交pr，请参考[GitHub pull request](https://help.github.com/articles/using-pull-requests/)。

使用gdb调试的好处是可以在没有GUI的服务器上调试代码，但是这样调试代码(至少对于我来说)不是很习惯，因为我已经习惯了IDE。有很多IDE可以使用，因为我使用Linux，所以用了vscode，这是微软开发的一个跨平台的工具(其实还不能叫IDE)，在主流平台都可以使用，我这里简单的介绍一下，仅供参考。

首先是安装vscode，然后需要安装[微软提供的C++插件](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)。然后新建一个workspace把kaldi目录加进去。不熟悉的读者可以参考[官方文档](https://code.visualstudio.com/docs)。

为了调试我们需要点击Debug视图，如下图所示：

<a name='vscode1'>![](/img/kaldidoc/vscode1.png)</a> 
*图：vscode的debug视图* 

然后点击"Add Configuration..."，如下图所示：


<a name='vscode2'>![](/img/kaldidoc/vscode2.png)</a> 
*图：Add Configuration* 

这是会出现一个配置"launch.json"，需要修改调试的程序路径，请参考作者的文件进行修改：
```
    "configurations": [
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "/home/lili/codes/kaldi/src/matrix/matrix-lib-test",
            "args": [],
```

保存后回到matrix-lib-test.cc文件在需要调试的地方增加断点，然后点击DEBUG旁边那个绿色的三角形就可以启动并且调试。原理其实还是使用gdb，只不过做了UI，使用起来更加方便。运行到断点后就可以查看变量，如下图所示：

<a name='vscode'>![](/img/kaldidoc/vscode.png)</a> 
*图：vscode调试进入断点* 

另外除了调试，vscode阅读代码也非常方便，强烈建议读者一试。如果读者想用vscode来运行makefile(而不是命令行运行make)，也可以参考[Quick Start to Use Visual Studio Code for C++ Programmers in Linux](https://www.codeproject.com/Articles/1184735/Quick-Start-to-Use-Visual-Studio-Code-for-Cplusplu)，它介绍了怎么在vscode里集成CMake和Makefile的项目。


### 声学模型代码

看一下gmm/diag-gmm.h，这个类存储一个GMM(高斯混合模型)。DiagGmm类看起来有点难懂，因为它有很多set和get函数。我们可以搜索"private:"来查看成员变量，Kaldi的成员变量总是以下划线结尾。
```
 private:
  /// Equals log(weight) - 0.5 * (log det(var) + mean*mean*inv(var))
  Vector<BaseFloat> gconsts_;
  bool valid_gconsts_;   ///< Recompute gconsts_ if false
  Vector<BaseFloat> weights_;        ///< weights (not log).
  Matrix<BaseFloat> inv_vars_;       ///< Inverted (diagonal) variances
  Matrix<BaseFloat> means_invvars_;  ///< Means times inverted variance
```

详细代码这里不介绍。注意这只是一个GMM而不是GMM的集合。看一下gmm/am-diag-gmm.h，这是GMM的集合。
```
 private:
  std::vector<DiagGmm*> densities_;
//  int32 dim_;
```
它只有一个std::vector\<DiagGmm*>，保存了很多GMM。注：最新的版本代码行"int32 dim_"已经注释掉了。

读者可能会问：这里只有GMM，那HMM的跳转概率，HMM的拓扑结构都在哪里？这些内容是和声学模型分开的，研究者通常会把声学模型替换掉(比如用DNN替代GMM)，所以这样的好处是修改或者提供新的声学模型不需要修改其它地方的代码。


### 特征提取代码

看一下feat/feature-mfcc.h，重点看一下MfccOptions这个struct，它是MFCC特征提取的一些选项(参数)。注意它的一些成员也是struct，比如"MelBanksOptions mel_opts;"。

```
struct MfccOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  int32 num_ceps;  // e.g. 13: num cepstral coeffs, counting zero.
  bool use_energy;  // use energy; else C0
  BaseFloat energy_floor;
  bool raw_energy;  // If true, compute energy before preemphasis and windowing
  BaseFloat cepstral_lifter;  // Scaling factor on cepstra for HTK compatibility.
                              // if 0.0, no liftering is done.
  bool htk_compat;  // if true, put energy/C0 last and introduce a factor of
                    // sqrt(2) on C0 to be the same as HTK.
```

再看一下Register函数：

```
  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    mel_opts.Register(opts);
    opts->Register("num-ceps", &num_ceps,
                   "Number of cepstra in MFCC computation (including C0)");
    opts->Register("use-energy", &use_energy,
                   "Use energy (not C0) in MFCC computation");
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in MFCC computation");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
    opts->Register("cepstral-lifter", &cepstral_lifter,
                   "Constant that controls scaling of MFCCs");
    opts->Register("htk-compat", &htk_compat,
                   "If true, put energy or C0 last and use a factor of sqrt(2) on "
                   "C0.  Warning: not sufficient to get HTK compatible features "
                   "(need to change other parameters).");
  }
```

这是Kaldi注册Option的标准方式，这样我们可以统一的命令行解析工具来解析Options。比如上面的代码会注册num-ceps这个选项，它定义MFCC的cepstra的个数，另外mel_opts本身也是个Option，因此调用"mel_opts.Register(opts)"来注册。

然后看一下featbin/compute-mfcc-feats.cc，这个文件会生成compute-mfcc-feats这个命令行工具，它最后会使用feature-mfcc.h。我们来看一下它的Register。
```
    // Register the MFCC option struct
    mfcc_opts.Register(&po);

    // Register the options
    po.Register("output-format", &output_format, "Format of the output "
                "files [kaldi, htk]");
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each "
                "feature file [CMS]; not recommended to do it this way. ");
    po.Register("vtln-warp", &vtln_warp, "Vtln warp factor (only applicable "
                "if vtln-map not specified)");
    po.Register("vtln-map", &vtln_map_rspecifier, "Map from utterance or "
                "speaker-id to vtln warp factor (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier, "Utterance to speaker-id map "
                "rspecifier (if doing VTLN and you have warps per speaker)");
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
                "0 -> left, 1 -> right)");
    po.Register("min-duration", &min_duration, "Minimum duration of segments "
                "to process (in seconds).");

    po.Read(argc, argv);
```
如果想看compute-mfcc-feats(或者任何Kaldi命令行工具)，都可以不带任何参数的执行它。我们发现有一些Option是在MfccOptions里注册的，而另外一些是在compute-mfcc-feats.cc的main函数里注册的。我们可以执行：
```
compute-mfcc-feats ark:/dev/null ark:/dev/null
```
它是可以正常运行的，因为/dev/null是特殊的"空"的输入和输出。我们也可以指定选项的值(而不是用默认值)：
```
compute-mfcc-feats --raw-energy=false ark:/dev/null ark:/dev/null
```

### 声学决策树和HMM拓扑的代码

看一下tree/build-tree.h，找到BuildTree函数。这是构建决策树的最上层函数。
```
EventMap *BuildTree(Questions &qopts,
                    const std::vector<std::vector<int32> > &phone_sets,
                    const std::vector<int32> &phone2num_pdf_classes,
                    const std::vector<bool> &share_roots,
                    const std::vector<bool> &do_split,
                    const BuildTreeStatsType &stats,
                    BaseFloat thresh,
                    int32 max_leaves,
                    BaseFloat cluster_thresh,  // typically == thresh.  If negative, use smallest split.
                    int32 P, 
                    bool round_num_leaves = true);
```

它的返回值的类型是EventMap指针。这是一个Map，Value是int，而Key是又是一个(key,value)对的list(std::vector)，它是在tree/event-map.h里定义的。(key,value)对的类型都是int，其中key表示phone的位置(对于triphone来说是0、1和2)，而value表示这个位置的phone。此外key还有一个特殊的-1，大致可以理解为它是用来说明triphone是HMM的哪个状态(比如常见的3状态HMM有3个状态)。BuildTree函数的主要输入参数是"BuildTreeStatsType &stats"，它的定义是：
```
typedef vector<pair<EventType, Clusterable*> > BuildTreeStatsType;
```
而EventType的定义是：
```
typedef std::vector<std::pair<EventKeyType, EventValueType> > EventType;
```

这里的EventKeyType就是EventMap的Key。我们来看一个EventType的例子：{ {-1, 1}, {0, 15}, {1, 21}, {2, 38} }。这个EventType有4个Pair，第一个Pair的Key是-1，Value是1，因此它表示这个triphone是HMM的第1个状态(更准确的说应该是pdf-class)；第二个Pair的Key是0，Value是15，这说明triphone的第一个位置是15。类似的后面两个Pair表示triphone的第二个位置和第三个位置的phone id是21和38。因此这个triphone是15-21+38，也许就是b-a+t(这是作者随便写的)。

因此我们可以认为EventType就是描述一个triphone的Key，而Clusterable*就是聚类的指针。因此BuildTreeStatsType可以看成一个Map(只不过是用Vector来存储)，我们可以根据triphone找到对应的聚类。类似的，前面我们提到的EventMap，它的Key也是EventType，而Value是EventAnswerType(其实也是int)，它的Key是triphone，value是这个triphone的pdf-id。

Clusterable是一个抽象基类，它的子类需要实现累加统计量和实现某种objective函数(比如计算似然)这两个接口(抽象方法)。我们大致可以认为是一个聚类，那么我们应该保存这个聚类的所有训练数据。但是在实际实现中，比如对角协方差矩阵的高斯模型里我们只需要保存其充分统计量就可以实现这些接口。

```
less exp/tri1/log/acc_tree.log
# 作者用的是less exp/tri1/log/acc_tree.1.log
# acc-tree-stats --ci-phones=1:2:3:4:5:6:7:8:9:10 exp/mono_ali_train_clean_5/final.mdl "ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_clean_5/split5/1/utt2spk scp:data/train_clean_5/split5/1/cmvn.scp scp:data/train_clean_5/split5/1/feats.scp ark:- | add-deltas  ark:- ark:- |" "ark:gunzip -c exp/mono_ali_train_clean_5/ali.1.gz|" exp/tri1/1.treeacc 
```

这个log文件不会有太多信息，但是我们可以看到这些命令的用法。这个程序会累加每个HMM状态(更准确的说是pdf-class，我们以后会介绍，这里理解为状态就行)的单高斯统计量。\-\-ci-phones告诉程序哪些是上下文无关的phone，即使在triphone模型里，通常我们也把silence等建模成上下文无关的phone。这个程序的输出可以认为就是上面我们讨论的BuildTreeStatsType。

接着运行：
```
less exp/tri1/log/train_tree.log
```
注：似乎文档比较老，现在train_tree变成了build_tree，但是意思都是一样的，也就是根据前面的BuildTreeStatsType进行决策树聚类构建决策树。

```
$ less exp/tri1/log/build_tree.log


# build-tree --verbose=1 --max-leaves=2000 --cluster-thresh=-1 exp/tri1/treeacc data/lang_nosp/phones/roots.int exp/tri1/questions.qst data/lang_nosp/topo exp/tri1/tree 
# Started at Wed Apr 24 15:28:19 CST 2019
#
build-tree --verbose=1 --max-leaves=2000 --cluster-thresh=-1 exp/tri1/treeacc data/lang_nosp/phones/roots.int exp/tri1/questions.qst data/lang_nosp/topo exp/tri1/tree 
LOG (build-tree[5.4.232-532f3]:main():build-tree.cc:104) Number of separate statistics is 81544
LOG (build-tree[5.4.232-532f3]:BuildTree():build-tree.cc:161) BuildTree: before building trees, map has 41 leaves.
LOG (build-tree[5.4.232-532f3]:SplitDecisionTree():build-tree-utils.cc:577) DoDecisionTreeSplit: split 1959 times, #leaves now 2000
LOG (build-tree[5.4.232-532f3]:BuildTree():build-tree.cc:187) Setting clustering threshold to smallest split 554.324

```

聚类前还需要自动生成问题，我们可以在steps/train_tri1.sh看看这个过程，如下图所示(注：作者的是steps/train_deltas.sh)：

<a name='buildtree'>![](/img/kaldidoc/buildtree.png)</a> 
*图：决策树聚类的脚本* 

最后看一下hmm/hmm-topology.h。类HmmTopology为phone定义HMM的拓扑结构。它包含了默认的跳转，这是用来初始化跳转概率的。我们来看一下代码注释提供的topo的"xml"定义：
```
 <Topology>
 <TopologyEntry>
 <ForPhones> 1 2 3 4 5 6 7 8 </ForPhones>
 <State> 0 <PdfClass> 0
 <Transition> 0 0.5
 <Transition> 1 0.5
 </State>
 <State> 1 <PdfClass> 1
 <Transition> 1 0.5
 <Transition> 2 0.5
 </State>
 <State> 2 <PdfClass> 2
 <Transition> 2 0.5
 <Transition> 3 0.5
 </State>
 <State> 3
 </State>
 </TopologyEntry>
 </Topology>
```
上面的"xml"是加了引号的，这只是看起来像xml而已，并不是严格的xml(就像很多程序员自己用字符串拼接出来看起来像xml的字符串就说是xml，其实根本没有考虑转义字符等等)。

我们简单的来解读一下这个topo：它有8个phone——1-8，都是这个topo结构。第一个状态(0)的PdfClass是0，它跳转到0(自己)的概率是0.5，跳到下一个状态1的概率也是0.5。状态1和2也是类似的，状态3是终止状态，所以它没有\<Transition>，或者说它是跳到下一个phone的开始状态。第一个状态0约定是初始状态。

这里定义的PdfClass不同的状态是不同的值，但是我们完全可以让它一样，一样的PdfClass就是Tie在一起的状态——也就是它们的GMM是共享的。




