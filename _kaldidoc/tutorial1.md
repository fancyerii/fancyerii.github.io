---
layout:     post
title:      "Kaldi教程(一)"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Kaldi tutorial](http://kaldi-asr.org/doc/tutorial.html)，这是第一部分。
 <!--more-->
 
**目录**
* TOC
{:toc}

## Prerequisites

本教程假设读者理解基于HMM-GMM的语音识别系统的基本原理，不了解的读者可以参考M. Gales和S. Young (2007)的文章[The Application of Hidden Markov Models in Speech Recognition](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf)。也可以参考[《深度学习理论与实战：提高篇》]({{ site.baseurl }}{% post_url 2019-03-14-dl-book %})中的相关内容。[HTK Book](http://htk.eng.cam.ac.uk/docs/docs.shtml)也是一个很好的资料(需要注册后才能下载)。但是除非有很强的数学背景和非常的投入，我们不鼓励在一个研究机构之外学习。因此本文的目标用户是语音识别的研究者或者是研究这个方向的研究生或者高年级本科生。【这和本文作者的观点并不一致，本文作者希望能够让更多的人通过自学能够进入这个行业，因为不管是国内还是国外，专业的研究机构非常有限，大部分人都不可能有这样的学习机会。】


我们假设读者熟悉C++和脚本编程。此外，我们需要LDC的[RM数据集](https://catalog.ldc.upenn.edu/LDC93S3A)。【这个数据集是要收费的，如果读者没有可能无法实际进行操作，但是即使这样也是建议详细阅读】。

另外当然需要安装好Kaldi和常见的工具，包括wget, git, svn, awk, perl等等。另外为了方便阅读，我们会标注出每一节实验需要的时间。

## Getting started(15分钟)

我们首先需要安装kaldi，我们首先使用git clone代码：
```
git clone https://github.com/kaldi-asr/kaldi.git
```
然后参考INSTALL文件进行安装。

## 使用Git进行版本控制 (5分钟) 

忽略，本文假设读者会使用git，不了解的读者可以上网寻找资料学习git的基本用法。【如果不想修改代码或者修改了代码也不需要提交PR，那么会git clone和git pull也就够了】

## Kaldi代码目录结构概览(20分钟) 

Kaldi的主目录包含了很多子目录和文件，最重要的子目录是"tools"、"src"和"egs"。egs包含了很多示例的recipe，我们后面会介绍。这里先介绍"tools"和"src"。

### tools(10分钟)

tools目录包含Kaldi需要的常见(外部)工具。读者可以参考INSTALL文件，它说明了怎么安装这些工具。这里最重要的就是openfst目录，它是一个符号链接，实际的目录可能是openfst-1.6.7。openfst目录下有bin和lib两个子目录，分别是编译后的二进制程序和库，Kaldi同时需要依赖它们。一些Kaldi的脚本会依赖二进制程序，另外一些Kaldi(C++)程序需要依赖openfst的库。"include/fst/"是openfst的头文件，Kaldi的代码编译需要它们，而lib是在链接时需要。要理解Kaldi就必须先理解OpenFst，更多关于OpenFst的内容请参考[官网](http://www.openfst.org/)。

现在我们只看一下include/fst/fst.h，打开这个文件，我们会发现有很多模板(template)的代码，如果想阅读Kaldi代码必须要熟悉C++的模板。

进入bin目录(或者把它加入到PATH)里，我们通过下面的例子来了解OpenFst的基本(命令行)用法。

```
$cat >text.fst <<EOF
0 1 a x .5
0 1 b y 1.5
1 2 c z 2.5
2 3.5
EOF
```
上面的脚本是创建一个text.fst文件，它是文本格式描述的WFST。最后一行只有两列，它说明状态2是终止状态，并且weight是3.5。其余的行都有5列，每一行描述WFST的一条边。其中第一列是起点id，第二列是终点id，第三列是输入符号，第四列是输出符号，第五列是weight。因此第一行表示的边为：从0到1的边，输入是a，输出是x，weight是0.5。

细心的读者可能会问：那么WFST的初始状态是哪个呢？OpenFst约定第一行的起点就是初始状态，一个WFST只有一个初始状态。weight也可以不指定，那么默认就是对于semi-ring的零元。

接下来我们定义输入符号表，也就是把输入符号a、b和c等映射成整数ID的文件：
```
$cat >isyms.txt <<EOF
<eps> 0
a 1
b 2
c 3
EOF
```
以及输出符号表的文件：
```
$cat >osyms.txt <<EOF
<eps> 0
x 1
y 2
z 3
EOF

```

注意：OpenFst要求符号的ID是非负整数，并且0必须代表特殊的符号ε，因此普通的符号只能从1开始。

为了能使用bin下面的OpenFst命令，我们把bin目录(假设当前在bin下)加到PATH里：
```
export PATH=.:$PATH
```

我们用下面的命令把文本描述的WFST编译成二进制的格式：
```
fstcompile --isymbols=isyms.txt --osymbols=osyms.txt text.fst binary.fst
```

我们可以用fstdraw把它输出成dot文件然后用dot命令绘制出来：
```
fstdraw --isymbols=isyms.txt --osymbols=osyms.txt binary.fst  > binary.dot
dot -Gorientation -Tpng binary.dot > binary.png
```


<a name='binary'>![](/img/kaldidoc/binary.png)</a>
*图：binary.fst*

然后我们把上面的WFST进行invert的操作：
```
fstinvert binary.fst > inverted.fst
```


<a name='inverted'>![](/img/kaldidoc/inverted.png)</a>
*图：inverted.fst*

最后我们把inverted.fst和binary.fst复合起来：
```
fstcompose inverted.fst binary.fst > binary2.fst
```

<a name='binary2'>![](/img/kaldidoc/binary2.png)</a>
*图：binary2.fst*

对于一个编译后的WFST，我们也可以用fstprint把它变成文本格式的，比如：
```
fstprint --isymbols=isyms.txt --osymbols=osyms.txt binary.fst
```

### src(10分钟)
我们首先来看Makefile，这个文件首先定义变量SUBDIRS，这个变量列举了src下的所有包含源代码的子目录。有些子目录名是以bin结尾，这表明这些子目录会build出一下可执行的工具，而其它的子目录只是构建Kaldi内部使用的库。

Makefile中有一个target是test，我们可以用"make test"来执行它。这个命令会进入各个子目录进行测试，通常这个命令应该成功，但是需要花费很长时间。我们也可以用"make valgrind"来检测Kaldi代码是否有内存泄露，这个命令可能会失败，我们暂时可以忽略(假设Kaldi代码没问题)。

Makefile中有一行"include kaldi.mk"，如果进入任意一个子目录(比如base)的Makefile，我们会看到"include ../kaldi.mk"。这个文件是前面的安装时生成的，里面会定义依赖的库的头文件和动态库的位置，比如作者的这个文件的部分内容为：
```
KALDI_FLAVOR := dynamic
KALDILIBDIR := /home/lili/codes/kaldi/src/lib
DOUBLE_PRECISION = 0
OPENFSTINC = /home/lili/codes/kaldi/tools/openfst/include
OPENFSTLIBS = /home/lili/codes/kaldi/tools/openfst/lib/libfst.so
OPENFSTLDFLAGS = -Wl,-rpath=/home/lili/codes/kaldi/tools/openfst/lib

ATLASINC = /home/lili/codes/kaldi/tools/ATLAS_headers/include
ATLASLIBS = /usr/lib/libatlas.so.3 /usr/lib/libf77blas.so.3 /usr/lib/libcblas.so.3 /usr/lib/liblapack_atlas.so.3



CXXFLAGS = -std=c++11 -I.. -I$(OPENFSTINC) $(EXTRA_CXXFLAGS) \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$(ATLASINC) \
           -msse -msse2 -pthread \
           -g # -O0 -DKALDI_PARANOID

```

为了调试方便，我们可能需要修改CXXFLAGS。我们可以把注释"\# -O0 -DKALDI_PARANOID"去掉，这样编译器不会做过多的优化，方便调试。

我们再看一下base/Makefile，指令"all:"告诉make这是一个top-level的target。顶级的Makefile里除了all之外还有clean、test等target，我们可以用"make clean"和"make test"执行它们。

打开base/Makefile，TESTFILES变量列举了测试的程序，我们可以选择一个(比如kaldi-math-test)来运行它，然后查看对应的源代码。注：默认我们在build Kaldi的时候只会生成.o目标文件，如果想生成测试的程序，需要在base目录下执行"make test"。然后就可以执行它："./kaldi-math-test"。

这个测试对应的代码是kaldi-math-test.cc，我们通过这些测试代码可以学习Kaldi的一些函数的用法。对于开源的代码，一般文档不会很多(即使很多公司的商业化代码大家也不喜欢太多文档)，因此单元测试一个很好的阅读代码的起点。

比如kaldi-math-test.cc就能看到Kaldi的Math库的常见用法。比如UnitTestLogAddSub函数可以看到函数LogAdd的用法：
```
void UnitTestLogAddSub() {
  for (int i = 0; i < 100; i++) {
    double f1 = Rand() % 10000, f2 = Rand() % 20;
    double add1 = Exp(LogAdd(Log(f1), Log(f2)));
    double add2 = Exp(LogAdd(Log(f2), Log(f1)));
    double add = f1 + f2, thresh = add*0.00001;
    KALDI_ASSERT(std::abs(add-add1) < thresh && std::abs(add-add2) < thresh);


    try {
      double f2_check = Exp(LogSub(Log(add), Log(f1))),
             thresh = (f2*0.01)+0.001;
      KALDI_ASSERT(std::abs(f2_check-f2) < thresh);
    } catch(...) {
      KALDI_ASSERT(f2 == 0);  // It will probably crash for f2=0.
    }
  }
}
```

LogAdd实现的是log域的加法，它的定义为：

$$
\begin{split}
LogAdd(x,y) & =log(e^x+e^y)
\end{split}
$$


当然我们可以直接安装定义计算，但是$e^x$可能很大或者很小，容易溢出，所以Kaldi用更加数值文档的方法实现了这个函数。当然我们这里的重点不是学习这个函数，而是看Kaldi的单元测试代码。Kaldi使用KALDI_ASSERT来检查代码执行是否和预期一样，写过单元测试的读者应该都能看懂。不过和业务逻辑的代码不一样，数值计算尤其是浮点数的运算是有误差的，因此比较结果时我们通常检查它们的误差再某个很小的范围内就算正确了。上面的检查依据是：

$$
\begin{split}
e^{LogAdd(log(x), log(y))} &=e^{log(e^{log(x)}+e^{log(y)})} \\
& = e^{log(x+y)}=x+y
\end{split}
$$

所以为了判断LogAdd函数是否正确，我们就可以判断Exp(LogAdd(Log(f1), Log(f2)))是否等于f1+f2。

我们再来看一下kaldi-math.h，熟悉一下Kaldi的代码习惯。
```
#ifndef KALDI_BASE_KALDI_MATH_H_
#define KALDI_BASE_KALDI_MATH_H_ 1

#ifdef _MSC_VER
#include <float.h>
#endif

#include <cmath>
#include <limits>
#include <vector>

#include "base/kaldi-types.h"
#include "base/kaldi-common.h"


#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-16
#endif

#define KALDI_ISNAN std::isnan
#define KALDI_ISINF std::isinf
#define KALDI_ISFINITE(x) std::isfinite(x)

inline double LogAdd(double x, double y) {
  double diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffDouble) {
    double res;
    res = x + Log1p(Exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}
```

Kaldi里的\#include都是相对于src目录的，所以我们的代码即使在base目录下，仍然写成\#include \"base/kaldi-types.h\"。另外所有的宏(除了特别明确的)都以KALDI_开头，这是为了防止冲突，因为宏是不局限于kaldi namespace的。函数的名称类似于LogAdd，每个单词的首字母都是大写，更多Coding Style请参考[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)。

为了了解其它风格，我们来看util/text-utils.h：
```

namespace kaldi {

/// Split a string using any of the single character delimiters.
/// If omit_empty_strings == true, the output will contain any
/// nonempty strings after splitting on any of the
/// characters in the delimiter.  If omit_empty_strings == false,
/// the output will contain n+1 strings if there are n characters
/// in the set "delim" within the input string.  In this case
/// the empty string is split to a single empty string.
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out);
```

所有的代码都是在kaldi这个namespace下，并且函数参数首先是输入参数，通常是常量引用(比如const std::string &full；const char *delim)。当然对于基本类型的输入参数，因为是值拷贝的，因此不需要是引用，比如这里的bool omit_empty_strings。接着是输出(或者可以修改的输入)参数，它的类型是指针(比如std::vector\<std::string> *out)，Kaldi不允许非常量的引用。也就是说输入参数只能是基本类型或者常量引用；输出参数只能是指针。更多Kaldi的Coding Sytle可以参考[这里](http://kaldi-asr.org/doc/style.html)。

接着我们进入gmmbin子目录然后执行：
```
lili@lili-Precision-7720:~/codes/kaldi/src/gmmbin$ ./gmm-init-model
./gmm-init-model 

Initialize GMM from decision tree and tree stats
Usage:  gmm-init-model [options] <tree-in> <tree-stats-in> <topo-file> <model-out> [<old-tree> <old-model>]
e.g.: 
  gmm-init-model tree treeacc topo 1.mdl
or (initializing GMMs with old model):
  gmm-init-model tree treeacc topo 1.mdl prev/tree prev/30.mdl

Options:
  --binary                    : Write output in binary mode (bool, default = true)
  --var-floor                 : Variance floor used while initializing Gaussians (double, default = 0.01)
  --write-occs                : File to write state occupancies to. (string, default = "")

Standard options:
  --config                    : Configuration file to read (this option may be repeated) (string, default = "")
  --help                      : Print out usage message (bool, default = false)
  --print-args                : Print the command line arguments (to stderr) (bool, default = true)
  --verbose                   : Verbose level (higher->more logging) (int, default = 0)

```
请仔细阅读帮助的内容。虽然Kaldi也可以使用\-\-config接受配置文件作为命令的输入，但是Kaldi和HTK不同。HTK完全是配置文件驱动的，而Kaldi更喜欢直接在命令行里指定参数。Kaldi的命令首先接受可选的选项(options)，然后是输入(一个或多个)，然后是输出(一个或者多个)。选项有分为标准的选项(Standard options)和这个命令特有的选项(Options)。标准的选项主要就是\-\-config、\-\-help、\-\-print-args和\-\-verbose，\-\-print-args会把命令行的参数打到stderr上，这样方便我们调试，默认是打开的。另外\-\-binary虽然不是标准的选项，但是很多命令都有，它指定输出的是二进制格式还是人可读的文本格式，默认是二进制(true)，但是为了调试我们可以输出文本格式的。

另外如果我们执行：
```
./gmm-init-model >/dev/null
```
会发现它的输出和前面是完全一样的。和很多普通的*nix系统命令不同，Kaldi命令的一些log信息都是打印到stderr上的，这样更加方便使用管道把不同命令连接起来。也就是说stdout是命令的输出，而stderr是一些debug信息。

为了了解Kaldi的构建过程，我们进入matrix子目录然后执行make：
```
lili@lili-Precision-7720:~/codes/kaldi/src/matrix$ make
g++ -std=c++11 -I.. -I/home/lili/codes/kaldi/tools/openfst/include  -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I/home/lili/codes/kaldi/tools/ATLAS_headers/include -msse -msse2 -pthread -g  -fPIC -DHAVE_CUDA -I/usr/local/cuda/include   -c -o kaldi-matrix.o kaldi-matrix.cc
g++ -std=c++11 -I.. -I/home/lili/codes/kaldi/tools/openfst/include  -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I/home/lili/codes/kaldi/tools/ATLAS_headers/include -msse -msse2 -pthread -g  -fPIC -DHAVE_CUDA -I/usr/local/cuda/include   -c -o kaldi-vector.o kaldi-vector.cc
g++ -std=c++11 -I.. -I/home/lili/codes/kaldi/tools/openfst/include  -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I/home/lili/codes/kaldi/tools/ATLAS_headers/include -msse -msse2 -pthread -g  -fPIC -DHAVE_CUDA -I/usr/local/cuda/include   -c -o packed-matrix.o packed-matrix.cc

```

我们可以看传递给编译器g++的参数，比如-std=c++11，这些参数是在../kaldi.mk里定义的。而这个文件又是我们./configure生成的，熟悉Linux的读者可能会猜测configure是通过automake自动检测系统后生成的，但其实不然，Kaldi的configure是手写的。

我们用vim打开configure，搜索makefiles，我们可以看到类似下面的内容：
```
     if [ "`uname -m`" == "x86_64" ]; then
       if [ "`uname`" == "Darwin" ]; then
         sed 's/lib64/lib/g' < makefiles/cuda_64bit.mk >> kaldi.mk
       else
         cat makefiles/cuda_64bit.mk >> kaldi.mk
       fi

```
可以看到kaldi.mk是通过一下"模板kaldi.mk"生成的，比如我们看一下makefiles/linux_openblas.mk，这表示在Linux下用OpenBlas的配置。
```
CXXFLAGS = -std=c++11 -I.. -I$(OPENFSTINC) $(EXTRA_CXXFLAGS) \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_OPENBLAS -I$(OPENBLASINC) \
           -msse -msse2 -pthread \
           -g # -O0 -DKALDI_PARANOID

```
如果在编译时碰到问题，我们可以修改kaldi.mk。最常见的问题是依赖的线性代数库比如BLAS和LAPACK找不到，关于线性代数库的更多内容，读者可以参考[External matrix libraries](http://kaldi-asr.org/doc/matrixwrap.html)。





