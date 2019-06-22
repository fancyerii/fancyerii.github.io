---
layout:     post
title:      "Kaldi的编译过程"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[The build process (how Kaldi is compiled) ](http://kaldi-asr.org/doc/build_setup.html)，介绍Kaldi的编译过程。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

 
本文介绍Kaldi的编译过程。

请参考[External matrix libraries](http://kaldi-asr.org/doc/matrixwrap.html)来了解Kaldi怎么使用外部的矩阵库已经相关的链接错误；你也可以参考[Downloading and installing Kaldi](http://kaldi-asr.org/doc/install.html)。

## 在Windows下编译

不建议在Windows下编译。

## configure脚本是怎么工作的

configure脚本有很多参数，其中一种用法是：
```
./configure --shared
```

上面的--shared会构建动态库，这样的程序会小一点，它也有参数指定使用那个矩阵库，比如可以指定使用OpenBlas。打开这个文件，前面的注释有一些示例用法，大家可以参考：
```
  11 #  Example command lines:
  12 # ./configure --shared  ## shared libraries.
  13 # ./configure
  14 # ./configure --mkl-root=/opt/intel/mkl
  15 # ./configure --mkl-root=/opt/intel/mkl --threaded-math=yes
  16 # ./configure --mkl-root=/opt/intel/mkl --threaded-math=yes --mkl-threading=tbb
  17 #        # This is for MKL 11.3, which does not seem  to provide Intel OMP libs
  18 # ./configure --openblas-root=../tools/OpenBLAS/install
  19 #        # Before doing this, cd to ../tools and type "make openblas".
  20 #        # Note: this is not working correctly on all platforms, do "make test"
  21 #        # and look out for segmentation faults.
  22 # ./configure --atlas-root=../tools/ATLAS/build
  23 # ./configure --use-cuda=no   # disable CUDA detection (will build cpu-only
  24 #                             # version of kaldi even on CUDA-enabled machine
  25 # ./configure --static --fst-root=/opt/cross/armv8hf \
  26 # --atlas-root=/opt/cross/armv8hf --host=armv8-rpi3-linux-gnueabihf
  27 #        # Cross compile for armv8hf, this assumes that you have openfst built
  28 #        # with the armv8-rpi3-linux-gnueabihf toolchain and installed to
  29 #        # /opt/cross/armv8hf. It also assumes that you have an ATLAS library
  30 #        # built for the target install to /opt/cross/armv8hf and that the
  31 #        # armv8-rpi3-linux-gnueabihf toolchain is available in your path
  32 # ./configure --static --openblas-root=/opt/cross/arm-linux-androideabi \
  33 # --fst-root=/opt/cross/arm-linux-androideabi --fst-version=1.4.1 \
  34 # --android-incdir=/opt/cross/arm-linux-androideabi/sysroot/usr/include \
  35 # --host=arm-linux-androideabi
  36 #        # Cross compile for Android on arm. The only difference here is the
  37 #        # addition of the the --android-includes flag because the toolchains
  38 #        # produced by the Android NDK don't always include the C++ stdlib
  39 #        # headers in the normal cross compile include path.
```
这个脚本支持Cygwin、Darwin和Linux，它会生成kaldi.mk，而kaldi.mk会被包含在Makefile里。


## 编辑kaldi.mk

运行configure之后你可以会修改kaldi.mk的如下内容：

* 修改debug级别
    * 默认是"-O1"
    * 为了调试方便可以去掉"-O0 -DKALDI_PARANOID"这一行的注释
    * 为了性能可能把上面改成"-O2 -DNDEBUG" 或者 "-O3 -DNDEBUG"
 
* 修改浮点数的精度 
    * 默认使用单精度浮点数，如果你怀疑bug等是由于浮点数的舍入引起的，可以把-DKALDI_DOUBLEPRECISION=0从0改成1

* 去掉警告
    * 为了去掉编译OpenFst时的有符号整数与无符号整数而带来的警告，可以个CXXFLAGS增加-Wno-sign-compare
 

我们也可以通过修改kaldi.mk来使用不同的数学库(比如使用CLAPACK代替LAPACK，LAPACK默认是使用ATLAS)或者动态而不是静态链接数学库。但是这个修改非常复杂，因此无法提供通用的指令来实现(你可以参考[External matrix libraries](http://kaldi-asr.org/doc/matrixwrap.html)，从而理解它们的编译过程)。更简单的办法是通过前面介绍的修改configure脚本来选择不同的数学库。



## Makefile定义的target

Makefile定义的target包括：

* "make depend" 会重新编译依赖。在编译Kaldi前最后重新编译依赖(比如通过git pull更新了代码)。如果.depend文件过期了，那么你可以会看到下面的错误信息：
```
    make[1]: *** No rule to make target `/usr/include/foo/bar', needed by `baz.o'.  Stop.
```
* "make all" (或者只是"make") 编译所有的代码，包括测试代码 

* "make test" 进行测试(如果你修改了代码，那么一定要进行测试)

* "make clean" 删除所有的二进制程序、.o(目标)文件和.a(archive)文件

* "make valgrind" 使用valgrind来检查内存泄露 

* "make cudavalgrind" 检查GPU代码的内存泄露


## 编译的二进制程序的位置


目前，编译后的二进制程序没有特殊的位置，它们都是和源代码放在一起。目前，二进制程序可能会放到bin/"、"gmmbin/"、featbin/"、"fstbin/"和"lm/"下，这些目录都在src下。以后我们可能会把它们放到一个专门统一的地方。

## Makefile怎么工作的


src/Makefile会调用所有子目录(src/base, src/matrix等等)下的Makefile。每个子目录都有自己的Makefile，它们的结果都是类似的，都包含如下行：
```
include ../kaldi.mk
```
这有点像C语言的#include。当阅读kaldi.mk的时候，一定要记得它是在子目录被调用的。下面是kaldi.mk的一个例子，它是用于Linux系统，我们删除了valgrind相关的规则。

```
ATLASLIBS = /usr/local/lib/liblapack.a /usr/local/lib/libcblas.a \
          /usr/local/lib/libatlas.a /usr/local/lib/libf77blas.a
CXXFLAGS = -msse -Wall -I.. \
      -DKALDI_DOUBLEPRECISION=0 -msse2 -DHAVE_POSIX_MEMALIGN \
     -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I ../../tools/ATLAS/include \
       -I ../../tools/openfst/include \
       #-g -O0 -DKALDI_PARANOID
LDFLAGS = -rdynamic
LDLIBS = ../../tools/openfst/lib/libfst.a -ldl $(ATLASLIBS) -lm
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
```

因此kaldi.mk复杂设置include的路径(gcc的-I选项)，定义预处理变量，设置编译器选项和链接哪些库。

## Kaldi可以在哪些平台编译


我们在Windows、Cygwin、各种Linux发行版(包括Ubuntu, CentOS, Debian, Red Hat和SUSE)和Darwin。我们推荐使用g++ 4.7以上的版本，当然其它编译器如llvm和英特尔的icc也是可以工作的。
