---
layout:     post
title:      "外部矩阵库"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[External matrix libraries](http://kaldi-asr.org/doc/matrixwrap.html)，介绍Kaldi依赖的外部矩阵库。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}
 
## Overview
 

Kaldi的矩阵代码基本是封装了BLAS和LAPCK这两个线性代数库(的接口)。代码设计的尽量灵活来适配不同的实现。目前支持4个选项：

* Intel MKL，它同时提供BLAS和LAPACK(当前最新版本的默认值，因为之前是收费的，老的版本不是默认值)
* OpenBLAS，提供BLAS和LAPACK
* ATLAS，它实现BLAS和部分LAPACK
* BLAS的参考实现和CLAPACK(注意：这里没有经过完善的测试)



代码必须知道使用了这个4个选项的哪一个，原因在于虽然理论上BLAS和LAPACK是标准的接口，但是不同的实现还是有一些差别。Kaldi代码需要HAVE_ATLAS, HAVE_CLAPACK, HAVE_OPENBLAS和HAVE_MKL这4个宏中的一个(且仅一个)，我们通常使用类似-DHAVE_ATLAS来告诉编译器。它必须在链接的时候链接到正确的库。kaldi-blas.h文件处理怎么include合适的库的头文件，怎么使用typedef把不同库的实现统一成一样的函数。但是这种封装并不能解决所有的问题，因为ATLAS和CLAPACK的高层routine的调用方法是不同的。此外ATLAS并没有实现全部的LAPACK，因此还需要Kaldi自己来实现一部分功能。


src目录下的configure脚本负责设置Kaldi使用哪个库。它会在src目录下创建kaldi.mk，这里会针对不同编译器设置合适的flags。如果调用时不设置任何参数，默认会在系统里的"常见"位置搜索Intel MKL，但这是可以配置的。请读者运行"./configure --help"来查看完整的帮助。

## 理解BLAS和LAPACK
因为在这里经常提及BLAS和LAPACK，所以在这里我们简要的解释一下它们。

### Basic Linear Algebra Subroutines (BLAS)
 
BLAS是一个底层矩阵-向量运算的subroutine的集合(BLAS最早是由Fortran语言实现，在Fortran里把没有返回值的"函数"叫做subroutine)。包括BLAS Level 1(向量-向量运算)、Level 2(向量-矩阵运算)和Level 3(矩阵-矩阵运算)。这些subroutine的名字类似与daxpy(是"double-precision a x plus y"的缩写，翻译成中文就是双精度的计算ax+y)，dgemm("double-precision general matrix-matrix multiply"的缩写，翻译成中文的就是双精度的通用矩阵-矩阵乘法)。BLAS只是一个接口(规范)，有很多不同的实现。最早的[BLAS的参考实现](http://www.netlib.org/blas/)可以追溯到1979年，这是由netlib维护的。参考实现没有任何优化，只是一个正确的实现，它的目的是用于验证其它实现的正确性。MKL、ATLAS和OpenBLAS提供了优化的BLAS实现。

CBLAS是C语言版本的BLAS接口。

### Linear Algebra PACKage (LAPACK)
 

LAPACK是一些线性代数的routine，最早使用Fortran实现。它包括比BLAS更高级的routine，比如矩阵求逆、SVD分解等等。[LAPACK的参考实现](https://github.com/Reference-LAPACK)也是由Netlib维护。LAPACK内部会使用BLAS。当然也可以混合使用的实现(比如使用ATLAS的BLAS来实现Netlib的LAPACK)。



CLAPACK是使用f2c工具自动把Fortran转换成C版本的代码。因此，在链接CLAPACK时需要f2c库(通常使用-lg2c或者-lf2c)。



MKL完全用C语言由自己来实现BLAS和LAPACK，因此不依赖额外的库。

## Intel Math Kernel Library (MKL)


Intel MKL提供了一个C语言版本的高效的BLAS和LAPACK实现，目前它是Kaldi最优先的CBLAS/CLAPACK提供者。为了使用MKL，需要使用-DHAVE_MKL编译选项。


之前MKL是收取license费用的。从2017年开始，Intel把它免费开放(包括用于商业目的)，但是原代码并不开放。

MKL提供了一个非常高度优化的线性代数函数的实现，尤其是对Intel的CPU。事实上，这个库包括多种代码路径，它会根据CPU的每种不同特性选择最优的代码路径。因此在MKL里，不需要任何手动配置，它会自动的使用所有CPU的特性和特殊指令集(比如AVX2和AVX512)。在CPU上，这些指令会极大的加速线性代数的运算。如果你的CPU的架构比较新，那么使用MKL通常是一个好的选择。


为了简化MKL在Linux上的安装，我们提供了tools/extras/install_mkl.sh这个脚本。我们只安装64位的MKL库，但是install_mkl.sh安装成功后，Intel的repo会注册到系统里，因此你也可以使用系统的包管理器安装32位的库。

对于Mac和Windows系统，需要从[这里](https://software.intel.com/mkl/choose-download)下载(可能需要注册)。如果你的Linux发行版不支持上面的脚本，那么也可以参考嗓门的安装方法。下载的安装程序可以让你选择32位或者64位的包。对于Kaldi，我们只需要64位的。


我们在Linux和Windows下大量的测试过了64位的库。


[MKL Link Line Advisor](http://software.intel.com/articles/intel-mkl-link-line-advisor/)是一个交互式的Web工具，它用于针对不同的系统和编译器可视化的配置编译器选项。

注意：不要使用多线程模式来训练Kaldi(选择"sequential"作为多线程选项)。我们的脚本和程序会在一条机器人运行多个进程，会把CPU用满，使用多线程的版本反而会影响性能。

## Automatically Tuned Linear Algebra Software (ATLAS)



ATLAS是一个常用的BLAS实现以及LAPACK的部分实现。ATLAS的基本想法是根据不同的处理器来进行自动的调整，因此它的编译过程非常复杂和费时。因此要编译ATLAS会非常tricky。对于UNIX类的系统，如果你不是root或者管理员的朋友，你甚至不能编译，因为它需要关闭CPU  throttling(CPU throttling指的是自动调整CPU的时钟频率，比如快没电的时候把时钟频率调低)；而在Windows，ATLAS只能在Cygwin里编译。更好的办法可能是复制相同平台上别人编译好的库，不过我们这里无法提供具体的建议。ATLAS通常比Netlib的参考实现的BLAS要高效。但是ATLAS值包含部分的LAPACK函数。它包括矩阵的求逆和Cholesky分解，但是没有SVD分解。因此我们实现了一些LAPACK函数(SVD和特征值分解)。

ATLAS遵循BLAS接口，但是LAPACK的接口和Netlib的不同(它更像C语言的风格而不是Fortran的风格)。因此，有很多#ifdef的代码来根据链接的是ATLAS还是CLAPACK来切换不同的调用风格。

### 在Windows下安装


在Windows下安装ATLAS(注意需要先安装Cygwin)，参考windows/INSTALL.atlas。注意：ATLAS在Windows下的维护并不活跃，因此不能保证它一定可以工作。

### Linux下的安装
 
如果系统没有安装ATLAS，你需要从源代码安装ATLAS。有时候即使系统已经安装了，它也不见得是针对你的系统的最优的实现，因此最好从源代码安装。安装最简单的方式是cd的tools目录然后运行./install_atlas.sh。如果这个脚本不work的话，请参考[官方安装文档](http://math-atlas.sourceforge.net/atlas_install/)。

在安装ATLAS之前可能需要使用"cpufreq-selector -g performance"关闭CPU throttling(cpufreq-selector可能在sbin目录下)。你可以首先运行install_atlas.sh，如果这个脚本不工作的话有可能是CPU throttling的问题。

## OpenBLAS

Kaldi现在支持链接OpenBLAS库了，它实现BLAS和部分LAPACK。OpenBLAS也会自动的编译Netlib的LAPACK。OpenBLAS是GotoBLAS项目的fork，目前GotoBLAS已经不再维护了。为了使用OpenBLAS，我们需要进入tools目录然后运行"make openblas"，然后进入src，给configure脚本合适的编译选项。

## Java Matrix Package (JAMA)

 
JAMA是Java实现的一个线性代数库，有NIST和MathWorks实现并且开源(参考math.nist.gov/javanumerics/jama)。我们使用它来填补ATLAS缺失的部分。特别是我们使用-DHAVE_ATLAS选项编译是，因为它的CLAPACK没有实现SVD和特征值分解，我们参考了JAMA的代码用C++实现了这些函数。参考[EigenvalueDecomposition](math.nist.gov/javanumerics/jama)的MatrixBase::JamaSvd函数。当然Kaldi的矩阵库的使用中不需要直接使用这些代码。

## 可能会遇到的链接错误


为了验证矩阵库是否正确编译，进入matrix目录执行"make"看它能否成功。很多make的问题通常是链接错误。这一节我们会总结一些常见的链接错误。

根据不同的编译选项(-DHAVE_CLAPACK, -DHAVE_LAPACK或者-DHAVE_MKL)，代码会链接不同的库。在排查链接错误的时候，请记住问题可能是编译选项和链接的库的不匹配。

### f2c或者g2c错误
 
如果链接CLAPACK，那么通常需要f2c库，因为CLAPACK=会使用f2c工具。注意：对于较新版本的gcc，你需要使用-lg2c而不是-lf2c。

如果找不到如下的符号则有可能是f2c的问题：
```
s_cat, pow_dd, r_sign, pow_ri, pow_di, s_copy, s_cmp, d_sign
```

###  CLAPACK链接错误
 
如果你使用了编译选项-DHAVE_CLAPACK但是有没有(正确)安装CLAPACK，那么可能会出现找不到下面的符号：

```
sgetrf_, sgetri_, dgesvd_, ssptrf_, ssptri_, dsptrf_, dsptri_, stptri_, dtptri_
```
这个库文件通常类似liblapack.a。如果使用动态库，那么需要使用"-llapack"。注意：这和ATLAS库的名字相同，但是它们提供的符号(函数)是不同的。CLAPACK版本的liblapack的符号类似于sgesvd_和sgetrf_，而ATLAS的版本的符号类似于clapack_sgetrf或者ATL_sgetrf。

### BLAS链接错误

在链接BLAS的实现是可能会出现这些错误。在链接的顺序不对的时候也可能出现。CLAPACK依赖BLAS，因此你需要在CLAPACK之后链接BLAS。

下面这些符号可能与BLAS的链接有关：
```
cblas_sger, cblas_saxpy, cblas_dapy, cblas_ddot, cblas_sdot, cblas_sgemm, cblas_dgemm
```

为了解决这些问题，可以使用libcblas.a这样的静态库，或者-lcblas(假设它在LD_LIBRARY_PATH搜索路径里)。这个库可能来自ATLAS(更快)或者来自Netlib。就我所知，它们的接口是相同的。

### cblaswrap链接错误


CLAPACK似乎依赖f2c_sgemm，这好像是cblas_sgemm的warpping。我不太清楚它是怎么wrapping的。但是如果使用Netlib的CLAPACK，则你需要链接libcblaswr.a或者使用-lcblaswr来动态链接。如果没有cblaswrap，则可能出现有关如下符号的错误：

```
f2c_sgemm, f2c_strsm, f2c_sswap, f2c_scopy, f2c_sspmv, f2c_sdot, f2c_sgemv
```

### ATLAS没有实现的BLAS
 

如果链接ATLAS实现的BLAS，如果只使用-lcblas(或者静态的libcblas.a)而没有-latlas(或者libatlas.a)，那么你可能碰到问题。因为ATLAS的BLAS内部会调用cblas_sger这类的函数。如果出现如下符号缺失的问题很可能就是这个原因：

```
ATL_dgemm, ATL_dsyrk, ATL_dsymm, ATL_daxpy, ATL_ddot, ATL_saxpy, ATL_dgemv, ATL_sgemv
```

### ATLAS没有实现LAPACK

在使用-DHAVE_ATLAS可能会出现这些错误。ATLAS的CLAPACK函数名字和CLAPCK自己的不同(它自己的有前缀clapack_)

如果找不到如下符号：

```
clapack_sgetrf, clapack_sgetri, clapack_dgetrf, clapack_dgetri
```

那么你链接的ATLAS找不到这些符号。你需要的库文件可能叫liblapack.a, libclapack.a 或者 liblapack_atlas.a，这么多文件你可以通过查找ATL_cgetrf符号来确定是哪一个("nm <library-name> | grep ATL_cgetrf")。你可以使用-llapack来动态的链接。注意：liblapack.a或者liblapack.so可能是CLAPACK也可能是ATLAS版本的CLAPACK。但是它们提供不同的符号。因此可以使用"nm"或者"strings"命令来确定。
