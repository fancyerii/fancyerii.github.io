---
layout:     post
title:      "CUDA矩阵库"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[The CUDA Matrix library](http://kaldi-asr.org/doc/cudamatrix.html)，介绍GPU上基于CUDA的矩阵库。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}
 
 


CUDA矩阵库通过用GPU来实习矩阵运算并且提供了类似Kaldi矩阵库的接口。


基本的原则为：如果你像把一部分计算放到GPU上，你可以声明CuMatrix或者CuVector来替代Matrix或者Vector。然后如果你配置好Kaldi使用GPU了而且Kaldi程序可以访问GPU，那么这些运算就会在GPU上运行。否则它们还是会在CPU上运行。如果你配置好了GPU并且程序初始化GPU设备后，CuMatrix和CuVector对象会把它们的值保存在GPU的内存里。


你不能把CuMatrix/CuVector与Matrix/Vector混合在一起进行运算，因为它们呆在不同的内存空间里，但是你可以在这两个空间里拷贝数据。Kaldi不会尝试自动决定哪些运算在GPU上运算更加高效：它完全需要程序员手动来控制。


如果configure脚本可以找到Nvidia的编译器nvcc，则它假设你需要编译GPU版本的Kaldi，它会定义HAVE_CUDA=1并且设置其它Makefile里的变量以便支持GPU的编译。如果你不想编译GPU版本的(即使有GPU)，那么可以使用\-\-use-cuda=no来禁止它。如果脚本无法在常见的位置找到CUDA toolkit(可能你把它安装到比较特殊的位置了)，那么也可以通过类似"\-\-cudatk-dir=/opt/cuda-4.2"来告诉脚本。如果你想知道configure脚本是否使用CUDA，可以在kaldi.mk里搜索CUDATKDIR，如果找到了就说明配置使用CUDA。我们也可以在编译后运行"cuda-compiled"这个命令。如果它正常结束(应该没有任何输出)，那么则表示安装成功，比如：

```
lili@lili-Precision-7720:~/codes/kaldinew/kaldi/egs/mini_librispeech/s5$ cuda-compiled
lili@lili-Precision-7720:~/codes/kaldinew/kaldi/egs/mini_librispeech/s5$ which cuda-compiled 
/home/lili/codes/kaldinew/kaldi/egs/mini_librispeech/s5/../../../src/nnet2bin/cuda-compiled
```




You can also tell from the logs whether a program is using the GPU. If it is using the GPU, you'll see lines like this near the top of the program's output: 

通过程序的log也能判断是否使用了GPU。如果使用了GPU，你通常会看到类似如下的输出：

```
LOG (nnet-train-simple:IsComputeExclusive():cu-device.cc:229) CUDA setup operating under Compute Exclusive Mode.
LOG (nnet-train-simple:FinalizeActiveGpu():cu-device.cc:194) The active GPU is [1]: Tesla K10.G2.8GB  \
    free:3519M, used:64M, total:3583M, free/total:0.982121 version 3.0
```

除了在Makefile配置使用CUDA之外，所有需要使用GPU的程序都需要增加如下的代码：

```
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
```

上面的use_gpu是一个字符串，通常是来自命令行的选项(比如"\-\-use-gpu=wait")，它的值为如下中的一个：

* yes 使用GPU(如果没有或者被别的进程占用则会crash)
* no 不使用GPU
* optional 如果有GPU则使用，没有则用CPU
* wait 和yes类似，但是如果GPU被别的进程使用，则它会等待直到有空闲的GPU

如果程序没有通过命令行选项\-\-use-gpu指定，那么即使它的代码里使用了CuVector和CuMatrix，它也不会GPU上的运算。通常我们只在神经网络的训练里使用GPU。

Nvidia的GPU(目前Kaldi只支持Nvidia的GPU)有许多不同的"计算模式(compute modes)"："default"、"process exclusive"和"thread exclusive"。它用于控制一个GPU是否同时运行多个进程。Kaldi推荐运行在独占模式(exclusive mode)，进程独占(process exclusive)或者线程独占(thread exclusive)都可以。你可以使用下面的命令来查询目前GPU的计算模式：
```
# nvidia-smi  --query | grep 'Compute Mode'
    Compute Mode                    : Exclusive_Process
```

如果模式不是独占，那么可以使用命令(需要root权限)：
```
# nvidia-smi -c Exclusive_Process
```
来设置。我们可能需要把上面的命令加到启动脚本里，以避免重启机器后恢复默认的"Default"模式。

 
Kaldi会cache之前释放的GPU内存(类似于简单的自动内存管理)而不是直接调用Nvidia提供的malloc和free函数，这可以避免malloc的overhead。这么做的原因是我们发现啊Amazon的云服务器上发现Nvidia的malloc非常慢。帧是由于虚拟机带来的问题，但是我们不确定这个问题释放还存在。如果你运行的模式是默认(非独占的)模式的话，它可能会造成内存分配失败。你可以在代码里加入"CuDevice::Instantiate().DisableCaching()"来禁用cache。
