---
layout:     post
title:      "第一章：Intro to JAX"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - Jax
---

<!--more-->
**目录**
* TOC
{:toc}

这一章涵盖了

* 什么是JAX以及它与NumPy的比较
* 为什么要使用JAX
* 以及JAX与TensorFlow/PyTorch的比较。

随着越来越多的研究人员开始使用JAX进行研究，并且像DeepMind这样的大公司也在为其生态系统做出贡献，JAX正变得越来越受欢迎。在本章中，我们将介绍JAX及其强大的生态系统。我们将解释JAX是什么，以及它与NumPy、PyTorch和TensorFlow的关系。我们将详细介绍JAX的优势，以便了解它们如何结合，为您提供一个非常强大的深度学习研究和高性能计算工具。


## 1.1 JAX是什么？

JAX是由谷歌（具体来说是Google Brain团队）开发的具有NumPy接口的Python数学库。它广泛用于机器学习研究，但并不局限于此，许多其他问题也可以使用JAX解决。

JAX的创建者将其描述为Autograd和XLA。如果您对这些名词不熟悉，不用担心，这是正常的，特别是如果您刚开始接触这个领域。

[Autograd](https://github.com/hips/autograd)是一个能够高效计算NumPy代码导数的库，也是JAX的前身。顺便说一下，Autograd库的主要开发者现在也在致力于JAX。简而言之，Autograd意味着您可以自动计算您的计算的梯度，这是深度学习和许多其他领域的核心，包括数值优化、物理模拟，以及更一般的可微分编程。

XLA是谷歌的面向特定领域的线性代数编译器，名为加速线性代数（Accelerated Linear Algebra）。它将您的Python函数与线性代数操作编译为在GPU或TPU上运行的高性能代码。

让我们从NumPy部分开始。

### 1.1.1 JAX作为NumPy

NumPy是Python数值计算的重要工具。它在工业和科学领域得到了广泛应用，以至于NumPy API已经成为Python中处理多维数组的事实标准。JAX提供了一个与NumPy兼容的API，但提供了许多NumPy中没有的新功能，因此有些人将JAX称为“强化版的NumPy”。

JAX提供了一个名为**DeviceArray**的多维数组数据结构，它实现了许多numpy.ndarray的典型属性和方法。还有一个名为jax.numpy的包，它实现了NumPy API，并包含许多众所周知的函数，如abs()、conv()、exp()等等。

JAX试图尽可能地遵循NumPy API，在许多情况下，您可以在不改变程序的情况下从numpy切换到jax.numpy。

但仍然存在一些限制，并非所有的NumPy代码都可以与JAX一起使用。JAX提倡函数式编程范式，并要求使用无副作用的纯函数。因此，JAX数组是不可变的，而NumPy程序经常使用原地更新，例如arr[i] += 10。JAX通过提供一个替代的纯函数式API来解决这个问题，用一个纯索引更新函数替换了原地更新。对于这种特定情况，可以使用arr = arr.at[i].add(10)。还有一些其他的区别，我们将在第三章中讨论它们。

因此，您可以几乎完全利用NumPy的强大功能，并按照您在使用NumPy时习惯的方式编写程序。但在这里您也有新的机会。


### 1.1.2 可组合的转换

JAX远不止于NumPy。它提供了一组可组合的函数转换，适用于Python+NumPy代码。在其核心，JAX是一个用于转换数值函数的可扩展系统，具有四个主要转换（但这并不意味着没有更多的转换即将到来！）：

**获取您的代码的梯度或对其进行微分。**这是深度学习和许多其他领域（包括数值优化、物理模拟和更普遍的可微分编程）的本质。JAX使用一种称为自动微分（或简称autodiff）的方法。自动微分帮助您专注于您的代码，而不是手动计算导数；框架会自动处理。通常通过grad()函数来实现，但也存在其他高级选项。我们将在第四章中对自动微分进行更详细的解释和深入讨论。

**使用jit()或即时编译将您的代码编译成机器代码。**JIT使用Google的XLA来编译和生成GPU（通常是NVIDIA的CUDA，尽管AMD ROCm平台支持正在进行中）和TPU（Google的Tensor处理单元）的高效代码。XLA是支持机器学习框架的后端，最初用于TensorFlow，在包括CPU、GPU和TPU在内的各种设备上运行。我们将在第五章专门讨论这个话题。

**使用vmap()自动矢量化您的代码，即矢量化映射。**如果您熟悉函数式编程，您可能知道什么是映射。如果不熟悉，也不要担心；我们稍后会详细描述它的含义。vmap()负责处理您数组的批次维度，并可以轻松将您的代码从处理单个数据项转换为同时处理多个数据项（称为批次）。您可以称之为自动批处理。通过这样做，您将计算矢量化，这通常会为现代硬件提供显著的性能提升，因为现代硬件可以有效地并行化矩阵计算。我们将在第六章中讨论这个话题。

**并行化**您的代码以在多个加速器上运行，比如GPU或TPU。这是通过pmap()来实现的，它有助于编写单程序多数据（SPMD）程序。pmap()会将一个函数与XLA编译，然后在其XLA设备上复制并并行执行每个副本。这个话题也将在第六章中讨论。




每个转换接受一个函数并返回一个函数。只要您使用函数式纯函数，您可以随意混合不同的转换。我们稍后会谈论这个，但简而言之，函数式纯函数是一种行为仅由输入数据确定的函数。它没有内部状态，不应产生副作用。对于那些来自函数式编程的人来说，这应该是一件自然的事情。对于其他人来说，采用这种编程方式并不难，我们会帮助您理解。

如果您遵循这些约束条件，您可以混合、链接和嵌套转换，并根据需要创建复杂的管道。JAX使得所有这些转换都可以任意组合。例如，您可以准备一个用于处理图像的神经网络的函数，然后自动生成另一个函数，使用vmap()处理一批图像，然后使用jit()将其编译成在GPU或TPU上运行的高效代码（或者通过pmap()以并行方式运行多个），最后生成一个函数，使用grad()计算梯度，以用梯度下降来训练我们的图像处理函数。我们将在途中看到一些令人兴奋的示例。

这些是您不希望在纯NumPy中自行实现的转换。即使您可以，也不再需要手动计算导数。强大的框架会为您处理，无论您的函数有多复杂——与自动矢量化和并行化一样。

图1.1可视化了NumPy是用于处理多维数组的引擎，具有许多有用的数学函数。JAX具有与其多维数组和许多函数兼容的NumPy API。但除此之外，JAX还提供了一套强大的函数转换。



<a>![](/img/jaxinaction/ch1/1.png)</a>

在某种程度上，JAX类似于Julia。Julia也具有即时编译（JIT）、良好的自动微分能力、机器学习库以及丰富的硬件加速和并行计算支持。但是使用JAX，您可以留在您熟悉的Python世界中。有时这是很重要的。

## 1.2 为什么使用JAX？

JAX现在正处于势头上升的阶段。2021年众所周知的“AI现状报告”称JAX是一个新的框架挑战者。

深度学习研究人员和实践者热爱JAX。越来越多的新研究正在使用JAX进行。在最近的研究论文中，我可以提到Google的Vision Transformer（ViT）和MLP-Mixer。Deepmind宣布他们正在使用JAX加速他们的研究，而且JAX易于采用，因为Python和NumPy都被广泛使用和熟悉。其可组合的函数转换有助于支持机器学习研究，JAX已经使得对新算法和架构的快速实验成为可能，并且现在支撑了许多DeepMind最近的出版物。其中，我要特别强调一种称为BYOL（“自举你自己的潜在”）的自监督学习的新方法，一种通用的基于Transformer的用于结构化输入和输出的架构称为Perceiver IO，以及使用2800亿参数的Gopher和700亿参数的Chinchilla进行的大规模语言模型的研究。

2021年中，Huggingface将JAX/Flax列为了他们著名的Transformers库中第三个官方支持的框架。截至2022年4月，Huggingface的预训练模型集合中JAX模型的数量（5530个）已经是TensorFlow模型数量（2221个）的两倍。PyTorch仍然领先于两者，拥有24467个模型，但是将模型从PyTorch迁移到JAX/Flax是一个持续进行的工作。

一个名为GPT-J-6B的开源大型类似GPT的模型，由EleutherAI开发，这个拥有60亿参数的Transformer语言模型是在Google Cloud上使用JAX进行训练的。作者表示这是开发大规模模型的正确工具组合。

目前，JAX可能不太适合在生产环境中部署，因为它主要侧重于研究方面，但这正是PyTorch发展的方式。研究和生产之间的差距很可能很快就会消除。Huggingface和GPT-J-6B案例已经在朝着正确的方向发展。

有趣的是，JAX与TensorFlow世界之间的互操作性随着时间的推移而越来越强。现在，JAX模型可以转换为TensorFlow的SavedModel，这样您就可以使用TensorFlow Serving、TFLite或TensorFlow.js进行部署。您甚至可以使用JAX编写Keras层。

考虑到Google的影响力和社区的迅速扩张，我期待JAX有一个光明的未来。

JAX不仅局限于深度学习。在JAX之上有许多令人兴奋的应用和库，涉及物理学领域，包括分子动力学、流体动力学、刚体模拟、量子计算、天体物理学、海洋建模等等。还有用于分布式矩阵分解、流式数据处理、蛋白质折叠和化学建模的库，而且不断有新的应用不断涌现。

让我们深入了解一下您可能想要使用的JAX功能。

### 1.2.1 计算性能

JAX提供了良好的计算性能。这部分涵盖了许多内容，包括使用现代硬件如TPU或GPU、XLA的JIT编译、自动向量化、跨集群的简单并行化，以及新的实验性xmap()命名轴(named-axis)编程模型，能够将您的程序从笔记本电脑CPU扩展到云中最大的TPU Pod。我们将在书的不同章节讨论所有这些主题。

您可以将JAX用作加速的NumPy，只需在程序开头将“import numpy as np”替换为“import jax.numpy as np”。在某种意义上，这是从NumPy切换到CuPy（用于使用NVIDIA CUDA或AMD ROCm平台的GPU）、Numba（同时具有JIT和GPU支持）或者PyTorch的替代选择，如果您希望对线性代数操作进行硬件加速。并非所有NumPy函数都在JAX中实现，并且有时您需要更多的替换导入之外的操作。我们将在第3章讨论这个话题。

通过GPU或TPU进行硬件加速可以加快矩阵乘法和其他可以从运行在这种大规模并行硬件上受益的操作的速度。对于这种加速类型，只需对放置在加速器内存中的多维数组进行计算即可。第3章将展示如何管理数据放置。

加速也可以来自于使用XLA编译器的JIT编译，它优化了计算图，并且可以将一系列操作融合为单个高效的计算或者消除一些冗余计算。这种加速甚至可以提高CPU上的性能，而无需任何其他硬件加速（尽管，现代CPU也有所不同，并且许多现代CPU提供了适用于深度学习应用的特殊指令）。

在图1.2中，您可以看到一个包含一些计算量的简单（并且相当无用）函数的截图，我们将使用纯NumPy、JAX使用CPU和GPU（在我的情况下是Tesla-P100）、以及JAX编译版本的相同函数在CPU和GPU后端上进行计算，比较它们的速度。我们将在第5章描述所有相关内容，并且深入探讨这个话题。相关的代码可以在书籍的代码存储库中找到：https://github.com/che-shr-cat/JAX-in-Action/blob/main/Chapter-1/JAX_in_Action_Chapter_1_JAX_speedup.ipynb。

图1.2 显示了用于比较NumPy和不同使用JAX方式的速度的代码。我们将评估计算f(x)所需的时间。

```
# a function with some amount of calculations
def f(x):
  y1 = x + x*x + 3
  y2 = x*x + x*x.T
  return y1*y2

# generate some random data
x = np.random.randn(3000, 3000).astype('float32')
jax_x_gpu = jax.device_put(jnp.array(x), jax.devices('gpu')[0])
jax_x_cpu = jax.device_put(jnp.array(x), jax.devices('cpu')[0])

# compile function to CPU and GPU backends with JAX
jax_f_cpu = jax.jit(f, backend='cpu')
jax_f_gpu = jax.jit(f, backend='gpu')

# warm-up
jax_f_cpu(jax_x_cpu)
jax_f_gpu(jax_x_gpu);
```


在图1.3中，我们比较了计算我们的函数的不同方式。

请暂时忽略诸如block_until_ready()或jax.device_put()之类的具体细节。前者是因为JAX使用异步调度，不会等待计算完成。在这种情况下，由于未计算所有的计算，测量结果会出错。后者是为了将数组粘附到特定的设备上。我们将在第3章讨论这些内容。

图1.3 显示了计算我们的函数f(x)的不同方式的时间测量。第一行是纯NumPy实现，第二行使用了JAX在CPU上，第三行是JAX JIT编译的CPU版本，第四行使用了JAX在GPU上，第五行是JAX JIT编译的GPU版本。

<a>![](/img/jaxinaction/ch1/2.png)</a>

在这个特定的例子中，经过CPU编译的JAX版本的函数几乎比纯NumPy原始版本快了五倍，然而非编译的JAX CPU版本略慢一些。您不需要编译函数就可以使用GPU，我们也可以忽略所有直接传输到GPU设备的操作，因为默认情况下，如果第一个GPU/TPU设备可用，所有数组都会在第一个GPU/TPU设备上创建。非编译的JAX函数仍然使用GPU，速度比经过JAX CPU编译的版本快5.6倍。而同一函数的GPU编译版本比非编译版本又快了2.9倍。与原始的NumPy函数相比，总的加速比接近77倍。在不改变函数代码太多的情况下，获得了相当不错的速度提升。

如果您的硬件资源和程序逻辑允许同时为多个项目执行计算，自动矢量化可以提供另一种加速计算的方法。

最后，您还可以在集群中并行化您的代码，并以分布式方式执行大规模计算，这是纯NumPy无法实现的，但可以通过类似Dask、DistArray、Legate NumPy等工具来实现。

当然，您也可以同时受益于所有提到的功能，这适用于训练大规模分布式神经网络、进行大规模物理模拟、执行分布式进化计算等场景。

EleutherAI的GPT-J-6B，这个拥有60亿个参数的变压器语言模型，是JAX上使用xmap()进行模型并行的一个很好的例子。作者表示，该项目“所需的人力小时数量远远少于其他大规模模型开发项目，这表明JAX + xmap + TPUs是快速开发大规模模型的正确工具组合。”

一些基准测试还显示JAX比TensorFlow更快。其他人表示，“JAX的性能在GPU和CPU上都非常有竞争力。它在两个平台上始终是顶级实现之一。”甚至在2020年，世界上最快的变压器也是用JAX构建的。



### 1.2.2 函数化(functional)方法

在JAX中，一切都是显而易见和明确的。由于采用了功能性方法，没有隐藏的变量和副作用（有关副作用的更多信息请参阅：https://ericnormand.me/podcast/what-are-side-effects），代码清晰明了，您可以随意更改任何内容，而且更容易偏离规则。正如我们之前提到的，研究人员喜欢JAX，并且很多新的研究都是使用JAX进行的。

这种方法需要您改变一些习惯。在PyTorch和TensorFlow中，代码通常是以类的形式组织的。您的神经网络是一个类，其中所有参数都是内部状态。您的优化器是另一个类，具有自己的内部状态。如果您使用强化学习，您的环境通常是另一个具有自己状态的类。您的深度学习程序看起来像是一个面向对象的程序，其中包含类实例和方法调用。

在JAX中，代码以函数的形式组织。您需要将任何内部状态作为函数参数传递，因此所有模型参数（神经网络中的权重集合，甚至是随机数的种子）都直接传递给函数。现在，生成随机数的函数需要您明确提供随机生成器状态（我们将在第7章深入讨论JAX的随机数生成器）。梯度是通过调用特殊函数（通过对您感兴趣的函数应用grad()转换来获取，这是第4章的主题）显式计算的。优化器状态和计算出的梯度也是优化器函数的参数。以此类推。

没有隐藏状态；一切都是可见的。副作用也不好，因为它们无法与JIT一起使用。

JAX还迫使您改变对if语句和for循环的习惯，因为它的编译方式。我们将在第5章讨论这个主题。

功能性方法还带来了丰富的组合性。由于所有先前提到的事物（自动微分、自动矢量化、使用XLA进行端到端编译、并行化）都是作为函数变换实现的，因此很容易将它们组合起来。

JAX丰富的组合性和表现力使其拥有强大的生态系统。

### 1.2.3 JAX生态系统

JAX为您构建神经网络提供了坚实的基础，但其真正的力量来自于不断壮大的生态系统。与其生态系统一起，JAX为您提供了一种令人兴奋的选择，作为当前两个最先进的深度学习框架——PyTorch和Tensorflow的替代品。

使用JAX，可以轻松地从不同模块组合解决方案。现在，您不必再使用包含所有内容的全能框架，如TensorFlow或PyTorch。这些是强大的框架，但有时很难用一个紧密集成的东西替换为另一个不同的东西。在JAX中，您可以通过组合所需的模块来构建自定义解决方案。

在JAX之前很久，深度学习领域就像是一种乐高式的东西，您可以使用不同形状和颜色的一组模块来创建解决方案。这就是乐高式的不同层次：从激活函数和层类型的底层到架构原语（如自注意力）、优化器、分词器等高级工程模块。

使用JAX，您有更多的自由度来组合最适合您需求的不同模块。您可以选择您想要的高级神经网络库，一个实现您想要使用的优化器的单独模块，自定义优化器以使不同层具有自定义学习率，使用您喜欢的PyTorch数据加载器，一个单独的强化学习库，添加蒙特卡洛树搜索，甚至使用其他库中的元学习优化器等等。

这非常类似于Unix的简单且模块化的设计方式，正如Douglas McIlroy所说：“编写只做一件事并且做得很好的程序。编写可以协同工作的程序。编写处理文本流的程序，因为这是一个通用的接口。”

嗯，除了文本流，这与JAX的哲学相关。然而，文本通信也可能成为未来不同（大型）神经网络之间的通用接口。谁知道呢。

于此同时，生态系统也随之出现。

生态系统已经非常庞大，而这只是一个开始。有优秀的高级神经网络编程模块（Google的Flax，DeepMind的Haiku等）、最先进优化器的模块（Optax）、强化学习库（DeepMind的RLax或Microsoft Research的Coax）、图神经网络库（Jraph）、分子动力学库（JAX, M.D.）等等。

JAX的生态系统已经包含了数百个模块，并且新的库不断涌现。我最近注意到的最新库包括用于进化计算的EvoJAX和Evosax、用于联邦学习的FedJAX、用于训练GPT-J-6B时使用的模型并行Transformer的Mesh Transformer JAX以及用于计算机视觉研究的Scenic库。但这远远不是全部。

Deepmind已经在JAX之上开发了一套库（其中一些已经在上面提到过）。还有新的用于蒙特卡洛树搜索、神经网络验证、图像处理等等的库。

Huggingface仓库中JAX模型的数量不断增长，而我们提到JAX根据这些数字是排名第二的框架。我们将把书的后半部分专门用于介绍JAX生态系统。

因此，JAX正在不断获得更多的动力，其生态系统也在不断增长。现在正是加入的绝佳时机！


## 1.3 JAX与TensorFlow/PyTorch有何不同？

我们已经讨论了JAX与NumPy的比较。现在让我们将JAX与两个现代深度学习框架PyTorch和TensorFlow进行比较。

我们提到，与PyTorch和TensorFlow常见的面向对象方法相比，JAX倡导函数式方法。这是当您开始使用JAX进行编程时面对的第一个非常明显的事情。它改变了您编写代码的结构方式，需要一些习惯的改变。同时，它为您提供了强大的函数转换功能，迫使您编写清晰的代码，并带来了丰富的组合性。

**[functorch](https://pytorch.org/functorch/stable/) is JAX-like composable function transforms for PyTorch.**

JAX的可组合函数转换对PyTorch产生了很大影响。在2022年3月，PyTorch 1.11发布时，其开发人员宣布了functorch（https://github.com/pytorch/functorch）库的beta版本，这是一个类似于JAX的可组合函数转换库。之所以这样做是因为许多用例在PyTorch中实现起来比较棘手，例如计算每个样本的梯度、在单台机器上运行模型集合、在元学习内循环中高效地批量处理任务、高效地计算Jacobian矩阵和Hessian矩阵及其批量版本。支持的转换列表仍然比JAX的转换要少。

另一个您很快会注意到的明显区别是，JAX非常简洁。它并没有实现所有功能。TensorFlow和PyTorch是两个最受欢迎和发展最完善的深度学习框架，几乎包含了所有可能的功能。与它们相比，JAX是一个非常简洁的框架，甚至很难称之为框架。它更像是一个库。

例如，JAX不提供任何数据加载器，因为其他库（例如PyTorch或TensorFlow）已经很好地实现了这一点。JAX的作者们并不想重新实现一切；他们想专注于核心功能。这正是您可以和应该将JAX与其他深度学习框架结合使用的地方。从PyTorch等框架中获取数据加载的内容并使用它是可以的。PyTorch有出色的数据加载器，所以让每个库发挥其优势。

另一个显著的区别是，JAX的原语非常底层，使用矩阵乘法编写大型神经网络可能会耗费时间。因此，您需要一种更高级的语言来指定这样的模型。JAX在开箱即用时并不提供这样的高级API（就像TensorFlow 1在TensorFlow 2添加高级Keras API之前的情况）。不提供这些功能并不是问题，因为JAX生态系统中有高级库可供使用。

您无需使用类似NumPy的原语编写神经网络。正如我们已经提到的，出色的高级库为您提供了所有必要的抽象，我们将把书的后半部分专门用于强大的JAX生态系统。

下图展示了PyTorch/TensorFlow与JAX之间的区别。由于JAX是一个可组合函数转换的可扩展系统，因此可以轻松地为每个功能构建单独的模块，并以任何您想要的方式进行混合。

<a>![](/img/jaxinaction/ch1/3.png)</a>

## 1.4 总结

* JAX是来自谷歌的低级Python库，广泛用于机器学习研究（但也可以用于其他领域，如物理模拟和数值优化）
* JAX为其多维数组和数学函数提供了与NumPy兼容的API
* JAX拥有强大的函数转换集合，包括自动微分（autodiff）、jit编译、自动矢量化和并行化，这些功能可以任意组合
* JAX通过利用现代硬件（如TPU或GPU）、XLA的JIT编译、自动矢量化和跨集群的轻松并行化，提供了良好的计算性能
* JAX采用函数式编程范式，并要求使用无副作用的纯函数
* JAX具有不断增长的模块化生态系统，您可以自由地组合不同的功能块，以最适合您需求的方式
* 与TensorFlow和PyTorch相比，JAX是一个相当简洁的框架，但由于其不断增长的生态系统，有许多优秀的库可以满足您特定的需求








