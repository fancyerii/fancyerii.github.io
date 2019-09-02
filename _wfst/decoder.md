---
layout:     post
title:      "基于WFST的语音识别解码器算法" 
author:     "lili" 
mathjax:   true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - 解码器
    - WFST
---



本文介绍基于WFST的语音识别解码器算法，主要是静态的解码算法。

<!--more-->

**目录**
* TOC
{:toc}


本文介绍把WFST用于语音识别的更多细节内容。首先我们简要的介绍基于WFST的语音识别系统，然后解释语音识别系统的不同模块怎么用WFST来表示以及怎么把这些WFST组织成单一的搜索网络。最后我们介绍使用完全复合后的WFST来进行识别的时间同步Viterbi Beam搜索算法。

## 基于WFST的解码器的概览
 

WFST提供里一个统一的形式来表示当前SOTA的大规模连续语音识别(LVCSR)系统的不同知识源(knowledge source)，比如HMM、声学模型、发音词典和N-gram语言模型。表示不同知识源的多个WFST可以通过复合运算整合成一个WFST，这个WFST表示的搜索网络的输入是HMM状态。然后这个WFST可以通过前面介绍的各种优化运算来去掉其中的冗余部分而变成等价的但是更加紧凑高效的WFST来加速解码过程。这里我们来简单的介绍一下基于WFST的语音识别系统的基本原理。


如前文介绍的，连续语音识别定义为则给定的输入语音信号O的条件下寻找最可能的词序列$\hat{W}$。似然可以使用贝叶斯公式变成$P(O \vert W)P(W)$，其中$P(W)$是语音模型的概率，而$P(O \vert W)$是声学模型概率。更具体的，通过发音词典引入发音概率$P(V \vert W)$，它表示给定词序列W的条件下phone序列V的概率。这3个概率分别通过声学模型、发音模型(词典)和语言模型来计算。因此前面的公式可以重写为：

$$
\begin{split}
\hat{W} &= \underset{W \in \mathcal{W}}{argmax} \sum_{V \in R(W)} P(O|V,W)P(V|W)P(W) \\
& \approx \underset{W \in \mathcal{W}}{argmax} \left \{ \sum_{V \in R(W)} P(O|V)P(V|W)P(W) \right \}
\end{split}
$$


这里的$P(O \vert V)$、$P(V \vert W)$和$P(W)$分别通过声学模型、发音词典和语言模型来计算。W是所有可能的词序列而R(W)是词序列W的所有可能发音的phone的序列。因为目前SOTA的LVCSR系统都使用子词单元的声学模型，因此声学似然$P(O \vert V,W)$被假设只依赖于phone的序列从而近似为$P(O \vert V)$。


为了实现方便，则Viterbi解码器里我们使用log运算来替代乘法。解码器会进行如下的搜索：

$$
\begin{split}
\hat{W} & \approx \underset{W \in \mathcal{W}}{argmax} \left \{\sum_{V \in R(W)} P(O|V)P(V|W)P(W) \right \} \\
& = \underset{W \in \mathcal{W}}{argmax} \left \{\max_{V \in R(W)} log P(O|V) + logP(V|W)+logP(W) \right \}
\end{split}
$$


在上面的公式里，因为是Viterbi近似，所以把求和变成了求最大值。为了简单，后面我们把对数似然和对数概率称为得分。$log P(O \vert V)$称为声学得分，$log P(V \vert W)$称为发音得分，$logP(W)$称为语言得分。WFST框架为我们提供里一种快速计算上式的方法。
 
在WFST框架里，语音识别问题被当作把输入语音信号O转换成词序列W的一个转换(transduction)过程。语音识别系统中的每一个模型都被解释为一个WFST，而模型的得分的取反(-log概率)被作为WFST的weight。对于上式的声学模型、发音词典和语言模型，我们分别定义H、L和G这3个WFST来表示它们。其中H把观察序列转换成phone序列V，其中$w_H(O \rightarrow V)=-logP(O \vert V)$；L把phone序列V转换成词序列W，其中$w_L(V \rightarrow W)=-logP(V \vert W)$；最后G把词序列W还是转换成W(G可以看成一个WFSA，它的作用是计算词序列W的概率，如果不接受可以认为概率是0)，其中$w_G(W \rightarrow W)=-logP(W)$。然后从观察序列O到词序列W的变换就是先后通过H、L和G计算得到。为了计算效率，我们可以提前把H、L和G使用复合运算组合成一个WFST来直接把观察序列O转换成词序列W：

$$
N=H \circ L \circ G
$$


上式中$\circ$是复合运算。因此，语音识别的过程被变成在N上搜索最小weight(因为是-log概率)的词序列：

$$
\begin{split}
\hat{W} & \approx \underset{W \in \mathcal{W}}{argmax} \left \{\max_{V \in R(W)} log P(O|V) + logP(V|W)+logP(W) \right \} \\
& = \underset{W \in \mathcal{W}}{argmax} \left \{\min_{V \in R(W)} (-log P(O|V)) + (-logP(V|W))+(-logP(W)) \right \} \\
& = \underset{W \in \mathcal{W}}{argmax} \left \{\min_{V \in R(W)} w_H(O \rightarrow V) \bigotimes w_L(V \rightarrow W) \bigotimes w_G(W \rightarrow W) \right \} \\
& = \underset{W \in \mathcal{W}}{argmax} \; w_N(O \rightarrow W)
\end{split}
$$

上式我们假设weight是定义在热带半环上的，所以+就是$\bigotimes$。

当前SOTA的系统都是triphone的模型，为了实现triphone，我们加入一个额外的WFST C，这个WFST的作用是把triphone序列转换成phone序列：

$$
N=H \circ C \circ L \circ G
$$
 

上面复合得到一个包含所有模型的搜索网络，其中跨词(cross-word)的triphone模型被完美的融入其中，这对于非WFST的传统解码方法来说是非常难以处理的。复合后的WFST可以使用前面介绍的各种WFST优化运算，比如确定化和最小化来进一步优化。这些优化运算可以在整个搜索网络(包含声学模型、发音词典和语言模型)上来进行优化，而传统的解码器优化通常被局限于某一个模型上。


只要复合的WFST构建完毕，解码器的工作就是对于给定的语音输入搜索最优的路径。如果模型没有发生变化，则WFST不需要更新。因此，WFST的解码器可以专注于使用优化的静态搜索网络来搜索最优路径；而传统的解码器通常是则非全局最优的搜索网络上进行搜索，而且因为内存等限制只能构造部分网络，在解码时还需要动态的扩展搜索网络。这是WFST的框架相对于传统方法的最重要的优点。此外，因为解码器被设计得可以处理任何WFST，所以解码器程序于具体的用WFST来表示的模型是无关的。这也是WFST框架的另一个好处——它让解码器更加通用和易于维护。
 

在后续的内容里，我们会介绍怎么构建语音识别系统中的各个模型的WFST，用于构建完整的复合WFST的一些常见复合和优化步骤以及使用WFST来解码的算法。如前面所述，基于WFST的语音识别系统把语音识别问题看成一个从输入语音观察序列到词序列的一个转换问题。但是我们需要注意：语音识别的转换问题其实是超出里WFST的定义。对于传统的语音识别系统，语音信号被转换成实数值的特征向量的序列。实数是连续的无穷的，因为WFST要求输入符号是有限的符号集合，所以无法用一个WFST把特征向量序列变成HMM状态序列。此外，给定一个HMM状态，计算观察向量的概率也是需要使用GMM模型on-the-fly的来计算，因此即使我们的WFST可以输入实数，它的weight和输入的关系也不是就简单对应关系，它需要根据概率密度函数公式来计算。因此，基于WFST的解码器会分成两部分，一部分处理连续的输入特征向量序列，而另一部分处理WFST。读者在后面的介绍里会更加清楚这一点。

## 语音识别系统各个部件(模型)的WFST构建


在本节，我们会介绍怎么用WFST来表示语音识别系统的不同部件，包括声学模型、phone上下文、发音词典和语言模型。具体来讲，我们着重关注语音识别中的标准模型，比如HMM模型、triphone上下文、简单的非概率的发音词典和n-gram语言模型。

### 声学模型(H)


在基于WFST的方法里，一些声学模型可以被看成一个转换机(transducer)，它把输入语音信号转化成一个(上下文相关的)phone序列，同时weight表示声学似然。下图是一个"假想"的上下文相关的HMM转换机。


<a name='img38'>![](/img/wfstbook/38.png)</a>
*图：HMM转换机*
 

上图的WFST里，包含了3个上下文相关的phone——s(t)、(s)s和(s)s(t)。其中(s)s(t)表示中心的phone是s，它的左边是s右边是t。而s(t)表示s的右边是t(没有左边的phone，因此它是一个开始的phone)，(s)s表示s的左边是s，而右边没有phone。这3个上下文相关的phone都是从左到右的3状态的HMM，每个状态可以跳转到自己，也可以跳转到下一个状态。x是一个特殊的符号，它代表任意的输入特征向量。"x : s(t)/w(x\|S0)"代表对于任意的输入特征向量x，WFST都可以从状态0跳转到状态1，并且输出是s(t)，weight是函数x(x\|S0)，这里的"S0"表示第0个共享的状态。所有共享的状态的发射概率$b_{S_k}(x)$都是一样的。对于热带半环或者log半环，w(x\|Sk)等于$-logb_{S_k}(x)$。除了从状态0开始的跳转，其它的跳转的输出符号都是ε，weight除了发射概率还包含状态的跳转概率(-log)。比如状态1的自跳转的weight为0.22 ⊗ w(x\|S0)，这里的0.22是状态自跳转的-log概率，因此真正的自跳转概率是0.8$(e^{-0.22}=0.8)$。⊗代表热点半环或者log半环上的乘法，因此从状态1到状态1的自跳转weight包含自跳转和发射概率的"相乘"。因此这个WFST的输入是声学特征，输出是上下文相关的phone。注意，这个WFST不考虑怎么把上下文相关的phone变成上下文无关的phone，那是后面介绍的C需要考虑的内容。



另外我们看一下哪些是共享的状态。比如(s)s和(s)s(t)的第一个状态(4和7)是共享的，因为它们的第一个状态都表示s的左边上下文是s，因此状态4和状态7的发射概率是相同的(但是跳转概率是不同的)，在图中都用w(x\|S4)表示。类似的，s(t)和(s)s(t)的最后一个状态也是共享的，因为它们的右边上下文是t。

但是这里的转换机并不是真正的WFST，因此不能直接在WFST的框架里用标准的WFST来表示H。原因在于WFST要求输入是有限的离散符号，但是这里的输入是连续的无穷的实数值的向量。因此这里我们需要引入一个特殊的符号x和一个weight函数$w(x \vert S_k)$。为了解决这个问题，H被分解为两个部分——HMM拓扑结构和声学匹配(acoustic matching)。前者可以使用标准的WFST来处理，而后者是用解码器程序里的特殊代码来处理。下图是上图的H的一种分解方式。


<a name='img39'>![](/img/wfstbook/39.png)</a>
*图：H的一种分解方式*




在这种分解方式里，代表拓扑结构的WFST编码了HMM的状态和状态的跳转概率，而声学匹配部分处理特殊的符号x以及发射概率$w(x \vert S_k)$，它们的复合就得到了前面的H。读者可以自行验证一下它们的复合确实是H，从而更好的理解分解。


除此之外还有别的分解方式，比如下图。


<a name='img40'>![](/img/wfstbook/40.png)</a>
*图：H的第二种分解方式*


在这种分解里，HMM拓扑结构只编码里每个phone的3个状态的顺序关系。状态转移概率都编码到声学匹配的WFST里了。比如我们看左边，它可以把输入符号序列"S0,S1,S3"转换成"s(t)"。而右边我们看到S0->S1的跳转得分1.6被编码到状态1跳回到初始状态0的ε跳转里了，因为在左边的拓扑结构里S0后面(这里是只可以)可以是S1。


<a name='img41'>![](/img/wfstbook/41.png)</a>
*图：H的第三种分解方式*

这种分解里，左边的拓扑结构更加简单，它的输入是一个符号(比如"S0,S1,S3"我们应该看成一个符号而不是符号序列)，输出是phone。同一个phone的所有的状态跳转顺序和概率都编码则右边的声学匹配WFST里了。
 

这三种分解方式的分解需要解码器则通用性(generality)和效率(efficiency)之间进行权衡。后两种的效率会更高一些，因为它们的拓扑结构WFST更加简单，而声学部分是由特定的程序实现，这些特定程序会比通用的WFST更加高效。但是第一种方式更加通用，它可以处理任何类型的HMM拓扑结构(只要拓扑结构可以用WFST来表示)。而后两种方式如果要处理新的HMM拓扑结构，则解码器的代码就要作相应的修改来适应这种特定的HMM拓扑结构。
 

把H分解后，拓扑结构会和C、L和G进行复合，因此我们后面的复合里的H指的是拓扑结构的H，它的输入是HMM共享状态的ID序列，输出是上下文相关的phone序列。因此$H \circ C \circ L \circ G$的输入也是共享状态的ID序列。在有的实现里，H可能是<a href='#img40'>上图(a)</a>的拓扑结构的WFST，然后链式(一个状态只能跳到另外一个状态)的跳转被合并成类似<a href='#img41'>上图(a)</a>输入，比如H、C、L和G复合后"S0,S1,S3"是一个链，为了效率更高，我们可以把它们合并成一个符号，这个过程后面会介绍到。而则另外一些实现里，H完全由程序来处理，只有$\text{CLG}=C \circ L \circ G$是提前构造和优化好的，然后由程序来实现H与CLG的复合。

### 上下文相关处理的FST(C)


前面我们介绍过，在大部分LVCSR系统里都是要上下文相关的phone单元来作为子词的声学建模。这些子词单元通过它们的上下文依赖在搜索网络中被连接起来。而处理这些上下文相关的phone的连接的FST就是本节要介绍的C。


如果上下文只依赖于左右各一个phone(也就是常见的triphone)，则C并不难构造。任何两个phone的pair都会作为一个状态，而每个triphone都会作为一条边。边的起点状态的phone pair必须匹配triphone的左边上下文和中心的phone，而终点状态的phone pair必须匹配triphone的中心phone和右边上下文。


下图是一个C的简单示例，它表示只有两个基本phone /t/和/s/的请看。每个跳转的输入是triphone，输出是上下文无关的triphone的中心phone。但是这里初始状态0是没有左边上下文而终止状态1是没有右边上下文的，因此这不是跨词(cross-word)的triphone。

<a name='img42'>![](/img/wfstbook/42.png)</a>
*图：triphone的转换机C的示例*

我们可以用一个例子来验证这个FST确实可以把一个上下文相关的triphone序列转换成上下文无关的phone序列。我们先反过来，假设要输出的phone序列是"s,s,t,s,t"，则它对应的triphone序列为"s(s)、(s)s(t)、(s)t(s)、(t)s(t)、(s)t"。下面我们来验证这个FST确实可以把triphone序列转换成phone序列。

完成这个转换的状态序列为"0->2->3->4->3->1"，请读者验证这条路径确实把上面的triphone序列变成里phone序列。

读者可能会问这个C是怎么构造出来的呢？本文没有介绍，感兴趣的读者可以参考[Speech Recognition with Weighted Finite State Transducers](https://cs.nyu.edu/~mohri/pub/hbka.pdf)和[Investigations on Search Methods for Speech Recognition using Weighted Finite-State Transducers](https://pdfs.semanticscholar.org/aedf/f1aeba8c2e825bed98ad68ccb5ddefb5ad6e.pdf)。

**未完待续**
