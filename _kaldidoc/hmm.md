---
layout:     post
title:      "HMM拓扑结构和跳转建模"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[HMM topology and transition modeling](http://kaldi-asr.org/doc/hmm.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}


## Introduction
 
本文介绍Kaldi里HMM的拓扑结构(topology)是怎么表示的以及HMM跳转(transition)的训练。这里会提及它怎么和决策树进行交互的；更多决策树的细节可以参考[How decision trees are used in Kaldi](http://kaldi-asr.org/doc/tree_externals.html)和[Decision tree internals](http://kaldi-asr.org/doc/tree_internals.html)。相关的类和函数列表可以在[Classes and functions related to HMM topology and transition modeling](http://kaldi-asr.org/doc/group__hmm__group.html)找到。

## HMM拓扑结构
 
用户可以使用工具类[HmmTopology](http://kaldi-asr.org/doc/classkaldi_1_1HmmTopology.html)来指定phone的HMM的拓扑结构。在普通的recipe里，我们创建一个文件，这个文件是HmmTopology对象的文本形式，然后在命令行程序里我们指定这个文件。为了了解这个对象的内容，下面给出一个HmmTopology对象的文本格式(这是一个3状态的Bakis模型，不了解Bakis模型的读者可以参考[基于HMM的语音识别(一)](/books/asr-hmm/#%E5%AE%9A%E4%B9%89))。

 

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
 

这是HmmTopology对象的一个TopologyEntry，它描述了ID从1到8的8个phone的HMM的拓扑结构(这个8个phone的HMM结构完全相同)。这里有3个发射状态(有概率密度分布pdf与之关联并且进入这个状态后会"发射"出特征向量)，每个状态都有自跳转和一个到下一个状态的跳转。此外还有3个非发射的状态3(没有\<PdfClass>)，没有从它出发的跳转(系统会把它连接到序列的下一个phone)。Kaldi会把第一个状态(这里是状态0)作为开始状态，而最后一个状态应该总是非发射的并且没有跳出的边。你可以把跳入最后一个终止状态的跳转概率当成HMM终止的概率。在这个特定的例子里，所有的3个发射状态的pdf都是不相同/共享的(因为PdfClass不同)。我们可以通过设置相同的\<PdfClass>来强占状态的pdf共享。这里给定的跳转概率只是用来则训练开始时初始化这个phone的HMM的跳转概率；训练完成后不同上下文相关HMM的跳转概率都是不同的，这些跳转概率会存在TransitionModel对象里。TransitionModel对象也会把HmmTopology对象作为它的一个成员变量保存下来，但是要记得HmmTopology里的跳转概率在初始化后通常就没有任何作用里。但是有一个例外：对于非发射的非终止状态(有跳出的边\<Transition>但是没有\<PdfClass>)，Kaldi不会这些特殊状态的跳转概率，而是直接使用HmmTopology对象里用户指定的跳转概率。这个做法是为了简化训练过程，而且通常情况下很少出现这种非发射的非终止状态，所以我们觉得这样做也没有什么问题。

## pdf-class

 

pdf-class是和HmmTopology对象相关的一个概念。HmmTopology对象指定(如前例所示一般是批量指定)每一个phone的HMM的结构。这个"原型"的HMM的每一个状态都有两个变量：forward_pdf_class和self_loop_pdf_class。self_loop_pdf_class是自跳转对应的pdf。默认情况下它等于forward_pdf_class，但是则某些特殊情况下这两者可以不同。【这是Kaldi和其它一些实现不同的地方，我们在论文里通常认为一个状态对应一个pdf(当然可以share)，不管它是自跳转到这个状态还是从别的状态跳转到这个状态，但是Kaldi里可以区分这两种情况。】让同一个状态的自跳转pdf不同于非自跳转，而不是完全的"基于跳转"的表示，这是一种折衷的选择，这是为了兼容"chain模型"(也就是lattice-free MMI)的拓扑结构。如果两个状态的pdf_class相同，则如果它们的上下文相同的话，那么它们总是共享相同的pdf。之所以这样的原因是决策树"看到"的不是HMM的状态，而是pdf-class。通常情况下pdf-class和HMM状态的下标(比如0,1,2)是相同的，但是pdf-class提供里一种强制共享的方式。这在这种常见非常有用：你想要更加灵活的跳转模型，但是想让声学模型一样。pdf-class的另一个作用是指定非发射的状态。如果某个HMM状态的pdf-class的值是常量kNoPdf(-1)，则这个状态是非发射的(没有关联的pdf)。这可以简单的则定义状态(\<State>)时省略\<PdfClass>就行，因为默认的pdf-class就是kNoPdf。
 
对于一个HMM原型(一个TopologyEntry)，pdf-class必须要从0开始并且连续(0,1,2,...)。这只是为了代码实现的简单，并且它也不会损失什么功能。

## 跳转模型(TransitionModel对象)
 
TransitionModel保存一个phone的跳转概率和HMM拓扑结构(它包含HmmTopology对象)。图的构建代码依赖于TransitionModel来获得拓扑结构与跳转概率(它还依赖于ContextDependencyInterface来把triphone变成pdf-id)。

### Kaldi里怎么表示跳转概率
 
很多跳转模型的代码都依赖于下面的决策：我们让一个上下文相关的HMM状态的跳转概率由下面的5个值确定：

* phone(哪个phone的HMM)
* 起点的HMM状态(在HmmTopology对象里解释过，通常是0,1,2)
* forward-pdf-id(这个状态关联的forward pdf的索引)
* self-loop-pdf-id(这个状态关联的self-loop pdf的索引)
* HmmTopology对象里跳转的下标
 
最后4个值用于确定这个跳转的终点状态。这么做的原因这是在不增加解码图大小的情况下最细粒度的跳转建模方式里。实际上，传统的方法不会这么细粒度的来建模跳转，比如HTK那种只是则monophone层次共享跳转就足够了。


### transition-id


TransitionModel则初始化的时候会创建一系列整数的mapping，其它部分的代码会用到这些mapping。除了上面提到的量之外，还包括transition-id、transition index(不同于transition id)和transition状态。我们引入这些id和mapping的原因是我们可以有一个完整的基于FST的训练recipe。基于FST的方法里最自然的是用pdf-id作为输入符号。但是由于决策树的存在，从pdf-id到一个phone的mapping并不总是唯一的，这会导致从输入符号序列到phone序列变大困难，从而出现一些麻烦的地方；此外它也会只使用FST的信息训练跳转概率变得困难。因此我们的FST的输入符号是transition-id，它除了可以映射成pdf-id之外还包含phone以及HMM里的特定跳转的信息。

### TransitionModel使用到的整数id
 
下面是TransitionModel接口里用到的一下id。它们的类型都是int32。注意它们中有些是从1开始的而另外一些是0开始的。为了和C++的数组兼容，我们会尽量避免1开始的下标，但是OpenFst把0当成特殊的$\epsilon$，所以我们把频繁作为FST的输入符号的id以1为开始下标。最重要的一点：transition-id是1开始的。因为我们觉得pdf-id不会经常被用作FST的输入，所以我们让它从0开始，如果某些特殊场景它被用作FST的输入，我们会对它加一。当阅读TransitionModel的代码时，当使用基于1开始的量来作为数组的索引是，某些地方我们需要对它减一而某些地方不需要；这通常在变量定义的地方会有文档说明。无论如何，这些都不是公开的接口，因此也不会导致太多的混淆。TransitionModel用到的所有整数id如下：

* phone(1开始)
    * 在Kaldi里广泛被使用；可以使用OpenFst的符号表把它转换成字符串。不要求连续。
* hmm状态(0开始)
    * HmmTopology::TopologyEntry里定义的HMM的状态ID，通常phone是3状态的，因此它的取值范围是{0, 1, 2}
* pdf-id(0开始)
    * 由决策树聚类产生的pdf的id。对于一个语音识别系统来说通常会有几千个pdf
* transition 状态(1开始)
    * 这是TransitionModel定义的id。每一个三元组(phone, hmm状态, pdf-id)定义为一个唯一的transition状态。我们可以把它想像成最细粒度的HMM状态。
* transition index(0开始)
    * 这是HmmTopology::HmmState里索引transitions的下标。每一个状态都会定义从它出发的跳转，这个下标就是索引这个跳转数组。
* transition-id(1开始)
    * 每一个都代表跳转模型里的一个唯一的跳转概率。它和(transition状态, transition index)一一对应。

除此之外还有下面两个与跳转模型有关的概念：

* 四元组(phone, hmm状态, forward pdf, self-loop pdf)存在于transition状态的一一对应关系。

* (transition状态, transition index)和transition-id之间存在一一对应。
 

## 跳转模型的训练
 

跳转模型的训练非常简单。我们创建的FST(包括训练和测试用的)输入符号是transition-id。在训练的时候我们使用Viterbi解码得到一个输入符号序列，它就是一个transition-id的序列(每一帧一个transition-id)。用来累计训练跳转的统计量只是每个transition-id出现的次数(代码虽然使用浮点数但其值只是整数)。函数Transition::Update(最新的代码叫做MleUpdate)根据统计对每个transition状态进行最大似然的参数估计。这个过程非常"直白"。当然里面有一些浮点数flooring的问题以及对于没有出现过的transition状态的处理，更多细节参考代码。

## Kaldi里的对齐

At this point we introduce the concept of an alignment. By "alignment", we generally mean something of type vector<int32>, which contains a sequence of transition-ids (c.f. Integer identifiers used by TransitionModel) whose length is the same as the utterance the alignment corresponds to. This sequence of transition-ids would generally be obtained from the decoder as the input-label sequence. Alignments are used in training time for Viterbi training, and in test time for adaptation. Because transition-ids encode the phone information, it is possible to work out the phonetic sequence from an alignment (c.f. SplitToPhones() and ali-to-phones.cc).

We often need to deal with collections of alignments indexed by utterance. To do this conveniently, we read and write alignments with tables; see I/O with alignments for more information.

The function ConvertAlignment() (c.f. the command-line program convert-ali) converts alignments from one transition-model to another. The typical case is where you have alignments created using one transition-model (created from a particular decision tree) and want to convert them to be valid for another transition model with a different tree. It optionally takes a mapping from the original phones to a new phone set; this feature is not normally needed but we have used it when dealing with simplified models based on a reduced (clustered) phone set.

Programs that read in alignments generally have the suffix "-ali".
