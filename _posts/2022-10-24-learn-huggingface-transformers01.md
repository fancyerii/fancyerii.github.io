---
layout:     post
title:      "Huggingface Transformers学习(一)——Transformers和Hugging Face Transformers简介" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - Huggingface
    - Transformer
    - 深度学习
---

本系列课程记录学习Huggingface Transformers的过程，主要参考了[官方教程](https://huggingface.co/course/chapter1/1)和[Natural Language Processing with Transformers: Building Language Applications with Hugging Face](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246)。

<!--more-->

**目录**
* TOC
{:toc}
 
## Transformer简介
随着2017年Google在[Attention Is All You Need](https://arxiv.org/abs/1706.03762)这篇论文提出Transformer来替代LSTM等RNN模型，Elmo、GPT和BERT等大规模自监督预训练模型以及这些大模型微调方法的出现，整个NLP领域都进入了新的范式。学术界提出了各种各样的大规模预训练模型，不断的刷新最好的成绩。下图是截止到2021年的一些主要模型，而在学术界每过多久就有更多的模型被提出来。

<a name='img1'>![](/img/learnhuggingface/1.png)</a>


这次NLP范式的变迁主要涉及如下三个技术：

* 编码器-解码器框架
* 注意力机制和Transformer
* 自监督预训练和迁移学习

### 编码器-解码器框架
在Transformer出现之前，LSTM这样的RNN模型是当时最主流的模型。RNN模型的特点是前一个时刻的隐状态会作为当前时刻的额外输入，因此可以认为RNN有一点的”记忆“功能。这样的特点使得RNN非常适合建模文本这样的序列数据。如下图所示，RNN除了从下到上的输入输出信息流，还会有从左到右的“记忆”信息的流动。

<a name='img2'>![](/img/learnhuggingface/2.png)</a>

如果把RNN按照时间展开，我们可以看到最后一个时刻的输出是依赖于它之前所以时刻的输入的。这样的一种方式使得RNN有很强的时序建模能力，更多信息请参考[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。

RNN的一个很重要应用就是叫做序列到序列(Seq2Seq)的一类问题，比如机器翻译、文本摘要和生成式问答都可以用Seq2Seq来解决。我们通常会用一个RNN来处理输入序列，这个过程通常叫做编码(encode)，所以这个RNN也叫做编码器(encoder)。RNN最后一个时刻的隐状态被认为"编码"了这个句子最重要的一些信息，然后被送入到另外一个RNN作为其初始隐状态。而第二个RNN被叫做解码器(decoder)，它采用自回归式的方式根据输入的初始化隐状态一步一步自回归的"解码"出输出序列。

<a name='img3'>![](/img/learnhuggingface/3.png)</a>

encoder-decoder框架非常灵活，因为它的输出和输入没有任何长度和顺序的约束关系。但是由于需要把输入的所有重要信息都编码到最后一个时刻的隐状态，所以会存在所谓的"信息瓶颈"的问题——也就是很难把所有的信息都塞到最后一个向量里。

幸运的是，我们可以通过一种机制让decoder可以访问encoder的所有隐状态而不是最后一个时刻的隐状态，这种机制就是注意力机制。

### 注意力机制和Transformer

注意力机制的核心思想是：encoder的每个时刻的隐状态都可以被decoder访问到，但是encoder的信息量很大，所以需要使用所谓的”注意力“来确定decoder的某个时刻到底用多大的”注意力“来加权encoder的所以隐状态。注意力机制的过程如下图所示：

<a name='img4'>![](/img/learnhuggingface/4.png)</a>

图中在计算第二个时刻的输出时会通过注意力机制确定输入的每个时刻的权重，也就是所谓的注意力。通过这种注意力，其实输出和输入就建立了一种软的(soft)对齐(alignment)。比如下图是机器翻译两个句子的注意力矩阵：

<a name='img5'>![](/img/learnhuggingface/5.png)</a>

这是一个英-法翻译的例子，我们可以看到输出的法语zone时的注意力主要集中在英文area上，所以可以认为模型学习到了这两个词的对齐关系。这种注意力的软对齐的好处是：它是模型自动学习到的依赖于上下文的对齐；可以容易的实现一对多和多对一以及乱序的对齐。

虽然基于注意力机制的编码器-解码器模型非常强大，但是它有一个很大的问题：RNN编码时的顺序计算效率底下，无法很好的利用GPU这样的资源进行并行计算。为了解决这个问题，Transformer模型把编码器和解码器都改成了自注意力机制(解码器的自注意力机制需要Mask掉未来信息)。自注意力机制可以替代RNN的作用，它在编码当前token的时候通过自注意力机制参考整个句子的其它token，从而能够有更强大的上下文表达能力。同时自注意力机制的计算就是矩阵和矩阵的乘法，完全可以并行计算。下图是编码器使用自注意力机制的图示：

<a name='img6'>![](/img/learnhuggingface/6.png)</a>

上图中编码器在编码are这个token时会同时参考其它token(当然也包括它自己)的上一层的隐状态。

有了计算效率更高的Transformer模型，最后一个问题就是怎么获取足够的训练数据来训练它！

### 自监督预训练和迁移学习

Transformer模型最初在机器翻译上大获成功，但是在NLP其它大部分任务上，训练数据都要比机器翻译少好几个数量级。如果训练数据不够的情况下，模型表示能力越强反而更容易过拟合。但是每个NLP任务都标注到千万级别的训练数据基本是不可能的，那怎么办呢？

在计算机视觉方向，大家通常会用很大规模的数据集(比如ImageNet)训练一个模型(比如ResNet)，然后在其它相关的任务上用少量数据进行fine-tuning模型，这就是一种常见的迁移学习方法。但是在NLP方向，这个思路一直没有应用。主要原因就是NLP的任务很多样，但是每个任务的训练数据都不是那么多。不过未标注的文本是非常多而且是可以很容易获取的，所以有人想到了用未标注的文本通过自监督的任务(主要就是语言模型)来预训练一个大模型，然后针对下游任务进行fine-tuning的思路。当然如果领域中也有不少未标注数据，那么也可以在预训练和fine-tuning中间加一个领域自适应的预训练环节。下图就是较早期的一个ULMFiT模型的流程图：

<a name='img7'>![](/img/learnhuggingface/7.png)</a>

首先是用海量的未标注文本(wiki)训练一个语言模型，之后再用领域的未标注数据(imdb)继续训练语言模型做领域自适应，最后用少量领域的标注数据微调模型。预训练模型是一个语言模型，通常可以是RNN或者Transformer的Encoder，模型的输出是下一个时刻的token，通过交叉熵损失就可以训练它。而微调的模型通常是一个分类的模型，我们通常把预训练语言模型的最后一层去掉，换成随机初始化的新的输出层，然后训练领域分类器。


GPT和BERT是自监督预训练模型的典型代表，它们都是用Transformer作为Encoder来自监督的预训练。差别在于GPT使用的是传统的语言模型，所以编码时需要Mask掉后面的文本。而BERT使用的是Masked语言模型，比如：
```
I looked at my [MASK] and saw that [MASK] was late.
```
对于一个句子，BERT预训练时会随机MASK掉一些Token让模型来预测，从而让模型学习到丰富的上下文语义。

## Hugging Face Transformers

在GPT和BERT大获成功之后，学术界不断涌现各种各样的预训练模型，它们大都是基于Transformer架构的。不同的机构的作者使用了不同的框架(Tensorflow和Pytorch是最主要的框架)，不同的实现方式。这样就导致一个问题，不同的预训练模型很难比较，而且使用起来也很复杂。

应用Transformer预训练模型来解决一个问题通常需要如下步骤：

* 用Tensorflow或者Pytorch等框架编码实现模型。
* 加载预训练模型
* 文本预处理、模型预测和文本后处理
* 实现dataloader加载数据，定义损失函数和优化器以训练模型

对于每个模型和每个特定任务，都需要特定的代码逻辑实现上面的所有步骤。研究机构在发布一篇论文时通常会公开相关的源代码和训练好的模型参数，但是每篇文章的实现方法都不相同。要把这些学术界最新的模型用到生产环境需要大量的工作。为了解决这些问题，Hugging Face(🤗)开源了Transformers库及相关库，致力于网罗常见的Transformer模型。让我们用极少量的代码和标准的接口来调用(几乎)所有常见的基于Transformer的预训练模型，它同时支持Tensorflow、Pytorch和Jax三个主流框架。并且在线的Hub提供海量的预训练模型下载。让我们更加轻松的用上最先进的Transformer模型。

下面我们体验一些Transformers库来完成一些NLP任务是多么简单。首先我们准备一段文本：

```
text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
```
### 搭建环境

有很多方法运行Hugging face Transformers，最简单的方法是在docker环境里运行，请参考[用Docker、Jupyter notebook和VSCode搭建深度学习开发环境](2022/10/19/docker-jupyter/)设置开发调试环境。

### 文本分类

Transformers库提供了多层次的API，其中最简单的一种就是pipeline。我们用如下代码初始化一个特定任务的pipeline：

```
from transformers import pipeline
classifier = pipeline("text-classification")
```

上面的代码构造了一个"text-classification"的pipeline，第一次运行的时候它会去Hugging face的Hub下载模型，因为没有制定模型，"text-classification"任务默认下载的是distilbert-base-uncased-finetuned-sst-2-english模型。下载后会缓存起来，第二次运行就不需要下载模型了。

有了pipeline的模型我们就可以进行预测了：

```
import pandas as pd
outputs = classifier(text)
pd.DataFrame(outputs)
```
结果为：


```
 	label 	          score
0 	NEGATIVE 	0.901546
```

也就是模型对这段文本的情感分类的是负面的，并且置信度非常高。


### 命名实体识别

类似的，我们可以用pipeline实现命名实体识别：


```
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)
```

运行结果为：
```
  entity_group 	  score 	word 	       start 	end
0 	ORG 	0.879011 	Amazon 	        5 	11
1 	MISC 	0.990859 	Optimus Prime 	36 	49
2 	LOC 	0.999755 	Germany 	90 	97
3 	MISC 	0.556571 	Mega 	208 	212
4 	PER 	0.590256 	##tron 	212 	216
5 	ORG 	0.669693 	Decept 	253 	259
6 	MISC 	0.498349 	##icons 	259 	264
7 	MISC 	0.775362 	Megatron 	350 	358
8 	MISC 	0.987854 	Optimus Prime 	367 	380
9 	PER 	0.812096 	Bumblebee 	502 	511
```

上面识别出了很多实体，除了对应的word，还有实体类型(ORG/LOC/PER)、置信度得分、开始和结束下标。我们这里使用的aggregation_strategy，这对于"Optimus Prime"这样的超过一个词的实体会合并成一个。另外"Decepticons"被拆分成了两个Token，由于tag不一样，因此并没有合并成一个词。读者可能会奇怪为什么Decepticons会被拆分成"Decept"和"##icons"这两个token，后文会讲到subword tokenizer的知识，这里暂不详细展开。


### 抽取式问答

```
reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
```

运行结果：
```
 	score 	        start 	end 	 answer
0 	0.631292 	335 	358 	an exchange of Megatron
```


输出除了answer文本，还有start和end下标，标明文本在context中的下标。这也能看出，这种问答是抽取式的，也就是给定一个问题和context，找到context中能够回答这个问题的一段文字(start和end下标)。

### 自动摘要

```
summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=80, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
```

运行结果：

```
No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6 and revision a4f8f3e (https://huggingface.co/sshleifer/distilbart-cnn-12-6).
Using a pipeline without specifying a model name and revision in production is not recommended.

 Bumblebee ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead. As a lifelong enemy of the Decepticons, I hope you can understand my dilemma.
```

### 翻译

```
translator = pipeline("translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en")
outputs = translator("今天天气怎么样？", clean_up_tokenization_spaces=True, min_length=5)
print(outputs[0]['translation_text'])
```

运行结果：
```
How was the weather today?
```

### 文本生成

我们可以用gpt这种模型来条件生成一些文字，比如给定上面用户的投诉文本，我们可以自动的生成一些回复文字：

```
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
```

输出为：
```
Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee.

Customer service response:
Dear Bumblebee, I am sorry to hear that your order was mixed up. An exchange of

the Optimus Prime action figure, for the Optimus Prime action figure will be not only

taken in full order, but also reissued as special offers. Please be advised that special orders, once

discovered, will not be refunded.

```

## Hugging Face生态系统

Hugging Face最早提供了Transformers库，然后慢慢逐渐发展成了一个平台，成为一整套生态系统。下图是目前相关的一些库和功能的集合：

<a name='img9'>![](/img/learnhuggingface/9.png)</a>

从上图可以看出，Hugging Face的生态主要包括两个部分：一系列Transformer相关代码库和Hub。前者主要是代码，实现Transformer相关的功能，包括文本预处理相关库Tokenizer、数据加载库Datasets和核心的Transformers库，此外还包括分布式训练的Accelerate库；后者包括预训练模型(Models)、数据集(Datasets)、性能评价指标(Metrics)和在线文档(Doc)。

### Hugging Face Hub

前面介绍了最新范式的核心是用大规模的未标注数据进行预训练，然后在下游任务上微调。对于大部分用户来说，他们很少会从零开始训练一个模型，通常他们针对自己的下游任务进行微调。网上散布这各种各样的预训练模型，但是它们的实现方式差异很大。而有了Transformers库之后，Hugging Face通过Hub收集了最常见的一些预训练模型，这样我们就不用费劲去各个地方搜集模型了。截止到目前，Hub上有8万多个各种类型的预训练模型，可以通过任务、语言和关键词对模型进行搜索，从而找到最适合的模型。

<a name='img10'>![](/img/learnhuggingface/10.png)</a>

除了模型，Hub上也收集了学术界常用的开源数据集和针对这些数据集的标注评价指标，从而让我们可以更快的进行科研。下图是模型的详细介绍，所以的模型都通过Model Card来提供标准化的信息，从而可以方便的搜索。

<a name='img11'>![](/img/learnhuggingface/11.png)</a>

而且从上图右边我们可以看到Hub的一个很好的功能——”Hosted inference API“。我们不需要下载模型到本地就可以在Hub上测试这些模型，比如在上图的BERT base model (uncased) ，我们可以测试一些Masked Language Model的预测功能。


### Hugging Face Tokenizers

不同的模型会有不同的文本预处理流程，所有这些差异Tokenizers库都帮我们处理好了，并且很多时候提供了高效的Rust版本实现，他们的速度比原始论文里的实现更快。


### Hugging Face Datasets

加载、处理和保存数据是非常繁琐的工作，尤其是数据量大到超过内存的时候。Datasets库提供了标注的数据处理结果，并且使用[Apache Arrow](https://arrow.apache.org/)作为数据的backend，可以通过Memory Mapping实现磁盘和内存高效的数据交换。从而可以帮助我们应对海量数据的预训练。


### Hugging Face Accelerate

如果我们用其它框架写过多卡或者多机训练的代码，就会发现需要修改很多代码和配置来应对单机和集群环境的差异。Hugging Face Accelerate库在训练的环节增加了一层抽象，使得我们增加极少量代码就可以无缝的实现集群上的训练。



