---
layout:     post
title:      "对BERT的pretraining改进的几篇文章"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - BERT
    - Pretraining
---

本文介绍对于BERT的Pretraining过程进行改进的几篇文章，包括[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf)、[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223)和[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/pdf/1907.12412v1)。

注意：这几篇文章都是对BERT模型的Pretraining阶段的Mask进行了不同方式的改进，但是对于BERT模型本身(基于Mask LM的Pretraining、Transformer模型和Fine-tuning)没有做任何修改。因此对于不需要Pretraining的用户来说只要把Google提供的初始模型替换成这些模型就可以直接享受其改进了(百度的ERNIE和ERNIE 2.0是基于PaddlePaddle的，Tensorflow和PyTorch用户需要借助第三方工具进行转换)。

阅读本文前，读者需要了解BERT的基本概念，不熟悉的读者可以先学习[BERT课程](/2019/03/05/bert-prerequisites/)、[BERT模型详解](/2019/03/09/bert-theory/)和[BERT代码阅读](/2019/03/09/bert-codes/)。

 <!--more-->
 
**目录**
* TOC
{:toc}
 

## Whole Word Masking
### 基本思想
注：虽然我这里介绍的是哈工大与科大讯飞的论文[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf)，但是Whold Word Mask其实是BERT的作者提出来的。他们并没有发论文，因为这是一个很简单(但是很有效)的改进。由于中文的特殊性，BERT并没有提供中文版的Pretraining好的Whole Word Masking模型。中文版的Pretraining Whole Word Masking模型可以在[这里](https://github.com/ymcui/Chinese-BERT-wwm)下载。


为了解决OOV的问题，我们通常会把一个词切分成更细粒度的WordPiece(不熟悉的读者可以参考[机器翻译·分词](/books/mt/#%E5%88%86%E8%AF%8D)和[WordpieceTokenizer](/2019/03/09/bert-codes/#wordpiecetokenizer))。BERT在Pretraining的时候是随机Mask这些WordPiece的，这就可能出现只Mask一个词的一部分的情况，比如下面的例子：

<a name='img1'>![](/img/bert-imp/1.png)</a>
*图：Whole Word Mask模型的示例*

probability这个词被切分成"pro"、"#babi"和"#lity"3个WordPiece。有可能出现的一种随机Mask是把"#babi" Mask住，但是"pro"和"#lity"没有被Mask。这样的预测任务就变得容易了，因为在"pro"和"#lity"之间基本上只能是"#babi"了。这样它只需要记住一些词(WordPiece的序列)就可以完成这个任务，而不是根据上下文的语义关系来预测出来的。类似的中文的词"模型"也可能被Mask部分(其实用"琵琶"的例子可能更好，因为这两个字只能一起出现而不能单独出现)，这也会让预测变得容易。

为了解决这个问题，很自然的想法就是词作为一个整体要么都Mask要么都不Mask，这就是所谓的Whole Word Masking。这是一个很简单的想法，对于BERT的代码修改也非常少，只是修改一些Mask的那段代码。对于英文来说，分词是一个(相对)简单的问题。哈工大与科大讯飞的论文对中文进行了分词，然后做了一些实验。

### 实现细节
#### 数据预处理
训练数据来自Wiki的[中文dump](https://dumps.wikimedia.org/zhwiki/latest/)，使用[WikiExtractor.py](https://github.com/attardi/wikiextractor/blob/master/WikiExtractor.py)对Wiki进行抽取，总共得到1,307个文件。这里同时使用了简体和繁体中文的Wiki文章。对网页进行抽取和处理之后得到13.6M行的文本。中文分词使用的是[哈工大LTP](http://ltp.ai/)。分完词后使用BERT官方代码提供的create_pretraining_data.py来生成训练数据，没有做任何修改(包括Mask的比例)。

#### Pretraining
Whole Word Masking可以看做是对原来BERT模型的一个改进，一种增加任务难度的方法，因此我们并不是从头开始Pretraining，而是基于原来Google发布的中文模型继续训练的(我感觉另外一个原因就是没有那么多计算资源从头开始)。这里使用了batch大小为2,560，最大长度为128训练了100k个steps，其中初始的learning rate是1e-4(warm-up ratio是10%）。然后又使用最大长度为512(batch大小改小为384)训练了100k个steps，这样让它学习更长距离的依赖和位置编码。

BERT原始代码使用的是AdamWeightDecayOptimizer，这里换成了LAMB优化器，因为它对于长文本效果更好。


#### Fine-tuning

Fine-tuning的代码不需要做任何修改，只不过是把初始模型从原来的基于WordPiece的Pretraining的模型改成基于Whole Word Masking的Pretraining模型。

### 实验结果

#### 阅读理解任务

实验了CMRC 2018、DRCD和CJRC，结果如下。

<a name='img2'>![](/img/bert-imp/2.png)</a>
*图：CMRC 2018数据集上的实验结果*


<a name='img3'>![](/img/bert-imp/3.png)</a>
*图：DRCD数据集上的实验结果*


<a name='img4'>![](/img/bert-imp/4.png)</a>
*图：CJRC数据集上的实验结果*

#### 命名实体识别任务

在人民日报(People Daily)和微软亚研究院NER(MSRA-NER)数据集做了实验，结果如下。


<a name='img5'>![](/img/bert-imp/5.png)</a>
*图：NER任务的实验结果*

#### 自然语言推理(Natural Language Inference)

对于自然语言推理任务XNLI中的中文数据进行了实验，结果如下。

<a name='img6'>![](/img/bert-imp/6.png)</a>
*图：XNLI中文数据集的实验结果*

#### 情感分类(Sentiment Classification)

在ChnSentiCorp和Sina Weibo两个数据集上的实验结果如下。

<a name='img7'>![](/img/bert-imp/7.png)</a>
*图：情感分类任务的实验结果*


#### 句对匹配(Sentence Pair Matching)

在LCQMC和BQ Corpu数据集上的实验结果为：

<a name='img8'>![](/img/bert-imp/8.png)</a>
*图：句对匹配任务的实验结果*

#### 文档分类

在THUCNews数据集上的实验结果如下表。

<a name='img9'>![](/img/bert-imp/9.png)</a>
*图：THUCNews数据集上的实验结果*

### 一些技巧

下面是论文作者在实现时的一些技巧总结：

* 初始的learning rate是最重要的超参数，一定要好好调。

* BERT和BERT-wwm的learning rate是相同的，但是如果使用ERNIE，则需要调整。

* BERT和BERT-wwm使用Wiki训练，因此在比较正式和标准的数据集上效果较好。而百度的ERNIE使用了网页和贴吧等数据，因此在非正式的语境中效果较好。

* 对于长文本，比如阅读理解和文档分类等任务，建议使用BERT和BERT-wwm。

* 如果任务的领域和Wiki等差异很大，并且我们有较多未标注数据，那么建议使用领域数据进行Pretraining。

* 对于中文(简体和繁体)，建议使用BERT-wwm(而不是BERT)的Pretraining模型。

### 更新

7月30日作者又使用了更大的数据训练了更大的模型，感兴趣的读者可以访问[这里](https://github.com/ymcui/Chinese-BERT-wwm)。

## ERNIE
### 基本思想
ERNIE是百度在论文[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223)提出的模型，它其实比Whole Word Masking更早的提出了已此为单位的Mask方法。虽然它的这个名字有些大：通过知识集成的增强表示(ERNIE)，但是它的思想其实和前面的Whole Word Masking非常类似，只是把Masking的整体从词多大到短语(phrase)和实体(entity)而已。这篇论文的发表时间其实比前面的Whole Word Masking要早，因此我们可以认为Whole Word Masking可能借鉴了其思想？(Arxiv上的这篇论文的时间是2019/4/19；而BERT更新英文Whole Word Masking模型的时间是2019/5/31) 我这里把它放到后面介绍的目的是为了能让百度的两篇论文放到一起。

<a name='img10'>![](/img/bert-imp/10.png)</a>
*图：ERNIE和BERT的对比*

如上图所示，ERNIE会把phrase "a series of"和entity "J. K. Rowling"作为一个整体来Mask(或者不Mask)。

当然，这里需要用NLP的工具来识别phrase和entity。对于英文的phrase，可以使用chunking工具，而中文是百度自己的识别phrase的工具。entity识别工具的细节论文里没有提及。

### Pretraining的数据
训练数据包括中文Wiki、百度百科、百度新闻和百度贴吧数据，它们的句子数分别是21M, 51M, 47M和54M，因此总共173M个句子。

此外把繁体中文都转换成了简体中文，英文都变成了小写，模型的词典大小是17,964。

### 多层级Mask的训练

ERNIE没有使用WordPiece，而是把词作为基本的单位。它包括3种层级的Masking：基本、phrase和entity，如下图所示。


<a name='img11'>![](/img/bert-imp/11.png)</a>
*图：一个句子的不同层级的Masking*

训练的时候首先基于基本级别的Masking训练，对于英语来说，用空格分开的就是词；对于中文来说，就是字。基本级别对于英文来说和BERT的Whole Word Masking类似，但是对于中文来说就是等价于最原始的BERT。

接着基于phrase级别的Masking进行训练，这样它可以学到短语等更高层的语义。最后使用entity级别的Masking来训练。


### Dialogue Language Model

除了Mask LM之外，对于对话数据(贴吧)，ERNIE还训练了所谓的Dialogue LM，如下图所示。


<a name='img13'>![](/img/bert-imp/13.png)</a>
*图：Dialogue Language Model*

原始对话为3个句子："How old are you?"、"8."和"Where is your hometown?"。模型的输入是3个句子(而不是BERT里的两个)，中间用SEP分开，而且分别用Dialogue Embedding Q和R分别表示Query和Response的Embedding，这个Embedding类似于BERT的Segment Embedding，但是它有3个句子，因此可能出现QRQ、QRR、QQR等组合。


### 实验

论文在XNLI、LCQMC、MSRA-NER、ChnSentiCorp和NLPCC-DBQA等5个中文NLP任务上做了和BERT的对比实验，如下表所示：


<a name='img12'>![](/img/bert-imp/12.png)</a>
*图：一个句子的不同层级的Masking*

### Ablation分析

论文使用了10%的数据来做分析，如下表所示

<a name='img14'>![](/img/bert-imp/14.png)</a>
*图：不同Mask策略的Ablation分析*

只用基本级别(中文的字)，测试集上的结果是76.8%，使用了phrase之后，能提高0.5%到77.3%，再加上entity之后可以提高到77.6%。

为了验证Dialog LM的有效性，论文对于10%的数据的抽取方式做了对比，包括全部抽取百科、百科(84%)+新闻(16%)和百科(71.2%)+新闻(13%)+贴吧(15.7%)的三种抽取数据方式。在XLNI任务上的对比结果为：

<a name='img15'>![](/img/bert-imp/15.png)</a>
*图：Dialog LM作用的Ablation分析*

可以看到Dialog LM对于XLNI这类推理的任务是很有帮助的。注：我认为这个实验有一定问题，提升可能是由于数据(贴吧)带来的而不是Dialog LM带来的。更好的实验可能是这样：再训练一个模型，数据为百科(84%)+新闻(16%)和百科(71.2%)+新闻(13%)+贴吧(15.7%)，但是这次不训练Dialog LM，而是普通的Mask LM，看看这个模型的结果才能说明Dialog LM的贡献。


### 完形填空对比

此外，论文还实验完形填空的任务对比了BERT和ERNIE模型。也就是把一些测试句子的实体去掉，然后让模型来预测最可能的词。下面是一些示例(不知道是不是精心挑选出来的)：


<a name='img16'>![](/img/bert-imp/16.png)</a>
*图：Dialog LM作用的Ablation分析*

## ERNIE 2.0


### 基本思想

ERNIE 2.0的名字又改了：A Continual Pre-training framework for Language Understanding。这个名字低调务实了一些，但是和缩写ERNIE似乎没有太大关系。

作者认为之前的模型，比如BERT，只是利用词的共现这个统计信息通过训练语言模型来学习上下文相关的Word Embedding。ERNIE 2.0希望能够利用多种无监督(弱监督)的任务来学习词法的(lexical)、句法(syntactic)和语义(semantic)的信息，而不仅仅是词的共现。

因为引入了很多新的任务，所以作为multi-task来一起训练是非常自然的想法。但是一下就把所有的任务同时来训练可能比较难以训练(这只是我的猜测)，因此使用增量的方式会更加简单：首先训练一个task；然后增加一个新的Task一起来multi-task Learning；然后再增加一个变成3个task的multi-task Learning……



### ERNIE 2.0框架

根据前面的介绍，ERNIE 2.0其实是一种框架(方法)，具体是用Transformer还是RNN都可以，当然更加BERT等的经验，使用Transformer会更好一些。ERNIE 2.0框架的核心就是前面的两点：构造多个无监督的任务来学习词法、句法和语义的信息；增量的方式来进行Multi-task learning。

ERNIE 2.0框架如下图所示。

<a name='img18'>![](/img/bert-imp/18.png)</a>
*图：ERNIE 2.0框架*


持续的(continual)pretraining过程包括两个步骤。第一步我们通过大数据和先验知识来持续的构造无监督任务。第二步我们增量的通过multi-task learning来更新ERNIE模型。



对于pre-training任务，我们会构造不同类型的任务，包括词相关的(word-aware)、结构相关的(structure-aware)和语义相关的(semantic-aware)任务，分别来学习词法的、句法的和语义的信息。所有这些任务都只是依赖自监督的或者弱监督的信号，这些都可以在没有人工标注的条件下从大量数据获得。对于multi-task pre-training，ERNIE 2.0使用增量的持续学习的方式来训练。具体来说，我们首先用一个简单的任务训练一个初始的模型，然后引入新的任务来更新模型。当增加一个新的任务时，使用之前的模型参数来初始化当前模型。引入新的任务后，并不是只使用新的任务来训练，而是通过multi-task learning同时学习之前的任务和新增加的任务，这样它就既要学习新的信息同时也不能忘记老的信息。通过这种方式，ERNIE 2.0可以持续学习并且累积这个过程中学到的所有知识，从而在新的下游任务上能够得到更好的效果。

如下图所示，持续pre-training时不同的task都使用的是完全相同的网络结构来编码上下文的文本信息，这样就可以共享学习到的知识。我们可以使用RNN或者深层的Transformer模型，这些参数在所有的pre-training任务是都会更新。


<a name='img19'>![](/img/bert-imp/19.png)</a>
*图：ERNIE 2.0框架的multi-task learning架构图*
 
如上图所示，我们的框架有两种损失函数。一种是序列级别的损失，它使用CLS的输出来计算；而另一种是token级别的损失，每一个token都有一个期望的输出，这样就可以用模型预测的和期望的值来计算loss。不同的pre-training task有它自己的损失函数，多个任务的损失函数会组合起来作为本次multi-task pre-training的loss。

### 模型的网络结构

模型采样和BERT类似的Transformer Encoder模型。为了让模型学习到任务特定的信息，ERNIE 2.0还引入了Task Embedding。每个Task都有一个ID，每个Task都编码成一个可以学习的向量，这样模型可以学习到与某个特定Task相关的信息。

网络结果如下图所示：


<a name='img20'>![](/img/bert-imp/20.png)</a>
*图：ERNIE 2.0框架的网络结构*

除了BERT里有的Word Embedding、Position Embedding和Sentence Embedding(基本等价于Segment Embedding)。图上还有多了一个Task Embedding。此外Encoder的输出会用来做多个Task(包括word-aware、structural-aware和semantic-aware)的multi-task learning，而不是像BERT那样只是一个Mask LM和next sentence prediction(而XLNet只有一个Permuatation LM)。

### Pre-training Tasks

#### Word-aware Tasks

##### Knowledge Masking Task

这其实就是ERNIE 1.0版本的任务，包括word、phrase和entity级别的mask得到的任务。

##### Capitalization Prediction Task

预测一个词是否首字母大小的任务。对于英文来说，首字符大小的词往往是命名实体，所以这个任务可以学习到一些entity的知识。

##### Token-Document Relation Task

预测当前词是否出现在其它的Document里，一个词如果出现在多个Document里，要么它是常见的词，要么它是这两个Document共享的主题的词。这个任务能够让它学习多个Document的共同主题。

#### Structure-aware Tasks 

##### Sentence Reordering Task

给定一个段落(paragraph)，首先把它随机的切分成1到m个segment。然后把segment随机打散(segment内部的词并不打散)，让模型来恢复。那怎么恢复呢？这里使用了一种最简单粗暴的分类的方法，总共有$k=\sum_1^m k!$种分类。这就是一个分类任务，它可以让模型学习段落的篇章结构信息。

##### Sentence Distance Task

两个句子的"距离"的任务，对于两个句子有3种关系(3分类任务)：它们是前后相邻的句子；它们不相邻但是属于同一个Document；它们属于不同的Document。

#### Semantic-aware Tasks

##### Discourse Relation Task

这个任务会让模型来预测两个句子的语义或者修辞(rhetorical)关系，follow的是[Mining discourse markers for unsuper-
vised sentence representation learning](https://arxiv.org/pdf/1903.11850.pdf)的工作，感兴趣的读者可以阅读这篇文章。

##### IR Relevance Task

这是利用搜索引擎(百度的优势)的数据，给定Query和搜索结果(可以认为是相关网页的摘要)，可以分为3类：强相关、弱相关和完全不相关。

### 实验

论文对于英文，在GLUE数据集上和BERT以及XLNet做了对比；在中文数据集上和BERT以及ERNIE1.0做了对比实验。

#### Pretraining数据

英文数据使用了Wiki和BookCorpus，这和BERT是一样的，此外还爬取了一些Reddit的数据。另外使用了Discovery数据集来作为篇章结构关系数据。对于中文，使用了百科、新闻、对话、搜索和篇章结构关系数据。详细情况如下表所示：

<a name='img17'>![](/img/bert-imp/17.png)</a>
*图：Pretraining数据详情*

#### Pretraining的设置


为了和BERT对比，我们使用和它完全一样的设置。基本(base)模型包含12层，每层12个self-attention head，隐单元大小是768.对于大(large)模型包含24层，16个self-attention head，隐单元1024。XLNet模型的设置和BERT也是一样的。


ERNIE 2.0的基本模型使用48块Nivida的v100 GPU来训练，而大模型使用64块v100 GPU训练。ERNIE 2.0使用[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)实现，这是百度开源的一个深度学习平台。论文使用了Adam优化器，其中$\beta_1=0.9, \beta_2=0.98$，batch大小是393216个token。英语的learning rate是5e-5而中文是1.28e-4。learning rate decay策略是[noam](https://arxiv.org/pdf/1706.03762)，前4000 step是warmup。为了结束内存，使用的是float16。每个pretraining都训练到收敛为止。

#### 实验任务

英文实验使用的是[GLUE](https://gluebenchmark.com/)；而中文任务和前面的Whole Word Masking差不多，这里就不详细列举了，感兴趣的读者可以参考论文或者代码。

#### 实验结果

英文的结果如下表所示：

<a name='img21'>![](/img/bert-imp/21.png)</a>
*图：GLUE数据集上的实验结果*

我们可以看到，不论是基本的模型还是大的模型，ERNIE 2.0在测试集上的效果都是SOTA的。

中文的结果如下表所示：

<a name='img22'>![](/img/bert-imp/22.png)</a>
*图：中文数据集上的实验结果*


#### 代码

感兴趣的读者可以去[这里](https://github.com/PaddlePaddle/ERNIE)下载Pre-training好的ERNIE 2.0模型和Fine-tuning的代码(但是没有Pre-training的代码和数据)。



## 展望

我们可以看到进入2019年之后，无监督的Contextual Word Embedding成为NLP领域最热门的研究方向，没过多久就会有新的模型出来刷榜。这一方面说明了在海量的未标注的文本里包含了大量有用的语义知识，因此我们希望通过这些数据进行无监督的pre-training。从ERNIE 2.0的成功来看，通过构造更多的任务可以让模型学习更多的知识，成功的关键是怎么构造任务的监督信号(显然不能人工标注)。另外，对于这种靠模型和训练数据的大小简单粗暴的刷榜的行为是否可持续，近期也是引起了学术界的持续关注。因为这个方向只能是有数据和计算资源的少数商业公司才有资本去这样玩，很多论文其实并没有太大的创新。另外很多论文的结论其实可能是有矛盾的。比如最近Facebook的[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692)，它并没任何模型结构，只是改进了pretraining的方法就在某些数据集上比XLNet的效果要好，这不禁让人怀疑XLNet的改进到底是有Permutation LM带来的还是只是因为它训练的比BERT要好。包括ERNIE 1.0提出的所谓的Knowledge(也就是phrase和entity)，但是哈工大和讯飞的简单的Whole Word Masking模型就做的比ERNIE 1.0更好，这也让人怀疑ERNIE 1.0的改进是不是由于它的数据带来的(它是有了wiki之外的百度百科、新闻和贴吧等数据)。

另一方面，如果大家把注意力都集中到刷榜，而不是从更本质的角度思考NLP甚至AI的问题，只是等待硬件的进步，这也是很让人担忧的事情。在[BERT的成功是否依赖于虚假相关的统计线索？](/2019/07/26/bert-spurious-stats-cue/)一文里我也分析了学术界对于这种暴力美学的担忧，有兴趣的读者也可以阅读一下这篇文章。
