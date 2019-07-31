---
layout:     post
title:      "BERT的成功是否依赖于虚假相关的统计线索？"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - BERT
---

本文介绍论文[Probing Neural Network Comprehension of Natural Language Arguments](https://arxiv.org/pdf/1907.07355)，讨论BERT在ACRT任务下的成绩是否依赖虚假的统计线索，同时分享一些个人对目前机器学习尤其是自然语言理解的看法。

 <!--more-->
 
**目录**
* TOC
{:toc}
 
## 论文解读

### Abstract

BERT在Argument Reasoning Comprehension Task(ARCT)任务上的准确率是77\%，这比没受过训练的人只底3个百分点，这是让人惊讶的好成绩。但是我们(论文作者)发现这么好的成绩的原因是BERT模型学习到了一些虚假相关的统计线索。我们分析了这些统计线索，发现很多其它的模型也是利用了这些线索。所以我们提出了一种方法来通过已有数据构造等价的对抗(adversarial)数据，在对抗数据下，BERT模型的效果基本等价于随机的分类器(瞎猜)。

### Introduction

论辩挖掘(argumentation mining)任务是找到自然语言论辩的结构(argumentative structure)。根据标注的结果分析，即使是人类来说也是一个很难的问题。

这个问题的一种解决方法是关注warrant——支持推理的世界知识。考虑一个简单的论证(argument)：(1) 因为现在在下雨； (2) 所以需要打伞。而能够支持(1)到(2)推理的warrant是：(3) 淋湿了不好。

[Argument Reasoning Comprehension Task, ACRT](https://arxiv.org/pdf/1708.01425)是一个关注推理并且期望模型发现隐含的warrant的任务。给定一个论证，它包括一个Claim(论点)和一个Reason，同时提供一个Warrant和一个Alternative。其中Warrant是支持从Reason到Claim推理的世界知识；而Alternative是一个干扰项，它无法支持从Reason到Claim的推理。用数理逻辑符号来表示就是：

$$
R \land W \rightarrow C \\
R \land A \rightarrow \neg C
$$

注意：ACRT数据集提供的Alternative一定能推出相反的结论。如果我们找一个随机的Alternative，它不能推导出C，但是也不一定能推导出$\neg C$。而这个数据集保证两个候选句子中一个是Warrant(一定能推导出C)，而另一个Alternative一定能推导出$\neg C$。这个特性在后面的构造adversarial数据集会非常有用。

下面是ACRT数据集的一个例子： 

| --- | ----------- |
| Claim | Google is not a harmful monopoly |
| Reason | People can choose not to use Google |
| Warrant | Other search engines don’t redirect to Google |
| Alternative | All other search engines redirect to Google |

论点是：Google不是一个寡头垄断。原因是：人们可以不使用Google。Warrant是：其它的搜索引擎不会重定向到Google。而Alternative是：其它的搜索引擎会重定向到Google。

因为其它搜索引擎不会重定向到Google，而且人们可以不使用Google，因此Google就不是一个垄断者。

因此这是一个二分类的问题，但是要做出正确的选择除了理解问题之外还需要很多的外部世界知识。在BERT之前，大部分模型的准确率都是达不到60%的准确率，而使用BERT可以达到77%的准确率，如下表所示。

<a name='img1'>![](/img/bert-bad/1.png)</a>
*图：ACRT任务上Baseline和BERT的效果* 

这比没有训练过的人只低3个点，这是非常让人震惊的成绩。因为训练数据里都没有提供这些世界知识，如果BERT真的表现这么好，那么唯一的解释就是它通过无监督的Pretraining从海量的文本里学到了这些世界知识。

为了研究BERT的决策，我们选择了那些多次训练BERT都比较容易正确预测的例子来分析。根据[SemEval-2018 Task 12: The Argument Reasoning Comprehension Task](https://www.aclweb.org/anthology/S18-1121)，Habernal等人在SemEval的任务上的做了类似的分析，和他们的分析类似(参考后面作者的观点)，我们发现BERT利用了warrant里的某些词的线索，尤其是"not"。通过寻根究底(probing)的设计实验来隔离这些效果(不让数据包含这种词的线索)，我们发现BERT效果好的原因就是它们利用了这些线索。

我们可以改进ACRT数据集，因为这个数据集上很好的特性$R \land A \rightarrow \neg C$，因此我们可以把结论反过来(加一个否定)，然后Warrant和Alternative就会互换，这样就可以保证模型无法根据词的分布来猜测哪个是Warrant哪个是Alternative。而通过这种方法得到的对抗(adversarial)数据集，BERT的准确率只有53%，比随机瞎猜没有强多少，因此这个改进的数据集是一个更好的测试模型的数据集。

### 任务描述和Baseline

假设$i=1,...,n$是训练集$\mathcal{D}$中每个训练数据的下标，因此$\vert \mathcal{D} \vert=n$。两个候选Warrant的下标$$j \in \{0,1\}$$，它的分布是均匀的，因此随机猜测正确的可能性是50%。输入是Claim $c^{(i)}$，Reason $r^{(i)}$，Warrant0 $w_0^{(i)}$和Warrant1 $w_1^{(i)}$；而输出是$$y^{(i)} \in \{0,1\}$$，是正确的Warrant的下标。


下图是解决这个问题的通用的模型结构，它会独立的考虑每一个Warrant。

<a name='img2'>![](/img/bert-bad/2.png)</a>
*图：实验的模型结构* 

因此给定$c^{(i)}$、$r^{(i)}$和$w_j^{(i)}$，模型最终会输出一个score，表示Warrant-j是正确的Warrant的可能性(logit)，然后使用softmax把两个logits变成概率。注意这个模型是独立考虑每一个Warrant的，每个Warrant的打分是和另外一个无关的，如果是相关的，则模型的输入要同时包含$w_0^{(i)}$和Warrant1 $w_1^{(i)}$。用数学公式描述其计算过程为：

$$
\begin{split}
z_j^{(i)} & =\theta[c^{(i)};r^{(i)};w_j^{(i)}] \\
p^{(i)} & =\text{softmax}(z_0^{(i)}, z_1^{(i)}) \\
\hat{y}^{(i)} & = argmax_jp^{(i)}
\end{split}
$$


模型$\theta$可以有很多种，这里的Baseline包括Bag of Vector(BoV)、双向LSTM(BiLSTM)、SemEval的冠军GIST和人类。结果如<a href='#img1'>上图</a>所示。对于所有的实验，我们都使用了网格搜索(grid search)的方法来选择超参数，同时我们使用了dropout和Adam优化算法。当在验证集上的准确率下降的话我们会把learning rate变为原来的1/10，最后的模型参数是在验证集上准确率最高的那组参数。BoV和BiLSTM的输入是300维的GloVe向量(从640B个Token的数据集上训练得到)。用于复现实验的代码、具体的超参数都放在[作者的GitHub](https://github.com/IKMLab/arct2)上。

### BERT

我们的BERT模型如下图所示。

<a name='img3'>![](/img/bert-bad/3.png)</a>
*图：处理argument-warrant对的BERT模型* 

我们把Claim和Reason拼接起来作为BERT的第一个"句子"(它们之间没有特殊的分隔符，因此只能靠句号之类的线索，这么做的原因是BERT最多输入两个句子，我们需要留一个句子给Warrant)，而Warrant是第二个"句子"，它们之间用特殊的SEP来分割，而最前面是特殊的CLS。CLS本身无语义，因此可以认为它编码了所有输入的语义，然后在它之上接入一个线性的全连接层得到一个logit $z_j^{(i)}$。两个Warrant都输入后得到$z_0^{(i)}$和$z_1^{(i)}$，最后用softmax变成概率。不熟悉BERT的读者可以参考[BERT课程](/2019/03/05/bert-prerequisites/)、[BERT模型详解](/2019/03/09/bert-theory/)和[BERT代码阅读](/2019/03/09/bert-codes/)。

整个模型(包括BERT和CLS上的线性层都会参与Fine-tuning)，learning rate是$2e^{−5}$，最大的Epoch数是20，选择在验证集上效果最好的那组参数。我们使用的是Hugging Face的[PyTorch实现](https://github.com/huggingface/pytorch-transformers)。

Devlin等人在[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)里指出，对于很小的数据集，BERT经常会无法收敛，从而得到很差的效果。ARCT是一个很小的数据集，它只有1,210个训练数据。在20次训练中有5次出现了这种情况，我们把它去掉后平均的准确率是71.6±0.04。这已经达到了之前最好的GIST的效果，而最好的一次是77%。我们只分析最好的77%的模型。


### 统计线索

虚假相关的统计线索主要来源于Warrant的不均匀的语言(词)分布，从而出现不同标签的不均匀词分布。虽然还有更复杂的线索，这里我们只考虑unigram和bigram。我们会分析如果模型利用这些线索会带来多大的好处，以及这种现象在这个数据集上有多么普遍。

形式化的，假设$\mathbb{T}_j^{(i)}$是第i个训练数据的第j(j=0或者1)个warrant的所有的Token的集合。我们定义一个线索(比如一个unigram或者bigram)的applicability $\alpha_k$为n个训练数据中只在一个标签里出现的次数。用数学语言描述就是：

$$
\alpha_k=\sum_{i=1}^n \mathbb{1}[\exists j,k \in \mathbb{T}_j^{(i)} \land k \notin \mathbb{T}_{\neg j}^{(i)}]
$$

用自然语言处理再来描述一下就是：如果某个线索(unigarm/词或者bigram)只在某个warrant里出现了，就加一。如果某个线索在两个warrant里都出现或者都不出现，则模型无法利用这个线索。最极端的，比如某个词只出现在warrant0里，那么模型可能就会学到错误的特征——一旦看到这个词出现就倾向于把它分到warrant0。注意：这个特征不见得就是错误的特征，比如情感分类任务里某个词或者某个词组(bigram)出现了确实就容易是正面或者负面的情感。但是对于ACRT这样的任务来说，我们一般认为(其实可能也可以argue)这样的特征是不稳定的，只有其背后的世界知识才是推理的真正原因，所以某些词(尤其是not这样的否定词)的出现与否与这个世界知识是无关的(我们可以用否定或者肯定来表示同样的语义：我很忧伤和我不高兴是一个语义，是肯定还是否定的表示方法与最终的结论无关)。

此外我们定义productivity $\pi_k$为：

$$
\pi_k = \frac{\sum_{i=1}^n \mathbb{1}[\exists j,k \in \mathbb{T}_j^{(i)} \land k \notin \mathbb{T}_{\neg j}^{(i)} \land y_i=j]}{\alpha_k}
$$

分母是$\alpha_k$，分子是$\alpha_k$里的并且模型分类和线索是同时出现的数量。比如"not"在n个训练数据里单独出现了5次，有3次只出现在warrant0，有2次只出现在warrant1。如果"not"只出现在warrant0的3次里有2次模型预测正确(预测为0)，"not"只出现在warrant1的2次里有1次预测正确(预测为1)，则分子就是2+1=3，分母就是5，则$\pi_{not}=\frac{3}{5}$。这个量是模型可能利用线索的"上限"，比如上面的例子，"not"单独出现了5次，模型预测正确了3次，则"not"这个特征对于分类正确"最大"的贡献就是0.6。


最后我们定义coverage $\xi_k=\alpha_k/n$。简单来说，productivity就是利用这个线索对于分类的好处，而coverage表示这个线索能够"覆盖"的数据范围。对于m(这里为2)分类的问题，如果$\pi_k>1/m$则说明这个线索对于分类是有帮助的。productivity和coverage最强的两个unigram线索是"not"这个词，它的productivity和coverage如下图所示。


<a name='img4'>![](/img/bert-bad/4.png)</a>
*图："not"的productivity和coverage*

它的意思就是平均来说，64%的数据都有"not"出现，如果只利用这个线索能够得到准确率为61%的分类结果。那么我们可以这么来得到一个分类器：如果"not"出现，我们就分类为0(假设训练数据中"not"出现更容易分类为0)；如果"not"不出现，我们就随机分类。那么这个分类器的准确率是多少呢？64%\*61%+(1-64%)\*50%=0.57。根据前面的描述，大部分分类器的准确率都没有超过0.6，而使用这样的特征就可以做到0.57。如果还有和"not"类似的特征，而且它们不完全相关(有一定的额外信息)，那么再加上其它的这类词特征可以进一步提高预测的准确率(前提是假设测试数据的词分布也是和训练数据一样的)。

### Probing实验

如果某个模型能只利用了Warrant的词的线索，那么我们只把warrant作为输入也应该会得到类似的准确率。当然这不是"真正"的解决了这个问题，因为你连claim(论点)和reason都没有看到，只看到warrant里出现了not就分类为0，这显然是不对的。举个前面的例子，我们的分类器会把"Other search engines don’t redirect to Google"分类为0，这显然是正确的，但是它分类的理由是：因为这个warrant包含了"not"。这显然是不对的。

类似的我们也可以只把warrant和claim、warrant和reason作为输入来训练模型，这样的模型学到的特征也肯定是不对的。但是实验的结果如下：


<a name='img5'>![](/img/bert-bad/5.png)</a>
*图：BERT Large、BoV和BiLSTM模型的probing实验*

我们看到在BERT Large模型里只用Warrant作为输入就可以得到71%的准确率，这和之前最好的GIST模型差不多，而把Warrant和Reason作为输入可以得到最高75%的准确率。因此ACRT数据集是有问题的，我们的输入不完整就可以得到75%的准确率，这就好比老师的题目还没写完，你就把答案写出来了，这只能说明你作弊或者是瞎猜的。


### 对抗测试数据

ACRT数据集的词的统计不均匀问题可以使用下面的技巧来解决。因为$R \land A \rightarrow \neg C$，我们可以把claim变成它的否命题，这样Warrant和Alternative就会互换，这样就能保证通用一个句子既是Warrant0也是Warrant1，从而在所有的Warrant里词的部分完全是均匀的。下图是一个示例。

<a name='img6'>![](/img/bert-bad/6.png)</a>
*图：一个原始的训练数据以及有它生成的对抗数据*


这个对抗例子为：我们可以使用其它的搜索引擎，但是其它搜索引擎最终会重定向到Google，所以Google是一个垄断者。也就是把Claim加一个否定(双重否定就是肯定，就可以去掉not)，原来的Alternative就变成的新的Warrant。这样"All other search engines redirect to Google"在Warrant0和Alternative都出现一次，从而词的分布是均匀的。

对于这种方法生成的对抗训练数据，我们做了两个实验。第一个实验使用原始的(没有加入对抗样本)训练数据和验证数据训练模型，然后在对抗数据集上测试，其结果比随机猜(50%)还差，因为它过拟合了某些根本不对的特征，测试数据根本不是这样的分布。第二个使用是使用对抗数据进行训练和测试，则BERT最好的效果只有53%。



### 补充一点

我个人觉得这篇文章还有一点小缺陷，那就是没有使用ACRT排名第一的模型GIST跑一下对抗数据集。因为GIST会使用SNLI等NLI任务的数据进行预训练，也许从这里可以学到一些世界知识用于解决ACRT的推理问题。我在[这里](https://github.com/IKMLab/arct2/issues/2)里提了这个问题，不过作者认为NLI的数据并不包含解决ACRT问题的世界知识，因此没有必要做这个实验(也许更主要的问题是不想重新实现一遍GIST模型?这也许说明了论文开源代码的价值，别的研究者可以很容易的check和利用其工作)。


## 相关讨论

这里注意收集了一些来自Reddit的帖子[BERT's success in some benchmarks tests may be simply due to the exploitation of spurious statistical cues in the dataset. Without them it is no better then random.](https://www.reddit.com/r/MachineLearning/comments/cfxpxy/berts_success_in_some_benchmarks_tests_may_be/)讨论里的一些观点。下面的内容都是我摘录和翻译(意译)的部分观点。

### 观点1(贴主orenmatar)

首先我们来看Reddit帖子的标题：BERT在某些benchmark上的成功是因为简单的利用了不相关的统计线索。如果不(能)利用这些线索，它的效果就是随机的瞎猜。这个观点相对比较客观，只是描述了一个事实。

### 观点2(neato5000)

赞成贴主的观点，并且提到[Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://arxiv.org/abs/1902.01007)里有类似的结论。



### 观点3(lysecret)

同意贴主的观点，但是认为BERT模型依然很有用，在3个个人项目中比其它模型好很多。贴主orenmatar同意BERT，他说自己发帖并没有否定BERT的价值，但是强调只是在ACRT这个特定集合上BERT学到的只是不相关的统计线索。


### 观点4(贴主orenmatar)


这种论文应该更多一些。我们每天都听到NLP的各种突破，每过几个月就出现更好大模型，得到超乎想象的结果。但是很少有人实际的想办法分析这些模型是否只是因为学习到一些无意义的特征。因此我们有必要往后一步，仔细看看数据集和分析一下模型到底学到了什么东西。

针对观点4，melesigenes认为这个观点对于文章大家结论过于扩大范围了(overgeneralizing)。BERT在ACRT数据集上没有学到什么并不代表在其它的数据集上没有学到有意义的东西。

### 观点5(dalgacik)

这个观点认为这篇论文并没有说明BERT模型有什么问题，只是指出了ACRT这个数据集有问题。

### 观点6(gamerx88)

很多进展其实都是模型过拟合了这个数据集而已，这在很多比赛类的任务都出现过。

### 观点7(fiddlewin)

我发现很多评论错误的解读了论文？论文只是说模型(包括BERT和其它)在某个特定任务(ARCT)上利用了统计线索(比如是否出现"not")。当引入对抗样本从而去掉这些线索之后，BERT的性能只有50%，和没有训练的人的80%相比差别很大，说明这个任务很难需要很深层次的语义理解。

但是这篇论文从来没有怀疑BERT在其它任务的能力，这也是符合常识的——学习算法解决问题的方法不一定是人(想象)那样的。

### 观点8(lugiavn)


这篇文章TLDR(Too long, don't read)的描述了一个很简单的事实：不平衡数据上训练的模型在和训练集不同分布的测试集上表现不会太好。这并没有什么稀奇。所有的机器学习模型都是这样。为什么要把BERT单独拎出来呢？

delunar对这个观点持不同态度，他认为这不是不平衡数据的问题。而是因为BERT错误的"理解"了文本的意思但是做出了相对程度正确的预测。


## 作者观点

这篇文章之所以引起大家的关注首先是因为BERT模型最近很火，另外一个原因其实就是很多研究者对于现在机器学习(深度学习)社区对于这种刷榜的研究风气的担忧。很多研究者不在模型结构和其它方面做创新，只是使用更大的模型和更多的数据追求在某些公开数据集上刷榜。而这篇文章正是在这样的背景下引起了极大的关注。

其实类似的文章还包括[SemEval-2018 Task 12: The Argument Reasoning Comprehension Task](https://www.aclweb.org/anthology/S18-1121)、[Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://arxiv.org/abs/1902.01007)和[Is It Worth the Attention? A Comparative Evaluation of Attention Layers for Argument Unit Segmentation](https://arxiv.org/abs/1906.10068)。

另外在计算机视觉领域最近也有一篇文章[Natural Adversarial Examples](https://arxiv.org/abs/1907.07174)，大家可能认为ImageNet已经是一个解决的问题。但是作者找到了很多"自然"(真实)的对抗性的数据，现有的模型在上面的分类准确率非常低，基本都到不了5%。即使通过一些办法来优化，作者也只能做的15%。下面是一些示例数据：



<a name='img7'>![](/img/bert-bad/7.png)</a>
*图：ImageNet-A中的自然对抗数据*


比如最右边的图，实际分类是牛蛙(bullfrog)，但是模型很容易分成黑松鼠(fox squirrel)。当然视觉和语言还是有较大的区别，但是现在的模型确实有可能学到的特征和人类(甚至动物)学到的有很大区别(当然也可以argue人或者动物大脑学到的也许是类似的东西)。


我们还是回到语言和BERT是否学到不相关的统计线索的问题上来。首先我认为BERT是非常用于的一种模型，它最大的优点是可以是无监督的Pretraining从海量的数据中学习Token(词)的上下文语义关系。因此Fine-tuning之后能在很多(浅层)的语言理解任务上去掉了很好的效果。但是BERT不是万能的，论文里也提到训练数据很少的情况下它可能不能训练。我们在工作中也发现了一个很有趣的意图分类的例子——有一个客户的数据量很少，大概1000+训练数据作用，而意图数(分类数)是100+。我们发现使用简单的Logistic Regression或者CNN都能达到非常好的效果——在测试集合上能有99%，但是使用BERT怎么调参也只有不到80%。后来我们分析数据发现这个客户的意图定义的很特别——只有包含某个词就作为一个意图(分类)，它并不需要特殊的泛化和上下文。因此我们猜测可能的原因是因为BERT的参数过多，而且同样一个词在不同的上下文可能会被编码成不同的向量，因此在训练数据不够的情况下反而没有学到这个任务最简单和重要的特征——只要有这个词就分为这个类别。


其次我认为就是现在的NLP模型(不管是BERT还是其它的模型)它没有办法获得(足够多的)常识(世界知识)。虽然海量的文本里包含了大量的世界知识(其实我觉得很多世界知识是不能在Wiki这样的地方找到的，比如前面的例子：下雨为什么要打伞，因为淋湿了不好。淋湿的感觉不舒服，那这个不舒服能用精确的语言描述吗？也许可以，也许在文学作品里有描述，但是很难用数学语言描述，很多要靠类比，但是淋过雨的人都有类似的感受)，但是现在的模型(包括BERT)都很难学习到这些知识。因为它看到的只是这些世界知识通过语法编码后的文字，通过分析文字的共现之类的方法可能发现一些浅层的语法和语义，但是很难学到更深层次的语义和逻辑。至少我们人类的学习不是这样的——给你100TB的火星文，然后遮住某个词让你猜测可能是哪个词。语言只不过是人类定义的用于沟通的符号系统，它背后的根源还是我们生存的这个宇宙以及我们通过视觉、听觉等感觉器官对这个世界的感觉。当然除了当下的感觉之外也包括很久以前的感觉甚至是我们出生前通过文化传承下来的先人们的感觉。如果抛开我们的身体和感觉器官，只是从符号的角度来研究自然语言，我觉得是不能根本解决这个问题的。当然这并不是说BERT这样的模型不重要，我们在还没有更好的方法的时候这些模型可以帮助我们解决一些问题，但是千万不能以为它们能解决所有问题。更多关于语言以及人工智能哲学的问题讨论，感兴趣的读者可以参考[《人工智能能否实现？》](/2019/03/14/philosophy/)。



