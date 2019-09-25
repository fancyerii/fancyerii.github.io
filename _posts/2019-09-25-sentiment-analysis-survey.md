---
layout:     post
title:      "情感分析综述" 
author:     "lili" 
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - 深度学习
    - 情感分析
    - sentiment analysis
---

本文是对情感分析这个问题(领域)的调研，对于问题的定义和介绍主要参考了[Sentiment Analysis: mining sentiments, opinions, and emotions](https://www.cs.uic.edu/~liub/FBS/sentiment-opinion-emotion-analysis.html)，虽然这本书写于2015年，很多当时SOTA的方法现在已经过时了，但是本书对于问题的介绍和描述非常详细和清晰，强烈建议对情感分析领域有兴趣的读者阅读。

<!--more-->

**目录**
* TOC
{:toc}

## 简介

### 术语解释

情感分析(sentiment analysis)，也叫作观点挖掘(opinion mining)，它的研究问题是分析人们通过文本表达的对某个实体(entity)的观点(opinion)、情感(sentiment)、评价(appraisal)、态度(attitude)和情绪(emotion)。这里的实体可以是产品、服务、组织机构、个人、事件和各种话题。除了sentiment analysis之外，在学术界还有很多类似的术语，包括观点挖掘(opinion mining)、观点分析(opinion analysis)、观点抽取(opinion extraction)、情感挖掘(sentiment mining)、主(客)观分析(subjectivity analysis)、情绪分析(affect analysis)、情感分析(emotion analysis)和评论挖掘(review mining)等等。注意这些词翻译成中文后很难区分了，所以最好看对应的英文术语，当然这些英文术语的区别也会细微，后文会介绍它们的区别。这么多术语的原因之一是因为它是一个比较新的领域，很多不同领域的研究者都开始研究这个问题，因此没有形成统一的术语。除了NLP社区，数据挖掘(data minin)和信息检索(information retrieval)等很多其它领域的研究者也都在研究这个问题。

下面我们来介绍一下sentiment和opinion的细微区别，同时形成统一的术语体系。根据Merriam-Webster词典，sentiment是态度(attitude)、想法(thought)和情绪感觉(feeling)激发出来的意见(judgment)。而opinion是人对某一个特点事物在大脑了形成的看法(view)、判断(judgment)。它们的区别非常细微，opinion更多关注一个人对某个事物的看法(view)而sentiment更多关注情绪和感受。比如句子"I am concerned about the current state of the economy"是sentiment，它表达了说话人对于当前经济经济状况的担忧(情感)；而"I think the economy is not doing well"表达了说话人对当前经济状况(不太好的)观点。对于的一个句子，我们的回复可能是"I share your sentiment"；而对于第二个句子的回复可能是"I agree/disagree with you"。当然这两个句子是有一定相关性的，第一个句子的负面情绪可能来源于第二个句子的判断结果——因为说话人对于当前经济状况的判断是不好的，所以可能产生担忧的情绪。通常表达观点的句子是正面(positive)或者负面(negative)的情感，但是也有一下表达观点的句子不带任何情感，比如"I think he will go to Canada next year"，这个句子只是表达了"我"对于他去加拿大的一个观点而已，并没有好坏的情感。

在业界几乎只使用情感分析(sentiment analysis)这个术语，而学术界情感分析和观点挖掘都有被使用。我们这里并不区分情感分析和观点挖掘这两个术语。我们这里用opinion表示情感(sentiment)、评估(evaluation)、评价(appraisal)、态度(attitude)等所有相关信息，比如我们把评论的对象叫做opinion target(评论目标)；而用sentiment表示这个opinion背后隐含的正面/负面情感。前面也说过，某些观点(opinion)并不带有情感，因此我们只研究带情感的观点。在讨论正面或者负面的情感时，我们通常用中立(neutral)情感来表示不带情感的句子。除了情感和观点之外，相关的词语还包括感受(affect)、情感(emotion)和情绪(mood)，后面我们会介绍它们。

表达观点和情感的句子通常是主观的(subjective)而不是客观的(objective)句子，因为客观的句子通常都是描述事实。但是客观的句子可能会隐含观点和情感，尤其是描述一些不想要的(undesirable)事实的句子。比如"I bought the car yesterday and it broke today"和"after sleeping on the mattress for a month, a valley has formed in the middle"，昨天刚买的车就不能开、毯子一个月就有裂缝，这显然不是说话人想要的结果。情感分析也需要处理客观句子里的这种隐式的观点和情感。简单来说，情感分析需要分析文本里显示或者隐式表达的正面或者负面观点/情感，同时还需要找到对应的情感目标(opinion target)——比如地一个句子的是car而第二个句子是mattress。

在2000年之前，由于数字化的文本数据很少，情感分析很少被研究。而在2000年之后，随着数字化的文本增多以及再后来社交媒体的兴起，越来越多的研究者开始研究这个问题。除了研究单一的评论文本，一些论坛的帖子和文章的评论之间是有类似对话的上下文关系，评论者除了表达观点之外也会表达对其他评论者的赞同/反对的立场(position)。此外研究方向也逐渐扩大到对评论者的分析，比如研究怎么给评论者建档案(profile)。而且随着大家对于情感/舆情的重视，也出现了很多作弊的手段(水军)，怎么去除这些非正常的评论也逐渐变得重要，这就不仅仅是一个NLP的问题，它往往还需要分析用户的行为数据才能作出更准确的判断。

### 研究问题


#### 分析的层级

情感分析通常可以分为三个层级：文档(document)级别、句子(sentence)级别和属性(aspect)级别。下面我们简单的介绍一下这三个层级。

**文档级别**，它把整个文档的情感分为正面或者负面。这可以看出一个二分类问题(不太可能有评论没有情感，因为写评论的目的就是要表达情感，即使是很弱的情感)，因此它也被叫做文档级别情感分类。比如对于一个商品的评论，我们可以分析整体上来看这个评论的情感是正面还是负面。它假设这个文档的所有评论句子针对的都是同一个实体。如果一个文档会点评多个实体，这个级别的分析就是有问题的。

**句子级别**，它对文档的每个句子判定它的情感是正面的、负面的还是中立的(neutral)。和前面的文档级别不同，一些描述性的句子其实是没有任何情感的，所以这里多了一个中立的分类，它表示没有情感。这个问题和主客观分类(subjectivity classification)有一些关系——主客观分类是判断一个句子是主观的(subjective)还是客观的(objective)。通常中立的句子是客观的，而正面或者负面情感的句子是主观的。但是它们并不是完全一样的。比如"We bought the car last month and the windshield wiper has fallen off"是客观的句子，但是它描述了一个不想要(undesirable)的事情，所以它隐含了负面的情感。而句子"I think he went home after lunch"虽然是主观的，但是它没有正面或者负面的情感。

**属性级别**，不管是文档级别的还是句子级别的分析都是不够精细的，比如句子级别的分析可以判断句子"I like the iPhone 5"的情感是正面的，但是它没有说明观点的对象(opinion target)是实体(entity) "iPhone 5"；而句子"The battery life of iPhone 5 is very long"的观点对象是实体"iPhone 5"的一个属性battery life。此外有点句子比如"Although the service is not great, I still love this restaurant"包含了两个观点——酒店的服务不行，酒店整体不错。从句子级别来看，上面的句子既有正面的又有负面的，而且正面的是放在转折(but)的子句里，所以也许我们从整体上可以判断句子的情感还是正面为主。但是从应用的角度来说，如果某个用户很关注服务，他看到这个评论后可能就不会选择这家酒店。最早属性是使用feature或者attribute这两词，但是在机器学习领域feature有特殊的含义，而attribute又是一个太宽泛的词汇，后来就使用了Aspect(方面)这个词来表示情感分析对象实体的某个具体属性。



除了按照层级分类，我们还可以把评论分析分类为普通的评论分析和比较的(comparative)评论分析。比如"Coke tastes very good"是一个普通的情感表示，它表示了说话人对实体Coke的味道。而"Coke tastes better than Pepsi"是一个比较的评论，它说明Coke的味道要比Pepsi好。

#### 基于情感词典的方法

判断情感倾向最重要的就是情感词语(sentiment word)，也叫做观点词语(opinion word)。比如"good"、"wonderful"和"amazing"是正面的情感词语，而"bad"、"poor"和"terrible"是负面的情感词语。除了词外，还有一些短语也能表达情感倾向，比如"cost an arm and a leg"(花费高昂代价)。因此构造一个情感词典来用于情感分类(规则或者把词典作为特征)是非常自然的想法。但是情感分析是一个非常复杂的问题，光参考情感词语或者短语是远远不够的。它不能解决如下的一些问题：

* 一个词在不同的上下文下的情感方向(orientation)/极性(polarity)是不同的。

    * suck这个词，通常表示负面的情绪，比如句子"This camera sucks"。但是在句子"This vacuum cleaner really sucks"里sucks是说明吸尘器很能吸，这是正面的。另外有一些词的极性依赖与修饰的名词，比如"大"没有特定的极性，说酒店房间大则是正面的，而说噪声大就是负面的。

* 包含情感词语的句子不见得有情感倾向

    * 比如句子"Can you tell me which Sony camera is good?"和"If I can find a good camera in the shop, I will buy it."这两个句子没有任何情感倾向。第一个句子是疑问句，一般(但不是绝对)不包含情感倾向；第二个句子good是在条件从句，所以也不表示情感。注意，并不是所有的疑问句和条件从句都没有情感，比如"Does anyone know how to repair this terrible printer?"和"If you are looking for a good car, get a Ford Focus."都有情感。

* 反讽和期望的句子很难处理

    * 比如"What a great car! It stopped working in two days."和"要是房间再大一点就好了"，如果只看情感词语，great car和"房间大"都是正面的，但是实际语义恰恰相反。

* 没有情感词语的句子也能表达情感

    * 比如"This washer uses a lot of water"没有情感词语，它是一个客观的陈述句，但是根据常识，它描述的是我们不想要的现象，因此可以推测这是一个负面的情感。

上面只是情感分析的部分困难示例，这些困难用情感词典很难解决。

#### Debate和Comment

社交媒体上有两种情感文本：独立的帖子(post)，比如review和blog；在线的对话，包括debate和comment。比如论坛的帖子，大家会对某个话题进行辩论(debate)。注意：review和comment翻译成中文都是评论，但是这里用review表示单个的独立的表达观点和情感的文档，而comment表示那种对话式的评论。comment可以是对主题贴(review)的看法——赞同或者反对，也可以针对评论的对象的提问，甚至是对前面提问的回答。这里涉及多个人，可能是主题贴的作者和评论者的对话，也可能是评论者之间的对话。

#### 挖掘意图(Mining Intention)

意图(Intention)是人们想要采取的行为(action)。意图和情感是两个不同的概念，但是它们也有一定的关联。比如"I am dying to see Life of Pi"，说话人想要看电影的意图非常强烈，说明它对这部电影很可能就是正面的情感。类似的，"I want to buy an iPhone 5"，想要购买某个商品可以推测他对这个商品是正面的情感。此外，某些表达意图的句子就是在表达情感，比如"我想把这个手机扔了"和"我想把手机退了"，明显就在表达非常负面的情绪。

#### 评论的Spam Detection和评论质量评估

社交媒体的优点是可以匿名的自由发表言论，但是这也会让一些不法分子有可乘之机。在各种利益的驱使下，很多人会发表"假的"评论来提高某些商品的评价得分。这些人被叫做opinion spammer，而他们的行为叫做opinion spamming。为了检测这些评论和评论者，我们不仅仅需要用的文本的语义内容(比如为了刷榜的评论一般比较短，用比较空虚的句子，当然也有很专业的五毛党写的让你看不出来)，还需要使用用户的行为数据。与之相关的一个问题是评估评论的质量，它的作用是把好的评论排序在前面让用户看到，另外在排序商品的时候一般也需要更多参考质量高的评论。

**未完待续**




