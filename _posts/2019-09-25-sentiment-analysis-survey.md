---
layout:     post
title:      "情感分析简介" 
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

## 问题的形式化定义

### 观点(Opinion)的定义

如前面所述，情感分析主要关注表达正面或者负面的情感的观点。我们用观点(Opinion)表示非常广的概念，它包括情感(sentiment)、评价和态度以及评价的对象(opinion target)和评论者(opinion holder)。而情感(sentiment)只表示观点(opinion)的情感极性是正面还是负面的。

#### Opinion的定义

我们使用下面的例子来说明Opinion的定义，为了方便引用对每个句子做了编号：

```
评论者：John Smith            日期：September 10, 2011
(1) I bought a Canon G12 camera six months ago. 
(2) I simply love it. 
(3) The picture quality is amazing. 
(4) The battery life is also long. 
(5) However, my wife thinks it is too heavy for her.
```

上面的评论对于产品Canon G12有许多评论。第2个句子对于它的整体持正面的评价。第3个句子对它的picture equality属性持正面评价。第4个句子对它的battery life属性也是正面评价。而第5个句子对于它的weight属性持负面评价。

一个观点有两个要素：对象(target)和情感(sentiment)。我们用(g,s)来表示这两个要素，其中g可以是某个实体(比如Canon G12)或者实体的某个属性(比如Canon G12的battery life)；而s表示情感极性，它的取值可以是正面的、负面的和中性的或者连续的实数值，比如Canon G12的battery life是正面的。

另外观点还包含观点的持有者(评论者)，比如第2个句子的观点持有者是"我"而第5个句子的持有者是"my wife"。

最好观点还有一个时间，这样我们可以分析大量观点随时间的变化规律。在上面的例子里，观点的(发布)时间是2011年9月10日。

因此，观点的定义是一个四元组(g, s, h, t)，其中：

* g是情感的对象
* s是情感
* h是持有者
* t是时间

这里需要说明一下的是观点有一个对象(target)，这有两个用处。首先对于一个包含两个对象的句子，我们可以分开处理。比如句子"Apple is doing very well in this poor economy"同时保护两个观点，如果用句子级别的情感分类是没有办法处理的。有了观点的对象，我们知道对Apple的情感是正面的，而对economy的是负面的。第二个好处是情感词语的极性有时和对象有关，比如"大"和"房间"搭配时就是正面的而与"噪声"搭配就是负面的。

这里定义的是普通的观点，对于比较性的观点需要不同的定义，后面我们再介绍。

#### 情感对象(sentiment target)

* 情感对象

    * 情感对象(sentiment target)也叫观点对象(opinion target)，是这个观点针对的实体或者实体的某个属性。

比如前面的例子里的第3个句子，情感对象是Canon G12的picture quality属性。注意：虽然句子中没有出现Canon G12，但是它隐含指代的对象是Canon G12。

* 实体(entity)

    * 一个实体e可能是一个产品、服务、主题、人、组织机构等。它可以表示为一个pair e=(T,W)，其中T是它的部件(part)的集合，而每一个部件又是一个实体(递归定义)；W是实体e的属性集合。e的部件或者子部件也可能有属性。

 
比如Canon G12这个特定的照相机就是一个实体。它包含picture qulity、size和weight等属性，同时它包含lens、viewfinder和battery等部件。battery也是一个实体，它包含battery life和battery weight等属性。上面的定义通过part-of关系定义了对象的树状层次结构。根节点是实体，比如Canon G12。我们评论的对象可以是某个子部件的属性，比如battery的battery life这个属性。当然我们也可以评论子部件整体，比如说battery很好。也就是说，这棵树的每一个节点(实体)和节点的属性都可以成为评论的对象。

#### 观点的情感(opinion sentiment)

* 观点的情感是某个观点表达的感受、态度、评价或者情绪。它可以表示为三元组(y, o, i)，其中
    * y是情感类型(type)
    * o是情感的极性
    * i是情感的强度

情感类型可以分为理性情感和感性情感。

* 理性(rational)情感
    * 理性的情感来自于理性的推理、实际的信念和实用主义的态度。它不(直接)包含强烈的个人情绪。

比如"The voice of this phone is clear."和"This car is worth the price."这两个句子就是理性的情感类型。

* 感性(emotional)情感
    * 感性情感来自于人内心深处的心理学状态，它是对某个实体无形的和情绪的反应。

比如句子"I love the iPhone", "I am so angry with their service people", "This is the best car ever"和"After our team won, I cried."就是感性的情感。感性的情感通常要比理性的情感更加强烈，对于营销来说，我们期望用户的情感能上升到感性层次，比如用户说"iPhone is good"，那他还不见得会买单，但是如果他说"I love iPhone"，那么离转化就不远了。

情感倾向(sentiment orientation)可以是正面的(positive)、负面的(negative)和中立的(neutral)。情感倾向也被叫做情感极性(polarity)和立场(valence)。

情感强度(sentiment intensity)表示情感的强烈程度。比如good要比excellent弱，而dislike要比detest弱。理论上我们可以用一个连续的实数值来表示情感的强弱，但是实际人都很难做这么细粒度的区分。因此在实际应用中，我们通常把情感强度分为5级别，比如1星到5星：

* 5星 感性的正面情感
* 4星 理性的正面情感
* 3星 中立的情感
* 2星 理性的负面情感
* 1星 感性的负面情感

#### 简化的观点定义

前面的观点定义关于实体的定义包含递归，使用起来过于复杂，这里我们介绍简化版本的定义。我们这里把对象简化成实体和aspect，也就是把子部件和属性都称为aspect。对于观点的情感(sentiment)，不管是正面的、负面的和中立的还是1星到5星，我们都可以用一个数值来表示，比如-1表示负面的而0表示中立的而1表示正面的。

这样我们得到简化版本的观点定义，它是一个五元组(e, a, s, h, t)，其中：

* e代表实体(entity)
* a代表aspect，可以是实体的属性，也可以是子部件或者子部件的属性
* s表示一个数值，表示不同的情感
* h是观点的持有者
* t是时间

如果评论的对象是entity这个整体，比如"I love iPhone 5"，则entity是"iPhone 5"，而aspect是特殊的"GENERAL"。上面这样定义的情感分析叫做基于aspect的情感分析(aspected based sentiment analysis)，用首字母缩写为ABSA。

#### 观点的理由(reason)和限定词(qualifier)

除了前面的五元组，我们还可以提前观点里的更多信息。比如句子"This car is too small for a tall person"，实体是"this car"，aspect是size，sentiment是负面的，观点持有者是说话人，时间是说话的时间。除此之外，我们还可以分析得到持有这个观点的理由是"too small"，并且这个观点还有一个限制条件"for a tall person"。因此对于一个不那么高的人来说，也许这个车的size并不是负面的。

* 观点的理由
    * 观点的理由是持有这个观点的理由或者解释

对于很多应用来说，知道理由是很有用的。比如"do not like the picture quality of this camera"只告诉我们照相机的拍摄质量有问题，但是"I do not like the picture quality of this camera because the pictures are quite dark."还能告诉我们拍摄质量不好的解释(现象)是因为拍出来的照片太黑，这可以指导厂家改进产品。

* 观点的限定词
    * 观点的限定词是一些限制条件，在满足这些条件下此观点才成立

比如"This car is too small for a tall person"，它说明只要对于高个来说车太小，但是并不是对所有的人来说都小。个子并不那么高的人可能不会在意这一点。并不是每个观点都有理由或者限定词，而且理由和限定词也可能和评论的主要内容出现在不同的句子里，这也会让抽取变得更加困难。比如"The picture quality of this camera is not great. Pictures of night shots are very dark."，理由出现在第二个句子里。而"I am six feet five inches tall. This car is too small for me."的限定词出现在地一个句子里。


#### 情感分析的目标和任务

有了前面观点的定义之后，我们就可以定义情感分析的目标和任务了。给定一个文档d，情感分析的目标就是找出其中所有的观点五元组(e, a, s, h, t)。对于更复杂的场景，我们可能还需要抽取观点的理由和限定词。

第一个抽取任务就是识别五元组中的实体。这和NLP的命名实体识别(NER)有一点关系，比如产品名称等通常就是命名实体。但是观点里的实体并不都是命名实体，比如"I hate tax increase"这个观点的实体是抽象的概念"tax increase"，这不是命名实体。

识别出实体之后，我们还需要"归一化"。我们需要定义一个实体类别(entity category)，比如iphone_5(它只是一个ID)，而它的不同文字描述比如"iPhone5"，"iphone 5"和"苹果5"都是对应这个实体类别。这些不同的文字叫做实体表示(entity expression)或者entity mention(不知道咋翻译好)。

aspect也是一样，比如camera的"picture", "image"和"photo"指得都是同一个aspect。因此我们也有aspect category和aspect expression。

aspect expression通常都是名词和名词短语，但是也可以是动词、形容词、副词等其它词类。aspect expression可以分为显式的和隐式的。

* 显式的aspect expression
    * 显式的aspect expression是文本里的名词或者名词短语。比如"The picture quality of this camera is great"中，picture quality这个aspect expression是名词短语。

* 隐式的aspect expression
    * 隐式的aspect expression通常通过形容词来隐含的表示。比如"This camera is expensive"，对应的aspect是price，这是通过形容词expensive来推断出来的。除了形容词，动词也可以隐含aspect，比如"I can install the software easily"，通过动词install可以推断出aspect是installation。此外还有更加复杂的隐式aspect expression，比如"This camera will not easily fit in my pocket"，能放到口袋里(fit in my pocket)说明它的size这个aspect；而"This restaurant closes too early"说明餐馆的下班时间这个aspect。为了理解这样的aspect，我们还需要世界知识，这是非常困难的。


下面我们定义情感分析的具体任务。

* 实体抽取(entity extraction)和指代消解(resolution)
    * 识别文档d里的所有entity expression，然后根据指代是否同一个实体把它们分组起来(或者叫归一化成一个ID)。

* aspect抽取和指代消解
    * 和实体类似，只不过处理aspect

* 观点持有者的抽取和指代消解
    * 抽取观点的持有者并进行指代消解

* 时间抽取和归一化

* aspect的情感分类/回归
    * 判断某个aspect的情感分类或者预测连续的情感强度

* 生成所有的观点五元组

* 抽取观点的原因

* 抽取观点的限定词

最后两个任务是非常困难的，而且很多观点并不包含，在实际应用中很少会处理它们。下面我们通过一个例子来说明不同的任务需要做哪些事情。我们的例子为：

```
Review B: Posted by bigJohn     Date: September 15, 2011
(1) I bought a Samsung camera and my friend brought a Canon camera yesterday.
(2) In the past week, we both used the cameras a lot. 
(3) The photos from my Samy are not clear for night shots, and the battery life is short too. 
(4) My friend was very happy with his camera and loves its picture quality. 
(5) I want a camera that can take good photos. 
(6) I am going to return it tomorrow.
```

第1个任务需要抽取entity expression Samsung和Samy，然后把它们放到一个cateory里面，而Canon是另外一个category。第2个任务需要抽取picture, photo和battery life，并且把picture和photo放到一起。而第3个任务需要抽取第3句话对应的观点的持有者是bigJohn，第4句话对于观点的持有者是bigJohn的朋友。任务4需要抽取时间为"September 15, 2011"。任务5需要识别第3句话对于Samy的photo这个aspect是负面的，battery life也是负面的；而第4个句子关于Canon整体(GENERAL这个aspect)是正面的，关于它的picture这个aspect也是正面的。

为了完成任务5，我们还需要知道句子his camera和its指代的是哪个实体。任务5应该生成如下的五元组：

```
1. (Samsung, picture_quality, negative, bigJohn, Sept-15-2011)
2. (Samsung, battery_life, negative, bigJohn, Sept-15-2011)
3. (Canon, GENERAL, positive, bigJohn’s_friend, Sept-15-2011)
4. (Canon, picture_quality, positive, bigJohn’s_friend, Sept-15-2011)
```

如果再加上任务6和7，则生成的结果为：

```
1. (Samsung, picture_quality, negative, bigJohn, Sept-15-2011)
    Reason for opinion: picture not clear
    Qualifier of opinion: night shots
2. (Samsung, battery_life, negative, bigJohn, Sept-15-2011)
    Reason for opinion: short battery life
    Qualifier of opinion: none
3. (Canon, GENERAL, positive, bigJohn’s_friend, Sept-15-2011)
    Reason for opinion: none
    Qualifier of opinion: none
4. (Canon, picture_quality, positive, bigJohn’s_friend, Sept-15-2011)
    Reason for opinion: none
    Qualifier of opinion: none
```

### 感受(Affect)、情感(Emotion)和情绪(Mood)

接下来我们讨论一下感性(emotional)情感，它涉及感受(Affect)、情感(Emotion)和情绪(Mood)。我们这里只是介绍这些概念的区别和联系，因为这些都是人的精神状态，很多时候都是通过行为和表情来表现。即使通过语言来描述自己的这些感受，也通常是非常困难的。

首先我们来看这些概念字面上的定义，这里还增加了Feeling。

* Affect
    * 感受或者情感，通常会通过面部表情和身体语言表现出来

* Emotion
    * 

* Mood
    * 精神或者情感的状态

* Feeling
    * 意识的情感状态

从字面上很难区分这些概念，下面我们通过一些例子来体会它们的细微区别。Affect是比较基本的感受，比如在看一个恐怖电影，我们会感受到恐怖这种感觉。我们的大脑会处理这种基本的感觉，然后产生一些Emotion，比如哭泣和尖叫。因此，Emotion是Affect的表现，它通常有一个对象，比如一个人，一件事或者一个物体，它是非常强烈并且持续时间不会太长。Mood和Emotion类似，都是一种情感状态。但是它通常没有特定的对象，而且持续时间比较长，我们也甚至不会意识到它。比如开心的情绪，它不是针对某个对象。

Emotion和Mood的区别是：Emotion通常很强烈而很短暂，而Mood更加分散和持久。比如生气就是一种Emotion，我们可能会非常生气，并且对某一个人或者事生气，但是我们很难一直保持生气。生气这种Emotion可能会转变成暴躁的情绪，这种情绪可能会持续很长时间。Emotion是非常具体的，比如某个事情可能会让我们生气。而Mood是比较模糊的，我们的暴躁情绪不是针对某一件事，我们在暴躁的情绪下可能对任何事情都容易生气。

不同的心理学研究者会定义各种不同的Emotion和Mood，因为我们从计算的角度来说不太能够准确的定义它们，而且实际应用中也比较少，所以这里不再分析这些具体的Emotion和Mood了。

### 观点(Opinion)的分类

观点可以从各种不同的维度来分类，下面我们讨论一些常见的分类方法，目的是了解不同观点的特点，从而为后面的算法提供一些sense。

#### 普通观点和比较观点

* 普通观点
    * 我们一般讨论的观点都是普通观点，它是针对一个实体或者一个实体的属性进行描述，它有可以分为直接(direct)观点和间接(indirect)u观点两大类。

直接观点直接描述一个实体或者实体的属性，比如"The picture quality is great"这个句子直接描述"picture quality"这个属性的好坏。而间接观点通过它的后果等间接描述，比如"After injection of the drug, my joints felt worse"，这个句子并没有直接说药效，而是说它的后果造成我的关节变得更痛了来间接说明效果不好，这通常需要世界知识来判断。

* 比较观点
    * 比较观点会对比两个实体，比如"Coke tastes better than Pepsi"。

#### 主观性观点和事实隐含类(Fact-Implied)观点

观点本身是主观的个人的看法，因此很多时候是通过主观的句子来表达观点。但是如前面介绍的，通过客观的描述一些我们想要或者不想要的事实，也能间接的表达观点，比如说车买回来没两天就坏了。

#### 第一人称观点和非第一人称观点

用第一人称观点直接表达说话人的观点，而第三人称观点描述其他人的观点。





