---
layout:     post
title:      "情感分析常见数据集介绍" 
author:     "lili" 
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - 深度学习
    - 情感分析
    - sentiment analysis
---

本文介绍情感分析领域最常见的一些数据集。

<!--more-->

**目录**
* TOC
{:toc}

## SemEval

###  SemEval-2014 Task 4: Aspect Based Sentiment Analysis

任务的介绍主要参考了[SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004.pdf)，官方网站为[SemEval-2014 Task 4](http://alt.qcri.org/semeval2014/task4/)。

这是SemEval-2014语义评测任务的第4个任务，它又包含4个子任务。

#### 子任务1：Aspect term extraction

给定针对某个entity(比如餐馆)的一些句子，识别其中的aspect term。比如句子"The food was nothing much, but I loved the staff"，我们需要识别"food"和"staff"这两个aspect term。一个句子里可能会出现多个(或者零个)aspect term。另外aspect term可能包含多个词，比如"The hard disk is very noisy"，这里的aspect term是"hard disk"。

#### 子任务2：Aspect term的极性分类

给定一个句子和这个句子里的所有aspect term，判定每一个term的情感极性。可能的极性包括正面(positive)、负面(negative)、中性(neutral)和冲突(conflict)。比如：

“I loved their **fajitas**” → {fajitas: positive}
“I hated their **fajitas**, but their **salads** were great” → {fajitas: negative, salads: positive}
“The **fajitas** are their first plate” → {fajitas: neutral}
“The **fajitas** were great to taste, but not to see” → {fajitas: conflict}


冲突的意思是在这个句子里既有正面的评价也有负面的评价，比如上面的第四个句子。

#### 子任务3：Aspect类别(category)识别

因为很多不同的aspect term都可以归为一类，比如fajitas和salads都是餐馆的菜品，我们希望把它们都归类到food。这个任务定义了几个类别，比如餐馆(restaurant)的数据集上定义里food, service, price, ambience, anecdotes/miscellaneous等5个类别。这个任务为：给定一个句子，识别出其中的类别(注意一个句子可能包含多个类别)。比如：

“The restaurant was too expensive”  → {**price**}
“The restaurant was expensive, but the menu was great” → {**price**, **food**}

有的读者可能回想，如果能识别aspect term，然后再判断aspect term是哪个category。这可能有一个问题，对于隐式的aspect，可能只有形容词而没有名词，比如第一个句子没有price这样的aspect term，我们需要根据形容词expensive来推测类别为price。

#### 子任务4：Aspect类别的情感分类

给定一个句子以及句子里的一个或者多个aspect类别，输出每个类别的情感分类。和前面的term分类一样，这里的分类也是正面(positive)、负面(negative)、中性(neutral)和冲突(conflict)。比如：

“The restaurant was too expensive” → {price: negative}
“The restaurant was expensive, but the menu was great” → {price: negative, food: positive}

对于上面的第一个例子，输入是句子和negative与food两个类别，输出是这两个类别的极性。

#### 示例数据

全部数据可以在[这里](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)下载，它包括餐馆和笔记本电脑两个数据集，其中餐馆数据集包含上面的4个子任务的标注，而笔记本电脑的数据只有前两个任务的标注数据(没有类别的标注)。

下面是餐馆的一个示例数据：

```
<sentence id="813">
          <text>All the appetizers and salads were fabulous, the steak was mouth watering and the pasta was delicious!!!</text>
          <aspectTerms>
                    <aspectTerm term="appetizers" polarity="positive" from="8" to="18"/>
                    <aspectTerm term="salads" polarity="positive" from="23" to="29"/>
                    <aspectTerm term="steak" polarity="positive" from="49" to="54"/>
                    <aspectTerm term="pasta" polarity="positive" from="82" to="87"/>
          </aspectTerms>
          <aspectCategories>
                    <aspectCategory category="food" polarity="positive"/>
          </aspectCategories>
</sentence>
```

下面是笔记本电脑的示例：

```
<sentence id="353">
          <text>From the build quality to the performance, everything about it has been sub-par from what I would have expected from Apple.</text>
          <aspectTerms>
                    <aspectTerm term="build quality" polarity="negative" from="9" to="22"/>
                    <aspectTerm term="performance" polarity="negative" from="30" to="41"/>
          </aspectTerms>
</sentence>
```




###  SemEval-2015 Task 12: Aspect Based Sentiment Analysis


任务的介绍主要参考了[SemEval-2015 Task 12: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S15-2082.pdf)，官方网站为[SemEval-2015 Task 12](http://alt.qcri.org/semeval2015/task12/)。

这是SemEval-2014任务的任务假定评论的都是给定实体(餐馆或者笔记本电脑)的某个属性，但是我们也可能点评这个实体的部件，比如笔记本电脑的鼠标。前面介绍过，aspect更加通用的表示方法是一棵树。不过这里的任务还是简化里一些，认为这棵树最多两层，树根是餐馆或者笔记本电脑，我们可以点评电脑的属性(比如价格)，也可以点评部件鼠标的属性(比如鼠标的灵敏度)。此外，有一些点评aspect的句子并不见得会出现对应的名词，比如下面的文字：

```
They sent it back with a huge crack in it and it still didn't work; and that was the fourth time I’ve sent it to them to get fixed
```

它点评的实体是餐馆的服务(service)，属性是服务的质量(quality)，但是文字中没有任何service或者quality相关的文字。这和前面的expensive的句子类似的。因此2015年的任务预定义里所有的Entity和属性，然后让我们识别文本中出现里哪些实体和属性的组合，也就是E#A。比如上面的句子，输出就是service#quality。另外这个任务的输入不是一个一个的句子，而是整段评论，这样我们可以利用上下文信息。当然标注和识别的粒度还是句子，只不过我们的算法可以(但大部分算法都没有)利用上下文的信息。



#### 任务1：In-domain任务

给定一个完整的评论，我们需要完成如下3个子任务。

##### Aspect类别识别

识别评论里所有的实体(E)和属性(A)对。E和A都是预定义集合中的某一个值，比如餐馆数据集，E包含laptop, keyboard, operating system, restaurant, food, drinks等实体和performance, design, price, quality等属性。

更具体的，对于笔记本电脑数据集来说，E共用22个实体类别(比如LAPTOP, DISPLAY, CPU, MOTHERBOARD, HARD DISC, MEMORY, BATTERY等)和9个属性标签(比如GENERAL, PRICE, QUALITY, OPERATION_PERFORMANCE等)。完整的实体列表和属性标签列表可以参考[这里](http://alt.qcri.org/semeval2015/task12/data/uploads/semeval2015_absa_laptops_annotationguidelines.pdf)，下面是一些示例：

```
(1)  It fires up in the morning in less than 30 seconds and I have never had any issues with it freezing. → {LAPTOP#OPERATION_PERFORMANCE}
(2)  Sometimes you will be moving your finger and the pointer will not even move.  → {MOUSE#OPERATION_PERFORMANCE}
(3)  The backlit keys are wonderful when you are working in the dark.  → {KEYBOARD#DESIGN_FEATURES}
(4)  I dislike the quality and the placement of the speakers. {MULTIMEDIA DEVICES#QUALITY}, {MULTIMEDIA DEVICES#DESIGN_FEATURES}
(5)  The applications are also very easy to find and maneuver.  → {SOFTWARE#USABILITY}
(6)  I took it to the shop and they said it would cost too much to repair it.  → {SUPPORT#PRICE}
(7)  It is extremely portable and easily connects to WIFI at the library and elsewhere. → {LAPTOP#PORTABILITY}, {LAPTOP#CONNECTIVITY}
```

比如第一个句子是说笔记本的操作响应很快，而第二个是说鼠标的操作很不灵敏。

对于餐馆数据集来说，E有6个实体类别(RESTAURANT, FOOD, DRINKS, SERVICE, AMBIENCE, LOCATION)和5个属性标签(GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS)，详细信息可以参考[这里](http://alt.qcri.org/semeval2015/task12/data/uploads/semeval2015_absa_restaurants_annotationguidelines.pdf)，下面是一些示例：

```
(1) Great for a romantic evening, but over-priced. → {AMBIENCE#GENERAL}, {RESTAURANT#PRICES}
(2) The fajitas were delicious, but expensive. → {FOOD#QUALITY}, {FOOD# PRICES}
(3)The exotic food is beautifully presented and is a delight in delicious combinations. → {FOOD#STYLE_OPTIONS}, {FOOD#QUALITY}
(4) The atmosphere isn't the greatest , but I suppose that's how they keep the prices down. → {AMBIENCE#GENERAL}, {RESTAURANT# PRICES}
(5) The staff is incredibly helpful and attentive. → {SERVICE# GENERAL}
```


##### Opinion Target Expression（OTE)识别

这个任务只有餐馆数据集上有标注数据。OTE任务的输入是所有的E#A对，需要识别E#A对里实体E对应的字符串。当隐式的表达实体时用特殊的"NULL"表示，比如代词"它"这样的代词，有的文本甚至根本找不到和E相关的字符串。下面是一些例子：

```
(1) Great for a romantic evening, but over-priced. → {AMBIENCE#GENERAL, “NULL”}, {RESTAURANT# PRICES, “NULL”}
(2) The fajitas were delicious, but expensive. → {FOOD#QUALITY, “fajitas”}, {FOOD# PRICES, “fajitas”}
(3) The exotic food is beautifully presented and is a delight in delicious combinations. → {FOOD#STYLE_OPTIONS, “exotic food”}, {FOOD# QUALITY, “exotic food”}
(4) The atmosphere isn't the greatest , but I suppose that's how they keep the prices down. → {AMBIENCE#GENERAL, “atmosphere”}, {RESTAURANT# PRICES, “NULL”}
(5) The staff is incredibly helpful and attentive. → {SERVICE# GENERAL, “staff”}
```
比如在第4个句子里，they指代的是餐馆，但是它不是OTE。

##### 情感分类

给定一个句子(有上下文)和所有的E#A对，判断其情感分类，可能的分类为正面、负面和中性。这和2014年的任务有所不同。下面是一些示例：

```
(1) The applications are also very easy to find and maneuver. → {SOFTWARE#USABILITY,  positive}
(2) The fajitas were great to taste, but not to see”→ {FOOD#QUALITY, “fajitas”, positive},  {FOOD#STYLE_OPTIONS, “fajitas”, negative }
(3) We were planning to get dessert, but the waitress basically through the bill at us before we had a chance to order.  → {SERVICE# GENERAL, “waitress”, negative}
(4) It does run a little warm but that is a negligible concern.  → {LAPTOP#QUALITY neutral}
(5) The fajitas are nothing out of the ordinary” → {FOOD#GENERAL, “fajitas”,  neutral}
(6) I bought this laptop yesterday. → {}
(7) The fajitas are their first plate  → {}
```

#### 任务2：Out-of-domain任务

增加里一个酒店的测试数据集(没有训练数据)，然后考察模型则不同领域的泛化能力。它的输入是E#A对和句子，要求我们输出这个E#A对的情感极性。

下面是一个完整的评论的标注数据：

```
Review id:"1004293"

 Judging from previous posts this used to be a good place, but not any longer.
 {target:"NULL" category:"RESTAURANT#GENERAL" polarity:"negative" from:"-" to="-"}

 We, there were four of us, arrived at noon - the place was empty - and the staff acted 
 like we were imposing on them and they were very rude.
 {target:"staff" category:"SERVICE#GENERAL" polarity:"negative" from:"75" to:"80"}

 They never brought us complimentary noodles, ignored repeated requests for sugar, 
 and threw our dishes on the table.
 {target:"NULL" category:"SERVICE#GENERAL" polarity:"negative" from:"-" to:"-"}

 The food was lousy - too sweet or too salty and the portions tiny.
 {target:"food" category="FOOD#QUALITY" polarity="negative" from:"4" to:"8"}
 {target:"portions" category:"FOOD#STYLE_OPTIONS" polarity:"negative" from:"52" to:"60"}

 After all that, they complained to me about the small tip.
 {target:"NULL" category:"SERVICE#GENERAL" polarity:"negative" from:"-" to:"-"}
			
 Avoid this place!
 {target:"place" category:"RESTAURANT#GENERAL" polarity:"negative" from:"11" to:"16"}

```

其中from和to表示OTE字符串在句子开始和结束的下标。


###  SemEval-2016 Task 5: Aspect Based Sentiment Analysis


任务的介绍主要参考了[SemEval-2016 Task 5: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S16-1002.pdf)，官方网站为[SemEval-2016 Task 5](http://alt.qcri.org/semeval2016/task5/)。


2016年的任务延续里2015年的任务，为它增加了新的测试数据(15年的训练数据和测试数据都变成16年的训练数据)，此外它还首次加入了英语之外的多种语言，包括中文。它包括如下几个子任务：

#### 句子级别的ABSA(Aspect-Based Sentiment Analysis)

给定某个实体(笔记本电脑、餐馆或者酒店)的一篇评论的一个句子，需要确定所有观点三元组的如下内容(slot)：

##### Aspect Category Detection

这个任务是确定文本里所有出现的E#A对，其中E来自预定义的实体类列表，A来自预定义的属性标签列表。

##### Opinion Target Expression (OTE)

和上年的任务一样，需要确定每个E#A对里实体对应的字符串的开始和结束下标，如果找不到则输出"NULL"。只有餐馆的数据有这个子任务。

##### 情感极性

判断每一个E#A对的情感分类，类别包括正面、负面和中性。

上面的任务在人工标注时每次处理一个句子，但是会参考它的前后上下文的其它句子。下面是一个笔记本的评论的标注示例：

```
S1:The So called laptop Runs to Slow and I hate it! →
{LAPTOP#OPERATION_PERFORMANCE, negative}, {LAPTOP#GENERAL, negative}
S2:Do not buy it! → {LAPTOP#GENERAL, negative}
S3:It is the worst laptop ever. → {LAPTOP#GENERAL, negative}
```

下面是餐馆的数据：

```
S1:I was very disappointed with this restaurant. →
{RESTAURANT#GENERAL, “restaurant”, negative, from="34" to="44"}
S2:I’ve asked a cart attendant for a lotus leaf wrapped rice and she replied back rice and just walked away. →{SERVICE#GENERAL, “cart attendant”, negative, from="12" to="26"}
S3:I had to ask her three times before she finally came back with the dish I’ve requested. →
{SERVICE#GENERAL, “NULL”, negative}
S4:Food was okay, nothing great. →
{FOOD#QUALITY, “Food”, neutral, from="0" to="4"}
S5:Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and rude family. →
{FOOD#QUALITY, “Chow fun”, negative, from="0" to="8"},
{FOOD#QUALITY, “pork shu mai”, negative, from="18" to="30"},
{AMBIENCE#GENERAL, “NULL”, negative}
S6:I/we will never go back to this place again. →
{RESTAURANT#GENERAL, “place”, negative, from="32" to="37"}
```

#### 文本级别的ABSA 

上面的句子级别的问题是模型不能参考上下文(人工标注是参考了的)，因此还有一个文本级别的ABSA任务。它的任务和前面是一样的，只不过输入是整个评论文本，下面是一些示例。

下面是整个评论文本：

```
Review id:LPT1 (Laptop)
"The So called laptop Runs to Slow and I hate it! Do not buy it! It is the worst laptop ever."
```

期望的输出(标注)为：
```
{LAPTOP#OPERATION_PERFORMANCE, negative}
{LAPTOP#GENERAL, negative}
```

但是它并不能简单的把文本分成句子，然后把所有句子的结果合并起来，因为一个段落里可能有多个句子都在说同一个E#A对，如果是这样的话需要判断最主要的情感倾向，比如下面的例子：

```
Review id:RST1 (Restaurant)
"I was very disappointed with this restaurant. I’ve asked a cart attendant for a lotus leaf wrapped rice and she replied back rice and just walked away. I had to ask her three times before she finally came back with the dish I’ve requested. Food was okay, nothing great. Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and rude family. I/we will never go back to this place again."
```

它的输出为：

```
{RESTAURANT#GENERAL, negative}
{SERVICE#GENERAL, negative}
{FOOD#QUALITY, negative}
{AMBIENCE#GENERAL, negative}
```

它就是前面句子级别的同一段文本，关于FOOD#QUALITY有一个中性两个负面的，因此总的情感倾向是负面的。如果多个句子的情感倾向是冲突的，比如一个正面一个负面，则需要识别为冲突(conflict)。比如下面的例子：

```
Review id: RST2 (Restaurant)
“This little place has a cute interior decor and affordable city prices. The pad seew chicken was delicious, however the pad thai was far too oily. I would just ask for no oil next time.”
```

它的输出为：

```
{AMBIENCE#GENERAL, positive}
{RESTAURANT#PRICES, positive}
{FOOD#QUALITY, conflict}
{RESTAURANT#GENERAL, positive}
```

FOOD#QUALITY既有正面的又有负面的，因此标注为冲突。

#### Out-of-domain ABSA

这个任务的测试数据的领域没有训练数据，它考察的是模型在不同领域的泛化能力。
 
## IMDB
 
电影的评论数据，二分类任务，包括25,000个训练数据和25,000个测试数据。可以在[这里](http://ai.stanford.edu/~amaas/data/sentiment/)下载。
