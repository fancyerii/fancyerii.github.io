---
layout:     post
title:      "Huggingface Transformers学习(二)——文本分类" 
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

文本分类是最常见的NLP任务之一，很多任务都可以看成文本分类问题。比如对一条评论判断它是正面的还是负面的；判断一封邮件是否垃圾邮件；给定一篇新闻判断它是体育、娱乐还是政治的。本文使用Twitter的情感分类问题来介绍文本分类，为了让模型能够快速训练，我们选择了[DistilBERT](https://arxiv.org/abs/1910.01108)，它可以让我们快速的进行实验。

为了用一个预训练的模型(比如DistilBERT)对一个文本分类任务进行微调，我们需要完成下图所示的步骤：

<a name='img12'>![](/img/learnhuggingface/12.png)</a>


## Dataset
 　
本文我们将实验来自论文[CARER: Contextualized Affect Representations for Emotion Recognition](http://
dx.doi.org/10.18653/v1/D18-1404)的数据，文本来自Twitter的消息。和普通的情感分类不同，这个数据把情感分成了生气(anger)、厌恶(disgust)、恐惧(fear)、开心(joy)、悲伤(sadness)和惊讶(surprise)。

### 探索Hugging Face Datasets

我们将会用Datasets库从[Hugging Face Hub](https://github.com/huggingface/datasets)下载数据。使用list_datasets()函数可以列举出Hub上所有的数据集：

```
from datasets import list_datasets
all_datasets = list_datasets()
print(f"There are {len(all_datasets)} datasets currently available on the Hub")
print(f"The first 10 are: {all_datasets[:10]}")

There are 12532 datasets currently available on the Hub
The first 10 are: ['acronym_identification', 'ade_corpus_v2', 'adversarial_qa', 'aeslc', 'afrikaans_ner_corpus', 'ag_news', 'ai2_arc', 'air_dialogue', 'ajgt_twitter_ar', 'allegro_reviews']

```

我们可以看到目前位置，Hub上总共有12532个数据集，并且这里输出了前10个。为了加载特定的数据集，比如我们要用到的情感分类的数据集，我们用load_dataset函数，传入数据集的名字即可：

```
from datasets import load_dataset
emotions = load_dataset("emotion")
emotions
```

输出为：
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 16000
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 2000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 2000
    })
})
```
我们看到load_dataset函数返回的是DatasetDict对象，它类似Python的Dict。DatasetDict里的不同key代表了数据集的不同split，比如emotion数据集包含"train"、"validation"和"test"三个split。DatasetDict的value是Dataset对象，它包含了一个split的全部数据。我们可以获取某个split的数据：

```
train_ds = emotions["train"]
train_ds
```

输出为：
```
Dataset({
    features: ['text', 'label'],
    num_rows: 16000
})
```

我们可以看到这个Dataset对象包括两个feature(列)——text和label，分布代表文本和标签。另外可以看到train数据集有16000行。和普通的Python的dict类似，我们可以用len函数获得其大小：

```
len(train_ds)
16000
```

我们也可以用下标来访问某一行的数据：

```
train_ds[0]
{'text': 'i didnt feel humiliated', 'label': 0}
```
每一行都是一个dict，所有可能的key为：

```
train_ds.column_names
['text', 'label']
```

Dataset对象基于[Apache Arrow](https://arrow.apache.org/)，它是一种列存储格式(columnar format)，而且使用了Memory Mapping技术来实现高效的内存和磁盘的数据交互，从而可以用有限的内存处理更大的数据集。接着我们看一下features的具体信息：

```
print(train_ds.features)

{'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None)}
```

train_ds共有两列(两个feature)：text的类型是string，表示Twitter文本；label类型是ClassLabel，可能的label是['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']。

我们用可以Python标注是slice语法输出多行数据：
```
print(train_ds[:5])

{'text': ['i didnt feel humiliated', 'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake', 'im grabbing a minute to post i feel greedy wrong', 'i am ever feeling nostalgic about the fireplace i will know that it is still on the property', 'i am feeling grouchy'], 'label': [0, 0, 3, 2, 3]}
```

可以看到输出还是一个dict(不是list)，text对应的是5条数据的文本组成的list，label是5条数据的标签的id组成的list。我们也可以输出前五条样本的text：

```
print(train_ds["text"][:5])

['i didnt feel humiliated', 'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake', 'im grabbing a minute to post i feel greedy wrong', 'i am ever feeling nostalgic about the fireplace i will know that it is still on the property', 'i am feeling grouchy']
```

### 加载本地数据

很多时候，我们的数据都是本地的文件，比如csv文件，我们可以依然可以用load_dataset加载数据。比如我们可以先下载一个csv文件：

```
dataset_url = "https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt"
!wget {dataset_url}

!head -n 1 train.txt
i didnt feel humiliated;sadness
```
然后用load_dataset函数加载它：
```
emotions_local = load_dataset("csv", data_files="train.txt", sep=";", names=["text", "label"])
```

注意，load_dataset的第一个参数是csv，第二个参数是csv文件路径，sep是告诉它csv使用分号分隔，最好给两列数据起一个名字。我们看一些emotion_local：

```
emotions_local

DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 16000
    })
})

emotions_local['train'].features

{'text': Value(dtype='string', id=None),
 'label': Value(dtype='string', id=None)}
```

虽然只有一个文件，依然会返回一个DatasetDict，默认的split是train。如果我们看features的话，发现label是文本类型，而不是ClassLabel。

### Datasets和DataFrames的转换

虽然Dataset类提供了很多数据操作的功能，但是把它转换成Pandas的DataFrame会更加方便。Dataset提供了set_format()函数来把Dataset转换成DataFrame。注意：set_format只是提供类似DataFrame的接口，但是底层的数据结构还是Apache Arrow。我们随时可以用set_format()函数把它恢复成Dataset。

```
import pandas as pd
emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()
```

输出：
```
 	text 							label
0 	i didnt feel humiliated 				0
1 	i can go from feeling so hopeless to so damned... 	0
2 	im grabbing a minute to post i feel greedy wrong 	3
3 	i am ever feeling nostalgic about the fireplac... 	2
4 	i am feeling grouchy 	3
```

注意：set_format(type="pandas")只是把Dataset变成DateFrame的格式，但它本身并不是DataFrame，所以需要复制它创建新的DataFrame对象。我们可以check一下：

```
type(emotions["train"])
datasets.arrow_dataset.Dataset

type(df)
pandas.core.frame.DataFrame
```

上面的数据转换成Pandas的DataFrame后，ClassLabel类型的label变成了id，所以我们可以再增加一个列表示label的文本：

```
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)

df.head()

	text 							label	label_name
0 	i didnt feel humiliated 	0 	sadness
1 	i can go from feeling so hopeless to so damned... 	0 	sadness
2 	im grabbing a minute to post i feel greedy wrong 	3 	anger
3 	i am ever feeling nostalgic about the fireplac... 	2 	love
4 	i am feeling grouchy 	3 	anger
```

这里用到emotions["train"].features["label"]，前面讲过，它是ClassLabel对象，可以调用它的int2str把id变成字符串。当然也可以反过来把字符串变成id：
```
emotions["train"].features["label"].str2int("sadness")
0
```

### 数据分布

在训练模型之前，我们先分析一下数据本身是非常重要的，首先我们需要确定一下各个标签的数据是否均衡的。

```
import matplotlib.pyplot as plt
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()
```

<a name='img13'>![](/img/learnhuggingface/13.png)</a>

我们可以看到，数据是非常不平衡的，最多的标签是joy和sadness，它们比最低的surprise多了10倍。为了解决数据不平衡，通常有如下做法：

* 随机过采样数据量少的标签
* 随机欠采样数据量多的标签
* 收集更多数据使得它们平衡

我们这里为了简单不考虑这些平衡数据的方法，直接用不平衡的数据训练模型。感兴趣的读者可以参考[imbalanced-learn库](https://imbalanced-learn.org/stable/)。如果要自己实现过采样，一定要在切分训练集和测试集之后再采样，否则就会可能出现信息泄露(测试数据在训练数据中出现过)。

### 文本长度

Transformer模型通常有个最大文本长度的限制，比如BERT的最大Token数是512。如果文本过长，模型就必须truncate掉一些文本。所以我们很有必要分析一下手头的数据的长度分别情况。

```
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()
```

<a name='img14'>![](/img/learnhuggingface/14.png)</a>

我们看到大部分Twitter文本的长度是15个词左右，最长的文本离512个Token也差距很多，所以我们不用担心这个问题。好了，到此为止，我们的数据分析就告一段落了，所以我们把Dataset的格式从DataFrame恢复成默认的格式：

```
emotions.reset_format()
```

## 把文本转换成Token

DistilBERT这样的模型不能直接输入文本，我们需要把文本切分成更小的单元然后变成整数的ID。在介绍DistilBERT的文本切分之前，我们看一看最常见的基于字符(Character)和词(Word)的切分方法。

### 字符切分

最简单的方法就是把每一个字符当成一个基本单元：

```
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text) 
```

输出：
```
['T', 'o', 'k', 'e', 'n', 'i', 'z', 'i', 'n', 'g', ' ', 't', 'e', 'x', 't', ' ', 'i', 's', ' ', 'a', ' ', 'c', 'o', 'r', 'e', ' ', 't', 'a', 's', 'k', ' ', 'o', 'f', ' ', 'N', 'L', 'P', '.']
```

字符还是不能作为神经网络的输入，我们需要把它变成整数的ID：

```
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)
```

输出：
```
{' ': 0, '.': 1, 'L': 2, 'N': 3, 'P': 4, 'T': 5, 'a': 6, 'c': 7, 'e': 8, 'f': 9, 'g': 10, 'i': 11, 'k': 12, 'n': 13, 'o': 14, 'r': 15, 's': 16, 't': 17, 'x': 18, 'z': 19}
```

这样我们就得到一个dict，把一个字符映射成一个整数id。有了这个dict，任意输入的文本就可以变成id的序列：

```
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)
```
输出：
```
[5, 14, 12, 8, 13, 11, 19, 11, 13, 10, 0, 17, 8, 18, 17, 0, 11, 16, 0, 6, 0, 7, 14, 15, 8, 0, 17, 6, 16, 12, 0, 14, 9, 0, 3, 2, 4, 1]
```
最后一步是把一维的ID序列变成二维的one-hot矩阵。为什么要把一个ID变成一个一维的one-hot向量呢？我们来看一个例子。比如说我们需要把变形金刚的角色编码成数字，那么最简单的方法就是用一个map：

```
categorical_df = pd.DataFrame({"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]})
categorical_df
```

输出：
```
 	Name 		Label ID
0 	Bumblebee 	0
1 	Optimus Prime 	1
2 	Megatron 	2
```
但是这种方法会给原本没有任何关系的3个名字(字符串)建立一种大小的顺序关系，直接把012这些数字输入模型，模型很可能从里面学到一些容易误导的特征。因此，更好的方法是使用one-hot的编码方法——创建长度为3(标签数量)的一个向量，向量的第一个位置表示Bumblebee是否出现，第二个位置表示Optimus Prime是否出现。我们可以用df.get_dummies()函数来实现one-hot编码：

```
pd.get_dummies(categorical_df["Name"])
```

```
	Bumblebee 	Megatron 	Optimus Prime
0 	1 		0 		0
1 	0 		0 		1
2 	0 		1 		0
```

在Pytorch里，我们可以使用one_hot()函数来实现类似的功能：

```
import torch
import torch.nn.functional as F
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
one_hot_encodings.shape
```
输出：
```
torch.Size([38, 20])
```

输入是长度为38的ID列表，输出是[38,20]的矩阵，矩阵的每一行代表一个one-hot编码的向量。通过检查第一个字符，我们可以确保代码是正确的：

```
print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One-hot: {one_hot_encodings[0]}")
```

输出：
```
Token: T
Tensor index: 5
One-hot: tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

我们发现字符T的ID是5，所以下标为5(第6个位置)的地方是1，其余的位置都是0。


字符切分的好处是词典非常小，比如英文来说基本就是字母数字和一些常见的标点，所以不会遇到OOV的问题。但是它的缺点也是非常明显的，那就是文本的Token树很多，模型需要学习字符组合成词，这需要很强大的模型、足够的计算资源和训练数据。

### 基于词的切分

另外一种切分的方法就是分词，对于英文这样的语言来说相对比较简单，用whitespace和表达就可以实现简单的切分。当然里面也有很多细节，比如英文句号(.)有的时候并不是句子结束的标点。而I'm这种应该算一个词还是要进一步切分或者是否要做一些词干的还原(比如把过去式变成原型等等)，大部分可以有规则，但是也有很多不规则变化的例外。对于中文这些没有明显字符边界的语言，分词也是很困难的，有的时候关于什么是词语言学家们都无法达成共识。Transformer里很少会用到分词，所以这里我们看看简单的用空格分割：

```
tokenized_text = text.split()
print(tokenized_text) 
```

输出：
```
['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP.']
```

基于词的切分相对字符来说粒度较粗，因此无法表达词法上的变化。比如cat和cats会被看出两个完全不同的词，虽然理论上通过上下文模型可以学到它们的相似之处，但是实际往往很困难。另外一个问题就是词的数量很多，而且会不断涌现出新词(尤其是新名词)，所以很难避免OOV的问题。

### 基于子词(subword)的切分

而介于字符和词直接的一类切分方法就是所谓的子词(subword)切分，它融合了字符和词切分的好处：一方面它会把一些高频的字符序列(可能是词也可能是一些前后缀)组合在一起，避免了字符模型输入过长的问题；同时对于不常见的甚至OOV的词汇，也能切分成一些可能见过的前后缀，就像我们可能没见过basically，但是通过basic，我们大致可以猜测出它的含义。大家提出了很多不同的子词切分的方法，大部分方法都是通过一个语料库文本的实际分布用统计方法的找到比较合适的子词的集合，并且基于这个集合和一些基于规则的方法进行切分。

我们首先来看一下[WordPiece](https://doi.org/10.1109/ICASSP.2012.6289079)算法，它是BERT和这里要用到的DistilBERT的分词算法。Transformers库提供了类AutoTokenizer来帮助我们轻松的加载某个模型(比如BERT)的分词算法，我们只需要调用它的from_pretrained()函数：

```
from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```

我们只需要知道模型的名字，比如这里的distilbert-base-uncased，然后AutoTokenizer.from_pretrained就会返回一个合适的Tokenizer：

```
type(tokenizer)
transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast
```

AutoTokenizer类的前缀Auto说明了这个类会智能的根据模型的名字加载合适的模型，比如这里我们传入的是distilbert-base-uncased，所以它自动会找到合适的DistilBERT模型。我们当然也可以明确的制定模型：

```
from transformers import DistilBertTokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
type(tokenizer)
```

输出结果是一样的：
```
transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast
```


我们来测试一下这个Tokenizer：
```
encoded_text = tokenizer(text)
print(encoded_text)
```

输出为：
```
{'input_ids': [101, 19204, 6026, 3793, 2003, 1037, 4563, 4708, 1997, 17953, 2361, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

我们看到Tokenizer对象实现了callable，因此可以直接把它当函数调用，返回的是一个dict，input_ids是对文本进行切分并且转换成了id；而attention_mask是输入的mask，后文会介绍，这里可以暂时忽略。Tokenizer对象也提供方法把id转换成token：

```
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)
```

输出：
```
['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '.', '[SEP]']
```

上面的输出有三点值得注意：第一，输出增加了[CLS]和[SEP]这样的特殊字符串，这是为了区分多个输入的边界，不同的模型可能会用不同的特殊字符串。第二，NLP等文本变成了小写，因为我们指定的是uncased的版本。第三，tokenizing被切分成了token+izing，而且非开头的部分加了特殊的两个#，这样我们就可以把这个token序列恢复成词序(但是从token序列可能是无法完全恢复成字符串序列的，因为它忽略了空格)。

不过我们不用自己写代码来实现token序列到文本的转换，因为每个模型都不同，这会很麻烦和容易出错，我们可以用Tokenizer的convert_tokens_to_string()方法来实现：


```
print(tokenizer.convert_tokens_to_string(tokens))

[CLS] tokenizing text is a core task of nlp. [SEP]

```

AutoTokenizer类的对象还有一些重要的属性，比如vocab_size记录了词典(subword)的大小：

```
tokenizer.vocab_size

30522
```

模型最大支持的Token数：
```
tokenizer.model_max_length

512
```


另外Tokenizer类的对象也记录了对应模型需要传入的参数名字：

```
tokenizer.model_input_names

['input_ids', 'attention_mask']
```

### 对整个训练集进行Token切分

为了对整个数据集进行Token切分，我们可以使用DatasetDict的map()函数。首先我们定义一个文本切分的函数：

```
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
```

这个函数的输入是一个batch的数据，padding为True指明我们会padding短的token序列到输入中最长的序列长度，truncation为True指明如果输入超出了模型的最大长度，那么就截掉过长的部分。我们来测试一下：

```
print(tokenize(emotions["train"][:2]))

{'input_ids': [[101, 1045, 2134, 2102, 2514, 26608, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1045, 2064, 2175, 2013, 3110, 2061, 20625, 2000, 2061, 9636, 17772, 2074, 2013, 2108, 2105, 2619, 2040, 14977, 1998, 2003, 8300, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

我们可以看到padding的效果，第一个文本较短，因此它的input_ids后面很多0，另外与0对应的位置的attention_mask都是0，这样模型在计算attention的时候就会去掉这些padding的位置。除了ID为0的[PAD]，我们也会看到101和102这些特殊字符串。BERT用到的所有特殊字符串和对应的ID如下表：


| 特殊Token      | [PAD] | [UNK] | [CLS] | [SEP] | [MASK] |
| 特殊Token的ID   | 0 | 100 | 101 | 102 | 103  |

attention_mask的作用是告诉模型一个batch的数据里哪些是padding的，从而可以忽略掉那些padding的部分，比如下图的例子：


<a name='img15'>![](/img/learnhuggingface/15.png)</a>

从图中可以看出，padding的位置的ID都是0，而对应的attention_mask也是0。最后会把input_ids和attention_mask变成(batch, max_length)大小的tensor。

定义好tokenize函数之后，我们就可以调用DatasetDict的map函数，从而对整个数据集进行处理：
```
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
```
默认情况下，map函数是对每条数据进行处理，所以我们设置batched为True。另外我们也设置了batch_size为None，从而把所有数据都弄成一个特别大的batch。这样做的目的是为了把所有的数据都变成一样长，便于后面使用非随机梯度下降的算法(那种一次需要feed进去所有数据)。如果使用深度学习常见的模型，就不能设置这么大的batch。


## 训练文本分类器

DistilBERT这样的预训练模型的任务是Masked Language Modeling，也就是预测Mask掉的那些词。这样的模型当然不能直接用于文本分类，为了找到需要做哪些修改，我们来看看DistilBERT这样的encoder模型的结构和怎么让它完成文本分类任务。

<a name='img16'>![](/img/learnhuggingface/16.png)</a>

文本通过前面的步骤变成了Token ID的list，这是模型的输入。第一步就是通过one-hot编码变成向量，one-hot编码的维度也就是所有可能的Token数量，在实践中通常是20k-200k的范围。这样得到的结果就是Token Encoding。接着把这个稀疏高维的one-hot向量变成低维稠密的Token Embedding，这从实现上可能是查找一个Embedding Lookup Tabel(GPU)或者乘以对应的一个矩阵(TPU)。Transformer模型除了Token Embedding之外还会有Position Embedding和Segment Embedding，它们的值都会逐个加到Token Embedding里。这些Embedding会传入很多层的自注意力层，而最后一层的Hidden States会接入一些全连接层和softmax来实现Mask掉的Token的预测。我们一般把自注意力层叫做Body，而上面加的全连接层和softmax叫Head。通过Masked Language Modeling任务的预训练，我们训练得到了比较好的Body和Head的参数。为了用它来完成下游的fine-tuning任务，我们会扔掉预训练的Head，然后增加适合文本分类的新Head，它的参数是随机初始化的。接下来我们通常有如下两种方法来训练新的模型：

* 特征提取——固定住预训练模型的body，把它当做特征提取器，只是训练新增加的Head的参数
* fine-tuning——body和新head同时训练

### Transformer作为特征提取器(Feature Extractor)

把Transformer当成特征提取器非常简单，我们只需要冻结住body的参数，然后把Body最后一层的隐状态作为模型的输入特征就行。模型甚至不需要是基于梯度的，比如随机森林这样的模型。而且通常我们也可以把特征提取器和后面的模型完全分开(当然对于后面加的是全连接层这样的神经网络来说也可以把整体当成一个模型，只是Body部分参数是不能调整的)。这种方法流程如下图所示：


<a name='img17'>![](/img/learnhuggingface/17.png)</a>

#### 使用预训练模型

接下来我们会使用Transformers库里的一个AutoModel类，和AutoTokenizer类似，它也有一个from_pretrained()的函数来加载预训练模型的参数权重。下面我们来加载预训练的DistilBERT模型的权重：

```
from transformers import AutoModel
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
```

这里我们首先判断是否有GPU，从而在from_pretrained加载模型后调用nn.Module.to()函数把它放到合适的设备上。AutoModel的输入是Token Encoding(Token ID)，输出是每个Token的隐状态。下面我们来看看怎么把这些隐状态抽取出来作为后面模型的特征。


#### 抽取最后一层隐状态

我们首先来看怎么抽取输入一个字符串(非batch)时的隐状态，第一步当然是需要把字符串通过Tokenizer变成Token ID，为了能够给后续的模型，我们需要传入参数return_tensors="pt"，否则默认它返回的是普通的python的list，但是我们的模型需要的是Pytorch的Tensor。

```
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")
type(inputs['input_ids'])
```

输出是：

```
Input tensor shape: torch.Size([1, 6])

torch.Tensor
```

可以看到返回的确实是Pytorch的Tensor，而且它的shape是(batch, len)。有了这个Token ID的Tensor之后，我们可以把它输入模型进行预测。在输入模型之前，我们需要把输入也放到合适的设备上。

```
inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
```

输出：
```
BaseModelOutput(last_hidden_state=tensor([[[-0.1565, -0.1862,  0.0528,  ..., -0.1188,  0.0662,  0.5470],
         [-0.3575, -0.6484, -0.0618,  ..., -0.3040,  0.3508,  0.5221],
         [-0.2772, -0.4459,  0.1818,  ..., -0.0948, -0.0076,  0.9958],
         [-0.2841, -0.3917,  0.3753,  ..., -0.2151, -0.1173,  1.0526],
         [ 0.2661, -0.5094, -0.3180,  ..., -0.4203,  0.0144, -0.2149],
         [ 0.9441,  0.0112, -0.4714,  ...,  0.1439, -0.7288, -0.1619]]],
       device='cuda:0'), hidden_states=None, attentions=None)
```

这里有一点Python的小技巧。我们并不知道模型有多少需要输入(不同的模型需要的输入个数可能不同)，但是我们知道Tokenizer的输出是可以作为AutoModel的输入的，我们唯一需要做的就是把Tokenizer输出的inputs(这是一个dict)的value(都是Tensor)放到GPU上。然后调用model的时候通过**inputs的语法填入参数。

另外在调用model时我们把它放到了torch.no_grad()这个context manager之下，这样可以避免自动梯度的计算。对于推理来说这是不必要的计算，去掉之后能够减少内存的使用量。输出取决于模型的配置，有可能返回隐状态、loss和attention。输出的值类似于Python的namedtuple。这里的输出是BaseModelOutput，我们可以查看它的last_hidden_state：

```
outputs.last_hidden_state.size()
```
输出：
```
torch.Size([1, 6, 768])
```

我们看到输出的shape是(batch, max_len, hidden_size)。对于文本分类来说，我们是整段文本输入返回一个标签，所以通常可以用第一个Token([CLS])的输出作为后续模型的输入，我们来看看这个Token的输出：

```
outputs.last_hidden_state[:,0].size()
```
输出：

```
torch.Size([1, 768])
```
不看batch维度，每个文本就对应[CLS]的768维的输出。我们知道了怎么抽取一个句子的特征，那么就可以实现一个函数来抽取一个batch的句子的特征，并且用DatasetDict的map方法抽取整个数据集的特征：

```
def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
```

这个函数和前面的唯一区别就是返回值从Pytorch的Tensor变成了numpy的ndarray。原因在于DatasetDict的map方法要求这个函数的输出是Python的对象或者numpy的对象。这是合理的要求，因为Datasets API的设计不应该依赖于具体的某种框架的Tensor，而是应该依赖numpy这种更底层通用的数据结构。

因为之前我们用map调用tokenizer函数没有传入return_tensors="pt"，所以得到的emotions_encoded是python的list。我们可以再跑一遍，不过DatasetDict提供了set_format来做不同格式的转换(不一定会做实际的转换，但是对于使用者来说可以看出它需要的类型)：

```
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

接下来就可以使用map函数提取特征了：

```
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
```

注意：我们在上面的map里没有指定batch_size为None，因此默认的batch_size是1000。因为如果所有数据都一次进行forward计算的话，可能会超出GPU的显存。而且前面我们tokenizer的时候是一次计算的，已经可以保证输入都是固定长度的(最大的长度为87)。所以这次处理就可以以较小的batch_size来做。我们可以看看输出包含的内容：

```
emotions_hidden["train"].column_names
```
输出
```
['text', 'label', 'input_ids', 'attention_mask', 'hidden_state']
```

和输入相比，输出多了hidden_state。


#### 创建特征矩阵

接下来我们希望用这个特征来训练模型，为了后续方便，我们把数据的格式变成Scikit-learn库能够用的输入和输出格式：

```
import numpy as np
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
X_train.shape, X_valid.shape
```

输出：
```
((16000, 768), (2000, 768))
```

在用隐状态的特征训练模型之前，我们首先来分析一下这些特征是否有足够的区分度从而可以让模型可以学习。


#### 训练集的可视化

因为人脑没法处理768维的数据，所以我们先使用[UMAP算法](https://arxiv.org/abs/1802.03426)把数据降维到2。因为UMAP算法在特征归一化到(0,1)的区间效果比较好，因此我们先使用MinMaxScaler对特征进行归一化，然后再使用umap-learn库来实现UMAP降维。这里需要先安装：


```
!pip install umap-learn
```

```
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
# Scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
# Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
df_emb.head()
```

输出为：

```
 	X 	Y 	label
0 	4.289374 	7.102280 	0
1 	-2.970790 	5.993966 	0
2 	5.233337 	3.682298 	3
3 	-2.182288 	3.740904 	2
4 	-3.031142 	3.906332 	3
```

上面的代码首先用MinMaxScaler().fit_transform()把X_train的每一列归一化到(0,1)区间。接着构造UMAP对象，n_components为2说明降到2维，metric是cosine表示用余弦计算向量的距离。最后得到mapper.embedding_，转成Pandas的DataFrame。最后把标签y_train也放到df_emb里面。

接下来是绘图：

```
fig, axes = plt.subplots(2, 3, figsize=(7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names
for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

这里用到hexbin函数，这里不做介绍，感兴趣的读者可以参考[Visualizing Data with Hexbins in Python](https://medium.com/@mattheweparker/visualizing-data-with-hexbins-in-python-39823f89525e)。

结果为：

<a name='img18'>![](/img/learnhuggingface/18.png)</a>

从图中我们可以看出，sadness、anger和fear三个负面的标签分布在类似的区域，只是分布细节有所区别。joy和love分布在和它们不同的区域，而它们两之间的区域比较重合。而surprise所有的区域都有分布。当然这是降维后的情况，在二维空间不可分并不代表在高维空间不可分。这里我们大致可以判断DistilBERT提取的特征还是不错的。所以我们赶紧用这些特征去训练分类器吧！

#### 训练一个简单分类器

下面我们用Scikit-learn训练一个非常简单的Logistic Regression分类器(多分类其实也可以叫softmax分类器)：

```
from sklearn.linear_model import LogisticRegression
# We increase `max_iter` to guarantee convergence
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)
```
输出：
```
0.633
```

这个准确率看起来比随机猜好不了多少，我们可以用Scikit-Learn的DummyClassifier来作为baseline。

```
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)
```
输出：
```
0.352
```

所以这个简单分类器比瞎猜的结果还是提升了15%以上。我们也可以画出这个分类器的混淆矩阵，来分析哪些类之间容易混淆：

```
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)
```


<a name='img19'>![](/img/learnhuggingface/19.png)</a>

上面的代码主要就是用confusion_matrix()函数来计算混淆矩阵，normalize为true表示进行归一化，从而可以在不平衡的类别自己方便对比。ConfusionMatrixDisplay函数需要传入混淆矩阵和每个类别对应的标签名字。

从混淆矩阵可以看出，anger和fear很容易与sadness混淆(看第四第五行的第一列)。love和surprise也很容易与joy混淆(看第三第六行的第二列)。

### fine-tuning一个Transformer模型

fine-tuning需要要求新增加的分类器head是可以微分的，从而可以把对loss的梯度传递给body部分。因此我们通常用神经网络作为新的head(当然LR/Softmax也可以看成一种网络结构)。在这种方式的过程如下图所示：

<a name='img20'>![](/img/learnhuggingface/20.png)</a>


fine-tuning的好处是body部分也是可以微调的，这样使得特征的表示更加适合我们特定的任务。缺点当然是训练的计算量会大很多，通常需要GPU训练才能在可接受的时间完成。下面我们讲基于Hugging Face Transformers库提供的Trainer API完成fine-tuning。

#### 加载预训练模型

和前面的方法类似，我们首先也需要加载一个预训练模型。这里我们需要用到AutoModelForSequenceClassification，它和AutoModel的区别是AutoModelForSequenceClassification会帮我们在body至少增加一个多分类的Head，我们只需要告诉它分类的标签数就行了。


```
from transformers import AutoModelForSequenceClassification
num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))
```

输出：
```
Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.bias', 'vocab_transform.weight', 'vocab_layer_norm.weight', 'vocab_transform.bias', 'vocab_projector.weight', 'vocab_projector.bias']
- This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.weight', 'pre_classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

我们会看到输出了一些警告信息，它说的是我们把一些预训练的权重扔掉了，因为这些是适合于Masked Language Modeling任务的Head。我们忽略掉它就行。

#### 定义评估效果的metrics

为了在训练的过程中监控效果，我们需要实现一个compute_metrics()函数，并把它提供给Trainer。这个函数的输入参数是一个EvalPrediction对象(它包含predictions和label_ids，分别表示模型的预测和真实的标签)，输出是一个dict，输出的key是metric的名字，value是metric的值。下面我们来实现这样的一个函数：


```
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

这里调用skllearn.metrics的f1_score和accuracy_score函数来计算分类的准确率和f1分。pred.predictions是logits，所以需要先用argmax得到预测分类。


#### 训练模型

要训练模型，我们需要先配置TrainingArguments，这里大部分可以用默认的配置，最重要的是设置output_dir、num_train_epochs、learning_rate和per_device_train_batch_size等，分别对应模型的输出目录、迭代的次数、学习率和batch大小：

```
from transformers import Trainer, TrainingArguments
batch_size = 8
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                num_train_epochs=2,
                                learning_rate=2e-5,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=0.01,
                                evaluation_strategy="epoch",
                                disable_tqdm=False,
                                logging_steps=logging_steps,
                                push_to_hub=False,
                                log_level="error")
```

有了TrainingArguments就可以构造Trainer来训练了：
```
from transformers import Trainer
trainer = Trainer(model=model, args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=emotions_encoded["train"],
                eval_dataset=emotions_encoded["validation"],
                tokenizer=tokenizer)
trainer.train();
```
结果为：
```
Epoch 	Training Loss 	Validation Loss 	Accuracy 	F1
1 	0.431400 	0.214688 	0.930500 	0.930070
2 	0.161300 	0.178166 	0.937500 	0.937274
```

这个训练在GPU上需要几分钟，两个epoch之后验证集上的准确率和F1都在93%以上。除了这两个指标，我们也希望看看混淆矩阵，所以我们先需要用模型对测试集进行预测。Trainer提供了predict方法：


```
preds_output = trainer.predict(emotions_encoded["validation"])
```

predict()函数返回的是PredictionOutput对象，这个对象包括predictions和label_ids两个属性，除此之外还有那些metrics。

```
preds_output.metrics
```
输出：
```
{'test_loss': 0.17816567420959473,
 'test_accuracy': 0.9375,
 'test_f1': 0.9372737827287936,
 'test_runtime': 8.8243,
 'test_samples_per_second': 226.647,
 'test_steps_per_second': 28.331}
```

由于predictions是logits，所以我们需要用argmax得到模型预测的结果：

```
y_preds = np.argmax(preds_output.predictions, axis=1)
```

接下来我们可以画出来混淆矩阵：

```
plot_confusion_matrix(y_preds, y_valid, labels)
```

结果为：

<a name='img22'>![](/img/learnhuggingface/22.png)</a>

可以看出，fine-tuning之后整体的效果好了很多。局部来看，love比较容易和joy混淆，surprise容易和fear以及joy混淆。不过混淆矩阵只能看到一个全局的情况，我们还需要做具体的badcase分析。

#### 错误分析

一种有用的错误分析方法是对验证集里的样本的loss进行排序，Trainer.predict不会返回loss，所以我们还是需要用Model。如果在调用Model时传入label，那么它会返回loss。


```
from torch.nn.functional import cross_entropy
def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}
```

forward_pass_with_label函数首先把batch的value放到GPU上，然后调用model得到output，然后根据output.logits得到预测从标签pred_label。同时用outout.logits和真正的标签batch["label"]去计算loss，为了不让它把一个batch的loss加起来，我们需要传入reduction为"none"。这样就可以得到每一个样本的loss。

接下来我们使用Dataset(和DatasetDict类似)的map函数对每一个batch的验证集数据都计算loss和预测标签：
```
# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label, batched=True, batch_size=16)
```

最后为了分析，我们把它转换成Pandas的DataFrame，并且把label从id变成容易阅读的文本：

```
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))
```

接下来我们会根据loss进行升序或者降序排列，目的是发现这两类问题：

* 错误的标注
* 难处理的case

我们先去看看loss最高的一些样本：
```
df_test.sort_values("loss", ascending=False).head(10)
```
输出：

```
 	text 	label 	predicted_label 	loss
1195 	i always think about are act the way i want to feel so even when im grumpy i still need to act pleasant and happy and then i will start to feel more that way 	anger 	joy 	9.087166
1963 	i called myself pro life and voted for perry without knowing this information i would feel betrayed but moreover i would feel that i had betrayed god by supporting a man who mandated a barely year... 	joy 	sadness 	8.655710
1500 	i guess we would naturally feel a sense of loneliness even the people who said unkind things to you might be missed 	anger 	sadness 	8.443667
318 	i felt ashamed of these feelings and was scared because i knew that something wrong with me and thought i might be gay 	fear 	sadness 	7.756294
1111 	im lazy my characters fall into categories of smug and or blas people and their foils people who feel inconvenienced by smug and or blas people 	joy 	fear 	7.677988
1870 	i guess i feel betrayed because i admired him so much and for someone to do this to his wife and kids just goes beyond the pale 	joy 	sadness 	7.350218
1801 	i feel that he was being overshadowed by the supporting characters 	love 	sadness 	7.221741
1840 	id let you kill it now but as a matter of fact im not feeling frightfully well today 	joy 	fear 	7.173310
1919 	i should admit when consuming alcohol myself in small amounts i feel much less inhibited ideas come to me more easily and i can write with greater ease 	fear 	sadness 	6.982317
1836 	i got a very nasty electrical shock when i was tampering with some electrical applainces 	fear 	anger 	6.703757
```

我们发现数据确实有一些问题，尤其是joy经常会被标错。也有一些文字看不出明显的类别。接下来我们看一些loss最低的一些例子：


```
 	text 	label 	predicted_label 	loss
908 	i said earlier that the overall feeling is joyful happy thankful and that s spoken in just about every other post i have of mason 	joy 	joy 	0.000424
578 	i got to christmas feeling positive about the future and hopeful that hospital admissions were finally behind me 	joy 	joy 	0.000427
1513 	i have also been getting back into my gym routine so im feeling positive about this now 	joy 	joy 	0.000428
669 	i am not feeling very joyful today its been a rough day 	joy 	joy 	0.000430
400 	i are just relaxing together and i feel ecstatic and blissfully happy because i know he loves me and i love him 	joy 	joy 	0.000431
1199 	i didn t feel ecstatic after each workout or anything like that 	joy 	joy 	0.000432
1425 	i see you the light in the room brightens i get a glow in my eyes i feel ecstatic 	joy 	joy 	0.000433
197 	i feel so cool like ice t huhwe neun gatda beoryeo priceless sesang ye ban bani namja neottaemune na ulji anha gucha hage neoreul jabgeo na mae dallil ireun jeoldae no 	joy 	joy 	0.000433
1873 	i feel practically virtuous this month i have not exceeded my target of only buying things 	joy 	joy 	0.000435
428 	i find enlightening and brilliant when i am feeling joyful can be annoying and slightly grating when the cluttered mind gets going 	joy 	joy 	0.000438
```

我们可以看到模型关于joy的预测是置信度最高的。




