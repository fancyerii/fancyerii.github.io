---
layout:     post
title:      "Huggingface Transformer教程(一)" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - Transformer
    - BERT
    - PyTorch
    - Tensorflow
    - Huggingface
---

本系列文章介绍Huggingface Transformer的用法。

<!--more-->

**目录**
* TOC
{:toc}

## 简介

目前各种Pretraining的Transformer模型层出不穷，虽然这些模型都有开源代码，但是它们的实现各不相同，我们在对比不同模型时也会很麻烦。[Huggingface Transformer](https://huggingface.co/transformers/)能够帮我们跟踪流行的新模型，并且提供统一的代码风格来使用BERT、XLNet和GPT等等各种不同的模型。而且它有一个[模型仓库](https://huggingface.co/models)，所有常见的预训练模型和不同任务上fine-tuning的模型都可以在这里方便的下载。截止目前，最新的版本是4.5.0。

## 安装

Huggingface Transformer 4.5.0需要安装Tensorflow 2.0+ **或者** PyTorch 1.1.0+，它自己的安装非常简单：

```
pip install transformers
```

如果想安装最新的master的版本，可以：
```
pip install git+https://github.com/huggingface/transformers
```

我们也可以从github clone源代码然后安装，这样的好处是如果我们需要修改源代码就会很方便：
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

当然，如果是普通的安装也是可以修改其源代码的，比如作者是使用virtualenv安装的transformers，那么它的位置在：
```
~/huggingface-venv/lib/python3.6/site-packages/transformers/
```
在Linux下，模型默认会缓存到~/.cache/huggingface/transformers/。

如果不想每次都去下载，可以设置环境变量：
```
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

说明：即使我们有缓存，默认它还是会去hugging face hub寻找最新版本的模型，如果有更新，它还是会重新下载。所以如果想保证代码的一致性，可以加上这两个环境变量。但是如果换一台机器就还是需要下载。另外一种方法就是直接通过模型的目录来初始化，后面会介绍。

## 基本原则

Transformers的目的是为了：

* 帮助NLP研究者进行大规模的transformer模型
* 帮助工业界的使用者微调模型并且不是到生产环境
* 帮助工程师下载预训练模型并且解决实际问题

它的设计原则包括：
* 易用
    * 只有[configuration](https://huggingface.co/transformers/main_classes/configuration.html)，[models](https://huggingface.co/transformers/main_classes/model.html)和[tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html)三个主要类。
    * 所有的模型都可以通过统一的from_pretrained()函数来实现加载，transformers会处理下载、缓存和其它所有加载模型相关的细节。而所有这些模型都统一在[Hugging Face Models](https://huggingface.co/models)管理。
    * 基于上面的三个类，提供更上层的pipeline和Trainer/TFTrainer，从而用更少的代码实现模型的预测和微调。
    * 因此它不是一个基础的神经网络库来一步一步构造Transformer，而是把常见的Transformer模型封装成一个building block，我们可以方便的在PyTorch或者TensorFlow里使用它。
* 尽量和原论文作者的实现一致
    * 每个模型至少有一个例子实现和原论文类似的效果
    * 尽量参考原论文的实现，因此有些代码不会那么自然

### 主要概念

* 诸如BertModel的**模型(Model)**类，包括30+PyTorch模型(torch.nn.Module)和对应的TensorFlow模型(tf.keras.Model)。
* 诸如BertConfig的**配置(Config)**类，它保存了模型的相关(超)参数。我们通常不需要自己来构造它。如果我们不需要进行模型的修改，那么创建模型时会自动使用对于的配置
* 诸如BertTokenizer的**Tokenizer**类，它保存了词典等信息并且实现了把字符串变成ID序列的功能。

所有这三类对象都可以使用from_pretrained()函数自动通过名字或者目录进行构造，也可以使用save_pretrained()函数保存。



## quicktour


### 使用pipeline

使用预训练模型最简单的方法就是使用pipeline函数，它支持如下的任务：
* 情感分析(Sentiment analysis)：一段文本是正面还是负面的情感倾向
* 文本生成(Text generation)：给定一段文本，让模型补充后面的内容
* 命名实体识别(Name entity recognition)：识别文字中出现的人名地名的命名实体
* 问答(Question answering)：给定一段文本以及针对它的一个问题，从文本中抽取答案
* 填词(Filling masked text)：把一段文字的某些部分mask住，然后让模型填空
* 摘要(Summarization)：根据一段长文本中生成简短的摘要
* 翻译(Translation)：把一种语言的文字翻译成另一种语言
* 特征提取(Feature extraction)：把一段文字用一个向量来表示

下面我们来看一个情感分析的例子：
```
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
```

当第一次运行的时候，它会下载预训练模型和分词器(tokenizer)并且缓存下来。分词器的左右是把文本处理成整数序列。最终运行的结果为：
```
[{'label': 'POSITIVE', 'score': 0.9997795224189758}]
```

注意，如果因为网络问题下载不了，可以尝试使用代理：
```
classifier = pipeline('sentiment-analysis', proxies={"http": "http://localhost:1080"})
```
但是上面的代码下载后会抛出如下的异常：
```
TypeError: __init__() got an unexpected keyword argument 'proxies'
```
感觉这是代码的bug，目前的hack方法是用代理下载成功后在去掉proxies参数。

我们也可以一次预测多个结果：
```
results = classifier(["We are very happy to show you the 🤗 Transformers library.",
           "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```

运行结果为：
```
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

上面的第二个句子被分成了负面的情感，这是因为模型是二分类的模型，不过它的得分是在0~1之间比较居中的部分。默认的"sentiment-analysis"会使用distilbert-base-uncased-finetuned-sst-2-english模型。它是把[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)模型在SST-2这个任务上fine-tuning后的结果。

我们也可以指定其它的情感分类模型，比如我们可能需要对法语进行情感分类，那么上面的模型是不适合的，我们可以去[model hub](https://huggingface.co/models)寻找合适的模型。比如我们可以使用：
```
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
```

除了通过名字来制定model参数，我们也可以传给model一个包含模型的目录的路径，也可以传递一个模型对象。如果我们想传递模型对象，那么也需要传入tokenizer。

我们需要两个类，一个是[AutoTokenizer](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoTokenizer)，我们将使用它来下载和加载与模型匹配的Tokenizer。另一个是[AutoModelForSequenceClassification](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModelForSequenceClassification)(如果用TensorFlow则是[TFAutoModelForSequenceClassification](https://huggingface.co/transformers/model_doc/auto.html#transformers.TFAutoModelForSequenceClassification))。注意：模型类是与任务相关的，我们这里是情感分类的分类任务，所以是AutoModelForSequenceClassification。

我们在使用前需要import：
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

为了下载和加载模型，我们只需要使用[from_pretrained函数](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModelForSequenceClassification.from_pretrained)：
```
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
```

### 原理

下面我们来看一下pipeline实际做了哪些工作。首先它会使用前面的from_pretrained函数加载Tokenizer和模型：
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### 使用Tokenizer
Tokenizer的作用大致就是分词，然后把词变成的整数ID，当然有些模型会使用subword。但是不管怎么样，最终的目的是把一段文本变成ID的序列。当然它也必须能够反过来把ID序列变成文本。关于Tokenizer更详细的介绍请参考[这里](https://huggingface.co/transformers/tokenizer_summary.html)，后面我们也会有对应的详细介绍。

下面我们来看一个例子：

```
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
```

Tokenizer对象是callable，因此可以直接传入一个字符串，返回一个dict。最主要的是ID的list，同时也会返回[attention mask](https://huggingface.co/transformers/glossary.html#attention-mask)：
```
print(inputs)
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

我们也可以一次传入一个batch的字符串，这样便于批量处理。这时我们需要指定padding为True并且设置最大的长度：
```
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```
truncation为True会把过长的输入切掉，从而保证所有的句子都是相同长度的，return_tensors="pt"表示返回的是PyTorch的Tensor，如果使用TensorFlow则需要设置return_tensors="tf"。

我们可以查看分词的结果：
```
>>> for key, value in pt_batch.items():
...     print(f"{key}: {value.numpy().tolist()}")
input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
```
pt_batch仍然是一个dict，input_ids是一个batch的ID序列，我们可以看到第二个字符串较短，所以它被padding成和第一个一样长。如果某个句子的长度超过max_length，也会被切掉多余的部分。

#### 使用模型
Tokenizer的处理结果可以输入给模型，对于TensorFlow来说直接输入就行，而对于PyTorch则需要使用**来展开参数：

```
# PyTorch
pt_outputs = pt_model(**pt_batch)
# TensorFlow
tf_outputs = tf_model(tf_batch)
```

Transformers的所有输出都是tuple，即使只有一个结果也会是长度为1的tuple：
```
>>> print(pt_outputs)
(tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>),)
```

Transformers的模型默认返回logits，如果需要概率，可以自己加softmax：
```
>>> import torch.nn.functional as F
>>> pt_predictions = F.softmax(pt_outputs[0], dim=-1)
```

得到和前面一样的结果：
```
>>> print(pt_predictions)
tensor([[2.2043e-04, 9.9978e-01],
        [5.3086e-01, 4.6914e-01]], grad_fn=<SoftmaxBackward>)
```

如果我们有输出分类对应的标签，那么也可以传入，这样它除了会计算logits还会loss：
```
>>> import torch
>>> pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
```
输出为：
```
SequenceClassifierOutput(loss=tensor(0.3167, grad_fn=<NllLossBackward>), logits=tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
```

from_pretrained返回的是PyTorch的[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)或者TensorFlow的[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)。因此我们可以很容易的把Transformer融入我们的代码里，自己来实现训练或者预测。但是Transformers包内置了一个[Trainer类](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer)，方便我们训练或者fine-tuning。

我们训练完成后就可以保存模型到一个目录中：
```
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```

之后我们想用的时候随时可以使用from_pretrained函数加载它们。Transformers包非常酷的一点就是它可以轻松的在PyTorch和TensorFlow之间切换。比如下面的例子是保存PyTorch的模型然后用TensorFlow加载：
```
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = TFAutoModel.from_pretrained(save_directory, from_pt=True)
```

如果用PyTorch加载TensorFlow模型，则需要设置from_tf=True：
```
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModel.from_pretrained(save_directory, from_tf=True)
```

除了logits，我们也可以返回所有的隐状态和attention：
```
pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states, all_attentions = pt_outputs[-2:]
```

#### 具体的模型类
AutoModel和AutoTokenizer只是方便我们使用，但最终会根据不同的模型(名字或者路径)构造不同的模型对象以及与之匹配的Tokenizer。前面的例子我们使用的是distilbert-base-uncased-finetuned-sst-2-english这个名字，AutoModelForSequenceClassification 会自动的帮我们加载[DistilBertForSequenceClassification](https://huggingface.co/transformers/model_doc/distilbert.html#transformers.DistilBertForSequenceClassification)模型。

知道原理后我们也可以自己直接构造这些模型：

```
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```

#### 自定义模型

如果你想自定义模型(这里指的是调整模型的超参数，比如网络的层数，每层的attention head个数等等，如果你要实现一个全新的模型，那就不能用这里的方法了)，那么你需要构造配置类。每个模型都有对应的配置类，比如[DistilBertConfig](https://huggingface.co/transformers/model_doc/distilbert.html#transformers.DistilBertConfig)。你可以通过它来指定隐单元的个数，dropout等等。如果你修改了核心的超参数(比如隐单元的个数)，那么就不能使用from_pretrained加载预训练的模型了，这时你必须从头开始训练模型。当然Tokenizer一般还是可以复用的。

下面的代码修改了核心的超参数，构造了Tokenizer和模型对象：
```
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)
```

如果我们只改变最后一层，这是很常见的，比如把一个两分类的模型改成10分类的模型，那么还是可以复用下面那些层的预训练模型。我们可以获取预训练模型的body，然后自己增加一个输出为10的全连接层。但是这里有更简单的方法，调用from_pretrained函数然后设置num_labels参数：

```
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```

我们可以看一下代码：
```
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()
```

预训练模型通常是语言模型(比如Masked LM或者NSP这样的任务)，所以DistilBertForSequenceClassification只会复用它的body部分，head部分是重新构造的。


## 术语
### 通用词汇
* autoencoding models
    * 参考MLM
* autoregressive models
    * 参考CLM
* CLM
    * 因果的(causal)语言模型，一种预训练任务，通过前面的文本预测后面的词。通常可以读入整个句子然后使用mask来屏蔽未来的输入。
* deep learning
    * 深度学习
* MLM
    * masked语言模型，一种预训练任务，通过破坏(corrupt)原始文本的某些部分，然后让模型来恢复(预测)它们。通常可以随机的mask某些词，然后让模型来预测。和CLM不同，它可以利用整句的上下文信息。
* multimodal
    * 除了文本还有其它类型的输入，比如图像。
* NLG
    * 自然语言生成类认任务，比如对话或者翻译
* NLP
    * 自然语言处理
* NLU
    * 自然语言理解
* 预训练模型
    * 在某个数据集(比如wiki)上训练好的模型，通常都是通过自监督的方式学习，比如使用CLM或者MLM任务进行学习
* RNN
    * 循环神经网络
* self-attention
    * 每个元素计算需要花多少的attention到其它元素
* seq2seq
    * 输入和输出都是序列的模型，而且通常输入和输出长度不同，不存在显式的对齐 
* token
    * 一个句子的一部分，通常是一个词，但是也可能是字词(subword,词的一部分)或者标点
* transformer
    * self-attention的深度学习模型结构

### 模型输入

虽然基于transformer的模型各不相同，但是可以把输入抽象成统一的格式。

#### 输入ID
输入ID通常是唯一必须的参数，虽然不同的tokenizer实现差异很大，但是它们的作用是相同的——把一个句子变成Token的序列，不同的Token有不同的整数ID。下面是BERT模型的Tokenizer，它的具体实现是[WordPiece](https://arxiv.org/pdf/1609.08144.pdf)。

```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence = "A Titan RTX has 24GB of VRAM"
```

我们可以用它把句子变成Token序列：
```
tokenized_sequence = tokenizer.tokenize(sequence)
```
输出为：
```
>>> print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

有了Token之后可以通过查词典(vocabulary)把它变成整数ID，不过我们可以直接调用tokenizer输入句子一步到位，而且它会使用Rust更快的实现上面的步骤。

```
inputs = tokenizer(sequence)
```

返回的inputs是一个dict，key为"input_ids"对应的值就是输入id序列：
```
>>> encoded_sequence = inputs["input_ids"]
>>> print(encoded_sequence)
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
```

细心的读者会发现ID的序列比Token要多两个元素。这是Tokenizer会自动增加一些特殊的Token，比如CLS和SEP。为了验证它，我们可以用decode来把ID解码成Token：

```
decoded_sequence = tokenizer.decode(encoded_sequence)
```
结果如下：
```
>>> print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

### Attention Mask

如果输入是一个batch，那么会返回Attention Mask，它可以告诉模型哪些部分是padding的，从而要mask掉。

比如我们可以单独对两个句子进行编码：

```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."
encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```

则返回的两个序列长度是不同的：
```
>>> len(encoded_sequence_a), len(encoded_sequence_b)
(8, 19)
```

这样没有办法把它们放到一个Tensor里。我们需要把短的padding或者截断(truncate)长的。我们可以让Tokenizer帮我们做这些琐碎的事情：
```
padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
print(padded_sequences["input_ids"])
[[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
```

我们可以看到第一个ID序列后面补了很多零。但这带来一个问题：模型并不知道哪些是padding的。我们可以约定0就代表padding，但是用起来会比较麻烦，所以通过一个attention_mask明确的标出哪个是padding会更加方便。

```
>>> padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```


### Token的类型

对于问答这样的问题来说，我们需要两个句子，这个时候需要明确的告诉模型某个Token到底属于哪个句子，这时候就需要Token类型了。虽然我们会在句子结束加一个[SEP]，但是显示的告诉模型效果会更好。我们可以调用Tokenizer是传入两个句子，这样它就会自动帮我们加上[SEP]：

```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"
encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict["input_ids"])
```

注意它和batch调用的区别。batch调用的输入是一个list，而这里是两个句子。它的输出为：

```
>>> print(decoded)
[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```

前面说了，BERT等模型需要明确知道某个Token属于哪个句子，这个时候就需要"token_type_ids"：
```
>>> encoded_dict['token_type_ids']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```
从上面的结果看到，第一个句子的每个Token对应的token类型是0，第二个句子是1。另外对于[XLNetModel](https://huggingface.co/transformers/model_doc/xlnet.html#transformers.XLNetModel)还会有token类型为2。


### 位置ID

和RNN不同，Transformer必须对位置进行编码，否则就会丢掉位置信息。位置编码对应的key是"position_ids"，如果输入给模型的位置编码为空，则默认使用绝对位置编码，其范围为[0, config.max_position_embeddings - 1]。

### 标签
标签是用于计算loss的，不同类型的任务需要的标签是不同的。

* 序列分类
    * 输入是一个序列，输出是一个分类，比如BertForSequenceClassification。则标签的shape是(batch, )，每个输入对应一个标签。
* token分类
    * 输出是给每个token打标签，比如BertForTokenClassification。则标签的shape是(batch, seq_length)。
* MLM
    * 同上，shape是(batch_size, seq_length)
* seq2seq
    * 标签的shape是(batch_size, tgt_seq_length)

基本的模型，比如BertModel，是不能接受标签的，它只能输出特征，如果要基于它做序列分类，则需要在之上加分类的head。

### Decoder输入ID

对于seq2seq任务，除了Encoder的输入，还有Decoder的输入(decoder_input_ids)。大部分encoder-decoder模型都能够通过labels计算decoder的decoder_input_ids。

### Feed Forward Chunking

这个是Reformer模型特有的参数，请参考[论文](https://arxiv.org/abs/2001.04451)。


## 任务汇总
本节介绍最常见的一些任务，包括问答、序列分类、命名实体识别和其它一些任务。这里的例子都是使用自动构造的模型(Auto Models)，它会从某个checkpoint恢复模型参数，并且自动构造网络。更多自动模型的介绍请参考[AutoModel](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModel)。

为了获得好的效果，我们需要找到适合这个任务的checkpoint。这些checkpoint通常是在大量无标注数据上进pretraining并且在某个特定任务上fine-tuning后的结果。这里有一些需要注意的：
* 并不是所有任务都有fine-tuning的模型，可以参考examples下run_$TASK.py脚本。
* fine-tuning的数据集不见得和我们的实际任务完全匹配，我们可能需要自己fine-tuning。

为了进行预测，Transformers提供两种方法：pipeline和自己构造模型。两种方法都会在下面的例子里被用到。

本节的例子都是使用在某个数据集上fine-tuning过的模型，如果不是fine-tuning后的模型，那么它只会加载body的预训练参数，而head会随机初始化，因此它的预测结果也是随机的。

### 分类

下面是使用pipeline进行情感分类的例子，它使用的是GLUE数据集的sst2任务进行fine-tuning后的模型。

```
>>> from transformers import pipeline

>>> nlp = pipeline("sentiment-analysis")

>>> result = nlp("I hate you")[0]
>>> print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: NEGATIVE, with score: 0.9991

>>> result = nlp("I love you")[0]
>>> print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9999
```

接下来是判断两个句子是否相同含义(paraphrase)的任务，步骤如下：

* 从checkpoint构造一个Tokenizer和模型。这里使用的是BERT模型。

* 给定两个输入句子，通过tokenizer的__call__方法正确的构造输入，包括token类型和attention mask。

* 把输入传给模型进行预测，输出logits。

* 计算softmax变成概率。

代码如下：
```
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

>>> classes = ["not paraphrase", "is paraphrase"]

>>> sequence_0 = "The company HuggingFace is based in New York City"
>>> sequence_1 = "Apples are especially bad for your health"
>>> sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

>>> paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
>>> not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

>>> paraphrase_classification_logits = model(**paraphrase).logits
>>> not_paraphrase_classification_logits = model(**not_paraphrase).logits

>>> paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
>>> not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

>>> # Should be paraphrase
>>> for i in range(len(classes)):
...     print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
not paraphrase: 10%
is paraphrase: 90%

>>> # Should not be paraphrase
>>> for i in range(len(classes)):
...     print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
not paraphrase: 94%
is paraphrase: 6%
```

### 抽取式问答
抽取式问答是从文字中抽取一个问题的答案，最常见的数据集是SQuAD。如果我们想fine-tuning它，可以参考run_qa.py和run_tf_squad.py。下面的例子是利用fine-tuning好的模型来进行问答(预测)的例子：

```
>>> from transformers import pipeline

>>> nlp = pipeline("question-answering")

>>> context = r"""
... Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
... question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
... a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
... """
```

返回的预测返回的结果如下：
```
>>> result = nlp(question="What is extractive question answering?", context=context)
>>> print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
Answer: 'the task of extracting an answer from a text given a question.', score: 0.6226, start: 34, end: 96

>>> result = nlp(question="What is a good example of a question answering dataset?", context=context)
>>> print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
Answer: 'SQuAD dataset,', score: 0.5053, start: 147, end: 161
```

它包括"answer"的文字，start和end下标。

我们也可以自己来构造Tokenizer和模型，步骤如下：

* 构造Tokenizer和模型。
* 定义文本和一些问题。
* 对每一个问题构造输入，Tokenizer会帮我们插入合适的特殊符合和attention mask。
* 输入模型进行预测，得到是开始和介绍下标的logits
* 计算softmax并且选择概率最大的start和end
* 最终根据start和end截取答案文本

```
>>> from transformers import AutoTokenizer, AutoModelForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
>>> model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

>>> text = r"""
... 🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
... architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
... Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
... TensorFlow 2.0 and PyTorch.
... """

>>> questions = [
...     "How many pretrained models are available in 🤗 Transformers?",
...     "What does 🤗 Transformers provide?",
...     "🤗 Transformers provides interoperability between which frameworks?",
... ]

>>> for question in questions:
...     inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
...     input_ids = inputs["input_ids"].tolist()[0]
...
...     outputs = model(**inputs)
...     answer_start_scores = outputs.start_logits
...     answer_end_scores = outputs.end_logits
...
...     answer_start = torch.argmax(
...         answer_start_scores
...     )  # Get the most likely beginning of answer with the argmax of the score
...     answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
...
...     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
...
...     print(f"Question: {question}")
...     print(f"Answer: {answer}")
Question: How many pretrained models are available in 🤗 Transformers?
Answer: over 32 +
Question: What does 🤗 Transformers provide?
Answer: general - purpose architectures
Question: 🤗 Transformers provides interoperability between which frameworks?
Answer: tensorflow 2 . 0 and pytorch
```

我们拿到token的开始和结束下标后，需要用tokenizer.convert_ids_to_tokens先把id变成token，然后用convert_tokens_to_string把token变成字符串。而前面的pipeline把这些工作都直接帮我们做好了。

### 语言模型

和前面的任务相比，语言模型本身一般很少作为一个独立的任务。它的作用通常是用来预训练基础的模型，然后也可以使用领域的未标注数据来fine-tuning语言模型。比如我们的任务是一个文本分类任务，我们可以基于基础的BERT模型在我们的分类数据上fine-tuning模型。但是BERT的基础模型是基于wiki这样的语料库进行预训练的，不一定和我们的任务很match。而且标注的成本通常很高，我们的分类数据量通常不大。但是领域的未标注数据可能不少。这个时候我们我们可以用领域的未标注数据对基础的BERT用语言模型这个任务进行再次进行pretraining，然后再用标注的数据fine-tuning分类任务。

#### MLM

如果我们需要fine-tuning MLM，可以参考 run_mlm.py。下面是用pipeline的例子：

```
>>> from transformers import pipeline
>>> nlp = pipeline("fill-mask")
>>> from pprint import pprint
>>> pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))
[{'score': 0.1792745739221573,
  'sequence': '<s>HuggingFace is creating a tool that the community uses to '
              'solve NLP tasks.</s>',
  'token': 3944,
  'token_str': 'Ġtool'},
 {'score': 0.11349421739578247,
  'sequence': '<s>HuggingFace is creating a framework that the community uses '
              'to solve NLP tasks.</s>',
  'token': 7208,
  'token_str': 'Ġframework'},
 {'score': 0.05243554711341858,
  'sequence': '<s>HuggingFace is creating a library that the community uses to '
              'solve NLP tasks.</s>',
  'token': 5560,
  'token_str': 'Ġlibrary'},
 {'score': 0.03493533283472061,
  'sequence': '<s>HuggingFace is creating a database that the community uses '
              'to solve NLP tasks.</s>',
  'token': 8503,
  'token_str': 'Ġdatabase'},
 {'score': 0.02860250137746334,
  'sequence': '<s>HuggingFace is creating a prototype that the community uses '
              'to solve NLP tasks.</s>',
  'token': 17715,
  'token_str': 'Ġprototype'}]
```

上面会用到nlp.tokenizer.mask_token，它就是特殊的<mask>这个token。我们也可以自己构造Tokenizer和模型，步骤为：

* 构造Tokenizer和模型。比如可以使用DistilBERT从checkpoint加载预训练的模型
* 构造输入序列，把需要mask的词替换成tokenizer.mask_token
* 用tokenizer把输入变成ID list
* 获取预测的结果，它的size是词典大小，表示预测某个词的概率
* 获取topk个概率最大的词

代码如下：
```
>>> from transformers import AutoModelWithLMHead, AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
>>> model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

>>> sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

>>> input = tokenizer.encode(sequence, return_tensors="pt")
>>> mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

>>> token_logits = model(input).logits
>>> mask_token_logits = token_logits[0, mask_token_index, :]

>>> top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
```
注意这里需要使用AutoModelWithLMHead构造模型。

输出结果：
```
>>> for token in top_5_tokens:
...     print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.

```

#### CLM

CLM根据当前的文本预测下一个词，如果我们想fine-tuning，可以参考run_clm.py。我们可以根据概率采样下一个词，然后不断的重复这个过程来生成更多的文本。这里我们只采样下一个词，根据模型输出的logits，我们可以使用top_k_top_p_filtering()函数把非top k的去掉。

top_k_top_p_filtering函数的作用是把非top-k的logits变成负无穷大(默认，也可以传入其它值)，这样softmax时这些项就是0。除了保留top-k，也可以传入参数top_p，它的含义是滤掉概率小于它的项目。采样的时候可以使用multinomial函数进行采样，完整代码如下：

```
>>> from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
>>> import torch
>>> from torch.nn import functional as F

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelWithLMHead.from_pretrained("gpt2")

>>> sequence = f"Hugging Face is based in DUMBO, New York City, and"

>>> input_ids = tokenizer.encode(sequence, return_tensors="pt")

>>> # get logits of last hidden state
>>> next_token_logits = model(input_ids).logits[:, -1, :]

>>> # filter
>>> filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

>>> # sample
>>> probs = F.softmax(filtered_next_token_logits, dim=-1)
>>> next_token = torch.multinomial(probs, num_samples=1)

>>> generated = torch.cat([input_ids, next_token], dim=-1)

>>> resulting_string = tokenizer.decode(generated.tolist()[0])
```

预测的结果(可能，但是因为采样会不确定)如下：
```
>>> print(resulting_string)
Hugging Face is based in DUMBO, New York City, and has
```

### 文本生成

我们可以用上面采样的方式一个接一个的生成更多的文本，但是Transformers帮我们实现了这些逻辑。
```
from transformers import pipeline
text_generator = pipeline("text-generation")
print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))
```
比如上面的代码，我们提供一段context的文本，指定最多生成50个Token，do_sample为False指定选择概率最大的而不是采样，从而每次运行的结果都是固定的。默认会使用gpt-2的模型来生成文本，执行的结果如下：
```
[{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
```
我们也可以通过手动构造的方式换成XLNet模型：
```

```

GPT-2、OpenAi-GPT、CTRL、XLNet、Transfo-XL和Reformer等模型都可以用于生成文本。根据[XLNet-gen](https://github.com/rusiaaman/XLNet-gen#methodology)，XLNet通常需要padding一下才会达到比较好的效果，而GPT-2则不需要。关于文本生成更多细节，感兴趣的读者可以参考[这篇博客](https://huggingface.co/blog/how-to-generate)。

### 命名实体识别

这里把命名实体识别当成一个序列标注任务，对于CoNLL-2003任务，共有人名(Person)、地名(Location)、机构名(Organization)和其它(Miscellaneous)四类，使用B/I标签共有8个，加上其它标签共有9个：
* O 当前Token不输于任何命名实体
* B-MIS 其它类别命名实体的开始
* I-MIS 
* B-PER
* I-PER
* B-LOC
* I-LOC
* B-ORG
* I-ORG

下面是pipeline的用法：
```
from transformers import pipeline
nlp = pipeline("ner")
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very"
           "close to the Manhattan Bridge which is visible from the window."
```

执行结果为：
```
>>> print(nlp(sequence))
[
    {'word': 'Hu', 'score': 0.9995632767677307, 'entity': 'I-ORG'},
    {'word': '##gging', 'score': 0.9915938973426819, 'entity': 'I-ORG'},
    {'word': 'Face', 'score': 0.9982671737670898, 'entity': 'I-ORG'},
    {'word': 'Inc', 'score': 0.9994403719902039, 'entity': 'I-ORG'},
    {'word': 'New', 'score': 0.9994346499443054, 'entity': 'I-LOC'},
    {'word': 'York', 'score': 0.9993270635604858, 'entity': 'I-LOC'},
    {'word': 'City', 'score': 0.9993864893913269, 'entity': 'I-LOC'},
    {'word': 'D', 'score': 0.9825621843338013, 'entity': 'I-LOC'},
    {'word': '##UM', 'score': 0.936983048915863, 'entity': 'I-LOC'},
    {'word': '##BO', 'score': 0.8987102508544922, 'entity': 'I-LOC'},
    {'word': 'Manhattan', 'score': 0.9758241176605225, 'entity': 'I-LOC'},
    {'word': 'Bridge', 'score': 0.990249514579773, 'entity': 'I-LOC'}
]
```

### 摘要

pipeline的代码为：

```
from transformers import pipeline
summarizer = pipeline("summarization")
ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

>>> print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
[{'summary_text': 'Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]

```

我们也可以自己构造模型，比如使用Google的T5摘要模型：
```
from transformers import AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, 	early_stopping=True)
```

### 翻译

pipeline的用法：

```
from transformers import pipeline
translator = pipeline("translation_en_to_de")
print(translator("Hugging Face is a technology company based in New York and Paris", max_length=40))
```

自定义模型：

```
from transformers import AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
```

中文翻译：

```
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)

text = "I like to study Data Science and Machine Learning"
translated_text = translation(text, max_length=40)[0]['translation_text']
print(translated_text)
```

翻译结果为：
```
我喜欢学习数据科学和机器学习
```

## 模型总结

## 预处理数据

本节介绍Transformers处理数据的方法，主要的工具是[tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html)。我们可以使用与某个模型匹配的特定tokenizer，也可以通过AutoTokenizer类自动帮我们选择合适的tokenizer。
 
Tokenizer的作用是把输入文本切分成Token，然后把Token变成整数ID，除此之外它也会增加一些额外的特殊Token以处理特定的任务。

注意：如果我们要使用预训练的模型，那么一定要使用它的Tokenizer。

使用AutoTokenizer非常简单：
```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
```

### 基本用法

[PreTrainedTokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer)有很多方法，但是最常用的就是__call__：
```
>>> encoded_input = tokenizer("Hello, I'm a single sentence!")
>>> print(encoded_input)
{'input_ids': [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

前面已经介绍过，它返回一个dict，包含input_ids、token_type_ids和attention_mask。我们可以用decode方法把ID"恢复"成字符串：
```
>>> tokenizer.decode(encoded_input["input_ids"])
"[CLS] Hello, I'm a single sentence! [SEP]"
```

它会增加一些特殊的Token，比如[CLS]和[SEP]。并不是所有的模型都需要增加特殊Token，我们可以使用参数add_special_tokens=False来禁用这个特性。

我们可以一次处理一个batch的输入：

```
>>> batch_sentences = ["Hello I'm a single sentence",
...                    "And another sentence",
...                    "And the very very last one"]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
               [101, 1262, 1330, 5650, 102],
               [101, 1262, 1103, 1304, 1304, 1314, 1141, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]}

```

我们看到每个句子的长度是不同的，但是对于大部分应用，batch的处理通常会需要：
* padding短的输入到最长的句子，从而使得所有的输入一样长
* 如果某个(些)句子太长，truncate到一个最大长度，因为大部分模型都有一个最大的长度限制
* 返回tensor(默认是python的list)

下面是一个例子：
```
>>> batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(batch)
{'input_ids': tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
                      [ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
                      [ 101, 1262, 1103, 1304, 1304, 1314, 1141,  102,    0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
```

注意：我们并没有指定最大的长度，因为大部分模型都有一个最大的长度。如果某个某些没有长度限制，则truncation不起作用。

### 处理两个输入

有时候我们需要两个输入，比如计算两个句子的相似度或者QA等任务。对于BERT，它会增加一些特殊字符，最后形成：[CLS] Sequence A [SEP] Sequence B [SEP]。我们可以给__call__方法传入两个参数(不是一个list的参数)：
```
>>> encoded_input = tokenizer("How old are you?", "I'm 6 years old")
>>> print(encoded_input)
{'input_ids': [101, 1731, 1385, 1132, 1128, 136, 102, 146, 112, 182, 127, 1201, 1385, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

我们看到返回的 token_type_ids 和前面不同，前面全是0，而这里第二个句子的token对应1。并不是所有的模型都需要token_type_ids，因此Tokenizer会根据模型是否需要而返回它。我们也可以通过参数return_token_type_ids强制要求返回。

我们可以check一下：
```
>>> tokenizer.decode(encoded_input["input_ids"])
"[CLS] How old are you? [SEP] I'm 6 years old [SEP]"
```

我们也可以传入两个list，从而进行batch处理：
```
>>> batch_sentences = ["Hello I'm a single sentence",
...                    "And another sentence",
...                    "And the very very last one"]
>>> batch_of_second_sentences = ["I'm a sentence that goes with the first sentence",
...                              "And I should be encoded with the second sentence",
...                              "And I go with the very last one"]
>>> encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 8667, 146, 112, 182, 170, 1423, 5650, 102, 146, 112, 182, 170, 5650, 1115, 2947, 1114, 1103, 1148, 5650, 102],
               [101, 1262, 1330, 5650, 102, 1262, 146, 1431, 1129, 12544, 1114, 1103, 1248, 5650, 102],
               [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 1262, 146, 1301, 1114, 1103, 1304, 1314, 1141, 102]],
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

我们可以再次通过decode方法来验证：
```
>>> for ids in encoded_inputs["input_ids"]:
>>>     print(tokenizer.decode(ids))
[CLS] Hello I'm a single sentence [SEP] I'm a sentence that goes with the first sentence [SEP]
[CLS] And another sentence [SEP] And I should be encoded with the second sentence [SEP]
[CLS] And the very very last one [SEP] And I go with the very last one [SEP]
```

和前面的batch方法一样，我们也可以padding和truncate以及返回tensor：
```
batch = tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors="pt")
```

### 关于padding和truncating

* padding
    * True或者'longest' 表示padding到batch里最长的长度
    * 'max_length' 表示padding到另一个参数'max_length'的值，如果没有传入max_length，则padding到模型的最大长度
    * False或者'do_not_pad'，不padding，默认值。
* truncation
    * True或者'only_first'，把输入truncating到参数'max_length'或者模型的最大长度。如果有两个输入，则truncate第一个。
    * 'only_second'，把输入truncating到参数'max_length'或者模型的最大长度，如果有两个输入，只truncate第二个
    * 'longest_first'，先truncate长的那个输入，如果等长了则一人truncate一个
    * False或者'do_not_truncate'，不truncating，默认值。

* max_length
    * padding或者truncating的最大长度。如果传入None，则使用模型的最大长度，如果None并且模型没有最大长度，则不truncating/padding。


### Pre-tokenized

Pre-tokenized指的是提前进行了分词，但是并没有进行subword的处理。如果输入是Pre-tokenized，则可以指的is_split_into_words=True。例如：

```
>>> encoded_input = tokenizer(["Hello", "I'm", "a", "single", "sentence"], is_split_into_words=True)
>>> print(encoded_input)
{'input_ids': [101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

输入是提前分好的5个词，最终的输出是9个token，其中第一个和最后一个是特殊的token，因此真正的token是7个，这是5个词进行subword处理后的结果。如果我们不想让它加入特殊token，可以传入add_special_tokens=False。

注意这里的输入是一个list的字符串，如果我们想处理一个batch，那么可以传字符串list的list：

```
batch_sentences = [["Hello", "I'm", "a", "single", "sentence"],
                   ["And", "another", "sentence"],
                   ["And", "the", "very", "very", "last", "one"]]
encoded_inputs = tokenizer(batch_sentences, is_split_into_words=True)
```

同样的，如果我们的每个输入都是两个句子，那么可以传入两个这样的字符串list的list：
```
batch_of_second_sentences = [["I'm", "a", "sentence", "that", "goes", "with", "the", "first", "sentence"],
                             ["And", "I", "should", "be", "encoded", "with", "the", "second", "sentence"],
                             ["And", "I", "go", "with", "the", "very", "last", "one"]]
encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences, is_split_into_words=True)
```

我们也可以padding/truncating并且返回tensor：
```
batch = tokenizer(batch_sentences,
                  batch_of_second_sentences,
                  is_split_into_words=True,
                  padding=True,
                  truncation=True,
                  return_tensors="pt")
```


## 训练和fine-tuning

Transformers包里的类可以无缝的和PyTorch或者TensorFlow集成。本节会介绍在PyTorch或者TensorFlow进行训练的例子，同时Transformers也提供了Trainer类，我们更推荐使用它进行训练。

### 在PyTorch里进行Fine-tuning

所有不以TF开头的模型类都是[PyTorch的Module](https://pytorch.org/docs/master/generated/torch.nn.Module.html)，因此它很容易集成在PyTorch里。

假设我们使用BERT来进行文本二分类，我们可以使用from_pretrained加载预训练的模型。而且Transformers针对文本分类提供了BertForSequenceClassification，它可以在BERT的body上面再加上用于分类的"head"层。比如代码：
```
BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```
它会从bert-base-uncased加载预训练的模型，同时会构造一个随机初始化的输出为2的全连接head层。默认的模型是在"eval"模式，我们可以调用train()把它设置成训练模式：

```
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()
```

我们可以使用PyTorch的optimizer，但是也可以使用transformers提供的[AdamW()](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.AdamW)，它实现了gradient bias correction和decay。

这个optimzer可以让我们对不同的参数设置不同的超参数，比如下面的代码：

```
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
```

对bias和LayerNorm.weight这两组(Transformer的不同层都会叫这个名字，但是前缀不同)参数使用0.01的weight_decay，而其它参数为0。

接下来把一个batch的文本通过Tokenizer进行处理：
```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
```


然后可以调用模型的__call__方法，需要传入label，这个时候会计算loss：

```
labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
```

通过loss.backward()和optimizer.step()，我们实现了一个mini-batch的训练。我们也可以自己计算loss，这个时候就不需要传入label：
```
from torch.nn import functional as F
labels = torch.tensor([1,0])
outputs = model(input_ids, attention_mask=attention_mask)
loss = F.cross_entropy(outputs.logits, labels)
loss.backward()
optimizer.step()
```
如果我们需要在GPU上训练，那么就把模型和输入都调用to('cuda')把模型放到GPU上。我们也可以使用get_linear_schedule_with_warmup进行learning rate decay：
```
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
```

上面的代码会进行num_warmup_steps步warm up，然后线性的衰减到0。加了scheduler后每次迭代都需要调用scheduler.step：
```
loss.backward()
optimizer.step()
scheduler.step()
```

如果我们想freeze某些参数也很容易：
```
for param in model.base_model.parameters():
    param.requires_grad = False
```

### 在TensorFlow 2里进行训练

和PyTorch类似，在TensorFlow 2里，也可以使用 from_pretrained()函数加载预训练的模型：
```
from transformers import TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
```
 
接下来我们使用tensorflow_datasets来加载GLUE评测的MRPC数据。Transformers包提供了glue_convert_examples_to_features()函数来对MRPC数据集进行tokenize并且转换成TensorFlow的Dataset。注意Tokenizer是与TensorFlow或者PyTorch无关的，所以它的名词前面不需要加TF前缀。

```
from transformers import BertTokenizer, glue_convert_examples_to_features
import tensorflow as tf
import tensorflow_datasets as tfds
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = tfds.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
```

from_pretrained得到的是Keras的模型，因此我们很容易就可以对它进行训练：

```
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
model.fit(train_dataset, epochs=2, steps_per_epoch=115)
```

因为Transformers包会对模型进行转换，我们甚至可以把训练好的TensorFlow模型保存下来然后用PyTorch进行加载：

```
from transformers import BertForSequenceClassification
model.save_pretrained('./my_mrpc_model/')
pytorch_model = BertForSequenceClassification.from_pretrained('./my_mrpc_model/', from_tf=True)
```

### Trainer

We also provide a simple but feature-complete training and evaluation interface through Trainer() and TFTrainer(). You can train, fine-tune, and evaluate any 🤗 Transformers model with a wide range of training options and with built-in features like logging, gradient accumulation, and mixed precision.

除了上面的两种方法之外，Transformers还提供了[Trainer](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer)和[TFTrainer](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TFTrainer)。

```
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-large-uncased")

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)
```

TrainingArguments参数指定了训练的设置：输出目录、总的epochs、训练的batch_size、预测的batch_size、warmup的step数、weight_decay和log目录。

然后使用trainer.train()和trainer.evaluate()函数就可以进行训练和验证。我们也可以自己实现模型，但是要求它的forward返回的第一个参数是loss。注意：TFTrainer期望的输入是tensorflow_datasets的DataSet。

如果我们想计算除了loss之外的指标，需要给Trainer传入compute_metrics函数：

```
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```





