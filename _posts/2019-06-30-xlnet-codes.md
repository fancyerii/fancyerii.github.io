---
layout:     post
title:      "XLNet代码分析" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 深度学习
    - XLNet
---

本文介绍XLNet的代码，读者阅读前需要了解XLNet的原理，不熟悉的读者请先阅读[XLNet原理](/2019/06/30/xlnet-theory/)。

<!--more-->

**目录**
* TOC
{:toc}

## 训练的代价(钱)

根据论文里实验部分的说明：训练XLNet-Large是在512核心的TPU v3芯片上进行(We train XLNet-Large on 512 TPU v3 chips for 500K steps with an Adam optimizer)。[reddit的帖子](https://www.reddit.com/r/MachineLearning/comments/c59ikz/r_it_costs_245000_to_train_the_xlnet_model512_tpu/)，认为需要花费\\$245,000来训练这个模型，这立即引起了极大的关注——这以后还让不让没钱的人搞科研(炼丹调参数)了！不过根据[James Bradbury的twitter](https://twitter.com/jekbradbury/status/1143397614093651969)，一个TPU有4个Core，512Core就是64个TPU，因此成本要除以4也就是\\$61,440。不过即使是\\$61,400，换成人民币也要40多万，这代价一般人也接受不了啊，万一不小心参数设置不合理，跑了两天效果不行，40万就没了！

目前XLNet只提供英语的模型，没有像BERT那样提供中文和多语言版本的模型，根据这个[Issue](https://github.com/zihangdai/xlnet/issues/3)，估计短期内都没有中文版的支持。但是一般的实验室或者个人都很难有近百万的预算(不能假设一次就跑成功吧)来做这个事情，因此只能等国内不缺钱的大公司来搞这个事情了。

不过即使我们无法从头开始训练模型，学习其代码也是有用的。因为没有中文模型，所以这里我们只能用英文作为例子。

## 安装


和BERT不同，BERT使用Python实现分词和WordPiece的切分；而XLNet使用了[Sentence Piece ](https://github.com/google/sentencepiece)来实现分词和WordPiece切分。因此我们需要首先安装Sentence Piece。

安装Sentence Piece需要[安装Bazel](https://docs.bazel.build/versions/master/install.html)，请读者自行参考文档安装(如果自己Build Tensorflow的话也是需要Bazel的)。有了Bazel之后我们就可以从源代码安装Sentence Piece了：

```
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
# Build C++库
bazel build src:all
# 安装Python模块
pip install sentencepiece
```

当然我们还需要clone XLNet的代码：
```
https://github.com/zihangdai/xlnet.git
```

此外还需要下载预训练的模型：
```
# 请科学上网
wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
unzip cased_L-24_H-1024_A-16.zip
```

此外运行XLNet需要Tensorflow 1.11+(论文作者是使用1.13.1和Python2)，我试了Tensorflow 1.11+Python3.6也是可以的。

## Pretraining

### 训练数据格式
我们首先来看Pretraining，我们需要准备训练数据，这里只是为了阅读代码，因此我们准备很少的数据就行。它的格式类似于：

```
cat pretrain.txt
This is the first sentence.
This is the second sentence and also the end of the paragraph.<eop>
Another paragraph.

Another document starts here.
```
当然上面的数据也太少了点，读者可以把这些内容复制个几百次。我们简单的介绍训练数据的格式。每一行代表一个句子。如一个空行代表一个新的文档(document)的开始，一篇文档可以包括多个段落(paragraph)，我们可以在一个段落的最后加一个<eop>表示这个段落的结束(和新段落的开始)。

比如上面的例子，总共有两篇文档，第一篇3个句子，第二篇1个句子。而第一篇的三个句子又分为两个段落，前两个句子是一个段落，最后一个句子又是一个段落。

### 运行预处理数据

xlnet提供了一个Python脚本来预处理数据，我们首先来运行它：
```
python data_utils.py \
 --bsz_per_host=8 \
 --num_core_per_host=1 \
 --seq_len=128 \
 --reuse_len=64 \
 --input_glob=pretrain.txt \
 --save_dir=traindata \
 --num_passes=20 \
 --bi_data=True \
 --sp_path=/home/lili/data/xlnet_cased_L-24_H-1024_A-16/spiece.model \
 --mask_alpha=6 \
 --mask_beta=1 \
 --num_predict=21
```

这里简单的解释一些参数的含义：

* bsz_per_host 每个host的batch大小，这里是8。
    * 因为它是多个TPU同时训练，所以可能有多个host，我们这里只有一个host。
* num_core_per_host 每个host的TPU的个数，我这里用CPU，只能是1。
    * 注意：在Tensorflow(和很多深度学习框架)里，即使主板上插了多个CPU，也只能算一个设备，因为CPU对于软件来说是透明的，软件很难控制进程调度再那个CPU的那个核上。但是一个主板上插两个GPU，那么就是两个设备。
* seq_len 序列长度，这里改成较小的128
* reuse_len cache的长度，这里是64
* input_glob 输入的训练数据，可以用\*这样的通配符
* save_dir 输出目录
* num_passes 生成多少趟(因为随机排列，所以每次都不同)
* bi_data 是否双向的batch，参考前面的理论部分
* sp_path sentencepiece的模型，模型下载后自带了一个
* mask_alpha 
* mask_beta
* num_predict 预测多少个词


sp_path是sentencepiece的模型，如果是自己的数据，可以使用spm_train工具来训练自己的WordPiece模型。这个工具的路径可能是：
```
$ which spm_train
/home/lili/.cache/bazel/_bazel_lili/36da2a1b0d95a6943be2977e45dfcacf/execroot/com_google_sentencepiece/bazel-out/k8-fastbuild/bin/src/spm_train
```

那么可以用下面的命令训练自己的模型(从github里复制过来的，我并没有执行过，仅供参考)：
```
spm_train \
	--input=$INPUT \
	--model_prefix=sp10m.cased.v3 \
	--vocab_size=32000 \
	--character_coverage=0.99995 \
	--model_type=unigram \
	--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> \
	--user_defined_symbols=<eop>,.,(,),",-,–,£,€ \
	--shuffle_input_sentence \
	--input_sentence_size=10000000
```

### data_utils.py代码分析

我们首先来看怎么生成训练数据的。它的main函数会调用create_data()函数，这个函数会调用_create_data来创建Pretraining的数据。这个函数的核心代码为：

```
def _create_data(idx, input_paths):
  # 加载sentence-piece模型 
  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.sp_path)

  input_shards = []
  total_line_cnt = 0
  for input_path in input_paths:
    # 处理每一个文件的过程 



  input_data_list, sent_ids_list = [], []
  prev_sent_id = None
  for perm_idx in perm_indices:
    # 把不同文件的数据拼成一个大的向量前的预处理
    # 主要是处理sent_ids
  
  # 最终得到一个大的向量 
  input_data = np.concatenate(input_data_list)
  sent_ids = np.concatenate(sent_ids_list)
  
  # 这是最核心的函数，后面会讲
  file_name, cur_num_batch = create_tfrecords(
      save_dir=tfrecord_dir,
      basename="{}-{}-{}".format(FLAGS.split, idx, FLAGS.pass_id),
      data=[input_data, sent_ids],
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      bi_data=FLAGS.bi_data,
      sp=sp,
  )

  ....
```

原始的代码有点长，我们分解为如下几个部分：

* 加载sentence-piece模型
    * 这个就是前两行代码

* 处理每一个文件的过程 

* 拼接前的预处理和拼接

* 调用create_tfrecords函数

#### 处理每一个文件的过程 

这个过程读取每一个文件的每一行，然后使用sp切分成WordPiece，然后变成id，放到数组input_data里。另外还有一个sent_ids，用来表示句子。

```
for input_path in input_paths:
  input_data, sent_ids = [], []
  sent_id, line_cnt = True, 0
  tf.logging.info("Processing %s", input_path)
  for line in tf.gfile.Open(input_path):
    if line_cnt % 100000 == 0:
      tf.logging.info("Loading line %d", line_cnt)
    line_cnt += 1

    if not line.strip():
      if FLAGS.use_eod:
        sent_id = not sent_id
        cur_sent = [EOD_ID]
      else:
        continue
    else:
      if FLAGS.from_raw_text:
        cur_sent = preprocess_text(line.strip(), lower=FLAGS.uncased)
        cur_sent = encode_ids(sp, cur_sent)
      else:
        cur_sent = list(map(int, line.strip().split()))

    input_data.extend(cur_sent)
    sent_ids.extend([sent_id] * len(cur_sent))
    sent_id = not sent_id

  tf.logging.info("Finish with line %d", line_cnt)
  if line_cnt == 0:
    continue

  input_data = np.array(input_data, dtype=np.int64)
  sent_ids = np.array(sent_ids, dtype=np.bool)

  total_line_cnt += line_cnt
  input_shards.append((input_data, sent_ids))
```


上面的代码看起来很长，其实不复杂。对于每一个文件(我们这里只有一个)，最终是为了得到"input_data, sent_ids = [], []"两个list。

input_data里是放到这个文件的每一个WordPiece对应的ID，而sent_ids用于判断句子的边界。比如下面的例子：
```
input_data=[  52   27   18 ... 3091  193    9]
sent_ids=[ True  True  True ... False False False]
```

因为第一个句子是"This is the first sentence."，使用sp切分后变成"['▁this', '▁is', '▁the', '▁first', '▁sentence', '.']"，最后变成ID得到[52, 27, 18, 89, 3833, 9]。

而sent_ids是[True, True, True, True, True, True]，这个读者可能不明白，我们暂时不解释。

接着我们处理第二个句子"this is the second sentence and also the end of the paragraph.<eop>"，它被切分成"['▁this', '▁is', '▁the', '▁second', '▁sentence', '▁and', '▁also', '▁the', '▁end', '▁of', '▁the', '▁paragraph', '.', '<eop>']"，最后也变成ID序列。

而第二个句子对应的sent_ids是[False, ..., False]。

最后把两个句子的ID和sent_ids都放到input_data和sent_ids：
```
input_data=[52, 27, 18, 89, 3833, 9, 52, 27, 18, 205, 3833, 21, 77, 18, 239, 20, 18, 11636, 9, 8]
sent_ids=[True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
```

因此input_data是每一个WordPiece对应的ID的数组，而sent_ids可以判断哪些ID是属于一个句子的，也就是sent_ids通过交替的True和False来告诉我们句子的边界，比如前面的sent_ids的前6个为True，因此我们可以知道前6个WordPiece属于第一个句子，而后面的14个连续False告诉我们第二个句子有14个WordPiece。那么如果第三个句子有5个WordPiece，则我们可以猜测后面应该出现连续5个True。


关于WordPiece，不了解的读者可以参考[BERT模型详解](/2019/03/09/bert-theory/#%E8%BE%93%E5%85%A5%E8%A1%A8%E7%A4%BA)。如果一个WordPiece以"▁"开始，则表明它是一个词的开始，而不以"▁"开始的表明它是接着前面的。上面的例子每个词都是一个WordPiece，但是也有多个WordPiece对应一个词的，比如下面的例子对应一个词"9886.75"。
```
'▁9,', '88', '6', '.', '75'
```

此外上面的代码还有处理空行，用于表示一个新的Document的开始(取决于选项FLAGS.use_eod)，则会加一个特殊的Token EOD_ID。而段落的结束是使用<eop>表示，下面是一些特殊的符号及其ID：
```
special_symbols = {
    "<unk>"  : 0,
    "<s>"    : 1,
    "</s>"   : 2,
    "<cls>"  : 3,
    "<sep>"  : 4,
    "<pad>"  : 5,
    "<mask>" : 6,
    "<eod>"  : 7,
    "<eop>"  : 8,
}
```


#### 拼接前的预处理和拼接

通过前面的代码，我们可以把每一个文件都变成一个(input_data, sent_ids)pair，放到input_shards这个list里。但是我们还需要把不同文件的(input_data, sent_ids)拼接成更大的一个(input_data, sent_ids)。input_data可以直接拼接，但是sent_ids不行，为什么呢？我们假设第一个文件有3个句子，因此它的sent_ids类似[True,True,False,False,True,True]，而第二个文件是两个句子，[True,False]，那么直接拼起来就变成[True,True,False,False,True,True,True,False]，拼接后本来应该是5个句子，但是现在变成了4个！

因为第一个文件是True结尾，但是第二个是True开始，因此我们需要把第二个文件的True和False反过来，这就是预处理的代码，关键的代码都有注释：
```
  for perm_idx in perm_indices:
    input_data, sent_ids = input_shards[perm_idx]
    # 如果上一个文件的最后的sent_id和这个文件的开始的sent_id相同
    # 那么就得把当前这个文件的sent_id反过来
    if prev_sent_id is not None and sent_ids[0] == prev_sent_id:
      sent_ids = np.logical_not(sent_ids)

    # append到临时的list
    input_data_list.append(input_data)
    sent_ids_list.append(sent_ids)

    # 更新 `prev_sent_id`
    prev_sent_id = sent_ids[-1]
```
最后拼接成两个大的向量：
```
  input_data = np.concatenate(input_data_list)
  sent_ids = np.concatenate(sent_ids_list)
```


#### create_tfrecords函数
##### 准备数据
首先看前面部分的代码：
```
  data, sent_ids = data[0], data[1]

  num_core = FLAGS.num_core_per_host
  bsz_per_core = bsz_per_host // num_core

  if bi_data:
    assert bsz_per_host % (2 * FLAGS.num_core_per_host) == 0
    fwd_data, fwd_sent_ids = batchify(data, bsz_per_host // 2, sent_ids)

    fwd_data = fwd_data.reshape(num_core, 1, bsz_per_core // 2, -1)
    fwd_sent_ids = fwd_sent_ids.reshape(num_core, 1, bsz_per_core // 2, -1)

    bwd_data = fwd_data[:, :, :, ::-1]
    bwd_sent_ids = fwd_sent_ids[:, :, :, ::-1]

    data = np.concatenate(
        [fwd_data, bwd_data], 1).reshape(bsz_per_host, -1)
    sent_ids = np.concatenate(
        [fwd_sent_ids, bwd_sent_ids], 1).reshape(bsz_per_host, -1)
  else:
    data, sent_ids = batchify(data, bsz_per_host, sent_ids)

  tf.logging.info("Raw data shape %s.", data.shape)
```

在阅读这部分代码前我们先来了解它的作用。这个函数的前面部分的作用是整个语料库(一个很长的data和对应sent_ids)分成batch。比如假设data为：
```
1 2 3 4 .... 1001
```
并且batch为8，bi_data为True(两个方向)，则上面的代码首先把1001个数据分成8/2=4个部分，不能整除的扔掉，因此变成：
```
1 2 ... 250
251 252 ... 500
501 502 ... 750
751 752 ... 1000
```

然后加上反过来的数据：
```
250 ... 2 1
500 ... 252 251
750 ... 502 501
100 ... 752 751
```
最终变成：
```
1 2 ... 250
251 252 ... 500
501 502 ... 750
751 752 ... 1000
250 ... 2 1
500 ... 252 251
750 ... 502 501
100 ... 752 751
```

它主要会用到batchify函数为：

```
def batchify(data, bsz_per_host, sent_ids=None):
  num_step = len(data) // bsz_per_host
  data = data[:bsz_per_host * num_step]
  data = data.reshape(bsz_per_host, num_step)
  if sent_ids is not None:
    sent_ids = sent_ids[:bsz_per_host * num_step]
    sent_ids = sent_ids.reshape(bsz_per_host, num_step)

  if sent_ids is not None:
    return data, sent_ids
  return data
```
我们假设输入data是[3239,]，并且bsz_per_host为4，则每个batch得到3239//4=3236/4=809个steps。3239去掉不能整除的最后3个，就是3236个ID。然后把它resahpe成(4, 809)，sent_ids也是类似的操作。

##### 生成Pretraining的数据

在阅读代码前，我们看一下最终生成的每一个数据的样子，它如下图所示：

<a name='img13'>![](/img/xlnet/13.png)</a>
*图：Pretraining的数据*

A和B有两种关系，第一种它们是连续的上下文；第二种B是随机在data中选择的句子。

接下来是一个大的for循环：
```
  while i + seq_len <= data_len:
    ....
    i += reuse_len
```

上面的大while循环就是每次移动64(reuse_len)，首先固定64个作为cache。然后从i+reuse_len位置开始不断寻找句子，直到这些句子的Token数大于61(128-64-3)。比如：
```
64  65-75 76-90 91-128
```
上面的例子找到3个句子，这三个句子的Token数大于61了。然后以50%的概率选择如下两种方案生成A和B：

* A和B是连续的，因此从3个句子里随机的选择前面一部分作为A，剩下的作为B。比如有可能前两个句子是A，后一个是B。

* A和B不连续，因此这3个句子随机选一部分作为A，比如前两个句子，接着随机的从整个data里寻找一部分作为B。

当然上面只是大致的思路，细节很多：比如这三个句子的长度超过61了，那么需要从A或者B里删除一部分；比如随机的从data里选择B，很可能B是句子的中间，那么需要向前后两个方向"扩充"B(当然同时要从A的尾部删除相应的个数的Token)。这里就不介绍了，读者知道它的作用后阅读代码就会比较容易了。

接下来就是对这128个Token进行"Mask"了，这是通过_sample_mask函数实现的。它首先对前64个memory进行Mask，然后对后面64个也进行Mask。_sample_mask的代码比较细节，我这里只介绍它的大致思路。

* 首先随机选择n-gram的n，n的范围是[1,5]，这里假设n为2
* 然后计算上下文 "ctx_size = (n * FLAGS.mask_alpha) // FLAGS.mask_beta" 这里为2*6=12
* 然后随机的ctx_size(12)切分成l_ctx和r_ctx，假设为5和7
* 然后下标后移5(l_ctx)，因为后移5之后可能不是一个词，因此持续后移找到n-gram开始的位置
* 寻找n-gram开始的位置寻找n个词(n个词可能多于n个Token)
* 然后从n-gram介绍的地方后移7(r_ctx)个位置，并且持续后移直到遇到词的开始(以"▁"开始的Token)

这样就找到了一个被Mask的n-gram以及它的左右(大致)l_ctx和r_ctx个Token。如果Mask的Token到达我们的预期(goal_num_predict)就退出，否则从结束的下标开始持续这个过程。最终我们需要得到的数据是feature，下面是一个feature的示例值：

```
input: [   52    27    18    89  3833     9    52    27    18   205  3833    21
    77    18   239    20    18 11636     9     8   245 11636     9     7
   245  2402  3091   193     9     7    52    27    18    89  3833     9
    52    27    18   205  3833    21    77    18   239    20    18 11636
     9     8   245 11636     9     7   245  2402  3091   193     9     7
    52    27    18    89  3833     9    52    27    18   205  3833    21
    77    18   239    20    18 11636     9     8   245 11636     9     7
   245  2402  3091   193     9     7    52    27    18    89  3833     9
    52    27    18   205  3833    21    77    18   239    20    18 11636
     9     8   245 11636     9     7   245  2402  3091   193     9     4
    52    27    18    89  3833     9     4     3]

tgt: [   27    18    89  3833     9    52    27    18   205  3833    21    77
    18   239    20    18 11636     9     8   245 11636     9     7   245
  2402  3091   193     9     7    52    27    18    89  3833     9    52
    27    18   205  3833    21    77    18   239    20    18 11636     9
     8   245 11636     9     7   245  2402  3091   193     9     7    52
    27    18    89  3833     9    52    27    18   205  3833    21    77
    18   239    20    18 11636     9     8   245 11636     9     7   245
  2402  3091   193     9     7    52    27    18    89  3833     9    52
    27    18   205  3833    21    77    18   239    20    18 11636     9
     8   245 11636     9     7   245  2402  3091   193     9     7    52
    27    18    89  3833     9    52     3     3]

is_masked: [False False False False False  True  True False False False False False
 False False False False False  True  True  True  True  True False  True
  True False False False False False False False False False False False
  True False False False False False False False False False False False
 False False  True False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False  True False False False False False False
 False False False False False False  True  True  True  True  True False
 False False False False False False False False False False  True  True
 False False False False False False False False False False False False
 False  True  True False False False False False]

seg_id: [0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]

label: 1
```

这些变量的含义是：

* input
    * 长度为128的输入，前64个是mem，后面64个是A和B(加上2个SEP一个CLS)
* tgt
    * 长度128，除了最后两个是CLS，前面126是input对应的下一个目标值
* label
    * 1表示A和B是连续的句子
* seg_id
    * 表示输入input的segment，mem+A+SEP是0，B+SEP是1，最后一个CLS是2
* is_masked
    * 表示这128个里哪些位置是Mask的

最终这5个变量都会作为features放到一个tf.train.Example写到TFRecord文件里。



下面是while循环的主要代码：

```
  while i + seq_len <= data_len:
    features = []
    for idx in range(bsz_per_host):
      inp = data[idx, i: i + reuse_len]
      tgt = data[idx, i + 1: i + reuse_len + 1]

      results = _split_a_and_b(
          data[idx],
          sent_ids[idx],
          begin_idx=i + reuse_len,
          tot_len=seq_len - reuse_len - 3,
          extend_target=True)
      if results is None:
        tf.logging.info("Break out with seq idx %d", i)
        all_ok = False
        break

      # unpack the results
      (a_data, b_data, label, _, a_target, b_target) = tuple(results)

      # sample ngram spans to predict
      reverse = bi_data and (idx // (bsz_per_core // 2)) % 2 == 1
      if FLAGS.num_predict is None:
        num_predict_0 = num_predict_1 = None
      else:
        num_predict_1 = FLAGS.num_predict // 2
        num_predict_0 = FLAGS.num_predict - num_predict_1
      mask_0 = _sample_mask(sp, inp, reverse=reverse,
                            goal_num_predict=num_predict_0)
      mask_1 = _sample_mask(sp, np.concatenate([a_data, sep_array, b_data,
                                                sep_array, cls_array]),
                            reverse=reverse, goal_num_predict=num_predict_1)

      # concatenate data
      cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                 sep_array, cls_array])
      seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] +
                [1] * b_data.shape[0] + [1] + [2])
      assert cat_data.shape[0] == seq_len
      assert mask_0.shape[0] == seq_len // 2
      assert mask_1.shape[0] == seq_len // 2

      # the last two CLS's are not used, just for padding purposes
      tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
      assert tgt.shape[0] == seq_len

      is_masked = np.concatenate([mask_0, mask_1], 0)
      if FLAGS.num_predict is not None:
        assert np.sum(is_masked) == FLAGS.num_predict

      feature = {
          "input": _int64_feature(cat_data),
          "is_masked": _int64_feature(is_masked),
          "target": _int64_feature(tgt),
          "seg_id": _int64_feature(seg_id),
          "label": _int64_feature([label]),
      }
      features.append(feature)

    if all_ok:
      assert len(features) == bsz_per_host
      for feature in features:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())
      num_batch += 1
    else:
      break

    i += reuse_len

```

运行后会在输出目录生成如下的内容：
```
~/codes/xlnet/traindata$ tree
.
├── corpus_info.json
└── tfrecords
    ├── record_info-train-0-0.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.json
    └── train-0-0.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.tfrecords

```


### 运行train_gpu.py

train.py是在TPU上训练的代码，如果是GPU(或者CPU)请使用这个脚本，下面是使用前面生成的数据进行训练的脚本：
```
python train_gpu.py \
   --record_info_dir=traindata/tfrecords \
   --train_batch_size=8 \
   --seq_len=128 \
   --reuse_len=64 \
   --mem_len=96 \
   --perm_size=32 \
   --n_layer=6 \
   --d_model=1024 \
   --d_embed=1024 \
   --n_head=16 \
   --d_head=64 \
   --d_inner=4096 \
   --untie_r=True \
   --mask_alpha=6 \
   --mask_beta=1 \
   --num_predict=21 \
   --model_dir=mymodel\
   --uncased=true \
   --num_core_per_host=1
```


### train_gpu.py代码
训练主要是调用函数train，它的主要代码为：
```
def train(ps_device):
  train_input_fn, record_info_dict = data_utils.get_input_fn(
      tfrecord_dir=FLAGS.record_info_dir,
      split="train",
      bsz_per_host=FLAGS.train_batch_size,
      seq_len=FLAGS.seq_len,
      reuse_len=FLAGS.reuse_len,
      bi_data=FLAGS.bi_data,
      num_hosts=1,
      num_core_per_host=1, # set to one no matter how many GPUs
      perm_size=FLAGS.perm_size,
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      uncased=FLAGS.uncased,
      num_passes=FLAGS.num_passes,
      use_bfloat16=FLAGS.use_bfloat16,
      num_predict=FLAGS.num_predict)
  ....

  # 忽略一个host上多个(num_core_per_host)设备(GPU)的代码

    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

      # The mems for each tower is a dictionary
      mems_i = {}
      if FLAGS.mem_len:
        mems_i["mems"] = create_mems_tf(bsz_per_core)

      loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
          is_training=True,
          features=examples[i],
          mems=mems_i)

    ....


  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
      gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]

    total_loss, prev_step = 0., -1
    while True:
      feed_dict = {}
      for i in range(FLAGS.num_core_per_host):
        for key in tower_mems_np[i].keys():
          for m, m_np in zip(tower_mems[i][key], tower_mems_np[i][key]):
            feed_dict[m] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)

      loss_np, tower_mems_np, curr_step = fetched[:3]
      total_loss += loss_np

```

如果忽略多设备(GPU)训练的细节，train的代码结构其实并不复杂，它大致可以分为3部分：

* 调用data_utils.get_input_fn得到train_input_fn

* 调用single_core_graph构造XLNet网络

* 使用session运行fetches进行训练


[未完待续！]
