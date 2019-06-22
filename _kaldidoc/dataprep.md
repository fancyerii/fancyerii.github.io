---
layout:     post
title:      "数据准备"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Data preparation](http://kaldi-asr.org/doc/data_prep.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}



## Introduction


学习完tutorial之后，你可能想知道怎么用自己的训练数据来训练模型(其实在[Kaldi for Dummies教程]({{ site.baseurl }}/kaldidoc/kaldi-for-dummy)已经实现了)。这里假设你使用最新的脚本(通常是s5，比如egs/rm/s5/)。除了本文之外，你也可以参考这些目录下的脚本。顶级的run.sh脚本(比如egs/rm/s5/run.sh)有一些命令是和数据准备相关的。名为local的目录总是与特定数据集相关的脚本。比如对于RM数据集的数据处理脚本是local/rm_data_prep.sh，run.sh里与之相关的代码为：

```
local/rm_data_prep.sh /export/corpora5/LDC/LDC93S3A/rm_comp

utils/prepare_lang.sh data/local/dict '!SIL' data/local/lang data/lang

local/rm_prepare_grammar.sh      # Traditional RM grammar (bigram word-pair)
local/rm_prepare_grammar_ug.sh   # Unigram grammar (gives worse results, but
                                 # changes in WER will be more significant.)

```

上面有三个数据处理脚本是local目录下的，这些脚本只能给rm数据集使用，而另外一个utils/prepare_lang.sh是"通用"的脚本，几乎每个recipe都会用到。

而wsj的run.sh里与数据准备相关的是：
```
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B


if [ $stage -le 0 ]; then
  # data preparation.
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

  utils/prepare_lang.sh data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;

  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

```

和前面差不多，也是3个local下的脚本并且使用utils/prepare_lang.sh来准备数据。

此外WSJ还有一些训练语言模型的脚本，但是上面列举的都是最重要的脚本。

数据准备阶段的输出包含两组数据。第一组是"data"(比如data/train)而另一组是"language"(比如data/lang)。data部分与你使用的录音相关，而lang部分是语言相关的部分，包括发音词典(lexicon)、phone集合、以及Kaldi需要的其它信息。如果你已经训练好了模型，只是想用它来识别(解码)，那么只需要准备data就行了。

## data部分


data部分的示例可以参考data/train。注意：目录名"data/train"并没有特殊之处。完全可以叫做其它的名字比如"data/eval2000"。我们看一下egs/swbd/s5的data部分：

```
s5# ls data/train
cmvn.scp  feats.scp  reco2file_and_channel  segments  spk2utt  text  utt2spk  wav.scp
```

并不是所有的文件都一样重要。有些recipe没有"segmentation"信息(也就是每个utterance是一个文件)，你只需要自己创建"utt2spk"、"text"和"wav.scp"文件，"segments"和"reco2file_and_channel"是可选的，其余的文件都是通过标准的脚本生成的。

下面我们会介绍这个目录下的文件，首先从需要自己创建的开始。


### 需要自己创建的文件

#### text
text文件包含每个utterance对应的transcript。
```
s5# head -3 data/train/text
sw02001-A_000098-001156 HI UM YEAH I'D LIKE TO TALK ABOUT HOW YOU DRESS FOR WORK AND
sw02001-A_001980-002131 UM-HUM
sw02001-A_002736-002893 AND IS
```
这个文件每一行都是一句话，用空格分开，第一列是utterance-id，后面是分词后的transcript。utterance-id可以是任意字符串，但是要求唯一。但如果需要告诉kaldi这句话是哪个speaker说的，通常的约定是把speaker-id作为utterance-id的前缀。比如A02是一个speaker-id，A02_000表示A02说的第一句话。我们用下划线来连接speaker-id和后面的字符串，更加安全的方法是用”-“来连接。不过这里因为保证后面的字符串都是一个定长的字符串，所以用下划线也不会有问题。如果是我们自己准备数据，尽量按照这种格式准备数据，否则kaldi可能出现奇怪的问题。此外很多脚本文件都会用空格来分隔，所有路径里最好不要用空格。

#### wav.scp

```
s5# head -3 data/train/wav.scp
sw02001-A /home/dpovey/kaldi-trunk/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 /export/corpora3/LDC/LDC97S62/swb1/sw02001.sph |
sw02001-B /home/dpovey/kaldi-trunk/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 /export/corpora3/LDC/LDC97S62/swb1/sw02001.sph |
```

这个文件告诉kaldi每个utterance-id对应的录音文件的位置。这里是非常简单的方法，用空格分开的两列。第一列是utterance-id(或者recording-id，参考segments)，第二列是录音文件的路径。我们这里的例子第二列是录音文件的路径，但是kaldi实际要求的是extended-filename，这个extended-filename可以是一个wav文件，但是也可以是一个命令，这个命令能产生一个wav文件。比如上面的例子，extended-filename是一个命令，这个命令的输出是wav文件。这里是使用sph2pipe工具把sph文件转成wav文件，注意最后有一个管道符号"\|"。


scp要求wav是单声道的，如果是立体声的需要用sox工具抽取特定的声道。

#### segments

```
s5# head -3 data/train/segments
sw02001-A_000098-001156 sw02001-A 0.98 11.56
sw02001-A_001980-002131 sw02001-A 19.8 21.31
sw02001-A_002736-002893 sw02001-A 27.36 28.93
```
这个文件的每行为4列，第1列是utterance-id，第2列是recording-id。什么是recording-id？如果有segments文件，那么wav.scp第一列就不是utterance-id而变成recording-id了。recording-id表示一个录音文件的id，而一个录音文件可能包含多个utterance。因此第3列和第4列告诉kaldi这个utterance-id在录音文件中开始和介绍的实际。比如上面的例子，utterance “sw02001-A_000098-001156”在sw02001-A这个录音文件里，开始和结束时间分别是0.98秒到11.56秒。


#### reco2file_and_channel

这个文件是用NIST的sclite工具打分时才需要。
```
s5# head -3 data/train/reco2file_and_channel
sw02001-A sw02001 A
sw02001-B sw02001 B
sw02005-A sw02005 A
```

它的格式为：
```
<recording-id> <filename> <recording-side (A or B)>
```

filename通常是sph文件的名字(不含路径)，但是也可以是stm文件中的任意ID。recording-side用于电话，表示这是哪个说话人(电话是两个人说话)。如果你没有stm文件，也不知道它是什么意思，那么通常不需要这个文件。

#### utt2spk

```
s5# head -3 data/train/utt2spk
sw02001-A_000098-001156 2001-A
sw02001-A_001980-002131 2001-A
sw02001-A_002736-002893 2001-A
```

这个文件告诉kaldi，utterance-id对应哪个speaker-id。比如上面的文件告诉我们A02_000这个句子的spaker-id是A02。speaker-id不一定要求准确的对应到某个人，只要大致准确就行。如果我们不知道说话人是谁，那么我们可以把speaker-id设置成utterance-id，这是比较安全的做法。一种常见的错误做法是把未知说话人的句子都对应到一个“全局”的speaker，这样的坏处是它会使得cepstral mean normalization变得无效。


#### spk2gender

这个文件不是必须的，它说明说话人的性别。

```
s5# head -3 ../../rm/s5/data/train/spk2gender
adg0 f
ahh0 m
ajp0 m
```

注意：上面的文件都要求排好序(并且使用LC_ALL=C)，如果没有排序，后续的脚本会出问题！需要排序的原因读者在学习[The Table concept](http://kaldi-asr.org/doc/io.html#io_sec_tables)之后就会理解。这是和Kaldi的I/O框架有关，排序最终的目的是为了能够在一个流上提供随机的访问能力，因为很多的流的来源是上一个命令的输出，这是不能像文件那样支持fseek的。此外很多Kaldi命令需要读取多个流，然后用类似归并排序的方式合并它们，这也要求它们是排好序的。

如果你的数据是来自NIST，那么可能还有stm和glm文件。这些文件用于计算WER。注意，我们在这里把score.sh放到local下，说明这个score.sh是只能用于这个数据集的，因为不是所有的脚本都能识别stm和glm文件。Switchboard的recipe里有一个egs/swbd/s5/local/score_sclite.sh，它就能处理stm和glm文件。这个脚本最终被egs/swbd/s5/local/score.sh调用，这个脚本是用于打分。

除了上面的几个文件，其它文件都是kaldi的脚本自动帮我们生成的。我们下面来逐个分析。

### 脚本生成的文件

#### spk2utt

这个文件和utt2spk反过来，它存储的是speaker和utterance的对应关系，这是一对多的关系，可以使用脚本来得到：
```
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
```

下面是一个示例spk2utt文件(不是Switchboard数据集的)：
```
A04 A04_000 A04_001 A04_002 A04_003 A04_005 A04_006 A04_007 A04_008 ...
```

它的意思是说话人A04说了"A04_000"、"A04_001"等文件。

#### feats.scp

```
s5# head -3 data/train/feats.scp
sw02001-A_000098-001156 /home/dpovey/kaldi-trunk/egs/swbd/s5/mfcc/raw_mfcc_train.1.ark:24
sw02001-A_001980-002131 /home/dpovey/kaldi-trunk/egs/swbd/s5/mfcc/raw_mfcc_train.1.ark:54975
sw02001-A_002736-002893 /home/dpovey/kaldi-trunk/egs/swbd/s5/mfcc/raw_mfcc_train.1.ark:62762
```
这个文件是提取的mfcc特征，第一列是utterance-id，第二列是mfcc特征文件。raw_mfcc_train.1.ark:24表示特征在raw_mfcc_train.1.ark这个文件，开始的位置是24，这样Kaldi会打开这个文件，然后fseek(24)，然后开始读取。我们通常可以用如下命令来生成feats.scp文件：
```
steps/make_mfcc.sh --nj 8 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
```
参数data/train是输入目录，exp/make_mfcc是log的目录，而$mfccdir是输出mfcc的目录。这个命令通常是在run.sh里调用。

#### cmvn.scp

这个文件包括倒谱(cepstral)均值和方程归一化(normalization)需要的统计信息，每个speaker一行。
```
s5# head -3 data/train/cmvn.scp
2001-A /home/dpovey/kaldi-trunk/egs/swbd/s5/mfcc/cmvn_train.ark:7
2001-B /home/dpovey/kaldi-trunk/egs/swbd/s5/mfcc/cmvn_train.ark:253
2005-A /home/dpovey/kaldi-trunk/egs/swbd/s5/mfcc/cmvn_train.ark:499
```

和feat.scp不同，cmvn.scp的key是speaker-id而不是utterance-id。通常使用下面的脚本来生成这个文件：
```
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
```

因为数据有可能格式不对，所有这里提供了脚本来检查：
```
utils/validate_data_dir.sh data/train
```

也可以使用下面的脚本修复数据的错误：
```
utils/fix_data_dir.sh data/train
```

上面的例子是对训练数据用检查脚本和修复脚本，实际也可以对任何(验证和测试)数据进行操作。上面的脚本主要是检查和修复排序的错误，此外它也会去掉这样的utterance——它们缺少对应的transcript或者特征文件。

## lang部分

下面我们来看lang部分：
```
s5# ls data/lang
L.fst  L_disambig.fst  oov.int	oov.txt  phones  phones.txt  topo  words.txt
```

可能还有类似的目录，比如data/lang_test，它比lang会多一个G.fst。G.fst是语言模型的FST，因为训练的时候是不需要语言模型的，只有在测试时才需要。
```
s5# ls data/lang_test
G.fst  L.fst  L_disambig.fst  oov.int  oov.txt	phones	phones.txt  topo
```
lang_test就是复制lang的内容然后加入G.fst。lang目录看起来很简单，没有几个文件，但是不要被表象迷惑了，因为phones是一个目录：
```
s5# ls data/lang/phones
context_indep.csl  disambig.txt         nonsilence.txt        roots.txt    silence.txt
context_indep.int  extra_questions.int  optional_silence.csl  sets.int     word_boundary.int
context_indep.txt  extra_questions.txt  optional_silence.int  sets.txt     word_boundary.txt
disambig.csl       nonsilence.csl       optional_silence.txt  silence.csl
```

phones子目录包括phone set的很多信息。它下面有一些同名但是不同后缀的文件，有3种后缀：txt、int和csl。其中txt是人可读的格式。我们不需要自己来生成这些文件，Kaldi提供了一个通用的utils/prepare_lang.sh，只需要我们为这个脚本提供一些参数。在介绍这个脚本之前，我们先了解一下lang下面文件的内容和格式。了解了它们的格式之后，我们再来介绍怎么使用prepare_lang.sh来快速的生成这个目录的内容。

## lang目录的内容


### phones.txt和words.txt
首先介绍phones.txt和words.txt，这两个文件都是符号表(符号到ID的映射)，这是OpenFst的格式。

```
<eps> 0
SIL 1
SIL_B 2
```
phones.txt是因子和id的映射，WFST里用的都是id。脚本utils/int2sym.pl用于把id转换成符号，而utils/sym2int.pl把符号转换成id。对于英文来说，最常见的音子集合是[ARPABET](https://web.stanford.edu/class/cs224s/arpabet.html)，对于汉语，我们常见的音子就是拼音，但是汉语是的韵母是有声调的，不同的声调是不同的因子。因此a1到a5分别表示ā、á、ǎ、à和a，代表阴平、阳平、上声、去声和轻声。

```
s5# head -3 data/lang/words.txt
<eps> 0
!SIL 1
-'S 2
```
words.txt是词与id的映射。

### L.fst和L_disambig.fst

L.fst是lexicon的WFST格式。L是发音词典的WFST，详细内容可以参考[Speech Recognition with Weighted Finite-State Transducers](http://www.cs.nyu.edu/~mohri/pub/hbka.pdf)。

而L_disambig.fst引入了#1、#2等消歧符号并且增加了输入为#0的self-loop。self-loop使得L可以把消歧符号传递到G中。可以参考[Disambiguation symbols](http://kaldi-asr.org/doc/graph.html#graph_disambig)。不过我们通常用脚本而不需要自己处理它。

### oov.txt
```
s5# cat data/lang/oov.txt
<UNK>
```
oov.txt里是未登录词(Out of Vocabulary Words)。这里把所有未登录词映射成一个特殊词<UNK>。字符串 "<UNK>" 并没有任何特殊之处，我们可以换成其它字符串(当然不能和正常的词相同)。比较重要的一点是这个词通常只有一个特殊的phone，而且这个特殊的phone用来对应所有的"garbage phone"，比如把各种噪声、特殊声音都用这个phone来表示。我们可以看一下它对应的phone：
```
s5# grep -w UNK data/local/dict/lexicon.txt
<UNK> SPN
```
grep 命令去lexicon.txt查找UNK，结果确实它对应的phone是特殊的SPN，SPN是spoken noise的缩写。

如果是RM recipe的话，oov.txt包含的是silence的词"!SIL"。这么做的原因是因为RM数据集没有OOV的词，所以随便选择了一个词，在RM里用"!SIL"之外的其它词也没区别。


### topo
```
s5# cat data/lang/topo
<Topology>
<TopologyEntry>
<ForPhones>
21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.75 <Transition> 1 0.25 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.75 <Transition> 2 0.25 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.75 <Transition> 3 0.25 </State>
<State> 3 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones>
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.25 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 2 <PdfClass> 2 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 3 <PdfClass> 3 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 4 <PdfClass> 4 <Transition> 4 0.75 <Transition> 5 0.25 </State>
<State> 5 </State>
</TopologyEntry>
</Topology>
```
这个文件是HMM的拓扑结构定义，通常在语音识别中我们使用的Bakis结构——一个状态可以自跳转保持状态不变，可以跳到下一个状态，也可以跳过下一个状态跳到下下个状态(说话特别快的时候)。phone 1-20是各种静音和噪声，这么多的原因是word-position的依赖。所谓的word-position是只同一个phone在不同的位置是不同的，比如：
```
good 
dog
```
这个两个词的phone /g/是不同的，第一个是g_B表示处于词开始的g，而dog里的g是g_E，表示它是一个词的最后一个音素。如果一个词由三个以上音素组成，那么中间的都是用_I表示。如果一个词只有一个音素，那么用_S表示，这种表示方法在NLP的序列标注里也非常常见。

普通的phone通常都是3状态的，而silence这里是5状态的，因为silence通常更长。

data/lang/phones/下的很多文件都是定义不同的phone set，它们通常包含三种格式。一种是.txt格式：
```
s5# head -3 data/lang/phones/context_indep.txt
SIL
SIL_B
SIL_E
```

一种是int格式：
```
s5# head -3 data/lang/phones/context_indep.int
1
2
3
```

还有csl格式：
```
s5# cat data/lang/phones/context_indep.csl
1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20
```

它们包含相同的信息，我们下面只介绍txt格式的文件。


### context_indep.txt


这个文件的内容是上下文无关的phone，对于这些phone，我们构建的决策树不会问关于它左边和右边上下文的问题。事实上，我们会构建一个很小的决策树，它只会问central phone和HMM状态(它是第几个状态)的问题。当然这依赖于"roots.txt"文件，下面我们会介绍这个文件。关于决策树的更多内容请参考[How decision trees are used in Kaldi](http://kaldi-asr.org/doc/tree_externals.html)。


这个文件包括silence(包括SIL、SIL_B和SIL_E)、SPN(spoken noise)、NSN(non-spoken noise)和LAU(laugh)。

```
# cat data/lang/phones/context_indep.txt
SIL
SIL_B
SIL_E
SIL_I
SIL_S
SPN
SPN_B
SPN_E
SPN_I
SPN_S
NSN
NSN_B
NSN_E
NSN_I
NSN_S
LAU
LAU_B
LAU_E
LAU_I
LAU_S
```

这些phone由于word-position的依赖有很多的变体；但并非每种变体都会用到。这里的SIL是可选的silence(在两个词之间可能会有静音)，SIL_B是作为一个词的开始(这是不可能的)，SIL_I是指一个词的中间有静音(这也不太可能存在)，SIL_E是一个词结尾的静音(这也不可能存在)，而SIL_S表示某个词只有这一个静音的phone。当我们的transcript显式的标注出静音的时候，我们可能定义一个叫"静音"的词的时候，我们可以用SIL_S来作为这个词的发音。当然通常很少会碰到这种情况，所以除了SIL，其它的SIL_\*很少会出现。

### silence.txt和nonsilence.txt
这两个文件分别包含silence的phone和非silence的phone，并且它们是没有交集的，而且它们的并集是所有的phone。在我们这个例子里，silence.txt和context_indep.txt是恰巧一样的。所谓的非silence的phone，我们认为是可以对它进行各种线性变换的phone。比如全局的LDA和MLLT；或者说话人相关的自适应变换比如fMLLR。根据我们的经验，silence是不值得做这些变换的(比如我们认为不同人说同一个"a"是有差别的，但是我们认为静音对于所有人都是一样的，比如背景噪音是没有必要根据说话人做什么变换的)。我们的实际经验是把所有的silence、背景噪声和发音噪声都作为"silence"的phone，而其它"真正"的音素当成非silence phone。

```
s5# head -3 data/lang/phones/silence.txt
SIL
SIL_B
SIL_E
s5# head -3 data/lang/phones/nonsilence.txt
IY_B
IY_E
IY_I
```

### disambig.txt

这个文件包含所有的消岐符号。关于消歧符号请参考[http://kaldi-asr.org/doc/graph.html#graph_disambig](http://kaldi-asr.org/doc/graph.html#graph_disambig)。

```
s5# head -3 data/lang/phones/disambig.txt
#0
#1
#2
```
消歧符号出现了phones.txt里，我们把它当成phone。

### optional_silence.txt
可选的silence，在两个词之间可能会有silence，因此需要定义可选的silence。 
```
s5# cat data/lang/phones/optional_silence.txt
SIL
```


通过在lexicon FST的每个词的结尾(也包括一个utterance的开始)增加一个可选的SIL，我们可以实现这两个词之间的可选的经验(可选的元音是两个词说得很快中间就没有静音了)。为什么在phones里指明而不只是出现了L.fst里的原因是比较复杂的，我们这里不深入讨论。

### sets.txt 
这个文件的每一行是一个phone集合，把phone聚类在一起的目的是用于创建上下文相关的问题。在Kaldi里构建决策树时我们并不使用语言学家定义的问题，而是自动聚类出来的问题，所谓的一个问题其实就是一个phone的集合，不清楚的读者可以参考[Kaldi教程(二)]({{ site.baseurl }}/kaldidoc/tutorial2monophone%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)。

在这里，sets.txt的每一行就是一个phone的所有word-postion的依赖的变体：
```
s5# head -3 data/lang/phones/sets.txt
SIL SIL_B SIL_E SIL_I SIL_S
SPN SPN_B SPN_E SPN_I SPN_S
NSN NSN_B NSN_E NSN_I NSN_S
```


### extra_questions.txt

除了自动聚类的问题，我们也可以通过这个文件提供决策树聚类的问题。

```
s5# cat data/lang/phones/extra_questions.txt
IY_B B_B D_B F_B G_B K_B SH_B L_B M_B N_B OW_B AA_B TH_B P_B OY_B R_B UH_B AE_B S_B T_B AH_B V_B W_B Y_B Z_B CH_B AO_B DH_B UW_B ZH_B EH_B AW_B AX_B EL_B AY_B EN_B HH_B ER_B IH_B JH_B EY_B NG_B
IY_E B_E D_E F_E G_E K_E SH_E L_E M_E N_E OW_E AA_E TH_E P_E OY_E R_E UH_E AE_E S_E T_E AH_E V_E W_E Y_E Z_E CH_E AO_E DH_E UW_E ZH_E EH_E AW_E AX_E EL_E AY_E EN_E HH_E ER_E IH_E JH_E EY_E NG_E
IY_I B_I D_I F_I G_I K_I SH_I L_I M_I N_I OW_I AA_I TH_I P_I OY_I R_I UH_I AE_I S_I T_I AH_I V_I W_I Y_I Z_I CH_I AO_I DH_I UW_I ZH_I EH_I AW_I AX_I EL_I AY_I EN_I HH_I ER_I IH_I JH_I EY_I NG_I
IY_S B_S D_S F_S G_S K_S SH_S L_S M_S N_S OW_S AA_S TH_S P_S OY_S R_S UH_S AE_S S_S T_S AH_S V_S W_S Y_S Z_S CH_S AO_S DH_S UW_S ZH_S EH_S AW_S AX_S EL_S AY_S EN_S HH_S ER_S IH_S JH_S EY_S NG_S
SIL SPN NSN LAU
SIL_B SPN_B NSN_B LAU_B
SIL_E SPN_E NSN_E LAU_E
SIL_I SPN_I NSN_I LAU_I
SIL_S SPN_S NSN_S LAU_S
```

You will observe that a question is simply a set of phones. The first four questions are asking about the word-position, for regular phones; and the last five do the same for the "silence phones". The "silence" phones also come in a variety without a suffix like _B, for example SIL. These may appear as optional silence in the lexicon, i.e. not inside an actual word. In setups with things like tone dependency or stress markings, extra_questions.txt may contain questions that relate to those features.

前面的4个问题(也就是4行)是对于普通的phone的位置的问题，比如第一行的问题用自然语言描述就是："请问这个phone是不是在词开始的位置？"

第五个问题是："这个phone是不是silence？" 而后面的问题是问silence phone的位置问题。

### word_boundary.txt

```
s5# head  data/lang/phones/word_boundary.txt
SIL nonword
SIL_B begin
SIL_E end
SIL_I internal
SIL_S singleton
SPN nonword
SPN_B begin
```

这个文件指明每个phone的"边界"信息。比如SIL是nonword，而SPN_B是位于词的开始。读者也许会说我们从名字_B就推测出它是词的开始，为什么还要多此一举搞个文件特别说明这一点呢？因为Kaldi是看不到这个字符串的，在WFST里把符号都变成了整数。我们需要这个文件是为了恢复出词的边界(比如lattice-align-words会读取这个文件的int版本——word_boundaray.int)。NIST sclite的打分是需要边界信息的。

### roots.txt

```
head data/lang/phones/roots.txt
shared split SIL SIL_B SIL_E SIL_I SIL_S
shared split SPN SPN_B SPN_E SPN_I SPN_S
shared split NSN NSN_B NSN_E NSN_I NSN_S
shared split LAU LAU_B LAU_E LAU_I LAU_S
...
shared split B_B B_E B_I B_S
```

这个文件说明怎么对决策树进行聚类。



在这里，我们暂时先忽略shared和split——这是构建决策树的特殊选项(更多信息请参考[How decision trees are used in Kaldi](http://kaldi-asr.org/doc/tree_externals.html))。这里我们我们重点关注一行里的那些phone，比如SIL SIL_B SIL_E SIL_I SIL_S，它是决策树的树根。也就是说每一行是一个决策树，不同行的phone是不可能聚类到一起的。对于有重音和带调的语言，我们通常把它们放到一行。我们通常把同一个音素的不同位置变体都放在一起。此外同一个HMM的三个不同状态(silence通常是五个状态)也放到一棵树的树根来聚类，当然Kaldi的问题会问它是第几个状态。这一点是和其它的系统不同的——在那里，比如一棵树的树根就是/iy/的中间(第二个)状态。而Kaldi里是把/iy/的三个状态放到一棵树下的，因此有可能b-iy+t的第一个状态和b-iy+d的第二个状态聚到一个叶子上，但是在其它系统是不可能的。


## 创建lang目录


前面我们看到data/lang下面有这么多文件，因此需要一个工具来自动的生成这些文件。
```
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
```

第一个参数是data/local/dict/，这是需要我们提提取准备好的目录，我们后面会讲，第二个参数指定OOV对应到哪个词，这里是<UNK>。我们前面的oov.txt就是根据这个参数生成的。第三个参数data/local/lang是临时的输出目录。第四个参数data/lang就是最终的输出。

而我们需要提取准备的目录就是data/local/dict/，我们来看一些这个目录里有些什么内容：
```
s5# ls data/local/dict
extra_questions.txt  lexicon.txt nonsilence_phones.txt  optional_silence.txt  silence_phones.txt
```
实际上这个目录还有一些文件，但它们都是临时文件，我们暂时忽略它们。我们来看一下这些文件的(部分)内容：
```
s5# head -3 data/local/dict/nonsilence_phones.txt
IY
B
D
s5# cat data/local/dict/silence_phones.txt
SIL
SPN
NSN
LAU
s5# cat data/local/dict/extra_questions.txt
s5# head -5 data/local/dict/lexicon.txt
!SIL SIL
-'S S
-'S Z
-'T K UH D EN T
-1K W AH N K EY
```

对于Switchboard recipe来说，这些文件非常简单。nonsilence_phones.txt里是"真实"的phone，而silence_phones.txt是silence的phone(那些position-position依赖的phone比如B_E都是prepare_lang.sh帮我们自动生成的)。extra_questions.txt是空的，表明我们提供没有额外的决策树聚类问题。


lexicon.txt是发音词典：
```
!SIL SIL
<SPOKEN_NOISE> SPN
<UNK> SPN
A EY1
A AH0
A''S EY1 Z
A'BODY EY1 B AA2 D IY0
A'COURT EY1 K AO2 R T
A'D EY1 D
A'GHA EY1 G AH0
```

它告诉我们每个词是由哪些phone组成，我们也可以提供一个lexiconp.txt。lexiconp.txt比lexicon.txt多一列，它的第二列是一个概率值。因为一个词可能有多种发音，但是它们的概率是不同的，所以我们可以指定概率。我们可以把lexicon.txt看成概率都是1.0。有些读者可能认为需要把概率归一化，也就是如果一个词有两个发音，那么这两个概率加起来是一。但是最佳实践并不是这样，最好的做法是让两个发音里最可能的值为1.0，而另外一个根据它和最可能的比例设置。因此这样两个值加起来是大于1的。

如果使用lexiconp.txt，可以参考egs/wsj/s5/run.sh，这个recipe是使用了lexiconp.txt的。

Switchboard的phone是没有重音和音调的，我们可以看一下 egs/wsj/s5/的情况：
```
s5# cat data/local/dict/silence_phones.txt
SIL
SPN
NSN
s5# head data/local/dict/nonsilence_phones.txt
S
UW UW0 UW1 UW2
T
N
K
Y
Z
AO AO0 AO1 AO2
AY AY0 AY1 AY2
SH
s5# head -6 data/local/dict/lexicon.txt
!SIL SIL
<SPOKEN_NOISE> SPN
<UNK> SPN
<NOISE> NSN
!EXCLAMATION-POINT  EH2 K S K L AH0 M EY1 SH AH0 N P OY2 N T
"CLOSE-QUOTE  K L OW1 Z K W OW1 T
s5# cat data/local/dict/extra_questions.txt
SIL SPN NSN
S UW T N K Y Z AO AY SH W NG EY B CH OY JH D ZH G UH F V ER AA IH M DH L AH P OW AW HH AE R TH IY EH
UW1 AO1 AY1 EY1 OY1 UH1 ER1 AA1 IH1 AH1 OW1 AW1 AE1 IY1 EH1
UW0 AO0 AY0 EY0 OY0 UH0 ER0 AA0 IH0 AH0 OW0 AW0 AE0 IY0 EH0
UW2 AO2 AY2 EY2 OY2 UH2 ER2 AA2 IH2 AH2 OW2 AW2 AE2 IY2 EH2
s5#
```
 
我们可以发现nonsilence_phones.txt的一行中有多个phone。它们是同一个元音的不同重音的变体。注意CMU词典的4个版本的重音：比如UW UW0 UW1 UW2，其中一个是没有数字后缀的。这些phone的顺序是无所谓的。我们建议一个"真实"phone的不同重音作为一行。

extra_questions.txt的第一个问题是问"当前phone是否是silence phone？"(事实上这个问题是不必要的，prepare_lang.sh会自动加上这个问题)，此外还有4个问题是问"当前phone的重音是否是1(0/2/null)？" 这4个问题是非常重要的。因为在nonsilence_phones.txt里"AY AY0 AY1 AY2"在一行，因此prepare_lang.sh生成的data/lang/phones/roots.txt和data/lang/phones/sets.txt会把它们放在一起作为决策树的树根来聚类。因此我们需要这4个问题来区分它们。注意：我们把同一个phone的4个重音变体放到一起的原因是如果分开的话，很多变体的训练数据会非常少，从而模型很不鲁棒。我们把它们放到一起，由数据来决定到底是要把它们聚类到同一个叶子节点还是分开。

utils/prepare_lang.sh脚本有很多参数，我们来看一下：
```
usage: utils/prepare_lang.sh <dict-src-dir> <oov-dict-entry> <tmp-dir> <lang-dir>
e.g.: utils/prepare_lang.sh data/local/dict <SPOKEN_NOISE> data/local/lang data/lang
options:
     --num-sil-states <number of states>             # default: 5, #states in silence models.
     --num-nonsil-states <number of states>          # default: 3, #states in non-silence models.
     --position-dependent-phones (true|false)        # default: true; if true, use _B, _E, _S & _I
                                                     # markers on phones to indicate word-internal positions.
     --share-silence-phones (true|false)             # default: false; if true, share pdfs of
                                                     # all non-silence phones.
     --sil-prob <probability of silence>             # default: 0.5 [must have 0 < silprob < 1]
```

重要的选项是\-\-share-silence-phones，它默认是false。如果是true，那么所有silence的phone，包括静音、噪声等的pdf(GMM)是共享参数的，但是它们的HMM的跳转概率是不共享的。把它设置为true对于IARPA的BABEL项目的广东话非常有用，我们也不知道这是为什么。可能原因是这个数据标注对齐的不是很好。

另外一个选项是\-\-sil-prob，但是我们没有做个太多实验，所以给不了太多建议。

## 创建语言模型或者语法(Grammar)


前面的tutorial没有介绍怎么创建G.fst，这是语言模型或者语法的FST表示。事实上，在有些recipe里，我们会有许多测试的目录，里面实验不同的语言模型的效果。比如WSJ的例子：

```
s5# echo data/lang*
data/lang data/lang_test_bd_fg data/lang_test_bd_tg data/lang_test_bd_tgpr data/lang_test_bg \
 data/lang_test_bg_5k data/lang_test_tg data/lang_test_tg_5k data/lang_test_tgpr data/lang_test_tgpr_5k
```
 

使用语言模型创建G.fst和文法创建G.fst的过程是不同的。在RM recipe里，使用了bigram文法，它只运行某些词的pair。我们让这个文法同一个点的出边的和是1。local/rm_data_prep.sh有如下的内容：
```
local/make_rm_lm.pl $RMROOT/rm1_audio1/rm1/doc/wp_gram.txt  > $tmpdir/G.txt || exit 1;
```


local/make_rm_lm.pl这个脚本会创建FST格式的文法。
```
s5# head data/local/tmp/G.txt
0    1    ADD    ADD    5.19849703126583
0    2    AJAX+S    AJAX+S    5.19849703126583
0    3    APALACHICOLA+S    APALACHICOLA+S    5.19849703126583
```

local/rm_prepare_grammar.sh会把上面的G.txt变成G.fst，使用的命令为：
```
fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt --keep_isymbols=false \
    --keep_osymbols=false $tmpdir/G.txt > data/lang/G.fst
```
 
如果你像创建自己的文法，那么你可能需要做类似的事情。注意：上面的过程只能使用特定的(正则)文法，比如你不能使用上下文无关文法(CFG)，因为它超出FST(也是一种有限状态自动机)的表示能力。在某些WFST的框架(比如Mike Riley的下推Transducer)可能实现，但是在Kaldi里不行。

在问任何关于语言模型或者文法之前，请先阅读Joshua Goodman的[A Bit of Progress in Language Modeling](https://arxiv.org/abs/cs/0108005)；然后去www.openfst.org去学习FST的tutorial。utils/format_lm.sh可以把ARPA格式的语言模型转成OpenFst的格式，下面是这个脚本的用法：
```
Usage: utils/format_lm.sh <lang_dir> <arpa-LM> <lexicon> <out_dir>
E.g.: utils/format_lm.sh data/lang data/local/lm/foo.kn.gz data/local/dict/lexicon.txt data/lang_test
Convert ARPA-format language models to FSTs.
```
这个脚本的关键命令是：
```
gunzip -c $lm \
  | arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$out_dir/words.txt - $out_dir/G.fst
```
arpa2fst命令把ARPA格式语言模型转换成FST(其实是FSA，其边的输入和输出符号是一样的)。

语言模型比较好用的工具是SRILM，Kaldi的很多例子都使用了它。它的文档和特性都会丰富，唯一的问题就是它的版权。utils/format_lm_sri.sh脚本用于把SRILM的语言模型变成FST：
```
Usage: utils/format_lm_sri.sh [options] <lang-dir> <arpa-LM> <out-dir>
E.g.: utils/format_lm_sri.sh data/lang data/local/lm/foo.kn.gz data/lang_test
Converts ARPA-format language models to FSTs. Change the LM vocabulary using SRILM.
```


## 关于OOV词

下面是解释Kaldi怎么处理OOV的，放到这里是因为找不到更合适的地方了。


很多语言模型工具多会在训练的时候把某些词(低频词，或者你指定了词典，那么训练数据里词典之外的词就会成为OOV)变成特殊的\<unk>或者\<UNK>，这样预测的时候就能处理OOV。你可以打开arpa文件看一看，通常都是上面两个中的一个。


在训练的时候，如果你的data目录下的text文件(前面说的transcript)出现了words.txt之外的词，那么Kaldi会把这些词变成data/lang/oov.txt里的词；这通常会是\<unk>、\<UNK> ，也可能是\<SPOKEN_NOISE>。这个oov.txt是你在运行prepare_lang.sh的第二个参数。如果这个词的概率不是零(你可以去arpa文件里看看)，那么Kaldi可能会识别出这个词来(然并卵？不一定。识别不出词来有时候比识别错强，所谓知之为知之不知为不知)。如果出现这种情况，通常是你把prepare_lang.sh的第二个参数指定为\<unk>，因为前面说过，很多语言模型工具把OOV变成特殊的\<unk>。此外因为解码的输出都是lexicon.txt里的词(否则我们怎么知道这个词是由哪些phone组成从而解码出来？)，因此你肯定也在lexicon.txt(或者lexiconp.txt)里定义了\<unk>。如果Kaldi(lexicon.txt)的未知词和语言模型的未知词不匹配，那么你不可能解码得到这个词。即使有可能，它的出现概率也是微乎及微的(否则说明模型有问题，比如OOV的LM概率太大)，因此它对WER没有太多影响。

当然把所有OOV都用一个phone来表示并不是最好的方案。在某些Kaldi recipe里比如tedlium/s5_r2/local/run_unk_model.sh，这些脚本会把\<unk>的phone用一个phone-level的语言模型来表示。注意：OOV是一个"高级话题"，建议语音识别的初学者暂时不用考虑它。
