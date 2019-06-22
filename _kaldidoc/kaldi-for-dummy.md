---
layout:     post
title:      "Kaldi for Dummies教程"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Kaldi for Dummies tutorial](http://kaldi-asr.org/doc/kaldi_for_dummies.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}

## Introduction


这是一个面向小白用户的手把手的使用Kaldi和你自己的数据来构建一个简单的语音识别系统的教程。要是在我开始学习Kaldi时有这样的教程就好了。这是作为一个语音识别和脚本编程的非专业的人士的学习经验。如果你之前阅读过官方的文档而不知所云，那么这个教程可能适合你。你会学习怎么安装Kaldi，怎么使用你自己的数据来运行一个语音识别系统。你会得到第一个语音识别的结果。

首先，你得知道Kaldi是什么以及你可以用它来做什么。在我看来使用Kaldi需要专业的语音识别技术。同时还需要了解基本的脚本语言(bash, perl, python)。如果想阅读源代码的话，C++也是必须要了解的。

## 准备环境

首先一定要使用Linux。虽然理论上Kaldi可以在Windows上运行，但是为了避免踩坑，最好使用Linux。我这里使用的是Ubuntu 14.10(译者使用的是16.04)。安装好了Ubuntu之后需要安装下面的软件：


* atlas – 根据系统自动选择最优的线性代码库的库
* autoconf 
* automake 
* git 版本控制工具
* libtool
* svn 目前似乎不需要了
* wget 下载工具
* zlib 数据压缩工具


下面这些软件系统默认可能已经安装，但是如果则需要自己安装。


* awk 命令行文本处理工具
* bash
* grep
* make
* perl

## 下载

请参考[官方安装文档](http://kaldi-asr.org/doc/install.html)下载和安装。

## Kaldi目录结构


* egs 各种recipe，也就是例子，这是我们重点使用的目录
* misc 附加的工具
* src 源代码
* tools 外部的工具
* windows windows系统下的工具

## 示例项目
 
为了完成本教程，我们假设你已经有一个数据集。然后阅读的过程中参考本文的过程，并且根据你的数据做对应的修改。如果你完全没有任何录音数据或者完全follow我的教程，那么就自己录音吧，这样你会学习到更多的东西。Let's Go。

## 数据准备

### 录音


原文档要求录制100个wav文件，每个文件都是用英语说3个数字，比如one-two-three，然后保存成1_2_3.wav。原文建议：

* 10个不同的说话人
* 每人说10句
* 总共100句，每个人的10句放到以说话人命名的子目录下
* 总共300个词(0到9的数字)
* 每个句子3个词


因为我找不到那么多人来录音，为了简化，我自己录制了100个wav文件，说的是中文。其中使用正常的发音录制50个wav，然后粗着嗓子和尖嗓子各录了25个wav。
大家可以选择喜欢的录音软件来录音，我使用的是Audacity，录音是使用单声道16KHz采样，直接使用笔记本电脑的内置麦克风录制。


### 操作

创建kaldi/egs/digits目录，然后再在下面创建digits_audio目录，再在digits_audio下面创建train和test两个子目录。然后寻找一个说话人的录音作为测试数据，其他说话人的作为训练数据。我这里把高音的25个wav作为测试数据。因此我的目录结构为：


```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ tree digits_audio/
digits_audio/
├── test
│   └── high
│       ├── 173.wav
│       ├── 185.wav
│       ├── 194.wav
│       ├── 285.wav
│       ├── 290.wav
│       ├── 296.wav
│       ├── 361.wav
...................
...................
└── train
    ├── low
    │   ├── 104.wav
    │   ├── 175.wav
    │   ├── 194.wav
    │   ├── 215.wav
    │   ├── 225.wav
    │   ├── 259.wav
    │   ├── 286.wav
...................
................... 
    │   └── 966.wav
    └── normal
        ├── 110.wav
        ├── 125.wav
        ├── 138.wav
        ├── 143.wav
        ├── 155.wav
        ├── 160.wav
        ├── 165.wav
        ├── 201.wav
...................
...................
        ├── 908.wav
        └── 953.wav

```



### 声学数据

接下来我们要准备一些训练声学模型需要的数据，这些都是文本文件。每一行都是一个字符串，Kaldi通常要求它是排序的。我们可以用utils/validate_data_dir.sh来检查是否合乎要求，也可以使用utils/fix_data_dir.sh来尝试fix它们。注意：现在还不能运行这些脚本，下面的导入工具部分创建符号链接后才能使用。另外如果我们自己用命令行的工具排序的话一定要记得执行如下命令：
```
export LC_ALL=C
```

### 操作

在kaldi/egs/digits目录下创建data目录，然后在data下面创建train和test子目录。然后自己想办法(手工输入或者编写脚本程序)生成如下的文件。注意：train和test的结构是一样的。

#### spk2gender 

这个文件告诉Kaldi每个说话人的性别，每个说话人有一个speakerID。f表示female，m表示male。比如下面是原文的示例：
```
cristine f
dad m
josh m
july f
# and so on...
```

我这里把自己的三种嗓音作为3个不同说话人：

```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ cat data/train/spk2gender 
low	m
normal	m
```

```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ cat data/test/spk2gender 
high	m
```


#### wav.scp

This file connects every utterance (sentence said by one person during particular recording session) with an audio file related to this utterance. If you stick to my naming approach, 'utteranceID' is nothing more than 'speakerID' (speaker's folder name) glued with *.wav file name without '.wav' ending (look for examples below).

这个文件指定每个utterance(也就是一个人说的一个句子)对于的录音文件的路径。下面是原文示例：
```
dad_4_4_2 /home/{user}/kaldi/egs/digits/digits_audio/train/dad/4_4_2.wav
july_1_2_5 /home/{user}/kaldi/egs/digits/digits_audio/train/july/1_2_5.wav
july_6_8_3 /home/{user}/kaldi/egs/digits/digits_audio/train/july/6_8_3.wav
# and so on...
```

我的例子是：
```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ head data/train/wav.scp 
low_104 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/104.wav
low_175 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/175.wav
low_194 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/194.wav
low_215 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/215.wav
low_225 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/225.wav
low_259 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/259.wav
low_286 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/286.wav
low_296 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/296.wav
low_300 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/300.wav
low_319 /home/lili/codes/kaldi/egs/digits/digits_audio/train/low/319.wav
```

当然我们可以手动输入，但是这太费劲。我们可以编写一个简单的脚本来完成这个事情，后面会介绍。

#### text 

这个文件知道每个utterance的文本(词序列)。下面是原文示例：
```
dad_4_4_2 four four two
july_1_2_5 one two five
july_6_8_3 six eight three
# and so on...
```

这是我的例子：
```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ head data/train/text    
low_104 一 零 四
low_175 一 七 五
low_194 一 九 四
low_215 二 一 五
low_225 二 二 五
low_259 二 五 九
low_286 二 八 六
low_296 二 九 六
low_300 三 零 零
low_319 三 一 九

```

当然这个文件也是上面的脚本生成的。

#### utt2spk 

这个文件指定每个utterance的说话人ID(speakerID)，比如：
```
dad_4_4_2 dad
july_1_2_5 july
july_6_8_3 july
# and so on...
```

这是我的例子：
```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ head data/train/utt2spk 
low_104 low
low_175 low
low_194 low
low_215 low
low_225 low
low_259 low
low_286 low
low_296 low
low_300 low
low_319 low
```

#### corpus.txt
这个是训练语言模型用的，它的位置是在kaldi/egs/digits/data/local下，通常我们只用训练数据来生成这个文件。它的每行就是一个句子，比如：
```
one two five
six eight three
four four two
# and so on...
```

下面是我的例子：
```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ head data/local/corpus.txt 
三 六 九
二 三 零
三 一 三
一 四 三
五 六 九
五 四 二
七 二 二
四 六 四
七 一 三
一 三 八
```

#### 生成这些文件的脚本

下面是我用来生成这些文件(spk2gender必须自己手动输入，不过这个工作量不大)的Python脚本，文件名为prepkaldidata.py。
```
import argparse
from os import listdir
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare Kaldi data")
    parser.add_argument('-i', '--inputdir', help='input dir', required=True, default=None)
    parser.add_argument('-o', '--outputdir', help='output dir', required=True, default=None)
    parser.add_argument('-l', '--localdir', help='local dir', required=False, default=None)

    args = parser.parse_args()

    if args.inputdir is None or args.outputdir is None:
        RuntimeError("Must specify inputdir and outputdir")

    number2cn = {"1": "一", "2": "二", "3": "三", "4": "四", "5": "五",
                 "6": "六", "7": "七", "8": "八", "9": "九", "0": "零"}


    scp_lines = []
    text_lines = []
    utt2spk = []
    corpus = []
    for subdir in listdir(args.inputdir):
        for file in listdir(os.path.join(args.inputdir,subdir)):
            uid = "%s_%s" %(subdir, file[:-4])
            path = os.path.join(args.inputdir,subdir,file)
            cns = [number2cn[i] for i in file[:-4]]
            text = " ".join(cns)
            scp_lines.append("%s %s\n" % (uid, path))
            text_lines.append("%s %s\n" % (uid, text))
            utt2spk.append("%s %s\n" % (uid, subdir))
            if args.localdir:
                corpus.append("%s\n" % (text))

    scp_lines.sort()
    with open(os.path.join(args.outputdir, "wav.scp", ), "w", encoding="utf-8") as file:
        file.writelines(scp_lines)

    text_lines.sort()
    with open(os.path.join(args.outputdir, "text",), "w", encoding="utf-8") as file:
        file.writelines(text_lines)

    utt2spk.sort()
    with open(os.path.join(args.outputdir, "utt2spk", ), "w", encoding="utf-8") as file:
        file.writelines(utt2spk)

    if args.localdir:
        with open(os.path.join(args.localdir, "corpus.txt", ), "w", encoding="utf-8") as file:
            file.writelines(corpus)

```

然后分别两次运行上面的脚本来生训练和测试的数据：

```
python prepkaldidata.py -i /home/lili/codes/kaldi/egs/digits/digits_audio/train -o /home/lili/codes/kaldi/egs/digits/data/train -l /home/lili/codes/kaldi/egs/digits/data/local
python prepkaldidata.py -i /home/lili/codes/kaldi/egs/digits/digits_audio/test -o /home/lili/codes/kaldi/egs/digits/data/test
```

### 语言相关的数据

在kaldi/egs/digits/data/local创建dict子目录。

### 操作

#### lexicon.txt
发音词典，英文的话可以参考egs/voxforge或者其它发音词典，比如CMU发音词典。中文的话我参考的是egs/thchs30。我们的词汇非常少，所以手动输入。
```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ cat data/local/dict/lexicon.txt 
!SIL sil
<UNK> spn
一	ii i1
一	ii iao1
七	q i1
三	s an1
九	j iu3
二	ee er4
五	uu u3
八	b a1
六	l iu4
四	s iy4
零	l ing2
```

注意一有两个发音："衣"和"幺"，另外这个文件是排过序的。

#### nonsilence_phones.txt

lexicon.txt里的非静音的phone。
```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ cat data/local/dict/nonsilence_phones.txt 
a1
an1
b
ee
er4
i1
iao1
ii
ing2
iu3
iu4
iy4
j
l
q
s
u3
uu
```


#### silence_phones.txt

```
sil
spn
```

#### optional_silence.txt

```
sil
```

## 准备工具和配置

### 工具

我们可以从kaldi/egs/wsj/s5复制utils和steps目录到kaldi/egs/digits下，但是创建符号理解更加省空间。这两个目录就是我们需要的工具。

```
ln -s ../wsj/s5/utils utils
ln -s ../wsj/s5/steps steps
```

### 算分的脚本

去kaldi/egs/voxforge/s5/local把score.sh复制到local下。
 
### 安装SRILM语言模型工具
默认的Kaldi安装可能是不按照它的，这里需要用到。

```
cd kaldi/tools
./install_srilm.sh
```

### 配置文件

在kaldi/egs/digits下面创建conf目录。

#### decode.config

在conf下创建decode.config，然后输入如下内容：
```
first_beam=10.0
beam=13.0
lattice_beam=6.0
```

#### mfcc.conf

内容为：
```
--use-energy=false
```


## 训练脚本

这里的脚本是参考egs/voxforge，这里训练一个monophone的模型一个一趟的triphone模型。

### cmd.sh

```
# Setting local system jobs (local CPU - no external clusters)
export train_cmd=run.pl
export decode_cmd=run.pl
```

### path.sh
```
# Defining Kaldi root directory
export KALDI_ROOT=`pwd`/../..
# Setting paths to useful tools
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH
# Defining audio data directory (modify it for your installation directory!)
export DATA_ROOT="/home/{user}/kaldi/egs/digits/digits_audio"
# Enable SRILM
. $KALDI_ROOT/tools/env.sh
# Variable needed for proper data sorting
export LC_ALL=C
```

### run.sh

```
#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1
nj=1       # number of parallel jobs - 1 is perfect for such a small dataset
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar
# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }
# Removing previously created data (from last run.sh execution)
rm -rf exp mfcc data/train/spk2utt data/train/cmvn.scp data/train/feats.scp data/train/split1 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/local/lang data/lang data/local/tmp data/local/dict/lexiconp.txt
echo
echo "===== PREPARING ACOUSTIC DATA ====="
echo
# Needs to be prepared by hand (or using self written scripts):
#
# spk2gender  [<speaker-id> <gender>]
# wav.scp     [<uterranceID> <full_path_to_audio_file>]
# text        [<uterranceID> <text_transcription>]
# utt2spk     [<uterranceID> <speakerID>]
# corpus.txt  [<text_transcription>]
# Making spk2utt files
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
echo
echo "===== FEATURES EXTRACTION ====="
echo
# Making feats.scp files
mfccdir=mfcc
# Uncomment and modify arguments in scripts below if you have any problems with data sorting
# utils/validate_data_dir.sh data/train     # script for checking prepared data - here: for data/train directory
# utils/fix_data_dir.sh data/train          # tool for data proper sorting if needed - here: for data/train directory
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/test exp/make_mfcc/test $mfccdir
# Making cmvn.scp files
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir
echo
echo "===== PREPARING LANGUAGE DATA ====="
echo
# Needs to be prepared by hand (or using self written scripts):
#
# lexicon.txt           [<word> <phone 1> <phone 2> ...]
# nonsilence_phones.txt [<phone>]
# silence_phones.txt    [<phone>]
# optional_silence.txt  [<phone>]
# Preparing language data
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
echo
echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo
loc=`which ngram-count`;
if [ -z $loc ]; then
        if uname -a | grep 64 >/dev/null; then
                sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
        else
                        sdir=$KALDI_ROOT/tools/srilm/bin/i686
        fi
        if [ -f $sdir/ngram-count ]; then
                        echo "Using SRILM language modelling tool from $sdir"
                        export PATH=$PATH:$sdir
        else
                        echo "SRILM toolkit is probably not installed.
                                Instructions: tools/install_srilm.sh"
                        exit 1
        fi
fi
local=data/local
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
lang=data/lang
arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang/words.txt $local/tmp/lm.arpa $lang/G.fst
echo
echo "===== MONO TRAINING ====="
echo
steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono  || exit 1
echo
echo "===== MONO DECODING ====="
echo
utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode
echo
echo "===== MONO ALIGNMENT ====="
echo
steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1
echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo
steps/train_deltas.sh --cmd "$train_cmd" 2000 11000 data/train data/lang exp/mono_ali exp/tri1 || exit 1
echo
echo "===== TRI1 (first triphone pass) DECODING ====="
echo
utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode
echo
echo "===== run.sh script is finished ====="
echo
```

## 结果

在命令行运行./run.sh就可以了，结果在kaldi/egs/digits/exp可以看到。比如我的：

```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ cat exp/tri1/decode/wer_15
compute-wer --text --mode=present ark:exp/tri1/decode/scoring/test_filt.txt ark,p:- 
%WER 49.33 [ 37 / 75, 14 ins, 9 del, 14 sub ]
%SER 92.00 [ 23 / 25 ]
Scored 25 sentences, 0 not present in hyp.

```

词错误率挺高的，在49%，原因很多，其中之一肯定是数据不好。不过我们这里的目的是先把代码跑起来，以后我们会研究怎么改进效果。

我们也可以看测试数据上解码的结果：
```
lili@lili-Precision-7720:~/codes/kaldi/egs/digits$ head exp.bak/tri1/decode/log/decode.1.log 
# gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 --allow-partial=true --word-symbol-table=exp/tri1/graph/words.txt exp/tri1/final.mdl exp/tri1/graph/HCLG.fst "ark,s,cs:apply-cmvn  --utt2spk=ark:data/test/split1/1/utt2spk scp:data/test/split1/1/cmvn.scp scp:data/test/split1/1/feats.scp ark:- | add-deltas  ark:- ark:- |" "ark:|gzip -c > exp/tri1/decode/lat.1.gz" 
# Started at Tue Jun  4 21:12:11 CST 2019
#
gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 --allow-partial=true --word-symbol-table=exp/tri1/graph/words.txt exp/tri1/final.mdl exp/tri1/graph/HCLG.fst 'ark,s,cs:apply-cmvn  --utt2spk=ark:data/test/split1/1/utt2spk scp:data/test/split1/1/cmvn.scp scp:data/test/split1/1/feats.scp ark:- | add-deltas  ark:- ark:- |' 'ark:|gzip -c > exp/tri1/decode/lat.1.gz' 
add-deltas ark:- ark:- 
apply-cmvn --utt2spk=ark:data/test/split1/1/utt2spk scp:data/test/split1/1/cmvn.scp scp:data/test/split1/1/feats.scp ark:- 
high_173 一 七 一 三 
LOG (gmm-latgen-faster[5.4.232-532f3]:DecodeUtteranceLatticeFaster():decoder-wrappers.cc:286) Log-like per frame for utterance high_173 is -7.28081 over 227 frames.
high_185 一 三 五 
LOG (gmm-latgen-faster[5.4.232-532f3]:DecodeUtteranceLatticeFaster():decoder-wrappers.cc:286) Log-like per frame for utterance high_185 is -7.04538 over 134 frames.

```

## 总结
恭喜你，通过这么简单的几步就可以在自己的训练数据上训练一个简单的模型了，虽然你可能还不懂每一步什么意思，但是不要着急，后面的文档会有介绍，我们以后也会回来改进它的识别效果。

