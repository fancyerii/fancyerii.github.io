---
layout:     post
title:      "timit例子"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 语音识别
    - Kaldi
---

本文介绍Kaldi的timit示例。

 <!--more-->
 
**目录**
* TOC
{:toc}

## 运行

修改cmd.sh

```
-export train_cmd="queue.pl --mem 4G"
-export decode_cmd="queue.pl --mem 4G"
+export train_cmd="run.pl --mem 4G"
+export decode_cmd="run.pl --mem 4G"
 # the use of cuda_cmd is deprecated, used only in 'nnet1',
-export cuda_cmd="queue.pl --gpu 1"
+export cuda_cmd="run.pl --gpu 1"
```

修改run.sh

```
 feats_nj=10
-train_nj=30
+train_nj=8
 decode_nj=5
 
 echo ============================================================================
@@ -36,8 +36,8 @@ echo "                Data & Lexicon & Language Preparation
 echo ============================================================================
 
 #timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT # @JHU
-timit=/mnt/matylda2/data/TIMIT/timit # @BUT
-
+#timit=/mnt/matylda2/data/TIMIT/timit # @BUT
+timit=/home/lili/databak/ldc/LDC/timit/TIMIT
```

## TIMIT数据集

TIMIT数据集是学术界常用的一个数据集，它除了标注每个句子(utterance)的文本，还包括细粒度的词和更细粒度的因子级别的标注，读者可以在[LDC](https://catalog.ldc.upenn.edu/LDC93S1)收费下载。


### 说话人统计

它共包含6300个录音文件，共630个人参与录制，每人录制10个。下表是630个人的地域和性别等统计信息：

```
   表1：  说话人的方言分布

      Dialect
      Region(dr)    #Male    #Female    Total
      ----------  --------- ---------  ----------
         1         31 (63%)  18 (27%)   49 (8%)
         2         71 (70%)  31 (30%)  102 (16%)
         3         79 (67%)  23 (23%)  102 (16%)
         4         69 (69%)  31 (31%)  100 (16%)
         5         62 (63%)  36 (37%)   98 (16%)
         6         30 (65%)  16 (35%)   46 (7%)
         7         74 (74%)  26 (26%)  100 (16%)
         8         22 (67%)  11 (33%)   33 (5%)
       ------     --------- ---------  ----------
         8        438 (70%) 192 (30%)  630 (100%)

方言区(Dialect Regions)分别是：
     dr1:  New England
     dr2:  Northern
     dr3:  North Midland
     dr4:  South Midland
     dr5:  Southern
     dr6:  New York City
     dr7:  Western
     dr8:  Army Brat (moved around)

```

可以看出，630人中男性占比70%，女性30%。区域的划分是指说话人童年时期(childhood)所在的区域，dr7是无法明确说话人方言的集合，dr8是在童年时期呆过多个区域的说话人。

### 录音脚本统计
给说话人的脚本(可以理解为稿子, prompts.doc)包含两个SRI设计的很考验方言的(dialect "shibboleth")句子(简称SA)、450个MIT设计的没有太多因子变化的(phonetically-compact)句子(简称SX)和1890个TI设计的因子变化较多的(phonetically-diverse)句子(简称SI)。两个SA句子因为方言差异很大，因此让630个人都录一遍。450个SX句子让630个人来录制，平均每个人录制5句，总共录制3150个文件，平均每个句子有7(3150/450)说过。1890个SI的句子也是630人来录制，每人录3个文件，因此这个1890个句子只有一个人说过。下表直观的总结了上面的描述：

```
    表2:  TIMIT语音材料

  Sentence Type   #Sentences   #Speakers   Total   #Sentences/Speaker
  -------------   ----------   ---------   -----   ------------------
  Dialect (SA)          2         630       1260           2
  Compact (SX)        450           7       3150           5
  Diverse (SI)       1890           1       1890           3
  -------------   ----------   ---------   -----    ----------------
  Total              2342                   6300          10

```

可以看成6300个录音文件，630个说话人，平均每个人录制10个文件。SI的每个句子只有一个人录制、SX的每个句子7个人录制、SA的每个句子630个人录制。

### 建议的数据集划分

数据集划分成训练集和测试集，划分的标准参考testset.doc文件。

#### 核心(Core)测试集

核心测试集数据包含24个说话人，他们来自8个dr，每个dr选择两男一女。

```
    表3:  核心测试集的24个说话人

     Dialect        Male      Female
     -------       ------     ------
        1        DAB0, WBT0    ELC0
        2        TAS1, WEW0    PAS0
        3        JMP0, LNT0    PKT0
        4        LLL0, TLS0    JLM0
        5        BPM0, KLT0    NLP0
        6        CMJ0, JDH0    MGD0
        7        GRT0, NJM0    DHC0
        8        JLN0, PAM0    MLD0
```

这24个人每人录制了5个SX的句子和3个SI的句子(显然不能用SA来测试，因为每人都说过这两个句子)，因此总共有192个测试数据。


#### 完整(Complete)测试集

对于核心测试集24个说话人说过的SX集合中的句子共120个(24 * 5 = 120)，这120个句子还被其他144个人说过，我们把其他人录制的8个句子(5SX+3SI)也加到测试集(这样保证测试的句子没有在训练集合里出现，从而更加合理)，这样就共有168个人的1344(168 * 8)个句子。剩下的4956(6300-1344)个句子就作为训练集。完整的信息如下表所示：

```
     表4:  完整测试集的方言分布

      Dialect    #Male   #Female   Total
      -------    -----   -------   -----
        1           7        4       11
        2          18        8       26
        3          23        3       26
        4          16       16       32
        5          17       11       28
        6           8        3       11
        7          15        8       23
        8           8        3       11
      -----      -----   -------   ------
      Total       112       56      168

```


#### 目录和文件结构

结构为/\<CORPUS>/\<USAGE>/\<DIALECT>/\<SEX>\<SPEAKER_ID>/\<SENTENCE_ID>.\<FILE_TYPE>

* CORPUS 固定的字符串"timit"

* USAGE train或者test

* DIALECT dr1...dr8

* SEX m(male)或者f(female)

* SPEAKER_ID 说话人ID，说话人的名字缩写再加上一个数字(避免重名)

* SENTENCE_ID 句子类型加编号，比如sx234表示SX集合的第234个句子

* FILE_TYPE 文件类型，wav、txt、wrd和phn中的一个，下面详细介绍

比如/timit/train/dr1/fcjf0/sa1.wav，我们可以知道它是训练集合中的数据，说话人来自dr1，fcjf0表示这个说话人是女性并且名字的缩写是cjf并且编号是0。sa1说明这是SA集合的第一个句子，文件类型是wav说明是原始的录音文件。

又比如/timit/test/dr5/mbpm0/sx407.phn，这是来自测试集合，说话人来自dr5，男性并且名字编码是bmp0。这是SX集合的第407个句子，phn是文件类型。

#### 文件类型

##### wav

原始的波形文件，采用SPHERE头的格式(我们不需要关心，kaldi有工具读取)，比如TRAIN/DR1/FCJF0/SA1.WAV：

```
NIST_1A
   1024
database_id -s5 TIMIT
database_version -s3 1.0
utterance_id -s8 cjf0_sa1
channel_count -i 1
sample_count -i 46797
sample_rate -i 16000
sample_min -i -2191
sample_max -i 2790
sample_n_bytes -i 2
sample_byte_format -s2 01
sample_sig_bits -i 16
end_head

....
下面是二进制的内容
```

我们可以看到这个wav文件的采样率是16KHz，共46797个采样点(大约2.9s)，每个样本点用16位表示，因此这是标准的PCM编码。

##### txt
句子(utterance)级别的标注，比如TRAIN/DR1/FCJF0/SA1.TXT：
```
0 46797 She had your dark suit in greasy wash water all year.

```
0表示开始的采样点，46797(不包含)表示结束的采样点。

##### wrd
词级别的标注，比如TRAIN/DR1/FCJF0/SA1.WRD：
```
3050 5723 she
5723 10337 had
9190 11517 your
11517 16334 dark
16334 21199 suit
21199 22560 in
22560 28064 greasy
28064 33360 wash
33754 37556 water
37556 40313 all
40313 44586 year
```
它的意思是第一个词"she"开始的样本点是3050(0.2s)，结束样本点(不包含)是5723。因此我们可以推测0.2s之前的录音都是静音(silence)。

##### phn

因子(phonetic)级别的标注，比如TRAIN/DR1/FCJF0/SA1.PHN：

```
0 3050 h#
3050 4559 sh
4559 5723 ix
5723 6642 hv
6642 8772 eh
8772 9190 dcl
9190 10337 jh
10337 11517 ih
11517 12500 dcl
12500 12640 d
12640 14714 ah
14714 15870 kcl
15870 16334 k
16334 18088 s

...省略...
```

可以把它和前面的wrd对比，我们发现"she"由"sh/ix"两个因子组成。


## local/timit_data_prep.sh

local下的脚本一般都是于具体某个任务(数据集)相关的脚本，因此不是通用的脚本。这个脚本主要输出data/local/data。和前面介绍的数据集划分方式不同，Kaldi的这个reciple里是这样划分的：核心测试集和的192个句子作为测试集，从完整测试集里挑出50个说话人，每人挑出8(5 SX + 3 SI)个句子作为开发集，462个训练集中的说话人的8个句子作为训练集。因此共有3696个训练数据、400个开放数据和192个测试数据，所有的SA句子都丢弃，另外完整测试集里的92(630-462-50-24)个说话人的数据都丢弃。

这个脚本的代码如下，我加上了详细的注释，即使不了解脚本的读者也至少应该大致了解它做了什么事情。

```

# 需要一个参数，传入TIMIT数据集的路径
if [ $# -ne 1 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi

# 输出和脚本都是相对当前路径，因此这个脚本一定要在s5目录下执行。
dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # 导入路径和环境变量。
# irstlm用于训练语音模型
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
# sph2pipe 读取sphere头的wav文件
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

# 检查conf/test_spk.list和conf/dev_spk.list是否存在
# test_spk.list包括24个核心测试集合里的说话人ID
# dev_spk.list包括用于dev的50个说话人
[ -f $conf/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";

# 检查train和test目录是否存在(可以是大写或者小写)
if [ ! -d $*/TRAIN -o ! -d $*/TEST ] && [ ! -d $*/train -o ! -d $*/test ]; then
  echo "timit_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to TIMIT directory"
  echo "with name like /export/corpora5/LDC/LDC93S1/timit/TIMIT"
  exit 1;
fi

# 判断目录名字是大写还是小写
uppercased=false
train_dir=train
test_dir=test
if [ -d $*/TRAIN ]; then
  uppercased=true
  train_dir=TRAIN
  test_dir=TEST
fi

# 临时目录为/tmp/kaldi.XXXX
tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
# 脚本退出的时候删除临时目录
# 如果想看中间结果的话可以注释掉下面这行。
trap 'rm -rf "$tmpdir"' EXIT

# 把test_spk.list里的24个核心说话人复制到$tmpdir/test_spk
# dev_spk.list里的50个说话人复制到$tmpdir/test_spk
# 把TRAIN目录里出现的所有说话人ID复制到$tmpdir/train_spk
if $uppercased; then
  tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$*"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
else
  tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:upper:]' '[:lower:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$*"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
fi

cd $dir
for x in train dev test; do
  # 首先，找到所有的si和sx的wav文件，然后挑出每个集合对应说话人的wav文件，写到[train|dev|test]_sph.flist里

  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $tmpdir/${x}_spk > ${x}_sph.flist

  sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' ${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
  paste $tmpdir/${x}_sph.uttids ${x}_sph.flist \
    | sort -k1,1 > ${x}_sph.scp

  cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids

  # 生成phn.trans，这是没有归一化的脚本
  # Get the transcripts: each line of the output contains an utterance
  # ID followed by the transcript.
  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.PHN' \
    | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist
  sed -e 's:.*/\(.*\)/\(.*\).PHN$:\1_\2:i' $tmpdir/${x}_phn.flist \
    > $tmpdir/${x}_phn.uttids
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
  done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
  paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
    | sort -k1,1 > ${x}.trans

  # 使用timit_norm_trans.pl把因子归一化
  cat ${x}.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map -to 48 | sort > $x.text || exit 1;

  # 通过_sph.scp生成_wav.scp
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp

  # 根据uttids生成utt2spk和spk2utt文件
  cut -f1 -d'_'  $x.uttids | paste -d' ' $x.uttids - > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;

  # 生成spk2gender
  cat $x.spk2utt | awk '{print $1}' | perl -ane 'chop; m:^.:; $g = lc($&); print "$_ $g\n";' > $x.spk2gender

  # 为sclite准备STM文件
  wav-to-duration --read-entire-file=true scp:${x}_wav.scp ark,t:${x}_dur.ark || exit 1
  awk -v dur=${x}_dur.ark \
  'BEGIN{
     while(getline < dur) { durH[$1]=$2; }
     print ";; LABEL \"O\" \"Overall\" \"Overall\"";
     print ";; LABEL \"F\" \"Female\" \"Female speakers\"";
     print ";; LABEL \"M\" \"Male\" \"Male speakers\"";
   }
   { wav=$1; spk=wav; sub(/_.*/,"",spk); $1=""; ref=$0;
     gender=(substr(spk,0,1) == "f" ? "F" : "M");
     printf("%s 1 %s 0.0 %f <O,%s> %s\n", wav, spk, durH[wav], gender, ref);
   }
  ' ${x}.text >${x}.stm || exit 1

  # 为sclite创建空的GLM文件:
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > ${x}.glm
done

echo "Data preparation succeeded"

```

上面的脚本最终会产生data/local/data目录里的一些重要文件。

* [train\|dev\|test]_wav.scp

这个文件包含录音文件的信息，第一列是utterance-id，第二列是录音文件的位置(或者处理的脚本)，具体参考[数据准备](http://kaldi-asr.org/doc/data_prep.html)。

```
$ head -2 train_wav.scp 
FAEM0_SI1392 /home/lili/codes/kaldi/egs/timit/s5/../../../tools/sph2pipe_v2.5/sph2pipe -f wav /home/lili/databak/ldc/LDC/timit/TIMIT/TRAIN/DR2/FAEM0/SI1392.WAV |
FAEM0_SI2022 /home/lili/codes/kaldi/egs/timit/s5/../../../tools/sph2pipe_v2.5/sph2pipe -f wav /home/lili/databak/ldc/LDC/timit/TIMIT/TRAIN/DR2/FAEM0/SI2022.WAV |
```

* [train\|dev\|test]_sph.scp

原始的WAV文件。

```
$ head -2 train_sph.scp
FAEM0_SI1392	/home/lili/databak/ldc/LDC/timit/TIMIT/TRAIN/DR2/FAEM0/SI1392.WAV
FAEM0_SI2022	/home/lili/databak/ldc/LDC/timit/TIMIT/TRAIN/DR2/FAEM0/SI2022.WAV
```

* [train\|dev\|test].text

录音对应的转录文字(transcripts)，注意，这里我们训练的因子级别的模型，因此这里没有词(word)的概念。我们可以把因子比如/ax/理解为词。

```
$ head -2 train.text 
FAEM0_SI1392 sil ax s uw m f ao r ix vcl z ae m cl p uh l ax s ix cl ch uw ey sh en w eh er f aa r m hh eh z ax cl p ae cl k iy ng sh eh vcl d sil ae n vcl d f iy l vcl s sil
FAEM0_SI2022 sil w ah dx aw f ix cl d uh sh iy vcl d r ay v f ao sil
```

* [train\|dev\|test].utt2spk
每个utterance-id对应的说话人id

```
$ head -2 train.utt2spk 
FAEM0_SI1392 FAEM0
FAEM0_SI2022 FAEM0
```

* [train\|dev\|test].spk2utt
每个说话人说的所以话(utterances)

```
head -2 train.spk2utt 
FAEM0 FAEM0_SI1392 FAEM0_SI2022 FAEM0_SI762 FAEM0_SX132 FAEM0_SX222 FAEM0_SX312 FAEM0_SX402 FAEM0_SX42
FAJW0 FAJW0_SI1263 FAJW0_SI1893 FAJW0_SI633 FAJW0_SX183 FAJW0_SX273 FAJW0_SX3 FAJW0_SX363 FAJW0_SX93
```

* [train\|dev\|test].spk2gender
说话人的性别

```
head -2 train.spk2gender 
FAEM0 f
FAJW0 f

```

* [train\|dev\|test].stm

stm格式，给sclite工具使用，我们暂时不用管它

* [train\|dev\|test].glm

glm格式，给sclite工具使用，我们暂时不用管它



## local/timit_prepare_dict.sh

这个脚本的输入上面的输出data/local/data，输出是data/local/dict。

```
# 相对s5所在目录运行脚本
srcdir=data/local/data
dir=data/local/dict
lmdir=data/local/nist_lm
tmpdir=data/local/lm_tmp

mkdir -p $dir $lmdir $tmpdir

[ -f path.sh ] && . ./path.sh

#(1) 准备词典(Dictionary):

# 创建因子符号表(phones symbol-table)，增加silence和噪声(都用SIL表示)

# silence，我们这里用SIL表示silence、可选的silence和噪声
echo sil > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# 非silence(nonsilence)因子，每行一个，代表实际的音素。

# 从data/local/data/train.text里找出所有的因子，去重排序后输出到phones.txt
cut -d' ' -f2- $srcdir/train.text | tr ' ' '\n' | sort -u > $dir/phones.txt
# lexicon.txt就是phones.txt，因为这里我们认为词(word)就是因子，所以这个文件很简单。
paste $dir/phones.txt $dir/phones.txt > $dir/lexicon.txt || exit 1;
# 把phones.txt里的非silence phone输出到nonsilence_phones.txt
grep -v -F -f $dir/silence_phones.txt $dir/phones.txt > $dir/nonsilence_phones.txt 

# 生成用于聚类的问题，把属于同一个音素的都放到一起，但是这里比较简单，没有重音等
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

# (2) 创建phone的bigram语言模型
if [ -z $IRSTLM ] ; then
  export IRSTLM=$KALDI_ROOT/tools/irstlm/
fi
export PATH=${PATH}:$IRSTLM/bin
if ! command -v prune-lm >/dev/null 2>&1 ; then
  echo "$0: Error: the IRSTLM is not available or compiled" >&2
  echo "$0: Error: We used to install it by default, but." >&2
  echo "$0: Error: this is no longer the case." >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_irstlm.sh" >&2
  exit 1
fi

# 使用train.txt产生用于训练语言模型的lm_train.text，主要是在句子开头和结尾加上<s>和</s>
cut -d' ' -f2- $srcdir/train.text | sed -e 's:^:<s> :' -e 's:$: </s>:' \
  > $srcdir/lm_train.text

# 使用lm_train.text训练语言模型，保存的lm_phone_bg.ilm.gz
build-lm.sh -i $srcdir/lm_train.text -n 2 \
  -o $tmpdir/lm_phone_bg.ilm.gz

# 把它转换成arpa的格式并且gzip压缩，arpa便于转换成WFST
compile-lm $tmpdir/lm_phone_bg.ilm.gz -t=yes /dev/stdout | \
grep -v unk | gzip -c > $lmdir/lm_phone_bg.arpa.gz 

echo "Dictionary & language model preparation succeeded"
```

上面的脚本主要产生如下文件：

* silence_phones.txt

    这里只有sil

* optional_silence.txt

    这里只是sil

* phones.txt

    所有的因子，包括sil。

```
$ head phones.txt 
aa
ae
ah
ao
aw
ax
ay
b
ch
cl
```

* nonsilence_phones.txt

    去掉silence phone后的phones。

* extra_questions.txt

    聚类的问题。对于英语，通常把silence和noise放到一起(我们这里把两者都要sil表示了)，把相同重音的放到一起。

```
head extra_questions.txt
sil 
aa ae ah ao aw ax ay b ch cl d dh dx eh el en epi er ey f g hh ih ix iy jh k l m n ng ow oy p r s sh t th uh uw v vcl w y z zh 
```

我们看一下librispeech的这个文件：

```
$ head extra_questions.txt 
SIL SPN 
AA2 AE2 AH2 AO2 AW2 AY2 EH2 ER2 EY2 IH2 IY2 OW2 OY2 UH2 UW2 
AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH 
AA1 AE1 AH1 AO1 AW1 AY1 EH1 ER1 EY1 IH1 IY1 OW1 OY1 UH1 UW1 
AA0 AE0 AH0 AO0 AW0 AY0 EH0 ER0 EY0 IH0 IY0 OW0 OY0 UH0 UW0 

```

再看看thchs30的，它是把相同音调的放到一起作为问题：

```
$ head extra_questions.txt 
sil
a1 ai1 an1 ang1 ao1 e1 ei1 en1 eng1 i1 ia1 ian1 iang1 iao1 ie1 in1 ing1 iong1 iu1 ix1 iy1 o1 ong1 ou1 u1 ua1 uai1 uan1 uang1 ueng1 ui1 un1 uo1 v1 van1 ve1 vn1 
a2 ai2 an2 ang2 ao2 e2 ei2 en2 eng2 er2 i2 ia2 ian2 iang2 iao2 ie2 in2 ing2 iong2 iu2 ix2 iy2 o2 ong2 ou2 u2 ua2 uai2 uan2 uang2 ui2 un2 uo2 v2 van2 ve2 vn2 
a3 ai3 an3 ang3 ao3 e3 ei3 en3 eng3 er3 i3 ia3 ian3 iang3 iao3 ie3 in3 ing3 iong3 iu3 ix3 iy3 o3 ong3 ou3 u3 ua3 uai3 uan3 uang3 ueng3 ui3 un3 uo3 v3 van3 ve3 vn3 
a4 ai4 an4 ang4 ao4 e4 ei4 en4 eng4 er4 i4 ia4 ian4 iang4 iao4 ie4 in4 ing4 iong4 iu4 ix4 iy4 iz4 o4 ong4 ou4 u4 ua4 uai4 uan4 uang4 ueng4 ui4 un4 uo4 v4 van4 ve4 vn4 
a5 ai5 an5 ang5 ao5 e5 ei5 en5 eng5 er5 i5 ia5 ian5 iang5 iao5 ie5 in5 ing5 iong5 iu5 ix5 iy5 iz5 o5 ong5 ou5 u5 ua5 uai5 uan5 uang5 ueng5 ui5 un5 uo5 v5 van5 ve5 vn5 
aa b c ch d ee f g h ii j k l m n oo p q r s sh t uu vv x z zh

```

* lexicon.txt

    发音词典，我们这里比较简单，因为这里的词就是音素。

```
$ head lexicon.txt 
aa	aa
ae	ae
ah	ah
ao	ao
aw	aw
ax	ax
ay	ay
b	b
ch	ch
cl	cl
```

我们来看一下librispeech的：

```
$ head lexicon.txt 
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

thchs30的：

```
$ head lexicon.txt 
# sil
<SPOKEN_NOISE> sil
SIL sil
一 ii i1
一一 ii i1 ii i1
一丁点 ii i4 d ing1 d ian3
一万 ii i2 uu uan4
一万元 ii i2 uu uan4 vv van2
一万多 ii i2 uu uan4 d uo1
一下 ii i2 x ia4
```

## prepare_lang.sh

```
utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang
```

这个脚本的输入是data/local/dict，输出是data/lang，而data/local/lang_tmp是临时目录。

```

# 这个脚本用来准备data/lang。它要求输入目录包含：
# (1) lexicon.txt，这是发音词典，说明每个词由哪些因子组成，其格式为：
# word phone1 phone2 ... phoneN
# 或者lexiconp.txt，比前面的lexicon.txt多了一列概率，格式为：
# word pron-prob phone1 phone2 ... phoneN
# 其中0.0 < pron-prob <= 1.0。注：如果两个文件都存在，则优先使用lexiconp.txt。
# 
# (2) silence_phones.txt和nonsilence_phones.txt
# 它们分别包含silence和non-silence的phones。
# silence出来真正的"无声"之外还包括噪声(noise)、笑声(laugh)、咳嗽(cough)和filled pauses(呃，那个)等
# nonsilence phones包括"真正的" phones。
# 这些文件的每一行都是一些phones，通常它们都是某个一个基本phone(base phone)的不同重音或者音调(比如汉语)的变体。

# (3) optional_silence.txt
# 当然有的时候把所有的silence都用一个SIL来表示，也就是用一个HMM来建模所有的这些silence_phones。

# (4) extra_questions.txt
# 可以空；通常每行是同一个基本phone的变体，也可以包括silence。它用于增强Kaldi的自动生成的问题。
# 注：自动生成的问题会把同一基本phone的变体看成一样，因此不会在决策树里问这样的问题。

# 这个脚本也会增加位置相关的(word-position-dependent)phone。

# 默认参数
# silence默认5个状态
num_sil_states=5
# non-silence默认3个状态
num_nonsil_states=3
# 默认使用位置相关的phone(比如SIL_B、SIL_E等等)
position_dependent_phones=true
# 当位置相关的phone和word_boundary.txt是不同源生成的，则position_dependent_phones也是false


# 不同的silence是否共享pdf，默认否
share_silence_phones=false

sil_prob=0.5
unk_fst=        # 如果需要make_unk_lm.sh创建的因子级别(phone-level)的语言模型来处理未登录词(<oov-dict-entry>)
                # 则需要传入这个参数，比如<work-dir>/unk_fst.txt
                # 这里的<work-dir>是make_unk_lm.sh脚本的第二个参数

phone_symbol_table=              # 如果设置了，就使用这个特定的phones.txt
extra_word_disambig_syms=        # 如果设置了，从这个文件增加额外的消歧符号(每行一个)
                                 # 这些消歧符号会加到phones/disambig.txt, phones/wdisambig.txt 和 words.txt

num_extra_phone_disambig_syms=1 # 可选silence的标准因子消歧符号设置为1个
                                # 我们也可以增加这个值，但是只有我们想把它引入L_disambig.fst才有意义。




echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "usage: utils/prepare_lang.sh <dict-src-dir> <oov-dict-entry> <tmp-dir> <lang-dir>"
  echo "e.g.: utils/prepare_lang.sh data/local/dict <SPOKEN_NOISE> data/local/lang data/lang"
  echo "<dict-src-dir> should contain the following files:"
  echo " extra_questions.txt  lexicon.txt nonsilence_phones.txt  optional_silence.txt  silence_phones.txt"
  echo "See http://kaldi-asr.org/doc/data_prep.html#data_prep_lang_creating for more info."
  echo "options: "
  echo "     --num-sil-states <number of states>             # 默认: 5, silence的状态个数"
  echo "     --num-nonsil-states <number of states>          # 默认: 3, non-silence phone的状态个数"
  echo "     --position-dependent-phones (true|false)        # 默认: true; 如果true, 使用_B, _E, _S & _I"
  echo "                                                     # 来标识因子在词中的位置 "
  echo "     --share-silence-phones (true|false)             # 默认: false; 如果true, 所有的silence phone共享pdf"

  echo "     --sil-prob <probability of silence>             # 默认: 0.5 [must have 0 <= silprob < 1]"
  echo "     --phone-symbol-table <filename>                 # 默认: \"\"; 如果不空，使用这个文件 "
  echo "                                                     # 作为phone符号表，这在换了一个新的词典时有用"
  echo "     --unk-fst <text-fst>                            # 默认: none.  比如exp/make_unk_lm/unk_fst.txt."
  echo "                                                     # 如果你想使用因子级别的LM而不是特殊的UNK来建模未登录词时有用

  echo "                                                     # (它在测试的时候比较有用)."
  echo "     --extra-word-disambig-syms <filename>           # 默认: \"\"; 如果非空，把这个文件中的额外消歧符号加到"
  echo "                                                     # phones/disambig.txt,"
  echo "                                                     # phones/wdisambig.txt 和 words.txt中。"
  exit 1;
fi

srcdir=$1
oov_word=$2
tmpdir=$3
dir=$4
mkdir -p $dir $tmpdir $dir/phones

# 检查是否有lexiconp_silprob.txt
silprob=false
[ -f $srcdir/lexiconp_silprob.txt ] && silprob=true

[ -f path.sh ] && . ./path.sh

# 检查输入目录是否合法(包含前面描述的文件
! utils/validate_dict_dir.pl $srcdir && \
  echo "*Error validating directory $srcdir*" && exit 1;

# 如果lexicon.txt不存在，那么用lexiconp.txt来创建lexicon.txt(去掉概率)
if [[ ! -f $srcdir/lexicon.txt ]]; then
  echo "**Creating $srcdir/lexicon.txt from $srcdir/lexiconp.txt"
  perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' < $srcdir/lexiconp.txt > $srcdir/lexicon.txt || exit 1;
fi
# 如果lexiconp.txt不存在，那么用lexicon.txt来创建它(概率设置为1)
if [[ ! -f $srcdir/lexiconp.txt ]]; then
  echo "**Creating $srcdir/lexiconp.txt from $srcdir/lexicon.txt"
  perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $srcdir/lexicon.txt > $srcdir/lexiconp.txt || exit 1;
fi

# 如果unk_fst变量存在，那么必须是文件
if [ ! -z "$unk_fst" ] && [ ! -f "$unk_fst" ]; then
  echo "$0: expected --unk-fst $unk_fst to exist as a file"
  exit 1
fi

# 再次检查输入目录
if ! utils/validate_dict_dir.pl $srcdir >&/dev/null; then
  utils/validate_dict_dir.pl $srcdir  # show the output.
  echo "Validation failed (second time)"
  exit 1;
fi

# 如果提供了phones.txt，也需要进行sanity检查
if [[ ! -z $phone_symbol_table ]]; then
  # 如果position_dependent_phones是true，那么要检查提供的phones.txt里有_BIES这样的phone
  n1=`cat $phone_symbol_table | grep -v -E "^#[0-9]+$" | cut -d' ' -f1 | sort -u | wc -l`
  n2=`cat $phone_symbol_table | grep -v -E "^#[0-9]+$" | cut -d' ' -f1 | sed 's/_[BIES]$//g' | sort -u | wc -l`
  $position_dependent_phones && [ $n1 -eq $n2 ] &&\
    echo "$0: Position dependent phones requested, but not in provided phone symbols" && exit 1;
  ! $position_dependent_phones && [ $n1 -ne $n2 ] &&\
      echo "$0: Position dependent phones not requested, but appear in the provided phones.txt" && exit 1;

  # 检查silence_phones.txt和nonsilence_phones.txt里的phone都出现在提供的phones.txt里
  cat $srcdir/{,non}silence_phones.txt | awk -v f=$phone_symbol_table '
  BEGIN { while ((getline < f) > 0) { sub(/_[BEIS]$/, "", $1); phones[$1] = 1; }}
  { for (x = 1; x <= NF; ++x) { if (!($x in phones)) {
      print "Phone appears in the lexicon but not in the provided phones.txt: "$x; exit 1; }}}' || exit 1;
fi

# 如果还有额外的词级别的消歧符号，那么我们也需要检查这些符号是合法的
if [ ! -z "$extra_word_disambig_syms" ]; then
  if ! utils/lang/validate_disambig_sym_file.pl --allow-numeric "false" $extra_word_disambig_syms; then
    echo "$0: Validation of disambiguation file \"$extra_word_disambig_syms\" failed."
    exit 1;
  fi
fi

# 有位置信息的phone的if分支
if $position_dependent_phones; then
  # 从$srcdir/lexiconp.txt创建$tmpdir/lexiconp.txt(或者
  # 从$srcdir/lexiconp_silprob.txt创建$tmpdir/lexiconp_silprob.txt)
  # 需要根据phone在词中的位置添加_B, _E, _S, _I
  # 在这里，silence也会添加上述后缀
  if "$silprob"; then
    perl -ane '@A=split(" ",$_); $w = shift @A; $p = shift @A; $silword_p = shift @A;
              $wordsil_f = shift @A; $wordnonsil_f = shift @A; @A>0||die;
         if(@A==1) { print "$w $p $silword_p $wordsil_f $wordnonsil_f $A[0]_S\n"; }
         else { print "$w $p $silword_p $wordsil_f $wordnonsil_f $A[0]_B ";
         for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
                < $srcdir/lexiconp_silprob.txt > $tmpdir/lexiconp_silprob.txt
  else
    perl -ane '@A=split(" ",$_); $w = shift @A; $p = shift @A; @A>0||die;
         if(@A==1) { print "$w $p $A[0]_S\n"; } else { print "$w $p $A[0]_B ";
         for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
         < $srcdir/lexiconp.txt > $tmpdir/lexiconp.txt || exit 1;
  fi

  # 创建$tmpdir/phone_map.txt
  # 它的格式为：
  # <original phone> <version 1 of original phone> <version 2> ...
  # 例如：
  # AA AA_B AA_E AA_I AA_S
  # B代表开始(Begin), E代表结束(End), I代表中间(Internal)，S代表单个phone的词(Singleton)
  # 对于SIL：
  # SIL SIL SIL_B SIL_E SIL_I SIL_S
  # [因为SIL本身也是一个变体]

  cat <(set -f; for x in `cat $srcdir/silence_phones.txt`; do for y in "" "" "_B" "_E" "_I" "_S"; do echo -n "$x$y "; done; echo; done) \
    <(set -f; for x in `cat $srcdir/nonsilence_phones.txt`; do for y in "" "_B" "_E" "_I" "_S"; do echo -n "$x$y "; done; echo; done) \
    > $tmpdir/phone_map.txt
else
  # 如果不是位置相关的phone，那么只是简单的复制lexiconp.txt
  if "$silprob"; then
    cp $srcdir/lexiconp_silprob.txt $tmpdir/lexiconp_silprob.txt
  else
    cp $srcdir/lexiconp.txt $tmpdir/lexiconp.txt
  fi

  cat $srcdir/silence_phones.txt $srcdir/nonsilence_phones.txt | \
    awk '{for(n=1;n<=NF;n++) print $n; }' > $tmpdir/phones
  # phone_map.txt也是自己到自己的映射 
  paste -d' ' $tmpdir/phones $tmpdir/phones > $tmpdir/phone_map.txt
fi

mkdir -p $dir/phones  创建phones子目录

# 用于聚类的phone集合，创建monophone。

# 构建roots.txt文件
if $share_silence_phones; then
  # 如果share_silence_phones，则我们在roots.txt里把所有的silence phones都放到一行，并且让它们共享pdf。#
  # 因此这些phone的3个状态都是共享的，但是不同phone的跳转概率是不同的(不共享)。
  # 'shared'/'not-shared' 的意思是我们是否共享HMM(phone)的3个状态。 
  # split/not-split 表示我们是否需要进行聚类(分裂)。
  # 'not-shared not-split' 的意思是3个状态不共享pdf，也不分裂，因此这一行的多个(或者一个)phone就对应3个pdf。

  cat $srcdir/silence_phones.txt | awk '{printf("%s ", $0); } END{printf("\n");}' | cat - $srcdir/nonsilence_phones.txt | \
    utils/apply_map.pl $tmpdir/phone_map.txt > $dir/phones/sets.txt
  cat $dir/phones/sets.txt | \
    awk '{if(NR==1) print "not-shared", "not-split", $0; else print "shared", "split", $0;}' > $dir/phones/roots.txt
else
  # 不同的silence phones有不同的pdf(GMMs)。
  # [注: 在这里，"shared split"的意思是所有的状态只有一个GMM]
  cat $srcdir/{,non}silence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt > $dir/phones/sets.txt
  cat $dir/phones/sets.txt | awk '{print "shared", "split", $0;}' > $dir/phones/roots.txt
fi

cat $srcdir/silence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt | \
  awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/silence.txt
cat $srcdir/nonsilence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt | \
  awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/nonsilence.txt
cp $srcdir/optional_silence.txt $dir/phones/optional_silence.txt
cp $dir/phones/silence.txt $dir/phones/context_indep.txt

# extra_questions.txt空也可以
cat $srcdir/extra_questions.txt 2>/dev/null | utils/apply_map.pl $tmpdir/phone_map.txt \
  >$dir/phones/extra_questions.txt

# 如果position_dependent_phones，那么给phone加上不同的后缀。
# 这里silence和non-silence的区别是：silence需要把不带后缀的也作为一个变体(也就是suffix为空字符串)。
if $position_dependent_phones; then
  for suffix in _B _E _I _S; do
    (set -f; for x in `cat $srcdir/nonsilence_phones.txt`; do echo -n "$x$suffix "; done; echo) >>$dir/phones/extra_questions.txt
  done
  for suffix in "" _B _E _I _S; do
    (set -f; for x in `cat $srcdir/silence_phones.txt`; do echo -n "$x$suffix "; done; echo) >>$dir/phones/extra_questions.txt
  done
fi

# add_lex_disambig.pl is responsible for adding disambiguation symbols to
# the lexicon, for telling us how many disambiguation symbols it used,
# and and also for modifying the unknown-word's pronunciation (if the
# --unk-fst was provided) to the sequence "#1 #2 #3", and reserving those
# disambig symbols for that purpose.
# The #2 will later be replaced with the actual unk model.  The reason
# for the #1 and the #3 is for disambiguation and also to keep the
# FST compact.  If we didn't have the #1, we might have a different copy of
# the unk-model FST, or at least some of its arcs, for each start-state from
# which an <unk> transition comes (instead of per end-state, which is more compact);
# and adding the #3 prevents us from potentially having 2 copies of the unk-model
# FST due to the optional-silence [the last phone of any word gets 2 arcs].
if [ ! -z "$unk_fst" ]; then  # if the --unk-fst option was provided...
  if "$silprob"; then
    utils/lang/internal/modify_unk_pron.py $tmpdir/lexiconp_silprob.txt "$oov_word" || exit 1
  else
    utils/lang/internal/modify_unk_pron.py $tmpdir/lexiconp.txt "$oov_word" || exit 1
  fi
  unk_opt="--first-allowed-disambig 4"
else
  unk_opt=
fi

if "$silprob"; then
  ndisambig=$(utils/add_lex_disambig.pl $unk_opt --pron-probs --sil-probs $tmpdir/lexiconp_silprob.txt $tmpdir/lexiconp_silprob_disambig.txt)
else
  ndisambig=$(utils/add_lex_disambig.pl $unk_opt --pron-probs $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt)
fi
ndisambig=$[$ndisambig+$num_extra_phone_disambig_syms]; # add (at least) one disambig symbol for silence in lexicon FST.
echo $ndisambig > $tmpdir/lex_ndisambig

# Format of lexiconp_disambig.txt:
# !SIL	1.0   SIL_S
# <SPOKEN_NOISE>	1.0   SPN_S #1
# <UNK>	1.0  SPN_S #2
# <NOISE>	1.0  NSN_S
# !EXCLAMATION-POINT	1.0  EH2_B K_I S_I K_I L_I AH0_I M_I EY1_I SH_I AH0_I N_I P_I OY2_I N_I T_E

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$dir/phones/disambig.txt

# In case there are extra word-level disambiguation symbols they also
# need to be added to the list of phone-level disambiguation symbols.
if [ ! -z "$extra_word_disambig_syms" ]; then
  # We expect a file containing valid word-level disambiguation symbols.
  cat $extra_word_disambig_syms | awk '{ print $1 }' >> $dir/phones/disambig.txt
fi

# Create phone symbol table.
if [[ ! -z $phone_symbol_table ]]; then
  start_symbol=`grep \#0 $phone_symbol_table | awk '{print $2}'`
  echo "<eps>" | cat - $dir/phones/{silence,nonsilence}.txt | awk -v f=$phone_symbol_table '
  BEGIN { while ((getline < f) > 0) { phones[$1] = $2; }} { print $1" "phones[$1]; }' | sort -k2 -g |\
    cat - <(cat $dir/phones/disambig.txt | awk -v x=$start_symbol '{n=x+NR-1; print $1, n;}') > $dir/phones.txt
else
  echo "<eps>" | cat - $dir/phones/{silence,nonsilence,disambig}.txt | \
    awk '{n=NR-1; print $1, n;}' > $dir/phones.txt
fi

# Create a file that describes the word-boundary information for
# each phone.  5 categories.
if $position_dependent_phones; then
  cat $dir/phones/{silence,nonsilence}.txt | \
    awk '/_I$/{print $1, "internal"; next;} /_B$/{print $1, "begin"; next; }
         /_S$/{print $1, "singleton"; next;} /_E$/{print $1, "end"; next; }
         {print $1, "nonword";} ' > $dir/phones/word_boundary.txt
else
  # word_boundary.txt might have been generated by another source
  [ -f $srcdir/word_boundary.txt ] && cp $srcdir/word_boundary.txt $dir/phones/word_boundary.txt
fi

# Create word symbol table.
# <s> and </s> are only needed due to the need to rescore lattices with
# ConstArpaLm format language model. They do not normally appear in G.fst or
# L.fst.

if "$silprob"; then
  # remove the silprob
  cat $tmpdir/lexiconp_silprob.txt |\
    awk '{
      for(i=1; i<=NF; i++) {
        if(i!=3 && i!=4 && i!=5) printf("%s\t", $i); if(i==NF) print "";
      }
    }' > $tmpdir/lexiconp.txt
fi

cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > $dir/words.txt || exit 1;

# In case there are extra word-level disambiguation symbols they also
# need to be added to words.txt
if [ ! -z "$extra_word_disambig_syms" ]; then
  # Since words.txt already exists, we need to extract the current word count.
  word_count=`tail -n 1 $dir/words.txt | awk '{ print $2 }'`

  # We expect a file containing valid word-level disambiguation symbols.
  # The list of symbols is attached to the current words.txt (including
  # a numeric identifier for each symbol).
  cat $extra_word_disambig_syms | \
    awk -v WC=$word_count '{ printf("%s %d\n", $1, ++WC); }' >> $dir/words.txt || exit 1;
fi

# format of $dir/words.txt:
#<eps> 0
#!EXCLAMATION-POINT 1
#!SIL 2
#"CLOSE-QUOTE 3
#...

silphone=`cat $srcdir/optional_silence.txt` || exit 1;
[ -z "$silphone" ] && \
  ( echo "You have no optional-silence phone; it is required in the current scripts"
    echo "but you may use the option --sil-prob 0.0 to stop it being used." ) && \
   exit 1;

# create $dir/phones/align_lexicon.{txt,int}.
# This is the method we use for lattice word alignment if we are not
# using word-position-dependent phones.

# First remove pron-probs from the lexicon.
perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' <$tmpdir/lexiconp.txt >$tmpdir/align_lexicon.txt

# Note: here, $silphone will have no suffix e.g. _S because it occurs as optional-silence,
# and is not part of a word.
[ ! -z "$silphone" ] && echo "<eps> $silphone" >> $tmpdir/align_lexicon.txt

cat $tmpdir/align_lexicon.txt | \
 perl -ane '@A = split; print $A[0], " ", join(" ", @A), "\n";' | sort | uniq > $dir/phones/align_lexicon.txt

# create phones/align_lexicon.int
cat $dir/phones/align_lexicon.txt | utils/sym2int.pl -f 3- $dir/phones.txt | \
  utils/sym2int.pl -f 1-2 $dir/words.txt > $dir/phones/align_lexicon.int

# Create the basic L.fst without disambiguation symbols, for use
# in training.

if $silprob; then
  # Add silence probabilities (modlels the prob. of silence before and after each
  # word).  On some setups this helps a bit.  See utils/dict_dir_add_pronprobs.sh
  # and where it's called in the example scripts (run.sh).
  utils/make_lexicon_fst_silprob.pl $tmpdir/lexiconp_silprob.txt $srcdir/silprob.txt $silphone "<eps>" | \
     fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
     --keep_isymbols=false --keep_osymbols=false |   \
     fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;
else
  utils/make_lexicon_fst.pl --pron-probs $tmpdir/lexiconp.txt $sil_prob $silphone | \
    fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
    --keep_isymbols=false --keep_osymbols=false | \
     fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;
fi

# The file oov.txt contains a word that we will map any OOVs to during
# training.
echo "$oov_word" > $dir/oov.txt || exit 1;
cat $dir/oov.txt | utils/sym2int.pl $dir/words.txt >$dir/oov.int || exit 1;
# integer version of oov symbol, used in some scripts.


# the file wdisambig.txt contains a (line-by-line) list of the text-form of the
# disambiguation symbols that are used in the grammar and passed through by the
# lexicon.  At this stage it's hardcoded as '#0', but we're laying the groundwork
# for more generality (which probably would be added by another script).
# wdisambig_words.int contains the corresponding list interpreted by the
# symbol table words.txt, and wdisambig_phones.int contains the corresponding
# list interpreted by the symbol table phones.txt.
echo '#0' >$dir/phones/wdisambig.txt

# In case there are extra word-level disambiguation symbols they need
# to be added to the existing word-level disambiguation symbols file.
if [ ! -z "$extra_word_disambig_syms" ]; then
  # We expect a file containing valid word-level disambiguation symbols.
  # The regular expression for awk is just a paranoia filter (e.g. for empty lines).
  cat $extra_word_disambig_syms | awk '{ print $1 }' >> $dir/phones/wdisambig.txt
fi

utils/sym2int.pl $dir/phones.txt <$dir/phones/wdisambig.txt >$dir/phones/wdisambig_phones.int
utils/sym2int.pl $dir/words.txt <$dir/phones/wdisambig.txt >$dir/phones/wdisambig_words.int

# Create these lists of phones in colon-separated integer list form too,
# for purposes of being given to programs as command-line options.
for f in silence nonsilence optional_silence disambig context_indep; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt >$dir/phones/$f.int
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt | \
   awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $dir/phones/$f.csl || exit 1;
done

for x in sets extra_questions; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$x.txt > $dir/phones/$x.int || exit 1;
done

utils/sym2int.pl -f 3- $dir/phones.txt <$dir/phones/roots.txt \
   > $dir/phones/roots.int || exit 1;

if [ -f $dir/phones/word_boundary.txt ]; then
  utils/sym2int.pl -f 1 $dir/phones.txt <$dir/phones/word_boundary.txt \
    > $dir/phones/word_boundary.int || exit 1;
fi

silphonelist=`cat $dir/phones/silence.csl`
nonsilphonelist=`cat $dir/phones/nonsilence.csl`

# Note: it's OK, after generating the 'lang' directory, to overwrite the topo file
# with another one of your choice if the 'topo' file you want can't be generated by
# utils/gen_topo.pl.  We do this in the 'chain' recipes.  Of course, the 'topo' file
# should cover all the phones.  Try running utils/validate_lang.pl to check that
# everything is OK after modifying the topo file.
utils/gen_topo.pl $num_nonsil_states $num_sil_states $nonsilphonelist $silphonelist >$dir/topo


# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra step where we create a loop to "pass through" the
# disambiguation symbols from G.fst.

if $silprob; then
  utils/make_lexicon_fst_silprob.pl $tmpdir/lexiconp_silprob_disambig.txt $srcdir/silprob.txt $silphone '#'$ndisambig | \
     fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
     --keep_isymbols=false --keep_osymbols=false |   \
     fstaddselfloops  $dir/phones/wdisambig_phones.int $dir/phones/wdisambig_words.int | \
     fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;
else
  utils/make_lexicon_fst.pl --pron-probs $tmpdir/lexiconp_disambig.txt $sil_prob $silphone '#'$ndisambig | \
     fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
     --keep_isymbols=false --keep_osymbols=false |   \
     fstaddselfloops  $dir/phones/wdisambig_phones.int $dir/phones/wdisambig_words.int | \
     fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;
fi


if [ ! -z "$unk_fst" ]; then
  utils/lang/internal/apply_unk_lm.sh $unk_fst $dir || exit 1

  if ! $position_dependent_phones; then
    echo "$0: warning: you are using the --unk-lm option and setting --position-dependent-phones false."
    echo " ... this will make it impossible to properly work out the word boundaries after"
    echo " ... decoding; quite a few scripts will not work as a result, and many scoring scripts"
    echo " ... will die."
    sleep 4
  fi
fi

echo "$(basename $0): validating output directory"
! utils/validate_lang.pl $dir && echo "$(basename $0): error validating output" &&  exit 1;

exit 0;

```

关于roots.txt的 share/not-share，[这里](https://groups.google.com/forum/#!msg/kaldi-help/u3GA2zMKs2Y/vQpluWvEKAAJ)有解释：(这个文件)同一行Phones总是在一棵决策树上。share/not-share的区别在于相同的phone的不同HMM状态是否在同一棵决策树上(从而有可能聚类到一起，当然也有可能聚类后不在一个叶子节点上)。



