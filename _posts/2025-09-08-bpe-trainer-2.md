---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第2部分：优化算法" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第三篇，优化之前的算法。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 算法优化

### 1.1 当前算法的瓶颈分析

上文介绍了最简单的算法实现，通过测试，在tinystory数据上总的训练时间是2000多秒。而我们的算法可以分成两步：第一步是切分文本并且统计词频，这主要有方法_pretokenize_and_count来实现；第二步是循环更新，具体的每一步包括统计pair的频率，找到频率最大的pair加入vocabulary和merges，然后更新word_encodings。实际跑下来，第一步的时间600多秒，第二步1400多秒。而第二步的时间又分为三个部分：统计pair的频率900多秒；找频率最大的pair为160多秒；更新450秒左右。

从上面的分析我们可以发现最应该优化的是统计pair的频率和更新word_encodings这两个部分。


### 1.2 优化思路

为了找到优化的方法，我们来看一下目前算法的两次循环过程。

假设第一步的统计结果为：

```
{low: 5, lower: 2, widest: 3, newest: 6, es: 2, st: 2}
```

#### 第一次循环

**循环的第一步**

统计pair的频率，这个需要遍历所有word，通过word_encodings找到这个pair的tokens，然后遍历tokens的相邻pair并统计数量：

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 11, st: 11, ne: 6, ew: 6}
```

**循环的第二步**

找到频率最大的pair是st和es，频率都是11，因为('s', 't') > ('e', 's') ，我们选择st。并且把st加到vocabulary和merges里。

**循环的第三步**

由于st合并成一个新的token，我们需要遍历word_encodings里的所有word，如果这个word的tokens里包含st这个pair，那么就需要更新它。

```
word_encodings['widest'] = ['w','i','d','e','st']
word_encodings['newest'] = ['n','e','s','e','st']
word_encodings['st'] = ['st']
```

#### 第二次循环

**循环的第一步**

统计pair的频率，这个需要遍历所有word，通过word_encodings找到这个pair的tokens，然后遍历tokens的相邻pair并统计数量：

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 2, st: 2, ne: 6, ew: 6, est: 9}
```
我们把它和第一次的对比一下：

```
第一次：{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 11, st: 11, ne: 6, ew: 6}
第二次：{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es:  2, st:  2, ne: 6, ew: 6, est: 9}
```

我们可以发现大部分都是相同的，不同点在于：第一次里统计的pair es和st变小了，因为st被合并了；第二次的统计新增了est，这是因为st合并后增加的新pair。

**循环的第二步**

找到频率最大的pair是('e', 'st')。我们把它合并并添加到vocabulary和merges里。

**循环的第三步**

由于('e', 'st')合并成一个新的token，我们需要遍历word_encodings里的所有word，如果这个word的tokens里包含('e', 'st')这个pair，那么就需要更新它。

```
word_encodings['widest'] = ['w','i','d','est']
word_encodings['newest'] = ['n','e','s','est']
```

### 1.3 优化方法

从上面的分析我们可以发现，不管是pair的频率还是word_encodings，其实大部分word是不受影响的。如果我们知道一个pair合并了，哪些word会受到影响，那么我们就可以只对这些受到影响的pair频率和word_encodings进行更新，这样就能极大的降低算法复杂度。现在的问题是：有一个pair合并了，哪些word包含这个pair呢？很容易想到的就是我们需要建立一个倒排索引，根据pair找到包含pair的word。


我们可以用一个dict pair_to_words来实现这个倒排索引，这个dict的key就是pair，而value是一个set，这个set里存放的是包含pair的所有word。根据pair_to_words，我们可以快速的找到所有受影响的word，然后增量更新这个word里的pair频率和word_encodings，当然pair更新后对应的倒排索引pair_to_words也要相应的更新。

我们具体来看一下这三步怎么增量更新。


为了获得新的pair的频率，我们首先根据第一次合并的是('s', 't')，我们根据pair_to_words找到包含这个pair的word有：

```
'widest'
'newest'
```

我们首先把这两个词里的pair都先减掉，也就是减去：

```
{wi: 3, id: 3, de: 3, es: 3, st: 3}
{ne: 6, ew: 6, we: 6, es: 6, st: 6}
```

这样得到：

```
{lo: 7, ow: 7, we: 2, er: 2, wi: 0, id: 0, de: 0, es: 2, st: 2, ne: 0, ew: 0}
```


接着这两个词的tokens变成了：

```
word_encodings['widest'] = ['w','i','d','e','st']
word_encodings['newest'] = ['n','e','s','e','st']
```

因此我们根据新的word_encodings计算新的pair的频次：

```
{wi: 3, id: 3, de: 3, est: 3}
{ne: 6, ew: 6, we: 6, est: 6}
```

把这些频次加回去得到：

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 2, st: 2, ne: 6, ew: 6, est: 9}
```


注意：我们这里实现的算法相对比较简单，先得到word的老的tokens，把所有老的pair都减掉。然后得到word合并后的新tokens，然后再加回去。更复杂的算法可以diff合并前后tokens的差异，比如wi其实不受st合并的影响，那么就可以避免它们的频率被减掉又加回去。不过这会使得算法变得非常复杂，我们这里先快速实现简单的算法，后面会再回来研究是否有必要实现更复杂的算法。


## 2. 代码

源代码可以参考[bpe_v2.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v2.py)。


### 2.1 train

```
    def train(self, input_path, vocab_size, special_tokens, *args):
        word_counts = self._pretokenize_and_count(input_path, special_tokens)
        vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            vocabulary[N_BYTES + i] = token.encode('utf-8')
        size = N_BYTES + len(special_tokens)
        merges = []

        # initial word encodings are utf-8
        word_encodings = {}
        for word in word_counts:
            word_encodings[word] = list(word.encode('utf-8'))

        pair_strings = {}
        pair_to_words = defaultdict(set)
        pair_counts = BPE_Trainer._count_pairs(word_counts, word_encodings, pair_strings, vocabulary, pair_to_words)

        while size < vocab_size:
            BPE_Trainer._merge_a_pair(pair_counts, pair_strings, vocabulary,
                                   pair_to_words, word_counts, word_encodings,
                                   merges, size)
            size += 1
      
        
        return vocabulary, merges
```

我们看train方法和之前大体相似，只是有些小的改动。由于我们只修改了第二步，所以第一步分词和统计词频的函数_pretokenize_and_count完全没有变。初始化word_encodings为word的utf8也没有变化。发生变化的部分是增加了：

```
pair_to_words = defaultdict(set)
```

这个dict就是实现从pair到words的倒排索引。

另外比较大的变化就是BPE_Trainer._count_pairs只是在while循环前调用了一次，后面词频的统计变成了增量更新，所以就不需要了。而之前while循环里的三步：统计pair频率，求max频率的pair以及更新word_encodings都放到一个新的函数BPE_Trainer._merge_a_pair里面。

下面我们来看一下改变或者新增的代码。

### 2.2 BPE_Trainer._count_pairs


```
    @staticmethod    
    def _count_pairs(word_counts, word_encodings, pair_strings, vocabulary, pair_to_words):
        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(0, len(encoding) - 1):
                pair = encoding[i], encoding[i + 1]
                pair_counts[pair] += count
                if pair not in pair_strings:
                    pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])

                pair_to_words[pair].add(word)

        return pair_counts
```

这个函数和bpe_v1基本相同，但是增加了对于倒排索引pair_to_words的修改：

```
pair_to_words[pair].add(word)
```

这个语句的作用就是维护pair到word的倒排索引。因为使用了defaultdict(set)，所以代码非常简单。如果这个pair没有出现过，那么会为这个pair新建一个set，并且把word加到set里，否则把pair加到原来的set里。因为一个word可以重复出现一个pair，所以value使用set而不是list。


### 2.3 _merge_a_pair

```
    @staticmethod
    def _merge_a_pair(pair_counts, pair_strings, vocabulary, pair_to_words, 
                   word_counts, word_encodings, merges, size):
        merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
        merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

        vocabulary[size] = merge_bytes
        new_id = size


        affected_words = pair_to_words[merge_pair]
        
        # update affected words' counts
        BPE_Trainer._updated_affected_word_count(merge_pair, affected_words, word_encodings,
                                                    word_counts, pair_counts,
                                                    pair_to_words, new_id, pair_strings, vocabulary)

        merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))
```

这个函数首先用max找到频率最大的pair，并且更新vocabulary和merges，这个和之前没有区别。主要的变化就是：


```
        affected_words = pair_to_words[merge_pair]
        
        # update affected words' counts
        BPE_Trainer._updated_affected_word_count(merge_pair, affected_words, word_encodings,
                                                    word_counts, pair_counts,
                                                    pair_to_words, new_id, pair_strings, vocabulary)
```

首先通过pair_to_words找到受merge_pair影响的所有words，然后调用_updated_affected_word_count增量更新word_counts和word_encodings，当然也包括倒排索引pair_to_words自己的更新。

### 2.4 _updated_affected_word_count

这个函数是本次算法优化的关键，我们来详细分析一下它的代码。

```
    @staticmethod
    def _updated_affected_word_count(merge_pair, affected_words, word_encodings, 
                                     word_counts, pair_counts, pair_to_words, 
                                     new_id, pair_strings, vocabulary):
            # we may update/delete words when iterate it.
            affected_words = affected_words.copy()

            for word in affected_words:
                word_tokens = word_encodings[word]
                wc = word_counts[word]

                for i in range(len(word_tokens) - 1):
                    old_pair = (word_tokens[i], word_tokens[i + 1])
                    pair_counts[old_pair] -= wc
                    if pair_counts[old_pair] <= 0:
                        # we accounted for all occurrences of this pair
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair)
                    else:
                        pair_to_words[old_pair].discard(word)


                i = 0
                new_tokens = []                
 
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1

                word_encodings[word] = new_tokens

                for i in range(len(new_tokens) - 1):
                    new_pair = (new_tokens[i], new_tokens[i + 1])
                    
                    pair_counts[new_pair] += wc
                    pair_to_words[new_pair].add(word)
                    if new_pair not in pair_strings:
                        pair_strings[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])
```

#### 复制affected_words

这个函数的主要循环就是遍历affected_words，然后根据合并的merge_pair增量修改word_counts、word_encodings和pair_to_words。因为affected_words来自pair_to_words[merge_pair]，所以在这个循环过程中pair_to_words会被修改，这样会出现问题。为了避免这个问题，我们先把affected_words复制一份在遍历它。因为affected_words这个set里是str，str是不可变的(immutable)，所以我们只需要用affected_words.copy()进行浅层拷贝即可。

#### 减去老的pair的频次

循环里第一步就是把受影响的word的老的pair都减去：

```
            for word in affected_words:
                word_tokens = word_encodings[word]
                wc = word_counts[word]

                for i in range(len(word_tokens) - 1):
                    old_pair = (word_tokens[i], word_tokens[i + 1])
                    pair_counts[old_pair] -= wc
                    if pair_counts[old_pair] <= 0:
                        # we accounted for all occurrences of this pair
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair)
                    else:
                        pair_to_words[old_pair].discard(word)
```

代码比较容易读懂，首先是pair_counts减去old_pair的频率。如果减去之后频率为0(这里写<=0是防御编程，理论上不应该出现小于0的情况)，那么就可以把这个pair从pair_counts里删掉，并且因为pair都没有了，倒排索引pair_to_words对应的value(set)应该也空了，所以可以把old_pair从pair_to_words里删除。如果old_pair的频率大于0，说明还有别的word包含它，那么只需要更新倒排索引，通过set.discard删除当前word。

注意：因为后面的代码会把一些pair又加进来，一些old_pair可能会被从pair_counts里删除，但是后面的代码又加进来。pair_to_words也是类似。还有一种方法是即使pair_counts为0了也不从pair_counts和pair_to_words里删除。这样也不影响代码的正确性。因为找频率最大的pair不可能找到频率为0的pair(除非所有非零次pair都加到vocabulary里了，也就是训练数据的所有pair都加入词典了，这基本上是不可能的事情，除非训练数据很小而我们要求的vocab_size又非常大)。但是这会出现pair_counts的某些key的value为零，某些pair_to_words的value为空的set。也就是这两个dict里存在一些垃圾数据，虽然避免了删除又加入的重复操作，但是也会导致dict变大，从而影响性能。至于这两种方法哪个更快，可能需要做实际的测试。我这里就略过这个比较了，感兴趣的读者可以试一试。

#### 计算新的word_encodings 

```
                i = 0
                new_tokens = []                
 
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1

                word_encodings[word] = new_tokens
```    

这段代码和之前相同，遍历原来的tokens，如果两个相邻的pair正好是merge_pair，那么合并成新的token。

#### 增加新的pair的统计

```
                for i in range(len(new_tokens) - 1):
                    new_pair = (new_tokens[i], new_tokens[i + 1])
                    
                    pair_counts[new_pair] += wc
                    pair_to_words[new_pair].add(word)
                    if new_pair not in pair_strings:
                        pair_strings[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])
```            

使用新的tokens重新统计pair的频次pair_counts和更新倒排索引pair_to_words，另外和前面类似，为了便于max函数，我们提前把新的pair的string加到pair_strings里。

## 3. 测试

### 3.1 单元测试

修改tests/adapters.py，把：

```
from cs336_basics import bpe_v1 as bpe
```

改成：

```
from cs336_basics import bpe_v2 as bpe
```

```
$ python -m pytest tests
============================================================ test session starts ============================================================
platform linux -- Python 3.12.1, pytest-8.4.1, pluggy-1.6.0
rootdir: ......../assignment1-basics-bpe
configfile: pyproject.toml
collected 3 items                                                                                                                           

tests/test_train_bpe.py::test_train_bpe_speed test_train_bpe_speed: 0.2742959549941588
PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

============================================================= 3 passed in 2.55s =============================================================
```

不错！我们通过了所有的单元测试，test_train_bpe_speed只花了0.27s。

### 3.2 tinystory训练

为了统计时间，我实现了[bpe_v2_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v2_time.py)

测试脚本为：

```
#!/bin/bash

for i in {1..3}; do
  python cs336_basics/test_trainer.py bpe_v2_time test_tiny_story_v2_${i} -d data/TinyStoriesV2-GPT4-train.txt -v 10000 > ts_v2_${i}.log
done
```

其中一次的测试结果为：

```
args=Namespace(trainer='bpe_v2_time', out_dir='test_tiny_story_v2_1', vocab_size=10000, data_path='data/TinyStoriesV2-GPT4-train.txt')
unknown_args=[]
_pretokenize_and_count time: 639.0596351698041
merge 1000: 5.967368541285396
merge 2000: 14.613370969891548
merge 3000: 25.7123233769089
merge 4000: 37.652114510536194
merge 5000: 50.475251752883196
merge 6000: 64.10010877996683
merge 7000: 77.75855202972889
merge 8000: 92.08696329593658
merge 9000: 106.85121286474168
merge time: 118.051194190979
total train time: 757.40 seconds
```

总时间757秒，比之前的2187快了差不多3倍。如果去掉没有优化的_pretokenize_and_count，那么时间分别是1565s和118s，速度快了13倍！

### 3.3 openweb训练

openweb比tinystory大了很多，我们以后的主要目标就是优化它。测试脚本类似，只需要修改vocab_size和训练文本路径：

```
#!/bin/bash

for i in {1..3}; do
  python cs336_basics/test_trainer.py bpe_v2_time test_openweb_v2_${i} -d ./data/owt_train.txt -v 32000 > ow_v2_${i}.log
done
```

一次运行结果的日志：

```
args=Namespace(trainer='bpe_v2_time', out_dir='test_openweb_v2_1', vocab_size=32000, data_path='./data/owt_train.txt')
unknown_args=[]
_pretokenize_and_count time: 2870.591514652595
merge 1000: 441.14712638035417
merge 2000: 688.2387115247548
merge 3000: 1041.7238410953432
merge 4000: 1486.6778947636485
merge 5000: 2020.9212533198297
merge 6000: 2638.192097246647
merge 7000: 3348.191169278696
merge 8000: 4135.424562651664
merge 9000: 4933.713395448402
merge 10000: 5686.200956711546
merge 11000: 6502.99085848406
merge 12000: 7370.24878706038
merge 13000: 8295.524110181257
merge 14000: 9270.619642425328
merge 15000: 10304.609711656347
merge 16000: 11381.702118359506
merge 17000: 12507.251110650599
merge 18000: 13666.351791091263
merge 19000: 14872.83940590918
merge 20000: 16118.41623338312
merge 21000: 17392.536006359383
merge 22000: 18637.891400933266
merge 23000: 19909.48292515427
merge 24000: 21211.815278911963
merge 25000: 22551.14059007913
merge 26000: 23926.073162287474
merge 27000: 25335.07080058381
merge 28000: 26784.249298969284
merge 29000: 28266.83105928637
merge 30000: 29766.008558433503
merge 31000: 31291.858460282907
merge time: 32437.584414267913
total train time: 35358.17 seconds
```

总的训练时长是35358s，其中_pretokenize_and_count花了2870s，而merge花了32437s。

## 4. 结果



版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v1_time | tinystory |2187/2264/2247|622/642/628| count_pair:944/995/984 max:167/173/174/174 update:453/453/460 |
bpe_v2_time | tinystory |757/738/746|639/621/627| 118/117/118 |
bpe_v2_time | openweb |35358/34265/35687|2870/2949/2930| 32437/31264/32708 |


## 本系列全部文章

* [第0部分：简介](/2025/09/05/bpe-trainer-0/) 介绍bpe训练的基本算法和相关任务，并且介绍开发环境。
* [第1部分：最简单实现](/2025/09/07/bpe-trainer-1/) bpe训练最简单的实现。
* [第2部分：优化算法](/2025/09/08/bpe-trainer-2/) 实现pair_counts的增量更新。
* [第3部分：并行分词和统计词频](/2025/09/09/bpe-trainer-3/) 使用multiprocessing实现多进程并行算法。
* [第4部分：一次失败的并行优化](/2025/09/10/bpe-trainer-4/) 尝试用多进程并行计算max pair。
* [第5部分：用C++实现Merge算法](/2025/09/12/bpe-trainer-5/) 用C++实现和Python等价的merge算法，并且比较std::unordered_map的两种遍历方式。
* [第6部分：用OpenMP实现并行求最大](/2025/09/15/bpe-trainer-6/) 用OpenMP并行求pair_counts里最大pair。
* [第7部分：使用flat hashmap替代std::unordered_map](/2025/09/18/bpe-trainer-7/) 使用flat hashmap来替代std::unordered_map。
* [第8部分：实现细粒度更新](/2025/09/19/bpe-trainer-8/) 使用倒排索引实现pair_counts的细粒度更新算法。
* [第9部分：使用堆来寻找最大pair](/2025/09/21/bpe-trainer-9/) 使用堆来求最大pair，提升性能。
* [第10部分：使用cython和pypy来加速](/2025/09/24/bpe-trainer-10/) 使用cython和pypy来加速python代码。
* [第11部分：使用cython封装c++代码](/2025/09/25/bpe-trainer-11/) 使用cython封装c++代码。
