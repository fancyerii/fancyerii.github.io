---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第1部分：最简单实现" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第二篇，实现一个最简单的算法。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 算法简介

本文实现最简单的BPE Tokenizer的训练算法，这个算法分为两步。第一步根据前文介绍的正则表达式把文档(document)切分成词(word)，并且统计词频。然后初始化vocabulary为0-255的字节，这样根据utf-8编码可以把每一个词切分成一个个字节，初始的时候每个字节就是一个token，这样可以得到每个词的编码(embedding)。接着就是根据词计算这些token组成的pair，找到频率最高的pair，把这个pair作为一个新的token加到vocabulary里，同时更新每个词新的编码。如果有两个或者多个pair的频率相同，那么就选择最大的那个pair。比如之前的例子：

```
(“A”, “B”), (“A”, “C”), (“B”, “ZZ”), (“BA”, “A”) 
```

上面4个pair的频率都一样，那么就需要比较大小。对于一个pair，我们首先比较第一个元素，如果第一个元素相同，我们再比较第二个元素。因此这4个pair的大小关系为：

```
(“BA”, “A”) > (“B”, “ZZ”) > (“A”, “C”) > (“A”, “B”)
```

比较单个元素时我们是把它看成bytes，因"BA"对应的bytes是[66, 65]，而"B"对应[66]，因此"BA" > "B"。

**注意**：我们不能把pair的两个字符串拼接起来再比较大小。比如("AB", "C")和("A", "BC")，如果拼接起来都是"ABC"，但是按照我们的规则，"AB" > "A"，所以"AB", "C") > ("A", "BC")。如果我们确定我们的字符串不会出现"\0"(null)，那么我们可以拼接的时候加一个"\0"。这样第一个pair变成"AB\0C"，第二个变成"A\0BC"，因为"\0"总是小于任何非"\0"，所以这是没有问题的。但是如果我们的字符串里可能出现"\0"，那就有问题了。比如("AB\0","C")和("AB","\0C")，如果按照拼接方法，两者是相同的，但是实际应该("AB\0","C") > ("AB","\0C")。虽然文本里出现"\0"的可能性不大，但这总是存在潜在的风险，除非我们对文本进行预处理去掉"\0"。


除此之外，我们还需要把原始的文本文件切分成文档，从而避免词或者token越过文档边界。这个过程通常取决于我们文本的来源。比如我们是抓取的网页，那么一个文档可能是一个网页(url)。如果我们的文本是一本书，那么可以把整本书当成一个文档，也可以按照章节切分为更细的单元。不管文档的来源如果，为了处理的效率，通常会把多个文档合并到一个文件里，否则文件太多了，大部分操作系统的文件系统无法高效的处理很多小文件。

所以我们通常假设把多个文档拼接成一个文本文件时会用一个特殊的字符串(比如`<|endoftext|>`)来区分它们。我们这个作业的训练数据data/owt_train.txt和data/TinyStoriesV2-GPT4-train.txt就是用这个特殊字符串来区分不同文档的。

## 2. 代码

源代码可以参考[bpe_v1.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v1.py)。

### 2.1 BPE_Trainer

为了统一接口，我们会定义一个类BPE_Trainer，这个类对外只有一个方法：

```
def train(self, input_path, vocab_size, special_tokens, *args)
```

这个方法有3个必选参数：

* input_path 用于训练的文本文件，假设用special_tokens切分文档
* vocab_size 输出的vocabulary大小，比如tinystory我们输出10000个词汇，而openweb我们输出32000
* special_tokens 用于切分文档的特殊字符串，我们的例子都是单个特殊字符串`<|endoftext|>`，注意这个输入类型是list

train的代码如下：

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

        while size < vocab_size:
            pair_counts = BPE_Trainer._count_pairs(word_counts, word_encodings, pair_strings, vocabulary)
            merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
            merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

            vocabulary[size] = merge_bytes
            new_id = size
            size += 1

            # update word_encodings
            for word, word_tokens in word_encodings.items():
                i = 0
                new_tokens = []                
                has_new_id = False
 
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        new_tokens.append(new_id)
                        i += 2
                        has_new_id = True
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1

                if has_new_id:
                    word_encodings[word] = new_tokens

            merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))
        return vocabulary, merges
```

代码大致分成两部分，第一部分是调用self._pretokenize_and_count实现文本切分成文档，文档再切分成word，然后再统计其频率。第二部分就是那个while循环，调用BPE_Trainer._count_pairs统计pair的频率，然后用max函数找到频率最高的加入到vocabulary和merges里。下面我们详细分析一下这两部分的代码。

#### _pretokenize_and_count

```
    def _pretokenize_and_count(self, input_path: str, special_tokens: list[str]):
        # pre-compile regex
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        word_counts = defaultdict(int)

        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            blocks = re.split(special_pattern, chunk)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    word_counts[text] += 1

        return word_counts
```

这段代码首先构造分词的正则表达式，这里使用的是[mrab-regex](https://github.com/mrabarnett/mrab-regex)，可以通过pip install regex安装。我们前面的环境打架部分已经通过uv安装好了这个依赖，只需要在开始import：

```
import regex as re
```

注意：如果使用Python标准库里的re库是无法编译这个正则表达式的。

然后构造切分文档的正则表达式，因为special_tokens可能多个，所以用'\|'。注意token里可能包含特殊字符(比如`<|endoftext|>`有\|)，所以要用re.escape函数来对它们进行转义。

最后就是调用BPE_Trainer._chunk_documents_streaming函数把文本切分成chunk(一个chunk包含多个文档)，然后用re.split切分成文档(block)，最后对每个block使用re.finditer进行分词，最后用defaultdict统计词频。

注意：re.split返回的是list，而re.finditer返回的是一个iterator。返回iterator会使用更少内存，但是速度可能会慢一点。split只能返回list，我们也可以用finditer来实现类似split的功能。我们甚至可以用Aho-Corasick(trie)算法来实现更高效的切分。不过根据优化的Amdahl's Law，我们在没有profile之前最好不要盲目的优化代码。



接着我们来看一下BPE_Trainer._chunk_documents_streaming，它是一个生成器函数。这里我参考了[这里的代码](https://github.com/Christine8888/cs336-assignment1-basics/blob/main/cs336_basics/bpe.py#L21)。

```
    @staticmethod
    def _chunk_documents_streaming(
        path: str,
        chunk_size: int = CHUNK_SIZE,
        special_token: str = "<|endoftext|>"
    ):
        """
        Reads 'path' in streaming fashion, yielding chunks of text that
        each end on a '<|endoftext|>' boundary.
        """

        leftover = ""
        token_len = len(special_token)

        with open(path, "r", encoding="utf-8") as f:
            while True:
                # read one chunk_size block of text
                block = f.read(chunk_size)
                if not block:
                    # no more data in file
                    break

                # combine leftover from previous iteration + new block
                block = leftover + block
                leftover = ""

                # find the *last* occurrence of the special token in 'block'
                last_eot_idx = block.rfind(special_token)

                if last_eot_idx == -1:
                    # no complete document in this chunk
                    # keep everything in leftover for the next read
                    leftover = block
                else:
                    # up through last_eot_idx is a complete set of docs
                    yield block[: last_eot_idx + token_len]
                    # keep everything after that boundary as leftover
                    leftover = block[last_eot_idx + token_len:]

        # yield leftover text
        if leftover:
            yield leftover
```

这段代码的思路是：从文件里读取大小为chunk_size(默认50K)的block，然后倒着回去(调用str的rfind函数)找special_token(`<|endoftext|>`)，然后返回`<|endoftext|>`之前(包括它本身)的文本，然后把它之后的文本暂存在leftover里，作为下一次读取的一部分。为了减少内存使用，这里通过yield实现生成器函数，而不是把所有结果一次性放到一个list。因为我们的文本可能非常大(比如openweb的训练数据有11GB)，都放内存会有问题。

### 2.2 train的其它代码

#### 初始化

```
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
```

在进入主要的while循环之前，我们需要构造初始化的vocabulary。vocabulary是一个dict，key是整数ID，value是这个token对应的utf8编码的bytes。此外每个词(word)的初始化编码(encoding)也是它们的utf8编码。

#### 主循环

```
        while size < vocab_size:
            pair_counts = BPE_Trainer._count_pairs(word_counts, word_encodings, pair_strings, vocabulary)
            merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
            merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

            vocabulary[size] = merge_bytes
            new_id = size
            size += 1

            # update word_encodings
            for word, word_tokens in word_encodings.items():
                i = 0
                new_tokens = []                
                has_new_id = False
 
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        new_tokens.append(new_id)
                        i += 2
                        has_new_id = True
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1

                if has_new_id:
                    word_encodings[word] = new_tokens

            merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))
```

第一步是调用BPE_Trainer._count_pairs统计所有的pair的频率。这个函数的代码如下：

```
    @staticmethod    
    def _count_pairs(word_counts, word_encodings, pair_strings, vocabulary):
        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(0, len(encoding) - 1):
                pair = encoding[i], encoding[i + 1]
                pair_counts[pair] += count
                if pair not in pair_strings:
                    pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])

        return pair_counts
```

这个函数处理统计pair = encoding[i], encoding[i + 1]的频率之外，还把这个pair的vocabulary存放到pair_strings里：

```
pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])
```

pair_strings的key是pair，value是一个tuple，对应的是pair[0]和pair[1]的vocabulary。我们把它存放到pair_strings的目的是为了下一步找pair时如果频率相同就可以找最大的那个pair。这就是下一个语句：

```
            merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
```

这里的关键是传给key的lambda函数，对于每一个pair/count，它生成一个tuple。tuple的第一个元素是pair的频率(x[1]，tuple的第二个元素是pair对应的string。这样max可以先比较频率，如果频率相同在比较string。如果没有pair_strings，则可以写成：

```
            merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], (vocabulary[x[0][0]], vocabulary[x[0][1]])))
```

因为pair_strings对于一个特定的key(pair)它的value是不变的，我们可以提前把它存放在pair_strings里。这样求max的时候就不用反复对vocabulary进行查找了。

找到需要合并的pair之后，就可以合并了：

```
            merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

            vocabulary[size] = merge_bytes
            new_id = size
            size += 1
            
            ...
            
            
            merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))
            
```

最后需要做是更新word_encodings，比如原来一个word的encoding是[1,2,1,1,2]，假设把1和2合并成了257，那么这个word的新的encoding就变成了[257,1,257]。代码为：

```
            for word, word_tokens in word_encodings.items():
                i = 0
                new_tokens = []                
                has_new_id = False
 
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        new_tokens.append(new_id)
                        i += 2
                        has_new_id = True
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1

                if has_new_id:
                    word_encodings[word] = new_tokens
```

这段代码从头到尾遍历原来的encoding(word_tokens)，如果两个相邻token可以合并，那么就用新的token替代，否则保持原来的。


## 3. 测试

### 3.1 单元测试

用pytest进行单元测试：
```
python -m pytest tests
```

结果为：

```
$ python -m pytest  tests
============================================================ test session starts ============================================================
platform linux -- Python 3.12.1, pytest-8.4.1, pluggy-1.6.0
rootdir: ..../..../assignment1-basics-bpe
configfile: pyproject.toml
collected 3 items                                                                                                                           

tests/test_train_bpe.py::test_train_bpe_speed test_train_bpe_speed: 4.37365837598918
FAILED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

三个测试有两个能通过，但是test_train_bpe_speed会超时。这说明我们的代码是正确的，但是太慢。有了正确的代码，我们下一步的目标就是优化速度。

不过优化速度能通过单元测试只是很小的目标，我们希望极限的优化性能，让它们能在更大的数据集tinystory和openweb上训练。因此我们会测试一下这个版本在tinystory上的时间，把它作为一个baseline。注意：由于这个版本太慢，我们无法在24小时内跑完openweb的训练。

### 3.2 tinystory训练

为了分析各个部分(统计词频和迭代合并)的时间，我实现了[bpe_v1_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v1_time.py)，它的代码和bpe_v1基本相同，只是在一些关键调用前后用time统计了时间。

另外为了比较不同版本的实现，我也实现了[test_trainer.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_trainer.py)。它可以根据命令行参数选择不同的版本和选项(比如cpu的个数等等)，完整代码就不介绍，感兴趣的读者可以自行阅读。它的用法比较简单，我把它写成了一个bash脚本可以直接运行：

```
#!/bin/bash

for i in {1..3}; do
  python cs336_basics/test_trainer.py  bpe_v1_time test_tiny_story_v1_${i} -d data/TinyStoriesV2-GPT4-train.txt -v 10000 > ts_v1_${i}.log
done

```

test_trainer.py的参数：

* 第一个参数是使用哪个版本的算法，比如这里是bpe_v1_time，
* 第二个是结果输出目录
* -d 可选参数指定训练文本，这里使用tinystory
* -v 指定输出vocabulary的大小

为了避免系统偏差，每个实验我都会跑3次，把日志重定向到对应的log文件里。比如我的第一次运行结果的log如下：

```
args=Namespace(trainer='bpe_v1_time', out_dir='test_tiny_story_v1_1', vocab_size=10000, data_path='data/TinyStoriesV2-GPT4-train.txt')
unknown_args=[]
_pretokenize_and_count time 622.017874084413
count_pairs_time: 944.907833525911, max_time: 167.04244575463235, update_time: 453.1098820846528
total time: 2187.1810638625175
total train time: 2187.20 seconds
```

可以看到，训练openwebtext总的时间是2187秒，这是超过了作业的目标(30分钟)。另外我们可以看到有944秒花在了调用BPE_Trainer._count_pairs上，max函数的时间是167秒，而更新word_encoding的时间是453秒。

第一个版本的实现虽然是正确的，但是它的速度很慢。优化它是我们的后续目标。

## 4. 结果

为了便于比较，每一篇文章都会汇总到当前版本的所有测试结果。


版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v1_time | tinystory |2187/2264/2247|622/642/628| count_pair:944/995/984 max:167/173/174/174 update:453/453/460 |


