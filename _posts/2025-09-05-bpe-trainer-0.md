---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第0部分：简介" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于15分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第一篇，内容包括这个任务的介绍，获取源代码和设置开发环境。

<!--more-->


**目录**
* TOC
{:toc}


## 1.任务介绍

我们的任务是完成Stanford CS336课程的[作业1](https://github.com/stanford-cs336/assignment1-basics)的一部分：实现BPE Tokenizer训练代码。关于作业的内容可以参考[CS336 Assignment 1 (basics): Building a Transformer LM](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)。这个作业的内容很多，我们这里只关注其中的**2.4 BPE Tokenizer Training**和**2.5 Experimenting with BPE Tokenizer Training**。

有关Tokenizer的原理可以参考youtube视频[Stanford CS336 Language Modeling from Scratch \| Spring 2025 \| Lecture 1: Overview and Tokenization](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=1)和[Course Materials](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_01.json)。

### BPE Tokenizer Training介绍

这部分内容来自[CS336 Assignment 1 (basics): Building a Transformer LM](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)的**2.4 BPE Tokenizer Training**

BPE 分词器（Tokenizer）的训练过程包含三个主要步骤。

#### 词汇表初始化 (Vocabulary initialization)

 分词器的词汇表是字节串token到整数 ID 的一对一映射。由于我们正在训练一个字节级别的 BPE 分词器，我们的初始词汇表就是所有字节的集合。因为总共有 256 种可能的字节值，所以我们初始词汇表的大小为 256。

#### 预分词 (Pre-tokenization)

有了词汇表之后，原则上你可以计算文本中哪些字节对相邻出现的频率最高，然后从频率最高的字节对开始合并它们。然而，这种做法在计算上非常昂贵，因为每次合并时我们都必须对整个语料库进行一次完整的遍历。此外，直接在整个语料库中合并字节可能会导致仅在标点符号上有所不同的标记（例如，dog! 和 dog.）。尽管这些标记的语义可能高度相似（因为它们只在标点符号上不同），但它们会获得完全不同的标记 ID。

为了避免这种情况，我们对语料库进行预分词。你可以将此视为一种粗粒度的语料库分词，它帮助我们统计字符对出现的频率。例如，单词“text”可能是一个预分词标记，它出现了 10 次。在这种情况下，当我们统计字符“t”和“e”相邻出现的频率时，我们会看到单词“text”中“t”和“e”是相邻的，我们可以将它们的计数增加 10，而不是遍历整个语料库。由于我们正在训练一个字节级别的 BPE 模型，每个预分词标记都被表示为一个 UTF-8 字节序列。

Sennrich 等人 [2016] 的原始 BPE 实现通过简单地按空白字符分割来预分词（即 s.split(" ")）。与此相反，我们将使用一个基于正则表达式的预分词器（被 GPT-2 [Radford et al., 2019] 使用），该分词器来自 github.com/openai/tiktoken/pull/234/files。

```
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

下面是Gemini对这个正则表达式的解释，仅供参考：


1.  `'(?:[sdmt]|ll|ve|re)`
    * `'`：匹配一个单引号。
    * `(?:...)`：这是一个**非捕获分组**，它将内部的模式作为一个整体对待，但不会为它创建反向引用。
    * `[sdmt]`：匹配单个字符 `s`、`d`、`m` 或 `t`。这通常用于处理英语中的缩写，例如 `it's` (is/has)、`I'd` (had/would)、`I'm` (am) 和 `don't` (not)。
    * `|ll|ve|re`：匹配两个字符的缩写，例如 `we'll` (will)、`we've` (have) 或 `they're` (are)。
    * **总结**：这一部分专门匹配常见的英语缩写，确保像 `'s`、`'ll` 这样的部分与前面的词语一起被视为一个整体，避免将它们单独分词。

2.  ` ?\p{L}+`
    * `?`：匹配它前面的字符（空格）零次或一次。这意味着它可以匹配一个可选的空格。
    * `\p{L}+`：匹配一个或多个 Unicode 字母字符。`\p{L}` 属性确保该模式不仅能匹配英文字母，还能匹配所有语言的字母（如中文、西里尔字母等）。
    * **总结**：这部分匹配一个或多个字母组成的单词，并且允许单词前有一个可选的空格。

3.  ` ?\p{N}+`
    * `?`：匹配一个可选的空格。
    * `\p{N}+`：匹配一个或多个 Unicode 数字字符。`\p{N}` 属性涵盖了所有数字。
    * **总结**：这部分匹配一个或多个数字，同样允许数字前有一个可选的空格。

4.  ` ?[^\s\p{L}\p{N}]+`
    * `?`：匹配一个可选的空格。
    * `[...]`：这是一个**字符集**。
    * `^`：在字符集内部表示“不匹配”后面的任何字符。
    * `\s`：匹配任何空白字符（如空格、制表符、换行符等）。
    * `\p{L}`：匹配任何 Unicode 字母。
    * `\p{N}`：匹配任何 Unicode 数字。
    * **总结**：这部分匹配一个或多个既不是空白、也不是字母、也不是数字的字符。这实际上涵盖了大多数标点符号，例如 `!`, `?`, `.` 等。

5.  `\s+(?!\S)`
    * `\s+`：匹配一个或多个空白字符。
    * `(?!\S)`：这是一个**否定式先行断言**。它断言当前位置后面**不是**一个非空白字符（`\S` 是 `\s` 的反义词）。
    * **总结**：这部分用于匹配字符串末尾或者大块空白区域中的空白字符。它是一个精巧的技巧，用于处理不与任何其他非空白字符相连的空白。

6.  `\s+`
    * 这部分是最后的“兜底”模式。
    * `\s+`：简单地匹配一个或多个空白字符。
    * **总结**：这部分确保了任何没有被前面更精确的模式匹配到的空白字符都能被捕获。

我们来看一个例子：

```
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过，在你的代码中使用时，应该使用 `re.finditer` 来避免在构建预分词到其计数的映射时，存储所有预分词的词语。


#### 计算 BPE 合并

现在，我们已经将输入文本转换为预分词标记，并将每个预分词标记表示为一个 UTF-8 字节序列，接下来我们就可以计算 BPE 合并（即训练 BPE 分词器）。

从宏观上看，BPE 算法会迭代地计算每对字节的出现频率，并找出频率最高的一对（“A”, “B”）。然后，这对出现频率最高的字节对（“A”, “B”）的每次出现都会被合并，即被一个新的标记“AB”所取代。这个新的合并标记会被添加到我们的词汇表中；因此，BPE 训练后的最终词汇表大小将是初始词汇表的大小（本例中为 256），加上训练过程中执行的 BPE 合并操作的数量。为了在 BPE 训练期间提高效率，我们不考虑跨越预分词边界的字节对。在计算合并时，当多个字节对的频率相同时，可以通过选择字典序上更大的字节对来确定性地打破僵局。例如，如果字节对（“A”, “B”）、（“A”, “C”）、（“B”, “ZZ”）和（“BA”, “A”）都具有最高频率，我们会合并（“BA”, “A”）：

```
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

#### 特殊标记 (Special tokens)

通常，一些字符串（例如 <\|endoftext\|> ）被用来编码元数据（例如文档之间的边界）。在对文本进行编码时，通常希望将某些字符串作为“特殊标记”来处理，这些标记永远不应被拆分为多个标记（即，它们将始终被保留为单个标记）。例如，序列结束字符串 <\|endoftext\|> 应始终被保留为单个标记（即一个单一的整数 ID），这样我们才知道何时停止从语言模型中生成文本。这些特殊标记必须被添加到词汇表中，以便它们有一个对应的固定标记 ID。

Sennrich 等人 [2016] 的算法 1 包含了一个效率低下的 BPE 分词器训练实现（基本遵循我们上面概述的步骤）。作为第一个练习，实现和测试这个函数可能有助于检验你的理解。

这是一个关于 BPE（字节对编码）分词器训练的示例，灵感来自 Sennrich 等人 [2016] 的研究。

#### 示例 (bpe\_example)

让我们考虑一个由以下文本组成的语料库：

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且词汇表中有一个特殊的标记 <\|endoftext\|> 。

**词汇表初始化**

我们用特殊标记  <\|endoftext\|>  和 256 个字节值来初始化我们的词汇表。

**预分词**

为了简化并专注于合并过程，我们在这个例子中假设预分词仅仅是按空白字符进行分割。当我们预分词并计数时，会得到下面的频率表：

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

将这个频率表表示为 `dict[tuple[bytes], int]` 的形式很方便，例如 `{(l,o,w): 5 ...}`。需要注意的是，在 Python 中，即使是单个字节也是一个 `bytes` 对象，就像 Python 中没有 `char` 类型来表示单个字符一样。


**合并**

我们首先查看每一对连续的字节，并累加它们出现的单词的频率，得到以下结果：`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。

字节对 `('es')` 和 `('st')` 的频率都是 9，相同。根据规则，我们取字典序上更大的那一对，即 `('st')`。

接下来，我们将预分词标记进行合并，最终得到：`{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

在第二轮中，我们看到 `(e, st)` 是最常见的字节对（出现 9 次），于是我们将它们合并，得到：`{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。

继续这个过程，最终我们将得到一系列的合并，顺序为：`['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果只进行 6 次合并，我们得到的合并序列是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，而我们的词汇表元素将包括： <\|endoftext\|>  ，加上 256 个字节，以及 `st`, `est`, `ow`, `low`, `west`, `ne`。

有了这个词汇表和这组合并，单词 `newest` 将会被分词为 `[ne, west]`。

###  BPE 分词器训练实现

这部分内容来自[CS336 Assignment 1 (basics): Building a Transformer LM](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)的**2.5 Experimenting with BPE Tokenizer Training**


#### 在 TinyStories 数据集上训练字节级 BPE 分词器

我们将在 TinyStories 数据集上训练一个字节级 BPE 分词器。该数据集的查找和下载说明可在第 1 节中找到。在开始之前，我们建议你先了解一下 TinyStories 数据集，感受一下数据的内容。

#### 预分词的并行化

你会发现，预分词步骤是一个主要的性能瓶颈。你可以利用 Python 内置的 `multiprocessing` 库来并行化代码，从而加快预分词的速度。具体来说，我们建议在预分词的并行实现中，对语料库进行分块，同时确保你的分块边界都出现在特殊标记的开头。你可以直接使用以下链接中的起始代码来获取分块边界，然后用这些边界将任务分配到不同的进程中：

`https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py`

这种分块方式总是有效的，因为我们从不希望跨文档边界进行合并。对于本次作业而言，你总能以这种方式进行分割。不用担心语料库非常大且不包含 <\|endoftext\|> 特殊标记的边缘情况。

#### 在预分词前移除特殊标记

在使用正则表达式模式运行预分词（使用 `re.finditer`）之前，你应该从你的语料库（或分块，如果你使用并行实现）中剥离所有特殊标记。确保你以特殊标记为分隔符进行分割，这样就不会在它们所分隔的文本之间发生合并。例如，如果你的语料库（或分块）是 `[Doc 1]<|endoftext|>[Doc 2]`，你应该以特殊标记 `<|endoftext|>` 为分隔符进行分割，然后分别对 `[Doc 1]` 和 `[Doc 2]` 进行预分词，这样就不会在文档边界处发生合并。这可以通过使用 `re.split` 并以 `re.escape` 处理后的 `re.escape("\|".join(special_tokens))` 作为分隔符来实现。`test_train_bpe_special_tokens` 测试用例将对这一点进行检查。

#### 优化合并步骤

上面示例中朴素的 BPE 训练实现之所以很慢，是因为在每次合并时，它都会遍历所有的字节对，以找出频率最高的那一对。然而，每次合并后，只有与被合并的字节对重叠的那些字节对的计数会发生变化。因此，可以通过**索引**所有字节对的计数并**增量更新**这些计数来提高 BPE 训练速度，而不是显式地遍历每个字节对来计算频率。通过这种缓存机制，你可以获得显著的性能提升，尽管我们需要注意的是，BPE 训练的合并部分在 Python 中无法并行化。

#### 低资源/降级技巧：性能分析 (Profiling)

你应该使用像 `cProfile` 或 `scalene` 这样的性能分析工具来找出你实现中的瓶颈，并专注于优化这些瓶颈。

#### 低资源/降级技巧：“降级” (Downscaling)

我们建议你不要直接在完整的 TinyStories 数据集上训练分词器，而是先在一个小的数据子集上进行训练，也就是一个“调试数据集”。例如，你可以先在 TinyStories 验证集上训练分词器，它有 2.2 万个文档，而不是 212 万个。这展示了一种通用的开发策略：尽可能“降级”来加快开发速度，比如使用更小的数据集、更小的模型规模等。选择调试数据集的大小或超参数配置需要仔细考量：你希望你的调试集足够大，能暴露出与完整配置相同的瓶颈（这样你所做的优化才具有普适性），但又不能大到运行起来遥遥无期。


#### 问题 (train_bpe)：BPE 分词器训练 (15 分)

**要求**：编写一个函数，给定一个输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数：

* **input_path**: `str`，BPE 分词器训练数据文本文件的路径。
* **vocab_size**: `int`，一个正整数，定义最终词汇表的最大大小（包括初始字节词汇表、合并产生的词汇项以及任何特殊标记）。
* **special_tokens**: `list[str]`，一个字符串列表，用于添加到词汇表中。这些特殊标记不会影响 BPE 的其他训练过程。

你的 BPE 训练函数应返回以下结果：

* **vocab**: `dict[int, bytes]`，分词器词汇表，将整数（词汇表中的标记 ID）映射到字节（标记的字节序列）。
* **merges**: `list[tuple[bytes, bytes]]`，训练过程中产生的 BPE 合并列表。列表中的每一项都是一个字节元组 (`<token1>`, `<token2>`)，表示 `<token1>` 与 `<token2>` 合并。合并应按照创建的顺序排列。

为了用我们提供的测试用例测试你的 BPE 训练函数，你首先需要实现在 `[adapters.run_train_bpe]` 的测试适配器。然后，运行 `uv run pytest tests/test_train_bpe.py`。

你的实现应该能够通过所有测试。作为可选任务（这可能需要投入大量时间），你可以使用某种系统语言（例如 C++ 或 Rust）来实现训练方法中的关键部分。如果你这样做，请注意哪些操作需要复制数据，哪些可以直接从 Python 内存中读取，并确保留下构建说明，或者确保它只使用 `pyproject.toml` 就能构建。另外请注意，GPT-2 的正则表达式在大多数正则表达式引擎中支持得不好，在那些支持的引擎中也会非常慢。我们已经验证过 Oniguruma 足够快并且支持否定式先行断言，但 Python 内置的 `regex` 包如果安装正确的话，甚至更快。

## 2.代码

### 克隆代码

```
git clone https://github.com/fancyerii/assignment1-basics-bpe.git
```

### 克隆子模块

如果需要运行cppupdate中的C++代码，还需要更新absl和emhash这两个第三方库。

```
cd assignment1-basics-bpe/
git submodule update --init --recursive
```

关于cppupdate和cppstep2的编译和运行，请参考后续文章。

### python环境

需要安装uv，uv的安装请参考[Installing uv](https://docs.astral.sh/uv/getting-started/installation/)。

```
uv sync --python 3.12
source .venv/bin/activate
```

### 测试

```
python -m pytest tests/
```

注意：这里测试的是第一个版本的实现，3个测试有两个会通过，但是另一个测试会超时。运行的输出结果类似：

```
tests/test_train_bpe.py::test_train_bpe_speed test_train_bpe_speed: 3.6250589709961787
FAILED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

从第二个版本开始我们的代码可以轻松的通过这个测试，具体内容请参考后续文章。

### 下载数据

```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


