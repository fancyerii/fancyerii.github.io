---
layout:     post
title:      "Building and Optimizing a BPE Tokenizer from Scratch—Part 0: Introduction" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford's CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm's training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This first article covers the task's introduction, how to get the source code, and how to set up the development environment.

<!--more-->


**Table of Content**
* TOC
{:toc}


## 1\. Task Introduction

Our task is to complete a part of Stanford's CS336 [Assignment 1](https://github.com/stanford-cs336/assignment1-basics): implementing the BPE Tokenizer training code. For the assignment details, you can refer to [CS336 Assignment 1 (basics): Building a Transformer LM](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf). The assignment has many parts, and we will focus only on **2.4 BPE Tokenizer Training** and **2.5 Experimenting with BPE Tokenizer Training**.

For the principles of Tokenizer, you can refer to the YouTube video [Stanford CS336 Language Modeling from Scratch \| Spring 2025 \| Lecture 1: Overview and Tokenization](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DSQ3fZ1sAqXI%26list%3DPLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_%26index%3D1) and [Course Materials](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_01.json).

### BPE Tokenizer Training

This section is from **2.4 BPE Tokenizer Training** in [CS336 Assignment 1 (basics): Building a Transformer LM](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf).

The BPE tokenizer training procedure consists of three main steps.

#### Vocabulary initialization

The tokenizer vocabulary is a one-to-one mapping from bytestring token to integer ID. Since we’re training a byte-level BPE tokenizer, our initial vocabulary is simply the set of all bytes. Since there are 256 possible byte values, our initial vocabulary is of size 256.

#### Pre-tokenization

Once you have a vocabulary, you could, in principle, count how often bytes occur next to each other in your text and begin merging them starting with the most frequent pair of bytes. However, this is quite computationally expensive, since we’d have to go take a full pass over the corpus each time we merge. In addition, directly merging bytes across the corpus may result in tokens that differ only in punctuation (e.g., dog! vs. dog.). These tokens would get completely different token IDs, even though they are likely to have high semantic similarity (since they differ only in punctuation).

To avoid this, we pre-tokenize the corpus. You can think of this as a coarse-grained tokenization over the corpus that helps us count how often pairs of characters appear. For example, the word 'text' might be a pre-token that appears 10 times. In this case, when we count how often the characters ‘t’ and ‘e’ appear next to each other, we will see that the word ‘text’ has ‘t’ and ‘e’ adjacent and we can increment their count by 10 instead of looking through the corpus. Since we’re training a byte-level BPE model, each pre-token is represented as a sequence of UTF-8 bytes.

The original BPE implementation of Sennrich et al. [2016] pre-tokenizes by simply splitting on whitespace(i.e., s.split(" ")). In contrast, we’ll use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019) from github.com/openai/tiktoken/pull/234/files:

```
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

Here is Gemini's explanation of this regular expression, for reference only:

1.  `'(?:[sdmt]|ll|ve|re)`

      * `'`: Matches a single apostrophe.
      * `(?:...)`: This is a **non-capturing group**, which treats the internal pattern as a whole without creating a backreference for it.
      * `[sdmt]`: Matches a single character `s`, `d`, `m`, or `t`. This is often used to handle English contractions, e.g., `it's` (is/has), `I'd` (had/would), `I'm` (am), and `don't` (not).
      * `|ll|ve|re`: Matches two-character contractions, e.g., `we'll` (will), `we've` (have), or `they're` (are).
      * **Summary**: This part is specifically for matching common English contractions, ensuring that parts like `'s`, `'ll` are treated as a single token with the preceding word, avoiding splitting them off.

2.  `  ?\p{L}+ `

      * `?`: Matches the preceding character (space) zero or one time. This means it can match an optional space.
      * `\p{L}+`: Matches one or more Unicode letter characters. The `\p{L}` property ensures that it matches letters from all languages (e.g., Chinese, Cyrillic, etc.), not just English.
      * **Summary**: This part matches words consisting of one or more letters, allowing for an optional leading space.

3.  `  ?\p{N}+ `

      * `?`: Matches an optional space.
      * `\p{N}+`: Matches one or more Unicode number characters. The `\p{N}` property covers all numbers.
      * **Summary**: This part matches one or more numbers, also allowing for an optional leading space.

4.  `  ?[^\s\p{L}\p{N}]+ `

      * `?`: Matches an optional space.
      * `[...]`: This is a **character set**.
      * `^`: Inside a character set, this means "does not match" any of the following characters.
      * `\s`: Matches any whitespace character (e.g., space, tab, newline, etc.).
      * `\p{L}`: Matches any Unicode letter.
      * `\p{N}`: Matches any Unicode number.
      * **Summary**: This part matches one or more characters that are not whitespace, letters, or numbers. This effectively covers most punctuation symbols, e.g., `!`, `?`, `.`, etc.

5.  `\s+(?!\S)`

      * `\s+`: Matches one or more whitespace characters.
      * `(?!\S)`: This is a **negative lookahead**. It asserts that what immediately follows the current position is **not** a non-whitespace character (`\S` is the opposite of `\s`).
      * **Summary**: This part is used to match whitespace at the end of a string or within large blocks of whitespace. It's a clever trick for handling whitespace not followed by another non-whitespace character.

6.  `\s+`

      * This is the final, "catch-all" pattern.
      * `\s+`: Simply matches one or more whitespace characters.
      * **Summary**: This part ensures that any remaining whitespace not matched by the more specific patterns is captured.

It may be useful to interactively split some text with this pre-tokenizer to get a better sense of its
behavior:

```
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized words
as you construct your mapping from pre-tokens to their counts.

#### Compute BPE Merges

Now that we’ve converted our input text into pre-tokens and represented each pre-token as a sequence of UTF-8 bytes, we can compute the BPE merges (i.e., train the BPE tokenizer). At a high level, the BPE algorithm iteratively counts every pair of bytes and identifies the pair with the highest frequency (“A”, “B”). Every occurrence of this most frequent pair (“A”, “B”) is then merged, i.e.,replaced with a new token “AB”. This new merged token is added to our vocabulary; as a result, the final vocabulary after BPE training is the size of the initial vocabulary (256 in our case), plus the number of BPEmerge operations performed during training. For efficiency during BPE training, we do not consider pairs that cross pre-token boundaries.2 When computing merges, deterministically break ties in pair frequency by preferring the lexicographically greater pair. For example, if the pairs (“A”, “B”), (“A”, “C”), (“B”, “ZZ”), and (“BA”, “A”) all have the highest frequency, we’d merge (“BA”, “A”):

```
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```


#### Special tokens

Often, some strings (e.g., <\|endoftext\|>) are used to encode metadata (e.g., boundariesbetween documents). When encoding text, it’s often desirable to treat some strings as “special tokens” that should never be split into multiple tokens (i.e., will always be preserved as a single token). For example, the end-of-sequence string <\|endoftext\|> should always be preserved as a single token (i.e., a single integer ID), so we know when to stop generating from the language model. These special tokens must be added to the vocabulary, so they have a corresponding fixed token ID. 

Algorithm 1 of Sennrich et al. [2016] contains an inefficient implementation of BPE tokenizer training (essentially following the steps that we outlined above). As a first exercise, it may be useful to implement and test this function to test your understanding.

#### Example (bpe_example): BPE training example

Here is a stylized example from Sennrich et al. [2016]. Consider a corpus consisting of the following text

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

and the vocabulary has a special token <\|endoftext\|>.

**Vocabulary** We initialize our vocabulary with our special token <\|endoftext\|> and the 256 byte values.

**Pre-tokenization** For simplicity and to focus on the merge procedure, we assume in this example
that pretokenization simply splits on whitespace. When we pretokenize and count, we end up with the
frequency table.

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 ...}. Note that even
a single byte is a bytes object in Python. There is no byte type in Python to represent a single byte,
just as there is no char type in Python to represent a single character.

**Merges** We first look at every successive pair of bytes and sum the frequency of the words where they
appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pair ('es')
and ('st') are tied, so we take the lexicographically greater pair, ('st'). We would then merge the
pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.

In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would
merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, the
sequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e',
'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'].

If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocab-
ulary elements would be [<\|endoftext\|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].

With this vocabulary and set of merges, the word newest would tokenize as [ne, west].

### Experimenting with BPE Tokenizer Training

This section is from **2.5 Experimenting with BPE Tokenizer Training** in [CS336 Assignment 1 (basics): Building a Transformer LM](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf).

Let’s train a byte-level BPE tokenizer on the TinyStories dataset. Instructions to find/download the dataset can be found in Section 1. Before you start, we recommend taking a look at the TinyStories dataset to get a sense of what’s in the data.

#### Parallelizing pre-tokenization

You will find that a major bottleneck is the pre-tokenization step. You can speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing. Concretely, we recommend that in parallel implementations of pre-tokenization, you chunk the corpus while ensuring your chunk boundaries occur at the beginning of a special token. You are free to use the starter code at the following link verbatim to obtain chunk boundaries, which you can then use to distribute work across your processes:

`https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py`

This chunking will always be valid, since we never want to merge across document boundaries. For the purposes of the assignment, you can always split in this way. Don’t worry about the edge case of receiving a very large corpus that does not contain <\|endoftext\|>.

#### Removing special tokens before pre-tokenization

Before running pre-tokenization with the regex pattern (using re.finditer), you should strip out all special tokens from your corpus (or your chunk, if using a parallel implementation). Make sure that you split on your special tokens, so that no merging can occur across the text they delimit. For example, if you have a corpus (or chunk) like [Doc 1]<\|endoftext\|>[Doc 2], you should split on the special token <\|endoftext\|>, and pre-tokenize [Doc 1] and [Doc 2] separately, so that no merging can occur across the document boundary. This can be done using re.split with "\|".join(special_tokens) as the delimiter (with careful use of re.escape since \| may occur in the special tokens). The test test_train_bpe_special_tokens will test for this.

#### Optimizing the merging step

The naïve implementation of BPE training in the stylized example above is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However, the only pair counts that change after each merge are those that overlap with the merged pair. Thus, BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can get significant speedups with this caching procedure, though we note that the merging part of BPE training is not parallelizable in Python.

#### Low-Resource/Downscaling Tip: Profiling

You should use profiling tools like cProfile or scalene to identify the bottlenecks in your implementation, and focus on optimizing those.

#### Low-Resource/Downscaling Tip: “Downscaling”

Instead of jumping to training your tokenizer on the full TinyStories dataset, we recommend you first train on a small subset of the data: a “debug dataset”. For example, you could train your tokenizer on the TinyStories validation set instead, which is 22K documents instead of 2.12M. This illustrates a general strategy of downscaling whenever possible to speed up development: for example, using smaller datasets, smaller model sizes, etc. Choosing the size of the debug dataset or hyperparameter config requires careful consideration: you want your debug set to be large enough to have the same bottlenecks as the full configuration (so that the optimizations you make will generalize), but not so big that it takes forever to run.


#### Problem (train_bpe): BPE Tokenizer Training (15 points)

**Deliverable**: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters: 

**input_path**: str Path to a text file with BPE tokenizer training data.

**vocab_size**: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

**special_tokens**: list[str] A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:

**vocab**: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).

**merges**: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (\<token1>, \<token2>), representing that \<token1> was merged with \<token2>. The merges should be ordered by order of creation.

To test your BPE training function against our provided tests, you will first need to implement the test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py.

Your implementation should be able to pass all tests. Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex package in Python is, if anything, even faster.

#### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

* (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <\|endoftext\|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?

    * **Resource requirements**: ≤ 30 minutes (no GPUs), ≤ 30GB RAM
    * **Hint** You should be able to get under 2 minutes for BPE training using multiprocessing during pretokenization and the following two facts:
        * (a) The <\|endoftext\|> token delimits documents in the data files.
        * (b) The <\|endoftext\|> token is handled as a special case before the BPE merges are applied.

    * **Deliverable**: A one-to-two sentence response.
* (b) Profile your code. What part of the tokenizer training process takes the most time?
    * **Deliverable**: A one-to-two sentence response.

#### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

* (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
    * **Resource requirements**: ≤ 12 hours (no GPUs), ≤ 100GB RAM
    * **Deliverable**: A one-to-two sentence response.
* (b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
    * **Deliverable**: A one-to-two sentence response.

#### 问题 (train_bpe_tinystories)：在 TinyStories 上训练 BPE (2 分)

* (a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词汇量为 10,000。确保将 TinyStories 的特殊标记 `<|endoftext|>` 添加到词汇表中。将得到的词汇表和合并操作序列化到磁盘，以供后续检查。训练耗时和占用的内存分别是多少？词汇表中包含的最长标记是什么？这合理吗？

  * **资源要求**: ≤ 30 分钟（无需 GPU），≤ 30GB 内存
  * **提示**：如果你在预分词阶段使用多进程并行化，并利用以下两个事实，你应该能够将 BPE 训练时间控制在 2 分钟以内：
    * (a) 数据文件中的文档以 `<|endoftext|>` 标记作为分隔符。
    * (b) 在应用 BPE 合并之前，`<|endoftext|>` 标记作为特殊情况被处理。
  * **提交内容**：一到两句话的回答。

* (b) 对你的代码进行性能分析。分词器训练过程中哪一部分最耗时？
  * **提交内容**：一到两句话的回答。


#### 问题 (train_bpe_expts_owt)：在 OpenWebText 上训练 BPE (2 分)

* (a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词汇量为 32,000。将得到的词汇表和合并操作序列化到磁盘，以供后续检查。词汇表中包含的最长标记是什么？这合理吗？
  * **资源要求**: ≤ 12 小时（无需 GPU），≤ 100GB 内存
  * **提交内容**：一到两句话的回答。

* (b) 比较和对比在 TinyStories 和 OpenWebText 上训练得到的分词器。
  * **提交内容**：一到两句话的回答。

## 2\. Code

### Clone the Code

```
git clone https://github.com/fancyerii/assignment1-basics-bpe.git
```

### Clone Submodules

If you need to run the C++ code in `cppupdate`, you must also update the third-party libraries `absl` and `emhash`.

```
cd assignment1-basics-bpe/
git submodule update --init --recursive
```

For information on compiling and running `cppupdate` and `cppstep2`, please refer to the subsequent articles.

### Python Environment

You need to install `uv`. For installation instructions, please refer to [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

```
uv sync --python 3.12
source .venv/bin/activate
```

### Testing

```
python -m pytest tests/
```

Note: This tests the first version of the implementation. Two out of three tests will pass, but one will time out. The output will be similar to this:

```
tests/test_train_bpe.py::test_train_bpe_speed test_train_bpe_speed: 3.6250589709961787
FAILED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

Starting with the second version, our code can pass all tests easily. For details, please refer to the next articles.

### Download Data

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


