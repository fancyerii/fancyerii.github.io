---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 1: The Simplest Implementation" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This article series is a sub-project of Stanford's CS336 Assignment 1, focusing on implementing an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, we managed to reduce the training time on OpenWebText from over 10 hours to less than 10 minutes. This series explains that entire optimization process, covering: algorithm optimization, data structure optimization, parallelization with OpenMP, Cython optimization, and the implementation and Cython integration of key components in C++. This is the second article, covering the implementation of the simplest algorithm.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Algorithm Overview

This article implements the simplest training algorithm for a BPE Tokenizer, which consists of two steps. The first step is to split the document into words based on the regular expression introduced previously and then count word frequencies. The vocabulary is then initialized with bytes from 0-255. Using UTF-8 encoding, each word can be split into individual bytes, with each byte initially treated as a token. This process generates an encoding (or embedding) for each word. The next step is to compute the pairs formed by these tokens within each word, find the most frequent pair, add this pair as a new token to the vocabulary, and simultaneously update the new encoding for each word. If two or more pairs have the same frequency, the largest pair is chosen. For example, using the previous pairs:

```
(“A”, “B”), (“A”, “C”), (“B”, “ZZ”), (“BA”, “A”) 
```

If all four pairs have the same frequency, we need to compare string order of the pair. For a pair, we first compare the first element; if they are the same, we then compare the second element. Therefore, the size relationship for these four pairs is:

```
(“BA”, “A”) > (“B”, “ZZ”) > (“A”, “C”) > (“A”, “B”)
```

When comparing a single element, we treat it as bytes. The bytes for "BA" are [66, 65], while for "B" it is [66], so "BA" \> "B".

**Note**: We cannot concatenate the two strings of a pair and then compare them. For example, ("AB", "C") and ("A", "BC") would both become "ABC" if concatenated, but according to our rule, "AB" \> "A", so ("AB", "C") \> ("A", "BC"). If we are certain that our strings will not contain `\0` (null), we can add a `\0` when concatenating. This would make the first pair "AB\\0C" and the second "A\\0BC". Since `\0` is always less than any non-`\0` character, this is not an issue. However, if our strings might contain `\0`, we would have a problem. For instance, ("AB\\0", "C") and ("AB", "\\0C") would be identical with the concatenation method, but in reality ("AB\\0", "C") \> ("AB", "\\0C"). Although `\0` is unlikely to appear in text, this is a potential risk unless we preprocess the text to remove it.

Furthermore, we need to split the original text file into documents to prevent words or tokens from crossing document boundaries. This process usually depends on the source of the text. For example, if we are scraping web pages, a document might be a URL. If our text is a book, we could treat the entire book as one document or split it into smaller units like chapters. Regardless of the document source, for processing efficiency, multiple documents are usually merged into a single file. Otherwise, having too many small files can be handled inefficiently by most operating systems' file systems.

Therefore, we typically assume that when multiple documents are concatenated into a single text file, a special string (e.g., `<|endoftext|>`) is used to separate them. Our training data for this assignment, `data/owt_train.txt` and `data/TinyStoriesV2-GPT4-train.txt`, uses this special string to distinguish different documents.

## 2\. Code

The source code can be found at [bpe\_v1.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v1.py).

### 2.1 BPE\_Trainer

To unify the interface, we'll define a class `BPE_Trainer`, which exposes only one public method:

```
def train(self, input_path, vocab_size, special_tokens, *args)
```

This method has three required parameters:

  * `input_path`: The text file for training, assuming documents are split by `special_tokens`.
  * `vocab_size`: The size of the output vocabulary. For TinyStories, we output 10,000 words, while for OpenWebText, we output 32,000.
  * `special_tokens`: The special strings used to split documents. In our example, this is a single special string `<|endoftext|>`. Note that this input type is a list.

The `train` method's code is as follows:

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

The code is roughly divided into two parts. The first part calls `self._pretokenize_and_count` to split the text into documents, then documents into words, and finally counts their frequencies. The second part is the `while` loop, which calls `BPE_Trainer._count_pairs` to count pair frequencies, then uses the `max` function to find the most frequent pair and adds it to the vocabulary and merges list. We'll analyze these two parts in detail below.

#### \_pretokenize\_and\_count

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

This code first constructs a regular expression for tokenization. It uses [mrab-regex](https://github.com/mrabarnett/mrab-regex), which can be installed with `pip install regex`. We've already installed this dependency in the environment setup section, so you just need to import it at the beginning:

```
import regex as re
```

Note: The standard Python `re` library cannot compile this regular expression.

Next, it constructs a regular expression for splitting documents. Since `special_tokens` can contain multiple strings, `|` is used to join them. Note that a token might contain special characters (e.g., `|` in `<|endoftext|>`), so `re.escape` is used to escape them.

Finally, it calls `BPE_Trainer._chunk_documents_streaming` to split the text into chunks (each chunk containing multiple documents), then uses `re.split` to get the documents (blocks), and finally uses `re.finditer` on each block to tokenize and use `defaultdict` to count word frequencies.

Note: `re.split` returns a list, while `re.finditer` returns an iterator. Returning an iterator uses less memory but might be slightly slower. `re.split` can only return a list, but we could implement similar functionality with `finditer`. We could even use the Aho-Corasick (trie) algorithm for more efficient splitting. However, according to the optimization principle of Amdahl's Law, it's best not to optimize blindly without profiling first.

Next, let's look at `BPE_Trainer._chunk_documents_streaming`, which is a generator function. Here, I referenced [this code](https://github.com/Christine8888/cs336-assignment1-basics/blob/main/cs336_basics/bpe.py#L21).

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

The idea behind this code is to: read a `chunk_size` (default 50K) block from the file, then use `str.rfind` to find the last occurrence of the `special_token` (`<|endoftext|>`) by searching backward. It then returns the text before (and including) that special token, and stores the remaining text in `leftover` to be part of the next read. To reduce memory usage, this is implemented as a generator function using `yield`, instead of putting all results into a list at once. This is because our text files can be very large (e.g., the OpenWebText training data is 11 GB), and loading it all into memory would be problematic.

### 2.2 Other `train` Code

#### Initialization

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

Before entering the main `while` loop, we need to construct the initial vocabulary. The vocabulary is a `dict` where the keys are integer IDs and the values are the corresponding UTF-8 encoded bytes for that token. Additionally, the initial encoding for each word is its UTF-8 encoding.

#### Main Loop

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

The first step is to call `BPE_Trainer._count_pairs` to count the frequencies of all pairs. The code for this function is as follows:

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

In addition to counting the frequency of the pair `encoding[i], encoding[i + 1]`, this function also stores the vocabulary for this pair in `pair_strings`:

```
pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])
```

The key of `pair_strings` is the pair, and the value is a tuple corresponding to the vocabularies of `pair[0]` and `pair[1]`. We store it in `pair_strings` so that in the next step, when searching for the pair, if frequencies are equal, we can find the largest one. This is what the next statement does:

```
            merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
```

The key here is the lambda function passed to `key`. For each pair/count item, it generates a tuple. The first element of the tuple is the pair's frequency (`x[1]`), and the second is the string corresponding to the pair. This allows `max` to first compare by frequency, and if they are equal, to then compare by string. Without `pair_strings`, the code would be:

```
            merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], (vocabulary[x[0][0]], vocabulary[x[0][1]])))
```

Because the value for a specific key (pair) in `pair_strings` is constant, we can store it in `pair_strings` in advance. This avoids repeated lookups in the vocabulary when finding the maximum.

Once the pair to be merged is found, the merge can be performed:

```
            merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

            vocabulary[size] = merge_bytes
            new_id = size
            size += 1

            ...

            merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))
```

Finally, we need to update `word_encodings`. For example, if a word's original encoding was `[1, 2, 1, 1, 2]`, and we merged 1 and 2 into 257, the new encoding for this word becomes `[257, 1, 257]`. The code for this is:

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

This code iterates through the original encoding (`word_tokens`) from beginning to end. If two adjacent tokens can be merged, they are replaced with the new token; otherwise, they remain the same.

## 3\. Testing

### 3.1 Unit Testing

We use `pytest` for unit testing:

```
python -m pytest tests
```

The results are:

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

Two of the three tests pass, but `test_train_bpe_speed` times out. This indicates that our code is correct, but too slow. With a correct implementation, our next goal is to optimize its speed.

However, passing the unit tests is a minor goal. We hope to achieve extreme performance optimization to train on larger datasets like TinyStories and OpenWebText. Therefore, we will test this version on TinyStories as a baseline. Note: this version is too slow to complete the OpenWebText training within 24 hours.

### 3.2 TinyStories Training

To analyze the time spent on different parts (counting word frequencies and iterative merging), I implemented [bpe\_v1\_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v1_time.py), which is nearly identical to `bpe_v1` but adds `time` logging around key function calls.

Additionally, to compare different versions, I also implemented [test\_trainer.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_trainer.py). It can select different versions and options (e.g., number of CPU cores) based on command-line arguments. I won't describe the full code here; readers who are interested can read it themselves. Its usage is quite simple, and I've written a bash script for direct execution:

```
#!/bin/bash

for i in {1..3}; do
  python cs336_basics/test_trainer.py  bpe_v1_time test_tiny_story_v1_${i} -d data/TinyStoriesV2-GPT4-train.txt -v 10000 > ts_v1_${i}.log
done
```

The parameters for `test_trainer.py` are:

  * The first argument specifies which algorithm version to use, in this case, `bpe_v1_time`.
  * The second is the output directory for the results.
  * `-d` is an optional argument to specify the training text, here using TinyStories.
  * `-v` specifies the size of the output vocabulary.

To avoid system bias, I run each experiment three times and redirect the logs to the corresponding `.log` file. For instance, my log from the first run is as follows:

```
args=Namespace(trainer='bpe_v1_time', out_dir='test_tiny_story_v1_1', vocab_size=10000, data_path='data/TinyStoriesV2-GPT4-train.txt')
unknown_args=[]
_pretokenize_and_count time 622.017874084413
count_pairs_time: 944.907833525911, max_time: 167.04244575463235, update_time: 453.1098820846528
total time: 2187.1810638625175
openwebtext time: 2187.20 seconds
```

As you can see, the total training time for OpenWebText is 2178 seconds, which is over the assignment's goal (30 minutes). Additionally, we can see that 944 seconds were spent on calling `BPE_Trainer._count_pairs`, the `max` function took 167 seconds, and updating `word_encoding` took 453 seconds.

The first version is correct but very slow. Optimizing it is our next objective.

## 4\. Results

For easy comparison, each article will summarize all test results up to the current version.

| Version | Data | Total Time (s) | Stat. Freq. Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v1\_time | tinystory | 2187/2264/2247 | 622/642/628 | count\_pair: 944/995/984 <br> max: 167/173/174/174 <br> update: 453/453/460 | |
