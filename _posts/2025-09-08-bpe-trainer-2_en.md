---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 2: Optimizing the Algorithm" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the third article, focusing on optimizing the previous algorithm.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Algorithm Optimization

### 1.1 Analysis of Current Algorithm's Bottlenecks

The previous article introduced the simplest algorithm implementation. Through testing, the total training time on the TinyStories dataset was over 2,000 seconds. Our algorithm can be broken down into two steps: the first is tokenizing the text and counting word frequencies, primarily handled by the `_pretokenize_and_count` method; the second is the iterative merging loop, which involves counting pair frequencies, finding the most frequent pair to add to the vocabulary and merges, and then updating `word_encodings`. In practice, the first step took over 600 seconds, and the second step took over 1,400 seconds. The time for the second step was further divided into three parts: counting pair frequencies (over 900 seconds), finding the most frequent pair (over 160 seconds), and updating (around 450 seconds).

From this analysis, we can see that the primary areas for optimization are the parts that count pair frequencies and update `word_encodings`.

### 1.2 Optimization Strategy

To find a way to optimize, let's look at two cycles of the current algorithm.

Assume the initial word frequency count from the first step is:

```
{low: 5, lower: 2, widest: 3, newest: 6, es: 2, st: 2}
```

#### First Cycle

**Step 1 of the cycle**

Count the frequencies of all pairs. This requires iterating through all words, finding their corresponding tokens via `word_encodings`, and then iterating through the adjacent pairs of tokens to count their occurrences:

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 11, st: 11, ne: 6, ew: 6}
```

**Step 2 of the cycle**

Find the most frequent pairs, which are `st` and `es`, both with a frequency of 11. Since `('s', 't') > ('e', 's')`, we choose `st`. We then add `st` to the vocabulary and merges.

**Step 3 of the cycle**

Since `st` is merged into a new token, we need to iterate through all words in `word_encodings` and update any that contain the `st` pair in their tokens.

```
word_encodings['widest'] = ['w','i','d','e','st']
word_encodings['newest'] = ['n','e','s','e','st']
word_encodings['st'] = ['st']
```

#### Second Cycle

**Step 1 of the cycle**

Count pair frequencies again by iterating through all words, finding their tokens via `word_encodings`, and counting adjacent pairs:

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 2, st: 2, ne: 6, ew: 6, est: 9}
```

Let's compare this with the first cycle:

```
First: {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 11, st: 11, ne: 6, ew: 6}
Second: {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es:  2, st:  2, ne: 6, ew: 6, est: 9}
```

We can see that most pairs have the same counts. The differences are: the counts for `es` and `st` have decreased because `st` was merged, and a new pair `est` was added due to the `st` merge.

**Step 2 of the cycle**

Find the most frequent pair, which is `('e', 'st')`. We merge it and add it to the vocabulary and merges.

**Step 3 of the cycle**

Since `('e', 'st')` is merged into a new token, we need to iterate through all words in `word_encodings` and update any that contain this pair.

```
word_encodings['widest'] = ['w','i','d','est']
word_encodings['newest'] = ['n','e','s','est']
```

### 1.3 Optimization Method

From the analysis above, we can see that for both pair frequencies and `word_encodings`, most words are unaffected. If we know which words are affected by a pair merge, we can just update the pair frequencies and `word_encodings` for those specific words, which would significantly reduce the algorithm's complexity. The problem now is: given a merged pair, which words contain that pair? The obvious solution is to build an **inverted index** that maps pairs to the words containing them.

We can use a dictionary `pair_to_words` to implement this inverted index. The key of this dictionary would be the pair, and the value would be a set containing all words that include that pair. With `pair_to_words`, we can quickly find all affected words and then incrementally update the pair frequencies and `word_encodings`. Of course, the `pair_to_words` inverted index itself must also be updated.

Let's look at how to implement these three incremental updates.

To get the new pair frequencies, we first identify the merged pair `('s', 't')`. Using our `pair_to_words` index, we find the words that contain this pair:

```
'widest'
'newest'
```

We first subtract the counts of all pairs in these two words:

```
{wi: 3, id: 3, de: 3, es: 3, st: 3}
{ne: 6, ew: 6, we: 6, es: 6, st: 6}
```

This gives us:

```
{lo: 7, ow: 7, we: 2, er: 2, wi: 0, id: 0, de: 0, es: 2, st: 2, ne: 0, ew: 0}
```

Next, the tokens for these two words become:

```
word_encodings['widest'] = ['w','i','d','e','st']
word_encodings['newest'] = ['n','e','s','e','st']
```

We then calculate the frequencies of the new pairs from these new encodings:

```
{wi: 3, id: 3, de: 3, est: 3}
{ne: 6, ew: 6, we: 6, est: 6}
```

Adding these back gives us:

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 2, st: 2, ne: 6, ew: 6, est: 9}
```

Note: The algorithm we've implemented here is relatively simple. It gets a word's old tokens, subtracts all of its old pair counts, then gets the word's new tokens after the merge, and adds the new pair counts. A more complex algorithm could diff the changes before and after the merge. For example, `wi` is not affected by the `st` merge, so its frequency doesn't need to be subtracted and then re-added. However, this would make the algorithm much more complex. For now, we'll quickly implement the simple version, and later we can reconsider whether a more complex algorithm is necessary.

## 2\. Code

The source code can be found at [bpe\_v2.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v2.py).

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

We can see that the `train` method is largely similar to before, with minor changes. Since we only modified the second step, the `_pretokenize_and_count` function for tokenization and word counting is completely unchanged. The initialization of `word_encodings` to the UTF-8 bytes of words also remains the same. The main change is the addition of:

```
pair_to_words = defaultdict(set)
```

This dictionary implements the inverted index from pairs to words.

Another significant change is that `BPE_Trainer._count_pairs` is now called only once before the `while` loop. The subsequent pair frequency counting becomes an incremental update, so it is no longer needed in the loop. The three steps from the previous `while` loop (counting pair frequencies, finding the max-frequency pair, and updating `word_encodings`) have all been moved into a new function, `BPE_Trainer._merge_a_pair`.

Now let's look at the changed or new code.

### 2.2 BPE\_Trainer.\_count\_pairs

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

This function is mostly the same as in `bpe_v1`, but it adds a modification for the `pair_to_words` inverted index:

```
pair_to_words[pair].add(word)
```

This statement's purpose is to maintain the inverted index from pairs to words. Because we use `defaultdict(set)`, the code is very simple. If the pair has not appeared before, a new set will be created for it, and the word will be added to the set. Otherwise, the word is added to the existing set. A `set` is used for the value because a single word can contain a pair multiple times.

### 2.3 \_merge\_a\_pair

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

This function first uses `max` to find the most frequent pair and then updates the vocabulary and merges list, which is the same as before. The main change is:

```
        affected_words = pair_to_words[merge_pair]

        # update affected words' counts
        BPE_Trainer._updated_affected_word_count(merge_pair, affected_words, word_encodings,
                                                 word_counts, pair_counts,
                                                 pair_to_words, new_id, pair_strings, vocabulary)
```

First, it finds all words affected by `merge_pair` using `pair_to_words`, and then it calls `_updated_affected_word_count` to incrementally update `word_counts` and `word_encodings`. This also includes updating the `pair_to_words` inverted index itself.

### 2.4 \_updated\_affected\_word\_count

This function is the key to this algorithm optimization. Let's analyze its code in detail.

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
                        has_new_id = True
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

#### Copy `affected_words`

The main loop of this function iterates through `affected_words` and incrementally modifies `word_counts`, `word_encodings`, and `pair_to_words` based on the merged `merge_pair`. Because `affected_words` comes from `pair_to_words[merge_pair]`, `pair_to_words` will be modified during the loop, which could cause problems. To avoid this, we make a copy of `affected_words` before iterating over it. Since `str` is immutable, a shallow copy with `affected_words.copy()` is sufficient.

#### Subtracting Old Pair Counts

The first step in the loop is to subtract the old pair counts for the affected words:

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

The code is relatively easy to understand. First, `pair_counts` is decremented by the word's count for the `old_pair`. If the count becomes 0 or less (the `less than 0` is defensive programming, as it shouldn't happen), we can delete the pair from `pair_counts`. And since the pair no longer exists, the corresponding value (set) in the inverted index `pair_to_words` should also be empty, so we can delete the `old_pair` from `pair_to_words` as well. If the `old_pair`'s count is greater than 0, it means other words still contain it, so we only need to update the inverted index by using `set.discard` to remove the current word.

Note: Since some pairs might be added back in later code, some `old_pair`s might be deleted from `pair_counts` only to be re-added later. The same goes for `pair_to_words`. An alternative is to not delete a pair from `pair_counts` and `pair_to_words` even if its count becomes 0. This wouldn't affect the code's correctness, as the search for the most frequent pair would never find a pair with a count of 0 (unless all non-zero-count pairs have been added to the vocabulary, which is practically impossible unless the training data is tiny and the requested `vocab_size` is very large). However, this would result in `pair_counts` having keys with zero values and `pair_to_words` having empty sets as values. This "garbage data" would make the dictionaries larger and potentially impact performance. It would require real-world testing to determine which method is faster. I will skip this comparison for now, but interested readers can try it.

#### Calculating New `word_encodings`

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

This code is the same as before. It iterates through the original tokens, and if two adjacent tokens form the `merge_pair`, they are replaced by the new token.

#### Adding the New Pair's Count

```
                for i in range(len(new_tokens) - 1):
                    new_pair = (new_tokens[i], new_tokens[i + 1])

                    pair_counts[new_pair] += wc
                    pair_to_words[new_pair].add(word)
                    if new_pair not in pair_strings:
                        pair_strings[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])
```

This section uses the new tokens to re-count the pair frequencies in `pair_counts` and update the inverted index `pair_to_words`. As before, we also add the string representation of the new pair to `pair_strings` to assist with the `max` function.

## 3\. Testing

### 3.1 Unit Testing

Modify `tests/adapters.py` by changing:

```
from cs336_basics import bpe_v1 as bpe
```

to:

```
from cs336_basics import bpe_v2 as bpe
```

Then run `pytest`:

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

Great\! We passed all unit tests. The `test_train_bpe_speed` test took only 0.27s.

### 3.2 TinyStories Training

To measure the time, I've implemented [bpe\_v2\_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v2_time.py).

The test script is:

```
#!/bin/bash

for i in {1..3}; do
  python cs336_basics/test_trainer.py bpe_v2_time test_tiny_story_v2_${i} -d data/TinyStoriesV2-GPT4-train.txt -v 10000 > ts_v2_${i}.log
done
```

The log from one of the runs is:

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

The total time is 757 seconds, which is almost 3 times faster than the previous 2,187 seconds. If we remove the unoptimized `_pretokenize_and_count` time, the merge times are 1,565s versus 118s, making the new version more than 13 times faster\!

### 3.3 OpenWebText Training

OpenWebText is much larger than TinyStories, and optimizing for it is our main goal. The test script is similar, just changing the `vocab_size` and the training text path:

```
#!/bin/bash

for i in {1..3}; do
  python cs336_basics/test_trainer.py bpe_v2_time test_openweb_v2_${i} -d ./data/owt_train.txt -v 32000 > ow_v2_${i}.log
done
```

The log from one of the runs is:

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

The total training time is 35,358 seconds. The `_pretokenize_and_count` step took 2,870s, while the merge step took 32,437s.

## 4\. Results

| Version | Data | Total Time (s) | Count Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v1\_time | tinystory | 2187/2264/2247 | 622/642/628 | count\_pair: 944/995/984 <br> max: 167/173/174/174 <br> update: 453/453/460 | |
| bpe\_v2\_time | tinystory | 757/738/746 | 639/621/627 | 118/117/118 | |
| bpe\_v2\_time | openweb | 35358/34265/35687 | 2870/2949/2930 | 32437/31264/32708 | |
 
 
## Full Series

  * [Part 0: Introduction](/2025/09/05/bpe-trainer-0_en/) Introduces the basic BPE training algorithm and related tasks, as well as the development environment.
  * [Part 1: The Simplest Implementation](/2025/09/07/bpe-trainer-1_en/) The simplest implementation of BPE training.
  * [Part 2: Optimized Algorithm](/2025/09/08/bpe-trainer-2_en/) Implements incremental updates for pair\_counts.
  * [Part 3: Parallel Tokenization and Frequency Counting](/2025/09/09/bpe-trainer-3_en/) Uses multiprocessing to implement a multi-process parallel algorithm.
  * [Part 4: A Failed Parallel Optimization](/2025/09/10/bpe-trainer-4_en/) An attempt to parallelize the max pair calculation using multiple processes.
  * [Part 5: Implementing the Merge Algorithm in C++](/2025/09/12/bpe-trainer-5_en/) Implements a C++ merge algorithm equivalent to the Python version, and compares two ways of iterating through std::unordered\_map.
  * [Part 6: Parallelizing the Max Pair Search with OpenMP](/2025/09/15/bpe-trainer-6_en/) Uses OpenMP to find the max pair in pair\_counts in parallel.
  * [Part 7: Using Flat Hashmap to Replace std::unordered\_map](/2025/09/18/bpe-trainer-7_en/) Uses flat hashmap to replace std::unordered\_map.
  * [Part 8: Implementing Fine-Grained Updates](/2025/09/19/bpe-trainer-8_en/) Implements a fine-grained update algorithm for pair\_counts using an inverted index.
  * [Part 9: Using a Heap to Find the Max Pair](/2025/09/21/bpe-trainer-9_en/) Uses a heap to find the max pair and improve performance.
  * [Part 10: Using Cython and PyPy for Acceleration](/2025/09/24/bpe-trainer-10_en/) Uses Cython and PyPy to accelerate Python code.
  * [Part 11: Wrapping C++ Code with Cython](/2025/09/25/bpe-trainer-11_en/) Wraps C++ code using Cython.
