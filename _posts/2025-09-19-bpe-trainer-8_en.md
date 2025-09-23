---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 8: Implementing Fine-Grained Updates" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the ninth article, which focuses on optimizing the update process for data structures like `pair_counts`.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Problem Analysis

We previously left one optimization point unaddressed: when updating `pair_counts`, many pairs' counts don't actually need to change, but for the sake of simple implementation, we first delete them and then add them back. More specifically, let's look at a previous example.

Suppose the word frequency count is:

```
{low: 5, lower: 2, widest: 3, newest: 6, es: 2, st: 2}
```

Then we count the pair frequencies:

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 11, st: 11, ne: 6, ew: 6}
```

Based on the merge rules, we choose to merge `('s', 't')`. Since they are merged, we need to update the pair frequency counts. Let's look at our current update algorithm (the code is in [BPE\_Trainer.\_updated\_affected\_word\_count](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v2.py#L55), though a more accurate name would be `_updated_pair_count_of_affected_word`).

To get the new pair frequencies, we use `pair_to_words` to find the words containing the pair `('s', 't')`:

```
'widest'
'newest'
```

We first subtract all pairs within these two words, which means subtracting:

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

Therefore, based on the new `word_encodings`, we calculate the new pair frequencies:

```
{wi: 3, id: 3, de: 3, est: 3}
{ne: 6, ew: 6, we: 6, est: 6}
```

Adding these frequencies back gives us:

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 2, st: 2, ne: 6, ew: 6, est: 9}
```

As you can see, the frequencies for pairs like `'wi'`, `'id'`, and `'de'` were subtracted and then added back, leading to unnecessary updates. Worse still, after `'wi'` is deleted, its frequency becomes 0, so it needs to be deleted from `pair_counts` and `pair_to_words`.

```
                    if pair_counts[old_pair] <= 0:
                        # we accounted for all occurrences of this pair
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair)
```

According to [Internals of sets and dicts](https://www.fluentpython.com/extra/internals-of-sets-and-dicts/), CPython's `dict` uses an open-addressing collision resolution strategy. Deleting a key from a `dict` just marks it (unless a rehash occurs), and adding the same key back will reuse the original address. So the impact on the `dict` itself is minimal, but the key and value objects still need to be reconstructed (since the `dict` stores pointers to them). If a separate chaining method like `std::unordered_map` is used, deleting and reinserting will mess up the original order of the linked list (the new element is inserted at the head). For example, if a bucket's elements were `a->b->c`, deleting `b` and reinserting it might make the new list `b->a->c`.

However, if we analyze carefully, we can identify which deletions are unnecessary. For instance, in our example, we are merging `('s', 't')`. In `'widest'`, the only pairs affected by the merge are `('e', 's')` and `('s', 't')` (there's no pair after `'t'`). The pairs `['w', 'i']`, `['i', 'd']`, and `['d', 'e']` are unaffected. Therefore, we only need to remove `'es'` and `'st'` and add `('e', 'st')`.

## 2\. Fine-Grained Frequency Update Algorithm

Based on the previous analysis, we can implement a fine-grained frequency update algorithm. The example we analyzed was a case where the pair to be merged appeared only once in a word. But what if the pair appears multiple times? One approach would be to fall back to the previous algorithm. The possibility of a pair appearing multiple times in a single word is not high. If it appears only once, it can be handled simply; if it appears multiple times, we can fall back to the old, coarse-grained update algorithm—first delete, then insert. I didn't think of this method before writing the article, which led me to a more complex solution. I created a complicated algorithm that can handle a pair appearing multiple times in one word. Although this algorithm seems perfect, the implementation is much more difficult, and the payoff is not that great. But since I've already implemented it, I'll continue with it.

For example, consider a hypothetical word `"abcststefstbbef"`. We can first find all indices where 'st' appears. To make it easier to see, we'll separate 'st' from other tokens with a space.

```
abc st st ef st bbef
```

Based on the previous analysis, the pairs before the first 'st' (`'ab'`, `'bc'`) are unaffected, and the pairs after the last 'st' (`'bb'`, `'be'`) are also unaffected. What about the pairs in the middle? Let's write them out:

```
cs st ts st te ef fs st tb
```

These pairs first need to be deleted, and then we generate new tokens for the affected section:

```
c st st e f st b
```

Then we add the new pairs:

```
('c','st'), ('st','st'), ('st','e'), ('e','f'), ('f','st'), ('st','b')
```

Besides decreasing the counts of the affected old pairs and increasing the counts of the new pairs, we also need to modify the inverted index `pair_to_words`. My earliest code would simply delete the affected pairs from it. But this has a bug. For example, `'ef'` is affected and needs its count reduced. However, `'ef'` also appears at the end of the word, not just in the middle of the three 'st' occurrences. If we simply delete `'ef'` from `pair_to_words`, the inverted index becomes incorrect. A "true" inverted index not only records that the pair `('e', 'f')` appeared in the word `'abcststefstbbef'` but also how many times it appeared. This would mean that the value of `pair_to_words` should be a `dict` mapping words to their pair frequencies, not a `set`. So we only know that the pair `('e', 'f')` appeared in `'abcststefstbbef'`, but we don't know how many times. This is where the problem lies. For example, in the case above, if we simply delete the word `'abcststefstbbef'` from `pair_to_words[('e','f')]`'s value set, the inverted index would be wrong. However, we must delete `'abcststefstbbef'` from the value of `pair_to_words[('t','e')]`, because the pair `('t', 'e')` appears only once in the word `'abcststefstbbef'`.

One solution is to change the `pair_to_words` code to a `{pair: {word: word_freq}}` structure. This is a clearer way to do it. However, at the time, I felt that this change would require too many modifications, and I wanted to limit the changes to a single function. So I used another method (which upon reflection is not ideal). This method is: find all the unaffected pairs and put them into a `set`. When modifying the inverted index, if the pair to be deleted is also in the set of unaffected pairs, we don't delete it. Logically, this algorithm is correct, but it makes the code exceptionally complex. And for a case that rarely occurs, it complicates the code and slows down the common cases, which is not a worthwhile trade-off.

Nonetheless, scanning all the unaffected pairs is relatively fast, and many later improved versions are based on this current algorithm, so modifying it would be a hassle. For the time being, I will not change it in this article.

## 3\. Code Implementation

The complete code is in [bpe\_v5.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v5.py). Let's look at the changes. First, the modification to `_updated_affected_word_count`:

```
    @staticmethod
    def _updated_affected_word_count(merge_pair, affected_words, word_encodings, 
                                     word_counts, pair_counts, pair_to_words, 
                                     new_id, pair_strings, vocabulary):
        # we may update/delete words when iterate it.
        affected_words = affected_words.copy()
        diff_pairs = defaultdict(int)

        new_pairs = set() 
        BPE_Trainer.fine_grained_pair_counter_diff(affected_words, word_encodings, word_counts, merge_pair, diff_pairs, 
                             new_id, pair_to_words, new_pairs)
        for pair, count in diff_pairs.items():
            if count == 0: continue
            pair_counts[pair] += count
            if pair_counts[pair] <= 0: # should not less than 0!
                del pair_counts[pair]
                pair_to_words.pop(pair, None)


        for new_pair in new_pairs:
            if new_pair not in pair_strings:
                pair_strings[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])


```

It becomes very simple. It calls a new function, `BPE_Trainer.fine_grained_pair_counter_diff`, to calculate the changes in pair counts after a merge. The main output of this function is `diff_pairs`. For example, a hypothetical output might be:

```
('a','s'): -3
('t','f'): -5
('st','b'):10
```

This indicates that due to the merge of `('s','t')`, the count of the old pair `('a','s')` decreased by 3, and `('t','f')` decreased by 5; while the new pair `('st','b')` increased by 10. The `for` loop below then uses this `diff` to update `pair_counts`. Additionally, the `BPE_Trainer.fine_grained_pair_counter_diff` function also returns the `new_pairs` that were created after the merge, and we need to update `pair_strings` with these new pairs as well. Let's focus on the `BPE_Trainer.fine_grained_pair_counter_diff` function.

```
    @staticmethod
    def fine_grained_pair_counter_diff(affected_words, word_encodings, word_counts, merge_pair, diff_pairs, new_id, pair_to_words, new_pairs):
        for word in affected_words:
            word_tokens = word_encodings[word]
            wc = word_counts[word]

            # find first and last pairs
            idx = 0
            unaffected_pairs = set()
            while idx < len(word_tokens) - 1:
                if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                    first_idx = idx
                    break
                idx += 1
            else:
                print(f"bug {merge_pair}, {word}, {word_tokens}")
                raise
            # assert first_idx exists

            idx = len(word_tokens) - 2
            while idx > first_idx + 1:
                if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                    last_idx = idx
                    break
                idx -= 1
            else:
                last_idx = first_idx

            start_idx = max(0, first_idx - 1) # inclusive
            end_idx = min(last_idx + 3, len(word_tokens)) # exclusive

            # unaffected [0, start_idx)
            
            for i in range(start_idx):
                pair = word_tokens[i], word_tokens[i + 1]
                unaffected_pairs.add(pair)
            # unaffected [end_idx-1, :-1]
            for i in range(end_idx - 1, len(word_tokens) - 1):
                pair = word_tokens[i], word_tokens[i + 1]
                unaffected_pairs.add(pair)                

            affected_tokens = word_tokens[start_idx: end_idx]
            for i in range(len(affected_tokens) - 1):
                old_pair = (affected_tokens[i], affected_tokens[i + 1])
                diff_pairs[old_pair] -= wc 
                if old_pair not in unaffected_pairs:   
                    pair_to_words[old_pair].discard(word)
                    

            new_tokens = []
            all_new_tokens = []
            for i in range(start_idx):
                all_new_tokens.append(word_tokens[i])
            
            i = 0

            # account for multiple occurrences of the pair
            while i < len(affected_tokens):
                if i < len(affected_tokens) - 1 and (affected_tokens[i], affected_tokens[i + 1]) == merge_pair:
                    new_tokens.append(new_id)
                    all_new_tokens.append(new_id)
                    # jump past pair
                    i += 2
                else:
                    new_tokens.append(affected_tokens[i])
                    all_new_tokens.append(affected_tokens[i])
                    i += 1
            

            for i in range(end_idx, len(word_tokens)):
                all_new_tokens.append(word_tokens[i])
            
            word_encodings[word] = all_new_tokens

            # add new pairs from the updated word
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])

                diff_pairs[new_pair] += wc
                pair_to_words[new_pair].add(word)

                new_pairs.add(new_pair)
```

This code is quite complex, especially the index handling. Let's look at it section by section.

**Finding the first and last occurrences of the merged pair**

We first need to find the first and last occurrence of the pair. If it only appears once, these two positions are the same. The range we care about is from the first element of the first occurrence to the last element of the last occurrence.

```
            idx = 0
            unaffected_pairs = set()
            while idx < len(word_tokens) - 1:
                if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                    first_idx = idx
                    break
                idx += 1
            else:
                print(f"bug {merge_pair}, {word}, {word_tokens}")
                raise
            # assert first_idx exists

            idx = len(word_tokens) - 2
            while idx > first_idx + 1:
                if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                    last_idx = idx
                    break
                idx -= 1
            else:
                last_idx = first_idx
```

The first `while` loop finds the first occurrence of the pair and stores its index in `first_idx`. The `else` block should not be executed (unless our inverted index has a bug), so it's a defensive exception. The second `while` loop works backward from the end to find the last occurrence, storing its index in `last_idx`. If it doesn't find it, it means the pair only appeared once, so `last_idx` will be the same as `first_idx`.

Using the previous example, we find that `first_idx` is 3 and `last_idx` is 9. As shown below:

```
"abcststefstbbef"
 012345678901234
    |     | 
first_idx |
       last_idx  
```

Next, we find all the unaffected pairs:

```
            start_idx = max(0, first_idx - 1) # inclusive
            end_idx = min(last_idx + 3, len(word_tokens)) # exclusive

            # unaffected [0, start_idx) 
            for i in range(start_idx):
                pair = word_tokens[i], word_tokens[i + 1]
                unaffected_pairs.add(pair)
                
            # unaffected [end_idx-1, :-1]
            for i in range(end_idx - 1, len(word_tokens) - 1):
                pair = word_tokens[i], word_tokens[i + 1]
                unaffected_pairs.add(pair) 
```

The index calculations here are particularly tricky, so I won't go into details. The final result is that the following unaffected pairs are stored in `unaffected_pairs`:

```
ab bc bb be ef
```

Next is to decrease the frequency of the affected pairs and update the inverted index:

```
            affected_tokens = word_tokens[start_idx: end_idx]
            for i in range(len(affected_tokens) - 1):
                old_pair = (affected_tokens[i], affected_tokens[i + 1])
                diff_pairs[old_pair] -= wc 
                if old_pair not in unaffected_pairs:   
                    pair_to_words[old_pair].discard(word)
```

For our example, assuming `wc` is 10, the code will execute:

```
diff_pairs[('c','s')] = -10
pair_to_words[('c','s')].discard(word)
```

For `('e','f')`, `pair_to_words[('e','f')].discard(word)` will not be executed because `('e','f')` is in `unaffected_pairs`.

Next, we calculate the new tokens:

```
            new_tokens = []
            all_new_tokens = []
            for i in range(start_idx):
                all_new_tokens.append(word_tokens[i])
            
            i = 0

            # account for multiple occurrences of the pair
            while i < len(affected_tokens):
                if i < len(affected_tokens) - 1 and (affected_tokens[i], affected_tokens[i + 1]) == merge_pair:
                    new_tokens.append(new_id)
                    all_new_tokens.append(new_id)
                    # jump past pair
                    i += 2
                else:
                    new_tokens.append(affected_tokens[i])
                    all_new_tokens.append(affected_tokens[i])
                    i += 1
            

            for i in range(end_idx, len(word_tokens)):
                all_new_tokens.append(word_tokens[i])
```

Note: Previously we only had `new_tokens`, but now we also have `all_new_tokens`. `new_tokens` is for the affected part, while `all_new_tokens` is for the entire word. So for our `"abcststefstbbef"` example, the results are:

```
new_tokens = ['c','st','st','e','f','st','b']
all_new_tokens = ['a','b'] + new_tokens + ['b','e','f']
```

Finally, we calculate the frequencies of the new pairs:

```
            word_encodings[word] = all_new_tokens

            # add new pairs from the updated word
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])

                diff_pairs[new_pair] += wc
                pair_to_words[new_pair].add(word)

                new_pairs.add(new_pair)
```

The first line updates `word_encodings` with `all_new_tokens`. For the example above:

```
word_encodings['abcststefstbbef'] = ['a','b', 'c','st','st','e','f','st','b', 'b','e','f']
```

Then it finds all the new pairs in `new_tokens`:

```
('c','st')
('st','st')
('st','e')
('e','f')
('f','st')
('st','b')
```

Here, "new" pairs don't necessarily mean they didn't exist before. For example, `('e','f')` did exist, but it was deleted and added back, which could result in `diff_pairs[('e','f')]==0`. This is why the code checks if `diff_pairs` is zero; if it is, it means there was no change. However, the code above will first delete the `('e','f')` pair with `diff_pairs[old_pair] -= wc` and then add it back later. But this only happens if a pair appears more than twice in a single word. If a pair appears only once, like in the word `'abcstbbef'`, the deleted pairs are only `[('c','s'), ('s','t'), ('t','b')]`, and it's impossible for them to be added back.

## 4\. Testing

The test code is in [bpe\_v5\_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v5_time.py).

Version         | Data      | Total Time (s) | Word Freq Time (s) | Merge Time (s) | Other
bpe\_v1\_time | tinystory |2187/2264/2247|622/642/628| count\_pair:944/995/984 max:167/173/174/174 update:453/453/460 |
bpe\_v2\_time | tinystory |757/738/746|639/621/627| 118/117/118 |
bpe\_v2\_time | openweb |35358/34265/35687|2870/2949/2930| 32437/31264/32708 |
bpe\_v3\_time | tinystory |250/90/90|80/90/90| 170/0/0 | num\_counter=8, num\_merger=1 skip
bpe\_v3\_time | openweb |391/401/32352|390/400/410| total:31883 max:31187 update:695 | num\_counter=8, num\_merger=1 skip
bpe\_v5\_time | tinystory |221/236/222 | 90/90/90 | total:130/146/132 max:127/143/129 update:3/3/3 |num\_counter=8, num\_merger=1
bpe\_v5\_time | openweb |34333/34853/35804 | 401/390 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 |num\_counter=8, num\_merger=1

Compared to `bpe_v3_time`, using fine-grained updates on `openweb` reduces the time from 695 seconds to a little over 500 seconds.

## 5\. Porting to C++

Similarly, I ported this fine-grained update algorithm to the C++ versions, including [bpe\_train\_updater\_fine\_grained](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained.cpp), [bpe\_train\_updater\_fine\_grained\_absl](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_absl.cpp), and [bpe\_train\_updater\_fine\_grained\_emhash8](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8.cpp).

I won't go into detail about the code, but interested readers can compare the C++ versions with the Python version.

The experimental results for these versions are:

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe\_train\_updater\_omp\_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v2\_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe\_train\_updater\_omp\_v2\_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe\_train\_updater\_opt\_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637
bpe\_train\_updater\_emhash8 | Boost hash | 479/485/485 | 398/401/401 | 80/83/83
bpe\_train\_updater\_opt\_emhash8 | Boost hash | 469/474/479 | 389/395/399 |79/78/79
bpe\_train\_updater\_opt\_emhash8\_hash| my hash | 2316/1951/1983 | 2250/1888/1918 |66/63/64
bpe\_train\_updater\_fine\_grained | Boost Hash | 8773/8873/7641 | 220/219/233 |8552/8653/7408
bpe\_train\_updater\_fine\_grained\_absl | Absl hash | 845/845/856 | 204/201/203 | 641/643/653
bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60

After implementing the fine-grained update, the total time for `bpe_train_updater_fine_grained_emhash8` is only **260 seconds**, with an update time of just 200 seconds. In contrast, `bpe_train_updater_emhash8` had a total time of 480 seconds, an update time of nearly 400 seconds, and a max time of 79 seconds. This shows that the fine-grained update reduces unnecessary deletions and insertions, cutting the update time by half. These unnecessary operations also tend to make the data structure messy, which is why the `max` time was slightly reduced as well.

## 6\. Using `emhash8::HashSet`/`emhash9::HashSet` to Replace `std::unordered_set`

Besides `pair_counts`, another frequently updated data structure is the inverted index `pair_wordids`:

```
std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> pair_wordids;
```

Here, in addition to the outer `std::unordered_map`, the value itself is an `std::unordered_set<int>`. We can replace `std::unordered_set` with `emhash8::HashSet` or `emhash9::HashSet`. The complete code is available in [`bpe_train_updater_fine_grained_emhash8_set.cpp`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8_set.cpp) and [`bpe_train_updater_fine_grained_emhash8_set9.cpp`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8_set9.cpp).

Using `hash_set4.hpp` (`emhash9::HashSet`) and `hash_table8.hpp` (`emhash8::HashMap`) together can cause some compilation warnings. However, according to [this issue](https://github.com/ktprime/emhash/issues/67#issuecomment-3262163325), we can ignore them. The experimental results are as follows:

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe\_train\_updater\_omp\_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v2\_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe\_train\_updater\_omp\_v2\_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe\_train\_updater\_opt\_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637
bpe\_train\_updater\_emhash8 | Boost hash | 479/485/485 | 398/401/401 | 80/83/83
bpe\_train\_updater\_opt\_emhash8 | Boost hash | 469/474/479 | 389/395/399 |79/78/79
bpe\_train\_updater\_opt\_emhash8\_hash| my hash | 2316/1951/1983 | 2250/1888/1918 |66/63/64
bpe\_train\_updater\_fine\_grained | Boost Hash | 8773/8873/7641 | 220/219/233 |8552/8653/7408
bpe\_train\_updater\_fine\_grained\_absl | Absl hash | 845/845/856 | 204/201/203 | 641/643/653
bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60
bpe\_train\_updater\_fine\_grained\_emhash8\_set | Boost Hash | 192/192/194 | 117/117/117 | 75/75/77
bpe\_train\_updater\_fine\_grained\_emhash8\_set9 | Boost Hash | 168/170/171 | 107/108/109 | 61/62/61

As you can see, by replacing `pair_wordids` with the faster `emhash8::HashMap`/`emhash8::HashSet`/`emhash9::HashSet`, the time was further reduced from over 260 seconds to around 170 seconds.
