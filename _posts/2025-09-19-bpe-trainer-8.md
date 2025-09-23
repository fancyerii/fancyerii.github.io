---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第8部分：实现细粒度更新" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第九篇，优化pair_counts等数据结构的更新过程。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 问题分析

我们之前遗留一个优化点，那就是在更新pair_counts时很多pair的计数本来是不需要变的，但是为了实现简单，我们先删除再加入。更具体的，我们来看一下之前的例子。

假设词频统计的结果为：

```
{low: 5, lower: 2, widest: 3, newest: 6, es: 2, st: 2}
```

然后统计pair的频率：

```
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 11, st: 11, ne: 6, ew: 6}
```


然后根据合并规则，我们选择合并('s', 't')。由于它们发生了合并，我们需要更新pair的频率统计。先看看目前我们的更新算法(代码在[BPE_Trainer._updated_affected_word_count](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v2.py#L55)，其实更准确的名字应该叫_updated_pair_count_of_affected_word)。


为了获得新的pair的频率，我们根据pair_to_words找到包含('s', 't')这个pair的word有：

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


可以发现'wi','id'和'de'这些pair的频次被减掉又加回去，这就带来没有必要的更新，更糟糕的是，比如'wi'删除后频率为0，那么就需要从pair_counts和pair_to_words里删除。

```
                    if pair_counts[old_pair] <= 0:
                        # we accounted for all occurrences of this pair
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair)
```

根据[Internals of sets and dicts](https://www.fluentpython.com/extra/internals-of-sets-and-dicts/)，CPython的dict使用的是开放地址的冲突解决方法。从dict里删除一个key只是做一个记号(除非rehash)，然后再增加同样的key会复用原来的地址。所以对dict本身的影响不大，但是key和value还是需要重新构造对象(dict里保存的是key和value的指针)。如果是std::unordered_map这样的分离链接（Separate Chaining）方法，那么删除再插入就会把链表原来的顺序打乱(链表是插入在表头)，比如某个桶原来的元素是a->b->c，现在把b删除再插入，新的链表就可能变成b->a->c。

但是如果我们仔细分析，我们就能知道哪些删除是不必要的。比如我们的例子，我们合并的是('s','t')，而在'widest'，受到合并影响的除了('s','t')之外，只有这个pair之前和之后的token，因此'widest'里的pair受到影响的只有['es','st']('t'后面没有pair了)，而['wi','id','de']不受影响，因此我们只需要删除'es','st'，然后增加'est'即可。

## 2. 细粒度的频率更新算法

根据前面的分析，我们可以实现细粒度的频率更新算法。前面分析的例子是一个要合并的pair在word里只出现一次的情况。但是如果要合并的pair出现多次呢？一种办法是fallback到之前的算法。因为一个word重复出现pair的可能性不大。如果只出现一次，那么可以比较简单的处理；如果出现多次，则fallback到之前的粗粒度的更新算法——先删除再插入。不过我在写文章之前没有想到这种方法，而是把问题搞得比较复杂。弄了一个复杂的算法能够处理pair在一个word里重复出现的算法。虽然这个算法看起来比较完美，但是实现的难度大了很多，而且收益也没有那么大。不过既然已经实现了，那么就继续吧。

比如一个假想的词"abcststefstbbef"，我们首先可以找到所有出现st的下标，为了看起来简单，我们用空格把st与其它token分开。

```
abc st st ef st bbef
```

根据前面的分析，第一个st之前的pair('ab','bc')不受影响，最后一个st之后的pair('bb')也不受影响。那么中间的pair呢？我们先写出来：

```
cs st ts st te ef fs st tb
```
这些pair首先得删除，然后我们把受到影响的部分生成新的token：

```
c st st e f st b
```

我们再加入新的pair：

```
('c','st'), ('st','st'), ('st','e'), ('e','f'), ('f','st'), ('st','b')
```

除了减少受影响的老pair的频次计数，增加新pair的计数，我们还需要修改倒排索引pair_to_words。我最早写的代码是把中间受影响的pair从中删除。但是这是有bug的。比如ef会受到影响，因此需要减少计数。但是ef除了在3个st的中间出现，它还出现在最后。如果我们直接从pair_to_words里删除ef，那么倒排索引就和实际不一致了。"真正"的倒排索引不仅仅记录pair('e','f')在词'abcststefstbbef'里出现了，而且还需要记录出现的次数。也就是说pair_to_words的key是pair，value也是dict，value这个dict是word/pair_freq。而我们现在的pair_to_words的value是一个set。因此我们只知道pair('e','f')在词'abcststefstbbef'里出现了，但是不知道出现几次。这就出现问题了。比如上面的例子我们之间从pair_to_words[('e','f')]里删除词'abcststefstbbef'，那么倒排索引就不对了，但是我们又必须从pair_to_words[('t','e')]的value里删除'abcststefstbbef'，因为在单词'abcststefstbbef'里pair('t','e')只出现一次。

一种解决办法就是修改pair_to_words相关代码，把它变成{pair: {word: word_freq}}这样的结构。这是比较清晰的改动方法。不过我当时觉得这么改需要修改的地方太多，我想把修改局限在一个函数里。因此我使用了另外一个办法(现在仔细思考其实并不太好)。这个办法是：找到不受影响的所有pair，把它放到一个set里。在修改倒排索引时如果发现要删除的pair在不受影响的pair也出现，那么不删除。从逻辑的角度这个算法也是正确的，但是它使得代码变得复杂。

不过总体来说扫描一遍不受影响的pair速度也比较快，而且后面的很多改进版本都基于当前算法，要改动比较麻烦。所以在写这篇文章的时候我就暂时不修改了。

## 3. 代码实现

完整的代码在[bpe_v5.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v5.py)。我们看一下改动的地方。首先是对_updated_affected_word_count的修改：

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

它调用一个新的函数BPE_Trainer._updated_affected_word_count变得非常简单，它调用一个新的函数BPE_Trainer.fine_grained_pair_counter_diff来计算合并pair后pair计数的变化。这个函数的主要输出是diff_pairs，比如假想的一个输出：

```
('a','s'): -3
('t','f'): -5
('st','b'):10
```

则它说明由于合并('s','t')，导致老的pair('a','s')的计数减少了3，('t','f')减少了5；而新的pair('st','b')增加了10。所以下面的for循环就用这个diff去更新pair_counts。另外BPE_Trainer.fine_grained_pair_counter_diff函数也会返回合并后新增的new_pairs，这些新出现的pair，我们也需要更新pair_strings。下面我们重点来看BPE_Trainer.fine_grained_pair_counter_diff函数。


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

这段代码比较复杂，尤其是下标的处理。我们分段来仔细看一下。

**找到合并pair的第一次和最后一次出现**

我们首先需要找到pair的第一次和最后一次出现的位置，如果只有一次，那么这两个位置是相同的。pair第一次出现的第一个元素和第二次出现第二个元素就是我们关注的范围。

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

第一个while循环找到pair出现的第一次位置，记录在first_idx里。else是不应该运行的(除非我们的倒排索引出现问题)，这是防御性的抛出异常。第二个while循环反过来从最后的位置开始找，记录在last_idx里，如果找不到则说明这个pair只出现了一次，则last_idx==first_idx。


用之前的例子来分析，我们找到的first_idx是3，last_idx是9。如下所示：

```
"abcststefstbbef"
 012345678901234
    |     | 
first_idx |
       last_idx  
```

接下来是找到所有不受影响的pair：

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

这里的下标计算特别要小心，我就不细讲了，总之最终的结果是找到如下不受影响的pair保存的unaffected_pairs里：

```
ab bc bb be ef
```

接下来就是减少受影响的pair的频次和更新倒排索引：
```
            affected_tokens = word_tokens[start_idx: end_idx]
            for i in range(len(affected_tokens) - 1):
                old_pair = (affected_tokens[i], affected_tokens[i + 1])
                diff_pairs[old_pair] -= wc 
                if old_pair not in unaffected_pairs:   
                    pair_to_words[old_pair].discard(word)
```

比如我们的例子，假设wc是10，则会执行：
```
diff_pairs[('c','s')] = -10
pair_to_words[('c','s')].discard(word)
```

而对于('e','f')，pair_to_words[('e','f')].discard(word)不会执行，因为('e','f')在unaffected_pairs里。

接下来是计算新的tokens：

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

注意：之前我们只有一个new_tokens，但是现在我们还有一个all_new_tokens。new_tokens是计算受影响的那部分，而all_new_tokens是整个word。因此对于我们之前"abcststefstbbef"的例子，它们的结果是：

```
new_tokens = ['c','st','st','e','f','st','b']
all_new_tokens = ['a','b'] + new_tokens + ['b','e','f']
```

最后是计算新的pair的频次：

```
            word_encodings[word] = all_new_tokens

            # add new pairs from the updated word
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])

                diff_pairs[new_pair] += wc
                pair_to_words[new_pair].add(word)

                new_pairs.add(new_pair)
```

第一行是更新word_encodings为all_new_tokens。比如前面的例子：
```
word_encodings['abcststefstbbef'] = ['a','b', 'c','st','st','e','f','st','b', 'b','e','f']
```

接着是找到new_tokens里的所有新pair：

```
('c','st')
('st','st')
('st','e')
('e','f')
('f','st')
('st','b')
```

这里所谓的新并不一定就是之前没有出现的pair，比如('e','f')之前其实就有，只不过被删除又加进来，因此可能出现diff_pairs[('e','f')]==0的情况，这也是前面代码会判断diff_pairs是否为0，如果为0，则说明它并没有改变。但是上面的代码会在前面diff_pairs[old_pair] -= wc先删除('e','f')这个pair，后面又再加进来。不过这种情况只有在一个word里包含pair两次以上才会出现。如果pair只出现一次，比如单词是'abcstbbef'，那么被删除的只有[('c','s'), ('s','t'), ('t','b')]，这些如果被删除就不可能再被加进来。


## 4. 测试

测试的代码是[bpe_v5_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v5_time.py)。

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v1_time | tinystory |2187/2264/2247|622/642/628| count_pair:944/995/984 max:167/173/174/174 update:453/453/460 |
bpe_v2_time | tinystory |757/738/746|639/621/627| 118/117/118 |
bpe_v2_time | openweb |35358/34265/35687|2870/2949/2930| 32437/31264/32708 |
bpe_v3_time | tinystory |250/90/90|80/90/90| 170/0/0 | num_counter=8, num_merger=1 skip
bpe_v3_time | openweb |391/401/32352|390/400/410| total:31883 max:31187 update:695 | num_counter=8, num_merger=1 skip
bpe_v5_time | tinystory |221/236/222 | 90/90/90 | total:130/146/132 max:127/143/129 update:3/3/3 |num_counter=8, num_merger=1
bpe_v5_time | openweb |34333/34853/35804 | 401/390/401 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 |num_counter=8, num_merger=1

和bpe_v3_time相比，使用细粒度的更新在openweb上的时间从695秒下降到500多秒。

## 5. 移植到c++

同样的，我把这个细粒度更新的算法移植到了c++的版本，包括[bpe_train_updater_fine_grained](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained.cpp)、[bpe_train_updater_fine_grained_absl](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_absl.cpp)和[bpe_train_updater_fine_grained_emhash8](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8.cpp)。

代码就不详细介绍，感兴趣的读者可以对照Python的版本阅读c++版本的代码。

这些版本的实验结果为：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v2_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe_train_updater_omp_v2_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe_train_updater_opt_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637
bpe_train_updater_emhash8 | Boost hash | 479/485/485 | 398/401/401 | 80/83/83
bpe_train_updater_opt_emhash8 | Boost hash | 469/474/479 | 389/395/399 |79/78/79
bpe_train_updater_opt_emhash8_hash| my hash | 2316/1951/1983 | 2250/1888/1918 |66/63/64
bpe_train_updater_fine_grained | Boost Hash | 8773/8873/7641 | 220/219/233 |8552/8653/7408
bpe_train_updater_fine_grained_absl | Absl hash | 845/845/856 | 204/201/203 | 641/643/653
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60

使用了细粒度更新后，bpe_train_updater_fine_grained_emhash8的总时间只有**260秒**，其中更新时间只有200秒。对比bpe_train_updater_emhash8，它的总时间是480秒，update时间接近400秒，max时间79秒。这说明细粒度更新减少了不必要的删除和插入，因此update时间变为原来的一半，而且这些不必要的删除和插入也会让数据结构变得混乱，因此max的时间也有少量减少。

## 6. 使用emhash8::HashSet/emhash9::HashSet替代std::unordered_set

除了pair_counts，还有一个比较频繁更新的就是倒排索引pair_wordids:

```
std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> pair_wordids;
```

这里除了外面的std::unordered_map，value本身也是一个std::unordered_set<int>。我们可以用emhash8::HashSet/emhash9::HashSet替代std::unordered_set。完整的代码在[bpe_train_updater_fine_grained_emhash8_set.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8_set.cpp)和[bpe_train_updater_fine_grained_emhash8_set9.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8_set9.cpp)。

同时使用hash_set4.hpp(emhash9::HashSet) and hash_table8.hpp(emhash8::HashMap)在编译时会出现一些警告，不过根据[这个issue](https://github.com/ktprime/emhash/issues/67#issuecomment-3262163325)，我们可以忽略它们。实验结果如下：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v2_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe_train_updater_omp_v2_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe_train_updater_opt_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637
bpe_train_updater_emhash8 | Boost hash | 479/485/485 | 398/401/401 | 80/83/83
bpe_train_updater_opt_emhash8 | Boost hash | 469/474/479 | 389/395/399 |79/78/79
bpe_train_updater_opt_emhash8_hash| my hash | 2316/1951/1983 | 2250/1888/1918 |66/63/64
bpe_train_updater_fine_grained | Boost Hash | 8773/8873/7641 | 220/219/233 |8552/8653/7408
bpe_train_updater_fine_grained_absl | Absl hash | 845/845/856 | 204/201/203 | 641/643/653
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60
bpe_train_updater_fine_grained_emhash8_set | Boost Hash | 192/192/194 | 117/117/117 | 75/75/77
bpe_train_updater_fine_grained_emhash8_set9 | Boost Hash | 168/170/171 | 107/108/109 | 61/62/61

可以看到pair_wordids用更快的emhash8::HashMap/emhash8::HashSet/emhash9::HashSet替换会后，时间进一步从260多秒降到了170秒。


