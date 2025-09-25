---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第7部分：使用flat hashmap替代std::unordered_map" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第八篇，使用flat hashmap来替代c++标准库的std::unordered_map来提高性能。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 问题分析

前文我们通过OpenMP的并行化，在32线程的情况下第二步merge的时间从6000多秒减少到了1000秒一下(bpe_train_updater_omp_v7)。这1000秒有500多秒是更新pair_counts等数据，另外的400多秒是求max。

在继续优化之前我们来回顾一下函数bpe_train_step2的主要数据读写，我们以串行版本的[bpe_train_updater_omp_v3.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3.cpp#L232)为例来分析。

求最大的pair主要代码是遍历pair_counts，只有在当前pair和当前最大pair的频次相同的时候才需要读取pair_strings[pair]。

更新pair_counts等变量的代码为：

```
        const int token_size = word_tokens.size();
        for(int i = 0; i < token_size - 1; ++i){
            std::pair<int, int> old_pair(word_tokens[i], word_tokens[i + 1]);
            pair_counts[old_pair] -= wc;
            if(pair_counts[old_pair] <= 0){
                pair_counts.erase(old_pair);
                pair_wordids.erase(old_pair);
            }else{
                pair_wordids[old_pair].erase(wordid);
            }
        }
        
        
        for(int i = 0; i < new_tokens_size - 1; ++i){
            std::pair<int, int> new_pair(new_tokens[i], new_tokens[i + 1]);
            pair_counts[new_pair] += wc;
            pair_wordids[new_pair].insert(wordid);
            if (pair_strings.find(new_pair) == pair_strings.end()) {
                pair_strings[new_pair] = {vocabulary[new_pair.first], vocabulary[new_pair.second]};
            }
        }
```

这里最主要操作的就是写操作：pair_counts[old_pair] -= wc; pair_counts[new_pair] += wc;此外就是在pair_counts[old_pair] <= 0时从pair_counts里删除old_pair，从pair_wordids里删除pair_wordids。

另外有一个读的操作if(pair_counts[old_pair] <= 0)。这个操作其实可以优化掉，下面我们再讨论。而且即使不优化，它所用的时间也是非常短的，一来unordered_map的查询比较快(更新则需要申请内存，插入链表和桶，并且可能导致rehash)；二来这个查询是紧跟在更新之后的，相关的数据应该还在cpu的cache里。

首先来看pair_counts的遍历，如果是串行版本，我们是通过unordered_map内部的单链表进行遍历；如果是并行版本，其实也是遍历链表的内容，只不过是通过bucket接口分块遍历。从实现来说，unordered_map是链表，它能够以O(1)的时间复杂度插入删除元素。但是我们的主要写操作是更新key对应的value。而链表的内存布局是非常零碎的，它的遍历比连续的内存布局要慢很多。因此对于我们的场景可以使用连续布局的hashmap，也就是所谓的flat hashmap。其主要特点是它将所有数据存储在一个连续的内存块中，而不是像标准哈希表那样使用链表或指针将数据分散在不同的内存位置。

一个标准的哈希表（例如 C++ 的 std::unordered_map）通常通过一个哈希函数将键（key）映射到不同的“桶”（bucket），每个桶里可能是一个链表，用于存放哈希值相同的元素。这导致元素在内存中是分散、不连续的。

而 flat hashmap 则完全不同，它的核心工作原理是：

* 连续内存布局: 它使用一个大数组来存储所有的键值对。由于数据是连续存放的，这极大地提高了缓存友好性（Cache-Friendly）。当处理器访问一个元素时，它通常会把周围的元素也一起加载到高速缓存中，这使得后续的访问变得非常快。
* 开放寻址（Open Addressing）: 为了处理哈希冲突（即两个不同的键哈希到同一个位置），flat hash map 不会使用链表。相反，它采用一种称为**探查（Probing）**的技术。当一个位置已经被占用时，它会按预定的规则（比如线性地、二次地）寻找下一个可用的空槽来存放数据。


**优点和缺点**

* 优点
    * 极佳的缓存性能：这是它最大的优势。在现代 CPU 中，缓存访问速度比内存访问快得多。由于数据是连续的，flat hashmap 的访问模式非常适合 CPU 缓存，尤其是在需要遍历或批量查找时。
    * 低内存开销：由于不需要为每个元素存储额外的指针，它的内存占用更小，特别是在存储大量小元素时。
    * 适用于高性能场景：在游戏开发、网络编程、高性能计算等对延迟要求极高的领域，flat hashmap 是一个理想的选择。
* 缺点
    * 删除复杂：由于探查机制的存在，删除元素不能简单地移除，需要用一个特殊的“墓碑”标记来确保后续查找操作的正确性。这可能会导致数组中出现碎片化，需要定期进行清理（重新哈希）。
    * 聚集问题（Clustering）：如果哈希函数不理想，或者冲突频繁，可能会导致数据在数组中聚集，形成“块”，这会使得探查过程变长，从而降低性能。
    * rehash开销大：当底层数组需要扩容时，需要分配一个更大的新数组，然后将所有元素复制过去，这个过程的开销相对较高。

不过对于我们的场景来说，flat hashmap是非常合适的：我们的主要操作是遍历；删除不多。

## 2. 对-=的优化

在用flat hashmap替代std::unordered_map之前，我们顺手做一个小优化来省去一次pair_counts的查询。

c++的unordered_map的-=是一个表达式(expression)，它会返回-=之后的值；而Python的dict(defaultdict)的-=则是一个语句(statement)，所以是没有返回值的。比如在c++里下面的代码是可以编译的：

```
int x = 1;
int y = (x += 1);
```

但是类似的代码：

```
x = 1
y = (x += 1)
```

会出现语法错误：

```
    y = (x += 1)
           ^
SyntaxError: invalid syntax
```

所以在Python里，我们只能在更新pair_counts[old_pair] -= wc之后再来一次查询。但是在c++里，我们可以直接获得-=之后的值。

我们可以把这个优化应用到所有的版本里(比如bpe_train_updater_omp_v3/bpe_train_updater_omp_v5/bpe_train_updater_omp_v7等等)，但由于这个优化是在上一个版本之后才发现的，所以我们先在最初版本的bpe_train_updater.cpp进行比较。如果有效就把它应用到最终的版本里。完整的代码在[bpe_train_updater_opt.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt.cpp#L165)，只有一处改动：

```
            int& c = (pair_counts[old_pair] -= wc);
            if(c <= 0){
                pair_counts.erase(old_pair);
                pair_wordids.erase(old_pair);
            }else{
                pair_wordids[old_pair].erase(wordid);
            }
```

测试对比如下：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_opt | Boost hash | 8153/7634/7362 | 411/396/389 |7741/7238/6973 |

我们比较update time，发现平均从450s下降到398.67，提升了11.41%。虽然对于整体来说不是很大，但是积少成多，这样的优化还是不错的。

## 3. flat hashmap调研

前面我们介绍了flat hashmap的原理，并且认为它更适合我们的场景。我们当然没有必要重复造轮子，所以接下来就是调研一下有哪些开源可用的flat hashmap库。通过搜索，发现有很多第三方库，[Comprehensive C++ Hashmap Benchmarks 2022](https://martin.ankerl.com/2022/08/27/hashmap-bench-01/)这篇文章测试了场景的c++ hashmap库，其中有不少是flat的hashmap。

根据这篇文章的测试总结，我尝试了absl::flat_hash_map和emhash8::HashMap。


## 4. absl::flat_hash_map

首先尝试它的原因主要是网上很多文章推荐，而且是google开源的，据说google内部都在使用，那还不够唬人的啊。

absl::flat_hash_map是[abseil-cpp](https://github.com/abseil/abseil-cpp)这个库的一部分，要使用absl::flat_hash_map就得牵连到一大堆东西，这好像也是google开源库的问题。我只需要用一个很小的功能，却因为它是整个大库的一部分，所以需要引入一大堆依赖。当然这个锅也不能完全由google来背，这主要是c++的问题。和Python/Java不同，c++没有跨平台的abi，这就意味着要使用c++的库基本上只能重新编译全部的依赖。而且c++早期没有maven/pypi这样的中央仓库(现在也有conan之类的，但是用的也不多)，每个库都得自己编译太麻烦，所以很多人就干脆用自己的代码或者复制别人的代码后进行魔改(这样就造成了分支，无法更新)。但是一个成熟的库的依赖是非常很多的，比如你写任何一个代码可能都需要log吧，你就需要引入一个依赖；你需要解析命令行参数吧，你要引入一个；你需要字符串处理吧，但是c++标准库的字符串没法用，你得自己造一个或者依赖一个吧。你需要正则表达式吧，你需要xml/json/yaml解析吧。这些c++标准库都没有。而相反，在Python/Java里要么标准库有了，要么经过几轮竞争都收敛到少数几个获胜的开源库了，这样整个生态系统就比较统一了。但是c++就没法收敛，除了c++标准库，每个公司都得自己造一堆轮子，有些公司觉得标准库都不好用，还得再造一遍。这样一来它们一旦要开源就是几乎开源一整套，没有办法把某个小功能剥离出来。

当然也有一些非常小的库，通常它们只有头文件，这样的库直接复制到自己的项目里就行了。即使这样，因为它们没有pypi/maven这样版本管理，你一旦使用它就基本上不会(也不敢)升级了。所以它只适合那种一旦发布就永远不修改的库了。但是再小的库也需要fix bug啊，你总不能自己diff打patch吧。

闲话少数，我们首先在代码引入absl::flat_hash_map，这需要abseil-cpp的完整代码。前面我们已经通过git submodule获得了代码，因此只需要修改CMakeLists.txt：

```
add_subdirectory(abseil-cpp)

target_link_libraries(bpe_train_updater_omp_v2 PRIVATE absl::flat_hash_map absl::hash)
```

然后在使用的地方包含头文件即可：

```
#include "absl/container/flat_hash_map.h"
```

absl::flat_hash_map的接口和std::unordered_map类似，只是没有bucket接口，所以无法通过bucket接口进行并行遍历。

使用absl::flat_hash_map的完整代码在[bpe_train_updater_omp_v2.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v2.cpp)。代码和bpe_train_updater.cpp几乎完全相同，只不过把std::ordered_map改成了absl::flat_hash_map。

原来编译cppupdate只需要几十秒，引入这个依赖后用8线程编译也需要几分钟。

## 5. absl::flat_hash_map测试

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v2 | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |

我们发现和bpe_train_updater相比，bpe_train_updater_omp_v2总的时间从8000秒降到了2000多秒。而且max time只有不到300秒，这比bpe_train_updater的7000秒相比快了20多倍，比32线程的bpe_train_updater_omp_v7还快。这说明flat hashmap的连续内存布局确实很适合遍历。但是update time从400多秒增加到了2000秒，这似乎不符合网上大家的测试。

我们前面说过，hashmap的hash函数非常重要。absl::flat_hash_map自带了hash函数，我们需要测试一下，另外我们之前也实现了非常简单的hash函数，我们也可以对比一下。使用简单hash的代码在[bpe_train_updater_omp_v2_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v2_hash.cpp)，使用absl自带hash的代码在[bpe_train_updater_omp_v2_hash2.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v2_hash2.cpp)。另外，我们也把之前的-=的优化加入到[bpe_train_updater_opt_absl](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt_absl.cpp)。

使用absl自带hash非常简单，不指定hash函数就会用自带的：

```
absl::flat_hash_map<std::pair<int, int>, int> pair_counts;
```

测试结果如下：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v2_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe_train_updater_omp_v2 | my hash | 6276/6349/6613 | 6030/6104/6364 | 245/245/249 |
bpe_train_updater_omp_v2_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe_train_updater_opt_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637

使用简单的hash(my hash)时，absl::flat_hash_map的max time也是200多秒，但是update的时间增加到6000多秒。而使用absl自带的hash函数时，max time增加到了600多秒(但是仍然远远小于bpe_train_updater的7000多秒)，但是update time只有500多秒。这个版本的效果非常好，甚至超过了bpe_train_updater_omp_v7在16线程的速度。而bpe_train_updater_opt_absl优化了-=，update时间减少到400秒左右。

## 6. emhash8::HashMap

关于emhash的原理可以参考[这个issue](https://github.com/ktprime/emhash/issues/33#issuecomment-1636618464)，我也没有深入研究，感兴趣的读者可以自行探索或者去项目的github提问。

[emhash](https://github.com/ktprime/emhash)相较absl::flat_hash_map就轻巧多了，我们用到的只是emhash8，因此只需要引入头文件就行，在CMakeLists.txt里增加target_include_directories即可：

```
target_include_directories(bpe_train_updater_emhash8 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)
```

修改也很简单，把std::unordered_map改成emhash8::HashMap就行，完整的代码在[bpe_train_updater_emhash8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_emhash8.cpp)。

## 7. emhash8::HashMap测试结果

为了对比，我们也实现了-=优化版本的[bpe_train_updater_opt_emhash8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt_emhash8.cpp)和-=优化版本+自定义hash函数的[bpe_train_updater_opt_emhash8_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt_emhash8_hash.cpp)。

结果如下：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v2_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe_train_updater_omp_v2 | my hash | 6276/6349/6613 | 6030/6104/6364 | 245/245/249 |
bpe_train_updater_omp_v2_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe_train_updater_opt_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637
bpe_train_updater_emhash8 | Boost hash | 479/485/485 | 398/401/401 | 80/83/83
bpe_train_updater_opt_emhash8 | Boost hash | 469/474/479 | 389/395/399 |79/78/79
bpe_train_updater_opt_emhash8_hash| my hash | 2316/1951/1983 | 2250/1888/1918 |66/63/64

emhash8::HashMap的速度比basl::flat_hash_map更快，总时长不到500秒。其中max时间更是小于100秒。使用-=优化还能快个十来秒。同时我们看到，如果使用简单的hash函数，它的性能也会下降。


## 8. 小结

使用了flat hashmap之后，max的时间急剧减少，这说明flat的连续内存结构确实适合我们的场景。这样一来，我们都不需要并行优化就达到了非常快的速度。所以这里我们学到的一课就是优化数据结构和算法有时比并行优化的效果更好。我们利用更少的资源做了同样的事情，而且做得更快。

至此，我们用c++优化第二步的工作就暂时告一段落，后面我们再次回到python的去优化。

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
