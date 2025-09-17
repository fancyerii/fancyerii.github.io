---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第5部分：用C++实现Merge算法" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第六篇，用C++实现和Python等价的merge算法，并且比较std::unordered_map的两种遍历方式，为下问的openmp并行求max做准备。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 目标

本文的目标是用c++重写bpe tokenizer训练的第二步，实现和BPE_Trainer._merge_a_pair完全相同的功能。

## 2. 代码编译

需要支持c++17的c++编译器，请根据您的系统安装合适的版本。首先获取代码。从git clone后还需要拉取submodule的代码。

```
# 获取代码
git clone https://github.com/fancyerii/assignment1-basics-bpe.git
cd assignment1-basics-bpe
# 获取submodule代码
git submodule update --init
```

然后是编译：

```
cd cppupdate/
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
cmake --build .
# cmake --build . -- -j 8(make/ninja)
```
## 3. 从Python导出中间结果

因为我们的c++版本只是完成bpe tokenizer训练的第二步，而第一步BPE_Trainer._pretokenize_and_count_mp里那些正则表达式的处理c++里没有完全等价的实现。而且前面我们的bpe_v3使用32核是第一步的时间已经优化到只有120秒了(用bpe_v3_bytes_time在64核上可以优化到70秒)，这个时间已经非常短了。所以就没有必要让c++来做这一步的工作了。理想情况当然是用c++实现第二部分然后通过Python/C API或者SWIG、CFFI、Pybind11或者Cython来集成，后文会我们会把c++实现的第二步编译成一个动态库然后通过cython集成。不过在这之前我们先要验证c++的可行性，如果c++实现的也没有太快，那就没有必要开发集成的代码了。因此我们先把Python第一步的结果通过json的方式(这是最简单跨语言的数据传输格式，虽然性能不是最优，但是最简单)导出为文件，然后再由c++导入到内存继续第二步。

导出openweb的第一步结果的代码在[dump_openweb_step1_results.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/dump_openweb_step1_results.py)。

它最终调用的是[bpe_v3_step1_result.BPE_Trainer.dump_step1_results](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_step1_result.py#L12)，代码为是执行bpe_v3_step1的第一步，然后把执行变量保存为json：

* word_counts: str -> int 
* vocabulary: int -> bytes 
* word_encodings: str - > list[int]
* pair_counts: tuple[int,int] -> int
* pair_strings: tuple[int,int] -> tuple[bytes,bytes]
* pair_to_words: tuple[int,int] -> set[str]


其中word_counts、word_encodings和pair_to_words里的word是字符串，在c++里处理unicode比较麻烦，而且word在第二步算法里只是作为一个独立的实体，我们并不会对它进行处理(比如求substring)，所以我们可以把word转换成一个id。这样在c++里就不用处理字符串了。

因此我们把这6个变量变成7个：

* word_ids: str -> int word到id的映射，输出为文件word_ids.json
* wordid_counts: int -> int 把word_counts的key从str转换成word对应的id，结果文件为wordid_counts.json
* wordid_encodings: int -> list[int]，结果文件为wordid_encodings.json
* pair_strings_json: str -> list[list[int], list[int]] 由于json的key只能字符串，所有把tuple[int,int]变成用逗号分隔的字符串，另外bytes也必须变成list[int]才能保存为json。结果文件为pair_strings.json
* vocabulary_json: int -> list[int]，在json里bytes必须变成list[int]，输出为vocabulary.json
* pair_to_wordids: str -> list[int]，key从pair变成逗号分隔的字符串，value从set[str]变成list[int]。输出文件pair_to_wordids.json
* pair_counts_json: str -> int，key从pair变成逗号分隔的字符串。输出pair_counts.json

上面7个变量的处理过程代码如下：

```
word_ids = {word:id for id, word in enumerate(word_counts)}

wordid_counts = {word_ids[word]:count for word, count in word_counts.items()}

wordid_encodings = {word_ids[word]:encoding for word, encoding in word_encodings.items()} 

pair_strings_json = {}
for pair, string in pair_strings.items():
    key = ",".join(str(item) for item in pair)
    value = [list(item) for item in string]
    pair_strings_json[key] = value
            
vocabulary_json = {key:list(value) for key, value in vocabulary.items()}     

pair_to_wordids = {}
for pair, words in pair_to_words.items():
    key = ",".join(str(item) for item in pair)
    wordids = [word_ids[word] for word in words]
    pair_to_wordids[key] = wordids  
    
pair_counts_json = {}
for pair, count in pair_counts.items():
    key = ",".join(str(item) for item in pair)
    pair_counts_json[key] = count      
```

## 4. 使用c++实现merge

代码在[bpe_train_updater.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater.cpp)。

这个程序的用法为：

```
mkdir res_updater
./build/bpe_train_updater ./data/openwebdump 32000 res_updater > bpe_train_updater.log
```

第一个参数为输入目录，路径就是上一步的输出目录，第二个参数是最终merge后词典的大小，第三个参数是保存vocabulary和merges的输出目录。

下面我们简单看一下它的代码。

### 4.1 read_pair_counts

第一步就是从输入目录里读取除了word_ids.json之外的其它6个json文件，因为在这里word_ids.json并没有什么用处。代码都差不多，我们这里就看一下其中之一读取pair_counts.json的代码。读取json使用的是[nlohmann/json](https://github.com/nlohmann/json)，因为它只需要一个头文件，所以我就没有弄git submodule，而是直接把[这个头文件](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/json.hpp)下载下来了。

```
std::unordered_map<std::pair<int, int>, int, pair_hash> read_pair_counts(std::string dir){
    std::ifstream f(dir + "/pair_counts.json");
    json data = json::parse(f);
    std::unordered_map<std::pair<int, int>, int, pair_hash> pair_counts;
    for (auto const& [key, val] : data.items()) {
        auto const key_vec = split_string_to_ints(key, ",");
        std::pair<int, int> pair {key_vec[0], key_vec[1]};
        pair_counts[pair] = int(val);
    }
    return pair_counts;
}
```

代码首先用nlohmann/json读取和解析json文件，然后用它构建pair_counts。和python的dict对应的就是std::unordered_map。我们在python里key是tuple[int,int]，c++里我们用std::pair<int,int>来作为key。与python不同，我们需要指定key的函数函数。我们这里复制的是boost的hash函数：

```
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};
```

**注意**：在python里我们很少自己实现，除非我们自己定义一个类作为key，那就需要重载__hash__ 和 __eq__这两个方法。hash函数对于unordered_map的性能至关重要，我们后面会实现一个更简单的hash函数来对比性能。


和list对应的当然就是std::vector。另外和python的set对应的是std::unordered_set，读者可以看看read_pair_wordids函数：

```
std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> read_pair_wordids(std::string dir)
```

而pair_strings的value是tuple[list[int],list[int]]，我用std::vector<std::vector<int>>来实现。

```
std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> read_pair_strings(std::string dir)
```

### 4.2 bpe_train_step2

读取完这6个变量之后就可以调用这个函数完成merge的过程，这个函数的原型为：

```
void bpe_train_step2(int vocab_size, 
                std::unordered_map<std::pair<int, int>, int, pair_hash> & pair_counts, 
                std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> & pair_strings, 
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges);
```

除了6个变量作为输入(其中vocabulary还作为输出)，还有两个参数：vocab_size表示合并后的词典大小，merges是输出参数。wordid_counts在函数里是不会被修改的，所以作为const引用传入。其它变量都是会被修改的，所以没有const，但是这些变量都非常大，所以都是作为引用传入。

这个函数分为两个部分，我们先看第一部分求频率最高的pair相关代码：

```
        int max_count = -1;
        std::pair<int, int> max_pair;
        std::vector<std::vector<int>> max_strings;
        for(const auto& [pair, count] : pair_counts){
            if(count > max_count){
                max_count = count;
                max_pair = pair;
                max_strings = pair_strings[pair];
            }else if(count == max_count){
                std::vector<std::vector<int>> strings = pair_strings[pair];
                ComparisonResult r1 = three_way_compare(strings[0], max_strings[0]);
                if(r1 == ComparisonResult::Greater){
                    max_count = count;
                    max_pair = pair;
                    max_strings = strings;
                }else if(r1 == ComparisonResult::Equal){
                    ComparisonResult r2 = three_way_compare(strings[1], max_strings[1]);
                    if(r2 == ComparisonResult::Greater){
                        max_count = count;
                        max_pair = pair;
                        max_strings = strings;                        
                    }
                }
            }
        }
```

代码和逻辑与Python完全相同，遍历pair_counts，如果count大于当前最大max_count，则更新max_count、max_pair和max_strings。如果count等于max_count，则还要比较pair_strings，这里把pair_strings的比较封装成了一个函数three_way_compare：

```
// 自定义的三元比较结果枚举
enum class ComparisonResult {
    Less,
    Equal,
    Greater
};

// 实现高效的三元比较函数
ComparisonResult three_way_compare(const std::vector<int>& a, const std::vector<int>& b) {
    // 1. 获取较小的 vector 的大小
    size_t min_size = std::min(a.size(), b.size());

    // 2. 逐个元素进行比较
    for (size_t i = 0; i < min_size; ++i) {
        if (a[i] < b[i]) {
            return ComparisonResult::Less;
        }
        if (a[i] > b[i]) {
            return ComparisonResult::Greater;
        }
    }

    // 3. 如果所有共同元素都相同，则比较大小
    if (a.size() < b.size()) {
        return ComparisonResult::Less;
    }
    if (a.size() > b.size()) {
        return ComparisonResult::Greater;
    }

    // 4. 如果大小和所有元素都相同
    return ComparisonResult::Equal;
}
```

接下来就是更新相关的代码：

```
        const std::vector<int>& bytes1 = vocabulary[max_pair.first];
        const std::vector<int>& bytes2 = vocabulary[max_pair.second];
        std::vector<int> merge_bytes;
        merge_bytes.reserve(bytes1.size() + bytes2.size());
        merge_bytes.insert(merge_bytes.end(), bytes1.begin(), bytes1.end());
        merge_bytes.insert(merge_bytes.end(), bytes2.begin(), bytes2.end());

        int size = vocabulary.size();
        vocabulary[size] = merge_bytes;

        auto& affected_words = pair_wordids[max_pair];

        
        updated_affected_word_count(max_pair, affected_words, wordid_encodings, wordid_counts,
                                    pair_counts, pair_wordids, size, pair_strings, vocabulary);
  
        merges.push_back({bytes1, bytes2});
```

它首先更新vocabulary，这和python是类似的，只不过c++比python要啰嗦很多。最主要的是调用updated_affected_word_count：

```
void updated_affected_word_count(const std::pair<int, int>& merge_pair,
        const std::unordered_set<int>& affected_words, 
        std::unordered_map<int, std::vector<int>> & wordid_encodings, 
        const std::unordered_map<int, long long> & wordid_counts,
        std::unordered_map<std::pair<int, int>, int, pair_hash> & pair_counts, 
        std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids, 
        int new_id,
        std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> & pair_strings, 
        std::unordered_map<int, std::vector<int>> & vocabulary){
    std::unordered_set<int> affected_words_copy(affected_words);

    for(int wordid : affected_words_copy){
        const auto& word_tokens = wordid_encodings[wordid];
        auto& wc = wordid_counts.at(wordid);

        // do not depend on LICM
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

        int i = 0;
        std::vector<int> new_tokens;

        while(i < token_size){
            if( (i < token_size - 1) && 
                (word_tokens[i] == merge_pair.first) &&
                (word_tokens[i + 1] == merge_pair.second)){
                new_tokens.push_back(new_id);
                i += 2;
            }else{
                new_tokens.push_back(word_tokens[i]);
                ++i;
            }
        }
        const int new_tokens_size = new_tokens.size();
        
        
        for(int i = 0; i < new_tokens_size - 1; ++i){
            std::pair<int, int> new_pair(new_tokens[i], new_tokens[i + 1]);
            pair_counts[new_pair] += wc;
            pair_wordids[new_pair].insert(wordid);
            if (pair_strings.find(new_pair) == pair_strings.end()) {
                pair_strings[new_pair] = {vocabulary[new_pair.first], vocabulary[new_pair.second]};
            }
        }
        
        // because we need move new_tokens
        // we move it below
        wordid_encodings[wordid] = std::move(new_tokens);
    }   
}
```

这段代码和python的几乎也是一样，即使对c++不熟悉，对照python的代码应该也能理解。

## 5. bpe_train_updater性能测试

因为tinystory比较小，我们这里只测试openweb上的时间。测试脚本为：

```
#!/bin/bash

for i in {1..3}; do
  mkdir -p res_updater_$i
  ./build/bpe_train_updater ./data/openwebdump 32000 res_updater_${i} > updater_${i}.log
done
```

测试结果为：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_v3_time          | | 31883 | 695 | 31187 |
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |

我们对比c++版本和python版本，可以发现update的时间从695秒减少到400多秒，max的时间从30000多秒减少到7000秒左右。算法没有做任何更改，只是用c++重写了一遍，收益还是非常明显的。另外我们可以看到用时最多的还是max部分，因此我们准备用并行算法来加速。不过在开始这个事情之前，我们需要做一个测试。之前的Python版本，我们无法直接对dict进行并行遍历，所以只能先把它复制到一个list里。但是复制list的时间就接近max了，那做并行计算就没有意义了。那么在c++的版本里，复制是不是快一些呢？我们需要测试一下。如果c++里复制std::unordered_map到std::vector也是很慢，那么我们就必须基于std::unordered_map做并行遍历。

## 6. bpe_train_updater_omp_v1

bpe_train_updater_omp_v1就是测试复制的时间，完整代码在[bpe_train_updater_omp_v1](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v1.cpp)。我们只是看一下它和bpe_train_updater.cpp不同的地方：

```
struct MaxData {
    int count = -1;
    std::pair<int, int> pair;
    std::vector<std::vector<int>> strings;
};


        // copy pair_counts to a vector        
        auto copy_start = std::chrono::steady_clock::now(); 
        std::vector pair_counts_vector(pair_counts.begin(), pair_counts.end());
        auto copy_end = std::chrono::steady_clock::now();
        copy_time += (copy_end - copy_start);

        auto max_start = std::chrono::steady_clock::now();
        MaxData max;

        const auto counter_size = pair_counts_vector.size();
        for(int i = 0; i < counter_size; ++i){
            const auto& pair = pair_counts_vector[i].first;
            const auto& count = pair_counts_vector[i].second;

            if(count > max.count){
                max.count = count;
                max.pair = pair;
                max.strings = pair_strings[pair];
            }else if(count == max.count){
                std::vector<std::vector<int>> strings = pair_strings[pair];
                ComparisonResult r1 = three_way_compare(strings[0], max.strings[0]);
                if(r1 == ComparisonResult::Greater){
                    max.count = count;
                    max.pair = pair;
                    max.strings = strings;
                }else if(r1 == ComparisonResult::Equal){
                    ComparisonResult r2 = three_way_compare(strings[1], max.strings[1]);
                    if(r2 == ComparisonResult::Greater){
                        max.count = count;
                        max.pair = pair;
                        max.strings = strings;                        
                    }
                }
            }
        }
        auto max_end = std::chrono::steady_clock::now();
        max_time += (max_end - max_start); 
```

代码几乎相同，只不过为了简化，我们把如下三个变量封装到一个MaxData结构体里：
```        
        int max_count = -1;
        std::pair<int, int> max_pair;
        std::vector<std::vector<int>> max_strings;
```

修改的部分就是增加了从unordered_map到vector的复制：

```
std::vector pair_counts_vector(pair_counts.begin(), pair_counts.end());
```

然后用下标对vector进行遍历：

```
        for(int i = 0; i < counter_size; ++i){
            const auto& pair = pair_counts_vector[i].first;
            const auto& count = pair_counts_vector[i].second;
```

## 7. bpe_train_updater_omp_v1测试

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_v3_time          | | 31883 | 695 | 31187 |
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v1测试 | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731

我们可以看到，增加了复制之后，bpe_train_updater_omp_v1总的时间满了很多。虽然vector的遍历很快，但是光copy花的时间就比bpe_train_updater的总时间还长。

所以把unordered_map复制到vector再并行的方案不可行，我们必须要直接对unordered_map进行并行max。

## 8. unordered_map的bucket api

通过搜索，我发现unordered_map有一个bucket api，它可以使得我们实现类似于vector的方式遍历，从而实现并行求max。关于unordered_map的实现，读者可以参考[C++ Unordered Map Under the Hood](https://jbseg.medium.com/c-unordered-map-under-the-hood-9540cec4553a)。我这里简单的介绍一下这篇文章的内容。

**注意**：这篇文章只是介绍unordered_map的大致实现逻辑，具体标准库的实现和这里描述的可能不同。c++的生态和python不同，python虽然定义了Python语言规范，理论上谁都可以实现一个Python解释器，但是主流的实现只有CPython(后面我们也会尝试pypy，但是很多第三方库会存在问题)。而c++没有一个CPython那样占据绝对垄断的解释器(这一点Python和Java有点类似，垄断也有垄断的好处，那就是决策会比较快，不像c++标准出来很久了，很多编译器还不支持它们)，g++/clang/msvc都有其自己的地盘，还有icc等其它编译器适合特定场景。所以c++标准只定义了unordered_map的接口，而不同的库(比如g++用到的libstdc++或者clang/llvm的libc++)可以用不同的方式实现，它们的性能差别可能非常大。后面我们会分析libstdc++的实现。

unordered_map的原理可以用下图来描述：

<a>![](/img/bpe/1.png)</a>

hashcode相同的key对应的pair都放在同一个桶(bucket)里，每个非空桶指向一个链表，用来存储这些hashcode相同的pair。这里解决冲突的是链表法(Separate Chaining)，除次之外常见的实现还有开放寻址法（Open Addressing）。

通常的实现(比如libstdc++)会把所有pair都用一个链表串联起来，我们用迭代器遍历时其实是遍历这个链表。但是根据上图，我们也可以通过每一个桶遍历，因为每个桶都有一个指针指向对应的开始部分。因此std::unordered_map也提供了bucket接口：

* [bucket_count](https://en.cppreference.com/w/cpp/container/unordered_map/bucket_count.html)
    * 查询总共的bucket个数
* [begin/cbegin](https://en.cppreference.com/w/cpp/container/unordered_map/begin2.html)
    * 返回某个桶的开始迭代器
* [end/cend](https://en.cppreference.com/w/cpp/container/unordered_map/end2.html)
    * 返回某个桶的结束迭代器
    
利用这个迭代器我们就可以实现并行遍历，不同的线程同时访问不同的桶。

## 9. bpe_train_updater_omp_v3

bpe_train_updater_omp_v3实现的就是使用bucket接口对pair_counts进行遍历。我们期望这个遍历的速度相比迭代器遍历不会慢太多，这样我们并行遍历才有价值。完整代码在[bpe_train_updater_omp_v3](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3.cpp)，我们这里看一下它给bpe_train_step2带来的改动：

```
        size_t num_buckets = pair_counts.bucket_count();
        for (size_t i = 0; i < num_buckets; ++i) {
            for (auto it = pair_counts.begin(i); it != pair_counts.end(i); ++it) {
                const auto& pair = it->first;
                const auto& count = it->second;
            ....
            }
        }
```

代码首先遍历每一个桶，然后对每一个桶通过pair_counts.begin(i)和pair_counts.end(i)对桶进行遍历。这里其实最好用cbegin和cend，因为我们不会修改pair_counts。不过const接口对最终的性能应该不会有影响。

## 10. bpe_train_updater_omp_v3的测试

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_v3_time          | | 31883 | 695 | 31187 |
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v1测试 | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731
bpe_train_updater_omp_v3的测试 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707

出乎意料！通过bucket接口求max竟然比使用迭代器的bpe_train_updater还要快！这不符合逻辑啊，因为如果bucket接口的方式遍历比迭代器快，那么完全可以用bucket接口的方式实现迭代器。

为了解答这个疑惑，我又开始深入研究std::unordered_map在libstdc++的代码。

## 11. bpe_train_updater_omp_v3比bpe_train_updater快的原因是什么

不过在研究一个问题的为什么之前，我们还是要先问问是不是确实是这样的。毕竟上面的代码逻辑很复杂，不只是简单的遍历。为了确定这一点，我写了一个小程序来测试。完整代码在[test_unordered_map_traverse_speed.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/test_unordered_map_traverse_speed.cpp)，这段代码就是测试随机生成的unordered_map的两种遍历方式的时间。

这个程序的用法是：

```
usage: test_unordered_map_traverse_speed loop_count map_size [key_range, seed]
```

第一个参数是循环次数；第二个是随机生成的map的大小；第三个(可选)参数是生成的key(int)的范围，默认是1-10000，第四个(可选)参数是随机种子，默认是1234。下面是测试结果(g++-11/libstdc++.so.6.0.32，和之前服务器上测试版本稍有不同，服务器是g++-9/libstdc++.so.6.0.28)：

```
$ ./test_unordered_map_traverse_speed 1000 10000 100000 1234
loop_count: 1000, map_size: 10000
key_range: 100000, seed: 1234
iter_duration:   48ms
bucket_duration: 188ms

$ ./test_unordered_map_traverse_speed 1000 10000 100000 1234
loop_count: 1000, map_size: 10000
key_range: 100000, seed: 1234
iter_duration:   51ms
bucket_duration: 192ms

$ ./test_unordered_map_traverse_speed 1000 100000 100000 1234
loop_count: 1000, map_size: 100000
key_range: 100000, seed: 1234
iter_duration:   443ms
bucket_duration: 1143ms

$ ./test_unordered_map_traverse_speed 1000 1000000 10000000 1234
loop_count: 1000, map_size: 1000000
key_range: 10000000, seed: 1234
iter_duration:   43885ms
bucket_duration: 66023ms
```

这个测试和前面相反，使用迭代器遍历要比bucket接口快。这是符合逻辑的，因为bucket接口需要两重循环，而且存在bucket为空的情况。

所以现在又有一个新问题：bucket接口遍历和迭代器遍历得到的结果是完全相同的吗？如果结果不同，那么可能解释为什么bpe_train_updater_omp_v3比bpe_train_updater快(后文会研究)。如果结果相同，那就很难解释了。

为了探究这个问题，我写了一个简单的程序[test_map_iter.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/test_map_iter.cpp)：

```

int main(){
    std::mt19937 gen(1234); 


    std::uniform_int_distribution<> key_dist(1, 1000); 
    std::uniform_int_distribution<> val_dist(1, 10000); 

    std::unordered_map<int, int> random_map;
    random_map.reserve(20);

    const int num_elements = 10;
    for (int i = 0; i < num_elements; ++i) {
        int random_key = key_dist(gen);
        int random_value = val_dist(gen);
        random_map[random_key] = random_value;
        std::cout << "insert: " << random_key << "->" << random_value << std::endl;
    }


    std::cout << "生成的随机 unordered_map 包含 " << random_map.size() << " 个元素:" << std::endl;
    std::cout << "for遍历" << std::endl;
    for (const auto& pair : random_map) {
        std::cout << "\t" << pair.first << "," << pair.second << std::endl;
    }
    
    std::cout << "iterator" << std::endl;
    for (auto it = random_map.begin(); it != random_map.end(); ++it) {
        std::cout << "\t" << it->first << "," << it->second << std::endl;
    }
    
    std::cout << "bucket api" << std::endl;
    size_t num_buckets = random_map.bucket_count();
    for (size_t i = 0, j = 0; i < num_buckets; ++i) {
        std::cout << "bucket " << i << std::endl;
        for (auto it = random_map.begin(i); it != random_map.end(i); ++it) {
            std::cout << "\t" << it->first << "," << it->second << std::endl;
        }
    }
}
```

运行的结果为：

```
insert: 192->4977
insert: 623->8179
insert: 438->6122
insert: 786->7714
insert: 780->8607
insert: 273->1507
insert: 277->1986
insert: 802->8152
insert: 959->1589
insert: 876->1162
生成的随机 unordered_map 包含 10 个元素:
for遍历
	959,1589
	802,8152
	273,1507
	780,8607
	786,7714
	277,1986
	438,6122
	876,1162
	623,8179
	192,4977
iterator
	959,1589
	802,8152
	273,1507
	780,8607
	786,7714
	277,1986
	438,6122
	876,1162
	623,8179
	192,4977
bucket api
bucket 0
bucket 1
	277,1986
	438,6122
bucket 2
	876,1162
	623,8179
bucket 3
bucket 4
	786,7714
bucket 5
bucket 6
bucket 7
bucket 8
	192,4977
bucket 9
bucket 10
bucket 11
bucket 12
bucket 13
bucket 14
bucket 15
bucket 16
	959,1589
bucket 17
bucket 18
bucket 19
bucket 20
	802,8152
	273,1507
bucket 21
	780,8607
bucket 22
```

我们发现用range-based的for循环和迭代器完全一样(因为range-based循环只是个语法糖)，但是和bucket接口访问不同。这个稍微有些出乎我的意料，为了探究原因，我们需要分析unordered_map的源代码。

关于unordered_map的源代码分析，可以参考[C++那些事之彻底搞懂STL HashTable](https://zhuanlan.zhihu.com/p/644205339)。b站上也有视频[手把手共读HashTable，彻底搞懂C++ STL](https://www.bilibili.com/video/BV1o8411U7vy/?vd_source=bb6532dcd5b1d6b26125da900adb618e)。

这篇文章使用的是gcc-4.9.1/libstdc++-v3，和我用的版本有些差异，不过差别也不多。另外补充一下，对细节不感兴趣的读者可以跳过。因为libstdc++要考虑兼容性，所以它的代码相对libc++的可读性要差很多。

我这里只分析和迭代器相关的代码，建议读者用调试器调试test_map_iter.cpp(建议用vscode)，在运行过程中一边阅读代码一边查看断点里的变量要比直接阅读代码容易很多。

<a>![](/img/bpe/2.jpg)</a>

这个图和前面类似，只不过把所有元素都用链表串联起来了，这是在libstdc++里的真实实现，而且迭代器遍历就是利用了这个链表。我们如果仔细比较前面程序的输出，我们会发现迭代器遍历的顺序接近插入顺序的逆序(但不完全是)，这个其实就和链表的插入有关，后面我们会看到。

在阅读代码之前我们先来阅读代码的文档，我这里的路径是/usr/include/c++/11/bits/hashtable.h，读者可以根据g++的版本搜索类似的路径：

```
 *  Each _Hashtable data structure has:
   *
   *  - _Bucket[]       _M_buckets
   *  - _Hash_node_base _M_before_begin
   *  - size_type       _M_bucket_count
   *  - size_type       _M_element_count
   *
   *  with _Bucket being _Hash_node_base* and _Hash_node containing:
   *
   *  - _Hash_node*   _M_next
   *  - Tp            _M_value
   *  - size_t        _M_hash_code if cache_hash_code is true
   *
   *  In terms of Standard containers the hashtable is like the aggregation of:
   *
   *  - std::forward_list<_Node> containing the elements
   *  - std::vector<std::forward_list<_Node>::iterator> representing the buckets
   *
   *  The non-empty buckets contain the node before the first node in the
   *  bucket. This design makes it possible to implement something like a
   *  std::forward_list::insert_after on container insertion and
   *  std::forward_list::erase_after on container erase
   *  calls. _M_before_begin is equivalent to
   *  std::forward_list::before_begin. Empty buckets contain
   *  nullptr.  Note that one of the non-empty buckets contains
   *  &_M_before_begin which is not a dereferenceable node so the
   *  node pointer in a bucket shall never be dereferenced, only its
   *  next node can be.
   *
   *  Walking through a bucket's nodes requires a check on the hash code to
   *  see if each node is still in the bucket. Such a design assumes a
   *  quite efficient hash functor and is one of the reasons it is
   *  highly advisable to set __cache_hash_code to true.
   *
   *  The container iterators are simply built from nodes. This way
   *  incrementing the iterator is perfectly efficient independent of
   *  how many empty buckets there are in the container.
   *
   *  On insert we compute the element's hash code and use it to find the
   *  bucket index. If the element must be inserted in an empty bucket
   *  we add it at the beginning of the singly linked list and make the
   *  bucket point to _M_before_begin. The bucket that used to point to
   *  _M_before_begin, if any, is updated to point to its new before
   *  begin node.
   *
   *  On erase, the simple iterator design requires using the hash
   *  functor to get the index of the bucket to update. For this
   *  reason, when __cache_hash_code is set to false the hash functor must
   *  not throw and this is enforced by a static assertion.
   *
   *  Functionality is implemented by decomposition into base classes,
   *  where the derived _Hashtable class is used in _Map_base,
   *  _Insert, _Rehash_base, and _Equality base classes to access the
   *  "this" pointer. _Hashtable_base is used in the base classes as a
   *  non-recursive, fully-completed-type so that detailed nested type
   *  information, such as iterator type and node type, can be
   *  used. This is similar to the "Curiously Recurring Template
   *  Pattern" (CRTP) technique, but uses a reconstructed, not
   *  explicitly passed, template pattern.
```

_Hashtable主要包含4个成员[bits/hashtable.h](https://gcc.gnu.org/onlinedocs/gcc-11.5.0/libstdc++/api/a17518_source.html)第391行：

```
    private:
      __buckets_ptr		_M_buckets		= &_M_single_bucket;
      size_type			_M_bucket_count		= 1;
      __node_base		_M_before_begin;
      size_type			_M_element_count	= 0;
      _RehashPolicy		_M_rehash_policy;
```

_M_bucket_count存储的就是bucket的个数，而_M_element_count存储当前hashtable的实际元素个数。_M_buckets的每个元素都是_Hash_node_base的指针，而时间存储数据的是_Hash_node，它继承了_Hash_node_base和_Hash_node_value。


```
  template<typename _Value, bool _Cache_hash_code>
    struct _Hash_node
    : _Hash_node_base
    , _Hash_node_value<_Value, _Cache_hash_code>
    {
      _Hash_node*
      _M_next() const noexcept
      { return static_cast<_Hash_node*>(this->_M_nxt); }
    };
```

_Hash_node_base代码为：

```
  struct _Hash_node_base
  {
    _Hash_node_base* _M_nxt;

    _Hash_node_base() noexcept : _M_nxt() { }

    _Hash_node_base(_Hash_node_base* __next) noexcept : _M_nxt(__next) { }
  };
```

_Hash_node_base很简单，就是一个单链表，它唯一的数据就是指向下一个元素的指针_M_nxt。

_Hash_node_value是真正存放key/value pair的地方，有一个特化的参数bool _Cache_hash_code来表示是否缓存hashcode。
```
  template<typename _Value, bool _Cache_hash_code>
    struct _Hash_node_value
    : _Hash_node_value_base<_Value>
    , _Hash_node_code_cache<_Cache_hash_code>
    { };
```

不过我们暂时不关心_Hash_node_value，只需要知道从数据结构来看，一个hashtable其实就是一个单链表，然后用std::vector存储了N个桶，每个非空桶都指向单链表中的某个元素(其实是它之前的)，表示这个桶的第一个元素。好奇的读者可能会问，那么非空桶是不是还应该存储这个通过的最后一个元素或者这个桶的元素个数？我之前也以为会这样，但是g++的实现并没有这样，它既没有保存每个桶的大小，也没有保存桶最后一个元素的指针。这样当然节省了空间，但是它怎么知道下一个元素到底是不是属于当前桶的呢？这里并没有魔法，它使用的方法是用时间来换取空间。也就是说它会判断下一个元素的hashcode和第一个元素的hashcode是否相同：如果相同，则说明属于当前桶，否则就不是。所以如果hash函数的计算很简单快速，那么就可以用少量的计算换取存储空间的节省。但是如果hash函数的计算非常复杂，那么这种方式就不一定是最优的了。

虽然_Hashtable提供了这个特化的参数，但是std::unordered_map并不同打开这个参数，如果我们需要cache hashcode，我们可以自己实现一个类作为key，比如

```
#include <iostream>
#include <unordered_map>
#include <utility>
#include <functional>

class PairWithCachedHash {
public:
    // 构造函数：接收一个 pair，并立即计算和缓存哈希值
    PairWithCachedHash(const std::pair<int, int>& p)
        : data_(p), hash_code_(std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1)) {
        // 你可以根据需要使用更复杂的哈希组合
    }

    // 必须实现相等性比较
    bool operator==(const PairWithCachedHash& other) const {
        return data_ == other.data_;
    }

    // 获取缓存的哈希值
    size_t get_hash_code() const {
        return hash_code_;
    }

private:
    std::pair<int, int> data_;
    size_t hash_code_;
};
```

感兴趣的读者可以测试一下，这种用空间换时间是否可以让程序更快一点。

那么还是回到之前的问题，为什么迭代器的遍历是接近于插入顺序的逆序呢？原因在于这里的插入逻辑是：如果当前插入的元素所在的桶是空的，那么就把这个元素插入到单链表的表头(因为单链表只有一个头指针，要快速插入只能插入表头)，然后当前桶指向这个元素(的前一个)；否则直接把这个元素插入到当前桶的第一个元素之前。

我们用前面的例子时间的来跑一下，为了避免rehash，我特意把reserve(20)，次数桶的总数为23(大于20的最小素数)。

我们第一个插入的key是192，根据hash函数 192 % 23 = 8，它放在下标为8个桶，此时的数据结果为：

```
linkedlist -> 192
              /|\
               |
               |
              [8]
```

第二个key是623，对应的桶是623 % 23 = 2，把它插入链表的头部，得到：

```
linkedlist -> 623  ->  192
              /|\      /|\
               |        |
               |        |
              [2]      [8]
```

接着是438：

```
linkedlist -> 438 -> 623  ->  192
              /|\    /|\      /|\
               |      |        |
               |      |        |
              [1]    [2]      [8]
```

接着是786：

```
linkedlist  -> 786 -> 438 -> 623  ->  192
               /|\    /|\    /|\      /|\
                |      |      |        |
                |      |      |        |
               [4]    [1]    [2]      [8]
```

780:

```
linkedlist  -> 780 -> 786 -> 438 -> 623  ->  192
               /|\    /|\    /|\    /|\      /|\
                |      |      |      |        |
                |      |      |      |        |
               [21]   [4]    [1]    [2]      [8]
```

273：

```
linkedlist  -> 273  -> 780 -> 786 -> 438 -> 623  ->  192
               /|\     /|\    /|\    /|\    /|\      /|\
                |       |      |      |      |        |
                |       |      |      |      |        |
               [20]    [21]   [4]    [1]    [2]      [8]
```

277 % 23 = 1，所以它不是插入到表头而是插入到438的前面：

```
linkedlist  -> 273  -> 780 -> 786 -> 277 -> 438 -> 623  ->  192
               /|\     /|\    /|\    /|\           /|\      /|\
                |       |      |      |             |        |
                |       |      |      |             |        |
               [20]    [21]   [4]    [1]           [2]      [8]
```

其它的都是类似，我就不再赘述。最后的结果为：

```
linkedlist -> 959  -> 802  -> 273  -> 780 -> 786 -> 277 -> 438 ->  876  -> 623  ->  192
              /|\     /|\             /|\    /|\    /|\            /|\              /|\
               |       |               |      |      |              |                |
               |       |               |      |      |              |                |
              [16]    [20]            [21]   [4]    [1]            [2]              [8]
```

因此迭代器的遍历顺序就是链表的顺序：959->802->273->780->786->277->438->876->623->192。
而bucket的遍历顺序是按照桶的顺序：[1]277,438 -> [2]876,623 -> [4]786 -> [8]192 ->[16]959 ->[20]802,273 ->[21]780
 
可以发现迭代器的遍历顺序比较接近插入的逆序，而bucket的遍历比较随机，和插入顺序关系不大，只有同一个桶里的元素会逆序。bucket接口begin最终调用的代码为：

```
  template<typename _Key, typename _Value, typename _Alloc,
	   typename _ExtractKey, typename _Equal,
	   typename _Hash, typename _RangeHash, typename _Unused,
	   typename _RehashPolicy, typename _Traits>
    auto
    _Hashtable<_Key, _Value, _Alloc, _ExtractKey, _Equal,
	       _Hash, _RangeHash, _Unused, _RehashPolicy, _Traits>::
    _M_bucket_begin(size_type __bkt) const
    -> __node_ptr
    {
      __node_base_ptr __n = _M_buckets[__bkt];
      return __n ? static_cast<__node_ptr>(__n->_M_nxt) : nullptr;
    }
```

不用看那一坨模板，真正代码只有两行，第一个是根据参数__bkt拿到桶的之前那个元素的指针\_\_n，如果\_\_n非空，则返回它的下一个节点。

end()直接返回空指针(下面的第二个参数nullptr)：

```
      const_local_iterator
      end(size_type __bkt) const
      { return const_local_iterator(*this, nullptr, __bkt, _M_bucket_count); }
```

而迭代器的重载++运算符最终调用的是(：

```
      void
      _M_incr()
      {
        __node_iter_base::_M_incr();
        if (this->_M_cur)
          {
            std::size_t __bkt = this->_M_h()->_M_bucket_index(*this->_M_cur,
                                                              _M_bucket_count);
            if (__bkt != _M_bucket)
              this->_M_cur = nullptr;
          }
      }
```

__node_iter_base::_M_incr()的作用就是把链表往后走一个元素，然后调用_M_bucket_index计算当前元素所在的桶(需要重新计算hashcode)，如果这个元素的桶和_M_bucket不同，则说明当前桶已经没有了，直接返回nullptr。



现在知道了迭代器和bucket接口遍历的区别，那么怎么解释bpe_train_updater_omp_v3比bpe_train_updater快呢？

一个可能的猜测就是，由于Python的dict在导出成json时key的顺序是插入顺序(Python 3.6之后的特性)，在统计词频时虽然有一些随机性，但是高频词出现在第一次的频率更高一下。举个极端的例子，假设只有两个词，第一个词出现了99次，第二个出现了一次。虽然每个词出现的概率是随机的，但是第一次出现第一个词的概率是99%。后续增加词频不会修改key，因此第一个词导出到json时出现在第二个词的前面的概率是99%。

所以c++版本拿到的dict，频率高的出现在前面的概率更大。但是插入到c++的pair_counts(std::unordered_map)之后，频率高的key在遍历时反而出现在后面。求max的操作除了遍历pair_counts的key/value，还需要用它和当前最大的pair的比较大小，如果它比当前的大，那么就要更新。用不同的顺序遍历虽然比较的次数都是N(N是pair_counts的大小)，但是更新的次数是不同的。如果遍历的时候是从大到小排列的，第一个就是最终的最大值，那么只需要更新1次；而如果遍历时从小到大排列，那么需要更新N次。这个更新的代码是：

```
        for(const auto& [pair, count] : pair_counts){
            if(count > max_count){
                max_count = count;
                max_pair = pair;
                max_strings = pair_strings[pair];
            }
```

上面只是当前count大于max_count的情况，如果相等也是有机会更新的。虽然更新只是3条赋值语句，但是它除了使得执行的指令增加，还会让CPU的分支预测失败，这会让流水线失效，从而进一步影响性能。

所以使用迭代器迭代从max算法的角度是很不好的，而使用bucket接口迭代，它就比较随机了(当然也不是最好的，最好的应该是保留Python词典的遍历顺序)。

当然，上面只是我的猜测，而且读者可能会问，虽然一开始是倒过来排序的，但是如果pair_counts发生rehash，这个顺序又会倒过来，这样不就补偿掉了吗？确实，pair_counts在迭代过程中会不断插入删除和rehash，这个过程很难准确预测。

解答这个疑问最好的办法就是用实际代码来测试。所以我修改了一下bpe_train_updater.cpp和bpe_train_updater_omp_v3.cpp，得到[bpe_train_updater_max_debug.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_max_debug.cpp)和[bpe_train_updater_omp_v3_max_debug.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3_max_debug.cpp)。它们在原来代码里增加了更新max的次数统计：

```
        for(const auto& [pair, count] : pair_counts){
            if(count > max_count){
                max_count = count;
                max_pair = pair;
                max_strings = pair_strings[pair];
                ++max_change;
            }else if(count == max_count){
                std::vector<std::vector<int>> strings = pair_strings[pair];
                ComparisonResult r1 = three_way_compare(strings[0], max_strings[0]);
                if(r1 == ComparisonResult::Greater){
                    max_count = count;
                    max_pair = pair;
                    max_strings = strings;
                    ++max_change;
                }else if(r1 == ComparisonResult::Equal){
                    ComparisonResult r2 = three_way_compare(strings[1], max_strings[1]);
                    if(r2 == ComparisonResult::Greater){
                        max_count = count;
                        max_pair = pair;
                        max_strings = strings;   
                        ++max_change;                     
                    }
                }
            }
        }
```

我们再来测试一下，结果如下：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_v3_time          | | 31883 | 695 | 31187 |
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v1 | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731
bpe_train_updater_omp_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707
bpe_train_updater_max_debug | Boost hash | 7332/7321/7156 | 446/416/400 | 6885/6905/6755 | max_change: 599657
bpe_train_updater_omp_v3_max_debug | Boost hash | 5489/5885/6138 | 399/413/418 | 5089/5471/5720 | max_change: 390506

可以发现和我的猜测符合，使用bucket接口遍历时更新的次数390506次，远小于迭代器的599657。就像前面分析的，虽然bucket接口的遍历速度要慢一些，但是由于它更接近随机访问，而迭代器接口最大值更可能在后面，所以导致更新次数变多。总体来说，bucket接口反而更快一下。

既然是这样，我们能不能在插入c++的std::unordered_map之前先从小到大排一次序，这样插入之后迭代器的遍历顺序就是大致从大到小了呢？我觉得这个有空是可以实验一下的。不过因为我们后面会转向bucket接口的并行遍历，而且依赖于g++的某个特定feature并不太好。比如换一个编译器甚至换一个版本也许它的实现都不相同了呢，所以这样的优化只能算是一个trick。我们还是尽量使得我们的优化算法是通用的，而不是依赖这样特定编译器的feature。

## 12. hash函数的影响

本文写得比我预期都要长很多，主要原因是在写的过程中我才发现了bucket接口比迭代器接口还快。之前并没有太关注它，以为只是实验的随机差异。所以调研std::unordered_map的底层实现增加了很多篇幅。

不过最后我们还是要实验一下不同hash函数对unordered_map的影响。我最早其实用的是很简单的hash函数：

```
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 << 1); 
    }
};
```

这个是Gemini帮我生成的，我当时也没有仔细考虑，直接就用了。用这个hash函数的代码是[bpe_train_updater_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_hash.cpp)和[bpe_train_updater_omp_v3_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3_hash.cpp)，它们和对应的版本代码几乎完全一样，只是修改了hash函数。下面是实验结果：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_v3_time          | | 31883 | 695 | 31187 |
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_hash | my hash | 9875/11434/11222 | 2753/3182/2834 | 7122/8251/8388 |
bpe_train_updater_omp_v1 | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731
bpe_train_updater_omp_v1_hash | Boost hash | 17317/16883  | 2716/2686 | 81/80 | copy:14519/14115
bpe_train_updater_omp_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707
bpe_train_updater_omp_v3_hash | my hash | 12592/11240/12355 | 3184/2939/3215 | 9408/8301/9139 | bucket:4355707

可以看到，使用不好的hash函数之后，更新速度慢了非常多。所以一个好的hash函数对于unordered_map的性能是非常重要的。


