---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 5: Implementing the Merge Algorithm in C++" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the sixth article in the series. It focuses on implementing the merge algorithm in C++ to be equivalent to Python's version. We also compare two traversal methods for `std::unordered_map` in preparation for a discussion on parallel maximum computation with OpenMP.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Goal

The goal of this article is to rewrite the second step of BPE tokenizer training in C++, achieving the exact same functionality as `BPE_Trainer._merge_a_pair`.

## 2\. Code Compilation

You will need a C++ compiler that supports C++17. Please install the appropriate version for your system. First, get the code. After cloning from git, you also need to fetch the submodule's code.

```
# Get the code
git clone https://github.com/fancyerii/assignment1-basics-bpe.git
cd assignment1-basics-bpe
# Get the submodule code
git submodule update --init
```

Next, compile:

```
cd cppupdate/
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
cmake --build .
# cmake --build . -- -j 8(make/ninja)
```



## 3\. Exporting Intermediate Results from Python

Since our C++ version only completes the second step of BPE tokenizer training, and the first step `BPE_Trainer._pretokenize_and_count_mp` involves regular expression processing that doesn't have a perfectly equivalent implementation in C++. Furthermore, we have already optimized the time for the first step with our `bpe_v3` to only 120 seconds using 32 cores (or 70 seconds on 64 cores with `bpe_v3_bytes_time`), which is already very fast. Therefore, there's no need for C++ to handle this step. The ideal scenario would be to implement the second part in C++ and integrate it using the Python/C API, SWIG, CFFI, Pybind11, or Cython. Later in this article, we'll compile the C++ implementation of the second step into a dynamic library and integrate it via Cython. But before that, we need to verify the feasibility of the C++ implementation. If the C++ version isn't significantly faster, there's no point in developing the integration code. So, we'll first export the results of the first step from Python to a file using JSON (the simplest cross-language data transfer format, although not the most performant, it's the easiest) and then import it into memory in C++ to continue with the second step.

The code to dump the first step's results for openweb is in [dump\_openweb\_step1\_results.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/dump_openweb_step1_results.py).

It ultimately calls [bpe\_v3\_step1\_result.BPE\_Trainer.dump\_step1\_results](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_step1_result.py#L12), which executes the first step of `bpe_v3_step1` and then saves the following variables as JSON:

  * `word_counts`: `str -> int`
  * `vocabulary`: `int -> bytes`
  * `word_encodings`: `str - > list[int]`
  * `pair_counts`: `tuple[int,int] -> int`
  * `pair_strings`: `tuple[int,int] -> tuple[bytes,bytes]`
  * `pair_to_words`: `tuple[int,int] -> set[str]`

The `word_counts`, `word_encodings`, and `pair_to_words` variables have string keys (`word`). Handling Unicode strings in C++ can be complicated, and since the `word` only acts as a unique entity in the second step's algorithm and we won't be processing it (e.g., finding substrings), we can convert the `word` into an ID. This way, we don't have to deal with strings in C++.

Therefore, we'll transform these six variables into seven:

  * `word_ids`: `str -> int` A mapping from `word` to `id`, saved as `word_ids.json`.
  * `wordid_counts`: `int -> int` Converts the keys of `word_counts` from strings to their corresponding IDs. The resulting file is `wordid_counts.json`.
  * `wordid_encodings`: `int -> list[int]`, The resulting file is `wordid_encodings.json`.
  * `pair_strings_json`: `str -> list[list[int], list[int]]` Since JSON keys can only be strings, we convert `tuple[int,int]` into a comma-separated string. Additionally, `bytes` must be converted to `list[int]` to be saved as JSON. The resulting file is `pair_strings.json`.
  * `vocabulary_json`: `int -> list[int]` In JSON, `bytes` must be a `list[int]`. Saved as `vocabulary.json`.
  * `pair_to_wordids`: `str -> list[int]` Keys are converted from `pair` to comma-separated strings, and values are converted from `set[str]` to `list[int]`. Saved as `pair_to_wordids.json`.
  * `pair_counts_json`: `str -> int` Keys are converted from `pair` to comma-separated strings. Saved as `pair_counts.json`.

The code to process these seven variables is as follows:

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



## 4\. Implementing the merge with C++

The code is located in [bpe\_train\_updater.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater.cpp).

The usage of this program is:

```
mkdir res_updater
./build/bpe_train_updater ./data/openwebdump 32000 res_updater > bpe_train_updater.log
```

The first argument is the input directory path (the output directory from the previous step). The second argument is the final vocabulary size after merging. The third argument is the output directory to save the vocabulary and merges.

Let's briefly look at the code.

### 4.1 `read_pair_counts`

The first step is to read the six JSON files (excluding `word_ids.json` as it's not needed here) from the input directory. The code for all of them is similar, so we'll just look at the `read_pair_counts.json` part. We're using [nlohmann/json](https://github.com/nlohmann/json) to read the JSON because it only requires a single header file, so I didn't use a git submodule and just downloaded [the header file](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/json.hpp) directly.

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

The code first reads and parses the JSON file using nlohmann/json, and then uses it to build `pair_counts`. The C++ equivalent of a Python dictionary is `std::unordered_map`. In Python, our key is `tuple[int,int]`, while in C++, we use `std::pair<int,int>`. Unlike Python, we need to specify a hash function for the key. We're using a copy of the Boost hash function here:

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

**Note**: In Python, we rarely implement our own hash functions unless we define a custom class as a key, which would require overloading the `__hash__` and `__eq__` methods. The hash function is crucial for `unordered_map` performance. We'll implement a simpler hash function later to compare performance.

The equivalent of a Python list is `std::vector`. The equivalent of a Python set is `std::unordered_set`. You can see this in the `read_pair_wordids` function:

```
std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> read_pair_wordids(std::string dir)
```

The value for `pair_strings` is `tuple[list[int],list[int]]`, which I implemented with `std::vector<std::vector<int>>`.

```
std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> read_pair_strings(std::string dir)
```

### 4.2 `bpe_train_step2`

After reading these six variables, we can call this function to perform the merge. The function prototype is:

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

In addition to the six variables as input (with `vocabulary` also being an output), there are two other parameters: `vocab_size` which is the size of the vocabulary after merging, and `merges` which is an output parameter. `wordid_counts` is not modified in the function, so it's passed as a `const` reference. The other variables are modified and are therefore passed as references without `const`. All of these variables are very large, so they are all passed by reference.

This function has two parts. Let's first look at the code for finding the most frequent pair:

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

The code and logic are exactly the same as in Python. We iterate through `pair_counts`. If `count` is greater than the current `max_count`, we update `max_count`, `max_pair`, and `max_strings`. If `count` is equal to `max_count`, we also compare `pair_strings`. Here, the comparison of `pair_strings` is encapsulated in a function called `three_way_compare`:

```
// Custom three-way comparison result enum
enum class ComparisonResult {
    Less,
    Equal,
    Greater
};

// Implements an efficient three-way comparison function
ComparisonResult three_way_compare(const std::vector<int>& a, const std::vector<int>& b) {
    // 1. Get the size of the smaller vector
    size_t min_size = std::min(a.size(), b.size());

    // 2. Compare elements one by one
    for (size_t i = 0; i < min_size; ++i) {
        if (a[i] < b[i]) {
            return ComparisonResult::Less;
        }
        if (a[i] > b[i]) {
            return ComparisonResult::Greater;
        }
    }

    // 3. If all common elements are the same, compare sizes
    if (a.size() < b.size()) {
        return ComparisonResult::Less;
    }
    if (a.size() > b.size()) {
        return ComparisonResult::Greater;
    }

    // 4. If sizes and all elements are the same
    return ComparisonResult::Equal;
}
```

Next is the code for updating the variables:

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

It first updates `vocabulary`, which is similar to Python, but C++ is a lot more verbose. The main call is to `updated_affected_word_count`:

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

This code is also almost identical to the Python version. Even if you're not familiar with C++, you should be able to understand it by comparing it to the Python code.



## 5\. `bpe_train_updater` Performance Test

Since tinystory is quite small, we'll only test the time on openweb. The test script is:

```
#!/bin/bash

for i in {1..3}; do
  mkdir -p res_updater_$i
  ./build/bpe_train_updater ./data/openwebdump 32000 res_updater_${i} > updater_${i}.log
done
```

The test results are:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bpe\_v3\_time | | 31883 | 695 | 31187 | |
| bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 | 6779/7376/8770 | |

Comparing the C++ version and the Python version, we can see that the update time decreased from 695 seconds to over 400 seconds, and the `max` time decreased from over 30,000 seconds to around 7,000 seconds. We made no changes to the algorithm; we just rewrote it in C++, and the benefits are very significant. We can also see that the `max` part still takes the most time, so we're going to use a parallel algorithm to speed it up. But before we start, we need to do a test. In the previous Python version, we couldn't directly iterate through a `dict` in parallel, so we had to first copy it into a `list`. But the time it takes to copy the list is almost as long as the `max` calculation itself, which makes parallel computing pointless. So, is copying faster in the C++ version? We need to test it. If copying `std::unordered_map` to `std::vector` in C++ is also slow, then we must perform parallel iteration directly on `std::unordered_map`.



## 6\. `bpe_train_updater_omp_v1`

`bpe_train_updater_omp_v1` is a test to measure the copy time. The full code is in [bpe\_train\_updater\_omp\_v1](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v1.cpp). We'll just look at the differences from `bpe_train_updater.cpp`:

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

The code is almost identical, except that to simplify things, we've encapsulated the following three variables into a `MaxData` struct:

```
        int max_count = -1;
        std::pair<int, int> max_pair;
        std::vector<std::vector<int>> max_strings;
```

The modified part is the addition of the copy from `unordered_map` to `vector`:

```
std::vector pair_counts_vector(pair_counts.begin(), pair_counts.end());
```

And then using an index to iterate through the `vector`:

```
        for(int i = 0; i < counter_size; ++i){
            const auto& pair = pair_counts_vector[i].first;
            const auto& count = pair_counts_vector[i].second;
```



## 7\. `bpe_train_updater_omp_v1` Test

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bpe\_v3\_time | | 31883 | 695 | 31187 | |
| bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 | 6779/7376/8770 | |
| bpe\_train\_updater\_omp\_v1 test | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731 |

We can see that after adding the copy, the total time for `bpe_train_updater_omp_v1` is much slower. Although `vector` iteration is very fast, the time spent on copying alone is longer than the total time for `bpe_train_updater`.

Therefore, the approach of copying `unordered_map` to `vector` and then parallelizing is not feasible. We must perform a parallel `max` directly on the `unordered_map`.



## 8\. `unordered_map` Bucket API

After searching, I found that `unordered_map` has a bucket API, which allows us to iterate similarly to a `vector` and thus perform a parallel `max`. For more on how `unordered_map` is implemented, you can refer to [C++ Unordered Map Under the Hood](https://jbseg.medium.com/c-unordered-map-under-the-hood-9540cec4553a). I'll briefly introduce the contents of that article here.

**Note**: This article only describes the general implementation logic of `unordered_map`. The specific implementation in the standard library may differ from what's described here. The C++ ecosystem is different from Python's. While Python's language specification allows for different implementations of the interpreter, the mainstream one is CPython (we'll also try pypy later, but many third-party libraries have issues with it). C++ does not have a single, dominant interpreter like CPython (which is similar to Java, where monopoly also has its advantages, such as faster decision-making, unlike C++ standards which are often not supported by many compilers for a long time). g++/clang/msvc all have their own territories, as do other compilers like icc which are suited for specific scenarios. Therefore, the C++ standard only defines the `unordered_map` interface, and different libraries (like libstdc++ used by g++ or libc++ by clang/llvm) can implement it in different ways, which can lead to significant performance differences. We will analyze the libstdc++ implementation later.

The principle of `unordered_map` can be described by the following diagram:

<a>![](/img/bpe/1.png)</a>

Pairs with the same hash code are placed in the same bucket. Each non-empty bucket points to a linked list that stores these pairs with identical hash codes. The collision resolution method used here is Separate Chaining. Another common implementation is Open Addressing.

The usual implementation (e.g., libstdc++) links all pairs together with a single linked list, and when we iterate with an iterator, we are actually traversing this list. However, based on the diagram above, we can also iterate through each bucket, since each bucket has a pointer to its starting part. Thus, `std::unordered_map` also provides a bucket interface:

  * [bucket\_count](https://en.cppreference.com/w/cpp/container/unordered_map/bucket_count.html)
      * Queries the total number of buckets.
  * [begin/cbegin](https://en.cppreference.com/w/cpp/container/unordered_map/begin2.html)
      * Returns the starting iterator for a specific bucket.
  * [end/cend](https://en.cppreference.com/w/cpp/container/unordered_map/end2.html)
      * Returns the ending iterator for a specific bucket.

Using this iterator, we can implement parallel traversal, with different threads accessing different buckets simultaneously.



## 9\. `bpe_train_updater_omp_v3`

`bpe_train_updater_omp_v3` implements the traversal of `pair_counts` using the bucket API. We expect that this traversal speed will not be much slower than the iterator traversal, making parallel traversal worthwhile. The full code is in [bpe\_train\_updater\_omp\_v3](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3.cpp). Let's look at the changes it brings to `bpe_train_step2`:

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

The code first iterates through each bucket, and then uses `pair_counts.begin(i)` and `pair_counts.end(i)` to iterate through that bucket. Here, it would be better to use `cbegin` and `cend` since we are not modifying `pair_counts`. However, the `const` interface should not affect the final performance.




## 10\. `bpe_train_updater_omp_v3` Test

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bpe\_v3\_time | | 31883 | 695 | 31187 | |
| bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 | |
| bpe\_train\_updater\_omp\_v1 test | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731 |
| bpe\_train\_updater\_omp\_v3 test | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707 |

This is unexpected\! Using the bucket API to find the max is actually faster than using the iterator in `bpe_train_updater`\! This doesn't make sense, because if the bucket API traversal was faster, the iterator could simply be implemented using the bucket API.

To solve this puzzle, I started to delve into the `std::unordered_map` source code in libstdc++.



## 11\. Why Is `bpe_train_updater_omp_v3` Faster Than `bpe_train_updater`?

Before researching the "why," we first have to ask if this is truly the case. After all, the code logic above is complex, not just a simple traversal. To confirm this, I wrote a small program to test it. The full code is in [test\_unordered\_map\_traverse\_speed.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/test_unordered_map_traverse_speed.cpp). This code simply tests the time for the two traversal methods on a randomly generated `unordered_map`.

The program usage is:

```
usage: test_unordered_map_traverse_speed loop_count map_size [key_range, seed]
```

The first argument is the number of loops; the second is the size of the randomly generated map; the third (optional) argument is the range for the generated `int` keys, defaulting to 1-10000; and the fourth (optional) is the random seed, defaulting to 1234. Here are the test results (g++-11/libstdc++.so.6.0.32, slightly different from the server version which was g++-9/libstdc++.so.6.0.28):

```
$ ./test_unordered_map_traverse_speed 1000 10000 100000 1234
loop_count: 1000, map_size: 10000
key_range: 100000, seed: 1234
iter_duration:    48ms
bucket_duration: 188ms

$ ./test_unordered_map_traverse_speed 1000 10000 100000 1234
loop_count: 1000, map_size: 10000
key_range: 100000, seed: 1234
iter_duration:    51ms
bucket_duration: 192ms

$ ./test_unordered_map_traverse_speed 1000 100000 100000 1234
loop_count: 1000, map_size: 100000
key_range: 100000, seed: 1234
iter_duration:    443ms
bucket_duration: 1143ms

$ ./test_unordered_map_traverse_speed 1000 1000000 10000000 1234
loop_count: 1000, map_size: 1000000
key_range: 10000000, seed: 1234
iter_duration:    43885ms
bucket_duration: 66023ms
```

This test shows the opposite result: iterating with an iterator is faster than using the bucket API. This makes sense, as the bucket API requires two loops, and there can be empty buckets.

So now a new question arises: are the results from the bucket API traversal and the iterator traversal exactly the same? If the results are different, it might explain why `bpe_train_updater_omp_v3` is faster than `bpe_train_updater` (we'll look into this later). If they are the same, it would be hard to explain.

To investigate this, I wrote a simple program [test\_map\_iter.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/test_map_iter.cpp):

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

The output is:

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

We can see that the range-based `for` loop and the iterator produce the exact same result (as the range-based loop is just a syntax sugar), but it's different from the bucket API access. This is slightly unexpected. To understand why, we need to analyze the `unordered_map` source code.

For an analysis of the `unordered_map` source code, you can refer to [C++那些事之彻底搞懂STL HashTable](https://zhuanlan.zhihu.com/p/644205339) or the Bilibili video [手把手共读HashTable，彻底搞懂C++ STL](https://www.bilibili.com/video/BV1o8411U7vy/?vd_source=bb6532dcd5b1d6b26125da900adb618e).

The article uses gcc-4.9.1/libstdc++-v3, which is slightly different from the version I'm using, but the differences are minor. As a side note, readers not interested in the details can skip this part. Because libstdc++ needs to consider compatibility, its code is much less readable than libc++.

I'll only analyze the iterator-related code. I recommend that you debug `test_map_iter.cpp` (with VS Code, for example) and read the code while checking the variables at breakpoints; this is much easier than reading the code directly.

<a>![](/img/bpe/2.jpg)</a>

This diagram is similar to the previous one, but all elements are linked together in a chain, which is the actual implementation in libstdc++. The iterator traversal uses this linked list. If we carefully compare the output of the previous program, we'll notice that the order of the iterator traversal is close to the reverse of the insertion order (but not exactly), which is related to the linked list insertion. We'll see this later.

Before reading the code, let's read the code documentation. My path is `/usr/include/c++/11/bits/hashtable.h`. You can search for a similar path based on your g++ version:

```
 * Each _Hashtable data structure has:
    *
    * - _Bucket[]       _M_buckets
    * - _Hash_node_base _M_before_begin
    * - size_type       _M_bucket_count
    * - size_type       _M_element_count
    *
    * with _Bucket being _Hash_node_base* and _Hash_node containing:
    *
    * - _Hash_node* _M_next
    * - Tp          _M_value
    * - size_t        _M_hash_code if cache_hash_code is true
    *
    * In terms of Standard containers the hashtable is like the aggregation of:
    *
    * - std::forward_list<_Node> containing the elements
    * - std::vector<std::forward_list<_Node>::iterator> representing the buckets
    *
    * The non-empty buckets contain the node before the first node in the
    * bucket. This design makes it possible to implement something like a
    * std::forward_list::insert_after on container insertion and
    * std::forward_list::erase_after on container erase
    * calls. _M_before_begin is equivalent to
    * std::forward_list::before_begin. Empty buckets contain
    * nullptr.  Note that one of the non-empty buckets contains
    * &_M_before_begin which is not a dereferenceable node so the
    * node pointer in a bucket shall never be dereferenced, only its
    * next node can be.
    *
    * Walking through a bucket's nodes requires a check on the hash code to
    * see if each node is still in the bucket. Such a design assumes a
    * quite efficient hash functor and is one of the reasons it is
    * highly advisable to set __cache_hash_code to true.
    *
    * The container iterators are simply built from nodes. This way
    * incrementing the iterator is perfectly efficient independent of
    * how many empty buckets there are in the container.
    *
    * On insert we compute the element's hash code and use it to find the
    * bucket index. If the element must be inserted in an empty bucket
    * we add it at the beginning of the singly linked list and make the
    * bucket point to _M_before_begin. The bucket that used to point to
    * _M_before_begin, if any, is updated to point to its new before
    * begin node.
    *
    * On erase, the simple iterator design requires using the hash
    * functor to get the index of the bucket to update. For this
    * reason, when __cache_hash_code is set to false the hash functor must
    * not throw and this is enforced by a static assertion.
    *
    * Functionality is implemented by decomposition into base classes,
    * where the derived _Hashtable class is used in _Map_base,
    * _Insert, _Rehash_base, and _Equality base classes to access the
    * "this" pointer. _Hashtable_base is used in the base classes as a
    * non-recursive, fully-completed-type so that detailed nested type
    * information, such as iterator type and node type, can be
    * used. This is similar to the "Curiously Recurring Template
    * Pattern" (CRTP) technique, but uses a reconstructed, not
    * explicitly passed, template pattern.
```

`_Hashtable` mainly contains four members [bits/hashtable.h](https://gcc.gnu.org/onlinedocs/gcc-11.5.0/libstdc++/api/a17518_source.html) line 391:

```
    private:
      __buckets_ptr     _M_buckets      = &_M_single_bucket;
      size_type         _M_bucket_count   = 1;
      __node_base       _M_before_begin;
      size_type         _M_element_count  = 0;
      _RehashPolicy     _M_rehash_policy;
```

`_M_bucket_count` stores the number of buckets, and `_M_element_count` stores the actual number of elements in the hashtable. Each element of `_M_buckets` is a pointer to `_Hash_node_base`, while the actual data is stored in `_Hash_node`, which inherits from `_Hash_node_base` and `_Hash_node_value`.

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

The `_Hash_node_base` code is:

```
  struct _Hash_node_base
  {
    _Hash_node_base* _M_nxt;

    _Hash_node_base() noexcept : _M_nxt() { }

    _Hash_node_base(_Hash_node_base* __next) noexcept : _M_nxt(__next) { }
  };
```

`_Hash_node_base` is very simple, it's just a singly linked list. Its only data member is `_M_nxt`, a pointer to the next element.

`_Hash_node_value` is where the key/value pair is actually stored. It has a specialized parameter `bool _Cache_hash_code` to indicate whether to cache the hash code.

```
  template<typename _Value, bool _Cache_hash_code>
    struct _Hash_node_value
    : _Hash_node_value_base<_Value>
    , _Hash_node_code_cache<_Cache_hash_code>
    { };
```

For now, we don't care about `_Hash_node_value`. We just need to know that from a data structure perspective, a hashtable is actually a singly linked list, and then a `std::vector` stores N buckets, where each non-empty bucket points to an element in the singly linked list (specifically, the one *before* the first element of that bucket). Curious readers might ask if a non-empty bucket should also store a pointer to the last element of the bucket or the number of elements in it. I also thought so at first, but the g++ implementation doesn't do this. It saves neither the size of each bucket nor a pointer to the last element. This saves space, but how does it know if the next element still belongs to the current bucket? There's no magic here; it trades time for space. It checks if the hash code of the next element is the same as the first element's hash code: if it's the same, it belongs to the current bucket; otherwise, it doesn't. So if the hash function is simple and fast to compute, this method can save storage space with a small amount of computation. But if the hash function is very complex, this method may not be optimal.

Although `_Hashtable` provides this specialized parameter, `std::unordered_map` does not enable it by default. If we need to cache the hash code, we can implement our own class as the key, for example:

```
#include <iostream>
#include <unordered_map>
#include <utility>
#include <functional>

class PairWithCachedHash {
public:
    // Constructor: receives a pair and immediately computes and caches the hash value
    PairWithCachedHash(const std::pair<int, int>& p)
        : data_(p), hash_code_(std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1)) {
        // You can use a more complex hash combination if needed
    }

    // Must implement equality comparison
    bool operator==(const PairWithCachedHash& other) const {
        return data_ == other.data_;
    }

    // Get the cached hash value
    size_t get_hash_code() const {
        return hash_code_;
    }

private:
    std::pair<int, int> data_;
    size_t hash_code_;
};
```

Interested readers can test if this space-for-time trade-off can make the program a bit faster.

Let's go back to the original question: why is the iterator traversal close to the reverse of the insertion order? The reason lies in the insertion logic: if the bucket for the element being inserted is empty, the element is inserted at the head of the singly linked list (since a singly linked list only has one head pointer, fast insertion can only happen at the head). The current bucket then points to this element (or the one before it). Otherwise, the element is inserted directly before the first element of the current bucket.

Let's run through our previous example. To avoid re-hashing, I specifically used `reserve(20)`, making the total number of buckets 23 (the smallest prime number greater than 20).

Our first inserted key is 192. Based on the hash function, 192 % 23 = 8. It's placed in bucket number 8. The data structure at this point is:

```
linkedlist -> 192
             /|\
              |
              |
             [8]
```

The second key is 623. The corresponding bucket is 623 % 23 = 2. It's inserted at the head of the linked list, resulting in:

```
linkedlist -> 623  ->  192
             /|\      /|\
              |        |
              |        |
             [2]      [8]
```

Next is 438:

```
linkedlist -> 438 -> 623  ->  192
             /|\    /|\      /|\
              |      |        |
              |      |        |
             [1]    [2]      [8]
```

Next is 786:

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

273:

```
linkedlist  -> 273  -> 780 -> 786 -> 438 -> 623  ->  192
              /|\    /|\    /|\    /|\    /|\      /|\
               |      |      |      |      |        |
               |      |      |      |      |        |
              [20]   [21]   [4]    [1]    [2]      [8]
```

277 % 23 = 1, so it's not inserted at the head, but before 438:

```
linkedlist  -> 273  -> 780 -> 786 -> 277 -> 438 -> 623  ->  192
              /|\    /|\    /|\    /|\            /|\      /|\
               |      |      |      |              |        |
               |      |      |      |              |        |
              [20]   [21]   [4]    [1]            [2]      [8]
```

The others are similar, so I won't go into detail. The final result is:

```
linkedlist -> 959  -> 802  -> 273  -> 780 -> 786 -> 277 -> 438 ->  876  -> 623  ->  192
              /|\    /|\            /|\    /|\    /|\            /|\            /|\
               |      |              |      |      |              |              |
               |      |              |      |      |              |              |
              [16]   [20]           [21]   [4]    [1]            [2]            [8]
```

Thus, the iterator traversal order is the order of the linked list: 959-\>802-\>273-\>780-\>786-\>277-\>438-\>876-\>623-\>192.
The bucket traversal order is according to the bucket order: [1]277,438 -\> [2]876,623 -\> [4]786 -\> [8]192 -\>[16]959 -\>[20]802,273 -\>[21]780

We can see that the iterator traversal order is closer to the reverse of the insertion order, while the bucket traversal is more random and less related to the insertion order. Only elements within the same bucket are in reverse order. The `begin` function of the bucket API ultimately calls:

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

Ignoring the template mess, the actual code is only two lines. The first gets a pointer to the element before the bucket, `__n`, based on the parameter `__bkt`. If `__n` is not null, it returns its next node.

`end()` returns a null pointer directly (the second parameter `nullptr` below):

```
      const_local_iterator
      end(size_type __bkt) const
      { return const_local_iterator(*this, nullptr, __bkt, _M_bucket_count); }
```

The overloaded `++` operator for the iterator eventually calls:

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

`__node_iter_base::_M_incr()` moves the linked list forward one element. It then calls `_M_bucket_index` to compute the bucket of the current element (which requires recomputing the hash code). If this element's bucket is different from `_M_bucket`, it means the current bucket has ended, so it returns `nullptr`.

Now that we know the difference between iterator and bucket API traversal, how do we explain why `bpe_train_updater_omp_v3` is faster than `bpe_train_updater`?

One possible guess is that because Python's `dict` preserves insertion order when exported to JSON (a feature since Python 3.8), and there's some randomness in word frequency counting, high-frequency words are more likely to appear first. As a simple example, if there are only two words, and the first word appears 99 times while the second appears once, the probability of the first word appearing first is 99%. Subsequent increments of word frequency don't change the key, so the first word has a 99% chance of being exported before the second word in the JSON.

Therefore, the `dict` received by the C++ version has a higher probability of having high-frequency words appearing earlier. However, after insertion into C++'s `pair_counts` (`std::unordered_map`), high-frequency keys may appear later during traversal. The `max` operation, besides traversing the `pair_counts` key/value pairs, also involves comparing each with the current maximum pair. If it's greater, it updates. Different traversal orders will have the same number of comparisons, N (where N is the size of `pair_counts`), but the number of updates will be different. If the traversal is from largest to smallest, the first element is the final maximum, so only one update is needed. If it's from smallest to largest, N updates are needed. The update code is:

```
        for(const auto& [pair, count] : pair_counts){
            if(count > max_count){
                max_count = count;
                max_pair = pair;
                max_strings = pair_strings[pair];
            }
```

The above only covers the case where the current `count` is greater than `max_count`. Updates can also happen when they're equal. Although an update is just three assignment statements, besides increasing the number of instructions executed, it can also cause CPU branch prediction failures, which stall the pipeline and further impact performance.

So, from an algorithm perspective, using an iterator for the max operation is not ideal, while using the bucket API is more random (though still not the best; the best would be to preserve the Python dictionary's traversal order).

Of course, this is just my guess. A reader might ask, even if the initial order is reversed, wouldn't a rehash of `pair_counts` reverse the order again, canceling out the effect? Indeed, `pair_counts` is continuously inserted into, deleted from, and rehashed during iteration, making the process difficult to predict accurately.

The best way to answer this question is with a real code test. So I modified `bpe_train_updater.cpp` and `bpe_train_updater_omp_v3.cpp` to create [bpe\_train\_updater\_max\_debug.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_max_debug.cpp) and [bpe\_train\_updater\_omp\_v3\_max\_debug.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3_max_debug.cpp). These versions add a count of `max` updates to the original code:

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

Let's test again. The results are:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bpe\_v3\_time | | 31883 | 695 | 31187 | |
| bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 | |
| bpe\_train\_updater\_omp\_v1 | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731 |
| bpe\_train\_updater\_omp\_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707 |
| bpe\_train\_updater\_max\_debug | Boost hash | 7332/7321/7156 | 446/416/400 | 6885/6905/6755 | max\_change: 599657 |
| bpe\_train\_updater\_omp\_v3\_max\_debug | Boost hash | 5489/5885/6138 | 399/413/418 | 5089/5471/5720 | max\_change: 390506 |

This confirms my guess: using the bucket API for traversal results in 390,506 updates, far fewer than the 599,657 updates with the iterator. As analyzed before, although the bucket API's traversal speed is slower, it's closer to random access, while the iterator is more likely to encounter the max value later on, leading to more updates. Overall, the bucket API is faster.

Given this, could we sort the `std::unordered_map` before inserting into C++ so the iterator traversal order is roughly from largest to smallest? This could be a good experiment to try. However, because we will move towards parallel traversal with the bucket API, relying on a specific feature of g++ is not a good idea. For instance, a different compiler or even a different version might have a different implementation. So, this optimization is more of a trick. We should strive for a general optimization algorithm that doesn't depend on such specific compiler features.



## 12\. Impact of the Hash Function

This article has become much longer than I anticipated, mainly because I discovered during the writing process that the bucket API is faster than the iterator interface. I hadn't paid much attention to it before, thinking it was just a random experimental variation. As a result, the research into the underlying implementation of `std::unordered_map` added a lot of content.

However, we should finally test the impact of different hash functions on `unordered_map`. I initially used a very simple hash function:

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

Gemini generated this for me, and I used it without much thought. The code using this hash function is [bpe\_train\_updater\_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_hash.cpp) and [bpe\_train\_updater\_omp\_v3\_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3_hash.cpp). They are almost identical to their respective versions, except for the change in the hash function. Here are the experimental results:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bpe\_v3\_time | | 31883 | 695 | 31187 | |
| bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 | |
| bpe\_train\_updater\_hash | my hash | 9875/11434/11222 | 2753/3182/2834 | 7122/8251/8388 | |
| bpe\_train\_updater\_omp\_v1 | Boost hash | 14151/16228/15253 | 415/447/459 | 62/62/61 | copy:13674/15718/14731 |
| bpe\_train\_updater\_omp\_v1\_hash | Boost hash | 17317/16883 | 2716/2686 | 81/80 | copy:14519/14115 |
| bpe\_train\_updater\_omp\_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707 |
| bpe\_train\_updater\_omp\_v3\_hash | my hash | 12592/11240/12355 | 3184/2939/3215 | 9408/8301/9139 | bucket:4355707 |

As you can see, using a poor hash function slows down the update speed significantly. Therefore, a good hash function is extremely important for `unordered_map` performance.
