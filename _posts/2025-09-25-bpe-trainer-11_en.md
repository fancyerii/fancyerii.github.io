---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 11: Wrapping C++ Code with Cython" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the twelfth and final article in a series on using Cython to wrap the previous C++ code into an extension module for Python.

<!--more-->


**Table of Content**
* TOC
{:toc}

Here's the English translation, with Markdown formatting and code blocks preserved:

-----

## 1\. Goal

In previous articles, we've done a lot of exploration. BPE training can be divided into two stages. The first stage is tokenization and counting, which we can perform in parallel using `multiprocessing`. Our `bpe_v3.py` implementation can complete this in 120 seconds using 32 cores, while the `bpe_v3_bytes_time.py` version, which reads bytes, can finish in 70 seconds using 64 cores. The second stage is merging. The Python version, `bpe_v8_v3.py`, takes about 500 seconds, with a total time of over ten minutes. The fastest C++ version, `bpe_train_updater_fine_grained_heap_emhash8_set9_opt.cpp`, has a merging time of around 100 seconds.

Today, our goal is to create the fastest version possible by wrapping the C++ code into an extension module using Cython, which can then be called by Python for the second stage.

## 2\. Wrapping the C++ Code into a Dynamic Library

We've conducted many experiments in `cppupdate`, and now we need to wrap the best-performing versions into a dynamic library for easy use with Cython.

We will create a new C++ project called `cppstep2`. The full code can be found at [cppstep2](https://github.com/fancyerii/assignment1-basics-bpe/tree/main/cppstep2).

Based on the `cppupdate` experiment results, I have selected the following files: [bpe\_train\_updater\_fine\_grained\_emhash8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8.cpp), [bpe\_train\_updater\_fine\_grained\_emhash8\_set.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8_set.cpp), [bpe\_train\_updater\_fine\_grained\_emhash8\_set9.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8_set9.cpp), [bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set9.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_emhash8_set9.cpp), and [bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set9\_opt.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_emhash8_set9_opt.cpp).

I chose `emhash` over `absl` for two reasons: it's faster and simpler to use. I previously `git clone`d the full `emhash` code, but to simplify, I've only copied the three necessary header files: `hash_set8.hpp`, `hash_set4.hpp`, and `hash_table8.hpp`. Additionally, `max_heap.h` and `max_heap.cpp` were copied directly from the `cppupdate` project.

Our dynamic library will consist of two main files: the header file [bpe\_train\_step2.h](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppstep2/bpe_train_step2.h) and the implementation file [bpe\_train\_step2.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppstep2/bpe_train_step2.cpp).

### 2.1 Header File

```
#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "emhash/hash_table8.hpp"

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

void bpe_train_step2(int vocab_size, 
                emhash8::HashMap<std::pair<int, int>, int, pair_hash> & pair_counts, 
                std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> & pair_strings, 
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges);

void bpe_train_step2_v2(int vocab_size,          
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges);

void bpe_train_step2_v3(int vocab_size,          
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges);   
                
void bpe_train_step2_v4(int vocab_size,          
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges);  

void bpe_train_step2_v5(int vocab_size,          
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges);  

void bpe_train_step2_v6(int vocab_size,          
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges); 
```

This header file defines five functions: `bpe_train_step2_v2`, `bpe_train_step2_v3`, `bpe_train_step2_v4`, `bpe_train_step2_v5`, and `bpe_train_step2_v6`. These correspond to `bpe_train_updater_fine_grained_emhash8.cpp`, `bpe_train_updater_fine_grained_emhash8_set.cpp`, `bpe_train_updater_fine_grained_emhash8_set9.cpp`, `bpe_train_updater_fine_grained_heap_emhash8_set9.cpp`, and `bpe_train_updater_fine_grained_heap_emhash8_set9_opt.cpp`, respectively.

`bpe_train_step2` is a direct copy of the `cppupdate` implementation and requires seven input parameters. As we'll see later, `pair_counts` and `wordid_counts` can be calculated inside C++ for better speed and fewer parameters passed. Thus, `bpe_train_step2_v2` and subsequent versions only require five parameters.

### 2.2 CMakeLists.txt

Our goal is to compile a dynamic library. We'll write the following in CMake syntax:

```
cmake_minimum_required(VERSION 3.20)

project(BPE_TRAIN_STEP2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(bpe_train_step2 SHARED bpe_train_step2.cpp max_heap.cpp)
target_include_directories(bpe_train_step2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bpe_train_step2 PUBLIC MY_LIBRARY_EXPORT)

target_include_directories(bpe_train_step2 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

install(TARGETS bpe_train_step2
        EXPORT bpe_train_step2_export_targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES "bpe_train_step2.h" 
        DESTINATION include
)
```

We create the `bpe_train_step2` target using `add_library`. `target_compile_definitions` seems to define a macro for Windows compatibility. I don't have a Windows environment to test this, so I'm unsure if the project will compile there.

The next `target_include_directories` includes the `emhash` headers. Finally, `install` copies the `bpe_train_step2` target (mainly `libbpe_train_step2.so`) and `bpe_train_step2.h` to the appropriate locations. I'll explain the installation process below.

### 2.3 bpe\_train\_step2

First, we will integrate it in the same way as `cppupdate`, which means we need to pass seven parameters, including `pair_counts`.

```
void bpe_train_step2(int vocab_size, 
                emhash8::HashMap<std::pair<int, int>, int, pair_hash> & pair_counts, 
                std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> & pair_strings, 
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges){
    auto start = std::chrono::steady_clock::now();
    while(vocabulary.size() < vocab_size){
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


    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "bpe_train_step2: " << duration.count() << "ms." << std::endl;
}
```

The code is a complete copy of [bpe\_train\_updater\_fine\_grained\_emhash8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_emhash8.cpp).

### 2.4 bpe\_train\_step2\_v2

If you've carefully reviewed the previous time statistics, you'll notice that the total time is about 50-60 seconds more than "tokenization time" + "merging time". This extra time is for calling the `BPE_Trainer._count_pairs` function:

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

This code calculates `pair_counts` based on `word_counts` and builds the initial inverted index `pair_to_words` and `pair_strings`. Its input parameters are `word_counts`, `word_encodings`, and `vocabulary`. This code can be implemented entirely in C++, which would not only speed it up but also reduce the number of parameters passed between Python and C++.

Therefore, the input parameters for `bpe_train_step2_v2` can be reduced by three:

```
void bpe_train_step2_v2(int vocab_size,          
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges){
    std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> pair_strings;
    emhash8::HashMap<std::pair<int, int>, int, pair_hash>  pair_counts;
    std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash>  pair_wordids;
    
    std::pair<int, int> pair;
    for(const auto& [wordid, count] : wordid_counts){
        const auto& word_tokens = wordid_encodings[wordid];
        for(int i = 0; i < word_tokens.size() - 1; ++i){
            pair.first = word_tokens[i];
            pair.second = word_tokens[i + 1];
            pair_counts[pair] += count;
            if (pair_strings.find(pair) == pair_strings.end()) {
                pair_strings[pair] = {vocabulary[pair.first], vocabulary[pair.second]};
            }
            pair_wordids[pair].insert(wordid);
        }
    }
    

    bpe_train_step2(vocab_size, pair_counts, pair_strings, vocabulary,
                    pair_wordids, wordid_counts, wordid_encodings, merges);
}
```

The `bpe_train_step2_v2` function first calculates `pair_counts`, `pair_strings`, and `pair_wordids` from `wordid_counts`, `wordid_encodings`, and `vocabulary`, and then calls `bpe_train_step2`.

The `bpe_train_step2_v3` and subsequent versions have the exact same interface as `bpe_train_step2_v2`, and the code is copied from the corresponding `cppupdate` versions, which I won't detail here.

### 2.5 Compile and Install

Use the following commands to compile and install:

```
cd cppstep2
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=../../lib_bpe_train_step2/  -D CMAKE_BUILD_TYPE=Release ..
cmake --build . -- -j8
# If you are using gcc, you can directly use make -j8
cmake --install .  
# Or make install
```

Since we don't want to install this library into system paths like `/usr/local` (which requires root privileges and is not convenient for Cython integration due to different paths on different systems), I'm installing it to `../../lib_bpe_train_step2/`, which is a subdirectory of the project root `assignment1-basics-bpe`:

```
$ ls assignment1-basics-bpe

cppstep2/
cppupdate/
cs336_basics/
data/
lib_bpe_train_step2/
```

After installation, the `lib_bpe_train_step2` directory will look like this:

```
lib_bpe_train_step2$ tree
.
├── include
│   ├── bpe_train_step2.h
│   └── emhash
│       └── hash_table8.hpp
└── lib
    └── libbpe_train_step2.so
```

Besides `bpe_train_step2.h`, `hash_table8.hpp` is also included because `bpe_train_step2.h` uses it. This is primarily because `bpe_train_step2`'s declaration needs it. If we only kept `bpe_train_step2_v2` and later versions, this header file would not need to be installed (though we'd still need all `emhash` headers for compilation).

## 3\. Wrapping with Cython into an Extension Module

For those new to wrapping C++ libraries with Cython, you can refer to the [Using C++ in Cython](https://cython.readthedocs.io/en/stable/src/userguide/wrapping_CPlusPlus.html) guide.

We need two files: [bpe\_train\_step2\_wrapper.pxd](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_train_step2_wrapper.pxd) and [bpe\_train\_step2\_wrapper.pyx](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_train_step2_wrapper.pyx).

### 3.1 bpe\_train\_step2\_wrapper.pxd

First, let's look at the `.pxd` file:

```
# distutils: language = c++

# 导入 C++ 标准库类型
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set


cdef extern from "../lib_bpe_train_step2/include/bpe_train_step2.h" :
    cppclass pair_hash:
        pass

cdef extern from "../lib_bpe_train_step2/include/emhash/hash_table8.hpp" namespace "emhash8":
    cppclass HashMap[K, V, H]:
        #ValueT& operator[](const KeyT& key) noexcept
        V& operator[](const K& key)


cdef extern from "../lib_bpe_train_step2/include/bpe_train_step2.h":
    void bpe_train_step2(int vocab_size,
                         HashMap[pair[int, int], int, pair_hash] & pair_counts,
                         unordered_map[pair[int, int], vector[vector[int]], pair_hash] & pair_strings,
                         unordered_map[int, vector[int]] & vocabulary,
                         unordered_map[pair[int, int], unordered_set[int], pair_hash] & pair_wordids,
                         const unordered_map[int, long long] & wordid_counts,
                         unordered_map[int, vector[int]] & wordid_encodings,
                         vector[pair[vector[int], vector[int]]] & merges) except +
    
    void bpe_train_step2_v2(int vocab_size,
                         unordered_map[int, vector[int]] & vocabulary,
                         const unordered_map[int, long long] & wordid_counts,
                         unordered_map[int, vector[int]] & wordid_encodings,
                         vector[pair[vector[int], vector[int]]] & merges) except +


```

The first line tells `distutils` (or `setuptools`) that you are building a C++ extension module, not the default C extension.

The next four lines import the C++ standard library types `std::vector`, `std::pair`, `std::unordered_map`, and `std::unordered_set` which are wrapped by Cython under `libcpp`. We can use them with `cimport`, which is a compile-time import. To see which C++ standard libraries are available, refer to [this link](https://github.com/cython/cython/tree/master/Cython/Includes/libcpp).

Next, we declare `pair_hash` with `cppclass`. Since we only need to know about the `pair_hash` symbol for the `lib_bpe_train_step2` library, we can leave the content empty with `pass`. Note that the path for `cdef extern from` is `"../lib_bpe_train_step2/include/bpe_train_step2.h"`, which requires `lib_bpe_train_step2` to be installed in the correct location as described earlier.

Next is the declaration of `emhash8::HashMap`. The `"namespace emhash8"` after `cdef extern from` tells Cython that `HashMap` is in the `emhash8` namespace. `cppclass HashMap[K, V, H]` indicates that `HashMap` is a template class for Key/Value/Hash function. We'll need `operator[]` in the `.pyx` file, so we also need to declare its prototype here.

Finally, we declare the `bpe_train_step2` functions. The syntax is similar to C++, but `< >` is replaced with `[ ]`, which might look a bit strange.

### 3.2 bpe\_train\_step2\_wrapper.pyx

Now, we'll wrap the C++ functions into Python-callable functions. The main task here is parameter conversion, such as converting a Python `dict` to a C++ `std::unordered_map`. Let's look at the most direct implementation first.

#### 3.2.1 py\_bpe\_train\_step2

```
cpdef py_bpe_train_step2(int vocab_size,
                             pair_counts_py,
                             pair_strings_py,
                             vocabulary_py,
                             pair_wordids_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef HashMap[pair[int, int], int, pair_hash] pair_counts_cpp
    cdef unordered_map[pair[int, int], vector[vector[int]], pair_hash] pair_strings_cpp
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[pair[int, int], unordered_set[int], pair_hash] pair_wordids_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    cdef pair[int, int] pair_key
    cdef vector[vector[int]] strings_value
    cdef vector[int] vector_value
    cdef unordered_set[int] set_value




    for p, count in pair_counts_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        pair_counts_cpp[pair_key] = count

    for p, string in pair_strings_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        strings_value.clear()
        value = [list(item) for item in string] 
        vector_value = value[0]
        strings_value.push_back(vector_value)
        vector_value = value[1]
        strings_value.push_back(vector_value)        
        pair_strings_cpp[pair_key] = strings_value     
    
    for k, v in vocabulary_py.items():
        value = list(v)
        vector_value = value
        vocabulary_cpp[k] = vector_value

    for p, wordids in pair_wordids_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]        
        set_value = wordids
        pair_wordids_cpp[pair_key] = set_value

    for k, v in wordid_counts_py.items():
        wordid_counts_cpp[k] = v

    for k, v in wordid_encodings_py.items():
        vector_value = v
        wordid_encodings_cpp[k] = vector_value

    # 调用 C++ 函数
    bpe_train_step2(vocab_size,
                    pair_counts_cpp,
                    pair_strings_cpp,
                    vocabulary_cpp,
                    pair_wordids_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp
```

Most of the code in this function converts Python variables to C++ variables before calling `bpe_train_step2`. Let's look at two typical examples:

```
    cdef HashMap[pair[int, int], int, pair_hash] pair_counts_cpp
    cdef pair[int, int] pair_key

    for p, count in pair_counts_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        pair_counts_cpp[pair_key] = count
```

The key for `pair_counts_cpp` is `pair[int,int]`, which is `std::pair<int,int>`. We can assign to its `first` and `second` members. We then use the `operator[]` to insert, which is why we needed to declare `V& operator[](const K& key)` earlier. The code iterates like a Python `for` loop.

Let's look at a more complex example:

```
    cdef unordered_map[pair[int, int], vector[vector[int]], pair_hash] pair_strings_cpp

    for p, string in pair_strings_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        strings_value.clear()
        value = [list(item) for item in string] 
        vector_value = value[0]
        strings_value.push_back(vector_value)
        vector_value = value[1]
        strings_value.push_back(vector_value)        
        pair_strings_cpp[pair_key] = strings_value   
```

The key for `pair_strings_py` is a `tuple`, and the value is also a `tuple` with two `bytes` elements. For example:

```
(111,110): (b'o', b'n')
```

The corresponding `pair_strings_cpp` is `unordered_map[pair[int, int], vector[vector[int]], pair_hash]`, so we need to convert the `bytes` to `list[int]`. This is done with the following statements:

```
value = [list(item) for item in string] 
vector_value = value[0]
strings_value.push_back(vector_value)
```

First, a list comprehension converts `tuple[bytes]` to `list[list[int]]`. Then, Python's `list[int]` is converted to C++'s `vector[int]` through assignment, and finally pushed back to `strings_value`.

#### 3.2.2 py\_bpe\_train\_step2\_v2

```
cpdef py_bpe_train_step2_v2(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    cdef pair[int, int] pair_key
    cdef vector[vector[int]] strings_value
    cdef vector[int] vector_value
    cdef unordered_set[int] set_value
 
    
    for k, v in vocabulary_py.items():
        value = list(v)
        vector_value = value
        vocabulary_cpp[k] = vector_value


    for k, v in wordid_counts_py.items():
        wordid_counts_cpp[k] = v

    for k, v in wordid_encodings_py.items():
        vector_value = v
        wordid_encodings_cpp[k] = vector_value

    # 调用 C++ 函数
    bpe_train_step2_v2(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp
```

This version is similar to the previous one, but with three fewer parameters.

Note: We return C++ variables `merges_cpp` and `vocabulary_cpp` at the end. Their types are:

```
cdef unordered_map[int, vector[int]] vocabulary_cpp
cdef vector[pair[vector[int], vector[int]]] merges_cpp
```

When they are returned to Python, Cython automatically converts them to `dict[int, list[int]]` and `list[tuple[list[int], list[int]]]`. We'll need to convert the `list[int]` back to `bytes` later.

In fact, not just for return values, but also when we copy a Python variable to a C++ variable or vice versa, Cython automatically performs these common conversions:

| Python type =\> | *C++ type* | =\> Python type |
| :--- | :--- | :--- |
| bytes | std::string | bytes |
| iterable | std::vector | list |
| iterable | std::list | list |
| iterable | std::set | set |
| iterable | std::unordered\_set | set |
| mapping | std::map | dict |
| mapping | std::unordered\_map | dict |
| iterable (len 2) | std::pair | tuple (len 2) |
| complex | std::complex | complex |

We can use this feature to simplify the variable conversions and get `py_bpe_train_step2_opt`:

```
cpdef py_bpe_train_step2_opt(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    bpe_train_step2_v2(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp
```

Here, we use just three assignment statements, and Cython automatically handles the conversion between Python and C++. The calls to other versions, like `py_bpe_train_step2_v3`, are identical to `py_bpe_train_step2_opt`:

```
cpdef py_bpe_train_step2_v3(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    bpe_train_step2_v3(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp
```

### 3.3 Modifying setup.py

```
project_root = os.path.dirname(os.path.abspath(__file__))


ext_modules = [
    Extension(
        name="cs336_basics.bpe_train_step2_wrapper",
        sources=["cs336_basics/bpe_train_step2_wrapper.pyx"],

        language="c++",
        #extra_compile_args=['-std=c++17', '-O3'],
        extra_compile_args=['-std=c++17'],
        libraries=["bpe_train_step2"],

        library_dirs=[f"{project_root}/lib_bpe_train_step2/lib"],
        runtime_library_dirs=[f"{project_root}/lib_bpe_train_step2/lib"],
        include_dirs=[f"{project_root}/lib_bpe_train_step2/include",
                      f"{project_root}/lib_bpe_train_step2/include/emhash"],
    )
]

setup(
    packages=['cs336_basics'],
    name='bpe_train_step2',
    ext_modules=cythonize(ext_modules),
)
```

We need to compile `bpe_train_step2_wrapper.pyx`.

  * `name` specifies the module name, allowing Python to `import cs336_basics.bpe_train_step2_wrapper`. Here, `cs336_basics` is the package name, and `bpe_train_step2_wrapper` is the module name.
  * `sources` specifies the source files to be compiled.
  * `language` specifies the language of the module as C++.
  * `extra_compile_args` specifies additional compilation options, in this case `'-std=c++17'`.
  * `libraries` lists the dependent libraries.
  * `library_dirs` specifies the library location for compilation.
  * `runtime_library_dirs` specifies the library location for runtime.
  * `include_dirs` specifies the header file location for compilation.

To avoid hard-coding, `project_root` is set to the directory where `setup.py` is located.

These options ultimately result in the following C++ compiler commands (for my GCC environment; they may vary):

```
c++ -pthread -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -fPIC -fPIC -Ics336_basics -I......codes/assignment1-basics-bpe/lib_bpe_train_step2/include -I......codes/assignment1-basics-bpe/lib_bpe_train_step2/include/emhash -I......codes/assignment1-basics-bpe/.venv/include -I.......local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/include/python3.12 -c cs336_basics/bpe_train_step2_wrapper.cpp -o build/temp.linux-x86_64-cpython-312/cs336_basics/bpe_train_step2_wrapper.o -std=c++17
c++ -pthread -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -fPIC -shared -Wl,--exclude-libs,ALL build/temp.linux-x86_64-cpython-312/cs336_basics/bpe_train_step2_wrapper.o -L......codes/assignment1-basics-bpe/lib_bpe_train_step2/lib -L.......local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/lib -Wl,--enable-new-dtags,-rpath,......codes/assignment1-basics-bpe/lib_bpe_train_step2/lib -lbpe_train_step2 -o build/lib.linux-x86_64-cpython-312/cs336_basics/bpe_train_step2_wrapper.cpython-312-x86_64-linux-gnu.so
```

### 3.4 Using it in Python

[bpe\_v9.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v9.py) calls `py_bpe_train_step2`; [bpe\_v10.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v10.py) calls `py_bpe_train_step2_v2`; [bpe\_v10\_v2.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v10_v2.py) calls `py_bpe_train_step2_opt`; [bpe\_v11.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11.py) calls `py_bpe_train_step2_v3`; [bpe\_v11\_bytes.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11_bytes.py) calls `py_bpe_train_step2_v3`; [bpe\_v11\_v2.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11_v2.py) calls `py_bpe_train_step2_v4`; [bpe\_v11\_v3.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11_v3.py) calls `py_bpe_train_step2_v5`; [bpe\_v11\_v3\_bytes.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11_v3_bytes.py) calls `py_bpe_train_step2_v5`; [bpe\_v11\_v4.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11_v4.py) calls `py_bpe_train_step2_v6`; [bpe\_v11\_v4\_bytes.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11_v4_bytes.py) calls `py_bpe_train_step2_v6`.

Their code is mostly the same. Let's look at [bpe\_v11.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v11.py):

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

        word_ids = {word:id for id, word in enumerate(word_counts)}

        wordid_counts = {word_ids[word]:count for word, count in word_counts.items()}

        wordid_encodings = {word_ids[word]:encoding for word, encoding in word_encodings.items()}      


        merges_cpp, vocabulary_cpp = py_bpe_train_step2_v3(vocab_size, 
                             vocabulary,
                             wordid_counts,
                             wordid_encodings,
                             merges)


        vocabulary = {k:bytes(v) for k, v in vocabulary_cpp.items()}
        merges = [(bytes(arr[0]), bytes(arr[1])) for arr in merges_cpp]
```

Before the call, we need to convert string words to integer IDs. This mapping is stored in `word_ids`. We then use `word_ids` to convert `word_counts` to `wordid_counts` and `word_encodings` to `wordid_encodings`. After the call, the returned `merges_cpp` and `vocabulary_cpp` need to have their `list[int]` converted back to `bytes`.

## 4\. Testing

| Version | Data | Total Time (s) | Tokenization Time (s) | Merging Time (s) | Other |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bpe\_v8\_v3 | open\_web | 897/899/951 | 395/399/395 | 442/438/493 | num\_counter=8, num\_merger=1 |
| bpe\_v9 | open\_web | 831/816/867 | 400/401/390 | prepare & convert: 93/86/94 py\_bpe\_train\_step2: 330/320/374 c++:289/281/326 | num\_counter=8, num\_merger=1 |
| bpe\_v10 | open\_web | 816/769/788 | 390/400/400 | prepare & convert: 21/17/19 py\_bpe\_train\_step2: 402/350/367 c++:338/296/309 | num\_counter=8, num\_merger=1 |
| bpe\_v10\_v2 | open\_web | 767/774/767 | 400/401/401 | prepare & convert: 18/17/17 py\_bpe\_train\_step2: 346/355/347 c++:292/298/294 | num\_counter=8, num\_merger=1 |
| bpe\_v10\_v2 | open\_web | 498/477/495 | 120/120/120 | prepare & convert: 20/19/18 py\_bpe\_train\_step2: 355/336/354 c++:299/282/298 | num\_counter=32, num\_merger=4 |
| bpe\_v11 | open\_web | 350/340/354 | 120/120/120 | prepare & convert: 21/19/19 py\_bpe\_train\_step2: 207/199/212 c++:183/175/190 | num\_counter=32, num\_merger=4 |
| bpe\_v11\_bytes | open\_web | 311/307/305 | 80/80/80 | prepare & convert: 18/19/18 py\_bpe\_train\_step2: 211/206/204 c++:189/183/182 | num\_counter=64, num\_merger=8, chunk\_size 8mb |
| bpe\_v11\_v2 | open\_web | 362/350/338 | 130/120/120 | prepare & convert: 19/18/19 py\_bpe\_train\_step2: 210/210/197 c++:189/190/176 | num\_counter=32, num\_merger=4 |
| bpe\_v11\_v3 | open\_web | 269/274/270 | 120/120/120 | prepare & convert: 18/19/18 py\_bpe\_train\_step2: 129/133/129 c++: 106/109/106 | num\_counter=32, num\_merger=4 |
| bpe\_v11\_v3\_bytes | open\_web | 218/219/215 | 72/74/69 | prepare & convert: 21/21/21 py\_bpe\_train\_step2: 123/122/123 c++: 101/100/101 | num\_counter=64, num\_merger=8, chunk\_size 8mb |
| bpe\_v11\_v4 | open\_web | 258/256/261 | 116/117/117 | prepare & convert: 19/18/19 py\_bpe\_train\_step2: 121/119/123 c++: 98/97/100 | num\_counter=32, num\_merger=4 |
| bpe\_v11\_v4\_bytes | open\_web | 210/206/207 | 71/69/70 | prepare & convert: 20/18/19 py\_bpe\_train\_step2: 117/117/117 c++: 95/96/95 | num\_counter=64, num\_merger=8, chunk\_size 8mb |

Comparing `bpe_v10` and `bpe_v10_v2`, the difference is between manually converting parameters in Python to C++ and letting Cython do it automatically. Automatic conversion is not only more convenient but also faster.

Ultimately, with `bpe_v11_v4_bytes` on 64 cores, the fastest training time is a little over 200 seconds, which is more than 100 times faster than our initial time of over ten hours\!

## 5\. Conclusion

This concludes the series. Here is a brief summary and links to the corresponding articles:


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
