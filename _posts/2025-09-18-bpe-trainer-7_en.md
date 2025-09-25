---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 7: Using Flat Hash Map instead of std::unordered_map" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the eighth article in the series, focusing on using a **flat hash map** to replace the C++ standard library's **`std::unordered_map`** for improved performance.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Problem Analysis

In the previous article, we used OpenMP parallelization to reduce the time for the second merge step from over 6000 seconds to less than 1000 seconds with 32 threads (bpe\_train\_updater\_omp\_v7). Out of that 1000 seconds, over 500 seconds were spent on updating data like pair\_counts, and another 400+ seconds were for finding the max pair.

Before continuing the optimization, let's review the main data read/write operations in the function `bpe_train_step2`. We'll use the serial version of the code, [bpe\_train\_updater\_omp\_v3.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v3.cpp%23L232), as an example.

The main code for finding the max pair iterates through `pair_counts`. It only needs to read `pair_strings[pair]` when the count of the current pair is the same as the current maximum.

The code for updating variables like `pair_counts` is as follows:

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

The most critical operations here are the write operations: `pair_counts[old_pair] -= wc;` and `pair_counts[new_pair] += wc;`. Additionally, when `pair_counts[old_pair] <= 0`, the code deletes `old_pair` from both `pair_counts` and `pair_wordids`.

There is also a read operation `if(pair_counts[old_pair] <= 0)`. This operation can actually be optimized away, which we'll discuss later. But even without optimization, the time it takes is very short. First, `unordered_map` lookups are fast (updates require memory allocation, insertion into a linked list and bucket, and may cause a rehash). Second, this lookup happens right after an update, so the relevant data should still be in the CPU's cache.

First, let's look at iterating through `pair_counts`. In the serial version, we iterate through the `unordered_map`'s internal singly-linked lists. In the parallel version, we also traverse the linked list contents, but we do so in chunks using the bucket interface. From an implementation perspective, `unordered_map` is a linked list, which allows for O(1) time complexity for element insertion and deletion. However, our main write operation is updating the value corresponding to a key. The memory layout of a linked list is very fragmented, and its traversal is much slower than a contiguous memory layout. Therefore, for our scenario, we can use a hash map with a contiguous layout, a so-called **flat hash map**. Its main feature is that it stores all data in a single, contiguous block of memory, unlike a standard hash map which disperses data using linked lists or pointers.

A standard hash map (e.g., C++'s `std::unordered_map`) typically uses a hash function to map keys to different "buckets," where each bucket might be a linked list to store elements with the same hash value. This results in elements being scattered and non-contiguous in memory.

A flat hash map, on the other hand, is completely different. Its core principles are:

  * **Contiguous Memory Layout**: It uses one large array to store all key-value pairs. Since the data is stored contiguously, this significantly improves **Cache-Friendliness**. When a processor accesses one element, it usually loads nearby elements into the cache as well, making subsequent accesses much faster.
  * **Open Addressing**: To handle hash collisions (when two different keys hash to the same location), a flat hash map does not use linked lists. Instead, it employs a technique called **Probing**. When a position is already occupied, it follows a predefined rule (e.g., linear or quadratic probing) to find the next available empty slot for the data.

**Pros and Cons**

  * **Pros**
      * **Excellent Cache Performance**: This is its biggest advantage. In modern CPUs, cache access is much faster than main memory access. Because data is contiguous, a flat hash map's access pattern is ideal for the CPU cache, especially for traversal or bulk lookups.
      * **Low Memory Overhead**: Since it doesn't need to store extra pointers for each element, its memory footprint is smaller, especially when storing a large number of small elements.
      * **Ideal for High-Performance Scenarios**: In fields with extremely low latency requirements, such as game development, network programming, and high-performance computing, the flat hash map is a great choice.
  * **Cons**
      * **Complex Deletion**: Due to the probing mechanism, elements cannot simply be removed. A special "tombstone" marker is needed to ensure that subsequent lookups in the chain are not broken. This can lead to fragmentation in the array and requires periodic cleanup (re-hashing).
      * **Clustering Issues**: If the hash function is not ideal or if collisions are frequent, data can cluster together in the array, forming "blocks." This makes the probing process longer and can degrade performance.
      * **High Rehash Cost**: When the underlying array needs to be resized, a larger new array must be allocated and all elements copied over, a process with a relatively high cost.

However, for our specific scenario, a flat hash map is very suitable: our main operation is traversal, and deletions are infrequent.

## 2\. Optimizing the `-=` Operation

Before replacing `std::unordered_map` with a flat hash map, let's do a small optimization to save one query to `pair_counts`.

In C++, `unordered_map`'s `-=` is an **expression** that returns the value after the operation; whereas in Python, a dict's `-=` is a **statement** with no return value. For example, the following code compiles in C++:

```
int x = 1;
int y = (x += 1);
```

But similar code in Python:

```
x = 1
y = (x += 1)
```

will cause a syntax error:

```
    y = (x += 1)
           ^
SyntaxError: invalid syntax
```

So in Python, we can only query `pair_counts[old_pair]` again after the update `pair_counts[old_pair] -= wc;`. But in C++, we can directly get the value after the `-=` operation.

We can apply this optimization to all versions (e.g., bpe\_train\_updater\_omp\_v3/bpe\_train\_updater\_omp\_v5/bpe\_train\_updater\_omp\_v7, etc.), but since we only discovered it after the last version, we'll first compare it with the original `bpe_train_updater.cpp`. If it proves effective, we'll apply it to the final version. The complete code is in [bpe\_train\_updater\_opt.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt.cpp#L165), with only one change:

```
            int& c = (pair_counts[old_pair] -= wc);
            if(c <= 0){
                pair_counts.erase(old_pair);
                pair_wordids.erase(old_pair);
            }else{
                pair_wordids[old_pair].erase(wordid);
            }
```

Here's the test comparison:

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe\_train\_updater\_opt | Boost hash | 8153/7634/7362 | 411/396/389 |7741/7238/6973 |

Comparing the update times, we see an average decrease from 450s to 398.67s, an improvement of 11.41%. While this may not be a huge gain for the overall runtime, every little bit helps.

## 3\. Flat Hash Map Survey

We've already introduced the principles of the flat hash map and concluded it's a good fit for our scenario. Of course, we don't need to reinvent the wheel, so the next step is to research available open-source flat hash map libraries. A search reveals many third-party libraries, and the article [Comprehensive C++ Hashmap Benchmarks 2022](https://martin.ankerl.com/2022/08/27/hashmap-bench-01/) tests several C++ hash map libraries, many of which are flat.

Based on the summary of this article, I've tried `absl::flat_hash_map` and `emhash8::HashMap`.

## 4\. absl::flat\_hash\_map

The first reason for trying this one is that many articles recommend it, and it's open-source from Google. Google supposedly uses it internally, which is quite impressive.

`absl::flat_hash_map` is part of the [abseil-cpp](https://github.com/abseil/abseil-cpp) library. To use `absl::flat_hash_map`, you have to pull in a huge number of other dependencies, which seems to be a common issue with Google's open-source libraries. You only want to use one small feature, but since it's part of a larger library, you have to include a bunch of dependencies. Of course, this isn't entirely Google's fault; it's a fundamental problem with C++. Unlike Python or Java, C++ lacks a cross-platform ABI, which means to use a C++ library, you generally have to recompile all of its dependencies. In the early days, C++ also lacked a central repository like Maven or PyPI (though things like Conan exist now, they're not widely used), and compiling every library from scratch was a hassle. As a result, many people just used their own code or copied and modified others' code, leading to fragmented branches that couldn't be easily updated. A mature library has many dependencies. For example, you probably need logging for any code you write, so you need a dependency for that. You need to parse command-line arguments, another dependency. You need string manipulation, but the C++ standard library strings aren't great, so you need to create your own or rely on one. You need regular expressions, XML/JSON/YAML parsing, etc. The C++ standard library lacks all of these. In contrast, in Python or Java, these features are either in the standard library or have converged on a few winning open-source libraries after several rounds of competition, leading to a more unified ecosystem. But C++ can't converge, so every company has to build its own set of tools, and some companies even rebuild tools that the standard library already has because they don't like them. This results in them open-sourcing almost an entire suite of libraries, making it impossible to separate a single small feature.

There are also some very small libraries, usually header-only, that you can just copy directly into your project. Even with these, because they don't have a package manager like PyPI or Maven, once you use them, you basically can't (or don't dare to) upgrade them. This makes them only suitable for libraries that will never be modified after their release. But even small libraries need bug fixes. You can't just `diff` and apply patches yourself.

Anyway, let's get back to the main point. We first include `absl::flat_hash_map` in our code, which requires the full `abseil-cpp` code. We've already got the code via `git submodule`, so we just need to modify `CMakeLists.txt`:

```
add_subdirectory(abseil-cpp)

target_link_libraries(bpe_train_updater_omp_v2 PRIVATE absl::flat_hash_map absl::hash)
```

Then, include the header file where you need to use it:

```
#include "absl/container/flat_hash_map.h"
```

The `absl::flat_hash_map` interface is similar to `std::unordered_map`, but it lacks the `bucket` interface, so parallel traversal via buckets is not possible.

The full code using `absl::flat_hash_map` is in [bpe\_train\_updater\_omp\_v2.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v2.cpp). The code is almost identical to `bpe_train_updater.cpp`, except that `std::unordered_map` has been replaced with `absl::flat_hash_map`.

Previously, compiling `cppupdate` took only tens of seconds. After introducing this dependency, it takes several minutes even with 8 threads.

## 5\. `absl::flat_hash_map` Test

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe\_train\_updater\_omp\_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v2 | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |

We found that compared to `bpe_train_updater`, the total time for `bpe_train_updater_omp_v2` dropped from around 8000 seconds to just over 2000 seconds. The max time is less than 300 seconds, which is more than 20 times faster than the 7000 seconds of `bpe_train_updater` and even faster than the 32-thread `bpe_train_updater_omp_v7`. This shows that the contiguous memory layout of the flat hash map is indeed well-suited for traversal. However, the update time increased from over 400 seconds to over 2000 seconds, which doesn't seem to align with the benchmarks from other people online.

As we mentioned before, the hash function for a hash map is crucial. `absl::flat_hash_map` comes with its own hash function, which we need to test. We also have a very simple hash function that we implemented previously, which we can compare. The code using the simple hash is in [bpe\_train\_updater\_omp\_v2\_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v2_hash.cpp), and the code using the native `absl` hash is in [bpe\_train\_updater\_omp\_v2\_hash2.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v2_hash2.cpp). Additionally, we incorporated the `-=` optimization into [bpe\_train\_updater\_opt\_absl](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt_absl.cpp).

Using the native `absl` hash is very simple; it's used by default if you don't specify a hash function:

```
absl::flat_hash_map<std::pair<int, int>, int> pair_counts;
```

Here are the test results:

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe\_train\_updater\_omp\_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v2\_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe\_train\_updater\_omp\_v2 | my hash | 6276/6349/6613 | 6030/6104/6364 | 245/245/249 |
bpe\_train\_updater\_omp\_v2\_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe\_train\_updater\_opt\_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637

When using a simple hash (`my hash`), `absl::flat_hash_map`'s max time is still around 200 seconds, but the update time increases to over 6000 seconds. However, when using the native `absl` hash function, the max time increases to over 600 seconds (still far less than `bpe_train_updater`'s 7000+ seconds), but the update time is only around 500 seconds. The performance of this version is excellent, even surpassing the 16-thread speed of `bpe_train_updater_omp_v7`. The `bpe_train_updater_opt_absl` version, with the `-=` optimization, further reduces the update time to around 400 seconds.

## 6\. emhash8::HashMap

You can refer to [this issue](https://github.com/ktprime/emhash/issues/33#issuecomment-1636618464) for the principles behind **emhash**. I haven't researched it in depth myself, so interested readers can explore it on their own or ask questions on the project's GitHub page.

[emhash](https://github.com/ktprime/emhash) is much lighter than `absl::flat_hash_map`. We only need to use `emhash8`, so we just need to include the header files. We can do this by adding a `target_include_directories` line to `CMakeLists.txt`:

```
target_include_directories(bpe_train_updater_emhash8 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)
```

The change is also very simple: just replace `std::unordered_map` with `emhash8::HashMap`. The full code is in [bpe\_train\_updater\_emhash8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_emhash8.cpp).

## 7\. `emhash8::HashMap` Test Results

For comparison, we also implemented an `-=` optimized version, [bpe\_train\_updater\_opt\_emhash8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt_emhash8.cpp), and an `-=` optimized version with a custom hash function, [bpe\_train\_updater\_opt\_emhash8\_hash.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_opt_emhash8_hash.cpp).

Here are the results:

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe\_train\_updater\_omp\_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,1000"
bpe\_train\_updater\_omp\_v2\_hash | Boost hash | 2201/2392/2281 | 1931/2120/2010 |269/272/270 |
bpe\_train\_updater\_omp\_v2 | my hash | 6276/6349/6613 | 6030/6104/6364 | 245/245/249 |
bpe\_train\_updater\_omp\_v2\_hash2 | Absl hash | 1170/1074/1071 | 545/456/449 |625/617/621
bpe\_train\_updater\_opt\_absl | Absl hash |1072/1012/1022 | 423/378/384 | 648/633/637
bpe\_train\_updater\_emhash8 | Boost hash | 479/485/485 | 398/401/401 | 80/83/83
bpe\_train\_updater\_opt\_emhash8 | Boost hash | 469/474/479 | 389/395/399 |79/78/79
bpe\_train\_updater\_opt\_emhash8\_hash| my hash | 2316/1951/1983 | 2250/1888/1918 |66/63/64

The speed of `emhash8::HashMap` is even faster than `absl::flat_hash_map`, with a total time of less than 500 seconds. The max time is even less than 100 seconds. Using the `-=` optimization shaves off another ten seconds or so. At the same time, we see that performance also degrades if we use a simple hash function.

## 8\. Summary

By using a flat hash map, the `max` time was drastically reduced, which proves that the flat, contiguous memory structure is indeed suitable for our scenario. As a result, we achieved a very high speed without even needing parallel optimization. The lesson here is that optimizing data structures and algorithms can sometimes yield better results than parallelization. We've accomplished the same task with fewer resources and at a faster speed.

With this, our work on optimizing the second step with C++ is temporarily concluded. Next, we will return to Python to continue our optimizations. 



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
