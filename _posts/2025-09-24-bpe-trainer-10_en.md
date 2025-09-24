---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 10: Using Cython and PyPy for Acceleration" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This article, the eleventh in the series, will cover using Cython and PyPy to accelerate Python code.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Problem Analysis

In the previous article, we have already performed extreme optimizations on the algorithm. The fastest Python version, `bpe_v7_maxheapc_opt_time`, has a merge time of over 500 seconds, while the fastest C++ version, `bpe_train_updater_fine_grained_heap_emhash8_set9_opt`, has a merge time of 100 seconds. Later, we will integrate the C++ code with the previous Python code using Cython. However, today, we want to perform some optimizations at the Python level, as using two languages is quite troublesome.

For the same logic, Python is slower than C++ for many reasons. The main reasons are: Python (more accurately, CPython) interprets code while C++ compiles to machine code; Python is dynamically typed while C++ is statically typed; Python's GIL prevents true multithreading parallelism; and Python's memory layout is not CPU-friendly.

The third reason was analyzed when comparing Python's multiprocessing and C++/OpenMP parallel max search. Furthermore, through algorithmic optimization, we no longer need to scan the entire `pair_counts` to find the max, so we can ignore this difference.

The fourth reason is also very important. Even in C++, if you use a linked-list memory layout like `std::unordered_map`, its traversal speed is much slower than `absl::flat_hash_map` and `emhash8::HashMap`. In Python, everything is an object, which from a memory layout perspective is essentially a `void *`. Therefore, although Python's `list` looks similar to `std::vector`, their memory layouts are completely different.

When we create a `dict` with integer keys and values, like `{1: 10, 2: 20}`, Python actually does the following in memory:

  * Creates a hash table structure for the dictionary itself.
  * Creates a `PyLongObject` for the key `1`.
  * Creates a `PyLongObject` for the value `10`.
  * Stores the references (memory addresses) of the key and value objects in a hash table entry.
  * Repeats the process for key `2` and value `20`.

This means that even if your keys and values are just small integers, each integer comes with the overhead of a full Python object (including type information, reference counting, etc.). This design makes `dict` extremely flexible, allowing it to store data of any type, but at the cost of consuming more memory.

However, at the Python level, it's difficult to optimize `dict`. This is the cost of using Python. If you want to optimize memory, the best way is to use other languages (like C/C++) and integrate them via the Python/C API. Our previous max-heap algorithm was integrated via the Python/C API, but this API is very complex. We will later use Cython to integrate the C++ code, as Cython ultimately compiles our C++ code into an extension module. Similarly, NumPy is integrated via the Python/C API.

The remaining two points are the focus of today's optimization: first, we'll try to rewrite some key code with Cython; second, we'll try to use PyPy to replace CPython.

## 2\. Introduction to Cython Principles

Cython is a programming language that combines the ease of use of Python with the high performance of C. It is designed to be a **superset of Python**, which means you can use it to write regular Python code while also adding C language features, such as static type declarations. The core idea of Cython is to compile your Python code into C language, and then compile it into machine code to form an importable Python module. This process solves the performance bottleneck of the Python interpreter.

For a detailed introduction to Cython, please refer to the [official documentation](https://cython.readthedocs.io/en/stable/index.html). Additionally, if readers want to learn Cython systematically, they can read the book [Cython: A Guide for Python Programmers](https://www.amazon.com/Cython-Programmers-Kurt-W-Smith/dp/1491901551). Although it's an old book, most of its content is still useful.

## 3\. Optimizing the `fine_grained_pair_counter_diff` function with Cython

Our optimization is based on `bpe_v7_opt`. According to our previous analysis, the time for `max` can be ignored, and most of the remaining time is spent in the `_updated_affected_word_count` function. This function mainly calls `fine_grained_pair_counter_diff` and then updates the `pair_counts` and `pair_to_words` dictionaries. Updating dictionaries is difficult to optimize with Cython, so we will first optimize the `fine_grained_pair_counter_diff` function.

We first create a [bpe\_updater.pyx](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_updater.pyx) file, which implements `fine_grained_pair_counter_diff` with Cython.

```
cpdef void fine_grained_pair_counter_diff(set affected_words, 
                                          word_encodings, 
                                          word_counts, 
                                          tuple merge_pair, 
                                          diff_pairs, 
                                          int new_id, 
                                          pair_to_words, 
                                          set new_pairs):
    cdef str word
    cdef int wc
    cdef int idx
    cdef int first_idx
    cdef int last_idx
    cdef int i
    cdef int tk_len

    for word in affected_words:
        word_tokens = word_encodings[word]
        wc = word_counts[word]

        # find first and last pairs
        idx = 0
        unaffected_pairs = set()
        tk_len = len(word_tokens)
        #first_idx = -1
        while idx < tk_len - 1:
            if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                first_idx = idx
                break
            idx += 1

        # assert first_idx exists

        idx = tk_len - 2
        while idx > first_idx + 1:
            if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                last_idx = idx
                break
            idx -= 1
        else:
            last_idx = first_idx

        start_idx = max_int(0, first_idx - 1) # inclusive
        end_idx = min_int(last_idx + 3, tk_len) # exclusive

        # unaffected [0, start_idx)


        for i in range(start_idx):
            pair = word_tokens[i], word_tokens[i + 1]
            unaffected_pairs.add(pair)
        # unaffected [end_idx-1, :-1]
        for i in range(end_idx - 1, tk_len - 1):
            pair = word_tokens[i], word_tokens[i + 1]
            unaffected_pairs.add(pair)                

        # TODO avoid slice copy
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

This code is essentially the same as the Python version, except that static type declarations have been added for the loop variables. This allows Cython to compile the loops into C-language versions, bypassing the Python iterator protocol.

Additionally, we can see in the code above:

```
affected_tokens = word_tokens[start_idx: end_idx]
```

This slice will copy the affected tokens. Our C++ implementation [code](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained.cpp%23L216) avoids this copy by using pointers:

```
        const int * affected_tokens = word_tokens.data() + start_idx;
        int affected_tokens_len = end_idx - start_idx;

        for(int i = 0; i < affected_tokens_len - 1; ++i){
            std::pair<int, int> old_pair(affected_tokens[i], affected_tokens[i + 1]);
            diff_pairs[old_pair] -= wc;
            if (unaffected_pairs.find(old_pair) == unaffected_pairs.end()) {
                pair_wordids[old_pair].erase(wordid);
            }
        }
```

Python's `list` cannot return a view, so we must manually handle the tedious indices to avoid copying. This gives us `fine_grained_pair_counter_diff_v2`. We won't go into the detailed changes here, but interested readers can refer to [fine\_grained\_pair\_counter\_diff\_v2](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_updater.pyx#L9).

Furthermore, based on the optimization from the previous article, we can remove `new_pairs`, which leads to [fine\_grained\_pair\_counter\_diff\_v3](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_updater.pyx#L205) based on `fine_grained_pair_counter_diff_v2`.

Next, we need to modify `setup.py`:

```
setup(
    packages=['cs336_basics'],
    name='cs336_basics.bpe_updater',
    ext_modules=cythonize("cs336_basics/bpe_updater.pyx"),
)
```

Then execute `python setup.py build_ext -i`, which compiles to `cs336_basics/bpe_updater.cpython-312-x86_64-linux-gnu.so`.

Finally, we use this `bpe_updater` extension module in our Python code, with the corresponding code in [bpe\_v8.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v8.py), [bpe\_v8\_v2.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v8_v2.py), and [bpe\_v8\_v3.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v8_v3.py). They call `fine_grained_pair_counter_diff`, `fine_grained_pair_counter_diff_v2`, and `fine_grained_pair_counter_diff_v3`, respectively.

## 4\. Test Results

Since gathering statistics for `max` and `update` introduces extra time overhead (e.g., `bpe_v7_time` is over 10 seconds slower than `bpe_v7`), I will only report the total merge time here.

| Version | Data | Total Time (s) | Word Freq Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v7\_time | open\_web | 1062/1035/1036 | 392/397/395 | total: 606/573/577<br>max:6/6/6<br>update: 599/567/571 | num\_counter=8, num\_merger=1 |
| bpe\_v7 | open\_web | 1051/1017/1023 | 399/389/398 | 590/568/568 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_opt | open\_web | 1021/1007/980 | 399/393/393 | 558/553/528 | num\_counter=8, num\_merger=1 |
| bpe\_v8 | open\_web | 934/930/935 | 390/393/393 | 479/472/476 | num\_counter=8, num\_merger=1 |
| bpe\_v8\_v2 | open\_web | 917/908/965 | 394/392/395 | 460/455/505 | num\_counter=8, num\_merger=1 |
| bpe\_v8\_v3 | open\_web | 897/899/951 | 395/399/395 | 442/438/493 | num\_counter=8, num\_merger=1 |

The implementation logic of `bpe_v8` and `bpe_v7` is identical. We can see that the Cython version `bpe_v8` has a merge time that is 17% faster than `bpe_v7`. `bpe_v8_v2` avoids copying `affected_tokens = word_tokens[start_idx: end_idx]`, and its first two runs are over 10 seconds faster than `bpe_v8`, but the third run is 30 seconds slower. The reason for this is unclear, perhaps due to fluctuations in server load. `bpe_v7_opt` made the `new_pairs` deletion optimization over `bpe_v7`, and it is compared with `bpe_v8_v3`, which also includes the copy-avoidance optimization from `bpe_v8_v2`. The average time for `bpe_v8_v3` is 457 seconds, which is 16% faster than `bpe_v7_opt`.

## 5\. Implementing the entire merge process with Cython

To go a step further, can implementing the entire merge process with Cython make it even faster? By refactoring `bpe_v8_v3`, I implemented [bpe\_updater\_v2.pyx](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_updater_v2.pyx) and [bpe\_v8\_v4.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v8_v4.py). Here, I encapsulated the merge process into a single function:

```
cpdef void bpe_train_step2(int vocab_size,
                      pair_counts,
                      pair_strings,
                      vocabulary,
                      pair_to_words,
                      word_counts,
                      word_encodings,
                      merges,
                      pair_heap):

    cdef int size = len(vocabulary)
    while size < vocab_size:
        _merge_a_pair(pair_counts, pair_strings, vocabulary,
                                pair_to_words, word_counts, word_encodings,
                                merges, size, pair_heap)
        size += 1
```

The test results are as follows:

| Version | Data | Total Time (s) | Word Freq Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v7\_time | open\_web | 1062/1035/1036 | 392/397/395 | total: 606/573/577<br>max:6/6/6<br>update: 599/567/571 | num\_counter=8, num\_merger=1 |
| bpe\_v7 | open\_web | 1051/1017/1023 | 399/389/398 | 590/568/568 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_opt | open\_web | 1021/1007/980 | 399/393/393 | 558/553/528 | num\_counter=8, num\_merger=1 |
| bpe\_v8 | open\_web | 934/930/935 | 390/393/393 | 479/472/476 | num\_counter=8, num\_merger=1 |
| bpe\_v8\_v2 | open\_web | 917/908/965 | 394/392/395 | 460/455/505 | num\_counter=8, num\_merger=1 |
| bpe\_v8\_v3 | open\_web | 897/899/951 | 395/399/395 | 442/438/493 | num\_counter=8, num\_merger=1 |
| bpe\_v8\_v4 | open\_web | 917/915/982 | 391/404/400 | 462/448/506 | num\_counter=8, num\_merger=1 |

`bpe_v8_v4` is even slower than `bpe_v8_v3`. The reason is that in the `fine_grained_pair_counter_diff` function, we used Cython's static type declarations to turn the Python iterator loop into a C loop, making it faster. However, the rest of the code operates on Python dictionaries. Implementing this code in Cython still reverts to Python, which is slower.

## 6\. Using PyPy

[PyPy](https://pypy.org/) is an alternative implementation of the Python language. Simply put, it's not a library or a framework, but a complete Python interpreter, just like the CPython we commonly use. PyPy's biggest highlight is its built-in **JIT (Just-In-Time) compiler**. This is the biggest difference from the standard CPython interpreter and the reason for its speed. CPython compiles Python code into bytecode, which is then executed one line at a time by a virtual machine. This process is relatively slow. PyPy also first compiles Python code into bytecode. However, while the program is running, PyPy's JIT compiler monitors which code is frequently executed. It then compiles this "hotspot" code directly into machine code and caches it. The next time it encounters the same code, PyPy executes the high-speed machine code directly instead of re-interpreting the bytecode. PyPy is quite similar to Java's JVM.

To switch the environment from CPython to PyPy, you don't need to modify any code, but you do need to reinstall the environment. Since we are using `uv` to manage our environment, switching is very simple; just run the following commands:

```
deactivate
UV_PROJECT_ENVIRONMENT=.venv_pypy uv sync --python pypy@3.11
source .venv_pypy/bin/activate
```

The first command exits the current virtual environment. Then, `uv sync` is used to create a new PyPy 3.11 environment. To avoid overwriting the original `.venv`, we use the `UV_PROJECT_ENVIRONMENT` environment variable to tell `uv` to create the new virtual environment in the `.venv_pypy` directory. Finally, we `source` this environment.

I first tested `bpe_v7`, and the results are as follows:

| Version | Data | Total Time (s) | Word Freq Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v7 | open\_web | 1028/1053/1045 | 395/397/393 | total: 575/589/590<br>max: 6/6/6<br>update: 569/583/583<br>make heap: 0.01/0.01<br>heap\_push\_time: 102/107/122 | num\_counter=8, num\_merger=1 |
| bpe\_v7(pypy) | open\_web | 2047/1694/1913 | 1644/1306/1552 | 403/388/361 | num\_counter=8, num\_merger=1 |

The results show that PyPy's overall runtime is much slower than CPython, increasing from over 1000 seconds to over 2000 seconds. However, PyPy's merge time is 34% faster than CPython. Why is there a speed discrepancy? Through a series of tests, I pinpointed the issue to the speed difference in regex matching, and then I wrote a program, [bpe\_v2\_time\_pypy.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v2_time_pypy.py). This program only executes the `_pretokenize_and_count` function and measures its time. The following are the results from testing on the `tinystory` dataset with a single CPU (`num_counter=1`) under PyPy 3.11, CPython 3.12, and CPython 3.11:

| Version | Data | split\_time (s) | match\_time (s) |
|---|---|---|---|
| bpe\_v2\_pypy(pypy 3.11) | tiny\_story | 40 | 888 |
| bpe\_v2\_pypy(cpython 3.12) | tiny\_story | 3 | 300 |
| bpe\_v2\_pypy(cpython 3.11) | tiny\_story | 3 | 288 |

`split_time` and `match_time` respectively measure the time for `regex.split` and `regex.finditer`:

```
        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            start_time = time.perf_counter()
            blocks = re.split(special_pattern, chunk)
            end_time = time.perf_counter()
            split_time += (end_time - start_time)
            for block in blocks:
                start_time = time.perf_counter()
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    text_len += len(text)
                end_time = time.perf_counter()
                match_time += (end_time - start_time)
```

As we can see, the PyPy version is much slower than CPython. To find the reason, I submitted an issue on the `regex` GitHub repository: [regex is much slower in pypy than cpython](https://github.com/mrabarnett/mrab-regex/issues/586). According to mattip's reply, the reason for PyPy's slowness is that `regex` uses CPython's C API, and PyPy can only implement this API through emulation, which is very slow. Not only is it slow, but the results can even be incorrect. According to the `regex` documentation:

> This module is targeted at CPython. It expects that all codepoints are the same width, so it won't behave properly with pypy outside U+0000..U+007F because pypy stores strings as UTF-8.

It's very unfortunate that while PyPy's JIT compilation does speed up the subsequent merge process, the unsupported `regex` library makes our overall PyPy run slower and even unusable. Of course, we could split the `train` function into two parts: the first part, `_pretokenize_and_count`, uses CPython, and the second part, `merge`, uses PyPy, with inter-process communication between them. However, inter-process communication is complex and data copying also introduces significant overhead.

Therefore, the biggest problem with PyPy is its lack of "compatibility" with CPython, while in reality, many high-performance third-party libraries use CPython's C API. For example, NumPy: PyPy cannot directly accelerate NumPy's underlying C code. Instead, it communicates with NumPy's C extension through a compatibility layer called `cpyext`. This compatibility layer is designed to allow PyPy to run CPython's C extensions. This means that when only using NumPy, PyPy can be even slower than CPython.

## 7\. Conclusion

Although PyPy can accelerate Python code execution, we have to give it up because of the lack of support from the third-party library `regex`. On the other hand, with Cython, we were able to achieve the fastest speed without modifying the Python code much (though some code was refactored for module calls). Of course, we can further optimize the Cython code, for example, by using C++'s `unordered_map` instead of Python's `dict`. However, a better approach is to perform these optimizations directly in C++ and then package them into a dynamic library for Cython to call. This will be the topic of our next article.
