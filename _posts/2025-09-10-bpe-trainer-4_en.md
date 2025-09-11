---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 4: A Failed Parallel Optimization" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the fifth article, documenting a failed attempt at parallel optimization. Through this failure, we can understand the problems that exist with Python's **`multiprocessing`** module and, in turn, learn when it should be used.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1. Algorithm Optimization

After a series of optimizations, our current algorithm (`bpe_v3_time`) on OpenWeb now takes only 120 seconds to perform the first step—token frequency counting (on a 32-core machine). However, the second step, merging, still takes over 30,000 seconds. This article focuses on optimizing the second step.

The second step is divided into two parts: finding the pair with the highest frequency using the `max` function, and incrementally updating the pair counts. The `max` operation alone takes 30,000 seconds, while the incremental update takes only a few hundred seconds. This analysis clearly shows that finding the most frequent pair is the part we should prioritize for optimization.

We use Python's built-in `max` function, whose source code can be found [here](https://github.com/python/cpython/blob/v3.12.11/Python/bltinmodule.c#L1872). It ultimately calls the `min_max` function, located [here](https://github.com/python/cpython/blob/v3.12.11/Python/bltinmodule.c#L1745C1-L1745C8):

```
PyObject *
min_max(PyObject *iterable, PyObject *args, PyObject *kwds, int op)
{
    // ... code ...
}
```

The code for this function is quite long, and much of it is related to the Python/C API. The core algorithm, however, is very simple and consists of the following snippet:

```
    while (( item = PyIter_Next(it) )) {
        /* get the value from the key function */
        if (keyfunc != NULL) {
            val = PyObject_CallOneArg(keyfunc, item);
            if (val == NULL)
                goto Fail_it_item;
        }
        /* no key function; the value is the item */
        else {
            val = Py_NewRef(item);
        }

        /* maximum value and item are unset; set them */
        if (maxval == NULL) {
            maxitem = item;
            maxval = val;
        }
        /* maximum value and item are set; update them as necessary */
        else {
            int cmp = PyObject_RichCompareBool(val, maxval, op);
            if (cmp < 0)
                goto Fail_it_item_and_val;
            else if (cmp > 0) {
                Py_DECREF(maxval);
                Py_DECREF(maxitem);
                maxval = val;
                maxitem = item;
            }
            else {
                Py_DECREF(item);
                Py_DECREF(val);
            }
        }
    }
```

The logic is as follows: It iterates through each `item` of the iterator `it`. For each `item`, it calls a `keyfunc` (if a `key` argument was passed, which we did), to get `val`. If it's the first element, `maxval` is set to `val`. Otherwise, `val` is compared to the current `maxval` using `PyObject_RichCompareBool(val, maxval, op)`. The third argument, `op`, specifies the comparison operator; we pass `Py_GT` to check if `val` is greater than `maxval`. If the return value is greater than 0, `val > maxval`, and `maxval` and `maxitem` are updated.

The built-in `max` function is implemented in C, making it very difficult to create a faster version at the Python level. So, what's the solution?

The most obvious approach, similar to our previous successful optimization, is to use a parallel algorithm. The `max` function is associative (`max(max(a,b),c) = max(a,max(b,c))`), which makes it easy to parallelize. For example, with N CPUs, we can split all the pairs into N chunks, find the local maximum in each chunk, and then find the global maximum from those N local maximums.



## 2\. Using `multiprocessing.Process` to Parallelize the `max` Operation

My first attempt was to use the successful strategy from the previous article: parallelizing the `max` operation with multiple processes. The code for this approach is in [`bpe_v4_mp.py`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp.py). Let's look at the changes relative to `bpe_v3.py`.

```
    @staticmethod
    def _merge_a_pair(pair_counts, pair_strings, vocabulary, pair_to_words, 
                   word_counts, word_encodings, merges, size, num_processes):
        _, _, merge_pair = BPE_Trainer._parallel_max(pair_counts, pair_strings, num_processes)
        
        ....
```

The main change is the call to `BPE_Trainer._parallel_max`. Let's examine this function:

```
    @staticmethod
    def _parallel_max(pair_counts, pair_strings, num_processes):
        # need a copy
        pair_counts = list(pair_counts.items())
        chunk_size = len(pair_counts) // num_processes
        data_chunks = [pair_counts[i:i + chunk_size] for i in range(0, len(pair_counts), chunk_size)]
        

        if len(data_chunks) > num_processes:
            data_chunks[num_processes - 1].extend(data_chunks.pop())

        processes = []
        queue = mp.Queue()
        for i in range(num_processes):
            p = mp.Process(target=BPE_Trainer._find_max_pair, 
                        args=(pair_strings, data_chunks[i], queue),
                        name=f"find_max_pair-{i+1}")
            p.start()
            processes.append(p)        
        for p in processes:
            p.join()

        local_maxes = []
        for i in range(num_processes):
            local_maxes.append(queue.get())

        return max(local_maxes)
```

To split the tasks, we first need to copy the contents of the `pair_counts` dictionary into a randomly accessible list, which requires an extra copy operation. The list is then divided into `num_processes` chunks. For example, if `pair_counts` has a size of 10 and `num_processes` is 4, the tasks are split so that the first three processes get 10 // 4 = 2 tasks each, and the remainder goes to the last process, resulting in a task distribution of [2, 2, 2, 4]. While not the most optimal distribution, it is acceptable since our number of processes is small (typically equal to the number of CPU cores) and `pair_counts` is large (hundreds of thousands of items).

To collect the results from each process, we create a `multiprocessing.Queue` to store the local maximum values. We then create `num_processes` processes to execute `BPE_Trainer._find_max_pair`. Once all processes finish, we collect their results into `local_maxes` and find the global maximum.

The `BPE_Trainer._find_max_pair` code is:

```
    @staticmethod
    def _find_max_pair(pair_strings, pair_counts, queue):
        max_pair, max_count = max(pair_counts, key = lambda x: (x[1], pair_strings[x[0]]))
        queue.put((max_count, pair_strings[max_pair], max_pair))
```

This is our original `max` function, but it puts the result into the queue.

For convenience during testing, a command-line argument was added to set the number of concurrent `max` processes:

```
        parser.add_argument("--num_max", 
                            type=int, 
                            default=NUM_MAX_PROCESS, 
                            help="number of processes for max")
```



## 3\. Test Results for `multiprocessing.Process`

To measure the performance, I implemented [`bpe_v4_mp_time.py`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp_time.py). The initial tests showed that it was incredibly slow on OpenWeb, so I switched to the smaller TinyStories dataset. The results are below:

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe_v3_time      | tinystory | 211/206/208 | 90/90/90 |total:120/115/117 <br> max_time: 117/112/115 <br> update_time: 3/3/3 |num_counter=8, num_merger=1
bpe_v4_mp_time      | tinystory | 730/715/740 | 80/90/80 |merge time: 650/624/659 <br> copy_time: 109/127/123 <br>max_time: 459/404/443 <br> compute_time: 221/238/231 |num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_time      | tinystory | 931/943/1007 | 90/90/90 |merge time: 841/852/917 <br> copy_time: 146/157/136 <br>max_time: 600/593/691 <br> compute_time: 125/128/118 |num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_time      | tinystory | 1354/1405/1431 | 80/90/80 |merge time: 1274/1315/1351 <br> copy_time: 143/158/160 <br>max_time: 1039/1055/1090 <br> compute_time: 80/85/86 |num_counter=8, num_merger=1 num_max=8

Here are the time metrics:

`merge time` is the total time for all merge operations:

```
        start_time = time.perf_counter()
        while size < vocab_size:
            BPE_Trainer._merge_a_pair(pair_counts, pair_strings, vocabulary,
                                pair_to_words, word_counts, word_encodings,
                                merges, size, num_max_processes, times)
            size += 1
        end_time = time.perf_counter()
        print(f"merge time: {end_time - start_time:.2f}s")
```

`copy_time` is the time to copy `dict.items()` to a list:

```
        start_time = time.perf_counter()
        pair_counts = list(pair_counts.items())
        end_time = time.perf_counter()
        times[0] += (end_time - start_time)
```

`max_time` is the total time for all parallel `max` operations:

```
        start_time = time.perf_counter()
        processes = []
        queue = mp.Queue()
        for i in range(num_processes):
            p = mp.Process(target=BPE_Trainer._find_max_pair, 
                        args=(pair_strings, data_chunks[i], queue),
                        name=f"find_max_pair-{i+1}")
            p.start()
            processes.append(p)        
        for p in processes:
            p.join()
        end_time = time.perf_counter()
        times[1] += (end_time - start_time)
```

`compute_time` is the maximum time taken by a single process to find its local maximum:

```
        local_maxes = []
        local_times = []
        for i in range(num_processes):
            t = queue.get()
            local_times.append(t[-1])
            local_maxes.append(t[:-1])
        times[2] += max(local_times)
```



The actual time calculation happens within the child process and is returned to the main process as a parameter:

```
    @staticmethod
    def _find_max_pair(pair_strings, pair_counts, queue):
        start_time = time.perf_counter()
        max_pair, max_count = max(pair_counts, key = lambda x: (x[1], pair_strings[x[0]]))
        end_time = time.perf_counter()
        queue.put((max_count, pair_strings[max_pair], max_pair, (end_time-start_time)))
```



## 4\. Analysis of the Results

Here are some key takeaways from the results:

  * The `merge time` for `bpe_v3_time` is only about 120 seconds, but for `bpe_v4_mp_time` it's around 700 seconds with 1 process, over 900 seconds with 4 processes, and more than 1,300 seconds with 8 processes.
  * The `max_time` for `bpe_v3_time` is about 110 seconds, whereas `bpe_v4_mp_time` takes over 400 seconds for 1 process, 600+ seconds for 4 processes, and 1000+ seconds for 8 processes.
  * The `compute_time` for `bpe_v4_mp_time` does decrease with more processes, from over 220 seconds to 120+ and 80+ seconds for 1, 4, and 8 processes, respectively.
  * The `copy_time` for `bpe_v4_mp_time` is between 120 and 160+ seconds.

The results clearly show that as the number of processes increases, the total time spent also increases. While the actual computation time (`compute_time`) does decrease, the total `max_time` increases significantly. This suggests that the overhead of **creating and destroying processes** is extremely high. Our previous `_pretokenize_and_count_mp` function only created and destroyed processes once, with most of the time spent on regex and counting. However, here, `_merge_a_pair` is called 10,000 times, which means `_parallel_max` is called 10,000 times, leading to 10,000 process creations and destructions. This cost is substantial.

Furthermore, for a parallel algorithm to work, we need to be able to randomly access `pair_counts`. Since Python's `dict` only allows sequential access via an iterator, we need to copy it to a list. This copy operation itself can take longer than the `max` operation. We saw from the `max` source code that it has a time complexity of O(n) and a space complexity of O(1), as it only needs to iterate once to find the maximum value. The copy operation also has O(n) time complexity and O(n) space complexity. To verify this, I wrote [`bpe_v4_time2.py`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_time2.py) to compare the `max` and copy times.

```
        start_time = time.perf_counter()
        merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
        end_time = time.perf_counter()
        times[0] += (end_time - start_time)

        start_time = time.perf_counter()
        merge_pair3, max_count3 = max(pair_counts.items(), key = lambda x: x[1])
        end_time = time.perf_counter()
        times[4] += (end_time - start_time)        
        
        start_time = time.perf_counter()
        pair_counts_list = list(pair_counts.items())
        end_time = time.perf_counter()
        times[1] += (end_time - start_time)

        start_time = time.perf_counter()
        pair_strings_list = [pair_strings[pair] for pair in pair_counts]
        end_time = time.perf_counter()
        times[2] += (end_time - start_time)   
```

The results showed that after only 7,000 merges, `max` took 70 seconds, while copying to a list took 55 seconds. When you add the overhead of process creation and destruction, this approach is clearly not viable.

The biggest issue with this approach isn't just the overhead of process creation/destruction; it's also that Python's dictionaries cannot be iterated in parallel. If the computation were very large, we could copy the dictionary to a list and use multiple CPUs to compute, and the parallel speed-up would compensate for the copy overhead. However, our `max` algorithm is primarily I/O-bound (iterating) with simple CPU comparisons, so this strategy is not worthwhile.

At this point, we could have abandoned this approach. However, a few questions remain:

  * Why is the `compute_time` for `bpe_v4_mp_time` with a single process (220+) so much slower than `bpe_v3_time` (110+)? One would expect iterating a list to be faster than iterating a dictionary.
  * Does using `multiprocessing.Pool` solve the high overhead of process creation and destruction?

To explore these questions, I tried implementing the parallel `max` algorithm using `multiprocessing.Pool`.



## 5\. Using `multiprocessing.Pool` for Parallel `max`

The full code is in [`bpe_v4_time.py`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_time.py).

### The `train` function

```
        with mp.Pool(processes=num_max_processes) as pool:
            times = [0] * 4
            start_time = time.perf_counter()
            while size < vocab_size:
                BPE_Trainer._merge_a_pair(pair_counts, pair_strings, vocabulary,
                                    pair_to_words, word_counts, word_encodings,
                                    merges, size, pool, num_max_processes, times)
                size += 1
      
            end_time = time.perf_counter()
            print(f"merge time: {end_time - start_time:.2f}s")
            print(f"copy_time: {times[0]:.2f}s, max_time: {times[1]:.2f}s, compute_time: {times[2]:.2f}s, copy_trunks_time: {times[3]:.2f}s")
```

We use `mp.Pool(processes=num_max_processes)` to create a fixed-size process pool. This approach avoids the need to create and destroy processes on every call to `_merge_a_pair`.

The `BPE_Trainer._merge_a_pair` function calls `BPE_Trainer._parallel_max`, which we'll examine next.

### `BPE_Trainer._parallel_max`

```
    @staticmethod
    def _parallel_max(pair_counts, pair_strings, pool, num_processes, times):
        # need a copy
        start_time = time.perf_counter()
        pair_counts = list(pair_counts.items())
        end_time = time.perf_counter()
        times[0] += (end_time - start_time)

        start_time = time.perf_counter()
        chunk_size = len(pair_counts) // num_processes
        data_chunks = [pair_counts[i:i + chunk_size] for i in range(0, len(pair_counts), chunk_size)]

        if len(data_chunks) > num_processes:
            data_chunks[num_processes - 1].extend(data_chunks.pop())
        end_time = time.perf_counter()
        times[3] += (end_time - start_time)

        start_time = time.perf_counter()
        find_max_pair = partial(BPE_Trainer._find_max_pair, pair_strings)
        local_maxes = pool.map(find_max_pair, data_chunks)
        end_time = time.perf_counter()
        times[1] += (end_time - start_time)

        computing_time = max([r[-1] for r in local_maxes])
        times[2] += computing_time
            
        return max(local_maxes)[:-1]
```

The first step is still to copy `pair_counts.items()` to a list. Next, the list is divided into `data_chunks`, just as before. I've added timing for this step, though it's negligible.

Our child process function is `_find_max_pair`:

```
    @staticmethod
    def _find_max_pair(pair_strings, pair_counts):
        start_time = time.perf_counter()
        max_pair, max_count = max(pair_counts, key = lambda x: (x[1], pair_strings[x[0]]))
        end_time = time.perf_counter() 
        return max_count, pair_strings[max_pair], max_pair, (end_time - start_time)
```

This is the actual function that finds the maximum value. To track the time, the run time is included in the return value.

Then, we call `Pool.map` to submit the tasks to the process pool. The first argument is the function to be called, and the second is an iterable of arguments for that function. Our `BPE_Trainer._find_max_pair` needs two arguments: `pair_strings` and `data_chunks[i]`. We could pass them as a tuple, but since `pair_strings` is the same for all child processes, we can use `functools.partial` to create a partial function that "freezes" this argument:

```
        find_max_pair = partial(BPE_Trainer._find_max_pair, pair_strings)
        local_maxes = pool.map(find_max_pair, data_chunks)
```

This makes the code cleaner. When `Pool.map` sends a task to a process, the arguments are serialized (using `pickle`), sent via inter-process communication (IPC), and then deserialized. While it might seem that using `partial` reduces the data being passed, the `find_max_pair` function itself, with `pair_strings` embedded, still needs to be transferred to each process in the pool.



## 6\. Test Results for `multiprocessing.Pool`

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe_v3_time      | tinystory | 211/206/208 | 90/90/90 |total:120/115/117 <br> max_time: 117/112/115 <br> update_time: 3/3/3 |num_counter=8, num_merger=1
bpe_v4_mp_time      | tinystory | 730/715/740 | 80/90/80 |merge time: 650/624/659 <br> copy_time: 109/127/123 <br>max_time: 459/404/443 <br> compute_time: 221/238/231 |num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_time      | tinystory | 931/943/1007 | 90/90/90 |merge time: 841/852/917 <br> copy_time: 146/157/136 <br>max_time: 600/593/691 <br> compute_time: 125/128/118 |num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_time      | tinystory | 1354/1405/1431 | 80/90/80 |merge time: 1274/1315/1351 <br> copy_time: 143/158/160 <br>max_time: 1039/1055/1090 <br> compute_time: 80/85/86 |num_counter=8, num_merger=1 num_max=8
bpe_v4_time      | tinystory | 981/967/960 | 90/90/90 |merge time: 890/876/870 <br> copy_time: 91/92/88 <br>max_time: 761/743/744 <br> compute_time: 79/79/81 <br> copy_trunks_time: 2/2/2 |num_counter=8, num_merger=1 num_max=1
bpe_v4_time      | tinystory | 1784/1688/1543 | 90/80/90 |merge time: 1693/1607/1452 <br> copy_time: 149/151/97 <br>max_time: 1483/1393/1311 <br> compute_time: 26/26/25 <br> copy_trunks_time: 3/3/2 |num_counter=8, num_merger=1 num_max=4
bpe_v4_time      | tinystory | 2660/2399/2503 | 90/90/90 |merge time: 2569/2308/2412 copy_time: 165/95/119 <br>max_time: 2336/2169/2238 <br> compute_time: 16/15/16 <br> copy_trunks_time: 3/2/2 |num_counter=8, num_merger=1 num_max=8



## 7\. Analysis of the Results

Here are a few observations:

  * `Pool` is even slower than `Process`. The total time (taking the median) for 1, 4, and 8 processes is roughly 900/1600/2500 seconds for `Pool` vs. 700/900/1300 seconds for `Process`.
  * `Pool`'s `compute_time` decreases as the number of processes increases. Interestingly, `Pool`'s `compute_time` for a single process is less than both `Process` and `bpe_v3_time`.
  * `Pool`'s `max_time` is significantly slower than `Process`.

The first and third results were unexpected. I thought that `Pool`, by reducing process creation/destruction overhead, would be faster than `Process`.

The second result is as expected, since `Pool`'s `max` iterates over a list, which should be faster than `bpe_v3_time`'s `dict` iteration.

However, the question remains: why is `Pool` slower than `Process` and why is `Pool`'s `compute_time` faster than `Process`'s?

After some research, I found the reason for `Pool` being slower than `Process`. For `Process`, I can only offer a guess.

My guess for `Process`'s slower `compute_time` is that a new process is created in each loop iteration, possibly on a different CPU core. This leads to cache misses when iterating through `pair_counts` as the data needs to be fetched from main memory. In contrast, `Pool` reuses processes. Since `pair_counts` is updated incrementally, and we always submit tasks in the same order, a given chunk of data is likely always handled by the same process running on the same core, which could improve cache hit rates. This is just a hypothesis, and to verify it, we would need to either shuffle the data chunks before submitting them to `Pool` or explicitly bind `Process` instances to specific CPUs using something like `psutil.Process.cpu_affinity()`.

The reason for the difference in `max_time` and total time between `Process` and `Pool` is that **`Process` (on Linux by default) uses `fork` to create child processes**. A child process created via `fork` inherits the parent's address space, avoiding the need for inter-process data copying. `Pool`, on the other hand, must go through three steps for data transfer: serialization via `pickle`, IPC communication, and deserialization.

I learned this key distinction between `Process` and `Pool` from this very problem. Many online resources are vague or incorrect about this. Let's verify this conclusion with some code.

The code to test this is [here](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_mp_fork.py).

```
from multiprocessing import Process, set_start_method

class Unpickleable:
    def __reduce__(self):    
        raise TypeError("I hate pickle")

    def __str__(self) -> str:
        return "Unpickleable"

def func(obj):
    print(obj)

if __name__ == '__main__':
    #set_start_method('spawn')
    set_start_method('fork')
    o = Unpickleable()
    p = Process(target=func, args=((o,)))
    p.start()
    p.join()   
```

We define an `Unpickleable` class that raises an exception if `pickle` is attempted. If we set the start method to `fork`, the code runs without error, demonstrating that `fork` inherits the parent's objects without serialization. If we set it to `spawn`, it raises an exception, proving that `spawn` requires serialization.

**Note**: `fork` is faster, but it has issues in multithreaded environments. As of Python 3.12, it issues a `DeprecationWarning` if the main process is multithreaded. The default start method will change to `spawn` in Python 3.14 on all platforms. If you are running on a newer Python version, you might need to manually set the start method to `fork`.



## 8\. Verifying the `multiprocessing.Process` Algorithm with `spawn`

Based on our analysis, `Process` has high overhead from creation/destruction and poor cache utilization. The only reason it was faster than `Pool` was `fork`'s lack of IPC data transfer. To test this, I modified the `Process`-based algorithm to use `spawn`. If my reasoning is correct, this version should be the slowest of all. The code is in [`bpe_v4_mp_spawn_time.py`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp_spawn_time.py), with the only change being:

```
mp.set_start_method('spawn')
```

The results are below:

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe_v3_time      | tinystory | 211/206/208 | 90/90/90 |total:120/115/117 <br> max_time: 117/112/115 <br> update_time: 3/3/3 |num_counter=8, num_merger=1
bpe_v4_mp_time      | tinystory | 730/715/740 | 80/90/80 |merge time: 650/624/659 <br> copy_time: 109/127/123 <br>max_time: 459/404/443 <br> compute_time: 221/238/231 |num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_time      | tinystory | 931/943/1007 | 90/90/90 |merge time: 841/852/917 <br> copy_time: 146/157/136 <br>max_time: 600/593/691 <br> compute_time: 125/128/118 |num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_time      | tinystory | 1354/1405/1431 | 80/90/80 |merge time: 1274/1315/1351 <br> copy_time: 143/158/160 <br>max_time: 1039/1055/1090 <br> compute_time: 80/85/86 |num_counter=8, num_merger=1 num_max=8
bpe_v4_time      | tinystory | 981/967/960 | 90/90/90 |merge time: 890/876/870 <br> copy_time: 91/92/88 <br>max_time: 761/743/744 <br> compute_time: 79/79/81 <br> copy_trunks_time: 2/2/2 |num_counter=8, num_merger=1 num_max=1
bpe_v4_time      | tinystory | 1784/1688/1543 | 90/80/90 |merge time: 1693/1607/1452 <br> copy_time: 149/151/97 <br>max_time: 1483/1393/1311 <br> compute_time: 26/26/25 <br> copy_trunks_time: 3/3/2 |num_counter=8, num_merger=1 num_max=4
bpe_v4_time      | tinystory | 2660/2399/2503 | 90/90/90 |merge time: 2569/2308/2412 copy_time: 165/95/119 <br>max_time: 2336/2169/2238 <br> compute_time: 16/15/16 <br> copy_trunks_time: 3/2/2 |num_counter=8, num_merger=1 num_max=8
bpe_v4_mp_spawn_time      | tinystory | 2317/2277/2318 | 90/90/90 |merge time: 2227/2186/2227 copy_time: 160/148/157 <br> max_time: 2003/1978/2007 <br> compute_time: 81/81/81 | num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_spawn_time      | tinystory | 6193/6173/6218 | 90/90/90 |merge time: 6102/6083/6127 copy_time: 160/146/161 <br> max_time: 5877/5876/5901 <br> compute_time: 26/26/26 | num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_spawn_time      | tinystory | 11071 | 90 |merge time: 10980 copy_time: 145 <br> max_time: 10774 <br> compute_time: 15 | num_counter=8, num_merger=1 num_max=8

This experiment confirms that `spawn` is much slower than `fork`, which is what we expected.

However, the `compute_time` is similar to `Pool`'s, which seems to contradict my cache-related hypothesis for why `Process` was slow.

If that's not the reason, why is iterating the `pair_counts` list in a `fork`-created child process slower than in a `spawn`- or `Pool`-created process? The key difference is that a `fork`-ed child process reads from the parent's `list(pair_counts.items())` (a shallow copy), while `spawn`- and `Pool`-created processes get a deep copy of the data via `pickle`. This deep copy might result in a more memory-compact list layout, leading to faster iteration.

To test this, I implemented [`bpe_v4_mp_deepcopy_time.py`](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp_deepcopy_time.py), which manually serializes and deserializes the list to mimic the deep copy. The only difference from `bpe_v4_mp_time.py` is:

```
        start_time = time.perf_counter()
        pair_counts = list(pair_counts.items())
        pickled_bytes = pickle.dumps(pair_counts)
        pair_counts = pickle.loads(pickled_bytes)
        end_time = time.perf_counter()
```

The results are below:

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe_v3_time      | tinystory | 211/206/208 | 90/90/90 |total:120/115/117 <br> max_time: 117/112/115 <br> update_time: 3/3/3 |num_counter=8, num_merger=1
bpe_v4_mp_time      | tinystory | 730/715/740 | 80/90/80 |merge time: 650/624/659 <br> copy_time: 109/127/123 <br>max_time: 459/404/443 <br> compute_time: 221/238/231 |num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_time      | tinystory | 931/943/1007 | 90/90/90 |merge time: 841/852/917 <br> copy_time: 146/157/136 <br>max_time: 600/593/691 <br> compute_time: 125/128/118 |num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_time      | tinystory | 1354/1405/1431 | 80/90/80 |merge time: 1274/1315/1351 <br> copy_time: 143/158/160 <br>max_time: 1039/1055/1090 <br> compute_time: 80/85/86 |num_counter=8, num_merger=1 num_max=8
bpe_v4_time      | tinystory | 981/967/960 | 90/90/90 |merge time: 890/876/870 <br> copy_time: 91/92/88 <br>max_time: 761/743/744 <br> compute_time: 79/79/81 <br> copy_trunks_time: 2/2/2 |num_counter=8, num_merger=1 num_max=1
bpe_v4_time      | tinystory | 1784/1688/1543 | 90/80/90 |merge time: 1693/1607/1452 <br> copy_time: 149/151/97 <br>max_time: 1483/1393/1311 <br> compute_time: 26/26/25 <br> copy_trunks_time: 3/3/2 |num_counter=8, num_merger=1 num_max=4
bpe_v4_time      | tinystory | 2660/2399/2503 | 90/90/90 |merge time: 2569/2308/2412 copy_time: 165/95/119 <br>max_time: 2336/2169/2238 <br> compute_time: 16/15/16 <br> copy_trunks_time: 3/2/2 |num_counter=8, num_merger=1 num_max=8
bpe_v4_mp_spawn_time      | tinystory | 2317/2277/2318 | 90/90/90 |merge time: 2227/2186/2227 copy_time: 160/148/157 <br> max_time: 2003/1978/2007 <br> compute_time: 81/81/81 | num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_spawn_time      | tinystory | 6193/6173/6218 | 90/90/90 |merge time: 6102/6083/6127 copy_time: 160/146/161 <br> max_time: 5877/5876/5901 <br> compute_time: 26/26/26 | num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_spawn_time      | tinystory | 11071 | 90 |merge time: 10980 copy_time: 145 <br> max_time: 10774 <br> compute_time: 15 | num_counter=8, num_merger=1 num_max=8
bpe_v4_mp_deepcopy_time      | tinystory | 1262/1393/1377 | 90/90/90 |merge time: 1171/1302/1287 copy_time: 590/668/644 <br> max_time: 492/538/549 <br> compute_time: 278/303/298 | num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_deepcopy_time      | tinystory | 1529/1481/1525 | 80/90/90 |merge time: 1449/1391/1434 copy_time: 661/712/680 <br> max_time: 687/571/651 <br> compute_time: 115/121/116 | num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_deepcopy_time      | tinystory | 1924/1872/1983 | 80/90/80 |merge time: 1843/1782/1903 copy_time: 730/682/721 <br> max_time: 1000/994/1070 <br> compute_time: 72/68/70 | num_counter=8, num_merger=1 num_max=8

The `compute_time` for `bpe_v4_mp_deepcopy_time` is similar to `bpe_v4_mp_time` (around 200+ seconds), which again refutes my guess. `pickle` serialization/deserialization doesn't seem to make list iteration faster.

A final question remains: Why is the subprocess created with `fork` slower at iterating through the `count_pairs` list than one created with `spawn` or `Pool`? A search on Gemini provided this answer:

A `fork` subprocess doesn't need to copy the arguments passed from the parent process. If the subprocess modifies a corresponding memory page, the operating system will create a private copy of that page. This is known as **Copy-on-Write (CoW)**. However, since our subprocess's traversal of `count_pairs` should be read-only, it shouldn't, in theory, trigger the CoW mechanism. I searched Gemini, and its reference answer is as follows.

Even if the subprocess only performs a read operation on `count_pairs`, it can still trigger Copy-on-Write (CoW). While this seems counterintuitive, the reason is typically related to the Python interpreter and underlying memory management mechanisms.

When a subprocess created with `fork` inherits the parent process's memory space, it gets read-only access to the `count_pairs` data. Ideally, if the subprocess only iterates through the data, it shouldn't trigger CoW. However, several scenarios can break this ideal state:

**1. Garbage Collection**

This is the most common and subtle reason. When the Python interpreter runs in the subprocess, it performs its own memory management and garbage collection. To ensure correct object reference counts or to mark and clear objects that are no longer in use, the interpreter may make small modifications to certain metadata in memory, such as reference counts.

* Python Objects: In Python, every object has an `ob_refcnt` (reference count) field.
* Subprocess Reading: When the subprocess iterates through the tuples in `count_pairs`, the interpreter might internally need to inspect or update the reference counts of these tuples.
* Triggering CoW: Even updating a single byte of a reference count is considered a modification by the operating system. To maintain the parent-child processes' independence, the kernel immediately creates a private copy of the entire memory page containing that object.

**2. Memory Alignment and Cache Lines**

When a processor accesses memory, it does so in units called **cache lines** (typically 64 bytes). When a memory page is accessed, the entire cache line is loaded into the processor's cache. If this cache line contains data that needs to be modified (even if it belongs to other, unrelated objects), the entire memory page may be flagged for copying.

**3. JIT Compilers and Other System Calls**

If the subprocess uses a Just-In-Time (JIT) compiler or other complex libraries, these libraries may perform internal memory operations that unintentionally trigger CoW. Even a simple memory access can have a complex interaction with the kernel's memory management.

**Conclusion**

The read operation on `count_pairs` in the subprocess triggers CoW not because of an explicit modification in your code, but because of small, background modifications made by the Python interpreter to maintain its internal memory state (most commonly, reference counts for garbage collection).

The `spawn` method doesn't have this issue because it creates a separate memory copy from the beginning. While this copying process (serialization and deserialization) is time-consuming at startup, once complete, the subprocess has its own fully independent memory space. Subsequent operations will not be affected by or affect the parent process's memory, so there's no CoW overhead.

To verify this, I wrote a program to test it:

```
import time
import multiprocessing as mp

def worker(lst):
    t0 = time.perf_counter()
    for x in lst:       
        pass
    print(f"traverse time: {time.perf_counter() - t0:.2f}s")

if __name__ == '__main__':
    mp.set_start_method('spawn')           
    lst = list(zip(range(5_000_000), range(5_000_000)))
    start_time = time.perf_counter()
    p = mp.Process(target=worker, args=(lst,))
    p.start()
    p.join()
    end_time = time.perf_counter()
    print(f"total time: {end_time - start_time:.2f}s")
```

With the `fork` method, the total time was 1.07s and the subprocess traversal time was 0.28s. With the `spawn` method, the total time was 4.8s and the subprocess traversal time was 0.08s. Therefore, even with the CoW issue, `fork` is still faster overall than `spawn`.



## 9\. Conclusion

This exploration shows that Python's `multiprocessing` is not suitable for tasks with frequent I/O and CPU overlap. While using multiple CPUs does reduce the `compute_time` for the `max` operation, the large overhead of inter-process communication makes it counterproductive. This type of problem is better suited for multithreading, where threads within the same process can share memory, thus avoiding IPC overhead. However, due to Python's Global Interpreter Lock (GIL), we cannot use multithreading to solve this.

Therefore, we will now turn to C++ to attempt to parallelize the `max` function using multithreading.
