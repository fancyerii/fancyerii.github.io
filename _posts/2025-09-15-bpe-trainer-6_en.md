---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 6: Implement parallel maximum finding with OpenMP" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the seventh article, exploring a method for using OpenMP to find the pair with the highest count in pair_counts in parallel.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Optimization Method

In the previous article, we implemented an algorithm to randomly traverse `std::unordered_map` using the bucket interface, which laid a good foundation for parallelizing the max algorithm. Furthermore, test results on the OpenWeb dataset showed that `bpe_train_updater_omp_v3` took over 400 seconds for updating data structures (serial), while the max algorithm (parallelizable) took over 6000 seconds. The time spent in the parallelizable code accounted for over 90% of the total time. Therefore, if we can parallelize the max algorithm, the total time will be significantly reduced.

## 2\. OpenMP

Since a large part of the parallel algorithm involves traversing the `std::unordered_map` `pair_counts`, and `pair_counts` is modified after each merge, a shared-memory parallel method like multithreading is the most suitable approach. The C++ standard library provides basic multithreading tools such as `std::thread`, `std::mutex`, `std::condition_variable`, `std::promise`, and `std::future`. The C++ standard library tends to provide low-level, general-purpose building blocks, so it lacks a thread pool and thread-safe queue. It is not easy to implement a correct and efficient thread pool and queue yourself. Some third-party libraries provide these tools, but introducing additional libraries increases dependencies, which would be more complicated when integrating with Cython later. More importantly, tools like thread pools and queues are still too low-level. Our task is relatively simple: we just want to parallelize a `for` loop to find the maximum value using multiple threads. OpenMP is very suitable for such a task. Unlike other multithreading libraries, OpenMP provides simple `#pragma omp xxx` compiler preprocessor directives. We use these directives to tell the compiler that our `for` loop needs to be parallelized, and OpenMP will automatically rewrite our code into a parallel version. It will automatically manage thread and task allocation. Many parameters, such as the number of threads and the dynamically allocated `chunk_size`, can be set through environment variables, which makes experimenting with different parameters very simple.

We will not introduce too much about OpenMP here. Readers who are unfamiliar with it can learn from the video course [Introduction to OpenMP](https://www.youtube.com/watch%3Fv%3DnE-xN4Bf8XI%26list%3DPLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG%26index%3D1). This video course is very suitable for beginners, as each video is short and provides hands-on exercises. If you only want to understand this article, knowing about `#pragma omp parallel` and `#pragma omp for schedule` is enough.

## 3\. Manual Task Splitting

Let's first try to implement the parallel algorithm by manually splitting tasks. The complete code is in [bpe\_train\_updater\_omp\_v4.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v4.cpp).

The main code is:

```
        int max_count = -1;
        std::pair<int, int> max_pair;
        std::vector<std::vector<int>> max_strings;
        size_t num_buckets = pair_counts.bucket_count();


        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int threads = omp_get_num_threads();
            int local_max_count = -1;
            std::pair<int, int> local_max_pair;
            std::vector<std::vector<int>> local_max_strings; 
            int buckets_per_threads = num_buckets / threads;
            int start_bucket = thread_id * buckets_per_threads; //inclusive
            int end_bucket = start_bucket + buckets_per_threads; //exclusive
            if(thread_id == threads - 1){
                end_bucket = num_buckets;
            }
            auto start = std::chrono::steady_clock::now();
            for (int i = start_bucket; i < end_bucket; ++i) {
                for (auto it = pair_counts.begin(i); it != pair_counts.end(i); ++it) {
                    const auto& pair = it->first;
                    const auto& count = it->second;
                    
                    if(count > local_max_count){
                        local_max_count = count;
                        local_max_pair = pair;
                        local_max_strings = pair_strings[pair];
                    }else if(count == local_max_count){
                        std::vector<std::vector<int>> strings = pair_strings[pair];
                        ComparisonResult r1 = three_way_compare(strings[0], local_max_strings[0]);
                        if(r1 == ComparisonResult::Greater){
                            local_max_count = count;
                            local_max_pair = pair;
                            local_max_strings = strings;
                        }else if(r1 == ComparisonResult::Equal){
                            ComparisonResult r2 = three_way_compare(strings[1], local_max_strings[1]);
                            if(r2 == ComparisonResult::Greater){
                                local_max_count = count;
                                local_max_pair = pair;
                                local_max_strings = strings;                    
                            }
                        }
                    }
                }
                
            }        
            auto end = std::chrono::steady_clock::now();
            omp_set_lock(&lock);
            if(local_max_count > max_count){
                max_count = local_max_count;
                max_pair = local_max_pair;
                max_strings = local_max_strings;
            }else if(local_max_count == max_count && local_max_count != -1){
                ComparisonResult r1 = three_way_compare(local_max_strings[0], max_strings[0]);
                if(r1 == ComparisonResult::Greater){
                    max_count = local_max_count;
                    max_pair = local_max_pair;
                    max_strings = local_max_strings;
                }else if(r1 == ComparisonResult::Equal){
                    ComparisonResult r2 = three_way_compare(local_max_strings[1], max_strings[1]);
                    if(r2 == ComparisonResult::Greater){
                        max_count = local_max_count;
                        max_pair = local_max_pair;
                        max_strings = local_max_strings;                  
                    }
                }
            }
            omp_unset_lock(&lock);
        }
```

This code is based on `bpe_train_updater_omp_v3`. The first change is placing the code that iterates through `pair_counts` inside a `#pragma omp parallel` block, so that the serial traversal can become a parallel one.

First, we split the tasks based on the current thread's `thread_id` (`omp_get_thread_num`) and the total number of threads (`omp_get_num_threads`). For example, with 10 buckets (`num_buckets`) and 4 threads, `10/4=2`. Thread 0 is assigned buckets 0-2 (exclusive); thread 1 gets 2-4; thread 2 gets 4-6; and thread 3 gets 6-10. Note: with this method, the last thread will have slightly more tasks, at most `threads-1` more than others.

To store the local maximum values, we define:

```
            int local_max_count = -1;
            std::pair<int, int> local_max_pair;
            std::vector<std::vector<int>> local_max_strings; 
```

Then each thread iterates through its assigned tasks:

```
            for (int i = start_bucket; i < end_bucket; ++i) {
                for (auto it = pair_counts.begin(i); it != pair_counts.end(i); ++it) {
                    const auto& pair = it->first;
                    const auto& count = it->second;

```

After a thread finishes, we need to update the global maximum value. Because of race conditions, we need to add a lock:

```
            omp_set_lock(&lock);
            if(local_max_count > max_count){
                max_count = local_max_count;
                max_pair = local_max_pair;
                max_strings = local_max_strings;
            }else if(local_max_count == max_count && local_max_count != -1){
                ComparisonResult r1 = three_way_compare(local_max_strings[0], max_strings[0]);
                if(r1 == ComparisonResult::Greater){
                    max_count = local_max_count;
                    max_pair = local_max_pair;
                    max_strings = local_max_strings;
                }else if(r1 == ComparisonResult::Equal){
                    ComparisonResult r2 = three_way_compare(local_max_strings[1], max_strings[1]);
                    if(r2 == ComparisonResult::Greater){
                        max_count = local_max_count;
                        max_pair = local_max_pair;
                        max_strings = local_max_strings;                  
                    }
                }
            }
            omp_unset_lock(&lock);
        }
```

## 4\. bpe\_train\_updater\_omp\_v4 Test Results and Analysis

We can test the speed with different numbers of threads by setting the environment variable `export OMP_NUM_THREADS=4/8/16` without changing any code. Here are the test results:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
|---|---|---|---|---|---|
| bpe\_v3\_time | | 31883 | 695 | 31187 | |
| bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 | 6779/7376/8770 | |
| bpe\_train\_updater\_omp\_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707 |
| bpe\_train\_updater\_omp\_v4 | Boost hash | 4896/4704/4214 | 543/484/440 | 4353/4219/3774 | OMP\_NUM\_THREADS=4 Cpu 150%-180% |
| bpe\_train\_updater\_omp\_v4 | Boost hash | 4313/4376/4674 | 516/472/630 | 3797/3904/4044 | OMP\_NUM\_THREADS=8 Cpu 230% |

Using 4 threads with `bpe_train_updater_omp_v4`, the max time decreased from 6000s to 4000s, a 1.5x speedup, which is consistent with the observed CPU usage peaking at 180%. With 8 threads, the max time decreased to 3900s, with a CPU peak of 230%.

However, this speedup is far from the ideal. Ideally, 4 threads should reduce the time from 6000s to 1500s, and 8 threads to 750s. While we can't expect the perfect scenario, we would hope for an 80%-90% speedup compared to the ideal.

A clear problem is that the CPU usage does not reach the expected 400% or 800%, which indicates that many threads are idle. This usually happens when the task allocation is not uniform. This is very likely because, although hash functions are theoretically designed to distribute keys evenly among all buckets, in practice, the distribution is not perfectly uniform. Moreover, even if each task has the same number of pairs, their value distribution is not uniform, as some might be already sorted, leading to a faster max search. If one of the 4 threads is slow, the other 3 threads, even if they are fast, will just wait idly.

Therefore, this simple static task partitioning is not suitable for our scenario; we need dynamic task allocation. This is similar to a producer-consumer model where each thread fetches "one" task from a task queue, and after completing it, fetches another. If the time for "one" task is not too long, the maximum idle time for other threads is limited to the time it takes to complete "one" task when the queue is empty. The definition of "one" task is critical here. If we define "one" task as too large, for example, making one loop iteration a task, we would spend a lot of time fetching data from the task queue. Since this involves data competition, multiple threads accessing the same queue require locking, which introduces overhead. On the other hand, if we define "one" task as too large, in the most extreme case defining each task as `num_buckets / threads`, it degenerates into our static situation, leading to idle threads. So, we need to find a suitable task granularity to achieve optimal performance.

The algorithm described above would be complex and tedious to implement ourselves, but with OpenMP, we only need to use the `#pragma omp for schedule` directive, which is very simple\! For example, to use dynamic scheduling for a parallel `for` loop with a chunk size of 10000, you just add the following before the loop:

```
#pragma omp parallel for schedule(dynamic, 10000)
for(int i = 0; i < N; ++i){

}
```

**Note**: If `N=25000` and there are a total of 4 threads, the first two threads will loop 10000 times, the third thread 5000 times, and the last thread will be idle.

## 5\. bpe\_train\_updater\_omp\_v5

Next, we use `#pragma omp for dynamic` to implement automatic task allocation for the `for` loop. The complete code is in [bpe\_train\_updater\_omp\_v5.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v5.cpp). Let's look at the main code:

```
        #pragma omp parallel
        {
            int local_max_count = -1;
            std::pair<int, int> local_max_pair;
            std::vector<std::vector<int>> local_max_strings; 
 
            #pragma omp for schedule(runtime)
            for (int i = 0; i < num_buckets; ++i) {
                for (auto it = pair_counts.begin(i); it != pair_counts.end(i); ++it) {
                    const auto& pair = it->first;
                    const auto& count = it->second;
                    
```

Furthermore, it differs slightly from `bpe_train_updater_omp_v4` in how the final max value is updated: it uses `#pragma omp critical` instead of a lock. The lock might have slightly better performance, but in our scenario, the difference should not be significant. Moreover, the possibility of multiple threads finishing their tasks at a very close time is small (we will set a large chunk size), so concurrent conflicts should not be frequent.

**Note**: For easy experimentation, we set the schedule to `runtime` here. This allows us to specify dynamic scheduling and set the chunk size via the environment variable `export OMP_SCHEDULE="dynamic,10000"`. If we were to hardcode `schedule(dynamic,10000)` in the code, we would have to recompile for every experiment, which would be too cumbersome.

## 6\. bpe\_train\_updater\_omp\_v5 Test Results

We first tested with a chunk size of 10000 (a very arbitrary choice, we will adjust it later) with 4, 8, and 16 threads. The results are as follows:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
|---|---|---|---|---|---|
| bpe\_v3\_time | | 31883 | 695 | 31187 | |
| bpe\_train\_updater | Boost hash | 7171/7856/9248 | 392/480/478 | 6779/7376/8770 | |
| bpe\_train\_updater\_omp\_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707 |
| bpe\_train\_updater\_omp\_v4 | Boost hash | 4896/4704/4214 | 543/484/440 | 4353/4219/3774 | OMP\_NUM\_THREADS=4 Cpu 150%-180% |
| bpe\_train\_updater\_omp\_v4 | Boost hash | 4313/4376/4674 | 516/472/630 | 3797/3904/4044 | OMP\_NUM\_THREADS=8 Cpu 230% |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 2574/2806/2220 | 576/513/440 | 1998/2292/1779 | export OMP\_NUM\_THREADS=4 export OMP\_SCHEDULE="dynamic,10000" cpu峰值390% |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1608/1570/1806 | 461/465/488 | 1147/1105/1318 | export OMP\_NUM\_THREADS=8 export OMP\_SCHEDULE="dynamic,10000" cpu峰值700% |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1605/1602/1531 | 577/564/518 | 1027/1037/1012 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,10000" cpu峰值 1331% |

As you can see, the results are much better with dynamic scheduling. The time for 4 threads is less than 2000s (ideal 1500s), and for 8 threads, it's 1100s (ideal 750s). However, the effect is not as pronounced with 16 threads, with the time being similar to that of 8 threads.

When running with 16 threads, I used `top` to continuously monitor CPU utilization and found that it was low at the beginning but eventually reached close to 1400%. My guess is that the chunk size is too large, especially in the early stages when the number of buckets is not very large, which leads to many idle threads. Therefore, a smaller chunk size might be needed. I first tried changing the chunk size to 5000/2000/1000/500/200/100. Here are the test results for 16 threads:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1605/1602/1531 | 577/564/518 | 1027/1037/1012 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,10000" cpu峰值 1331% |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1389/1340/1325 | 563/524/533 | 825/815/792 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,5000" cpu峰值 1400% |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,2000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1442/1287/1323 | 592/486/499 | 849/800/824 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1347/1363/1319 | 490/507/485 | 856/855/834 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,500" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1386/1355/1468 | 492/487/570 |893/868/897 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,200" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1485/1462/1530 | 526/489/561 | 959/972/968 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,100" |

From the experiments above, it can be seen that a chunk size of 500/1000/2000 is relatively good. If it's too large (10,000), threads will be idle; if it's too small (100), it increases scheduling overhead. Compared to 8 threads, the max time for 16 threads at this point decreased from over 1100s to over 800s.

Next, we tested the speed with 32 threads using chunk sizes of 500/1000/2000/5000:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,2000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 995/1058/974 | 571/602/531 | 424/456/443 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,500" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1002/1004/1080 | 572/580/635 | 430/423/445 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1075/1068/1036 | 634/611/590 | 440/457/445 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,2000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1197/1173/1173 | 605/600/596 | 592/572/576 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,5000" |

From 16 to 32 threads, the max time decreased from over 800s to over 400s, which is close to a linear speedup and is very good.

Next, we tried 64 threads:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,2000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1002/1004/1080 | 572/580/635 | 430/423/445 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 986/925/934 | 712/655/660 | 274/270/273 | export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 980/946/969 | 683/643/669 | 296/302/299 | export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,2000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1265/1253/1220 | 672/673/636 | 593/579/584| export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,5000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1658/1688/1658 | 634/675/640 | 1024/1012/1018| export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,10000" |

The speedup from 32 to 64 threads is not as good. There are two observations here:

  * With 64 threads and a chunk size of 1000/2000, the max time decreased from over 400s (with 32 threads) to around 300s, but with a chunk size of 5000/10000, the time was even longer than with 32 threads. This is because a large chunk size results in fewer tasks, and with more threads, idle threads appear.
  * With 64 threads, although the max time decreased, the update time increased significantly, resulting in a total time (over 950s) that was not much faster than with 32 threads.

Regarding the first point, why is the max time for 64 threads longer than for 32 threads? This is because even though the number of threads increases, there isn't enough work for them. Moreover, the cost of thread synchronization increases with more threads, causing the total time to increase.

The thread synchronization overhead here is mainly the locking operations on the task queue during task allocation, which is unavoidable. Another point is the critical section code for updating the global maximum at the end:

```
            #pragma omp critical
            if(local_max_count > max_count){
                max_count = local_max_count;
                max_pair = local_max_pair;
                max_strings = local_max_strings;
            }else if(local_max_count == max_count && local_max_count != -1){
                ComparisonResult r1 = three_way_compare(local_max_strings[0], max_strings[0]);
                if(r1 == ComparisonResult::Greater){
                    max_count = local_max_count;
                    max_pair = local_max_pair;
                    max_strings = local_max_strings;
                }else if(r1 == ComparisonResult::Equal){
                    ComparisonResult r2 = three_way_compare(local_max_strings[1], max_strings[1]);
                    if(r2 == ComparisonResult::Greater){
                        max_count = local_max_count;
                        max_pair = local_max_pair;
                        max_strings = local_max_strings;                  
                    }
                }
            }
```

We previously assumed that the execution time of each task is long and the number of threads is small, so the possibility of them reaching the critical section at the same time is small. However, the reality is that the time for each loop iteration is not long (especially in the early stages of the `while` loop), and the total max time is only over 200 seconds (with 64 threads), with an average max time of 6.3ms per iteration. So the shortest iteration time might be one or two orders of magnitude smaller than 6.3ms, making it very likely for multiple threads to reach the critical section at nearly the same time.

## 7\. Optimizing Thread Conflicts

To avoid using a critical section (lock), a common optimization method is to eliminate shared variables by using an array to store local maximums, where each thread writes to a different index in the array. However, here you must be careful to avoid false sharing—where the elements of this array are in the same cache line. To avoid false sharing, we use padding:

```
struct LocalMax {
    int max_count = -1;
    std::pair<int, int> max_pair;
    std::vector<std::vector<int>> max_strings;
    char padding[CACHE_LINE];
};


std::vector<LocalMax> local_maxes(omp_get_max_threads());
```

We put the 3 related max values into a struct and add a padding array. When we define the `local_maxes` array (vector), the spacing between its elements will be greater than the cache line size. Here we need to know the size of the cache line, which we can query with the command:

```
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
```

My system outputs 64. However, this method makes the code difficult to port. C++17 can use (but it seems not to work in my g++-13):

```
static constexpr std::size_t CACHE_LINE = std::hardware_constructive_interference_size;
```

This requires C++17 and the `<new>` header file. To avoid issues with some compilers, we use:

```
#if defined(__cpp_lib_hardware_interference_size)
    static constexpr std::size_t CACHE_LINE = std::hardware_constructive_interference_size;
#else
    static constexpr std::size_t CACHE_LINE = 64;
#endif
```

If your compiler does not support it, you will need to manually query and change the hardcoded 64 to the corresponding value (however, most common CPUs use 64).

The complete code is in [bpe\_train\_updater\_omp\_v7.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v7.cpp). Let's look at the differences from `bpe_train_updater_omp_v5`:

```
        LocalMax max;
        size_t num_buckets = pair_counts.bucket_count();

        std::vector<LocalMax> local_maxes(omp_get_max_threads()); 

        #pragma omp parallel
        {
            LocalMax local_max;
 
            #pragma omp for schedule(runtime)
            for (int i = 0; i < num_buckets; ++i) {
                for (auto it = pair_counts.begin(i); it != pair_counts.end(i); ++it) {
                    const auto& pair = it->first;
                    const auto& count = it->second;
                    
                    if(count > local_max.max_count){
                        local_max.max_count = count;
                        local_max.max_pair = pair;
                        local_max.max_strings = pair_strings[pair];
                    }else if(count == local_max.max_count){
                        std::vector<std::vector<int>> strings = pair_strings[pair];
                        ComparisonResult r1 = three_way_compare(strings[0], local_max.max_strings[0]);
                        if(r1 == ComparisonResult::Greater){
                            local_max.max_count = count;
                            local_max.max_pair = pair;
                            local_max.max_strings = strings;
                        }else if(r1 == ComparisonResult::Equal){
                            ComparisonResult r2 = three_way_compare(strings[1], local_max.max_strings[1]);
                            if(r2 == ComparisonResult::Greater){
                                local_max.max_count = count;
                                local_max.max_pair = pair;
                                local_max.max_strings = strings;                  
                            }
                        }
                    }
                }
                
            }        
            
            int thread_id = omp_get_thread_num();
            local_maxes[thread_id] = local_max;

        }

        for(const auto& local_max : local_maxes){
            // 有些线程可能没有任务
            if(local_max.max_count == -1){
                continue;
            }
            compare_and_update(local_max, max);
        }
```

First, we define `local_maxes` to store the local maximum for each thread. But inside the loop, we still use a thread-local variable `local_max` and update it within the `for` loop. It's only after the thread completes all its loops that it updates the global `local_maxes` once:

```
            int thread_id = omp_get_thread_num();
            local_maxes[thread_id] = local_max;
```

Then, after all threads have finished and synchronized (the `#pragma omp parallel` has an implicit synchronization point), we find the global maximum in a serial region:

```
        for(const auto& local_max : local_maxes){
            // Some threads might not have any tasks
            if(local_max.max_count == -1){
                continue;
            }
            compare_and_update(local_max, max);
        }
```

To avoid code verbosity, we defined a function `compare_and_update`:

```
inline void compare_and_update(const LocalMax& curr, LocalMax& max){
    if(curr.max_count > max.max_count){
        max = curr;
    }else if(curr.max_count == max.max_count){
        ComparisonResult r1 = three_way_compare(curr.max_strings[0], max.max_strings[0]);
        if(r1 == ComparisonResult::Greater){
            max = curr;
        }else if(r1 == ComparisonResult::Equal){
            ComparisonResult r2 = three_way_compare(curr.max_strings[1], max.max_strings[1]);
            if(r2 == ComparisonResult::Greater){
                max = curr;                  
            }
        }        
    }
}
```

It is important to note here: some threads might not have any tasks, so their `max_count` would be the initial value of -1. These need to be skipped, otherwise `compare_and_update` would fail.

## 8\. bpe\_train\_updater\_omp\_v7 Testing

For `bpe_train_updater_omp_v7`, I didn't do extensive chunk size experiments and directly chose 1000, which performed relatively well for `bpe_train_updater_omp_v5` with 16, 32, and 64 threads. The results are as follows:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,2000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 1002/1004/1080 | 572/580/635 | 430/423/445 | export OMP\_NUM\_THREADS=16 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v5 | Boost hash | 986/925/934 | 712/655/660 | 274/270/273 | export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v7 | Boost hash | 986/876/971 | 730/618/709 | 253/256/259 | export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,1000" |

It can be seen that `bpe_train_updater_omp_v7` performs slightly better than `bpe_train_updater_omp_v5` with 16 and 32 threads.

In summary, by increasing the number of threads to 32, the max time dropped from over 6000s (for the serial `bpe_train_updater_omp_v3`) to over 400s, which is roughly what we expected. However, the serial update time for the dictionary increased from over 400s to over 500s. With 64 threads, the update time even increased to 700s. This is because frequent read/write operations on `std::unordered_map` by multiple threads can cause the main thread's CPU cache to become invalid, thereby increasing the update time.

## 9\. bpe\_train\_updater\_omp\_v8

In addition, I also tried extending the OpenMP parallel region to the `while` loop, hoping to avoid the overhead of creating and destroying a thread pool for each `for` loop. However, this attempt was not successful. Although it reduced thread management overhead, the serial code had to be completed by one thread (I chose the main thread), and threads had to synchronize frequently. Since threads are relatively lightweight and the creation cost is not that high, the results were similar to `bpe_train_updater_omp_v7`:

| program | hash function | total time(sec) | update time(sec) | max time(sec) | other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_omp\_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v7 | Boost hash | 986/876/971 | 730/618/709 | 253/256/259 | export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v8 | Boost hash | 972/1003/912 | 558/590/520 | 413/412/391 | export OMP\_NUM\_THREADS=32 export OMP\_SCHEDULE="dynamic,1000" |
| bpe\_train\_updater\_omp\_v8 | Boost hash | 954/913/977 | 701/660/721 | 251/252/255 | export OMP\_NUM\_THREADS=64 export OMP\_SCHEDULE="dynamic,1000" |

The complete code is in [bpe\_train\_updater\_omp\_v8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v8.cpp), so I won't describe it further here.


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
