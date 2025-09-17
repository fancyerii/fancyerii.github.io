---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第6部分：用OpenMP实现并行求最大" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第七篇，探索使用OpenMP并行求pair_counts里最大pair的方法。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 优化方法

前一篇文章我们已经实现了通过bucket接口使用下标随机遍历std::unordered_map的算法，这为max算法的并行化打下了很好的基础。而且在openweb数据集上测试的结果，bpe_train_updater_omp_v3在更新数据结构(串行)的时间是400多秒，而max(可并行)的时间是6000多秒。可并行代码的时间占据了总时间的90%以上，因此如果我们能够并行化max，那么总的时间会减少很多。

## 2. OpenMP

由于并行算法有很大一部分是在遍历pair_counts这个std::unordered_map，而且pair_counts在每次merge之后都会修改，所以最适合的并行算法是共享内存的并行方法，比如多线程。c++的标准库提供了基础的多线程工具，比如std::thread、std::mutex、std::condition_variable、std::promise 和 std::future等等。C++ 标准库倾向于提供低层级、通用的构建模块，所以没有线程池和线程安全的队列。如果要自己实现一个正确高效的线程池和队列并不容易。一些第三方库提供了这些工具，但是引入额外的库会增加依赖，我们后面通过cython集成第三方库会麻烦一点。更重要的是，即使是线程池和队列这些工具还是太底层。而我们的任务比较简单，就是希望把一个for循环用多个线程并行求max。这样的任务使用OpenMP是非常合适的。OpenMP和那些多线程的工具库不同，它提供了简单的#pragma omp xxx编译器预处理指令。我们通过这些指令告诉编译器，我们的这个for循环需要并行，OpenMp就会自动的把我们的代码改写成并行的版本。它会自动管理线程和任务的分配。很多参数，比如线程数、动态分配的chunk_size都可以通过环境变量来设置，这样我们需要实验不同的参数也非常简单。

关于OpenMP的知识这里不多做介绍，不熟悉的读者可以通过[Introduction to OpenMP](https://www.youtube.com/watch?v=nE-xN4Bf8XI&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG&index=1)这个视频课程来学习。这个视频课程非常适合入门学习OpenMP，每个视频都不长，而且看完视频后可以动手练习。如果只是为了阅读本文，掌握\#pragma omp parallel和\#pragma omp for schedule也就足够了。

## 3. 手动切分任务

我们先来尝试用手动切分任务的方式实现并行算法，完整代码在[bpe_train_updater_omp_v4.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v4.cpp)。

最主要的代码是：

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

这个代码是基于bpe_train_updater_omp_v3进行修改的。首先的改动是把遍历pair_counts的代码放到了#pragma omp parallel块下，这样就可以把这个串行遍历变成并行遍历。

我们首先根据当前线程的thread_id(omp_get_thread_num)和总的线程数threads(omp_get_num_threads)来切分任务，比如有10个桶(num_buckets)和4个线程，10/4=2，那么线程0分配的是0-2(exclusive)；线程1分配的是2-4；线程2分配的是4-6；线程3分配的是6-10。注意：这种分配方法最后一个线程会多一些任务，最多比其它线程多threads-1个任务。

为了保存局部的最大值，我们定义：

```
            int local_max_count = -1;
            std::pair<int, int> local_max_pair;
            std::vector<std::vector<int>> local_max_strings; 
```

然后每个线程遍历自己的任务：

```
            for (int i = start_bucket; i < end_bucket; ++i) {
                for (auto it = pair_counts.begin(i); it != pair_counts.end(i); ++it) {
                    const auto& pair = it->first;
                    const auto& count = it->second;

```

当前线程结束后，我们需要更新全局的最大值。因为存在race condition，我们需要加锁：

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
```

## 4. bpe_train_updater_omp_v4测试结果及结果分析

我们可以通过设置环境变量export OMP_NUM_THREADS=4/8/16来测试不同线程时的速度，而不需要修改任何代码。下面是测试结果：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_v3_time          | | 31883 | 695 | 31187 |
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707
bpe_train_updater_omp_v4 | Boost hash | 4896/4704/4214 | 543/484/440 | 4353/4219/3774 | OMP_NUM_THREADS=4 Cpu 150%-180%
bpe_train_updater_omp_v4 | Boost hash | 4313/4376/4674 | 516/472/630 | 3797/3904/4044 | OMP_NUM_THREADS=8 Cpu 230%

bpe_train_updater_omp_v4使用4个线程时，max的时间从6000s减少到4000s，快了1.5倍，这和我观察到的cpu使用率在峰值时180%是相符合的。使用8个线程时max时间减少到3900s，cpu的峰值在230%。

但是这个加速比与理想的情况还是差了很多的。最理想的情况下4个线程应该从6000s减少到1500s，8个线程减少到750s。虽然我们不能期望达到最理想的情况，但是也期望能够达到理想情况80%-90%的加速比。

很明显的问题就是CPU的使用率没有达到期望的400%或者800%，这说明很多线程是处于空闲状态的。这种情况通常是因为任务的分配并不均匀。这是很有可能的，虽然理论上hash函数会把key均匀的分散到所有的桶里，但是实际上不可能那么均匀。而且即使每个任务的pair是相同的，根据前面的分析，它们的值的分布也不均匀，有些可能排好序，那么max就会更快。如果4个线程中有一个很慢，其它3个线程即使很快，也只能闲在那里等待。

因此这种简单的静态任务划分并不适合我们的场景，我们需要动态的任务分配。我们类似于生产者消费者模式里的任务队列，每个线程去任务队列里取“一个”任务，做完了之后再去取。如果“一个”任务的时间不太长，那么其它线程闲置的最大时间就是当任务队列空的时候等待“一个”任务的完成时间。这里的“一个”任务的定义比较关键，如果我们把“一个”任务定义的太多，比如把for循环的一次循环作为一个任务，那么我们需要花大量时间去任务队列取数据。因为涉及到数据竞争，多个线程访问同一个队列是需要加锁，这会带来额外的开销。另一方面，如果我们把“一个”任务定义的太大，最极端的就是把每个任务定义为num_buckets / threads，那就退化到我们的静态的情况了，这时就会导致线程闲置。所以我们需要找到一个合适的任务粒度，从而才能达到最优的性能。

上面的这种算法描述如果要我们自己实现会比较复杂和繁琐，而使用OpenMP只需要使用#pragma omp for schedule指令就可以了，非常简单！比如要使用动态调度来多线程并行一个for循环，并且chunk size是10000，那么只需要在for之前加上：

```
#pragma omp parallel for schedule(dynamic, 10000)
for(int i = 0; i < N; ++i){

}
```

**注意**：如果N=25000，并且总共4个线程，那么前两个线程循环10000次，第三个线程循环5000次，最后一个线程会闲置。

## 5. bpe_train_updater_omp_v5

接下来我们使用#pragma omp for dynamic实现for循环任务的自动分配，完整的代码在[bpe_train_updater_omp_v5.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v5.cpp)。我们来看一下它的主要代码：

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

另外和bpe_train_updater_omp_v4在更新最终的max时稍有不同，没有使用锁，而是使用了#pragma omp critical。lock的性能应该稍好一些，不过在我们的场景应该没有太大区别。而且多个线程在非常接近的时间完成任务的可能性并不大(我们会设置较大的chunk_size)，所以并发的冲突应该不多。

**注意**：为了实验方便，我们这里的schedule是runtime，这样就可以通过环境变量export OMP_SCHEDULE="dynamic,10000"来指定使用动态调度，并且设置chunk size是10000。如果我们用schedule(dynamic,10000)写死在代码里，那么每次实验都要重复编译代码，那就太麻烦了。

## 6. bpe_train_updater_omp_v5测试结果

我们首先用chunk size是10000(这个是很随意的，我们后面会调整)在4/8/16线程下进行了测试，结果为：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_v3_time          | | 31883 | 695 | 31187 |
bpe_train_updater | Boost hash | 7171/7856/9248 | 392/480/478 |6779/7376/8770 |
bpe_train_updater_omp_v3 | Boost hash | 6439/7016/6857 | 436/463/473 | 6003/6552/6383 | buckets:4355707
bpe_train_updater_omp_v4 | Boost hash | 4896/4704/4214 | 543/484/440 | 4353/4219/3774 | OMP_NUM_THREADS=4 Cpu 150%-180%
bpe_train_updater_omp_v4 | Boost hash | 4313/4376/4674 | 516/472/630 | 3797/3904/4044 | OMP_NUM_THREADS=8 Cpu 230%
bpe_train_updater_omp_v5 | Boost hash | 2574/2806/2220 | 576/513/440 | 1998/2292/1779 | export OMP_NUM_THREADS=4 export OMP_SCHEDULE="dynamic,10000" cpu峰值390%
bpe_train_updater_omp_v5 | Boost hash | 1608/1570/1806 | 461/465/488 | 1147/1105/1318 | export OMP_NUM_THREADS=8 export OMP_SCHEDULE="dynamic,10000" cpu峰值700%
bpe_train_updater_omp_v5 | Boost hash | 1605/1602/1531 | 577/564/518 | 1027/1037/1012 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,10000" cpu峰值 1331%

可以发现，使用动态调度之后，效果好了很多，4个线程时的时间小于2000s(理想1500s)，8个线程1100s(理想750s)。但是到了16个线程时效果就不明显了，和8线程的时间差不多。

在16线程运行时我用top不断查看cpu的利用率，发现刚开始时很低，到了后面才达到接近1400%。我猜测可能是因为chunk size过大，尤其是早期bucket的数量不是太大，这就导致很多线程闲置。因此可能需要把chunk size改小一点。我首先尝试把chunk size改成5000/2000/1000/500/200/100，下面是16线程时的测试结果：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_omp_v5 | Boost hash | 1605/1602/1531 | 577/564/518 | 1027/1037/1012 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,10000" cpu峰值 1331%
bpe_train_updater_omp_v5 | Boost hash | 1389/1340/1325 | 563/524/533 | 825/815/792 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,5000" cpu峰值 1400%
bpe_train_updater_omp_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,2000" 
bpe_train_updater_omp_v5 | Boost hash | 1442/1287/1323 | 592/486/499 | 849/800/824 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,1000" 
bpe_train_updater_omp_v5 | Boost hash | 1347/1363/1319 | 490/507/485 | 856/855/834 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,500" 
bpe_train_updater_omp_v5 | Boost hash | 1386/1355/1468 | 492/487/570 |893/868/897 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,200" 
bpe_train_updater_omp_v5 | Boost hash | 1485/1462/1530 | 526/489/561 | 959/972/968 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,100" 


从上面的实验可以看出：chunk size在500/1000/2000是比较好的，太大(10,000)了会导致线程空闲，太小(100)了增加调度开销。和8线程相比，此时16线程的max time从1100多秒降到800多秒。

接下来我们用chunk size是500/1000/2000/5000测试32线程的速度：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_omp_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,2000" 
bpe_train_updater_omp_v5 | Boost hash | 995/1058/974 | 571/602/531 | 424/456/443 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,500" 
bpe_train_updater_omp_v5 | Boost hash | 1002/1004/1080 | 572/580/635 | 430/423/445 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000" 
bpe_train_updater_omp_v5 | Boost hash | 1075/1068/1036 | 634/611/590 | 440/457/445 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,2000" 
bpe_train_updater_omp_v5 | Boost hash | 1197/1173/1173 | 605/600/596 | 592/572/576 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,5000" 

从16线程到32线程，max time从800多秒讲到了400多秒，接近线性加速，非常不错。

再接下来我们尝试64线程：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_omp_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,2000" 
bpe_train_updater_omp_v5 | Boost hash | 1002/1004/1080 | 572/580/635 | 430/423/445 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000" 
bpe_train_updater_omp_v5 | Boost hash | 986/925/934 | 712/655/660 | 274/270/273 | export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,1000" 
bpe_train_updater_omp_v5 | Boost hash | 980/946/969 | 683/643/669 | 296/302/299 | export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,2000"
bpe_train_updater_omp_v5 | Boost hash | 1265/1253/1220 | 672/673/636 | 593/579/584| export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,5000"
bpe_train_updater_omp_v5 | Boost hash | 1658/1688/1658 | 634/675/640 | 1024/1012/1018| export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,10000"
 
从32线程到64线程就没那么好了。这里有两点观察现象：

* 64线程时chunk size在1000/2000时，max time从32线程的400多秒降到300秒左右，但是chunk size在5000/10000的时长甚至超过32线程，这是因为chunk太大导致任务数少，而线程又变多，因此出现线程闲置。
* 64线程时虽然max time下降，但是update time增加了很多，导致总时长（950多秒）和32线程相比并没有显著变快。

对于第一点，为什么64线程的max time比32线程还长呢？因为线程虽然增多，但是并没有活可干，而且由于线程增多，线程同步的成本增加，从而导致线程增多时间反而变长。

这里线程的同步开销主要是任务分配时任务队列的加锁等操作，这个无法避免。另外就是在最后更新全局最大的临界区代码：

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

我们之前假设任务的执行时间很长，线程也不多，因此它们同时到达临界区的可能性不大。但是实际情况是每一层循环的时间并不长(尤其是在while循环的初期)，我们总共执行32000-257次迭代，总的max time只有200多秒(64线程时)，平均每次max的时间6.3ms。那么最短的迭代时间可能比6.3ms还要小一两个数量级，因此多个线程接近同时到底临界区的可能性还是非常大的。

## 7. 优化线程冲突

为了避免使用临界区(锁)，可以使用常见的一种优化方法来消除共享变量——那就是用一个数组来存储局部最大值，每个线程写到数组的不同下标里。不过这里一定要注意避免伪共享——也就是这个数组的元素在同一个cache line里。为了避免伪共享，我们使用padding的方法：

```
struct LocalMax {
    int max_count = -1;
    std::pair<int, int> max_pair;
    std::vector<std::vector<int>> max_strings;
    char padding[CACHE_LINE];
};


std::vector<LocalMax> local_maxes(omp_get_max_threads());
```

我们把3个相关的最大值放到一个结构体里，另外增加padding数组，这样我们定义local_maxes数组(vector)时它们之间的间隔就大于cache line的大小了。这里我们需要知道cache line的大小，我们可以用命令查询：

```
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
```

我的系统输出是64。但是这种方法使得代码很难移植，c++-17可以使用(但是我在g++-13里好像不行)：

```
static constexpr std::size_t CACHE_LINE = std::hardware_constructive_interference_size;
```

这个需要c++-17，并且需要包含\<new>这个头文件。为了避免有的编译器不支持，我们使用：

```
#if defined(__cpp_lib_hardware_interference_size)
    static constexpr std::size_t CACHE_LINE = std::hardware_constructive_interference_size;
#else
    static constexpr std::size_t CACHE_LINE = 64;
#endif
```

如果你的编译器不支持，那么就需要手动查询然后把硬编码的64修改成对应的值(不过目前常用的cpu基本都是64)。

完整的代码在[bpe_train_updater_omp_v7.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v7.cpp)，下面我们来看一下它和bpe_train_updater_omp_v5的不同之处：

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

首先我们定义local_maxes，用来为每个线程存储局部最大。但是在循环里面，我们还是使用了一个thread local的局部变量local_max，并且在for循环里更新它。只是在线程完成所有的循环之后，更新全局的local_maxes一次：

```
            int thread_id = omp_get_thread_num();
            local_maxes[thread_id] = local_max;
```

然后等待所有的线程都完成并同步后(#pragma omp parallel会有隐式的同步)，我们在串行区域求全局最大：

```
        for(const auto& local_max : local_maxes){
            // 有些线程可能没有任务
            if(local_max.max_count == -1){
                continue;
            }
            compare_and_update(local_max, max);
        }
```

为了避免代码冗长，我们定义了一个函数compare_and_update：

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

这里需要注意：有些线程可能没有任务，那么它的max_count是初始值-1，需要跳过，否则compare_and_update会出错。

## 8. bpe_train_updater_omp_v7的测试

bpe_train_updater_omp_v7的chunk size我没有再做太多实验，直接选择了bpe_train_updater_omp_v5在16/32/64线程都算比较好的1000。结果如下：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_omp_v5 | Boost hash | 1327/1330/1393 | 505/539/569 |821/791/823 | export OMP_NUM_THREADS=16 export OMP_SCHEDULE="dynamic,2000" 
bpe_train_updater_omp_v5 | Boost hash | 1002/1004/1080 | 572/580/635 | 430/423/445 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000" 
bpe_train_updater_omp_v5 | Boost hash | 986/925/934 | 712/655/660 | 274/270/273 | export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,1000" 

bpe_train_updater_omp_v7 | Boost hash | 1268/1196/1215 | 548/473/481 | 719/723/734 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v7 | Boost hash | 986/876/971 | 730/618/709 | 253/256/259 | export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,1000"


可以发现bpe_train_updater_omp_v7在16/32线程时比bpe_train_updater_omp_v5要好一些。

总结一下，我们把线程数增加到32，max time从bpe_train_updater_omp_v3(串行)的6000多秒降到了400多秒，这是基本符合预期的。但是串行更新词典的update time却从400多秒增加到了500多秒，如果增加到64线程，update time更是增加到700秒。这是因为多线程对于std::unordered_map的频繁读写会导致主线程的cpu的cache失效，从而导致update time增加。

## 9. bpe_train_updater_omp_v8

除此之外，我还尝试了把omp的并行区域扩展到while循环，这样期望避免每次for循环都创建和销毁线程池的开销。但是这个尝试并不成功，原因是虽然减少了线程的管理开销，但是串行的代码必须由某一个线程(我选择主线程)来完成，而线程之间需要频繁同步。而线程还是比较轻量级的，创建的成本也并没有那么高，所以它的结果和bpe_train_updater_omp_v7差不多：

bpe_train_updater_omp_v7 | Boost hash | 907/908/955 | 514/503/554 | 391/403/400 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v7 | Boost hash | 986/876/971 | 730/618/709 | 253/256/259 | export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v8 | Boost hash | 972/1003/912 | 558/590/520 | 413/412/391 | export OMP_NUM_THREADS=32 export OMP_SCHEDULE="dynamic,1000"
bpe_train_updater_omp_v8 | Boost hash | 954/913/977 | 701/660/721 | 251/252/255 | export OMP_NUM_THREADS=64 export OMP_SCHEDULE="dynamic,1000"

完整代码在[bpe_train_updater_omp_v8.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_omp_v8.cpp)，我这里就不介绍了。

