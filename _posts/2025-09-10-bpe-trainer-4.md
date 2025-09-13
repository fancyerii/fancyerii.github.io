---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第4部分：一次失败的并行优化" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第五篇，记录一次失败的并行优化尝试，通过这次失败，我们可以了解Python multiprocessing存在的问题，从而理解什么时候应该用multiprocessing什么时候不应该用它。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 算法优化


通过前面的一系列优化，我们当前的算法(bpe_v3_time)在openweb上的第一步统计词频的时间缩短到120秒(32核)，但是第二步合并还需要30000多秒。所以本文尝试优化第二步。第二步又可以分为max函数求频率最大的pair和增量更新pair统计。其中max的时间是30000秒，而增量更新pair统计只有几百秒。这么一分析，很明显我们最应该优化的就是求频率最大的pair这一步。

我们这里调用的是python的内置函数max，它的源代码在[这里](https://github.com/python/cpython/blob/v3.12.11/Python/bltinmodule.c#L1872)。它最终调用的是[min_max函数](https://github.com/python/cpython/blob/v3.12.11/Python/bltinmodule.c#L1745C1-L1745C8)：



这个函数的代码看起来有点长，而且很多都是关于Python/C API的东西。但真正算法相关的很简单，只是下面这一段：

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

代码的实现逻辑是：遍历迭代器it的每一个元素item，对item调用keyfunc(如果传入了key参数的话，我们调用时是有的)得到val。如果是第一个元素，那么把当前maxval设置为val。否则比较val和当前最大值maxval，这里调用的是`PyObject_RichCompareBool(val, maxval, op)`。这个函数的前两个参数是要比较的数，第三个参数op表示我们要比较运算符，我们这里传入的是Py_GT，表示我们想知道val是否大于maxval。如果返回值小于0，说明比较出现异常，如果返回值大于0，说明val > maxval，如果返回值等于0，则说明val <= maxval。

所以在else if (cmp > 0)的分支里，val大于maxval，因此更新maxval为val，同时更新maxitem为当前item。

这里我们主要了解一下内置max实现的大概逻辑就行，PyObject_RichCompareBool函数后面我们还会用到，所以这里稍微也介绍了一下。

系统的max函数是用c语言实现的，在Python层面上我们很难再实现一个比它更快的版本了。那怎么办呢？

当然最容易想到的办法就是和上一篇类似，使用并行算法来加速。max函数满足结合律，也就是max(max(a,b),c) = max(a,max(b,c))，因此也是非常容易并行化的。具体来说，比如我们有N个CPU，那么我们可以把所有的pair切分成N个部分，然后每个部分分别求max，然后再在N个局部最大里求全局最大。


## 2. 使用multiprocessing.Process来并行求max

我的第一次尝试就是借鉴上文的成功经验，使用多进程来并行求max。这个方法的代码在[bpe_v4_mp.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp.py)。我们来看相对于bpe_v3.py的改动部分。

```
    @staticmethod
    def _merge_a_pair(pair_counts, pair_strings, vocabulary, pair_to_words, 
                   word_counts, word_encodings, merges, size, num_processes):
        _, _, merge_pair = BPE_Trainer._parallel_max(pair_counts, pair_strings, num_processes)
        
        ....
```

它的主要代码是调用BPE_Trainer._parallel_max。我们来看一下这个函数：

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

为了能够切分任务，我们需要把pair_counts这个dict的内容放到一个可以随机访问的list里，这里就需要一次额外的复制。接着就是根据进程数num_processes切分任务。比如pair_counts的大小是10，num_processes是4。那么切分的结果就是前3个进程的任务是10 // 4 = 2，剩余除不断的都分到第4个进程。这样得到的任务数是[2,2,2,4]。当然这不是最优的划分方式，这样划分最坏的情况下最后一个进程的任务数比其它进程多(num_processes-1)个。最优的划分方式应该是把除不断的2个分给2个进程，比如给前两个进程，从而得到[3,3,2,2]。不过由于我们的进程数量不会太多(一般设置成CPU的个数)，而pair_counts的大小很大(几十万)，因此这么简单的切分也没有太大问题。

为了能够收集每个进程的结果，我们创建了一个队列queue，用于存放局部最大值。然后创建num_processes个进程来执行BPE_Trainer._find_max_pair。等到这些进程都结束后，我们再把它们的结果收集到local_maxes里，最后再用max求全局最大。

BPE_Trainer._find_max_pair的代码为：

```
    @staticmethod
    def _find_max_pair(pair_strings, pair_counts, queue):
        max_pair, max_count = max(pair_counts, key = lambda x: (x[1], pair_strings[x[0]]))
        queue.put((max_count, pair_strings[max_pair], max_pair))
```

它就是我们之前的max函数，只不过把结果放到队列里而已。

另外为了实验时方便设置max的并发进程数，增加如下命令行参数：

```
        parser.add_argument("--num_max", 
                            type=int, 
                            default=NUM_MAX_PROCESS, 
                            help="number of processes for max")
```

## 3. multiprocessing.Process实现并行算法的测试结果

为了测试时间，我实现了[bpe_v4_mp_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp_time.py)。测试发现它的运行速度非常慢，在openweb上根本跑不动，所以就在tinystory上进行了测试。测试结果如下：

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe_v3_time      | tinystory | 211/206/208 | 90/90/90 |total:120/115/117 <br> max_time: 117/112/115 <br> update_time: 3/3/3 |num_counter=8, num_merger=1
bpe_v4_mp_time      | tinystory | 730/715/740 | 80/90/80 |merge time: 650/624/659 <br> copy_time: 109/127/123 <br>max_time: 459/404/443 <br> compute_time: 221/238/231 |num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_time      | tinystory | 931/943/1007 | 90/90/90 |merge time: 841/852/917 <br> copy_time: 146/157/136 <br>max_time: 600/593/691 <br> compute_time: 125/128/118 |num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_time      | tinystory | 1354/1405/1431 | 80/90/80 |merge time: 1274/1315/1351 <br> copy_time: 143/158/160 <br>max_time: 1039/1055/1090 <br> compute_time: 80/85/86 |num_counter=8, num_merger=1 num_max=8


其中merge time是统计所有的merge的时间，代码为：

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

copy_time是把dict.items()复制到list的时间：

```
        start_time = time.perf_counter()
        pair_counts = list(pair_counts.items())
        end_time = time.perf_counter()
        times[0] += (end_time - start_time)
```
max_time是并行求max的总时间：

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

而compute_time是多个进程求max的最大时间：

```
        local_maxes = []
        local_times = []
        for i in range(num_processes):
            t = queue.get()
            local_times.append(t[-1])
            local_maxes.append(t[:-1])
        times[2] += max(local_times)
```

实际的时间计算发生在子进程里，作为参数传回主进程：

```
    @staticmethod
    def _find_max_pair(pair_strings, pair_counts, queue):
        start_time = time.perf_counter()
        max_pair, max_count = max(pair_counts, key = lambda x: (x[1], pair_strings[x[0]]))
        end_time = time.perf_counter()
        queue.put((max_count, pair_strings[max_pair], max_pair, (end_time-start_time)))
```

## 4. 结果分析

下面是几点结果总结：

* bpe_v3_time的merge time只有120秒，但是bpe_v4_mp_time在1个进程的并行算法merge时间是700秒，4个进程需要900多秒，8个进程需要1300多秒
* bpe_v3_time的max_time是110多秒，bpe_v4_mp_time在1/4/8个进程的时间分别是400+/600+/1000+秒
* bpe_v4_mp_time在1/4/8进程的时间计算时间compute_time是220+/120+/80+秒
* bpe_v4_mp_time的copy_time是120+/140+/150+秒

从上面的结果可以看出，随着进程的增多，总体时间反而花的更多。不过真正max的时间是compute_time确实是随着进程数的增多而减少的。在1/4/8个进程时，虽然compute_time是减少了，但是max_time反而增多了。这说明创建和销毁进程带来了极大的额外开销。之前的_pretokenize_and_count_mp函数只创建和销毁了一次进程，然后大量时间都是花在正则表达式和统计上。但是我们这里总共有10000次_merge_a_pair调用，也就是有10000次_parallel_max的调用，对应10000次进程的创建和销毁。这个成本是非常高的。

此外，如果要进行并行算法，我们就必须要能够随机访问pair_counts，但是Python的dict只能通过迭代器顺序访问，我们即使只是把dict复制到list里，它花的时间也可能要超过max。因为我们刚才看过max的代码了，它的时间复杂度是O(n)，它就是遍历一次，比较N次就能求出最大值，空间复杂度是O(1)，因为只需要保存当前的最大值。而复制的时间复杂度也是O(n)，而且需要O(n)的空间来保存list(虽然list里存放的是实际对象的指针)。当然这只是理论分析，所以我写了[bpe_v4_time2.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_time2.py)来比较max和copy的时间：

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

运行的结果为：

```
size: 7000 original time 70.3815951757133, copy pair: 55.18748151510954, copy pairstring: 34.67586521431804, parallel max: 0.0, max counter: 34.84415685944259

```

这里只跑了7000次merge，max的时间70s，copy到list就花了55s。这里还没有计算copy带来的内存使用的gc时间，因为这个无法测量。再加上进程创建和销毁的开销，这个方案基本不行。

这个方案最大的问题可能还不是进程创建/销毁的开销，而是Python的dict无法并行遍历。如果计算量很大，我们可以把dict先复制到list里，然后用多个CPU来同时进行计算，虽然复制有一定的开销，但是并行的加速能够弥补这个时间。但是我们的max算法其实主要是I/O，也就是遍历，CPU的计算是简单的比较，所以这样做就得不偿失了。

到此为止，其实我们就可以放弃这个方案了。不过这里还有几个疑点：

* 为什么bpe_v4_mp_time在一个进程时的compute_time(220+)要比bpe_v3_time的110+多那么多？按说bpe_v3_time的max是遍历dict，而bpe_v4_mp_time的max是遍历list，后者反而应该更快才是。
* multiprocessing.Process的进程创建和销毁开销很大，那么用multiprocessing.pool是否能解决这个问题？


为了探索这些疑点，我们再次来尝试用multiprocessing.Pool来实现并行求max。

## 5. 使用multiprocessing.Pool来实现并行max算法

完整代码在[bpe_v4_time](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_time.py)。

### train函数

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

我们使用mp.Pool(processes=num_max_processes)来创建了一个固定大小的进程池，这样的好处就是每次调用_merge_a_pair时不需要创建和销毁进程了。

BPE_Trainer._merge_a_pair最终调用的是BPE_Trainer._parallel_max，我们来看一下这个函数。

### BPE_Trainer._parallel_max

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

第一步还是把pair_counts.items()复制到list里(为了避免代码改动还是把这个list叫做了pair_counts)。接着把pair_counts划分成data_chunks，这个和之前一样，不过我增加了时间的统计(事实上后面可以看到划分的时间非常少，几乎可以忽略)。

我们的子进程执行的函数是_find_max_pair：

```
    @staticmethod
    def _find_max_pair(pair_strings, pair_counts):
        start_time = time.perf_counter()
        max_pair, max_count = max(pair_counts, key = lambda x: (x[1], pair_strings[x[0]]))
        end_time = time.perf_counter() 
        return max_count, pair_strings[max_pair], max_pair, (end_time - start_time)
```

它是真正求max的函数，为了统计时间，在返回结果里也增加了max的运行时间。

接着就是要调用Pool.map来把任务提交到进程池了。Pool.map的第一个参数是要调用的函数，第二个参数是一个迭代器，是传给子进程函数的参数。我们的BPE_Trainer._find_max_pair函数需要两个参数：pair_strings和data_chunks[i]。一种方法是把它们作为一个tuple：

```
args = [(pair_strings, chunk) for chunk in data_chunks]
local_maxes = pool.map(find_max_pair, args)
```

这样一来需要修改BPE_Trainer._find_max_pair，使得它接受一个tuple参数而不是两个参数。

因为不同的子进程的pair_strings是相同的，只有pair_counts参数不同，因此我们可以用functools.partial来创建偏函数从而把这个固定的参数嵌入其中：

```
        find_max_pair = partial(BPE_Trainer._find_max_pair, pair_strings)
        local_maxes = pool.map(find_max_pair, data_chunks)
```

这也是为什么我们的BPE_Trainer._find_max_pair把第一个参数变成了pair_strings，这是因为functools.partial只能从左到右来。

使用partial让我们的代码更加简洁，但是它们的作用完全相同。有的读者可能会说使用partial之后我们好像少了一个参数，这是不是能提高效率呢？

我们知道主进程提交任务到进程池时，参数是需要通过IPC传递给对应的进程。具体的过程是通过pickle把参数序列化成bytes，然后通过操作系统提高的API实现IPC，最后接受进程再用pickle反序列化成对象。因此如果是pool.map(find_max_pair, args)，那么args就需要传给对应的进程。假设args的len是8，则需要把(pair_strings, data_chunks[0])传递给一个进程，(pair_strings, data_chunks[1])传给另一个进程。pair_strings需要传递8次，data_chunks总共传递一次。

如果是local_maxes = pool.map(find_max_pair, data_chunks)呢？这看起来似乎只需要传递data_chunks。其实不然，Pool.map除了参数，函数也是需要从主进程传递给每一个进程池里的进程，而pair_strings是被partial嵌入在find_max_pair函数里的，因此每一个进程池里的子进程还是会需要接受一次这个函数的。

## 6. multiprocessing.Pool的测试结果

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe_v3_time      | tinystory | 211/206/208 | 90/90/90 |total:120/115/117 <br> max_time: 117/112/115 <br> update_time: 3/3/3 |num_counter=8, num_merger=1
bpe_v4_mp_time      | tinystory | 730/715/740 | 80/90/80 |merge time: 650/624/659 <br> copy_time: 109/127/123 <br>max_time: 459/404/443 <br> compute_time: 221/238/231 |num_counter=8, num_merger=1 num_max=1
bpe_v4_mp_time      | tinystory | 931/943/1007 | 90/90/90 |merge time: 841/852/917 <br> copy_time: 146/157/136 <br>max_time: 600/593/691 <br> compute_time: 125/128/118 |num_counter=8, num_merger=1 num_max=4
bpe_v4_mp_time      | tinystory | 1354/1405/1431 | 80/90/80 |merge time: 1274/1315/1351 <br> copy_time: 143/158/160 <br>max_time: 1039/1055/1090 <br> compute_time: 80/85/86 |num_counter=8, num_merger=1 num_max=8
bpe_v4_time      | tinystory | 981/967/960 | 90/90/90 |merge time: 890/876/870 <br> copy_time: 91/92/88 <br>max_time: 761/743/744 <br> compute_time: 79/79/81 <br> copy_trunks_time: 2/2/2 |num_counter=8, num_merger=1 num_max=1
bpe_v4_time      | tinystory | 1784/1688/1543 | 90/80/90 |merge time: 1693/1607/1452 <br> copy_time: 149/151/97 <br>max_time: 1483/1393/1311 <br> compute_time: 26/26/25 <br> copy_trunks_time: 3/3/2 |num_counter=8, num_merger=1 num_max=4
bpe_v4_time      | tinystory | 2660/2399/2503 | 90/90/90 |merge time: 2569/2308/2412 copy_time: 165/95/119 <br>max_time: 2336/2169/2238 <br> compute_time: 16/15/16 <br> copy_trunks_time: 3/2/2 |num_counter=8, num_merger=1 num_max=8

## 7. 结果分析

* Pool比Process还要慢，1/4/8进程时总时间(取中间数)是700/900/1300 vs 900/1600/2500。
* Pool的compute_time也是随着进程数增加而减少的，而且Pool在一个进程时的compute_time比Process甚至bpe_v3_time都要少。
* Pool的max_time时间比Process要慢很多

第一个和第三结果是出乎我意料之外的，因为按照我之前的想法，Pool减少了进程创建和销毁的开销，应该要比Process的方法更快才是。

第二个结果是符合预期的，因为Pool的max是对list进行遍历，而bpe_v3_time是对dict遍历。

但是不能解释的是为什么Pool比Process要慢，而Pool的compute_time要比Process的compute_time少。

经过反复推敲和搜索，我找到了Pool比Process慢的原因。而Process的compute_time比Pool慢的原因我当时是做了如下的猜测。

我猜测Process和Pool相比，每次循环都需要创建一个新的进程，新创建的进程可能会被放到不同的CPU/核上执行，这样它在遍历pair_counts时缓存中没有命中，从而需要到内存中获取数据。而Pool是复用进程，而pair_counts虽然在每次循环都会被修改，但是我们的算法是增量更新，大部分其实是没有被修改的。Pool的大小正好等于任务数，而且我们每次都是按照顺序向Pool提交任务的，所以很大可能pair_couts的第一部分总是提交给第一个进程，这个进程一直跑在CPU1上；第二个任务对应的数据是pair_counts的第二部分，它总是提交给进程池的第二个进程，它总是跑在CPU2上。如果是这样的话CPU的缓存中大部分也是有效的。当然，我这个只是猜想，如果要验证的话可能有两种方法：每次Pool提交任务时对data_chunks随机shuffle，这样cache可能失效；第二就是我们用Process创建进程时指定CPU，这可以通过psutil.Process.cpu_affinity()来设置(当然在测试的时候要保证这些cpu没有其它进程在使用)。这个猜测似乎有一些道理，但是后面的实验推翻了这个猜测。

而第一个和第三个结果出现的原因是：Process(在Linux默认)是使用fork来创建进程的，子进程直接继承父进程的地址空间，所以免去了进程间的数据拷贝，而Pool则必须通过pickle序列化/IPC通信/反序列化这三个步骤实现两个进程的数据传递。

Process和Pool的这个区别我也是通过这个问题才学习到的，在网上搜索了很多资料都是错误的，或者说模糊的。下面我们通过实际的代码来验证这个结论。

测试这个结论的代码在[这里](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_mp_fork.py)。

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

我们首先定义一个类Unpickleable，我们重载了它的特殊方法__reduce__，这个方法是实现自定义pickle协议的方法，我们让它抛出异常。这样如果发生了序列化，就会抛出异常，我们就能知道这个类的对象被序列化了。

multiprocessing.Process在Linux系统默认(CPython 3.12)是用fork的方式创建子进程，这个默认行为可以通过multiprocessing.set_start_method函数改变。

上面的代码如果set_start_method('fork')，则不会抛出异常，这说明fork的子进程不需要pickle序列化就可以获得父进程里的对象o。而如果设置set_start_method('spawn')，则会抛出异常。

**注意**：fork的方式启动进程虽然比spawn更快，但是在多线程环境会存在很多问题。根据[文档](https://docs.python.org/3/library/multiprocessing.html)，如果主进程使用了多线程，从Python 3.12之后会出现DeprecationWarning。更多细节请参考[os.fork()](https://docs.python.org/3/library/os.html#os.fork)。不过我们这里的主进程没有使用多线程，所以没有问题。因为fork存在这些问题，Python 3.14之后默认的启动方式在所有平台都会改成spawn。如果读者在新版本上运行bpe_v4_mp_time或者bpe_v4_time，可能需要设置启动方法为fork。

## 8. 把之前使用multiprocessing.Process的算法改用spawn来验证

根据之前的分析，Process存在额外的进程创建和销毁的开销，而且无法利用cache。它比Pool快的唯一原因就是Pool需要进程间传输数据。为了再次验证这一点，我们把Process实现的算法做一个改动，把默认的启动方法从fork改成spawn。如果之前的理由是正确的话，那么这个版本应该是最慢的一个版本。为了验证，我实现了[bpe_v4_mp_spawn_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp_spawn_time.py)。它和bpe_v4_mp_time.py唯一的区别就是：

```
mp.set_start_method('spawn')
```

这个版本的结果为：

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

这个实验的结果验证了fork比spawn要慢很多，这是符号预期的。

但是比较奇怪的是：它的compute_time和Pool差不多。这似乎推翻了我之前的推测——就是Process的compute_time慢的原因是因为新启动的进程cache不命中。

如果不是这个原因，那么为什么spawn和Pool的进程的compute_time差不多，而fork的进程就慢呢？它们的区别就是fork的子进程直接读取父进程的pair_counts：

```
pair_counts = list(pair_counts.items())
```

这里用list实现了浅层拷贝，而真正的数据在原来的dict里，pair_counts.items()返回的是原始dict的一个视图。而spawn和Pool的子进程会通过pickle深度拷贝了一份。这样得到的list在内存中的布局应该比dict更加紧凑，所以遍历的速度更快。

为了验证这个猜测是否正确，我对bpe_v4_mp_time的list手动做了一次pickle的序列化和反序列化，写了[bpe_v4_mp_deepcopy_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v4_mp_deepcopy_time.py)。它和bpe_v4_mp_time.py的区别是：

```
        start_time = time.perf_counter()
        pair_counts = list(pair_counts.items())
        pickled_bytes = pickle.dumps(pair_counts)
        pair_counts = pickle.loads(pickled_bytes)
        end_time = time.perf_counter()
```

测试结果如下：

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

bpe_v4_mp_deepcopy_time的compute_time和bpe_v4_mp_time差不多，都是200+秒。这又推翻了我的猜想，pickle序列化和反序列化并没有使得list的遍历变快。

最后还存在一个疑问：为什么fork的子进程遍历count_pairs这个list要比spawn和Pool慢？通过问Gemini，它给的答案是：

fork的子进程不需要复制父进程传入的参数，如果子进程修改了相应的内存页，操作系统就会为这个内页生成一个私有副本，这就是所谓的写时复制（Copy-On-Write，CoW）。但是我们的子进程遍历count_pairs应该是只读的，理论上不应该触发COW机制。我搜索了一下Gemini，它给的参考答案是如下。

即使子进程只对count_pairs进行读取操作，也可能触发写时复制（Copy-On-Write, CoW）。这听起来有点反直觉，但原因通常与 Python 解释器和底层的内存管理机制有关。

当子进程在 fork 后继承父进程的内存空间时，它获得了对 count_pairs 数据的只读访问权限。理想情况下，如果子进程只遍历数据，就不应该触发 CoW。然而，有几种情况会打破这个理想状态：

**1.垃圾回收 (Garbage Collection)**

这是最常见也最微妙的原因。当 Python 解释器在子进程中运行时，它会进行自己的内存管理和垃圾回收。为了确保对象的引用计数正确，或者为了标记和清除不再使用的对象，解释器可能会对内存中的某些元数据（比如引用计数）进行微小的修改。

* Python 对象：在 Python 中，每个对象都有一个ob_refcnt（引用计数）字段。
* 子进程读取：当子进程遍历 count_pairs 中的元组时，解释器可能会在内部需要检查或更新这些元组的引用计数。
* 触发 CoW：即使只是更新一个字节的引用计数，操作系统也会认为这个内存页被修改了。为了保持父子进程的独立性，内核会立即为包含这个对象的整个内存页创建一个私有副本。

**2.内存对齐和缓存行**

处理器在访问内存时，会以缓存行（通常是 64 字节）为单位。当一个内存页被访问时，整个缓存行会被载入处理器缓存。如果这个缓存行包含了需要被修改的数据（即使是其他不相关的对象），那么整个内存页都可能被标记为需要复制。

**3.JIT 编译器和其他系统调用**

如果子进程使用了 JIT（Just-In-Time）编译器或其他一些复杂的库，这些库可能会在内部进行内存操作，从而意外地触发 CoW。即使是简单的内存访问，也可能与内核的其他内存管理操作产生复杂的交互。

**结论**

子进程对 count_pairs 的读取之所以会触发 CoW，并不是因为你代码中显式地进行了修改，而是因为 Python 解释器在后台为了维护自身的内存状态（最常见的是垃圾回收的引用计数）而进行的微小修改。

spawn 方式之所以没有这个问题，是因为它从一开始就创建了一个独立的内存副本。虽然这个复制过程（序列化和反序列化）在启动时很耗时，但一旦完成，子进程就拥有了完全独立的内存空间，后续的任何操作都不会再影响或被父进程的内存所影响，因此也就不存在 CoW 的开销。


为了验证这一点，我写了一个程序来测试：

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

如果是fork方式，总时间是1.07s，子进程遍历时间是0.28s。如果是spawn方式，总时间是4.8s，子进程遍历时间是.08s。所以即使存在触发CoW的问题，总体来说fork还是要比spawn更快。

## 9. 总结

通过这一次探索，我们发现Python的multiprocessing对于这种I/O和CPU交叠的运算任务是不合适的。虽然使用多个CPU确实能够减少计算max的时间(compute_time)，但是由于进程间通信的开销过大，反而得不偿失。这种场景应该是使用多线程来解决，因为同一个进程的多个线程可以共享内存，从而避免了进程间通信的开销。但是由于CPython本身GIL的限制，我们无法使用多线程来解决这个问题。所以下面的内容我们暂时转向C++，尝试使用多线程来并行化max函数。


