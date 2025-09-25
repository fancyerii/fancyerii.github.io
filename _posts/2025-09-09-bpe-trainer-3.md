---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第3部分：并行分词和统计词频" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第四篇，使用并行算法加速。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 算法优化

### 1.1 当前算法瓶颈分析

根据前文的实验，在tinystory数据集上，_pretokenize_and_count的时间是600多秒，合并的时间是100多秒；在openweb上_pretokenize_and_count的时间接近3000秒，而合并的时间是30000多秒。这两个步骤都需要优化，但是本文我们先优化第一个步骤。先优化第一个步骤的原因有二：第一，我们希望先解决简单的问题，第一步的主要计算是正则表达式的匹配和词频统计，这非常容易用并行算法实现加速；第二，对于实际的LLM预训练，训练的文本远远大于10GB。第一步时间复杂度O(n)，而第二步根据[Heaps' law](https://en.wikipedia.org/wiki/Heaps%27_law)，词相对于语料库的增长速度是小于线性的，因此如果训练文本增大100倍，那么第一步的时间也会线性的增大100倍。但是第二步是基于词频来合并的，它的增速相对是比较小的。

补充说明一下，Heaps' law用来描述**语料库大小（$N$）**和**词典大小（$V$）**之间的关系。词典大小是指语料库中不同词汇的总数（即词表大小）。


$$
V = K \cdot N^{\beta}
$$

其中：
* **$V$** 是词典大小（Vocabulary size）。
* **$N$** 是语料库大小（Corpus size），通常用词的数量来衡量。
* **$K$** 是一个常数，其值取决于具体的语料库。
* **$\beta$** 是一个指数，通常在 $0.4$ 到 $0.6$ 之间，最常见的近似值是 $0.5$。


### 1.2 优化方法

_pretokenize_and_count是易于并行的(Embarrassingly parallel)，我们可以把原始文本按照文档为单位切分成多个部分，然后用多个cpu来进行正则表达式匹配和统计词频，最后再合并各个部分的词频。只有最后一个合并步骤串行的，其余部分非常容易并行。

为了利用多个CPU进行并行计算，最常见的就是使用多线程。但是由于python的GIL，python的多线程(threading)只能实现并发(concurrency)而不是并行(parallel)，python的多线程只适合I/O密集型任务，而不适合CPU密集型任务。对于CPU密集型任务，比较合适的是使用多进程(multiprocessing)。

另外，我们的计算需要分成很多步骤，因此使用队列和生产者/消费者模式来实现是比较清晰的。具体来说我们的流程如下：

* 文档切割
    * 利用文档分隔特殊字符串`<|endoftext|>`把原始文本切分成多个chunk，每个chunk包含一个或多个文档。结果放到`chunk_queue`。
* 正则匹配和词频统计
    * 把chunk用`<|endoftext|>`切分成文档，再把文档切分成词，然后统计词频。处理的结果放到`counter_queue`。
* 词频合并
    * 从`counter_queue`取出chunk的词频统计结果，然后进行合并，结果放到`merged_queue`。
* 最终合并
    * 主进程把`merged_queue`里的统计结果进行合并。


这里需要注意切分的粒度，比如在**正则匹配和词频统计**这一步，我们是把整个chunk的词频统计结果一次性放到`counter_queue`，这样可以减少进程间数据的传输。我们也可以把每一个文档的词频统计放到`counter_queue`，那么统计的工作并不会增加，只是把工作转移到了后面**词频合并**步骤里而已，但是这种方法会使得数据传输增大。比如一个chunk有1000个文档，原来这1000个文档的词频合并发生在第二步**正则匹配和词频统计**，现在这个工作发生在第三步**词频合并**。但是这种方法就会使得`counter_queue`的数据可能增大1000倍。

比如有两个文档：

```
{"a":1, "b":2, "c":3, "d":4}
{"a":4, "b":3, "c":2, "e":1}
```

如果单独传输，需要传输8个key/value对，但是如果先合并：

```
{"a":5, "b":5, "c":5, "d":4, "e":1}
```
则只需要传输5个key/value对。最理想的情况是这些文档的key都是相同的，那么合并后能减少2倍。但是即使不能达到理想情况，根据前面的Heaps' law和[Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law)，那些高频词总是能合并的。

由于我们使用的是multiprocessing，进程间无法想线程那样共享内存地址空间，Python只能把要传输的数据先从对象序列化(pickle)成二进制流，然后通过进程间通信的系统API传输数据，最后还得把二进制流反序列化成对象。所以减少进程间的通信能极大的优化Python多进程的效率。

## 2. 代码

源代码可以参考[bpe_v3.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3.py)。


### 2.1 train

```
    def train(self, input_path, vocab_size, special_tokens, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_counter", 
                            "-c",
                            type=int, 
                            default=NUM_COUNTER_PROCESS, 
                            help="number of processes for counting")
        parser.add_argument("--num_merger", 
                            "-m",
                            type=int, 
                            default=NUM_MERGER_PROCESS, 
                            help="number of processes for merging")
        parser.add_argument("--do_monitor",
                            action="store_true",
                            help="Enable queue monitor. (default: False)"
        )        
        
        args = parser.parse_args(args)
        print(f"train: {args=}")
        num_counter = args.num_counter
        num_merger = args.num_merger
        do_monitor = args.do_monitor

        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count_mp(input_path, special_tokens, num_counter, num_merger, do_monitor)
        end_time = time.perf_counter()

        print(f"_pretokenize_and_count_mp: {end_time - start_time}")
        vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            vocabulary[N_BYTES + i] = token.encode('utf-8')
        size = N_BYTES + len(special_tokens)
        merges = []

        # initial word encodings are utf-8
        word_encodings = {}
        for word in word_counts:
            word_encodings[word] = list(word.encode('utf-8'))

        pair_strings = {}
        pair_to_words = defaultdict(set)
        pair_counts = BPE_Trainer._count_pairs(word_counts, word_encodings, pair_strings, vocabulary, pair_to_words)

        while size < vocab_size:
            BPE_Trainer._merge_a_pair(pair_counts, pair_strings, vocabulary,
                                   pair_to_words, word_counts, word_encodings,
                                   merges, size)
            size += 1
      
        
        return vocabulary, merges
```

train函数和之前版本相比，主要是把_pretokenize_and_count换成了_pretokenize_and_count_mp。另外为了方便实验和调试，增加了3个参数：

* --num_counter/-c _chunk_counter_process进程的数量，也就是前面4步里第二步的进程数量
* --num_merger/-m _merge_counter_process进程的数量，也就是第三步的进程数量
* --do_monitor 是否定期(10秒)打印当前队列大小，这是为了便于调试，如果哪个队列堆积，则说明相应步骤的处理速度不够快

### 2.2 _pretokenize_and_count_mp

```
    def _pretokenize_and_count_mp(self, input_path: str, special_tokens: list[str],
                                  num_counter, num_merger, do_monitor):
        # pre-compile regex
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

        chunk_queue = mp.Queue(maxsize=1_000_000)
        counter_queue = mp.Queue(maxsize=1_000_000)
        merged_queue = mp.Queue(maxsize=num_merger)
        counter_processes = []
     
        for i in range(num_counter):
            p = mp.Process(target=BPE_Trainer._chunk_counter_process, 
                        args=(chunk_queue, counter_queue, 
                              pattern, special_token_pattern),
                        name=f"counter_process-{i+1}")
            p.start()
            counter_processes.append(p)

        merge_processes = []
        for i in range(num_merger):
            p = mp.Process(target=BPE_Trainer._merge_counter_process, 
                        args=(counter_queue, merged_queue),
                        name=f"merge_process-{i+1}")
            p.start()
            merge_processes.append(p)        

        

        # stop_event.set() for unit test, we should stop monitor to pass speed test
        # because monitor process will sleep 30s 
        if do_monitor:
            stop_event = mp.Event()

            monitor_process = mp.Process(target=BPE_Trainer._queue_moniter_process, 
                                  args=(chunk_queue, counter_queue, merged_queue, stop_event))
            monitor_process.start()



        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            chunk_queue.put(chunk)

        for i in range(num_counter):
            chunk_queue.put(None)

        for p in counter_processes:
            p.join()


        for _ in range(num_merger):
            counter_queue.put(None)



        # use main process to merge into final counter
        if num_merger == 1:
            word_counts = merged_queue.get()
        else:
            word_counts = merged_queue.get()
            for _ in range(num_merger - 1):
                counter = merged_queue.get()
                for k,v in counter.items():
                    word_counts[k] += v

        # stop moniter and join all processes
        
        for p in merge_processes:
            p.join() 

        if do_monitor:
            stop_event.set()
            monitor_process.join()   

        return word_counts
```

这个函数比较长，我们分段解释。

#### 构建正则表达式和队列

```
        # pre-compile regex
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

        chunk_queue = mp.Queue(maxsize=1_000_000)
        counter_queue = mp.Queue(maxsize=1_000_000)
        merged_queue = mp.Queue(maxsize=num_merger)
```

首先我们构建分词和切分文档的正则表达式pattern和special_token_pattern，这和之前的版本没有区别。接着我们构建了3个队列`chunk_queue`、`counter_queue`和`merged_queue`，它们的作用前面已经说过了。注意：`merged_queue`的大小和num_merger一样大。因为每个合并进程最终只产生一个词频统计的结果。

#### 启动_chunk_counter_process进程和_merge_counter_process进程

```
        counter_processes = []
        for i in range(num_counter):
            p = mp.Process(target=BPE_Trainer._chunk_counter_process, 
                        args=(chunk_queue, counter_queue, 
                              pattern, special_token_pattern),
                        name=f"counter_process-{i+1}")
            p.start()
            counter_processes.append(p)

        merge_processes = []
        for i in range(num_merger):
            p = mp.Process(target=BPE_Trainer._merge_counter_process, 
                        args=(counter_queue, merged_queue),
                        name=f"merge_process-{i+1}")
            p.start()
            merge_processes.append(p)        

        

        # stop_event.set() for unit test, we should stop monitor to pass speed test
        # because monitor process will sleep 30s 
        if do_monitor:
            stop_event = mp.Event()

            monitor_process = mp.Process(target=BPE_Trainer._queue_moniter_process, 
                                  args=(chunk_queue, counter_queue, merged_queue, stop_event))
            monitor_process.start()  
```

_chunk_counter_process进程的输入是`chunk_queue`队列，输出是`counter_queue`队列。而_merge_counter_process进程的输入是`counter_queue`，输出是`merged_queue`。这是处理流程的第二步和第三步。

而第一步和第四步我们理论上也可以分别用一个或者多个进程来完成。但是我们知道第四步只能串行，所以我们就直接用主进程来做了。因为如果单独在起一个进程来做，需要多一次进程间的通信。我们需要把最终的词频统计结果发给主进程来完成后续的处理，用主进程来做就可以省去这个通信过程，而且主进程在等待期间也没事可做。第一步我们也是由主进程来完成，理由是文件的读取速度相对于后面的流程来说是非常快的，而且使用多进程读取并不比单进程快(后面我们会分析文件读取性能极限)。

此外上面的代码还会根据do_monitor参数来启动_queue_moniter_process，这个进程的作用就是定期打印队列的大小，方便调试。

#### 主进程读取文件生成chunk

```
        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            chunk_queue.put(chunk)

        for i in range(num_counter):
            chunk_queue.put(None)
```

这段代码就是主进程调用之前的_chunk_documents_streaming函数(其实是生成器函数)来读取文件并且生成chunk。注意：这里我们的chunk_queue是有大小限制的(我们设置成了1_000_000)，这样如果后面的流程比较慢，队列满了之后chunk_queue.put(chunk)就会阻塞，从而避免内存爆掉。这也是_chunk_documents_streaming用yield来实现生成器函数的好处。如果不用生成器函数而是直接读取chunk到list里，那么很可能内存就OOM了(比如读取一个1TB的文件)。

最后如果所有的chunk都放到chunk_queue里了，主进程会再放num_counter个None到num_counter里。这是生产者/消费者模式里生产者告诉消费者任务已经完成的一种常见方式——哨兵（Sentinel）。通过特殊的对象，消费者可以知道所有的任务都已经完成，从而可以结束自己。


#### 主进程等待_chunk_counter_process结束并且通知_merge_counter_process进程


```
        for p in counter_processes:
            p.join()

        for _ in range(num_merger):
            counter_queue.put(None)
```

p.join()等待所有的_chunk_counter_process结束(它们能够自己结束自己，因为前面的哨兵None)，然后再放num_merger个None到counter_queue，这些哨兵可以告诉_merge_counter_process什么时候可以结束自己。


#### 主进程合并

```
        # use main process to merge into final counter
        if num_merger == 1:
            word_counts = merged_queue.get()
        else:
            word_counts = merged_queue.get()
            for _ in range(num_merger - 1):
                counter = merged_queue.get()
                for k,v in counter.items():
                    word_counts[k] += v
```


接下来主进程合并词频统计。如果num_merger，那么就不需要合并里，直接获得结果。否则需要从队列里取num_merger个结果进行合并。

#### 等待进程结束并返回结果

```
        for p in merge_processes:
            p.join() 

        if do_monitor:
            stop_event.set()
            monitor_process.join()   

        return word_counts
```

主进程等待所有合并进程结束，如果启动了监控进程，通过Event通知它结束自己。

**注意**：我们这里的主进程是先合并merged_queue的结果再等待所有merge_processes结束。可能有的读者会问：为什么不是先等所有merge_processes结束再合并merged_queue？因为merge_processes并不是同时完成，而是有先有后。如果先合并，只要任何_merge_counter_process完成并且把结果放到merged_queue，主进程都可以第一时间拿到它并进行合并，这样在最后一个_merge_counter_process完成时，前面num_merger-1个进程的结果可能已经合并完了(最理想情况)，只需要合并最后一个即可。

我们这里是自己通过Process来管理进程，所以比较繁琐。如果使用[进程池](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool)，则可以使用[pool.imap_unordered](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap_unordered)。感兴趣的读者也尝试使用进程池和提交任务的方式来实现。不过我觉得生产者/消费者模式逻辑更加清晰更容易理解。

#### 监控进程

我们先从最简单的监控进程开始。

```
    @staticmethod
    def _queue_moniter_process(chunk_queue, counter_queue, merged_queue, event):
        while not event.is_set():
            print(f"chunk_queue: {chunk_queue.qsize()}, counter_queue: {counter_queue.qsize()}, merged_queue: {merged_queue.qsize()}")
            time.sleep(10)
```

它的代码很简单，循环检查Event是否设置，如果没有设置就打印队列大小并且sleep 10秒。否则退出。
注意：如果一开始Event没有设置，那么这个进程至少要运行10秒，所以如果要通过单元测试，必须不能设置--do_monitor(默认是False)。

#### _chunk_counter_process进程


```
    @staticmethod
    def _chunk_counter_process(chunk_queue, counter_queue, 
                               pattern, special_token_pattern):
        while True:
            chunk = chunk_queue.get()
            if chunk == None:
                break
            blocks = re.split(special_token_pattern, chunk)
            counter = defaultdict(int)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    counter[text] += 1
            counter_queue.put(counter)
```

代码和之前差不多，while循环里首先从chunk_queue获取chunk。如果chunk是None，说明任务已经完成，就可以break结束自己。


#### _merge_counter_process进程

```
    @staticmethod
    def _merge_counter_process(counter_queue, merged_queue):
        merged_counter = defaultdict(int)

        while True:
            counter = counter_queue.get()
            if counter == None:        
                break

            for k,v in counter.items():
                merged_counter[k] += v

        merged_queue.put(merged_counter)
```

退出方式和上面一样。

### 2.3 错误的通信方式

在实现过程中，我可能使用了一种错误的通信方式，从而导致了死锁。这一部分内容不感兴趣的读者可以跳过。

错误的代码在[bpe_v3_bug.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_bug.py)。

在前面的介绍里，主进程等待所有_chunk_counter_process进程结束之后放置哨兵None到counter_queue队列。而在这个错误的版本里，我是在_chunk_counter_process进程里放置哨兵None到counter_queue队列。

```

        running_counters = mp.Value('i', NUM_COUNTER_PROCESS)
        lock = mp.Lock()  


    @staticmethod
    def _chunk_counter_process(chunk_queue, counter_queue, 
                               pattern, special_token_pattern,
                               num_mergers, running_counters, lock):
        process_name = mp.current_process().name

        while True:
            chunk = chunk_queue.get()
            if chunk == None:
                break
            blocks = re.split(special_token_pattern, chunk)
            counter = defaultdict(int)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    counter[text] += 1
            debug_data = (process_name, counter)
            counter_queue.put(debug_data)

        with lock:
            running_counters.value -= 1
            finished = running_counters.value == 0

        
        if finished:
            print(f"{process_name} is the last one")
            for _ in range(num_mergers):
                counter_queue.put(None)            

        print(f"{process_name} is done")      
```

代码逻辑是：每个进程结束后会加锁并且给mp.Value减一，在主进程里我们构造running_counters等于_chunk_counter_process进程数。如果running_counters.value == 0，则说明当前进程是最后一个完成的进程，那么就可以由它来放置哨兵。

这个逻辑看起来没有什么问题，但是运行后会死锁，通过分析会发现counter_queue的None后面还会有非None的正常数据。为什么会出现这个我也不是特别清楚，我也没有查到Queue是怎么实现的，也暂时没有精力研究CPython的源代码。因为操作系统提供的进程间通信API通常是点对点的，虽然逻辑上应该有一个公共的地方来放队列里的数据，但是我猜测实现的时候也许还是生产者进程和消费者进程的点对点通信。而且实现时可能是counter_queue.put()把数据放到缓冲区里就返回了，然后通过后台线程把它发送到消费者的内存。因为counter_queue的消费者提前收到哨兵None退出，导致_chunk_counter_process进程的后台线程无法结束，因此p.join()就会无限等待。


多线程/多进程的同步非常tricky，不像Java或者C++的内存模型，我找不到Python的多进程同步详细的资料。关于这个问题，我也在[discuss.python.org](https://discuss.python.org/t/multiprocessing-queue-put-order/101975)提问了，但是并没有找到答案。如果读者找到了答案，请告诉我，谢谢！

## 3. 测试

为了统计时间，我实现了[bpe_v3_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_time.py)，这个和bpe_v3相比，除了增加计时外还增加了两个参数：

```
        parser.add_argument("--skip_merge",
                            action="store_true",
                            help="skip merge (default: False)"
        ) 
        parser.add_argument("--chunk_size", 
                            type=str, 
                            default=f"{CHUNK_SIZE}", 
                            help="chunk_size")
```

* --skip_merge 它的作用是跳过第二步，因为我们只修改了第一步，没有必要跑第二步，尤其是在openweb上第二步需要十几个小时。
* --chunk_size 它的作用是设置chunk的大小


## 4. 测试结果

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v1_time | tinystory |2187/2264/2247|622/642/628| count_pair:944/995/984 max:167/173/174/174 update:453/453/460 |
bpe_v2_time | tinystory |757/738/746|639/621/627| 118/117/118 |
bpe_v2_time | openweb |35358/34265/35687|2870/2949/2930| 32437/31264/32708 |
bpe_v3_time | tinystory |250/90/90|80/90/90| 170/0/0 | num_counter=8, num_merger=1 skip
bpe_v3_time | openweb |391/401/32352|390/400/410| total:31883 max:31187 update:695 | num_counter=8, num_merger=1 skip
bpe_v3_time | openweb | |210/210/220|   | num_counter=16, num_merger=2
bpe_v3_time | openweb | |120/120/120|   | num_counter=32, num_merger=4
bpe_v3_time | openweb | |120/130/130|   | num_counter=64, num_merger=4


## 5. 结果分析

从openweb数据集上，我们看到从8个进程、16个进程到32个进程，时间都接近减半，这是比较理想的加速比。但是到了64个进程，时间并没有减少，这是什么原因呢？我们来看一次运行的日志：

```
args=Namespace(trainer='bpe_v3_time', out_dir='test_openweb_v3_64_8_1', vocab_size=32000, data_path='./data/owt_train.txt')
unknown_args=['-c', '64', '-m', '8', '--do_monitor', '--skip_merge']
train: args=Namespace(num_counter=64, num_merger=8, do_monitor=True, skip_merge=True)
chunk_queue: 0, counter_queue: 0, merged_queue: 0
chunk_queue: 684, counter_queue: 3, merged_queue: 0
chunk_queue: 1797, counter_queue: 5, merged_queue: 0
chunk_queue: 3541, counter_queue: 0, merged_queue: 0
chunk_queue: 4123, counter_queue: 3, merged_queue: 0
chunk_queue: 7059, counter_queue: 2, merged_queue: 0
chunk_queue: 9378, counter_queue: 2, merged_queue: 0
chunk_queue: 9969, counter_queue: 1, merged_queue: 0
chunk_queue: 6295, counter_queue: 3, merged_queue: 0
chunk_queue: 2093, counter_queue: 2, merged_queue: 0
chunk_queue: 250, counter_queue: 4, merged_queue: 0
chunk_queue: 0, counter_queue: 0, merged_queue: 4
chunk_queue: 0, counter_queue: 0, merged_queue: 0
_pretokenize_and_count_mp: 130.14884584583342
skip merge
total train time: 130.37 seconds
```

监控进程每10秒打印一次队列的大小。我们发现counter_queue几乎是空的，说明第三步_merge_counter_process进程的速度是没有问题的。但是chunk_queue队列比较大，说明_chunk_counter_process进程速度不够。但是实时运行时，我监控发现64个进程用的时间CPU差不多总cpu的50%(测试服务器有64个cpu)，这说明_chunk_counter_process进程没有完全把cpu跑满。而且在32个进程测试时，我监控系统发现程序是完全把32个CPU用满的。我猜测是因为Python的队列处理速度跟不上，所以我把chunk_size从默认的50K增加到了2M。测试结果如下：

```
rgs=Namespace(trainer='bpe_v3_time', out_dir='test_openweb_v3_64_8_2mb_1', vocab_size=32000, data_path='./data/owt_train.txt')
unknown_args=['-c', '64', '-m', '8', '--chunk_size', '2mb', '--do_monitor', '--skip_merge']
train: args=Namespace(num_counter=64, num_merger=8, do_monitor=True, skip_merge=True, chunk_size='2mb')
chunk_size=2097152
chunk_queue: 0, counter_queue: 0, merged_queue: 0
chunk_queue: 3, counter_queue: 1, merged_queue: 0
chunk_queue: 2, counter_queue: 0, merged_queue: 0
chunk_queue: 1, counter_queue: 0, merged_queue: 0
chunk_queue: 1, counter_queue: 1, merged_queue: 0
chunk_queue: 2, counter_queue: 0, merged_queue: 0
chunk_queue: 1, counter_queue: 0, merged_queue: 0
chunk_queue: 1, counter_queue: 1, merged_queue: 0
chunk_queue: 1, counter_queue: 1, merged_queue: 0
chunk_queue: 0, counter_queue: 2, merged_queue: 0
chunk_queue: 0, counter_queue: 0, merged_queue: 6
chunk_queue: 0, counter_queue: 0, merged_queue: 1
_pretokenize_and_count_mp: 120.22874877601862
skip merge
total train time: 120.42 seconds
```

从这个日志可以看到，chunk_queue和counter_queue几乎也是空，这说明系统瓶颈可能在chunk的生产者_chunk_documents_streaming。

为了验证，我做了一些测试。首先是测试磁盘的读取速度。

```
dd if=data/owt_train.txt of=/dev/null bs=1M status=progress
11920511059字节(12 GB)已复制，34.6439 秒，344 MB/秒
```

发现读取owt_train.txt需要34秒，读取速度只有344 MB/秒。这个速度明显太慢，通过分析，才发现这个路径是挂着在NFS上的，因此这个速度是网速的上限。于是我把owt_train.txt复制到本地磁盘，再进行测试：

```
dd if=~/data/owt_train.txt of=/dev/null bs=1M status=progress
11920511059字节(12 GB)已复制，7.15933 秒，1.7 GB/秒
```

这个速度就快多了，能到1.7GB/秒。为了验证，又用cat读取一遍：

```
time cat ~/data/owt_train.txt > /dev/null

real    0m7.066s
user    0m0.004s
sys     0m4.001s
```

和dd的速度差不多，这说明磁盘的速度就是1.7GB/秒。

那会不会是Python读取文件的速度很慢呢？于是写了个简单程序[test_readspeed.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed.py)进行测试：

```
read from nfs
time: 53.79s
total_chars=11815998173
            
read from local disk
time: 51.55s
total_chars=11815998173
```

Python逐行读取文件比dd和cat慢了很多，不管是从nfs还是本地磁盘，速度都差不多200 MB/s。这是什么原因呢？通过各种搜索，猜测可能是Python读取utf-8文件时需要解码导致的。为了验证，我写了个读取二进制格式的代码[test_readspeed_rb.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed_rb.py)，测试如下：

```
本地磁盘
python cs336_basics/test_readspeed_rb.py ~/data/owt_train.txt (buffer 64k, read 8k)
time: 9.19s
total_bytes=11920511059 

nfs
python cs336_basics/test_readspeed_rb.py data/owt_train.txt 
time: 30.48s
total_bytes=11920511059
```

上面的代码是每次读取8k字节。但是我们需要按行处理，因此也写了[test_readspeed_rb_line.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed_rb_line.py)，测试结果如下：

```
python cs336_basics/test_readspeed_rb_line.py ~/data/owt_train.txt
time: 25.95s
total_bytes=11920511059, total_lines=94568885
```

用二进制读取固定大小的8k字节，本地磁盘和nfs的速度都接近dd命令。但是用readline按行读取就慢了不少，我猜测是因为每行的字符较少，这样循环的次数就增多。owt_train.txt总共11920511059，如果每次读取8k，那么只需要循环1422次。但是按行读取需要94568885行。


如果想要优化，可能想到的方法就是使用多个进程读取文件的不同部分。但这样优化行得通吗？我也写了一个程序测试多进程读取，代码在[test_readspeed_mp.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed_mp.py)。

通过测试发现多进程的速度比单进程要慢很多！在tinystory上8进程要慢10倍以上，openweb就慢的我忍不住ctrl+c了。分析原因也很简单，磁盘顺序读取是最快的，如果多个进程并发的读取不同位置，那么磁头就必须反复定位，这样反而更慢。


我们目前的程序32进程的时间是120秒。这120秒有50秒是从磁盘读取到主进程的内存的时间，我猜测从主进程的内存复制到其它进程内存也需要不少时间(没有测试过，虽然内存到内存的复制要比磁盘到内存快，但是Python的进程间通信成本也很高)。如果我们想继续优化它，期望在64个CPU的时候能减少到60秒。那么可能的思路是_chunk_documents_streaming函数用二进制的方式把数据读取一个chunk，chunk不是str而是bytes，按照前面的测试只需要25.95s。然后多个_chunk_counter_process进程首先把bytes解码成str，然后再进行后续的处理。

## 6. 用bytes优化

代码在[bpe_v3_bytes_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_bytes_time.py)，把所有str都改成了bytes：

```
    @staticmethod
    def _chunk_documents_streaming(
        path: str,
        chunk_size: int = CHUNK_SIZE,
        special_token: str = "<|endoftext|>"
    ):

        leftover = b""
        special_token_bytes = special_token.encode("utf-8")
        token_len = len(special_token_bytes)

        with open(path, "rb") as f:
            while True:
                # read one chunk_size block of text
                block = f.read(chunk_size)
                if not block:
                    # no more data in file
                    break

                # combine leftover from previous iteration + new block
                block = leftover + block
                leftover = b""

                # find the *last* occurrence of the special token in 'block'
                last_eot_idx = block.rfind(special_token_bytes)

                if last_eot_idx == -1:
                    # no complete document in this chunk
                    # keep everything in leftover for the next read
                    leftover = block
                else:
                    # up through last_eot_idx is a complete set of docs
                    yield block[: last_eot_idx + token_len]
                    # keep everything after that boundary as leftover
                    leftover = block[last_eot_idx + token_len:]

        # yield leftover text
        if leftover:
            yield leftover
```

```
    @staticmethod
    def _chunk_counter_process(chunk_queue, counter_queue, 
                               pattern, special_token_pattern):
        while True:
            chunk = chunk_queue.get()
            if chunk == None:
                break
            chunk = chunk.decode("utf-8")
            blocks = re.split(special_token_pattern, chunk)
            counter = defaultdict(int)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    counter[text] += 1
            counter_queue.put(counter)
```

当我们用64个cpu，并且chunk_size是8mb时，时间减少到70秒。基本达到了我们的预期。实际监控系统发现cpu的利用率也提高到了90%以上，说明文件读取速度已经不是瓶颈。这可以通过日志来印证：

```
args=Namespace(trainer='bpe_v3_bytes_time', out_dir='test_openweb_v3_bytes_64_4_8mb_1', vocab_size=32000, data_path='~/data/owt_train.txt')
unknown_args=['--chunk_size', '8mb', '-c', '64', '-m', '4', '--do_monitor', '--skip_merge']
train: args=Namespace(num_counter=64, num_merger=4, do_monitor=True, skip_merge=True, chunk_size='8mb')
chunk_size=8388608
chunk_queue: 0, counter_queue: 0, merged_queue: 0
chunk_queue: 955, counter_queue: 11, merged_queue: 0
chunk_queue: 1004, counter_queue: 5, merged_queue: 0
chunk_queue: 745, counter_queue: 0, merged_queue: 0
chunk_queue: 472, counter_queue: 2, merged_queue: 0
chunk_queue: 200, counter_queue: 0, merged_queue: 0
chunk_queue: 0, counter_queue: 0, merged_queue: 2
_pretokenize_and_count_mp: 70.26211958006024
skip merge
total train time: 70.52 seconds
```

我们可以看到chunk_queue队列不为空，这说明_chunk_counter_process进程把64个cpu用满了也无法跟上文件读取的速度，因此系统目前的瓶颈从I/O再次变为CPU。

说明：最后一个bpe_v3_bytes_time.py的优化是写文章的时候顺便做的，后面很多的代码很多还是基于bpe_v3.py进行优化的。


## 7. 测试结果

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v1_time | tinystory |2187/2264/2247|622/642/628| count_pair:944/995/984 max:167/173/174/174 update:453/453/460 |
bpe_v2_time | tinystory |757/738/746|639/621/627| 118/117/118 |
bpe_v2_time | openweb |35358/34265/35687|2870/2949/2930| 32437/31264/32708 |
bpe_v3_time | tinystory |250/90/90|80/90/90| 170/0/0 | num_counter=8, num_merger=1 skip
bpe_v3_time | openweb |391/401/32352|390/400/410| total:31883 max:31187 update:695 | num_counter=8, num_merger=1 skip
bpe_v3_time | openweb | |210/210/220|   | num_counter=16, num_merger=2
bpe_v3_time | openweb | |120/120/120|   | num_counter=32, num_merger=4
bpe_v3_time | openweb | |120/130/130|   | num_counter=64, num_merger=4
bpe_v3_bytes_time | openweb | |80/90/80|   | num_counter=64, num_merger=8, chunk_size 1mb
bpe_v3_bytes_time | openweb | |70/70/80|   | num_counter=64, num_merger=8, chunk_size 4mb
bpe_v3_bytes_time | openweb | |70/70/70|   | num_counter=64, num_merger=8, chunk_size 8mb


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
