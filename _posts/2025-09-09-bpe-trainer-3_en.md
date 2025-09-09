---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 3: Parallel word segmentation and word frequency counting" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the fourth article, using parallel algorithms for acceleration.

<!--more-->


**Table of Content**
* TOC
{:toc}

-----

Here is the English translation of the provided content, preserving the markdown formatting.

## 1\. Algorithm Optimization

### 1.1 Current Algorithm Bottleneck Analysis

Based on previous experiments, on the tinystory dataset, the `_pretokenize_and_count` step took over 600 seconds, and the merge step took over 100 seconds. On the openweb dataset, `_pretokenize_and_count` took close to 3000 seconds, while the merge step took over 30,000 seconds. Both of these steps need optimization, but in this article, we will first optimize the first step. There are two reasons for this: First, we want to solve the simpler problem first. The main computation in the first step is regular expression matching and word frequency counting, which are very easy to accelerate using parallel algorithms. Second, for actual LLM pre-training, the text data is far larger than 10GB. The first step has a time complexity of O(n), while the second step, according to [Heaps' law](https://en.wikipedia.org/wiki/Heaps%27_law), has a sub-linear growth rate of vocabulary relative to the corpus. Therefore, if the training text increases 100-fold, the time for the first step will also increase linearly by 100-fold. However, the second step is based on merging word frequencies, and its growth rate is relatively small.

To clarify, Heaps' law describes the relationship between the **corpus size ($N$)** and the **vocabulary size ($V$)**. The vocabulary size is the total number of unique words in the corpus.

$$V = K \cdot N^{\beta}$$

where:

  * **$V$** is the Vocabulary size.
  * **$N$** is the Corpus size, typically measured in the number of words.
  * **$K$** is a constant, whose value depends on the specific corpus.
  * **$\\beta$** is an exponent, usually between $0.4$ and $0.6$, with the most common approximation being $0.5$.

### 1.2 Optimization Method

`_pretokenize_and_count` is **embarrassingly parallel**. We can split the original text into multiple parts, using documents as the unit, and then use multiple CPUs to perform regular expression matching and count word frequencies. Finally, we can merge the counts from each part. Only the final merge step is serial; the rest of the process is very easy to parallelize.

To leverage multiple CPUs for parallel computing, the most common approach is to use multithreading. However, due to Python's GIL (Global Interpreter Lock), Python's `threading` can only achieve **concurrency** rather than **parallelism**. Python's multithreading is only suitable for I/O-bound tasks, not CPU-bound tasks. For CPU-bound tasks, it's more appropriate to use **multiprocessing**.

Furthermore, since our computation needs to be divided into many steps, using a queue and the producer-consumer pattern is a very clear way to implement it. Specifically, our workflow is as follows:

  * **Document Splitting**
      * Use the special document separator string `<|endoftext|>` to split the original text into multiple chunks, with each chunk containing one or more documents. The results are placed in `chunk_queue`.
  * **Regex Matching and Word Counting**
      * Split the chunk into documents using `<|endoftext|>`, then split the documents into words, and count their frequencies. The processed results are placed in `counter_queue`.
  * **Frequency Merging**
      * Take the word count results for each chunk from `counter_queue` and merge them. The final result is placed in `merged_queue`.
  * **Final Merge**
      * The main process merges the results from `merged_queue`.

A point to note is the granularity of the splitting. For example, in the **Regex Matching and Word Counting** step, we put the word count results for the entire chunk into `counter_queue` at once. This reduces the amount of data transferred between processes. We could also put the word count for each individual document into `counter_queue`, which wouldn't increase the total work, just shift the work to the **Frequency Merging** step. However, this method would increase the data transfer to `counter_queue` by a potential factor of 1000, if a single chunk contains 1000 documents.

For example, with two documents:

```
{"a":1, "b":2, "c":3, "d":4}
{"a":4, "b":3, "c":2, "e":1}
```

If transmitted separately, 8 key/value pairs need to be transferred. If merged first:

```
{"a":5, "b":5, "c":5, "d":4, "e":1}
```

Then only 5 key/value pairs need to be transferred. The ideal case is that the keys are all the same, which would reduce data transfer by half. But even in a non-ideal scenario, according to Heaps' law and [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law), high-frequency words can always be merged.

Because we are using `multiprocessing`, processes cannot share the same memory address space like threads can. Python must first serialize (pickle) the data to be transferred into a binary stream, transmit it through inter-process communication system APIs, and then deserialize the binary stream back into an object. Therefore, reducing inter-process communication can significantly optimize the efficiency of Python multiprocessing.

## 2\. Code

The source code can be found at [bpe\_v3.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3.py).

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

Compared to the previous version, the `train` function mainly replaced `_pretokenize_and_count` with `_pretokenize_and_count_mp`. Additionally, for easier experimentation and debugging, three parameters have been added:

  * `--num_counter/-c` The number of `_chunk_counter_process` processes, which corresponds to the second step in our four-step process.
  * `--num_merger/-m` The number of `_merge_counter_process` processes, corresponding to the third step.
  * `--do_monitor` A flag to periodically (every 10 seconds) print the current queue sizes, which is useful for debugging. If any queue is backed up, it indicates that the corresponding step is not processing fast enough.

### 2.2 \_pretokenize\_and\_count\_mp

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

This function is quite long, so let's break it down into sections.

#### Building Regex and Queues

```
        # pre-compile regex
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

        chunk_queue = mp.Queue(maxsize=1_000_000)
        counter_queue = mp.Queue(maxsize=1_000_000)
        merged_queue = mp.Queue(maxsize=num_merger)
```

First, we build the regex patterns `pattern` and `special_token_pattern` for tokenization and document splitting, which is the same as the previous version. We then create three queues: `chunk_queue`, `counter_queue`, and `merged_queue`. Their purposes were described earlier. Note that `merged_queue`'s size is set to `num_merger` because each merging process only produces one final word count result.

#### Starting `_chunk_counter_process` and `_merge_counter_process`

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

The `_chunk_counter_process` takes `chunk_queue` as input and produces `counter_queue` as output. The `_merge_counter_process` takes `counter_queue` as input and produces `merged_queue` as output. This represents the second and third steps of our process flow.

The first and fourth steps could also, in theory, be completed by separate processes. However, we know the fourth step must be serial, so we handle it directly in the main process to avoid an extra inter-process communication step. The main process also handles the first step because file reading is very fast relative to the rest of the process, and using multiple processes to read a single file does not make it faster (we will analyze file reading performance limits later).

Additionally, the code above will start a `_queue_moniter_process` based on the `do_monitor` argument. This process periodically prints the queue sizes for debugging purposes.

#### Main Process Reads File to Generate Chunks

```
        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            chunk_queue.put(chunk)

        for i in range(num_counter):
            chunk_queue.put(None)
```

This code block shows the main process calling the `_chunk_documents_streaming` function (which is a generator function) to read the file and produce chunks. Note that our `chunk_queue` has a size limit (we set it to 1,000,000). This prevents memory overflow if subsequent steps are slow, as `chunk_queue.put(chunk)` will block when the queue is full. This is one of the benefits of using a generator function with `yield`. If we were to read all chunks into a list directly, we would likely run into an Out-of-Memory (OOM) error (e.g., when reading a 1TB file).

Once all chunks have been put into `chunk_queue`, the main process places `num_counter` `None` objects into it. This is a common method in the producer-consumer pattern using a **sentinel** to signal that all tasks have been completed. A special object is used so consumers know when to safely stop.

#### Main Process Waits for `_chunk_counter_process` and Notifies `_merge_counter_process`

```
        for p in counter_processes:
            p.join()

        for _ in range(num_merger):
            counter_queue.put(None)
```

`p.join()` waits for all `_chunk_counter_process` to finish (they terminate themselves after receiving the sentinel `None`). Then, `num_merger` `None` objects are put into `counter_queue`, which serve as sentinels to tell the `_merge_counter_process` to terminate.

#### Main Process Merges Results

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

Next, the main process merges the word counts. If `num_merger` is 1, no merging is needed. Otherwise, it retrieves the `num_merger` results from the queue and merges them.

#### Waiting for Processes to Finish and Returning Results

```
        for p in merge_processes:
            p.join() 

        if do_monitor:
            stop_event.set()
            monitor_process.join()    

        return word_counts
```

The main process waits for all merge processes to finish. If a monitor process was started, it is signaled to stop via the `Event` object.

**Note**: Here, the main process merges the `merged_queue` results **before** waiting for all `merge_processes` to finish. Some might wonder why not wait for all processes to finish first. This is because the `merge_processes` don't all finish at the same time. By merging as soon as a result is available in `merged_queue`, the main process can start its work immediately. In the ideal case, by the time the last `_merge_counter_process` finishes, the results from the other `num_merger - 1` processes have already been merged.

We are manually managing processes here, which can be tedious. If using a [process pool](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool), you could use [pool.imap\_unordered](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap_unordered). I encourage readers to try implementing this using a process pool and task submission. However, I find the producer-consumer pattern to be logically clearer and easier to understand.

#### Monitor Process

Let's start with the simplest monitor process.

```
    @staticmethod
    def _queue_moniter_process(chunk_queue, counter_queue, merged_queue, event):
        while not event.is_set():
            print(f"chunk_queue: {chunk_queue.qsize()}, counter_queue: {counter_queue.qsize()}, merged_queue: {merged_queue.qsize()}")
            time.sleep(10)
```

The code is simple: it loops, checking if the `Event` is set. If not, it prints the queue sizes and sleeps for 10 seconds. Otherwise, it exits.
Note: If the `Event` is not set initially, this process will run for at least 10 seconds, so to pass unit tests, `--do_monitor` must be set to `False` (which is the default).

#### `_chunk_counter_process`

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

The code is similar to before. In the while loop, it gets a chunk from `chunk_queue`. If the chunk is `None`, it breaks to terminate itself.

#### `_merge_counter_process`

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

The exit method is the same as above.

### 2.3 Incorrect Communication Method

During implementation, I used an incorrect communication method that led to a deadlock. Readers not interested in this can skip this section.

The buggy code is at [bpe\_v3\_bug.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_bug.py).

In the previous explanation, the main process waits for all `_chunk_counter_process` processes to finish before placing the `None` sentinels into `counter_queue`. In this buggy version, I had the `_chunk_counter_process` processes themselves place the `None` sentinels into `counter_queue`.

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

The logic seems fine: each process, upon completion, acquires a lock, decrements an `mp.Value`, and checks if it's the last one. If so, it places the sentinels. The `running_counters` is initialized by the main process to the number of `_chunk_counter_process` processes.

This logic seems fine at first glance, but it leads to a deadlock. Through analysis, I discovered that non-`None` data, which should be regular tasks, still appears in the `counter_queue` after the `None` sentinel. I'm not entirely sure why this happens, as I haven't been able to find documentation on how Python's `Queue` is implemented and currently don't have the time to dive into the CPython source code.

While message queue typically use a central location to store data, my guess is that the implementation relies on point-to-point communication between the producer and consumer processes. It's possible that `counter_queue.put()` places data into a buffer and returns immediately, with a background thread then sending it to the consumer's memory. Since the consumer receives the `None` sentinel early and exits, the background thread in the `_chunk_counter_process` can't finish, which causes `p.join()` to wait indefinitely.

Synchronization in multithreading/multiprocessing is very tricky, and I haven't found detailed documentation on Python's multiprocessing memory model, unlike Java or C++. I also posted this question on [discuss.python.org](https://discuss.python.org/t/multiprocessing-queue-put-order/101975) but haven't found an answer. If any reader knows the reason, please let me know\!

## 3\. Testing

To measure the timing, I implemented [bpe\_v3\_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_time.py). Compared to `bpe_v3`, this version adds timing and two additional arguments:

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

  * `--skip_merge` Skips the second step. Since we only modified the first step, it's unnecessary to run the second step, especially on openweb where it takes over ten hours.
  * `--chunk_size` Sets the size of each chunk.

## 4\. Test Results

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe\_v1\_time      | tinystory | 2187/2264/2247| 622/642/628        | count\_pair:944/995/984 max:167/173/174/174 update:453/453/460 |
bpe\_v2\_time      | tinystory | 757/738/746   | 639/621/627        | 118/117/118 |
bpe\_v2\_time      | openweb   | 35358/34265/35687| 2870/2949/2930 | 32437/31264/32708 |
bpe\_v3\_time      | tinystory | 250/90/90     | 80/90/90           | 170/0/0 | num\_counter=8, num\_merger=1 skip
bpe\_v3\_time      | openweb   | 391/401/32352 | 390/400/410        | total:31883 max:31187 update:695 | num\_counter=8, num\_merger=1 skip
bpe\_v3\_time      | openweb   |               | 210/210/220        |               | num\_counter=16, num\_merger=2
bpe\_v3\_time      | openweb   |               | 120/120/120        |               | num\_counter=32, num\_merger=4
bpe\_v3\_time      | openweb   |               | 120/130/130        |               | num\_counter=64, num\_merger=4

## 5\. Result Analysis

From the openweb dataset, we see that the time taken is almost halved when increasing the number of processes from 8 to 16, and then to 32, which is a near-ideal speedup. However, at 64 processes, the time does not decrease. Why is this? Let's look at a log from a run:

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

The monitor process prints the queue sizes every 10 seconds. We can see that `counter_queue` is almost empty, which means the speed of the third step (`_merge_counter_process`) is not the bottleneck. However, `chunk_queue` is quite large, indicating that the `_chunk_counter_process` speed is insufficient. During the real-time run, I monitored the system and found that the 64 processes were using about 50% of the total CPU (the test server has 64 CPUs), which suggests that the `_chunk_counter_process` processes are not fully utilizing the CPUs. In the 32-process test, I observed that the program was fully utilizing all 32 CPUs. I suspect that the bottleneck might be Python's queue processing speed. So I increased the `chunk_size` from the default 50K to 2MB. The test results are as follows:

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

From this log, we can see that `chunk_queue` and `counter_queue` are almost empty, which means the system bottleneck might be the chunk producer, `_chunk_documents_streaming`.

To verify this, I ran some tests. First, I tested the disk read speed.

```
dd if=data/owt_train.txt of=/dev/null bs=1M status=progress
11920511059 bytes (12 GB) copied, 34.6439 s, 344 MB/s
```

Reading `owt_train.txt` took 34 seconds, with a speed of only 344 MB/s. This speed is too slow. Upon analysis, I found that the path was a network mount on NFS, so this speed was limited by the network speed. I then copied `owt_train.txt` to a local disk and tested again:

```
dd if=~/data/owt_train.txt of=/dev/null bs=1M status=progress
11920511059 bytes (12 GB) copied, 7.15933 s, 1.7 GB/s
```

This speed is much better, reaching 1.7 GB/s. To verify, I also used `cat`:

```
time cat ~/data/owt_train.txt > /dev/null

real     0m7.066s
user     0m0.004s
sys      0m4.001s
```

The speed is similar to `dd`, which confirms the disk speed is about 1.7 GB/s.

But is Python's file reading slow? I wrote a simple script [test\_readspeed.py](https://www.google.com/search?q=https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed.py) to test this:

```
read from nfs
time: 53.79s
total_chars=11815998173
             
read from local disk
time: 51.55s
total_chars=11815998173
```

Python's line-by-line reading is much slower than `dd` and `cat`, regardless of whether the file is on NFS or a local disk, with speeds around 200 MB/s. Why? After some searching, I suspect it's due to the overhead of decoding UTF-8. To test this, I wrote code to read in binary mode, [test\_readspeed\_rb.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed_rb.py):

```
Local disk
python cs336_basics/test_readspeed_rb.py ~/data/owt_train.txt (buffer 64k, read 8k)
time: 9.19s
total_bytes=11920511059 

NFS
python cs336_basics/test_readspeed_rb.py data/owt_train.txt 
time: 30.48s
total_bytes=11920511059
```

Reading a fixed size of 8KB at a time in binary mode, the speeds on both local disk and NFS were close to the `dd` command. However, using `readline` to read line by line was much slower, as shown by the test of [test\_readspeed\_rb\_line.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed_rb_line.py):

```
python cs336_basics/test_readspeed_rb_line.py ~/data/owt_train.txt
time: 25.95s
total_bytes=11920511059, total_lines=94568885
```

This is because each line is short, increasing the number of loops. `owt_train.txt` has a total of 11,920,511,059 bytes. Reading 8KB at a time only requires about 1,422,000 loops, while reading line by line requires over 94 million loops.

One might consider using multiple processes to read different parts of the file to optimize. Does this approach work? I wrote a program to test multi-process reading, [test\_readspeed\_mp.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_readspeed_mp.py).

The tests showed that multi-process reading is much slower than single-process reading\! On tinystory, 8 processes were more than 10 times slower, and for openweb, it was so slow I had to `ctrl+c` out of it. The reason is simple: sequential disk reads are the fastest. If multiple processes concurrently read from different locations, the disk head must constantly reposition, which slows things down.

Our current program with 32 processes takes 120 seconds. Of this time, 50 seconds are for reading from disk into the main process's memory. I suspect that copying from the main process's memory to other processes also takes a significant amount of time (I haven't tested this, but Python's inter-process communication overhead is high). If we want to continue optimizing and aim for a 60-second runtime with 64 CPUs, a possible solution is to read the data in binary mode in the `_chunk_documents_streaming` function, making the chunk a `bytes` object instead of a `str`. According to my tests, this step would only take 25.95 seconds. Then, the `_chunk_counter_process` processes would decode the `bytes` to `str` before proceeding with their tasks.

## 6\. Optimization with Bytes

The code for this is at [bpe\_v3\_bytes\_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v3_bytes_time.py), where all strings have been changed to bytes.

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

When using 64 CPUs and a `chunk_size` of 8MB, the time was reduced to 70 seconds, which is a big improvement. My system monitoring confirmed that CPU utilization increased to over 90%, showing that file reading is no longer the bottleneck. The log confirms this:

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

We can see that `chunk_queue` is not empty, which indicates that the `_chunk_counter_process` processes are maxing out all 64 CPUs and still cannot keep up with the file reading speed. The system's bottleneck has shifted from I/O back to CPU.

Note: The optimization in `bpe_v3_bytes_time.py` was done while writing this article, so much of the code of latter articles is based on `bpe_v3.py`.

## 7\. Final Test Results

Version           | Data      | Total Time(s) | Word Count Time(s) | Merge Time(s) | Other
bpe\_v1\_time      | tinystory | 2187/2264/2247| 622/642/628        | count\_pair:944/995/984 max:167/173/174/174 update:453/453/460 |
bpe\_v2\_time      | tinystory | 757/738/746   | 639/621/627        | 118/117/118 |
bpe\_v2\_time      | openweb   | 35358/34265/35687| 2870/2949/2930 | 32437/31264/32708 |
bpe\_v3\_time      | tinystory | 250/90/90     | 80/90/90           | 170/0/0 | num\_counter=8, num\_merger=1 skip
bpe\_v3\_time      | openweb   | 391/401/32352 | 390/400/410        | total:31883 max:31187 update:695 | num\_counter=8, num\_merger=1 skip
bpe\_v3\_time      | openweb   |               | 210/210/220        |               | num\_counter=16, num\_merger=2
bpe\_v3\_time      | openweb   |               | 120/120/120        |               | num\_counter=32, num\_merger=4
bpe\_v3\_time      | openweb   |               | 120/130/130        |               | num\_counter=64, num\_merger=4
bpe\_v3\_bytes\_time| openweb   |               | 80/90/80           |               | num\_counter=64, num\_merger=8, chunk\_size 1mb
bpe\_v3\_bytes\_time| openweb   |               | 70/70/80           |               | num\_counter=64, num\_merger=8, chunk\_size 4mb
bpe\_v3\_bytes\_time| openweb   |               | 70/70/70           |               | num\_counter=64, num\_merger=8, chunk\_size 8mb
 
