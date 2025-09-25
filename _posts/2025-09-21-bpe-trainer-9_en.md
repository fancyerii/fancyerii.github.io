---
layout:     post
title:      "Implementing and Optimizing a BPE Tokenizer from Scratch—Part 9: Using a Heap to Find the Maximum Pair" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

This series of articles implements a subtask of Stanford’s CS336 Assignment 1: building an efficient training algorithm for a BPE Tokenizer. Through a series of optimizations, our algorithm’s training time on OpenWebText was reduced from over 10 hours to less than 10 minutes. This series explains these optimizations, including algorithmic improvements, data structure enhancements, parallelization with OpenMP, Cython optimization, and implementing key code in C++ along with its integration via Cython. This is the tenth article, where we use the heap data structure to replace the process of finding the maximum pair, thereby improving performance.

<!--more-->


**Table of Content**
* TOC
{:toc}

## 1\. Problem Analysis

In our `bpe_v5_time` from above, the update time was just over 500 seconds, but the time for finding the max was over 30,000 seconds. How can we optimize this time? We previously tried a parallel algorithm for finding the max, but the Python version was not successful because the Python GIL prevents multiple threads from using multiple CPUs simultaneously, and multiprocessing involves a lot of inter-process communication. We then tried a C++ implementation with OpenMP for parallelism, which reduced the max time to 400 seconds with 32 threads. We then replaced `std::unordered_map` with a faster flat hashmap, which brought the max time down to under 100 seconds without parallelism.

However, let's go back to Python. Besides optimizing data structures, is there a way to optimize the algorithm itself? Optimizing data structures means doing the same amount of work faster, while optimizing an algorithm means achieving the same goal with less work. Of course, the two are not completely separate; sometimes, a different algorithm requires designing an appropriate data structure.

To find the maximum value in a set, we must iterate through the entire set, and this time cannot be reduced. This means the time for the first traversal cannot be reduced. However, the second traversal can be optimized because only a portion of the pair counts change between traversals. At the beginning, the affected words are numerous due to high-frequency terms, so relatively more pairs are affected. Later on, as word frequencies decrease, fewer words are affected. The simplest idea is to sort the `pair_counts`. When some pair counts change, we would only need to re-sort those specific pairs. Since there are only two possibilities for changes in word frequency—counts of old pairs decrease, and counts of new pairs increase—we could also use some heuristic rules. For example, if a pair's current rank is greater than 32,000 (our target vocabulary size) and its frequency has decreased, we can ignore it.

But there is a better way to find the maximum value in a set: using a heap data structure.

## 2\. Heap

In computer science, "heap" has two completely different meanings: one refers to a data structure, and the other refers to a type of memory allocation. Interestingly, the counterpart to the heap as a memory allocation type is the stack, which also represents both a memory allocation method and a data structure. The stack as a data structure (LIFO) is closely related to the stack as memory allocation, as it leverages the LIFO principle for function calls and returns. However, the heap as a data structure and the heap as memory allocation have no connection; their only link is that someone, for some reason, used the same word for two entirely different concepts.

Here, we're focusing on the heap as a data structure. It's typically used to implement priority queues and can also be used for heapsort. I won't go into a detailed introduction of heaps; readers who are unfamiliar can find plenty of resources in any data structures and algorithms book or online, for example, on [Wikipedia](https://en.wikipedia.org/wiki/Heap_\(data_structure\)).

## 3\. Using a Heap to Find the Max

Typically, when using a heap, we only use three main operations: converting an array into a heap (**heapify**), popping the top element from the heap (**heappop**), and pushing an element into the heap (**heappush**). We first need to use `heapify` to convert an array into a heap (satisfying the heap property: the root is larger/smaller than every element in its subtrees). Then, we can repeatedly call `heappop` and `heappush`, and the array will remain a heap after these operations.

There's a problem here: after finding the maximum (with `heappop`), we merge a pair. This decreases the count of some old pairs and introduces some new ones. Adding new pairs is not an issue; we just need to call `heappush`. But how do we modify the counts of old pairs and keep the structure a valid heap? Let's look at an example:

```
         11
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   / \
6  4  5   7
```

If we change 8 to 10, meaning an element has increased, we need to call a "sift up" operation (`siftup`) starting from 8:

```
         11
       /    \
      /      \
     10       4
   /   \     / \
  7     9   3   1 
 / \   / \
6  4  5   7
```

Conversely, if we change 8 to 6, we need to call a "sift down" operation (`siftdown`) starting from 8:

```
         11
       /    \
      /      \
     10       4
   /   \     / \
  7     7   3   1 
 / \   / \
6  4  5   6
```

But there's a problem: how do we find the element to be modified? Recall that our main data is `pair_counts`, which is a `dict`. However, a heap operates on a list. So we need to copy the elements from `pair_counts` into a `pair_heap` list. But a list doesn't support fast lookups; if we had to sequentially scan the list to find an element for modification, it would be a counterproductive effort (we've already found the max).

One solution is to store the index of the element in the list within the `pair_counts` dictionary, so `pair_counts` would become:

```
pair -> (count, index_in_pair_heap)
```

Then, when we add a new pair to `pair_counts`, we also add it to the correct position in `pair_heap` using `heappush`. This would require the `heappush` function to not only add an element but also return its index in `pair_heap`. This way, we can save the index in `pair_counts`. Later, when the count of a pair changes (in our case, it will only decrease), we can find its position in `pair_heap` via `pair_counts` and then call `siftdown` on that element.

This would require modifying `heappush` and maintaining the relationship between `pair_counts` and `pair_heap`, which makes the code quite complex. Readers who are interested can try to implement this algorithm.

However, I'm using a different approach—**lazy modification**. With this method, we do nothing when a pair's count changes. We only check if its count has changed when we call `heappop`, which we can discover by comparing the heap element's count with the value in `pair_counts`. If it has changed (it will only get smaller, which is a very important assumption), we re-insert the new count using `heappush`. We then continue to `heappop` until we find an element whose count has not been modified. That element is the current maximum.

This sounds complicated, so let's walk through an example. Suppose our current heap is:

```
         11
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   / \
6  4  5   7
```

The maximum should be 11, but let's assume that due to a merge, its count has changed to 10. When we pop the pair with 11, we query `pair_counts` and find its latest count is 10. Since it's smaller, we can't be sure it's the max. So, we first pop 11, which gives us:

```
          9
       /    \
      /      \
     8        4
   /   \     / \
  7     7   3   1 
 / \   /
6  4  5 
```

Then we need to re-push 10, resulting in:

```
         10
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   /  \
6  4  5    7
```

Next, we pop the current maximum, 10. This time, a query to `pair_counts` reveals its count is up-to-date, so we've found the current maximum pair is 10.

**Important Note**: The crucial assumption that allows for this lazy update is that the count of an old pair will **only decrease**. If this assumption were not true, for example, if our heap were:

```
         11
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   / \
6  4  5   7
```

And we changed 1 to 12, we would have to `siftup` 1 immediately, otherwise the max we find would be 11, which is incorrect.

Using this algorithm, we don't need to maintain `pair_heap` indices in `pair_counts`, and we don't need to call `siftdown`, which is a private function in Python's `heapq` (`_siftdown`), making its use risky as it might not be available in future versions.

## 4\. Python's `heapq` Module

Python's standard library provides the `heapq` module. The main functions we need are `heappush`, `heappop`, and `heapify`. Readers who are unfamiliar can refer to [The Python heapq Module: Using Heaps and Heappushs](https://realpython.com/python-heapq-module/).

However, there is an issue here: we need a **max heap**, but Python's `heapq` module provides a **min heap**. Later, when we analyze its code, we'll see that it has already implemented a max heap internally. But for now, we need to discuss how we can use a min heap to achieve the functionality of a max heap.

A common trick is to reverse the elements. For example, if the heap elements are positive integers, we can store their corresponding negative integers to simulate a max heap. Here's an example:

```python
import heapq
arr = [3, 5, 1, 2, 6, 8, 7]

arr2 = [-i for i in arr]

arr2
[-3, -5, -1, -2, -6, -8, -7]

heapq.heapify(arr2)
arr2
[-8, -6, -7, -2, -5, -1, -3]

arr3 = [-i for i in arr2]
arr3
[8, 6, 7, 2, 5, 1, 3]
```

But now we need to put tuples `(count, pair_string[pair], pair)` into the heap. If `count` is large, the tuple is large; if `count` is the same, we compare the pair's string and choose the larger one. The pair is included as the last element of the tuple for convenience.

Following the above method, we can put `-count` into the heap. But what about `pair_strings[pair]`? `pair_strings[pair]` is a tuple of bytes. If the bytes are of fixed length, since a byte's range is 0-255, we can reverse it by subtracting each byte from 255. For example:

```python
b1 = b'us'

b2 = b'ua'

b1 > b2
True

c1 = bytes([255 - b for b in b1])
c1
b'\x8a\x8c'

c2 = bytes([255 - b for b in b2])

c1 < c2
True
```

But if the strings are of variable length, this will cause problems. For example:

```python
b1 = b'us'

b2 = b'usb'

b1 < b2
True

c1 = bytes([255 - b for b in b1])

c2 = bytes([255 - b for b in b2])

c1 > c2
False
```

`b1` has one less character than `b2`. After reversing by subtracting from 255, the leading `b'us'` parts are the same, but regardless of reversal, the shorter string is always considered smaller than the longer one.

Therefore, we need a max heap.

## 5\. Implementing a Max Heap

We can slightly modify Python's built-in `heapq` module to turn it into a max heap. We'll copy the source code of `heapq` and modify it into `maxheap_py.py`. The `heapq` module has many functions, but we only need to keep `heappush`, `heappop`, and `heapify`. These three functions, in turn, depend on `_siftdown` (which I've renamed `_siftdown_max`) and `_siftup` (renamed `_siftup_max`).

I won't go into the full code, but here's a comparison of `_siftdown_max` and `_siftdown`:

```python
def _siftdown_max(heap, startpos, pos):
    'Maxheap variant of _siftdown'
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem
```

```python
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem
```

The only difference is the single line: `"if parent < newitem"` versus `"if newitem < parent"`. This assumes that the `newitem` and `parent` values have an overloaded or implemented `__lt__` operator, so even when finding the larger element, it's done by swapping the order and using the `__lt__` function.

## 6\. Optimizing `maxheap_py` with a C Module

Readers who are not interested in C module development can skip this section.

If we compare the speed of our version to the CPython version, ours is much slower because the CPython version internally calls a corresponding C module. If we look closely at the `heapq` source code, we'll find:

```python
# If available, use C implementation
try:
    from _heapq import *
except ImportError:
    pass
try:
    from _heapq import _heapreplace_max
except ImportError:
    pass
try:
    from _heapq import _heapify_max
except ImportError:
    pass
try:
    from _heapq import _heappop_max
except ImportError:
    pass
```

This means it will (attempt to) call the corresponding C module implementation. The specific code is in `_heapqmodule.c`.

I've copied this to implement the max heap functionality. The full code is in `_maxheapqmodule.c`.

Let's look at just one function, `siftdown`:

```c
static int
siftdown(PyListObject *heap, Py_ssize_t startpos, Py_ssize_t pos)
{
    PyObject *newitem, *parent, **arr;
    Py_ssize_t parentpos, size;
    int cmp;

    assert(PyList_Check(heap));
    size = PyList_GET_SIZE(heap);
    if (pos >= size) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    /* Follow the path to the root, moving parents down until finding
       a place newitem fits. */
    arr = _PyList_ITEMS(heap);
    newitem = arr[pos];
    while (pos > startpos) {
        parentpos = (pos - 1) >> 1;
        parent = arr[parentpos];
        Py_INCREF(newitem);
        Py_INCREF(parent);
        cmp = PyObject_RichCompareBool(newitem, parent, Py_GT);
        Py_DECREF(parent);
        Py_DECREF(newitem);
        if (cmp < 0)
            return -1;
        if (size != PyList_GET_SIZE(heap)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "list changed size during iteration");
            return -1;
        }
        if (cmp == 0)
            break;
        arr = _PyList_ITEMS(heap);
        parent = arr[parentpos];
        newitem = arr[pos];
        arr[parentpos] = newitem;
        arr[pos] = parent;
        pos = parentpos;
    }
    return 0;
}
```

The two comparison codes have only a one-line difference:

```c
cmp = PyObject_RichCompareBool(newitem, parent, Py_GT);
cmp = PyObject_RichCompareBool(newitem, parent, Py_LT);
```

As mentioned before, `PyObject_RichCompareBool` compares the first two arguments based on the third. A return value less than 0 indicates an error; a return value greater than 0 means the third argument's comparison (e.g., greater than/less than) is true; and a return value of 0 means it's false.

Therefore, to change a min heap to a max heap, we just need to change the comparison operator from `Py_LT` (less than) to `Py_GT` (greater than).

Of course, there are other minor details to modify. For example, I've changed the module name to `maxheapqc`, so the corresponding code needs to be updated:

```c
static struct PyModuleDef _heapqmodule = {
    PyModuleDef_HEAD_INIT,
    "maxheapqc",
    module_doc,
    0,
    heapq_methods,
    heapq_slots,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_maxheapqc(void)
{
    return PyModuleDef_Init(&_heapqmodule);
}
```

To compile this C module, we need to modify `setup.py`:

```python
include_path = sysconfig.get_path('include')
internal_include_path = os.path.join(include_path, 'internal')

print(f"{internal_include_path=}")
project_root = os.path.dirname(os.path.abspath(__file__))

maxheapqc_module = Extension('cs336_basics.maxheapqc', sources=['cs336_basics/_maxheapqmodule.c'],
        extra_compile_args=[
            f'-I{internal_include_path}',
        ]
        )

setup(
    name='maxheapqc',
    version='1.0',
    description='maxheapqc',
    packages=['cs336_basics'],
    ext_modules=[maxheapqc_module]
)
```

When we compile `_maxheapqmodule.c`, it depends on:

```c
#include "Python.h"
#include "pycore_list.h" 
```

For `Python.h`, `setuptools` will include it for us. Its location is typically something like `/usr/local/include/python3.12/Python.h`. But our `_maxheapqmodule.c` also needs to operate on lists via the C API, which requires `pycore_list.h`. This file is located at a path like `/usr/local/include/python3.12/internal/pycore_list.h`. We don't want to hard-code this path, and the same system might have multiple Python versions and environments like Conda. So we can use `sysconfig.get_path('include')` to get the current Python interpreter's path and then find `internal` within it.

We need to add this path during compilation using the `-I` flag:

```python
maxheapqc_module = Extension('cs336_basics.maxheapqc', sources=['cs336_basics/_maxheapqmodule.c'],
        extra_compile_args=[
            f'-I{internal_include_path}',
        ]
        )
```

This `-I` parameter seems to be the way `gcc` adds header files. I'm not sure if it's universal for non-Linux systems or other compilers. If you are using a different compiler, please refer to its manual for adding appropriate header file paths.

Run the following command to compile the C module:

```bash
python setup.py build_ext -i
```

After compilation, you'll get a file like `cs336_basics/maxheapqc.cpython-312-x86_64-linux-gnu.so`. We can then write a `maxheapq.py` to call it.

The code for `maxheapq.py` is similar to `heapq.py`; it implements max heap functionality in Python and attempts to load `maxheapqc`:

```python
# If available, use C implementation
try:
    from cs336_basics.maxheapqc import *
    print("load c!")
except ImportError:
    pass
try:
    from cs336_basics.maxheapqc import _heapreplace_max
except ImportError:
    pass
try:
    from cs336_basics.maxheapqc import _heapify_max
except ImportError:
    pass
try:
    from cs336_basics.maxheapqc import _heappop_max
except ImportError:
    pass
```

## 7\. Using `heapq` to Implement a Max Heap

In fact, if we read the `heapq` source code, we'll find that it already implements `_heapify_max`, `_siftdown_max`, and `_siftup_max`. Among these, `_heapify_max` has a corresponding C module implementation and is very fast. However, `_siftdown_max` and `_siftup_max` are still implemented in Python. We can look [here](https://github.com/python/cpython/blob/3.12/Modules/_heapqmodule.c%23L540) to find which functions have C implementations:

```c
static PyMethodDef heapq_methods[] = {
    _HEAPQ_HEAPPUSH_METHODDEF
    _HEAPQ_HEAPPUSHPOP_METHODDEF
    _HEAPQ_HEAPPOP_METHODDEF
    _HEAPQ_HEAPREPLACE_METHODDEF
    _HEAPQ_HEAPIFY_METHODDEF
    _HEAPQ__HEAPPOP_MAX_METHODDEF
    _HEAPQ__HEAPIFY_MAX_METHODDEF
    _HEAPQ__HEAPREPLACE_MAX_METHODDEF
    {NULL, NULL}           /* sentinel */
};
```

We can use these three functions from `heapq` to implement a max heap. The code is in `maxheap_heapq.py`. Let's take a look:

```python
def heappush(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def heappop(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt

def heapify(heap):
    heapq._heapify_max(heap)
```

Here, `heapify` calls the C interface and is fast, but `heappush` and `heappop` are still Python implementations.

## 8\. Performance Testing of Different Max Heap Implementations

I've written a simple test script, `test_heap_speed.py`. The results are as follows:

```
maxheapq_py: 40.50480842590332
maxheap_heapq: 4.577162265777588
maxheapq: 4.334884405136108
```

The majority of the time here is spent on `heapify`. `maxheapq_py` is implemented in Python, so it's much slower.

And `test_heap_speed2.py` primarily tests `heappush` and `heappop`. The results are as follows:

```
maxheapq_py: 1.892250157892704
maxheap_heapq: 1.8967562650796026
maxheapq: 0.3524886401137337
```

As you can see, `maxheapq` is much faster than the Python implementations.

## 9\. Using a Max Heap to Find the Max

The complete code is in `bpe_v6.py`. Let's just look at the differences between it and `bpe_v5.py`.

```python
from cs336_basics import maxheap_py as maxheap

    def train(self, input_path, vocab_size, special_tokens, *args):
        ...
        pair_heap = []
        for pair, count in pair_counts.items():
            maxheap.heappush(pair_heap, (count, pair_strings[pair], pair))
            
            
            

```

First, we import `maxheap_py`, and for easy switching between different max heap implementations, I've aliased it as `maxheap`. Since all different max heaps have the same interface, switching to `maxheapq` or `maxheap_heapq` only requires changing this one line of code.

After the initial `pair_counts` are calculated, we need to construct a `pair_heap`. This is done by continuously calling `maxheap.heappush`. Another implementation method is to use `heapify`:

```python
        pair_heap = []
        for pair, count in pair_counts.items():
            pair_heap.append((count, pair_strings[pair], pair))
        maxheap.heapify(pair_heap)
```

Both methods will eventually build a heap, but their results may not be exactly the same. Theoretically, the `heapify` method is faster, but actual testing shows the time difference is minimal, as `pair_counts` starts with fewer than 20,000 entries.

Next is the main modification: finding the current maximum pair using `heappop`:

```python
    @staticmethod
    def _merge_a_pair(pair_counts, pair_strings, vocabulary, pair_to_words, 
                   word_counts, word_encodings, merges, size, pair_heap):
        
        while pair_heap:
            count, string_priority, merge_pair = maxheap.heappop(pair_heap)
            
            # check pair validity
            if merge_pair in pair_counts and pair_counts[merge_pair] == count:
                break
            elif merge_pair in pair_counts:
                # update count (lazily)
                maxheap.heappush(pair_heap, (pair_counts[merge_pair], 
                                               string_priority, 
                                               merge_pair))
        else:
            # no valid pairs found
            return False
```

The algorithm's implementation is as described earlier: first, pop the current maximum `merge_pair` from the top of the heap. Then, check if its count has been updated by comparing it to `pair_counts[merge_pair]`. If the count hasn't been updated, then `merge_pair` is the current maximum, and we break the loop. If it has been updated and the new count is greater than 0 (`merge_pair in pair_counts`), we re-insert the new count into the heap using `heappush`. We then continue the loop to find the maximum pair.

The final change is that whenever a new pair is created during a merge, we also need to add it to the heap:

```python
    @staticmethod
    def _updated_affected_word_count(merge_pair, affected_words, word_encodings, 
                                     word_counts, pair_counts, pair_to_words, 
                                     new_id, pair_strings, vocabulary, pair_heap):



        for new_pair in new_pairs:
            if new_pair not in pair_strings:
                pair_strings[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])

            maxheap.heappush(pair_heap, (pair_counts[new_pair], pair_strings[new_pair], new_pair))
```

## 10\. Testing

To test the time, I also implemented `bpe_v6_time.py`. The test results are as follows:

| Version | Data | Total Time (s) | Word Count Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v5\_time | openweb | 34333/34853/35804 | 401/390/401 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 | num\_counter=8, num\_merger=1 |
| bpe\_v6\_time | open\_web | 1036/1107/1046 | 395/395/398 | total: 576/641/591 max:6/7/6 update: 570/633/584 | num\_counter=8, num\_merger=1 |

After using a heap to find the max, the merge time decreased from over 30,000 seconds to just over 570 seconds. The `heappush` time is over 100 seconds. Can we optimize it with a faster max heap?

By changing `maxheap_py` to `maxheap_heapq` or `maxheapq`, we get `bpe_v7.py` and `bpe_v7_maxheapc.py`. The test results are as follows:

| Version | Data | Total Time (s) | Word Count Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v5\_time | openweb | 34333/34853/35804 | 401/390/401 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 | num\_counter=8, num\_merger=1 |
| bpe\_v6\_time | open\_web | 1036/1107/1046 | 395/395/398 | total: 576/641/591 max:6/7/6 update: 570/633/584 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_time | open\_web | 1062/1035/1036 | 392/397/395 | total: 606/573/577 max:6/6/6 update: 599/567/571 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_maxheapc\_time | open\_web | 1069/1017/1011 | 400/401/399 | total: 606/556/555 max: 3/3/3 update: 602/552/552 | num\_counter=8, num\_merger=1 |

Using a faster heap implementation did not speed things up. I speculate this is because the time for `heappush` accounts for a relatively small proportion of the total update time, so a minor change in its time doesn't have a large impact on the overall performance.

## 11\. Experiments with a Larger `vocab_size`

If we compare `bpe_v7` with the C++ version `bpe_train_updater_fine_grained_emhash8`:

| Version | Data | Total Time (sec) | Word Count Time (sec) | Merge Time (sec) | Other |
|---|---|---|---|---|---|
| bpe\_v7\_time | open\_web | 1028/1053/1045 | 395/397/393 | total: 575/589/590 max: 6/6/6 update: 569/583/583 make heap: 0.01/0.01 heap\_push\_time: 102/107/122 | num\_counter=8, num\_merger=1 |

| Program | Hash Function | Total Time (sec) | Update Time (sec) | Max Time (sec) | Other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60 | |

The total merge time for `bpe_v7` is around 600 seconds, while `bpe_train_updater_fine_grained_emhash8` is around 260 seconds. But what if we increase the number of merges? Since the vocabularies of modern large models are getting larger, let's test the results for `vocab_size` of 64,000 and 96,000:

| Version | Data | Total Time (s) | Word Count Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v7\_time | open\_web | 1028/1053/1045 | 395/397/393 | total: 575/589/590 max: 6/6/6 update: 569/583/583 make heap: 0.01/0.01 heap\_push\_time: 102/107/122 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_time2 | open\_web | 1111/1174/1100 | 393/420/403 | total: 655/686/639 max: 9/10/10 update: 645/675/628 make heap: 0.01/0.01/0.01 heap\_push\_time: 128/157/108 | vocab\_size=64000 |
| bpe\_v7\_time2 | open\_web | 1129/1130/1123 | 394/406/393 | total: 675/666/670 max: 13/12/12 update: 661/653/657 make heap: 0.01/0.01/0.01 heap\_push\_time: 143/152/120 | vocab\_size=96000 |

As you can see, because the later parts deal with low-frequency words and pairs, the time barely changes. Now let's look at `bpe_train_updater_fine_grained_emhash8`:

| Program | Hash Function | Total Time (sec) | Update Time (sec) | Max Time (sec) | Other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60 | |
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 413/414/406 | 233/227/223 | 179/187/182 | 64k |
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 664/593/606 | 305/255/269 | 358/338/337 | 96k |

The time for `bpe_train_updater_fine_grained_emhash8` gradually increases. This is because as the number of pairs in `pair_counts` grows, the time to find the max will only increase, not decrease.

## 12\. Porting the Max Heap to C++

We can also port this algorithm to C++. The C++ standard library's `<algorithm>` header has functions like `std::make_heap`, `std::push_heap`, and `std::pop_heap`, which can provide the same functionality as Python. The C++ heap is a max heap by default, so we just need to overload `operator<`. We can define:

```cpp
struct HeapItem{
    int count;
    std::vector<std::vector<int>> pair_string;
    std::pair<int,int> pair;

    bool operator<(const HeapItem& other) const;
};
```

You can refer to `test_max_heap.cpp` for how to use the standard library.

I didn't use the standard library here; instead, I rewrote the Python version in C++ and tested its speed. It seems to be slightly faster than the standard library. The full code for the max heap is in `max_heap.cpp`.

Based on `bpe_train_updater_fine_grained`, I've implemented `bpe_train_updater_fine_grained_heap.cpp`. Based on `bpe_train_updater_fine_grained_emhash8_set`, I've implemented `bpe_train_updater_fine_grained_heap_emhash8_set.cpp`. Based on `bpe_train_updater_fine_grained_emhash8_set9`, I've implemented `bpe_train_updater_fine_grained_heap_emhash8_set9.cpp`.

The implementation logic is completely consistent with the Python version. Readers who are interested can read the code themselves. Below are the comparison test results:

| Program | Hash Function | Total Time (sec) | Update Time (sec) | Max Time (sec) | Other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60 | |
| bpe\_train\_updater\_fine\_grained\_emhash8\_set | Boost Hash | 192/192/194 | 117/117/117 | 75/75/77 | |
| bpe\_train\_updater\_fine\_grained\_emhash8\_set9 | Boost Hash | 168/170/171 | 107/108/109 | 61/62/61 | |
| bpe\_train\_updater\_fine\_grained\_heap | Boost Hash | 200/228/211 | 194/220/208 |2/3/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8 | Boost Hash | 199/200/210 | 193/195/203 | 2/2/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set | Boost Hash | 139/122/128 | 136/118/123 | 2/2/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set9 | Boost Hash | 111/121/122 | 109/116/116 | 2/2/2 | |

As you can see, the time to find the max using a heap is less than 2 seconds. Now let's look at its performance with vocabularies of 64,000 and 96,000:

| Program | Hash Function | Total Time (sec) | Update Time (sec) | Max Time (sec) | Other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60 | |
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 413/414/406 | 233/227/223 | 179/187/182 | 64k |
| bpe\_train\_updater\_fine\_grained\_emhash8 | Boost Hash | 664/593/606 | 305/255/269 | 358/338/337 | 96k |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8 | Boost Hash | 199/200/210 | 193/195/203 |2/2/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8 | Boost Hash | 224/228/227 | 216/218/217 | 3/3/3 | 64k |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8 | Boost Hash | 249/255/240 | 239/238/230 | 5/6/4 | 96k |

By comparing them, we can see that for the `bpe_train_updater_fine_grained_heap_emhash8` algorithm, the update time only increases by about 20 seconds when the vocabulary increases from 32k to 64k, and the max time is almost unchanged. In contrast, the max time for `bpe_train_updater_fine_grained_emhash8` doubles.

## 13\. Optimizing `_updated_affected_word_count`

If you carefully read the `_updated_affected_word_count` function and `fine_grained_pair_counter_diff`, you can see there's redundancy between `new_pairs` and `diff_pairs`. Recall that `diff_pairs` records the change in a pair's count: a value greater than zero indicates an increase (new pair), and a value less than zero indicates a decrease (old pair). `new_pairs` represents all affected pairs (including pairs that may not have changed, which happens when the `merge_pair` appears multiple times with other pairs in between). However, based on our previous algorithm, we only need to add new pairs to the heap with `heappush`, and old pairs can be updated lazily. So, we can remove `new_pairs`, and any pair in `diff_pairs` with a count \> 0 must be a new pair.

This leads to `bpe_v7_maxheapc_opt_time.py`. Other versions can be modified similarly.

The main code changes are:

```python
    @staticmethod
    def _updated_affected_word_count(merge_pair, affected_words, word_encodings, 
                                     word_counts, pair_counts, pair_to_words, 
                                     new_id, pair_strings, vocabulary, pair_heap):
        # we may update/delete words when iterate it.
        affected_words = affected_words.copy()
        diff_pairs = defaultdict(int)

        BPE_Trainer.fine_grained_pair_counter_diff(affected_words, word_encodings, word_counts, merge_pair, diff_pairs, 
                             new_id, pair_to_words)
        for pair, count in diff_pairs.items():
            if count == 0: continue
            pair_counts[pair] += count
            if count > 0: # new pair
                pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])
                maxheap.heappush(pair_heap, (pair_counts[pair], pair_strings[pair], pair))
            
            if pair_counts[pair] <= 0: # should not less than 0!
                del pair_counts[pair]
                pair_to_words.pop(pair, None)
```

`BPE_Trainer.fine_grained_pair_counter_diff` no longer needs to take `new_pairs` as an output parameter. When iterating through `diff_pairs`, we check the count; if `count > 0`, it's a new pair, and we save its `pair_strings` and add it to the heap with `heappush`.

| Version | Data | Total Time (s) | Word Count Time (s) | Merge Time (s) | Other |
|---|---|---|---|---|---|
| bpe\_v5\_time | openweb | 34333/34853/35804 | 401/390/401 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 | num\_counter=8, num\_merger=1 |
| bpe\_v6\_time | open\_web | 1036/1107/1046 | 395/395/398 | total: 576/641/591 max:6/7/6 update: 570/633/584 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_time | open\_web | 1062/1035/1036 | 392/397/395 | total: 606/573/577 max:6/6/6 update: 599/567/571 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_maxheapc\_time | open\_web | 1069/1017/1011 | 400/401/399 | total: 606/556/555 max: 3/3/3 update: 602/552/552 | num\_counter=8, num\_merger=1 |
| bpe\_v7\_maxheapc\_opt\_time | open\_web | 984/965/1000 | 394/394/403 | total: 532/514/538 max: 0.8/0.8/0.9 update: 531/513/537 | num\_counter=8, num\_merger=1 |

`bpe_v7_maxheapc_opt_time` is about 7% faster than `bpe_v7_maxheapc_time`.

## 14\. Porting the Optimization to C++

The optimized versions are `bpe_train_updater_fine_grained_heap_opt.cpp`, `bpe_train_updater_fine_grained_heap_emhash8_opt.cpp`, `bpe_train_updater_fine_grained_heap_emhash8_set_opt.cpp`, and `bpe_train_updater_fine_grained_heap_emhash8_set9_opt.cpp`.

Test results on the OpenWeb dataset:

| Program | Hash Function | Total Time (sec) | Update Time (sec) | Max Time (sec) | Other |
|---|---|---|---|---|---|
| bpe\_train\_updater\_fine\_grained\_heap | Boost Hash | 200/228/211 | 194/220/208 |2/3/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_opt | Boost Hash | 190/192/190 | 187/189/187 |0/0/0 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8 | Boost Hash | 199/200/210 | 193/195/203 | 2/2/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_opt | Boost Hash | 188/213/192 | 185/209/189 | 0/0/0 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set | Boost Hash | 139/122/128 | 136/118/123 | 2/2/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set\_opt | Boost Hash | 110/130/110 | 108/128/108 | 0/0/0 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set9 | Boost Hash | 111/121/122 | 109/116/116 | 2/2/2 | |
| bpe\_train\_updater\_fine\_grained\_heap\_emhash8\_set9\_opt | Boost Hash | 105/102/104 | 102/100/101 | 0/0/0 | |

`bpe_train_updater_fine_grained_heap_emhash8_set9_opt` is 11% faster in update time than `bpe_train_updater_fine_grained_heap_emhash8_set9`. `bpe_train_updater_fine_grained_heap_emhash8_set_opt` is 8% faster than `bpe_train_updater_fine_grained_heap_emhash8_set`. This conclusion is consistent with the Python version.


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
