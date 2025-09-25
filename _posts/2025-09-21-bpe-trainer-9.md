---
layout:     post
title:      "动手实现和优化BPE Tokenizer的训练——第9部分：使用堆来寻找最大pair" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - cs336
    - bpe tokenizer
---

本系列文章完成Stanford CS336作业1的一个子任务——实现BPE Tokenizer的高效训练算法。通过一系列优化，我们的算法在OpenWebText上的训练时间从最初的10多个小时优化到小于10分钟。本系列文章解释这一系列优化过程，包括：算法的优化，数据结构的优化，并行(openmp)优化，cython优化，用c++实现关键代码和c++库的cython集成等内容。本文是第十篇，使用堆(heap)这个数据结构来替代求最大pair，提升性能。

<!--more-->


**目录**
* TOC
{:toc}


## 1. 问题分析

在上面我们的bpe_v5_time中,update time只有500多秒，但是max的时间是30000多秒。怎么才能优化这个时间呢？之前我们尝试的办法是用并行算法求max，Python的版本并不成功，因为Python的GIL使得多线程无法同时使用多个CPU，而多进程又存在大量的通信。我们又尝试了在c++里用OpenMP实现了并行，在32线程的时候max的时间降低到400秒。接着我们又用更快的flat hashmap替代了std::unordered_map，使得max的时间在不用并行算法就能把max时间降低到100秒之内。

不过我们还是回到Python，除了优化数据结构，我们还有没有办法能够优化算法呢？优化数据结构就是让同样的工作做得更快，而优化算法是用更少的工作达到同样的目的。当然这两者也不是完全分开的，有的时候为了实现某种不同的算法就需要设计合适的数据结构来完成。


为了找到一个集合中的最大值，我们当然需要遍历整个集合，这个时间不可能更少。也就是说第一次遍历整个集合的时间是不可能变少的。但是第二次遍历时我们就有可能优化了，因为两次遍历之间只有部分pair的计数是发生变化的。刚开始都是高频词，因此affected_words很多，所以受到影响的pair相对较多。而到了后面，词频没有那么高了，affected_words就少了。最简单的想法就是我们可以给pair_counts排序，然后某些pair的计数发生了变化，那么就可以只对这些pair重新排序。因为我们的词频发生变化只有两种可能：老的pair词频减少，新增的pair词频增加。所以我们还可以用一些启发式规则，比如某个pair当前的序号是大于32000(我们最终需要的词表大小)，并且它的频次变低了，那么我们可以忽略它。

不过我们还有更好的方法来找到一个集合的最大值，那就是使用堆这种数据结构。

## 2. 堆(heap)

在计算机领域，堆有两个完全不同的意思：一是表示一种数据结构；二是一种内存分配方式。有趣的是在表示内存分配方式时和堆对应的是栈(stack)，而栈除了表示内存分配方式之外也表示一种数据结构。不过作为数据结构的栈和内存分配的栈还是存在紧密联系的，比如数据结构的栈是一种后进先出的数据结构(和先入先出的队列对应)，而内存分配的栈(函数)正是利用了这种后进先出的特点，使得函数的调用和返回顺序符合我们的期望。但是数据结构的堆和内存分配的堆就没有什么联系了，它们唯一的联系就是当初起名字的人不知怎的偶然用了相同的词表示了完全不同两个概念。

我们这里关注的是作为数据结构的堆，它通常用于实现优先队列，也可以用来实现堆排序。我这里不再详细介绍堆，如果不熟悉的读者随便找一本数据结构和算法的书，或者上网搜索都有很多介绍，比如[wiki](https://en.wikipedia.org/wiki/Heap_(data_structure))。

## 3. 使用堆求max

一般使用堆的时候我们只是使用如下3个操作：把一个数组变成一个堆(heapify)、从堆顶弹出一个元素(heappop)和把一个元素加入堆(heappush)。我们首先需要通过heapify把一个数组变成堆(满足堆的定义：树根比它的子树中的每一个都大/小)，然后不断的调用heappop/heappush，这两个操作完成之后的数组依然还是堆。


这里有一个问题，那就是每次找完最大之后(heappop)，我们会合并pair，这会使得某些老的pair的计数减少，同时一些新的pair出现(原来没有)。增加新的pair没有问题，我们只需要调用heappush就可以了。但是怎么修改老的pair的计数并且使得修改之后还是堆呢？我们来看一个例子：


```
         11
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   / \
6  4  5   7
```
如果我们把8改成10，也就是一个元素变大了，那么需要从8开始往上调用siftup：

```
         11
       /    \
      /      \
     10       4
   /   \     / \
  7     9   3   1 
 / \   / \
6  4  5   7
```

而反正如果我们把8改成了6，那么需要从8开始往下调用siftdown：

```
         11
       /    \
      /      \
     10       4
   /   \     / \
  7     7   3   1 
 / \   / \
6  4  5   6
```

但是这里有一个问题，那就是我们怎么找到这个修改的元素。请读者回顾一下，我们的主要数据是pair_counts，这是一个dict，但是堆需要操作的是一个数组(list)，因此我们需要把pair_counts中的元素复制一份到pair_heap这个list里。但是list无法支持快速查找，如果为了修改某个元素就需要顺序扫描一遍list，那就得不偿失了(max都求出来了)。

一种解决办法是在pair_counts里记录它在list里的下标，这样pair_counts变成：

```
pair -> (count, index_in_pair_heap)
```

那么新增一个pair到pair_counts的同时也把这pair通过heappush添加到pair_heap的合适位置，这就要求heappush函数不但可以把一个元素添加到pair_heap数组，而且还需要返回它在pair_heap的下标。这样我们才能把这个下标保存到pair_counts里。等到后面某个pair的计数发生改变(我们这里只会减少)，我们就可以通过pair_counts找到它在pair_heap的位置，然后对这个位置的元素调用siftdown。

这就需要修改heappush，而且还需要维护pair_counts和pair_heap的关系，这会使得代码变得很复杂。感兴趣的读者可以尝试实现一下这种算法。

不过我这里使用的是另外一种方法——lazy的修改。这种方法当某个pair的计数发生改变时，我们什么也不做。只有当我们调用heappop时需要检查一下它的计数是否修改了，这可以通过和pair_counts里的对比来发现。如果修改了(只会变小，这个假设非常重要)，那么我们把新的计数重新通过heappush加进去。然后不断的heappop，直到某个元素没有被修改，那么它就是当前的最大值。

说起来比较复杂，我通过一个例子来看。比如当前的堆为：
```
         11
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   / \
6  4  5   7
```

最大的应该是11，但是我们假设因为合并11变成了8，那么当我们弹出11这个pair时，我们查询pair_counts得知它的最新计数是10，因为它变小了，所以我们不能确定它是否最大。所以我们首先把11弹出，得到：

```
          9
       /    \
      /      \
     8        4
   /   \     / \
  7     7   3   1 
 / \   /
6  4  5 
```

然后还需要把10重新heappush进去，得到：
```
         10
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   /  \
6  4  5    7
```

接着，我们再次弹出当前最大10，这个时候查询pair_counts得知它的计数是最新的，因此就找到了当前的最大pair是10。

**注意**：我们可以lazy更新的一个重要假设是老pair的计数只会变小。如果这个假设不成立，比如当前我们的堆是：

```
         11
       /    \
      /      \
     9        4
   /   \     / \
  7     8   3   1 
 / \   / \
6  4  5   7
```
我们把1变成了12，那么我们就必须马上对1进行siftup，否则我们找到的最大是11，但这是不正确的。

使用这种算法，我们不需要在pair_counts里维护pair_heap的下标，而且我们也不需要调用siftdown，这在Python的heapq里是一个私有函数_siftdown，使用私有函数是有风险的，也许新的版本就没有这个函数了。

## 4. Python的heapq模块

Python的标准库提供了[heapq](https://docs.python.org/3/library/heapq.html)。我们需要用到的主要函数是heappush、heappop和heapify。不熟悉的读者可以参考[The Python heapq Module: Using Heaps and Priority Queues](https://realpython.com/python-heapq-module/)。

不过这里有一个问题，我们需要的是一个大堆(max heap)，但是Python的heapq模块提供的是小堆(min heap)。我们后面分析其代码时会发现它的内部已经实现了大堆。不过这里我们需要讨论如果只有小堆我们能不能用它来实现大堆的功能。

一种常见的技巧是把元素逆转来实现大堆。比如如果堆的元素是正整数，那么我们可以存入它对应的负整数，这样就可以得到大堆了，比如下面的例子：

```
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

但是现在我们需要放入堆的是(count, pair_string[pair], pair)。如果count大，那么这个tuple就大；如果count相同，我们再比较这个pair的字符串，选择大的那个。tuple最后一个放入pair是为了使用方便。

如果按照上面的办法，count可以放入-count。pair_strings[pair]呢？pair_strings[pair]里是个tuple，这个tuple的每个元素都是bytes。如果是定长的bytes，因为一个byte的范围是0~255，那么我们可以用255减去这个数来逆转它。比如：

```
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

但是如果字符串不定长，就会出问题，比如：

```
b1 = b'us'

b2 = b'usb'

b1 < b2
True

c1 = bytes([255 - b for b in b1])

c2 = bytes([255 - b for b in b2])

c1 > c2
False
```

b1比b2少一个字符串。用255减去逆转之后前面的b'us'是相同的，但是不管是否逆转，位数少的都是小于位数多的。

所以我们需要一个大堆。

## 5. 实现一个大堆

我们只需要稍微修改一下Python自带的heapq模块就可以把它改成大堆，所以我们把heapq的源代码复制一遍然后修改成[maxheap_py.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/maxheap_py.py)。heapq的函数很多，我们只需要保留heappush、heappop和heapify。另外这3个函数又依赖_siftdown(我把它改名为_siftdown_max)和_siftup(改名_siftup_max)。

完整代码我就不展开，这里就对比一下_siftdown_max和_siftdown：

```
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

```
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

它们唯一的区别就是"if newitem < parent"和"if parent < newitem"这两行代码。这里假设newitem和parent值实现/重载了小于运算符，所以即使想求较大的元素也是通过调换顺序通过小于函数来实现的。

## 6. 用c模块优化maxheap_py

对于c模块开发不感兴趣的读者可以跳过本节内容。

但是如果我们比较自己版本速度，它会把CPython的版本慢很多，原因在于CPython的版本内部是调用了对应的c模块。如果我们仔细阅读[heap源代码](https://github.com/python/cpython/blob/3.12/Lib/heapq.py)，我们会发现：

```
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

也就是它会(尝试)调用对应的c模块实现。具体的代码在[_heapqmodule.c](https://github.com/python/cpython/blob/3.12/Modules/_heapqmodule.c)里。

我复制它实现了大堆的功能，完整代码在[_maxheapqmodule.c](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/_maxheapqmodule.c)。

我们只看其中一个函数siftdown：

```
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

两种对比代码只有一行区别：

```
cmp = PyObject_RichCompareBool(newitem, parent, Py_GT);
cmp = PyObject_RichCompareBool(newitem, parent, Py_LT);
```

前面介绍过PyObject_RichCompareBool，它就是根据第三个参数比较前两个参数。如果返回值小于0，表示异常；如果返回值大于0，则表示第三个参数，第一个比第二个大/小是true；如果返回值是0，则表示第一个参数比第二个参数大/小是false。

因此我们把小堆改成大堆只需要把比较运算符从Py_LT(小于)变成Py_GT(大于)就行了。

当然，其它还有一些细节需要修改，比如我把模块名字改成了maxheapqc，那么就需要修改相应的代码：

```
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

另外为了编译这个C模块，需要修改setup.py：

```
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

我们在编译_maxheapqmodule.c时，需要依赖：

```
#include "Python.h"
#include "pycore_list.h" 
```

对于Python.h，setuptools会帮我们包含这个头文件，它的位置通常类似于：/usr/local/include/python3.12/Python.h。但是我们的_maxheapqmodule.c还需要通过c接口操作list，这需要pycore_list.h，它的位置在类似于/usr/local/include/python3.12/internal/pycore_list.h的地方。我们不想硬编码这个位置，而且同一个系统里可能安装了很多版本的python，还有很多conda这样的环境。所以我们可以通过sysconfig.get_path('include')得到当前Python解释器的路径，然后再在里面找internal。

我们在编译时需要通过-I添加这个路径：

```
maxheapqc_module = Extension('cs336_basics.maxheapqc', sources=['cs336_basics/_maxheapqmodule.c'],
        extra_compile_args=[
            f'-I{internal_include_path}',
        ]
        )
```

这个-I参数好像是gcc添加头文件的方式，我不知道在非linux系统非gcc编译器下能否通用，如果读者使用的是其它编译器，请参考编译器的手册添加合适的头文件路径。

运行下面的命令编译c模块：

```
python setup.py build_ext -i
```

编译后可以得到类似cs336_basics/maxheapqc.cpython-312-x86_64-linux-gnu.so。接下来我们就可以写一个[maxheapq.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/maxheapq.py)来调用它了。

maxheapq.py的代码和heapq.py类似，它用Python实现了大堆的功能，并且会尝试加载maxheapqc：

```
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

## 7. 用heapq实现大堆

其实如果我们阅读了heapq的代码，我们会发现它已经实现了_heapify_max、_siftdown_max和_siftup_max。而其中_heapify_max是有对应的c模块代码，速度比较快。但是_siftdown_max和_siftup_max还是用Python实现的。我们可以查看[这里](https://github.com/python/cpython/blob/3.12/Modules/_heapqmodule.c#L540)来找到哪些函数有c语言实现：

```
static PyMethodDef heapq_methods[] = {
    _HEAPQ_HEAPPUSH_METHODDEF
    _HEAPQ_HEAPPUSHPOP_METHODDEF
    _HEAPQ_HEAPPOP_METHODDEF
    _HEAPQ_HEAPREPLACE_METHODDEF
    _HEAPQ_HEAPIFY_METHODDEF
    _HEAPQ__HEAPPOP_MAX_METHODDEF
    _HEAPQ__HEAPIFY_MAX_METHODDEF
    _HEAPQ__HEAPREPLACE_MAX_METHODDEF
    {NULL, NULL}           /* sentinel */
};
```

我们可以利用heapq的这3个函数来实现大堆。代码在[maxheap_heapq.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/maxheap_heapq.py)。我们来看一下：

```
def heappush(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def heappop(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt

def heapify(heap):
    heapq._heapify_max(heap)
```

这里heapify会调用c接口，速度比较快，但是heappush和heappop还是Python实现。

## 8. 不同的大堆实现的性能测试

我写了一个简单的测试代码[test_heap_speed.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_heap_speed.py)。结果如下：

```
maxheapq_py: 40.50480842590332
maxheap_heapq: 4.577162265777588
maxheapq: 4.334884405136108
```

这里主要的时间用在heapify，maxheapq_py是Python实现，所以比较慢。

而[test_heap_speed2.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/test_heap_speed2.py)主要测试heappush和heappop，结果如下：

```
maxheapq_py: 1.892250157892704
maxheap_heapq: 1.8967562650796026
maxheapq: 0.3524886401137337
```

可以看到maxheapq要比Python的实现快。

## 9. 用大堆求max

完整的代码在[bpe_v6.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v6.py)。我们只看一下它和bep_v5.py的区别。

```
from cs336_basics import maxheap_py as maxheap

    def train(self, input_path, vocab_size, special_tokens, *args):
        ...
        pair_heap = []
        for pair, count in pair_counts.items():
            maxheap.heappush(pair_heap, (count, pair_strings[pair], pair))
            
            
            

```

首先需要import maxheap_py，为了便于切换不同的大堆实现，我把它重命名为maxheap。因为所有不同大堆的接口都相同，所以要切换为maxheapq或者maxheap_heapq只需要修改这一行代码就可以了。

然后在第一次统计了pair_counts之后，我们需要构造一个pair_heap，这里是不停调用maxheap.heappush。另外一种实现方法是使用heapify：

```
        pair_heap = []
        for pair, count in pair_counts.items():
            pair_heap.append((count, pair_strings[pair], pair))
        maxheap.heapify(pair_heap)
```

两种方法最终都会构建出一个堆，但是它们的结果可能不完全相同。理论上用heapify的方法更快一点，不过实际测试发现它们的时间差很小，因为pair_counts一开始不到两万。

接下来是主要的修改之处，通过heappop求当前最大的pair：

```
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

这个算法的实现和前面描述的一样：首先从堆顶弹出当前最大的merge_pair，然后通过对比pair_counts[merge_pair]来检查它的count有没有更新，如果没有更新，那么merge_pair就是当前的最大pair，break出循环。如果有了更新而且新的count>0(merge_pair in pair_counts)，则需要把新的count重新插入到堆里。然后再循环去找最大的pair。

最后一处修改就是每次merge产生新的pair时，我们需要把它也加到堆里：

```
    @staticmethod
    def _updated_affected_word_count(merge_pair, affected_words, word_encodings, 
                                     word_counts, pair_counts, pair_to_words, 
                                     new_id, pair_strings, vocabulary, pair_heap):



        for new_pair in new_pairs:
            if new_pair not in pair_strings:
                pair_strings[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])

            maxheap.heappush(pair_heap, (pair_counts[new_pair], pair_strings[new_pair], new_pair))
```

## 10. 测试

为了测试时间，我也实现了[bpe_v6_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v6_time.py)。测试结果如下：

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v5_time | openweb |34333/34853/35804 | 401/390/401 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 |num_counter=8, num_merger=1
bpe_v6_time | open_web | 1036/1107/1046 | 395/395/398 | total: 576/641/591 max:6/7/6 update: 570/633/584 | num_counter=8, num_merger=1


使用堆来求最大之后，合并时间从30000多秒减少到了570多秒。heappush的时间是100多秒，我们还能通过其它更快的大堆来优化它吗？

把maxheap_py改成maxheap_heapq或者maxheapq就得到[bpe_v7.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v7.py)和[bpe_v7_maxheapc.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v7_maxheapc.py)，测试结果如下：

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v5_time | openweb |34333/34853/35804 | 401/390/401 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 |num_counter=8, num_merger=1
bpe_v6_time | open_web | 1036/1107/1046 | 395/395/398 | total: 576/641/591 max:6/7/6 update: 570/633/584 | num_counter=8, num_merger=1
bpe_v7_time | open_web | 1062/1035/1036 | 392/397/395 | total: 606/573/577 max:6/6/6 update: 599/567/571 | num_counter=8, num_merger=1
bpe_v7_maxheapc_time | open_web | 1069/1017/1011 | 400/401/399 | total: 606/556/555 max: 3/3/3 update: 602/552/552 | num_counter=8, num_merger=1

使用更快的堆实现并没有加快速度。我猜测是因为heappush的时间占总update时间的比例比较低，因此它的微小时间变化对于整体影响不大。

## 11. 更大vocab_size的实验

如果我们拿bpe_v7和c++版本的bpe_train_updater_fine_grained_emhash8相比：

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v7_time | open_web | 1028/1053/1045 | 395/397/393 | total: 575/589/590 max: 6/6/6 update: 569/583/583 make heap: 0.01/0.01 heap_push_time: 102/107/122 | num_counter=8, num_merger=1

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60

bpe_v7的merge总时间是600秒左右，而bpe_train_updater_fine_grained_emhash8的总时间是260秒左右。但是如果我们把合并的次数增多呢？因为现在的大模型的词典越来越大，所以我们测试一下vocab_size是64000和96000的结果：

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v7_time | open_web | 1028/1053/1045 | 395/397/393 | total: 575/589/590 max: 6/6/6 update: 569/583/583 make heap: 0.01/0.01 heap_push_time: 102/107/122 | num_counter=8, num_merger=1
bpe_v7_time2 | open_web | 1111/1174/1100 | 393/420/403 | total: 655/686/639 max: 9/10/10 update: 645/675/628 make heap: 0.01/0.01/0.01 heap_push_time: 128/157/108 | vocab_size=64000
bpe_v7_time2 | open_web | 1129/1130/1123 | 394/406/393 | total: 675/666/670 max: 13/12/12 update: 661/653/657 make heap: 0.01/0.01/0.01 heap_push_time: 143/152/120 | vocab_size=96000

可以看到由于后面处理的都是低频词和低频的pair，所以几乎时间没有什么变化。我们再看bpe_train_updater_fine_grained_emhash8：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 413/414/406 | 233/227/223 | 179/187/182 | 64k
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 664/593/606 | 305/255/269 | 358/338/337 | 96k

可以看到bpe_train_updater_fine_grained_emhash8的时间逐渐增大，这是因为随着pair_counts中pair的增多，max的时间只会增加不会减少。

## 12. 把最大堆移植到c++

我们也可以把这个算法移植到c++，c++的标准库的<algorithm>里有std::make_heap、std::push_heap和std::pop_heap等函数，可以实现和python相同的功能，而且c++的堆是大堆，我们只需要实现operator<的重载即可，我们可以定义：

```
struct HeapItem{
    int count;
    std::vector<std::vector<int>> pair_string;
    std::pair<int,int> pair;

    bool operator<(const HeapItem& other) const;
};
```

标准库的用法可以参考[test_max_heap.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/test_max_heap.cpp)。

我这里没有使用标准库，而是把Python版本的用c++重写了，测试了一下速度，似乎比标准库的还要稍微快一点点。大堆的完整代码在[max_heap.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/max_heap.cpp)。

基于bpe_train_updater_fine_grained实现了[bpe_train_updater_fine_grained_heap.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap.cpp)，基于bpe_train_updater_fine_grained_emhash8_set实现了[bpe_train_updater_fine_grained_heap_emhash8_set.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_emhash8_set.cpp)，基于bpe_train_updater_fine_grained_emhash8_set9实现了[bpe_train_updater_fine_grained_heap_emhash8_set9.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_emhash8_set9.cpp)。

代码实现逻辑和Python版本完全一致，感兴趣的读者可以自行阅读。下面是对比实验结果：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60
bpe_train_updater_fine_grained_emhash8_set | Boost Hash | 192/192/194 | 117/117/117 | 75/75/77
bpe_train_updater_fine_grained_emhash8_set9 | Boost Hash | 168/170/171 | 107/108/109 | 61/62/61
bpe_train_updater_fine_grained_heap | Boost Hash | 200/228/211 | 194/220/208 |2/3/2
bpe_train_updater_fine_grained_heap_emhash8 | Boost Hash | 199/200/210 | 193/195/203 | 2/2/2
bpe_train_updater_fine_grained_heap_emhash8_set | Boost Hash | 139/122/128 | 136/118/123 | 2/2/2
bpe_train_updater_fine_grained_heap_emhash8_set9 | Boost Hash | 111/121/122 | 109/116/116 | 2/2/2

可以看到，使用堆来求max的时间不到2秒。我们再来看一下它在合并词典64000和96000时的结果：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 261/259/261 | 200/198/200 | 61/60/60
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 413/414/406 | 233/227/223 | 179/187/182 | 64k
bpe_train_updater_fine_grained_emhash8 | Boost Hash | 664/593/606 | 305/255/269 | 358/338/337 | 96k
bpe_train_updater_fine_grained_heap_emhash8 | Boost Hash | 199/200/210 | 193/195/203 |2/2/2
bpe_train_updater_fine_grained_heap_emhash8 | Boost Hash | 224/228/227 | 216/218/217 | 3/3/3 | 64k
bpe_train_updater_fine_grained_heap_emhash8 | Boost Hash | 249/255/240 | 239/238/230 | 5/6/4 | 96k

通过对比我们可以发现，使用堆的bpe_train_updater_fine_grained_heap_emhash8算法词典从32k增加到64k时update时间值增加了20秒左右，max时间几乎没变；而bpe_train_updater_fine_grained_emhash8的max时间增加了一倍。

## 13. 优化_updated_affected_word_count

如果仔细阅读_updated_affected_word_count函数和fine_grained_pair_counter_diff，可以发现其中new_pairs和diff_pairs存在冗余。回顾一下，diff_pairs记录了pair的count的变化，如果大于零表示增加(新pair)；如果小于零表示减少(老pair)。而new_pairs表示受到影响的所有pair(还包括可能没有变化的pair，这种情况发生在merge_pair出现多次时夹在中间的那些没有变化的pair)。但是根据之前的算法，我们只需要把新pair通过heappush加入堆就行，老的pair可以lazy的方式更新。所以我们可以去掉new_pairs，而diff_pairs里count>0的pair就一定是新pair。

这样就得到[bpe_v7_maxheapc_opt_time.py](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cs336_basics/bpe_v7_maxheapc_opt_time.py)，其它版本也可以参考这个修改。

主要改动的代码是：

```
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

BPE_Trainer.fine_grained_pair_counter_diff不再需要传入new_pairs这个出参。在遍历diff_pairs时我们判断一下count，如果count>0则说明是一个新加入的pair，我们保存它的pair_strings然后通过heappush加入堆中。

版本         | 数据      | 总时间(s) | 统计词频时间(s) | 合并时间(s) | 其它
bpe_v5_time | openweb |34333/34853/35804 | 401/390/401 | total:33879/34401/35347 max:33353/33820/34816 update:525/579/530 |num_counter=8, num_merger=1
bpe_v6_time | open_web | 1036/1107/1046 | 395/395/398 | total: 576/641/591 max:6/7/6 update: 570/633/584 | num_counter=8, num_merger=1
bpe_v7_time | open_web | 1062/1035/1036 | 392/397/395 | total: 606/573/577 max:6/6/6 update: 599/567/571 | num_counter=8, num_merger=1
bpe_v7_maxheapc_time | open_web | 1069/1017/1011 | 400/401/399 | total: 606/556/555 max: 3/3/3 update: 602/552/552 | num_counter=8, num_merger=1
bpe_v7_maxheapc_opt_time | open_web | 984/965/1000 | 394/394/403 | total: 532/514/538 max: 0.8/0.8/0.9 update: 531/513/537 | num_counter=8, num_merger=1

bpe_v7_maxheapc_opt_time比bpe_v7_maxheapc_time要快7%。

## 14. 把上面的优化移植到c++

优化后的版本是[bpe_train_updater_fine_grained_heap_opt.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_opt.cpp)、[bpe_train_updater_fine_grained_heap_emhash8_opt.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_emhash8_opt.cpp)、[bpe_train_updater_fine_grained_heap_emhash8_set_opt.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_emhash8_set_opt.cpp)和[bpe_train_updater_fine_grained_heap_emhash8_set9_opt.cpp](https://github.com/fancyerii/assignment1-basics-bpe/blob/main/cppupdate/bpe_train_updater_fine_grained_heap_emhash8_set9_opt.cpp)。

在OpenWeb数据集上的测试结果为：

program              | hash function | total time(sec) | update time(sec) | max time(sec) | other
bpe_train_updater_fine_grained_heap | Boost Hash | 200/228/211 | 194/220/208 |2/3/2
bpe_train_updater_fine_grained_heap_opt | Boost Hash | 190/192/190 | 187/189/187 |0/0/0
bpe_train_updater_fine_grained_heap_emhash8 | Boost Hash | 199/200/210 | 193/195/203 | 2/2/2
bpe_train_updater_fine_grained_heap_emhash8_opt | Boost Hash | 188/213/192 | 185/209/189 | 0/0/0
bpe_train_updater_fine_grained_heap_emhash8_set | Boost Hash | 139/122/128 | 136/118/123 | 2/2/2
bpe_train_updater_fine_grained_heap_emhash8_set_opt | Boost Hash | 110/130/110 | 108/128/108 | 0/0/0
bpe_train_updater_fine_grained_heap_emhash8_set9 | Boost Hash | 111/121/122 | 109/116/116 | 2/2/2
bpe_train_updater_fine_grained_heap_emhash8_set9_opt | Boost Hash | 105/102/104 | 102/100/101 | 0/0/0


bpe_train_updater_fine_grained_heap_emhash8_set9_opt比bpe_train_updater_fine_grained_heap_emhash8_set9的update时间快11%。bpe_train_updater_fine_grained_heap_emhash8_set_opt比bpe_train_updater_fine_grained_heap_emhash8_set快8%。这个结论和python版本是一致的。


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

