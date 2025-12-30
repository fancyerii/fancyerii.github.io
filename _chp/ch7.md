---
layout:     post
title:      "第七章：内存管理"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - c++
---



<!--more-->


**目录**
* TOC
{:toc}
 
  
经过前面章节的学习，我们处理内存的方式会对性能产生巨大影响，这一点应该不再令人意外。CPU 将大量时间花费在 CPU 寄存器和主内存之间的数据传输上（从主内存加载数据和向主内存存储数据）。正如 第 4 章，《数据结构》 中所示，CPU 使用内存缓存来加速对内存的访问，程序需要对缓存友好才能快速运行。

本章将揭示计算机处理内存的更多方面，以便你知道在调整内存使用时必须考虑哪些因素。此外，本章还涵盖：

* 自动内存分配和动态内存管理。
* C++ 对象的生命周期以及如何管理对象所有权。
* 高效的内存管理。有时，存在严格的内存限制，迫使我们保持数据表示的紧凑；有时，我们有充足的内存，但需要通过使内存管理更高效来加快程序运行速度。
* 如何最大限度地减少动态内存分配。动态内存的分配和释放相对昂贵，有时我们需要避免不必要的分配，以使程序运行得更快。

在本章开始，我们将解释一些在你深入研究 C++ 内存管理之前需要理解的概念。本介绍将解释虚拟内存和虚拟地址空间、栈内存与堆内存、分页 (Paging) 和交换空间 (Swap Space)。


## 计算机内存 (Computer memory)

计算机的物理内存由系统上运行的所有进程共享。如果一个进程使用了大量内存，其他进程很可能会受到影响。但是，从程序员的角度来看，我们通常不必担心其他进程正在使用的内存。这种内存隔离是由于当今大多数操作系统都是虚拟内存操作系统，它们提供了一个进程独占所有内存的错觉。

每个进程都有自己的虚拟地址空间 (virtual address space)。


### 虚拟地址空间 (The virtual address space)

程序员看到的虚拟地址空间中的地址，由操作系统和内存管理单元（MMU）（它是处理器的一部分）映射到物理地址。这种映射或翻译发生在每次我们访问一个内存地址时。

额外的这一层间接性使得操作系统可以只对进程当前正在使用的部分使用物理内存，而将其余的虚拟内存备份到磁盘上。从这个意义上讲，我们可以将物理主内存视为虚拟地址空间的缓存，而虚拟地址空间驻留在辅助存储器上。用于备份内存页的辅助存储区域通常被称为交换空间 (swap space)、交换文件 (swap file) 或简称 页文件 (pagefile)，具体取决于操作系统。

虚拟内存使得进程可以拥有一个大于物理地址空间的虚拟地址空间，因为未使用的虚拟内存不必占用物理内存。


### 内存页 (Memory pages)

当今实现虚拟内存最常见的方式是将地址空间划分为固定大小的块，称为内存页 (memory pages)。当一个进程访问一个虚拟地址的内存时，操作系统会检查该内存页是否由物理内存（一个页帧/页框 page frame）支持。如果该内存页没有映射到主内存中，就会发生硬件异常，并且该页会从磁盘加载到内存中。这种类型的硬件异常被称为页错误 (page fault)。这不是一个错误，而是从磁盘加载数据到内存所必需的中断。但是，你可能已经猜到，与读取已驻留在内存中的数据相比，这个过程非常慢。

当主内存中没有更多可用的页帧时，必须驱逐一个页帧。如果将被驱逐的页是脏的 (dirty)（即它自上次从磁盘加载后已被修改），它需要在被替换之前被写回磁盘。这种机制被称为分页 (paging)。如果内存页没有被修改，则该内存页会被简单地驱逐。

并非所有支持虚拟内存的操作系统都支持分页。例如，iOS 确实有虚拟内存，但脏页永远不会存储在磁盘上；只有干净页可以从内存中驱逐。如果主内存已满，iOS 将开始终止进程，直到有足够的可用内存。Android 使用了类似的策略。不将内存页写回移动设备闪存的原因之一是这会消耗电池，并且也会缩短闪存本身的寿命。

下面的图表显示了两个正在运行的进程。它们都有自己的虚拟内存空间。一些页映射到物理内存，而另一些则没有。如果进程 1 需要使用起始地址为 `0x1000` 的内存页中的内存，就会发生页错误。然后，该内存页将被映射到一个空闲的内存帧。另请注意，虚拟内存地址与物理地址不同。进程 1 的第一个内存页（起始虚拟地址为 `0x0000`）映射到一个起始物理地址为 `0x4000` 的内存帧：

<a>![](/img/chp/ch7/1.png)</a>


### 抖动 (Thrashing)

当系统物理内存不足，因此不断地进行分页操作时，就会发生抖动 (Thrashing)。每当一个进程获得 CPU 调度时间时，它就会尝试访问已被换出到页文件中的内存。加载新的内存页意味着其他页必须首先存储到磁盘上。在磁盘和内存之间来回移动数据通常非常慢；在某些情况下，这会或多或少地使计算机停滞，因为系统将所有时间都花在了分页上。观察系统的页错误频率是确定程序是否开始抖动的一个好方法。

了解硬件和操作系统如何处理内存的基础知识对于性能优化非常重要。接下来，我们将了解在 C++ 程序执行期间内存是如何处理的。



## 进程内存 (Process memory)

栈 (stack) 和 堆 (heap) 是 C++ 程序中两个最重要的内存段。还有静态存储 (static storage) 和线程局部存储 (thread local storage)，但我们稍后会详细讨论。实际上，从正式的 C++ 规范角度来说，C++ 并没有谈论栈和堆；相反，它谈论的是自由存储区 (free store)、存储类 (storage classes) 和对象的存储期 (storage duration)。然而，由于栈和堆的概念在 C++ 社区中被广泛使用，并且我们所知道的所有 C++ 实现都使用栈来实现函数调用和管理局部变量的自动存储，因此理解栈和堆是什么非常重要。

在本书中，我也会使用栈和堆这两个术语，而不是使用对象的存储期。我将堆和自由存储区交替使用，不作区分。

栈和堆都位于进程的虚拟内存空间中。栈是所有局部变量驻留的地方，这也包括函数的参数。每次调用函数时，栈增长；当函数返回时，栈收缩。每个线程都有自己的栈，因此栈内存可以被认为是线程安全的。另一方面，堆是一个全局内存区域，由运行进程中的所有线程共享。当我们使用 `new`（或 C 库函数 `malloc()` 和 `calloc()`）分配内存时，堆增长；当我们使用 `delete`（或 `free()`）释放内存时，堆收缩。通常，堆从低地址开始并向上增长，而栈从高地址开始并向下增长。图 7.2 展示了栈和堆在虚拟地址空间中如何向相反方向增长：

<a>![](/img/chp/ch7/2.png)</a>

接下来的部分将提供关于栈和堆的更多细节，并解释在我们编写的 C++ 程序中何时使用了这些内存区域。

 

### 栈内存 (Stack memory)

栈在许多方面与堆不同。以下是栈的一些独有属性：

  * 栈是一个连续的内存块。
  * 它有一个固定的最大大小。如果程序超过最大栈大小，程序将崩溃。这种情况称为栈溢出 (stack overflow)。
  * 栈内存永远不会发生碎片化。
  * 从栈分配内存几乎）总是很快的。可能发生页错误，但很少见。
  * 程序中的每个线程都有自己的栈。

本节中的后续代码示例将探讨其中一些属性。让我们从分配和释放开始，以了解程序中如何使用栈。

我们可以通过检查栈分配数据的地址，轻松找出栈的增长方向。以下示例代码演示了在进入和离开函数时栈如何增长和收缩：

```cpp
void func1() {
    auto i = 0;
    std::cout << "func1(): " << std::addressof(i) << '\n';
}

void func2() {
    auto i = 0;
    std::cout << "func2(): " << std::addressof(i) << '\n';
    func1();
}

int main() {
    auto i = 0;
    std::cout << "main(): " << std::addressof(i) << '\n';
    func2();
    func1();
}
```

运行该程序的一个可能输出可能如下所示：

```
main(): 0x7ea075ac
func2(): 0x7ea07594
func1(): 0x7ea0757c
func1(): 0x7ea07594
```

通过打印栈分配的整型变量的地址，我们可以确定在我的平台上栈增长了多少以及朝哪个方向增长。每次我们进入 `func1()` 或 `func2()` 时，栈向下增长 24 字节。整型变量 `i`（它将被分配在栈上）长 4 字节。剩余的 20 字节包含函数结束时所需的数据，例如返回地址，以及可能的对齐填充。

下图说明了程序执行期间栈如何增长和收缩。第一个框说明了程序刚进入 `main()` 函数时的内存情况。第二个框显示了执行 `func1()` 时栈如何增加，依此类推：

<a>![](/img/chp/ch7/3.png)</a>

为栈分配的总内存是一个在线程启动时创建的固定大小的连续内存块。那么，栈有多大？当我们达到栈的限制时会发生什么？

如前所述，每次程序进入一个函数时，栈会增长；当函数返回时，栈会收缩。当我们在同一个函数内创建新的栈变量时，栈也会增长；当这些变量超出作用域时，栈会收缩。栈溢出最常见的原因是深度递归调用和/或在栈上使用大型的自动变量。栈的最大大小在不同平台之间有所不同，也可以针对单个进程和线程进行配置。

让我们试着编写一个程序，看看在我的系统上栈的默认大小是多少。我们将从编写一个将无限递归的函数 `func()` 开始。在每个函数的开头，我们将分配一个 1 千字节的变量，每次我们进入 `func()` 时，该变量都会被放置到栈上。每次 `func()` 执行时，我们都打印栈的当前大小：

```cpp
void func(std::byte* stack_bottom_addr) {
    std::byte data[1024];
    std::cout << stack_bottom_addr - data << '\n';
    func(stack_bottom_addr);
}

int main() {
    std::byte b;
    func(&b);
}
```

栈的大小只是一个估计值。我们通过将 `main()` 中第一个局部变量的地址减去 `func()` 中定义的第一个局部变量的地址来计算它。

当我在 Clang 中编译代码时，我收到了一个关于 `func()` 永远不会返回的警告。通常，这是一个我们不应该忽略的警告，但这次，这正是我们想要的结果，所以我们忽略警告并运行程序。程序在栈达到其限制后不久就崩溃了。在程序崩溃之前，它成功打印了数千行栈的当前大小。输出的最后几行看起来像这样：

```
...
8378667
8379755
8380843
```

由于我们正在减去 `std::byte` 指针，所以大小是以字节为单位的，因此看起来在我的系统上栈的最大大小约为 8 MB。在类 Unix 系统上，可以使用带有 `-s` 选项的 `ulimit` 命令来设置和获取进程的栈大小：

```bash
$ ulimit -s
$ 8192
```

`ulimit`（`user limit` 的缩写）返回以千字节为单位的最大栈大小的当前设置。`ulimit` 的输出证实了我们实验的结果：如果我不明确配置它，我的 Mac 上的栈大约是 8 MB。

在 Windows 上，默认栈大小通常设置为 1 MB。如果栈大小未正确配置，在一个 macOS 上运行良好的程序可能会在 Windows 上因栈溢出而崩溃。

通过这个例子，我们也可以得出结论，我们不希望栈内存用尽，因为那样程序就会崩溃。在本章后面，我们将看到如何实现一个基本内存分配器来处理固定大小的分配。届时我们将理解，栈只是另一种类型的内存分配器，由于其使用模式始终是顺序的，因此可以非常高效地实现。我们总是在栈的顶部（连续内存的末端）请求和释放内存。这确保了栈内存永远不会碎片化，并且我们可以通过只移动一个栈指针来分配和释放内存。
 

### 堆内存 (Heap memory)

堆（或自由存储区，这是 C++ 中更正确的术语）是具有动态存储期数据驻留的地方。如前所述，堆在多个线程之间共享，这意味着堆的内存管理需要考虑并发性。这使得堆中的内存分配比栈分配（栈分配是每个线程局部的）更复杂。

栈内存的分配和释放模式是顺序的，因为它总是按照与分配相反的顺序释放内存。另一方面，对于动态内存，分配和释放可以任意发生。对象的动态生命周期和内存分配的可变大小增加了内存碎片化的风险。

理解内存碎片化问题的一个简单方法是看一个内存碎片化如何发生的例子。假设我们有一个 16 KB 的小连续内存块，我们正在从中分配内存。我们正在分配两种类型的对象：类型 A（1 KB）和类型 B（2 KB）。我们首先分配一个类型 A 的对象，接着是一个类型 B 的对象。这个过程重复，直到内存看起来像下面的图：

<a>![](/img/chp/ch7/4.png)</a>

接下来，类型 A 的所有对象都不再需要了，因此可以被释放。内存现在看起来像这样：

<a>![](/img/chp/ch7/5.png)</a>

现在有 10 KB 的内存在使用中，6 KB 可用。现在，假设我们想分配一个新的类型 B 对象，它是 2 KB。尽管有 6 KB 的空闲内存，但我们找不到一个 2 KB 的连续内存块，因为内存已经碎片化了。

现在你对计算机内存的结构以及它在运行进程中的使用方式有了很好的理解，是时候探讨 C++ 对象如何在内存中生存了。

## 内存中的对象 (Objects in memory)

我们在 C++ 程序中使用的所有对象都驻留在内存中。在这里，我们将探讨对象是如何在内存中创建和删除的，并描述对象在内存中的布局方式。


### 创建和删除对象 (Creating and deleting objects)

在本节中，我们将深入探讨使用 `new` 和 `delete` 的细节。考虑以下使用 `new` 在自由存储区创建对象，然后使用 `delete` 删除对象的方式：

```cpp
auto* user = new User{"John"}; // 分配内存并构造对象
user->print_name();           // 使用对象
delete user;                  // 析构对象并释放内存
```

我不建议你以这种方式显式调用 `new` 和 `delete`，但我们暂时忽略这一点。直奔主题；正如注释所示，`new` 实际上做了两件事，即：

1.  分配内存：分配内存以容纳 `User` 类型的新对象。
2.  构造对象：通过调用 `User` 类的构造函数，在分配的内存空间中构造一个新的 `User` 对象。

`delete` 也是如此，它：

1.  析构对象：通过调用其析构函数来析构 `User` 对象。
2.  释放内存：释放/归还放置 `User` 对象的内存。

实际上，在 C++ 中可以分离这两个动作（内存分配和对象构造）。这很少使用，但在编写库组件时有一些重要且合理的用例。

#### Placement New (定位 new)

C++ 允许我们将内存分配与对象构造分离。例如，我们可以使用 `malloc()` 分配一个字节数组，并在该内存区域中构造一个新的 `User` 对象。请看以下代码片段：

```cpp
auto* memory = std::malloc(sizeof(User));
auto* user = ::new (memory) User("john");
```

这种可能看起来不熟悉的、使用 `::new (memory)` 的语法被称为 Placement New（定位 new）。它是 `new` 的一种不分配内存的形式，只构造对象。`new` 前面的双冒号 (`::`) 确保是从全局命名空间进行解析，以避免选取到 `operator new` 的重载版本。

在前面的示例中，Placement New 构造了 `User` 对象并将其放置在指定的内存位置。由于我们使用 `std::malloc()` 分配了单个对象的内存，因此可以保证它是正确对齐的（除非 `User` 类被声明为过度对齐 (overaligned)）。稍后，我们将探讨在使用 Placement New 时必须考虑对齐的情况。

没有 Placement Delete，因此为了析构对象并释放内存，我们需要显式调用析构函数，然后释放内存：

```cpp
user->~User();
std::free(memory);
```

> 注意： 只有在你使用 Placement New 创建对象时，才应该像这样显式调用析构函数。绝不应该在其他情况下如此调用析构函数。

C++17 在 `<memory>` 头文件中引入了一组用于构造和销毁对象而不分配或释放内存的实用函数。因此，现在可以使用 `<memory>` 中的一些以 `std::uninitialized_` 开头的函数，而不是调用 Placement New，用于在未初始化内存区域构造、复制和移动对象。而且，现在可以使用 `std::destroy_at()` 在特定内存地址销毁对象而不释放内存，而不是显式调用析构函数。

前面的示例可以使用这些新函数重写。它看起来会像这样：

```cpp
auto* memory = std::malloc(sizeof(User));
auto* user_ptr = reinterpret_cast<User*>(memory);
std::uninitialized_fill_n(user_ptr, 1, User{"john"});
std::destroy_at(user_ptr);
std::free(memory);
```

C++20 还引入了 `std::construct_at()`，这使得可以用以下代码替换 `std::uninitialized_fill_n()` 调用：

```cpp
std::construct_at(user_ptr, User{"john"}); // C++20
```

请记住，我们展示这些裸露的底层内存工具是为了更好地理解 C++ 中的内存管理。在 C++ 代码库中，应将使用 `reinterpret_cast` 和此处演示的内存工具保持在绝对最低限度。

接下来，你将看到当我们使用 `new` 和 `delete` 表达式时会调用哪些操作符。


### New 和 Delete 操作符 (The new and delete operators)

当调用 `new` 表达式时，`operator new` 函数负责分配内存。`operator new` 可以是全局定义的函数，也可以是类的静态成员函数。可以重载全局的 `operator new` 和 `operator delete`。在本章后面，我们将看到这在分析内存使用情况时很有用。

下面是重载方式：

```cpp
auto operator new(size_t size) -> void* {
    void* p = std::malloc(size);
    std::cout << "allocated " << size << " byte(s)\n";
    return p;
}

auto operator delete(void* p) noexcept -> void {
    std::cout << "deleted memory\n";
    return std::free(p);
}
```

我们可以验证在创建和删除一个 `char` 对象时，我们的重载操作符是否确实被使用了：

```cpp
auto* p = new char{'a'}; // Outputs "allocated 1 byte(s)"
delete p;                // Outputs "deleted memory"
```

当使用 `new[]` 和 `delete[]` 表达式创建和删除对象数组时，会使用另一对操作符，即 `operator new[]` 和 `operator delete[]`。我们可以用同样的方式重载这些操作符：

```cpp
auto operator new[](size_t size) -> void* {
    void* p = std::malloc(size);
    std::cout << "allocated " << size << " byte(s) with new[]\n";
    return p;
}

auto operator delete[](void* p) noexcept -> void {
    std::cout << "deleted memory with delete[]\n";
    return std::free(p);
}
```

请记住，如果你重载了 `operator new`，你也应该重载 `operator delete`。分配和释放内存的函数是成对出现的。内存应该由分配该内存的分配器来释放。例如，使用 `std::malloc()` 分配的内存应始终使用 `std::free()` 释放，而使用 `operator new[]` 分配的内存应使用 `operator delete[]` 释放。

也可以重写类特定的 `operator new` 或 `operator delete`。这可能比重载全局操作符更有用，因为我们更有可能需要为特定类定制动态内存分配器。

在这里，我们为 `Document` 类重载了 `operator new` 和 `operator delete`：

```cpp
class Document {
    // ...
public:
    auto operator new(size_t size) -> void* {
        return ::operator new(size);
    }
    auto operator delete(void* p) -> void {
        ::operator delete(p);
    }
};
```

当我们创建新的动态分配的 `Document` 对象时，将使用类特定的 `new` 版本：

```cpp
auto* p = new Document{}; // Uses class-specific operator new
delete p;
```

如果我们想改为使用全局 `new` 和 `delete`，仍然可以通过使用全局作用域 (`::`) 来实现：

```cpp
auto* p = ::new Document{}; // Uses global operator new
::delete p;
```

我们将在本章后面讨论内存分配器，届时我们将看到重载的 `new` 和 `delete` 操作符的应用。

总结一下我们到目前为止所看到的内容，一个 `new` 表达式涉及两件事：分配和构造。`operator new` 分配内存，你可以全局或按类重载它以自定义动态内存管理。Placement New 可用于在已分配的内存区域中构造对象。

另一个我们需要理解才能高效使用内存的重要但相当底层的主题是内存对齐 (alignment of memory)。


## 内存对齐 (Memory alignment)

CPU 一次读取一个字 (word) 的内存到其寄存器中。在 64 位架构上，字大小是 64 位；在 32 位架构上，字大小是 32 位，依此类推。为了让 CPU 在处理不同数据类型时高效工作，它对不同类型对象所处的地址有限制。C++ 中的每种类型都有一个对齐要求 (alignment requirement)，它定义了某一类型对象在内存中应位于的地址。

如果一个类型的对齐度是 1，则意味着该类型的对象可以位于任何字节地址。如果一个类型的对齐度是 2，则意味着连续允许地址之间的字节数是 2。或者引用 C++ 标准的说法：

> "对齐度是一个由实现定义的整数值，表示给定对象可以分配的连续地址之间的字节数。"

我们可以使用 `alignof` 来找出某个类型的对齐度：

```cpp
// 可能的输出是 4
std::cout << alignof(int) << '\n';
```

当我运行这段代码时，它输出 4，这意味着在我的平台上，`int` 类型的对齐要求是 4 字节。

下图显示了来自一个具有 64 位字的系统的两个内存示例。上排包含三个 4 字节的整数，它们位于 4 字节对齐的地址上。CPU 可以高效地将这些整数加载到寄存器中，并且在访问其中一个 `int` 成员时永远不需要读取多个字。将此与第二排进行比较，第二排包含两个 `int` 成员，它们位于未对齐的地址上。第二个 `int` 甚至跨越了两个字边界。在最好的情况下，这只是效率低下；但在某些平台上，程序会崩溃：

<a>![](/img/chp/ch7/6.png)</a>

假设我们有一个对齐要求为 2 的类型。C++ 标准没有说明有效地址是 $1, 3, 5, 7...$ 还是 $0, 2, 4, 6...$。我们知道的所有平台都从地址 $0$ 开始计数，因此实际上我们可以通过使用取模操作符 (`%`) 检查对象的地址是否是对齐度的倍数来判断它是否正确对齐。

然而，如果我们要编写完全可移植的 C++ 代码，我们需要使用 `std::align()` 而不是取模来检查对象的对齐度。`std::align()` 是 `<memory>` 中的一个函数，它会根据我们作为参数传递的对齐度来调整指针。如果我们传递给它的内存地址已经对齐，则指针不会被调整。因此，我们可以使用 `std::align()` 来实现一个名为 `is_aligned()` 的小型实用函数，如下所示：

```cpp
bool is_aligned(void* ptr, std::size_t alignment) {
    assert(ptr != nullptr);
    assert(std::has_single_bit(alignment)); // 2 的幂次
    auto s = std::numeric_limits<std::size_t>::max();
    auto aligned_ptr = ptr;
    std::align(alignment, 1, aligned_ptr, s);
    return ptr == aligned_ptr;
}
```

首先，我们确保 `ptr` 参数不为空，并且 `alignment` 是 2 的幂次，这是 C++ 标准中规定的要求。我们使用 `<bit>` 头文件中的 C++20 `std::has_single_bit()` 来检查这一点。接下来，我们调用 `std::align()`。`std::align()` 的典型用例是当我们有一个某个大小的内存缓冲区，我们想在其中存储一个具有某种对齐要求的对象时。在这种情况下，我们没有缓冲区，也不关心对象的大小，所以我们说对象的大小为 1，缓冲区大小是 `std::size_t` 的最大值。然后，我们可以比较原始的 `ptr` 和调整后的 `aligned_ptr`，看看原始指针是否已经对齐。在接下来的示例中，我们将使用到这个实用工具。

当使用 `new` 或 `std::malloc()` 分配内存时，我们返回的内存应该对我们指定的类型正确对齐。以下代码显示，在我当前的平台上，为 `int` 分配的内存至少是 4 字节对齐的：

```cpp
auto* p = new int{};
assert(is_aligned(p, 4ul)); // True
```

实际上，`new` 和 `malloc()` 保证始终返回适用于任何标量类型的适当对齐的内存（如果它成功返回内存的话）。`<cstddef>` 头文件为我们提供了一个名为 `std::max_align_t` 的类型，其对齐要求至少与所有标量类型一样严格。稍后，我们将看到这种类型在编写自定义内存分配器时很有用。因此，即使我们只在自由存储区请求 `char` 的内存，它也会被对齐到适用于 `std::max_align_t` 的程度。

以下代码显示，从 `new` 返回的内存对 `std::max_align_t` 和任何标量类型都是正确对齐的：

```cpp
auto* p = new char{};
auto max_alignment = alignof(std::max_align_t);
assert(is_aligned(p, max_alignment)); // True
```

让我们用 `new` 连续分配两次 `char`：

```cpp
auto* p1 = new char{'a'};
auto* p2 = new char{'b'};
```

那么，内存可能看起来像这样：

<a>![](/img/chp/ch7/7.png)</a>

`p1` 和 `p2` 之间的空间取决于 `std::max_align_t` 的对齐要求。在我的系统上，它是 16 字节，因此每个 `char` 实例之间有 15 字节的填充，即使 `char` 的对齐度只有 1。

可以使用 `alignas` 说明符在声明变量时指定比默认对齐要求更严格的自定义对齐要求。假设我们的缓存行大小是 64 字节，并且由于某种原因，我们希望确保两个变量位于不同的缓存行上。我们可以这样做：

```cpp
alignas(64) int x{};
alignas(64) int y{};
// x 和 y 将被放置在不同的缓存行上
```

也可以在定义类型时指定自定义对齐。以下是一个在使用时将恰好占据一个缓存行的结构体：

```cpp
struct alignas(64) CacheLine {
    std::byte data[64];
};
```

现在，如果我们创建一个 `CacheLine` 类型的栈变量，它将根据 64 字节的自定义对齐进行对齐：

```cpp
int main() {
    auto x = CacheLine{};
    auto y = CacheLine{};
    assert(is_aligned(&x, 64));
    assert(is_aligned(&y, 64));
    // ...
}
```

在堆上分配对象时，更严格的对齐要求也会得到满足。为了支持动态分配具有非默认对齐要求的类型，C++17 引入了 `operator new()` 和 `operator delete()` 的新重载，它们接受一个 `std::align_val_t` 类型的对齐参数。`<cstdlib>` 中还定义了一个 `aligned_alloc()` 函数，可用于手动分配对齐的堆内存。

下面是一个示例，我们在其中分配一个恰好占据一个内存页的堆内存块。在这种情况下，当使用 `new` 和 `delete` 时，将调用对齐感知版本的 `operator new()` 和 `operator delete()`：

```cpp
constexpr auto ps = std::size_t{4096}; // 页面大小
struct alignas(ps) Page {
    std::byte data_[ps];
};

auto* page = new Page{};       // 内存页
assert(is_aligned(page, ps));  // True
// 使用 page ...
delete page;
```

内存页不是 C++ 抽象机器的一部分，因此没有可移植的方法来以编程方式获取当前运行系统的页面大小。但是，你可以使用 `boost::mapped_region::get_page_size()` 或平台特定的系统调用，例如在 Unix 系统上的 `getpagesize()`。

最后一个需要注意的告诫是，支持的对齐集合由你正在使用的标准库的实现定义，而不是 C++ 标准。



## 填充 (Padding)

编译器有时需要向我们用户定义的类型中添加额外的字节，即填充 (padding)。当我们在类或结构体中定义数据成员时，编译器被迫按我们定义的相同顺序放置这些成员。

然而，编译器还必须确保类内部的数据成员具有正确的对齐度；因此，如果需要，它需要在数据成员之间添加填充。例如，假设我们定义了一个类如下：

```cpp
class Document {
    bool is_cached_{};
    double rank_{};
    int id_{};
};

std::cout << sizeof(Document) << '\n'; // 可能的输出是 24
```

可能的输出为 24 的原因是编译器在 `bool` 和 `int` 之后插入了填充，以满足各个数据成员和整个类的对齐要求。编译器将 `Document` 类转换为类似如下的内容：

```cpp
class Document {
    bool is_cached_{};
    std::byte padding1[7]; // 编译器插入的不可见填充
    double rank_{};
    int id_{};
    std::byte padding2[4]; // 编译器插入的不可见填充
};
```

`bool` 和 `double` 之间的第一个填充是 7 字节，因为 `double` 类型的 `rank_` 数据成员的对齐度为 8 字节。在 `int` 之后添加的第二个填充是 4 字节。这是为了满足 `Document` 类本身的对齐要求。具有最大对齐要求的成员也决定了整个数据结构的对齐要求。在我们的示例中，这意味着 `Document` 类的总大小必须是 8 的倍数，因为它包含一个 8 字节对齐的 `double` 值。

【译注：id已经对齐到4字节，为什么后面还需要padding2？因为一个结构体的大小必须是其最大元素(这里是rank_，float 8字节)的倍数，否则申请Document数组时就会出现最大元素不对齐的情况。假设没有padding2，则Document的size是20字节，如果我们申请Document[2]，那么第二个Document的rank_就没有8字节对齐。

】

我们现在意识到，我们可以通过从具有最大对齐要求的类型开始，重新排列 `Document` 类中数据成员的顺序，从而最大限度地减少编译器插入的填充。让我们创建一个新版本的 `Document` 类：

```cpp
// Document 类的版本 2
class Document {
    double rank_{};     // 重新排列的数据成员
    int id_{};
    bool is_cached_{};
};
```

通过重新排列成员，编译器现在只需要在 `is_cached_` 数据成员之后进行填充，以调整 `Document` 的对齐。这是填充后该类的外观：

```cpp
// 填充后的 Document 类的版本 2
class Document {
    double rank_{};
    int id_{};
    bool is_cached_{};
    std::byte padding[3]; // 编译器插入的不可见填充
};
```

新的 `Document` 类的大小现在只有 16 字节，而第一个版本是 24 字节。这里的要点是，仅通过更改其成员的声明顺序，对象的大小就可以改变。我们也可以通过对更新后的 `Document` 版本再次使用 `sizeof` 操作符来验证这一点：

```cpp
std::cout << sizeof(Document) << '\n'; // 可能的输出是 16
```

下图显示了 `Document` 类版本 1 和版本 2 的内存布局：

<a>![](/img/chp/ch7/8.png)</a>

通常的规则是，你可以将最大的数据成员放在开头，将最小的成员放在末尾。通过这种方式，你可以最大限度地减少由填充引起的内存开销。稍后，我们将看到，当我们将在分配的内存区域中放置对象时，我们需要考虑对齐，因为我们事先不知道正在创建的对象的对齐度。

从性能角度来看，在某些情况下，你可能希望将对象对齐到缓存行，以最大限度地减少一个对象跨越的缓存行数。当我们谈论缓存友好性时，还应该提到，将频繁一起使用的多个数据成员放置在一起也是有益的。

保持数据结构的紧凑性对性能很重要。许多应用程序受到内存访问时间的限制。内存管理的另一个重要方面是永远不要泄露或浪费不再需要的对象的内存。我们可以通过清晰明确地界定资源的所有权来有效避免各种资源泄露。这是下一节的主题。
 
 
## 内存所有权 (Memory ownership)

资源的所有权是编程时需要考虑的一个基本方面。资源的所有者负责在资源不再需要时将其释放。资源通常是一块内存块，但也可能是数据库连接、文件句柄等等。无论你使用哪种编程语言，所有权都很重要。然而，在 C 和 C++ 等语言中，它更为明显，因为默认情况下动态内存没有垃圾回收机制。

每当我们在 C++ 中分配动态内存时，都必须考虑该内存的所有权。幸运的是，现在该语言通过使用智能指针来表达各种类型的所有权，提供了非常好的支持，我们将在本节后面讨论这些智能指针。

标准库中的智能指针帮助我们指定动态变量的所有权。其他类型的变量已经有了明确的所有权。例如，局部变量由当前作用域所有。当作用域结束时，在该作用域内创建的对象将自动销毁：

```cpp
{
    auto user = User{};
} // user 在超出作用域时自动销毁
```

静态变量和全局变量由程序所有，并将在程序终止时销毁：

```cpp
static auto user = User{};
```

数据成员由它们所属的类实例所有：

```cpp
class Game {
    User user; // 一个 Game 对象拥有 User 对象
    // ...
};
```

只有动态变量没有默认所有者，程序员有责任确保所有动态分配的变量都有一个所有者来控制变量的生命周期：

```cpp
auto* user = new User{}; // 现在谁拥有 user？
```

使用现代 C++，我们可以在大部分代码中不显式调用 `new` 和 `delete`，这是一件好事。手动跟踪 `new` 和 `delete` 的调用非常容易导致内存泄漏、二次删除以及其他令人讨厌的错误。裸指针不表达任何所有权，这使得如果我们只使用裸指针来引用动态内存，所有权就很难跟踪。

我建议你使所有权清晰和明确，但要努力最大限度地减少手动内存管理。通过遵循一些相当简单的处理内存所有权规则，你将增加代码干净、正确且不会泄露资源的可能性。接下来的部分将指导你完成一些实现此目的的最佳实践。


### 隐式处理资源 (Handling resources implicitly)

首先，让你的对象隐式地处理动态内存的分配/释放：

```cpp
auto func() {
    auto v = std::vector<int>{1, 2, 3, 4, 5};
}
```

在前面的示例中，我们使用了栈内存和动态内存，但我们不必显式调用 `new` 和 `delete`。我们创建的 `std::vector` 对象是一个自动对象，它将存在于栈上。因为它由作用域所有，所以当函数返回时它将自动销毁。`std::vector` 对象本身使用动态内存来存储整数元素。当 `v` 超出作用域时，它的析构函数可以安全地释放动态内存。这种让析构函数释放动态内存的模式使得避免内存泄漏变得相当容易。

当我们谈论释放资源时，我认为有必要提及 RAII。RAII 是一种众所周知的 C++ 技术，是 Resource Acquisition Is Initialization (资源获取即初始化) 的缩写，其中资源的生命周期由对象的生命周期控制。这种模式很简单，但对于处理资源（包括内存）极其有用。

但假设，为了换个例子，我们需要的资源是某种用于发送请求的连接。每当我们使用完连接后，我们（所有者）必须记得关闭它。以下是手动打开和关闭连接以发送请求的示例：

```cpp
auto send_request(const std::string& request) {
    auto connection = open_connection("http://www.example.com/");
    send_request(connection, request);
    close(connection);
}
```

如你所见，我们必须记住在使用连接后关闭它，否则连接将保持打开状态（泄露）。在这个示例中，这似乎很难忘记，但一旦代码变得更复杂（在插入适当的错误处理和多个退出路径后），就很难保证连接将始终被关闭。

RAII 通过依赖自动变量的生命周期以可预测的方式为我们处理这一事实来解决这个问题。我们需要的是一个与我们从 `open_connection()` 调用获得的连接具有相同生命周期的对象。我们可以为此创建一个名为 `RAIIConnection` 的类：

```cpp
class RAIIConnection {
public:
    explicit RAIIConnection(const std::string& url)
        : connection_{open_connection(url)} {}
    
    ~RAIIConnection() {
        try {
            close(connection_);
        }
        catch (const std::exception&) {
            // 处理错误，但永远不要从析构函数中抛出异常
        }
    }
    
    auto& get() { return connection_; }

private:
    Connection connection_;
};
```

`Connection` 对象现在被包装在一个控制连接（资源）生命周期的类中。现在，我们可以让 `RAIIConnection` 为我们处理连接，而不是手动关闭它：

```cpp
auto send_request(const std::string& request) {
    auto connection = RAIIConnection("http://www.example.com/");
    send_request(connection.get(), request);
    // 无需关闭连接，它由 RAIIConnection 析构函数自动处理
}
```

RAII 使我们的代码更安全。即使 `send_request()` 在此处抛出异常，`connection` 对象仍将被析构并关闭连接。我们可以将 RAII 用于许多类型的资源，不仅限于内存、文件句柄和连接。另一个例子是 C++ 标准库中的 `std::scoped_lock`。它在创建时尝试获取一个锁（互斥体），然后在析构时释放该锁。你可以在第 11 章，《并发》中阅读有关 `std::scoped_lock` 的更多信息。

现在，我们将探索更多使 C++ 中内存所有权明确的方法。


### 容器 (Containers)

你可以使用标准容器来处理对象的集合。你使用的容器将拥有存储你添加到其中的对象所需的动态内存。这是最大限度地减少代码中手动 `new` 和 `delete` 表达式的一种非常有效的方法。

也可以使用 `std::optional` 来处理可能存在或可能不存在的对象的生命周期。`std::optional` 可以看作是一个最大大小为 1 的容器。

我们在这里将不再讨论容器，因为它们已在第 4 章，《数据结构》中介绍过。


### 智能指针 (Smart pointers)

标准库中的智能指针包装了一个裸指针，并使其指向的对象的所有权变得明确。正确使用时，毫不疑问谁应该负责删除一个动态对象。三种智能指针类型是：`std::unique_ptr`、`std::shared_ptr` 和 `std::weak_ptr`。正如它们的名称所示，它们代表了对象的三种所有权类型：

  * 独占所有权 (Unique ownership)：表示我，且只有我拥有该对象。当我使用完它时，我将删除它。
  * 共享所有权 (Shared ownership)：表示我与其他人共同拥有该对象。当没有人再需要该对象时，它将被删除。
  * 弱所有权 (Weak ownership)：表示如果该对象存在，我将使用它，但不要仅仅为了我而保持它存活。

我们将在接下来的部分中分别处理这些类型。

#### 独占指针 (Unique pointer)

最安全且最不复杂的所用权是独占所有权，并且应该是你在考虑智能指针时首先想到的。独占指针代表独占所有权；即，一个资源由恰好一个实体所有。独占所有权可以转移给其他人，但不能复制，因为那会破坏其唯一性。以下是如何使用 `std::unique_ptr`：

```cpp
auto owner = std::make_unique<User>("John");
auto new_owner = std::move(owner); // 转移所有权
```

独占指针也非常高效，因为与普通裸指针相比，它们增加的性能开销非常小。这种轻微的开销是由于 `std::unique_ptr` 具有一个非平凡的析构函数，这意味着（与裸指针不同）在作为参数传递给函数时，它不能在 CPU 寄存器中传递。这使得它们比裸指针慢一些。

【译注：


`std::unique_ptr` 的核心设计理念是 零开销抽象 (Zero-Overhead Abstraction)：

* 内存开销为零： 在运行时，`std::unique_ptr` 内部只包含一个原始的裸指针。它的 `sizeof()` 大小与裸指针完全相同，不包含额外的引用计数（这是 `std::shared_ptr` 的开销）。
* 运行时操作开销低： 访问它所管理的对象（通过 `*` 或 `->`）与使用裸指针一样快，因为编译器可以直接将其优化成对裸指针的访问。



轻微开销源于 `std::unique_ptr` 的生命周期管理，特别是在作为函数参数传递时。


* 裸指针： `char* p;` 的析构函数是平凡的 (trivial)，它不执行任何操作。
* `std::unique_ptr`： 它的析构函数 `~unique_ptr()` 是非平凡的。它必须包含逻辑来检查指针是否非空，并在超出作用域时调用 `delete` 来释放底层资源。


在 C++ 调用约定中，如果一个类或结构体的析构函数是非平凡的，那么它在作为值（by value）传递给函数时，通常不能直接在 CPU 寄存器中传递。

* 裸指针 (`T*`)： 由于其平凡的析构函数和简单的结构，它可以直接被放入 CPU 寄存器中传递，这速度极快。
* `std::unique_ptr<T>`： 尽管其大小与裸指针相同，但因为它有一个非平凡的析构函数（即它需要执行清理工作），编译器可能需要将其视为一个复杂的对象。这可能导致在函数调用时：
    1.  需要在栈上为其分配空间。
    2.  需要执行额外的指令来复制或移动它。

结论： 这种对调用约定的限制，使得 `std::unique_ptr` 在作为值传递时，理论上比裸指针需要更多的 CPU 指令，从而引入了轻微的性能开销。

> 最佳实践： 尽管有理论开销，但在现代 C++ 编程中，出于安全和资源管理（RAII）的考虑，仍然强烈推荐使用 `std::unique_ptr`。为了最小化开销，当将 `std::unique_ptr` 传递给函数时，应始终通过引用 (`T&`) 或仅传递裸指针 (`T*`) 来避免触发非平凡的移动/复制语义。

】

#### 共享指针 (Shared pointer)

共享所有权意味着一个对象可以有多个所有者。当最后一个所有者不复存在时，该对象将被删除。这是一种非常有用的指针类型，但也比独占指针更复杂。

`std::shared_ptr` 对象使用引用计数来跟踪对象拥有的所有者数量。当计数器达到 0 时，对象将被删除。计数器需要存储在某个地方，因此与独占指针相比，它确实有一些内存开销。此外，`std::shared_ptr` 在内部是线程安全的，因此计数器需要原子地更新以防止竞争条件。

创建由共享指针拥有的对象的推荐方法是使用 `std::make_shared<T>()`。它比手动使用 `new` 创建对象然后将其传递给 `std::shared_ptr` 构造函数更安全（从异常安全角度来看）和更高效。通过再次重载 `operator new()` 和 `operator delete()` 来跟踪分配，我们可以进行一个实验来找出为什么使用 `std::make_shared<T>()` 更高效：

```cpp
auto operator new(size_t size) -> void* {
    void* p = std::malloc(size);
    std::cout << "allocated " << size << " byte(s)" << '\n';
    return p;
}

auto operator delete(void* p) noexcept -> void {
    std::cout << "deleted memory\n";
    return std::free(p);
}
```

现在，让我们首先尝试推荐的方法，使用 `std::make_shared()`：

```cpp
int main() {
    auto i = std::make_shared<double>(42.0);
    return 0;
}
```

运行程序时的输出如下：

```
allocated 32 bytes
deleted memory
```

现在，让我们通过使用 `new` 显式分配 `int` 值，然后将其传递给 `std::shared_ptr` 构造函数：

```cpp
int main() {
    auto i = std::shared_ptr<double>{new double{42.0}};
    return 0;
}
```

程序将生成以下输出：

```
allocated 4 bytes
allocated 32 bytes
deleted memory
deleted memory
```

我们可以得出结论，第二种版本需要两次分配，一次用于 `double`，一次用于 `std::shared_ptr`，而第一种版本只需要一次分配。这也意味着，通过使用 `std::make_shared()`，我们的代码将由于空间局部性而更具缓存友好性。

#### 弱指针 (Weak pointer)

弱所有权不会保持任何对象存活；它只允许我们在别人拥有该对象时使用它。为什么你需要像弱所有权这样模糊的所有权呢？使用弱指针的一个常见原因是打破引用循环。当两个或多个对象使用共享指针相互引用时，就会发生引用循环。即使所有外部 `std::shared_ptr` 构造函数都消失了，对象也会因为相互引用而保持存活。

为什么不直接使用裸指针呢？弱指针不正是裸指针已经具备的功能吗？根本不是。弱指针是安全使用的，因为除非对象实际存在，否则我们无法引用它，而悬空裸指针的情况并非如此。一个示例将澄清这一点：

```cpp
auto i = std::make_shared<int>(10);
auto weak_i = std::weak_ptr<int>{i};

// 也许 i.reset() 发生在这里，导致 int 被删除...

if (auto shared_i = weak_i.lock()) {
    // 我们设法将弱指针转换为共享指针
    std::cout << *shared_i << '\n';
}
else {
    std::cout << "weak_i 已过期，shared_ptr 为 nullptr\n";
}
```

每当我们尝试使用弱指针时，我们需要首先使用成员函数 `lock()` 将其转换为共享指针。如果对象尚未过期，则共享指针将是该对象的有效指针；否则，我们将获得一个空的 `std::shared_ptr`。这样，我们在使用 `std::weak_ptr` 而非裸指针时，就可以避免悬空指针。

这将结束我们关于内存中对象的这一节。C++ 为处理内存提供了出色的支持，既包括对齐和填充等底层概念，也包括对象所有权等高层概念。

对所有权、RAII 和引用计数有深入的理解，对于使用 C++ 工作非常重要。刚接触 C++ 且之前未接触过这些概念的程序员可能需要一些时间才能完全掌握。同时，这些概念并非 C++ 所独有。在大多数语言中，它们更加分散，但在其他语言中则更为突出（Rust 是后者的一个例子）。因此，一旦掌握，它也会提高你在其他语言中的编程技能。思考对象所有权将对你编写的程序设计和架构产生积极影响。

现在，我们将转向一种优化技术，该技术将减少动态内存分配的使用，并尽可能使用栈。

## 小对象优化 (Small object optimization)

像 `std::vector` 这样的容器的一个巨大优势在于，它们在需要时会自动分配动态内存。然而，有时候，对于那些只包含少量元素的容器对象使用动态内存可能会损害性能。更高效的做法是将元素保留在容器本身内部，只使用栈内存，而不是在堆上分配小的内存区域。

大多数现代的 `std::string` 实现都利用了一个事实：在正常的程序中，很多字符串都很短，而且处理短字符串时不使用堆内存会更高效。

一种替代方案是在 `string` 类本身中保留一个小的独立缓冲区，当字符串内容较短时可以使用它。但这会增加 `string` 类的大小，即使不使用这个短缓冲区也是如此。

因此，一种更节省内存的解决方案是使用 联合体 (union)，当字符串处于短模式时，它可以容纳一个短缓冲区；否则，它容纳处理动态分配缓冲区所需的数据成员。这种用于优化容器以处理小数据的技术，通常对于字符串被称为小字符串优化 (Small String Optimization, SSO)，而对于其他类型则被称为小对象优化 (Small Object Optimization) 和小缓冲区优化 (Small Buffer Optimization)。我们为我们所爱的事物起了很多名字。

一个简短的代码示例将演示 LLVM 的 libc++ 中的 `std::string` 在我的 64 位系统上的行为：

```cpp
auto allocated = size_t{0};

// 重载 operator new 和 delete 以跟踪分配
void* operator new(size_t size) {
    void* p = std::malloc(size);
    allocated += size;
    return p;
}

void operator delete(void* p) noexcept {
    return std::free(p);
}

int main() {
    allocated = 0;
    auto s = std::string{""}; // 尝试不同的字符串大小
    std::cout << "stack space = " << sizeof(s)
              << ", heap space = " << allocated
              << ", capacity = " << s.capacity() << '\n';
}
```

代码首先重载了全局的 `operator new` 和 `operator delete`，目的是跟踪动态内存分配。现在我们可以开始测试不同大小的字符串 `s`，看看 `std::string` 的行为。在我的系统上以发布模式构建并运行前面的示例，它生成了以下输出：

```
stack space = 24, heap space = 0, capacity = 22
```

这个输出告诉我们，`std::string` 在栈上占用 24 字节，并且它在不使用任何堆内存的情况下拥有 22 个字符的容量。让我们通过用一个 22 个字符的字符串替换空字符串来验证这是否属实：

```cpp
auto s = std::string{"1234567890123456789012"};
```

程序仍然产生相同的输出，并验证了没有分配动态内存。但是，当我们改为将字符串增加到包含 23 个字符时会发生什么呢？

```cpp
auto s = std::string{"12345678901234567890123"};
```

现在运行程序会产生以下输出：

```
stack space = 24, heap space = 32, capacity = 31
```

`std::string` 类现在被迫使用堆来存储字符串。它分配了 32 字节，并报告容量是 31。这是因为 libc++ 在内部总是存储一个空终止字符串，因此末尾需要一个额外的字节用于空字符。尽管如此，`string` 类只有 24 字节，却能容纳 22 个字符长度的字符串而无需分配任何内存，这仍然是相当了不起的。它是如何做到的呢？

正如前面提到的，通常通过使用具有两种不同布局的联合体来节省内存：一种用于短模式 (short mode)，一种用于长模式 (long mode)。在真实的 libc++ 实现中，有很多巧妙的设计来最大限度地利用这 24 字节的可用空间。这里的代码经过简化，旨在演示这个概念。

长模式 (Long Mode) 的布局如下所示：

```cpp
struct Long {
    size_t capacity_{};
    size_t size_{};
    char* data_{};
};
```

长布局中的每个成员都是 8 字节，因此总大小是 24 字节。`char` 指针 `data_` 是指向将保存长字符串的动态分配内存的指针。

短模式 (Short Mode) 的布局看起来像这样：

```cpp
struct Short {
    unsigned char size_{};
    char data_[23]{};
};
```

在短模式中，不需要使用变量来表示容量，因为它是一个编译时常量。也可以在这个布局中使用一个更小的类型作为 `size_` 数据成员，因为我们知道如果它是一个短字符串，字符串的长度只能在 0 到 22 之间。

这两种布局通过一个联合体结合起来：

```cpp
union u_ {
    Short short_layout_;
    Long long_layout_;
};
```

然而，还缺少一个部分：`string` 类如何知道它当前存储的是短字符串还是长字符串？需要一个标志来指示这一点，但它存储在哪里呢？

结果是，libc++ 使用长模式中 `capacity_` 数据成员的最低有效位，以及短模式中 `size_` 数据成员的最低有效位。对于长模式，这个位无论如何都是冗余的，因为字符串总是分配 2 的倍数的内存大小。在短模式中，可以只使用 7 个位来存储大小，以便一个位可以用于标志。

当编写这段代码来处理大端字节序时，情况会变得更加复杂，因为无论我们使用的是联合体的 `Short` 结构体还是 `Long` 结构体，该位都需要放置在内存中的相同位置。你可以在 libc++ 的实现中查找详细信息：`https://github.com/llvm/llvm-project/tree/master/libcxx`。

图 7.9 总结了我们简化的（但仍然相当复杂的）高效小字符串优化实现所使用的联合体的内存布局：

<a>![](/img/chp/ch7/9.png)</a>

像这样的巧妙技巧，就是你应该努力使用标准库提供的高效且经过良好测试的类，而不是尝试自己实现的原因。尽管如此，了解这些优化以及它们的工作原理非常重要且有用，即使你永远不需要自己编写一个。


## 自定义内存管理 (Custom memory management)

本章我们已经进行了很长时间的学习。我们涵盖了虚拟内存、栈和堆的基础知识、`new` 和 `delete` 表达式、内存所有权以及对齐和填充。但在结束本章之前，我们将展示如何在 C++ 中自定义内存管理。我们将看到本章前面介绍过的那些知识点，在编写自定义内存分配器时是如何派上用场的。

但首先，什么是自定义内存管理器，我们为什么需要它？

当使用 `new` 或 `malloc()` 分配内存时，我们使用的是 C++ 内置的内存管理系统。`operator new` 的大多数实现都会调用 `malloc()`，这是一个通用内存分配器。设计和构建一个通用内存管理器是一项复杂的任务，许多人已经在这个课题上花费了大量时间进行研究。尽管如此，你可能仍希望编写一个自定义内存管理器，原因有很多。以下是一些示例：

  * 调试和诊断： 我们在本章中已经通过重载 `operator new` 和 `operator delete` 来打印一些调试信息，以此进行过几次这样的操作。
  * 沙盒： 自定义内存管理器可以为不允许分配不受限制内存的代码提供一个沙盒。沙盒还可以跟踪内存分配，并在沙盒代码执行完成后释放内存。
  * 性能： 如果我们需要动态内存且无法避免分配，我们可能需要编写一个自定义内存管理器，以更好地满足我们的特定需求。稍后，我们将介绍一些可以利用来超越 `malloc()` 性能的情形。

话虽如此，许多经验丰富的 C++ 程序员从未遇到过真正需要他们自定义系统自带的标准内存管理器的问题。这很好地说明了当今通用内存管理器有多么优秀，尽管它们必须在对我们的特定用例一无所知的情况下满足所有要求。我们对应用程序中的内存使用模式了解得越多，我们编写出比 `malloc()` 更高效的东西的机会就越大。例如，还记得栈吗？由于栈不需要处理多线程，并且保证了释放总是按逆序发生，因此从栈分配和释放内存比从堆分配和释放内存要快得多。

构建自定义内存管理器通常从分析精确的内存使用模式开始，然后实现一个内存竞技场（arena）。


### 构建内存竞技场 (Building an arena)

在处理内存分配器时，两个经常使用的术语是 arena（竞技场） 和 memory pool（内存池）。在本书中，我们不会区分这些术语。我所说的 arena 是指一块连续的内存块，包括一套分发和回收该内存块中部分内存的策略。

竞技场在技术上也可以被称为内存资源（memory resource）或分配器（allocator），但这些术语将用于指代标准库中的抽象。我们稍后将开发的自定义分配器将使用我们在这里创建的竞技场来实现。

在设计竞技场时，可以使用一些通用策略，这些策略可能会使分配和释放的性能优于 `malloc()` 和 `free()`：

  * 单线程： 如果我们知道一个竞技场只会被一个线程使用，就不需要用同步原语（如锁或原子操作）来保护数据。客户使用竞技场时不会有被其他线程阻塞的风险，这在实时环境中很重要。
  * 固定大小分配： 如果竞技场只分发固定大小的内存块，就可以通过使用空闲列表（free list）有效地回收内存，而不会产生内存碎片。
  * 有限生命周期： 如果你知道从竞技场分配的对象只需要在有限且明确定义的生命周期内存活，那么竞技场可以推迟回收并一次性释放所有内存。一个例子是服务器应用程序在处理请求期间创建的对象。当请求完成后，可以在一个步骤中回收在请求期间分发的所有内存。当然，竞技场需要足够大，能够处理请求期间的所有分配而无需连续回收内存；否则，此策略将不起作用。

我不会进一步深入探讨这些策略的细节，但了解在寻找改进程序中内存管理的方法时存在的可能性是很好的。就像优化软件通常的情况一样，关键在于理解程序运行的环境并分析特定的内存使用模式。我们这样做是为了找到与通用内存管理器相比，改进自定义内存管理器的方法。

接下来，我们将看一个简单的竞技场类模板，它可用于需要动态存储期，但所需内存通常很小可以放在栈上的小对象或少量对象。这段代码基于 Howard Hinnant 的 `short_alloc`，发布于 *[https://howardhinnant.github.io/stack\_alloc.html](https://howardhinnant.github.io/stack_alloc.html)*。如果你想更深入地研究自定义内存管理，这是一个很好的起点。我认为它是一个很好的演示示例，因为它能够处理多种大小的对象，这需要正确处理对齐。

但请再次记住，这是一个用于演示概念的简化版本，而不是为你提供可用于生产环境的代码：

```cpp
template <size_t N> class Arena {

  static constexpr size_t alignment = alignof(std::max_align_t);

public:
  Arena() noexcept : ptr_(buffer_) {}
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  auto reset() noexcept { ptr_ = buffer_; }
  static constexpr auto size() noexcept { return N; }
  auto used() const noexcept { return static_cast<size_t>(ptr_ - buffer_); }
  auto allocate(size_t n) -> std::byte*;
  auto deallocate(std::byte* p, size_t n) noexcept -> void;

private:
  static auto align_up(size_t n) noexcept -> size_t {
    return (n + (alignment - 1)) & ~(alignment - 1);
  }
  auto pointer_in_buffer(const std::byte* p) const noexcept -> bool {
    return std::uintptr_t(buffer_) <= std::uintptr_t(p) &&
           std::uintptr_t(p) < std::uintptr_t(buffer_) + N;
  }
  alignas(alignment) std::byte buffer_[N];
  std::byte* ptr_{};
};
```

竞技场包含一个 `std::byte` 缓冲区，其大小在编译时确定。这使得可以在栈上或作为具有静态或线程局部存储期的变量来创建竞技场对象。缓冲区可能是在栈上分配的；因此，除非我们对数组应用 `alignas` 说明符，否则不能保证它会为 `char` 以外的类型对齐。

如果你不习惯位操作，辅助函数 `align_up()` 可能看起来很复杂。然而，它基本上只是向上舍入到我们使用的对齐要求。此版本分发出的内存将与使用 `malloc()` 时相同，因为它适用于任何类型。如果我们将竞技场用于具有较小对齐要求的较小类型，这会有点浪费，但我们在这里忽略这一点。

【译注：

这段话解释了一个 C++ 中常见的位操作技巧，用于实现向上对齐（Round Up to Alignment）的功能。

**`align_up` 函数解释**

这个 `align_up` 函数的目的是接收一个字节大小 `n`，并将其向上调整到下一个满足特定对齐要求 (`alignment`) 的边界。

```cpp
static auto align_up(size_t n) noexcept -> size_t {
    return (n + (alignment-1)) & ~(alignment-1);
}
```

**1. 核心目标：向上取整**

假设 `alignment` 是 $A$（例如 8 或 16）。函数的目标是计算：

$$
\text{aligned_n} = \lceil \frac{n}{A} \rceil \times A
$$

**2. 位操作原理（假设 $A$ 是 2 的幂）**

这种位操作技巧只在对齐值 $A$ 是 2 的幂（例如 $2^3=8, 2^4=16$）时有效，这在内存管理中是常态。

**第一步：加法偏置 (`n + (alignment - 1)`)**

  * `alignment - 1` 是一个掩码（Mask），其二进制形式全部是 1，直到达到对齐边界。
      * 示例 (A=8)： $A-1 = 7$。
      * 目的： 确保 $n$ 只要超过前一个对齐边界一点点，就会跨越到下一个对齐块。
          * 如果 $n=8$：$8 + 7 = 15$
          * 如果 $n=9$：$9 + 7 = 16$
          * 如果 $n=15$：$15 + 7 = 22$
          * 如果 $n=16$：$16 + 7 = 23$

**第二步：按位取反和按位与 (`& ~(alignment - 1)`)**

  * `~(alignment - 1)` 是一个特殊的掩码，它清除了数字的最低几位。
      * 示例 (A=8)： $A-1 = 7$ (二进制 `0111`)。$\sim 7$ (二进制 `...1000`)。
      * 目的： 将第一步结果的最低位强制归零，从而保证结果是 $A$ 的倍数。
          * 15 (`...1111`) & $\sim 7$ (`...1000`) $\rightarrow$ 8
          * 16 (`...0000`) & $\sim 7$ (`...1000`) $\rightarrow$ 16
          * 22 (`...0110`) & $\sim 7$ (`...1000`) $\rightarrow$ 16
          * 23 (`...0111`) & $\sim 7$ (`...1000`) $\rightarrow$ 16

**3. 代码的实际含义**

这段代码确保了 `allocate` 函数返回的内存指针将满足指定的对齐要求。

  * `std::max_align_t`： 示例代码中的 `alignment` 被设置为 `alignof(std::max_align_t)`。这意味着分配的内存块将适用于任何 C++ 基本类型，包括 `double` 或 `long long`。
  * 浪费问题： 如果竞技场只存储 `char` 或 `int`（它们通常只需要 4 字节对齐），而你使用了 8 字节或 16 字节的对齐要求，那么你可能浪费了缓冲区中的一些空间，但这保证了分配器的通用性。

】

在回收内存时，我们需要知道被要求回收的指针是否确实属于我们的竞技场。`pointer_in_buffer()` 函数通过将指针地址与竞技场的地址范围进行比较来检查这一点。顺便提一下，对不相交对象的裸指针进行关系比较是未定义行为；优化编译器可能会利用这一点并导致出乎意料的结果。为了避免这种情况，我们在比较地址之前将指针强制转换为 `std::uintptr_t`。如果你对这背后的细节感到好奇，可以在 Raymond Chen 的文章 *How to check if a pointer is in range of memory* 中找到详尽的解释，网址是 *[https://devblogs.microsoft.com/oldnewthing/20170927-00/?p=97095](https://devblogs.microsoft.com/oldnewthing/20170927-00/?p=97095)*。

接下来，我们需要实现 `allocate` 和 `deallocate`：

```cpp
template <size_t N> auto Arena<N>::allocate(size_t n) -> std::byte* {
  const auto aligned_n = align_up(n);
  const auto available_bytes =
      static_cast<decltype(aligned_n)>(buffer_ + N - ptr_);
  if (available_bytes >= aligned_n) {
    auto* r = ptr_;
    ptr_ += aligned_n;
    return r;
  }
  return static_cast<std::byte*>(::operator new(n));
}
```

`allocate()` 函数返回一个指向正确对齐、具有指定大小 `n` 的内存的指针。如果缓冲区中没有足够的可用空间来满足请求的大小，它将回退到使用 `operator new`。

下面的 `deallocate()` 函数首先检查要释放的内存的指针是来自缓冲区，还是用 `operator new` 分配的。如果它不是来自缓冲区，我们简单地用 `operator delete` 删除它。否则，我们检查要释放的内存是否是我们从缓冲区中分发出去的最后一块内存，然后通过移动当前的 `ptr_` 来回收它，就像栈所做的那样。我们简单地忽略其他回收内存的尝试：

```cpp
template <size_t N>
auto Arena<N>::deallocate(std::byte* p, size_t n) noexcept -> void {
  if (pointer_in_buffer(p)) {
    n = align_up(n);
    if (p + n == ptr_) {
      ptr_ = p;
    }
  } else {
    ::operator delete(p);
  }
}
```

差不多就是这样了；我们的竞技场现在可以使用了。让我们在分配 `User` 对象时使用它：

```cpp
auto user_arena = Arena<1024>{};

class User {
public:
    auto operator new(size_t size) -> void* {
        return user_arena.allocate(size);
    }
    auto operator delete(void* p) -> void {
        user_arena.deallocate(static_cast<std::byte*>(p), sizeof(User));
    }
    auto operator new[](size_t size) -> void* {
        return user_arena.allocate(size);
    }
    auto operator delete[](void* p, size_t size) -> void {
        user_arena.deallocate(static_cast<std::byte*>(p), size);
    }
private:
    int id_{};
};

int main() {
    // 创建用户时没有分配动态内存
    auto user1 = new User{};
    delete user1;
    auto users = new User[10];
    delete [] users;
    auto user2 = std::make_unique<User>();
    return 0;
}
```

在这个示例中创建的 `User` 对象都将驻留在 `user_area` 对象的缓冲区中。也就是说，当我们在这里调用 `new` 或 `make_unique()` 时，没有分配动态内存。但是，还有其他创建 C++ 中 `User` 对象的方法是这个示例没有展示的。我们将在下一节中介绍它们。

 

### 自定义内存分配器 (A custom memory allocator)

当我们尝试用一个特定类型来使用我们的自定义内存管理器时，它工作得非常好！但是，有一个问题。事实证明，类特有的 `operator new` 并不是在我们可能期望的所有情况下都被调用的。考虑以下代码：

```cpp
auto user = std::make_shared<User>();
```

当我们想要一个包含 10 个用户的 `std::vector` 时会发生什么？

```cpp
auto users = std::vector<User>{};
users.reserve(10);
```

在这两种情况下，我们的自定义内存管理器都没有被使用。为什么？

从共享指针开始，我们必须回到前面看到的示例，其中我们看到 `std::make_shared()` 实际上为引用计数数据和它应该指向的对象都分配了内存。`std::make_shared()` 没有办法只用一次分配就使用像 `new User()` 这样的表达式来创建用户对象和计数器。相反，它分配内存并使用 Placement New 来构造用户对象。

`std::vector` 对象也类似。当我们调用 `reserve()` 时，它不会默认在数组中构造 10 个对象。这需要用于 `vector` 的所有类都具有默认构造函数。相反，它分配内存，当对象被添加时，这些内存可用于容纳 10 个用户对象。同样，Placement New 是实现这一点的工具。

幸运的是，我们可以向 `std::vector` 和 `std::shared_ptr` 提供一个自定义内存分配器，以便让它们使用我们的自定义内存管理器。标准库中的其余容器也是如此。如果我们不提供自定义分配器，容器将使用默认的 `std::allocator<T>` 类。因此，为了使用我们的竞技场，我们需要编写一个可以被容器使用的分配器。

自定义分配器在 C++ 社区中一直是一个热门话题。许多自定义容器的实现是为了控制内存的管理方式，而不是使用带有自定义分配器的标准容器，这可能是有充分理由的。

然而，C++11 改进了编写自定义分配器的支持和要求，现在情况好多了。在这里，我们将只关注 C++11 及以后的分配器。

C++11 中最小的分配器现在看起来像这样：

```cpp
template<typename T>
struct Alloc {
    using value_type = T;
    Alloc();
    template<typename U> Alloc(const Alloc<U>&);
    T* allocate(size_t n);
    auto deallocate(T*, size_t) const noexcept -> void;
};

template<typename T>
auto operator==(const Alloc<T>&, const Alloc<T>&) -> bool;

template<typename T>
auto operator!=(const Alloc<T>&, const Alloc<T>&) -> bool;
```

由于 C++11 的改进，代码量真的不多了。使用分配器的容器实际上使用的是 `std::allocator_traits`，如果分配器省略了一些特性，它会提供合理的默认值。我建议你查看 `std::allocator_traits`，看看哪些特性可以配置以及它们的默认值是什么。

通过使用 `malloc()` 和 `free()`，我们可以很容易地实现一个最小的自定义分配器。在这里，我们将展示著名的 Mallocator，它最初由 Stephan T. Lavavej 在一篇博客文章中发布，演示如何使用 `malloc()` 和 `free()` 编写一个最小的自定义分配器。从那时起，它已针对 C++11 进行了更新，使其更加精简。它的样子如下：

```cpp
template <class T>
struct Mallocator {
    using value_type = T;
    Mallocator() = default;
    template <class U>
    Mallocator(const Mallocator<U>&) noexcept {}
    template <class U>
    auto operator==(const Mallocator<U>&) const noexcept {
        return true;
    }
    template <class U>
    auto operator!=(const Mallocator<U>&) const noexcept {
        return false;
    }
    auto allocate(size_t n) const -> T* {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length{};
        }
        void* const pv = malloc(n * sizeof(T));
        if (pv == nullptr) {
            throw std::bad_alloc{};
        }
        return static_cast<T*>(pv);
    }
    auto deallocate(T* p, size_t) const noexcept -> void {
        free(p);
    }
};
```

`Mallocator` 是一个无状态分配器（stateless allocator），这意味着分配器实例本身没有任何可变状态；相反，它使用全局函数进行分配和释放，即 `malloc()` 和 `free()`。无状态分配器应始终与相同类型的分配器比较相等。它表明用 `Mallocator` 分配的内存也应该用 `Mallocator` 释放，无论 `Mallocator` 实例是什么。无状态分配器是编写起来最不复杂的分配器，但它也受到限制，因为它依赖于全局状态。

为了将我们的竞技场用作栈分配对象，我们需要一个有状态分配器（stateful allocator），它可以引用竞技场实例。在这里，我们实现的竞技场类才真正开始有意义。例如，假设我们想在函数中使用标准容器之一来进行一些处理。我们知道，在大多数情况下，我们处理的数据量非常小，可以放在栈上。但一旦我们使用了标准库中的容器，它们就会从堆分配内存，在这种情况下会损害我们的性能。

使用栈来管理数据并避免不必要的堆分配的替代方案是什么？一种替代方案是构建一个自定义容器，它使用我们为 `std::string` 研究过的小对象优化的变体。

也可以使用 Boost 中的容器，例如 `boost::container::small_vector`，它基于 LLVM 的 `small vector`。我们建议你查看一下：*[http://www.boost.org/doc/libs/1\_74\_0/doc/html/container/non\_standard\_containers.html](https://www.google.com/search?q=http://www.boost.org/doc/libs/1_74_0/doc/html/container/non_standard_containers.html)*。

然而，另一种替代方案是使用自定义分配器，这也是我们接下来要探讨的。由于我们已经有一个现成的竞技场模板类，我们可以简单地在栈上创建一个竞技场实例，并让一个自定义分配器使用它进行分配。那么我们需要做的就是实现一个有状态分配器，它可以持有对栈分配的竞技场对象的引用。

同样，我们将要实现的这个自定义分配器是 Howard Hinnant 的 `short_alloc` 的简化版本：

```cpp
template <class T, size_t N>
struct ShortAlloc {
    using value_type = T;
    using arena_type = Arena<N>;
    ShortAlloc(const ShortAlloc&) = default;
    ShortAlloc& operator=(const ShortAlloc&) = default;
    ShortAlloc(arena_type& arena) noexcept : arena_{&arena} { }
    template <class U>
    ShortAlloc(const ShortAlloc<U, N>& other) noexcept
        : arena_{other.arena_} {}
    template <class U> struct rebind {
        using other = ShortAlloc<U, N>;
    };
    auto allocate(size_t n) -> T* {
        return reinterpret_cast<T*>(arena_->allocate(n*sizeof(T)));
    }
    auto deallocate(T* p, size_t n) noexcept -> void {
        arena_->deallocate(reinterpret_cast<std::byte*>(p), n*sizeof(T));
    }
    template <class U, size_t M>
    auto operator==(const ShortAlloc<U, M>& other) const noexcept {
        return N == M && arena_ == other.arena_;
    }
    template <class U, size_t M>
    auto operator!=(const ShortAlloc<U, M>& other) const noexcept {
        return !(*this == other);
    }
    template <class U, size_t M> friend struct ShortAlloc;
private:
    arena_type* arena_;
};
```

分配器持有对竞技场的引用。这是分配器拥有的唯一状态。`allocate()` 和 `deallocate()` 函数只是将它们的请求转发给竞技场。比较运算符确保 `ShortAlloc` 类型的两个实例使用的是同一个竞技场。

现在，我们实现的分配器和竞技场可以与标准容器一起使用，以避免动态内存分配。当我们处理小数据时，我们可以使用栈来处理所有的分配。让我们看一个使用 `std::set` 的示例：

```cpp
int main() {
    using SmallSet =
        std::set<int, std::less<int>, ShortAlloc<int, 512>>;
    auto stack_arena = SmallSet::allocator_type::arena_type{};
    auto unique_numbers = SmallSet{stack_arena};

    // 从标准输入读取数字
    auto n = int{};
    while (std::cin >> n)
        unique_numbers.insert(n);

    // 打印唯一的数字
    for (const auto& number : unique_numbers)
        std::cout << number << '\n';
}
```

该程序从标准输入读取整数，直到到达文件末尾（在类 Unix 系统上是 Ctrl + D，在 Windows 上是 Ctrl + Z）。然后，它按升序打印唯一的数字。根据从标准输入读取的数字数量，程序将使用栈内存或动态内存，这是通过我们的 `ShortAlloc` 分配器实现的。
 
 

## 使用多态内存分配器 (Using Polymorphic Memory Allocators)

如果你已经学习了本章，那么现在你知道如何实现一个可以与任意容器（包括标准库容器）一起使用的自定义分配器了。假设我们想将这个新的分配器用于代码库中处理 `std::vector<int>` 类型缓冲区的现有代码，像这样：

```cpp
void process(std::vector<int>& buffer) {
// ...
}

auto some_func() {
auto vec = std::vector<int>(64);
process(vec);
// ...
}
```

我们渴望尝试使用我们新的、利用栈内存的分配器，并试图像这样将其注入：

```cpp
using MyAlloc = ShortAlloc<int, 512>; // 我们的自定义分配器

auto some_func() {
auto arena = MyAlloc::arena_type();
auto vec = std::vector<int, MyAlloc>(64, arena);
process(vec);
// ...
}
```

在编译时，我们痛苦地意识到 `process()` 是一个期望接收 `std::vector<int>` 的函数，而我们的 `vec` 变量现在是另一种类型。GCC 报出以下错误：

```
error: invalid initialization of reference of type 'const
std::vector<int>&' from expression of type 'std::vector<int,
ShortAlloc<int, 512> >
```

类型不匹配的原因在于，我们想要使用的自定义分配器 `MyAlloc` 被作为模板参数传递给了 `std::vector`，因此它成为了我们实例化类型的一部分。结果就是，`std::vector<int>` 和 `std::vector<int, MyAlloc>` 无法互换。

这对你正在处理的用例来说可能是个问题，也可能不是。你可以通过让 `process()` 函数接受一个 `std::span` 或者使其成为一个使用 Range 而不是要求 `std::vector` 的泛型函数来解决它。无论如何，重要的是要认识到：当使用标准库中分配器感知（allocator-aware）的模板类时，分配器实际上成为了类型的一部分。

 

那么，`std::vector<int>` 使用的是什么分配器呢？答案是 `std::vector<int>` 使用了默认模板参数，即 `std::allocator`。所以，编写 `std::vector<int>` 等价于 `std::vector<int, std::allocator<int>>`。模板类 `std::allocator` 是一个空类，它在执行容器的分配和释放请求时，使用的是全局的 `new` 和全局的 `delete`。

这也意味着使用空分配器的容器的大小要小于使用我们自定义分配器的容器：

```cpp
std::cout << sizeof(std::vector<int>) << '\n';
// 可能的输出: 24
std::cout << sizeof(std::vector<int, MyAlloc>) << '\n';
// 可能的输出: 32
```

检查 `libc++` 中 `std::vector` 的实现，我们可以看到它使用了一种精妙的类型，称为 compressed pair（压缩对），它反过来是基于空基类优化（empty base-class optimization）来消除通常由空类成员占用的不必要存储空间的。我们在这里不涵盖细节，但如果你感兴趣，可以查看 Boost 版本的 `compressed_pair`。

 

这种由于使用不同分配器而导致类型不同的问题在 C++17 中得到了解决，它引入了一个额外的间接层（layer of indirection）。

在 `std::pmr` 命名空间下的所有标准容器都使用相同的分配器，即 `std::pmr::polymorphic_allocator`，它将所有的分配/释放请求转发给一个内存资源（memory resource）类。

因此，我们不必编写新的自定义内存分配器，而是可以使用名为 `std::pmr::polymorphic_allocator` 的通用多态内存分配器，并转而编写新的自定义内存资源，这些资源将在构造时被传递给多态分配器。内存资源类似于我们前面的 `Arena` 类，而 `polymorphic_allocator` 则是那个额外的间接层，其中包含一个指向该资源的指针。

下图展示了控制流：`vector` 委托给它的分配器实例，而分配器又委托给它所指向的内存资源：

<a>![](/img/chp/ch7/10.png)</a>

要开始使用多态分配器，我们需要将命名空间从 `std` 更改为 `std::pmr`：

```cpp
auto v1 = std::vector<int>{};             // 使用 std::allocator
auto v2 = std::pmr::vector<int>{/*...*/}; // 使用 polymorphic_allocator
```

编写自定义内存资源相对简单，尤其是有了内存分配器和竞技场（Arena）的知识。但是，我们甚至可能不必编写自定义内存资源就能实现我们的目标。C++ 已经为我们提供了一些有用的实现，在编写自己的资源之前应该考虑使用它们。

 

所有的内存资源都派生自基类 `std::pmr::memory_resource`。以下内存资源位于 `<memory_resource>` 头文件中：

  * `std::pmr::monotonic_buffer_resource`：这与我们的 `Arena` 类非常相似。它适用于创建许多生命周期较短的对象的场景。内存仅在该 `monotonic_buffer_resource` 实例被析构时才会被释放，这使得分配非常快。
  * `std::pmr::unsynchronized_pool_resource`：这使用内存池（也称为 "slabs"），其中包含固定大小的内存块，从而避免了每个池内的碎片化。每个池为特定大小的对象提供内存。如果你正在创建许多具有几种不同大小的对象，使用此类会很有益处。此内存资源不是线程安全的，不能从多个线程使用，除非你提供外部同步。
  * `std::pmr::synchronized_pool_resource`：这是 `unsynchronized_pool_resource` 的线程安全版本。

 

内存资源可以链接起来。在创建一个内存资源实例时，我们可以为其提供一个上游内存资源（upstream memory resource）。当当前资源无法处理请求时（类似于我们在 `ShortAlloc` 中在小缓冲区满时使用 `malloc()` 的做法），或当资源本身需要分配内存时（例如 `monotonic_buffer_resource` 需要分配它的下一个缓冲区时），就会使用这个上游资源。

`<memory_resource>` 头文件提供了返回全局资源对象的指针的自由函数，这些函数在指定上游资源时非常有用：

  * `std::pmr::new_delete_resource()`：使用全局的 `operator new` 和 `operator delete`。
  * `std::pmr::null_memory_resource()`：当被要求分配内存时，总是抛出 `std::bad_alloc` 的资源。
  * `std::pmr::get_default_resource()`：返回全局默认内存资源，该资源可以通过 `set_default_resource()` 在运行时设置。最初的默认资源是 `new_delete_resource()`。

让我们看看如何重写上一节的示例，但这次使用 `std::pmr::set`：

```cpp
int main() {
    auto buffer = std::array<std::byte, 512>{};
    auto resource = std::pmr::monotonic_buffer_resource{
        buffer.data(), buffer.size(), std::pmr::new_delete_resource()};
    auto unique_numbers = std::pmr::set<int>{&resource};
    auto n = int{};
    while (std::cin >> n) {
        unique_numbers.insert(n);
    }
    for (const auto& number : unique_numbers) {
        std::cout << number << '\n';
    }
}
```

我们向内存资源传递了一个栈上分配的缓冲区，然后为它提供了 `new_delete_resource()` 返回的对象作为上游资源，以便在缓冲区满时使用。如果我们省略上游资源，它将使用默认内存资源，在本例中结果是一样的，因为我们的代码没有更改默认内存资源。


 

##  实现自定义内存资源 (Implementing a Custom Memory Resource)

实现一个自定义内存资源相当简单。我们需要公有继承自 `std::pmr::memory_resource`，然后实现三个纯虚函数，这些函数将被基类（`std::pmr::memory_resource`）调用。

让我们实现一个简单的内存资源，它会打印分配和释放信息，然后将请求转发给默认内存资源：

```cpp
class PrintingResource : public std::pmr::memory_resource {
public:
    // 构造函数中保存默认资源
    PrintingResource() : res_{std::pmr::get_default_resource()} {}
private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        std::cout << "allocate: " << bytes << '\n';
        return res_->allocate(bytes, alignment);
    }

    void do_deallocate(void* p, std::size_t bytes,
                       std::size_t alignment) override {
        std::cout << "deallocate: " << bytes << '\n';
        return res_->deallocate(p, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other)
    const noexcept override {
        return (this == &other);
    }

    std::pmr::memory_resource* res_; // 默认资源指针
};
```

请注意，我们在构造函数中保存了默认资源，而不是直接在 `do_allocate()` 和 `do_deallocate()` 中调用 `get_default_resource()`。这样做的原因是，在分配和释放之间，有人可能通过调用 `set_default_resource()` 更改了默认资源。

我们可以使用自定义内存资源来跟踪由 `std::pmr` 容器所做的分配。以下是使用 `std::pmr::vector` 的示例：

```cpp
auto res = PrintingResource{};
auto vec = std::pmr::vector<int>{&res};
vec.emplace_back(1);
vec.emplace_back(2);
```

运行程序时可能得到的输出是：

```
allocate: 4
allocate: 8
deallocate: 4
deallocate: 8
```
 

使用多态分配器时需要非常小心的一点是，我们传递的是指向内存资源的裸指针（raw non-owning pointers）。这并非多态分配器独有的问题；我们在 `Arena` 类和 `ShortAlloc` 中也遇到过类似的问题，但当使用 `std::pmr` 容器时，这个问题更容易被遗忘，因为这些容器都使用相同的分配器类型。

请考虑以下示例：

```cpp
auto create_vec() -> std::pmr::vector<int> {
    auto resource = PrintingResource{};
    auto vec = std::pmr::vector<int>{&resource}; // 裸指针
    return vec; // 噢！resource 在这里被销毁了
} // resource 的生命周期结束

auto vec = create_vec();
vec.emplace_back(1); // 未定义行为 (Undefined behavior)
```

由于 `resource` 在 `create_vec()` 函数结束时超出作用域而被销毁，我们新创建的 `std::pmr::vector` 变得毫无用处，并且在使用时极有可能导致程序崩溃。

 

##  章节总结与建议

这总结了我们关于自定义内存管理的部分。这是一个复杂的主题，如果你想通过使用自定义内存分配器来提高性能，我建议你仔细测量和分析应用程序中的内存访问模式，然后再使用和/或实现自定义分配器。

通常，应用程序中只有一小部分类或对象真正需要通过自定义分配器进行调整。同时，减少应用程序中的动态内存分配数量或将对象分组到特定的内存区域中，可以对性能产生显著影响。

在你开始实现自己的容器或自定义内存分配器之前，请记住，在你之前很多人可能遇到过非常相似的内存问题。因此，很有可能适合你的工具已经在某个库中存在了。构建快速、安全、稳健的自定义内存管理器是一项挑战。

在下一章中，你将学习如何从 C++ 中新引入的 Concepts 特性中受益，以及我们如何使用模板元编程让编译器为我们生成代码。
