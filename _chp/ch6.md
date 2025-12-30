---
layout:     post
title:      "第六章：范围与视图"
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
 
  
本章将从上一章关于算法及其局限性的内容继续。Ranges 库中的 Views（视图）是对 Algorithm 库的强大补充，它允许我们将多个转换操作组合成对元素序列的惰性求值视图。阅读本章后，你将理解什么是范围视图，以及如何将它们与标准库中的容器、迭代器和算法结合使用。

具体来说，我们将涵盖以下主要主题：

  * 算法的可组合性
  * 范围适配器 (Range adaptors)
  * 将视图具化 (Materializing) 为容器
  * 在范围中生成、转换和采样元素

在我们深入了解 Ranges 库本身之前，我们先讨论一下为什么它被添加到 C++20 中，以及为什么我们想要使用它。



## Ranges 库的动机

随着 Ranges 库引入 C++20，我们在实现算法时从标准库中受益的方式有了一些重大改进。以下列表展示了新特性：

  * 定义了对迭代器和范围需求的 Concepts（概念），现在可以被编译器更好地检查，并在开发过程中提供更多帮助。
  * `<algorithm>` 头文件中的所有函数都有了新的重载，它们受到上述 Concepts 的约束，并接受范围作为参数，而不是迭代器对。
  * 迭代器头文件中的受约束迭代器 (Constrained iterators)。
  * 范围视图 (Range views)，使得算法的可组合成为可能。

本章将重点讨论最后一点：视图 (views) 的概念，它允许我们组合算法以避免不必要的数据复制到拥有的容器中。为了充分理解其重要性，我们首先通过演示 Algorithm 库中缺乏可组合性来开始。


### Algorithm 库的局限性 


标准库算法在一个基本方面有所欠缺：可组合性。让我们通过回顾 第 5 章，《算法》 中的最后一个例子来解释这意味着什么，我们在那里简要讨论了这一点。如果你还记得，我们有一个结构体来表示特定年份和特定考试分数的 `Student`：

```cpp
struct Student {
    int year_{};
    int score_{};
    std::string name_{};
    // ...
};
```

如果我们要从一大批学生中找到二年级学生的最高分数，我们可能会对 `score_` 使用 `max_element()`，但由于我们只想考虑特定年份的学生，这就变得棘手了。通过使用接受范围和投影 (projections) 的新算法（参考 第 5 章，《算法》），我们可能会得到这样的代码：

```cpp
auto get_max_score(const std::vector<Student>& students, int year) {
    auto by_year = [=](const auto& s) { return s.year_ == year; };
    // The student list needs to be copied in
    // order to filter on the year
    auto v = std::vector<Student>{};
    std::ranges::copy_if(students, std::back_inserter(v), by_year);
    auto it = std::ranges::max_element(v, std::less{}, &Student::score_);
    return it != v.end() ? it->score_ : 0;
}
```

这里是如何使用它的示例：

```cpp
auto students = std::vector<Student>{
    {3, 120, "Niki"},
    {2, 140, "Karo"},
    {3, 190, "Sirius"},
    {2, 110, "Rani"},
    // ...
};
auto score = get_max_score(students, 2);
std::cout << score << '\n';
// Prints 140
```

`get_max_score()` 的这个实现易于理解，但它在使用 `copy_if()` 和 `std::back_inserter()` 时创建了不必要的 `Student` 对象副本。

你现在可能会想，`get_max_score()` 可以写成一个简单的 for-循环，从而避免 `copy_if()` 带来的额外内存分配：

```cpp
auto get_max_score(const std::vector<Student>& students, int year) {
    auto max_score = 0;
    for (const auto& student : students) {
        if (student.year_ == year) {
            max_score = std::max(max_score, student.score_);
        }
    }
    return max_score;
}
```

虽然在这个小例子中很容易实现，但我们希望能够通过组合小的算法构建块来实现这个算法，而不是使用一个单一的 for-循环从头开始实现。

我们想要的是一种像使用算法一样可读的语法，但又能够在算法的每一步中避免构造新的容器。这就是 Ranges 库中的 views 发挥作用的地方。虽然 Ranges 库包含的不仅仅是 views，但与 Algorithm 库的主要区别在于它能够将本质上是不同类型迭代器的东西组合成一个惰性求值的范围。

如果使用 Ranges 库中的 views 来编写上述示例，它将如下所示：

```cpp
auto max_value(auto&& range) {
    const auto it = std::ranges::max_element(range);
    return it != range.end() ? *it : 0;
}

auto get_max_score(const std::vector<Student>& students, int year) {
    const auto by_year = [=](auto&& s) { return s.year_ == year; };
    return max_value(students
        | std::views::filter(by_year)
        | std::views::transform(&Student::score_));
}
```

现在我们又回到了使用算法的方式，因此可以避免可变变量、for-循环和 if-语句。在我们最初的例子中，用于存放特定年份学生的额外 `vector` 已经被消除了。相反，我们组合了一个范围视图，它表示所有经过 `by_year` 谓词过滤，然后转换为只暴露 `score` 的学生。该视图随后被传递给一个小的工具函数 `max_value()`，该函数使用 `max_element()` 算法来比较选定学生的分数，从而找到最大值。

这种通过链式组合算法，同时避免不必要的复制的方式，就是我们开始使用 Ranges 库中 views 的动力。

##  理解 Ranges 库中的 Views 

Views 在 Ranges 库中是对一个范围进行惰性求值 (lazy evaluated) 的迭代。从技术上讲，它们只是带有内置逻辑的迭代器，但在语法上，它们为许多常见操作提供了非常简洁愉快的表达方式。

以下是使用视图对向量中的每个数字进行平方的示例（通过迭代）：

```cpp
auto numbers = std::vector{1, 2, 3, 4};
auto square = [](auto v) { return v * v; };
auto squared_view = std::views::transform(numbers, square);

for (auto s : squared_view) { // square lambda 在此被调用
    std::cout << s << " ";
}
// Output: 1 4 9 16
```

变量 `squared_view` 不是一个包含了平方值的 `numbers` 向量的副本；它是一个用于 `numbers` 的代理对象 (proxy object)，只有一点细微差别——每次你访问一个元素时，`std::transform()` 函数都会被调用。这就是我们说视图是惰性求值的原因。

从外部来看，你仍然可以像迭代任何常规容器一样迭代 `squared_view`，因此可以执行像 `find()` 或 `count()` 这样的常规算法，但在内部，你没有创建另一个容器。

如果你想存储该范围，可以使用 `std::ranges::copy()` 将视图具化 (materialized) 到一个容器中。（这将在本章后面进行演示。）一旦视图被复制回一个容器，原始容器和转换后的容器之间就不再有任何依赖关系。
 
 

使用 Ranges，还可以创建一个过滤视图，其中只有范围的一部分是可见的。在这种情况下，迭代视图时，只有满足条件的元素是可见的：

```cpp
auto v = std::vector{4, 5, 6, 7, 6, 5, 4};
auto odd_view =
    std::views::filter(v, [](auto i){ return (i % 2) == 1; });

for (auto odd_number : odd_view) {
    std::cout << odd_number << " ";
}
// Output: 5 7 5
```

 
 

Ranges 库多功能性的另一个例子是它提供了创建视图的可能性，该视图可以迭代多个容器，就好像它们是一个单一的列表：

```cpp
auto list_of_lists = std::vector<std::vector<int>> {
    {1, 2},
    {3, 4, 5},
    {5},
    {4, 3, 2, 1}
};
auto flattened_view = std::views::join(list_of_lists);

for (auto v : flattened_view)
    std::cout << v << " ";
// Output: 1 2 3 4 5 5 4 3 2 1

auto max_value = *std::ranges::max_element(flattened_view);
// max_value is 5
```

现在我们已经简要地看了一些使用视图的例子，接下来让我们研究一下所有视图共有的要求和属性。
 
### 视图的可组合性

视图的全部威力来自于组合它们的能力。由于视图不复制实际数据，你可以在一个数据集上表达多个操作，而在内部，只需要迭代一次。为了理解视图是如何组合的，让我们看看最初的示例，但这次不使用管道操作符来组合视图，而是直接构造实际的视图类。代码如下：

```cpp
auto get_max_score(const std::vector<Student>& s, int year) {
    auto by_year = [=](const auto& s) { return s.year_ == year; };
    auto v1 = std::ranges::ref_view{s}; // 将容器包装成视图
    auto v2 = std::ranges::filter_view{v1, by_year};
    auto v3 = std::ranges::transform_view{v2, &Student::score_};
    auto it = std::ranges::max_element(v3);
    return it != v3.end() ? *it : 0;
}
```

我们首先创建了一个 `std::ranges::ref_view`，它是一个围绕容器的薄包装器 (thin wrapper)。在我们的例子中，它将向量 `s` 转换成一个复制成本很低的视图。我们需要这样做，因为我们的下一个视图 `std::ranges::filter_view` 要求它的第一个参数是一个视图。正如你所见，我们通过引用链中的前一个视图来组合我们的下一个视图。

这种可组合视图的链当然可以任意长。`max_element()` 算法不需要了解整个链条的任何信息；它只需要像对待一个普通容器一样迭代范围 `v3` 即可。以下图表是 max_element() 算法、视图和输入容器之间关系的简化视图：

<a>![](/img/chp/ch6/1.png)</a>

现在，这种组合视图的风格有点冗长，如果我们尝试去除中间变量 `v1` 和 `v2`，我们最终会得到类似这样的代码：

```cpp
using namespace std::ranges; // _view 类位于 std::ranges 命名空间
auto scores =
    transform_view{filter_view{ref_view{s}, by_year},
                   &Student::score_};
```

这种语法看起来可能不够优雅。通过摆脱中间变量，我们得到了一些即使对于训练有素的人来说也难以阅读的代码。我们还被迫从内到外阅读代码才能理解其依赖关系。

幸运的是，Ranges 库为我们提供了范围适配器 (range adaptors)，这是组合视图的首选方式。

 

### 范围视图伴随着范围适配器


正如你前面所见，Ranges 库还允许我们使用范围适配器和管道操作符来组合视图，以获得更加优雅的语法（你将在 第 10 章，《代理对象和惰性求值》 中学习更多关于在自己的代码中使用管道操作符的知识）。前面的代码示例可以使用范围适配器对象重写，代码如下：

```cpp
using namespace std::views; // 范围适配器位于 std::views 命名空间
auto scores = s | filter(by_year) | transform(&Student::score_);
```

能够从左到右阅读语句，而不是从内到外阅读，使代码更易于理解。如果你使用过 Unix shell，你可能对这种用于链式命令的表示法很熟悉。

Ranges 库中的每个视图都有一个对应的范围适配器对象，可以与管道操作符一起使用。使用范围适配器时，我们还可以跳过额外的 `std::ranges::ref_view`，因为范围适配器直接作用于 viewable\_ranges（即可安全转换为视图的范围）。

你可以将范围适配器视为一个全局无状态对象，它实现了两个函数：`operator()()` 和 `operator|()`。这两个函数都构造并返回视图对象。在前面的示例中使用的就是管道操作符。但也可以使用调用操作符（Call Operator）通过带括号的嵌套语法来形成视图，如下所示：

```cpp
using namespace std::views;
auto scores = transform(filter(s, by_year), &Student::score_);
```

同样，使用范围适配器时，不需要将输入容器包装在 `ref_view` 中。

总结一下，Ranges 库中的每个视图都由以下部分组成：

  * 一个类模板（实际的视图类型），作用于视图对象，例如 `std::ranges::transform_view`。这些视图类型可以在 `std::ranges` 命名空间下找到。
  * 一个范围适配器对象，用于从范围创建视图类的实例，例如 `std::views::transform`。所有范围适配器都实现了 `operator()()` 和 `operator|()`，使得可以使用管道操作符或嵌套方式组合转换。范围适配器对象位于 `std::views` 命名空间下。


### 视图是非拥有的范围且具有复杂性保证

在上一章中介绍了范围 (range) 的概念。任何提供 `begin()` 和 `end()` 函数的类型，其中 `begin()` 返回一个迭代器，`end()` 返回一个哨兵 (sentinel)，都符合范围的条件。我们得出结论，所有标准容器都是范围。容器拥有它们的元素，因此我们可以称它们为拥有的范围 (owning ranges)。

视图也是一个范围，即它提供 `begin()` 和 `end()` 函数。然而，与容器不同，视图不拥有它所跨越的范围内的元素。

视图的构造被要求是 O(1) 的常数时间操作。它不能执行任何依赖于底层容器大小的工作。视图的赋值、复制、移动和析构也是如此。这使得在使用视图组合多个算法时，很容易对性能做出推断。这也使得视图不可能拥有元素，因为那样将在构造和析构时需要线性时间复杂度。

### 视图不会改变底层容器

乍一看，视图可能看起来像输入容器的一个变异版本 (mutated version)。然而，容器根本没有被改变：所有的处理都在迭代器中执行。视图仅仅是一个代理对象，当被迭代时，看起来像是一个变异的容器。

这也使得视图可以暴露与输入元素类型不同的元素类型。以下代码片段演示了视图如何将元素类型从 `int` 转换为 `std::string`：

```cpp
auto ints = std::list{2, 3, 4, 2, 1};
auto strings = ints
    | std::views::transform([](auto i) { return std::to_string(i); });
```

也许我们有一个对容器进行操作的函数，我们想使用范围算法对其进行转换，然后将其返回并存储回一个容器中。例如，在上面的例子中，我们可能希望实际将这些字符串存储在一个单独的容器中。你将在下一节学习如何实现这一点。


### 视图可以具化为容器

有时，我们希望将视图存储在容器中，即具化 (materialize) 视图。

所有视图都可以具化为容器，但这并不像你希望的那么简单。一个名为 `std::ranges::to<T>()` 的函数模板（可以将视图转换为任意容器类型 `T`）曾被提议用于 C++20，但最终未能实现。希望我们能在未来的 C++ 版本中得到类似的功能。在那之前，我们需要自己做更多的工作来具化视图。

在前面的示例中，我们将 `ints` 转换为了 `std::strings`，如下所示：

```cpp
auto ints = std::list{2, 3, 4, 2, 1};
auto r = ints
    | std::views::transform([](auto i) { return std::to_string(i); });
```

现在，如果我们要将范围 `r` 具化为一个 `vector`，我们可以像这样使用 `std::ranges::copy()`：

```cpp
auto vec = std::vector<std::string>{};
std::ranges::copy(r, std::back_inserter(vec));
```

具化视图是一个常见的操作，因此如果我们有一个通用的实用工具会很方便。假设我们想将某个任意视图具化为一个 `std::vector`；我们可以使用一些泛型编程来得出以下方便的实用函数：

```cpp
auto to_vector(auto&& r) {
    std::vector<std::ranges::range_value_t<decltype(r)>> v;
    if constexpr(std::ranges::sized_range<decltype(r)>) {
        v.reserve(std::ranges::size(r));
    }
    std::ranges::copy(r, std::back_inserter(v));
    return v;
}
```

> 此代码片段摘自 Timur Doumler 的博客文章 `https://timur.audio/how-to-make-a-container-from-a-c20-range`，该文章非常值得一读。

我们在这本书中还没有太多地讨论泛型编程，但接下来的几章将解释 `auto` 参数类型和 `if constexpr` 的用法。

我们使用 `reserve()` 来优化此函数的性能。它将为范围内的所有元素预先分配足够的空间，以避免后续的分配。但是，我们只能在知道范围大小时调用 `reserve()`，因此我们必须使用 `if constexpr` 语句在编译时检查该范围是否是 `size_range`。

有了这个实用工具，我们就可以将某种类型的容器转换为保存另一种任意类型元素的 `vector`。让我们看看如何使用 `to_vector()` 将一个整数列表转换为一个 `std::string` 的向量。这是一个示例：

```cpp
auto ints = std::list{2, 3, 4, 2, 1};
auto r = ints
    | std::views::transform([](auto i) { return std::to_string(i); });
auto strings = to_vector(r);
// strings is now a std::vector<std::string>
```

请记住，一旦视图被复制回一个容器，原始容器和转换后的容器之间就不再有任何依赖关系。这也意味着具化是一个急切 (eager) 的操作，而所有视图操作都是惰性的。


### 视图是惰性求值的

视图执行的所有工作都是惰性发生的。这与 `<algorithm>` 头文件中的函数相反，后者在被调用时立即对所有元素执行工作。

你已经看到 `std::views::filter` 视图可以替换算法 `std::copy_if()`，`std::views::transform` 视图可以替换 `std::transform()` 算法。当我们将视图用作构建块并将它们链接在一起时，我们通过避免急切算法所需的不必要的容器元素复制，从而从惰性求值中受益。

但是 `std::sort()` 呢？有对应的排序视图吗？答案是没有，因为它需要视图首先急切地收集所有元素才能返回第一个元素。相反，我们必须通过显式地对视图调用 `sort` 来自己完成这项工作。在大多数情况下，我们还需要在排序之前具化视图。我们可以通过一个例子来澄清这一点。假设我们有一个数字向量，我们已经通过某个谓词对其进行了过滤，如下所示：

```cpp
auto vec = std::vector{4, 2, 7, 1, 2, 6, 1, 5};
auto is_odd = [](auto i) { return i % 2 == 1; };
auto odd_numbers = vec | std::views::filter(is_odd);
```

如果我们尝试使用 `std::ranges::sort()` 或 `std::sort()` 来排序我们的视图 `odd_numbers`，我们会得到一个编译错误：

```cpp
std::ranges::sort(odd_numbers); // 不会编译
```

编译器会抱怨 `odd_numbers` 范围提供的迭代器类型。排序算法需要随机访问迭代器 (random access iterators)，但这不是我们的视图提供的迭代器类型，即使底层输入容器是 `std::vector`。我们需要做的是在排序之前具化视图：

```cpp
auto v = to_vector(odd_numbers);
std::ranges::sort(v);
// v is now 1, 1, 5, 7
```

但为什么这是必要的呢？答案是这是惰性求值的一个结果。当求值需要通过一次读取一个元素来保持惰性时，`filter` 视图（以及许多其他视图）无法保留底层范围（在本例中为 `std::vector`）的迭代器类型。

那么，有没有可以排序的视图呢？有，一个例子是 `std::views::take`，它返回范围中的前 $n$ 个元素。下面的示例在排序之前不需要具化视图，可以正常编译和运行：

```cpp
auto vec = std::vector{4, 2, 7, 1, 2, 6, 1, 5};
auto first_half = vec | std::views::take(vec.size() / 2);
std::ranges::sort(first_half);
// vec is now 1, 2, 4, 7, 2, 6, 1, 5
```

迭代器的质量得到了保留，因此可以对 `first_half` 视图进行排序。最终结果是底层向量 `vec` 的前一半元素已被排序。

你现在对 Ranges 库中的视图是什么以及它们如何工作有了很好的理解。在下一节中，我们将探讨如何使用标准库中包含的视图。


## 标准库中的 Views（视图）

到目前为止，在本章中我们一直在讨论 Ranges 库中的视图。如前所述，这些视图类型需要以常数时间构造，并且也具有常数时间的复制、移动和赋值操作符。然而，在 C++ 中，早在 Ranges 库被添加到 C++20 之前，我们就已经讨论过视图类。这些视图类是非拥有的类型，就像 `std::ranges::view` 一样，但没有复杂性保证。

在本节中，我们将首先探讨与 `std::ranges::view` 概念相关的 Ranges 库中的视图，然后转向与 `std::ranges::view` 不相关的 `std::string_view` 和 `std::span`。

 

### 范围视图 

Ranges 库中已经有很多视图，我相信我们会在未来的 C++ 版本中看到更多。本节将简要概述一些可用的视图，并根据它们的功能将其放入不同的类别中。

#### 生成视图 

生成视图产生值。它们可以生成有限或无限的范围值。这一类别中最明显的例子是 `std::views::iota`，它在半开区间内产生值。以下代码片段打印值 -2、-1、0 和 1：

```cpp
for (auto i : std::views::iota(-2, 2)) {
    std::cout << i << ' ';
}
// Prints -2 -1 0 1
```

通过省略第二个参数，`std::views::iota` 将按需生成无限数量的值。

#### 转换视图 

转换视图是转换范围元素的值和/或结构的视图。一些示例包括：

  * `std::views::transform`：转换每个元素的值和/或类型。
  * `std::views::reverse`：返回输入范围的反转版本。
  * `std::views::split`：将一个元素拆开并将每个元素拆分成一个子范围。结果范围是一个范围的范围。
  * `std::views::join`：与 `split` 相反；扁平化所有子范围。

以下示例使用 `split` 和 `join` 从逗号分隔值的字符串中提取所有数字：

```cpp
auto csv = std::string{"10,11,12"};
auto digits = csv
    | std::views::split(',') // [ [1, 0], [1, 1], [1, 2] ]
    | std::views::join;      // [ 1, 0, 1, 1, 1, 2 ]

for (auto i : digits) {
    std::cout << i;
}
// Prints 101112
```

#### 采样视图  

采样视图是选择范围中元素子集的视图，例如：

  * `std::views::filter`：仅返回满足所提供的谓词的元素。
  * `std::views::take`：返回范围的前 $n$ 个元素。
  * `std::views::drop`：在丢弃前 $n$ 个元素后，返回范围中所有剩余的元素。

在本章中你已经看到了大量使用 `std::views::filter` 的示例；它是一个极其有用的视图。`std::views::take` 和 `std::views::drop` 都有一个 `_while` 版本，它接受一个谓词而不是一个数字。这是一个使用 `take` 和 `drop_while` 的示例：

```cpp
auto vec = std::vector{1, 2, 3, 4, 5, 4, 3, 2, 1};
auto v = vec
    | std::views::drop_while([](auto i) { return i < 5; })
    | std::views::take(3);

for (auto i : v) { std::cout << i << " "; }
// Prints 5 4 3
```

此示例使用 `drop_while` 从前面丢弃小于 5 的值。剩余的元素被传递给 `take`，后者返回前三个元素。

现在是我们的最后一类范围视图。

#### 实用视图  

在本章中你已经看到了一些实用视图的作用。当你有一些想要转换或视作视图的东西时，它们会派上用场。

这一类视图的一些示例是 `ref_view`、`all_view`、`subrange`、`counted` 和 `istream_view`。

以下示例向你展示了如何读取一个包含浮点数的文本文件并打印它们。

假设我们有一个名为 `numbers.txt` 的文本文件，里面装满了重要的浮点数，如下所示：

```
1.4142 1.618 2.71828 3.14159 6.283 ...
```

然后，我们可以通过使用 `std::ranges::istream_view` 来创建一个浮点数视图：

```cpp
auto ifs = std::ifstream("numbers.txt");
for (auto f : std::ranges::istream_view<float>(ifs)) {
    std::cout << f << '\n';
}
ifs.close();
```

通过创建一个 `std::ranges::istream_view` 并向其传递一个 `istream` 对象，我们有了一种简洁的方式来处理来自文件或任何其他输入流的数据。

Ranges 库中的视图是经过仔细选择和设计的。在标准的后续版本中，很可能会有更多的视图加入。了解不同类别的视图有助于我们将它们区分开来，并在需要时轻松找到它们。

 

### 重新审视 `std::string_view` 和 `std::span`

值得注意的是，标准库在 Ranges 库之外还为我们提供了其他视图。在 第 4 章，《数据结构》 中介绍的 `std::string_view` 和 `std::span` 都是非拥有的范围，它们非常适合与 Ranges 视图结合使用。

不能保证这些视图可以以常数时间构造，而 Ranges 库中的视图则有此保证。例如，从以空字符结尾的 C 风格字符串构造 `std::string_view` 可能会调用 `strlen()`，这是一个 $O(n)$ 的操作。

假设出于某种原因，我们有一个函数可以重置范围内的前 $n$ 个值：

```cpp
auto reset(std::span<int> values, int n) {
    for (auto& i : std::ranges::take_view{values, n}) {
        i = int{};
    }
}
```

在这种情况下，不需要对 `values` 使用范围适配器，因为 `values` 本身就是一个视图。通过使用 `std::span`，我们可以传递内置数组或像 `std::vector` 这样的容器：

```cpp
int a[]{33, 44, 55, 66, 77};
reset(a, 3);
// a is now [0, 0, 0, 66, 77]

auto v = std::vector{33, 44, 55, 66, 77};
reset(v, 2);
// v is now [0, 0, 55, 66, 77]
```

以类似的方式，我们可以将 `std::string_view` 与 Ranges 库一起使用。以下函数将 `std::string_view` 的内容拆分为 `std::string` 元素的 `std::vector`：

```cpp
auto split(std::string_view s, char delim) {
    const auto to_string = [](auto&& r) -> std::string {
        const auto cv = std::ranges::common_view{r};
        return {cv.begin(), cv.end()};
    };
    return to_vector(std::ranges::split_view{s, delim}
        | std::views::transform(to_string));
}
```

Lambda 表达式 `to_string` 将一个 `char` 范围转换为一个 `std::string`。`std::string` 构造函数要求迭代器和哨兵类型相同，因此，该范围被包装在 `std::ranges::common_view` 中。实用工具 `to_vector()` 将视图具化并返回一个 `std::vector<std::string>`。`to_vector()` 在本章前面已定义。

我们的 `split()` 函数现在可以与 `const char*` 字符串和 `std::string` 对象一起使用，如下所示：

```cpp
const char* c_str = "ABC,DEF,GHI"; // C 风格字符串
const auto v1 = split(c_str, ','); // std::vector<std::string>

const auto s = std::string{"ABC,DEF,GHI"};
const auto v2 = split(s, ','); // std::vector<std::string>

assert(v1 == v2); // true
```

现在，我们将通过讨论 C++ 未来版本中 Ranges 库的发展前景来结束本章。
 

## Ranges 库的未来 

被 C++20 接受的 Ranges 库基于 Eric Niebler 编写的一个库，该库可在 `https://github.com/ericniebler/range-v3` 上获取。目前，该库的组件中只有一小部分进入了标准，但很可能很快会添加更多内容。

除了许多尚未被接受的有用视图，例如 `group_by`、`zip`、`slice` 和 `unique` 之外，还有一个 actions（动作）的概念，它可以像视图一样使用管道操作。然而，与视图的惰性求值不同，动作会对范围执行急切的变异 (eager mutations)。排序就是典型动作的一个例子。

如果你等不及这些功能被添加到标准库中，我建议你查看 `range-v3` 库。


## 总结

本章介绍了使用 Range 视图来构建算法的一些动机。通过使用视图，我们可以利用管道操作符，以简洁的语法高效地组合算法。你还了解了视图类意味着什么，以及如何使用将范围转换为视图的范围适配器。

视图不拥有其元素。构造范围视图被要求是常数时间操作，并且所有视图都是惰性求值的。你已经看到了如何将容器转换为视图，以及如何将视图具化回拥有的容器的示例。

最后，我们简要概述了标准库附带的视图，以及 C++ 中 Ranges 可能的未来。

本章是关于容器、迭代器、算法和范围系列文章的最后一章。我们现在将转向 C++ 中的内存管理。
