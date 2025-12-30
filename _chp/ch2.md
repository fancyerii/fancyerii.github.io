---
layout:     post
title:      "第二章：C++ 基础技巧"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - c++
---



<!--more-->


目录
* TOC
{:toc}

在本章中，我们将深入探讨一些基本的 C++ 技术，例如移动语义、错误处理和 lambda 表达式——这些技术将在本书中贯穿使用。其中一些概念即使对经验丰富的 C++ 程序员来说仍然容易混淆，因此我们将同时从它们的使用场景和底层原理进行讲解。

本章将涵盖以下主题：

* 自动类型推导，以及在声明函数和变量时如何使用 auto 关键字。
* 移动语义，以及“五法则”（rule of five）与“零法则”（rule of zero）。
* 错误处理与契约（contracts）。尽管这些主题不属于现代 C++ 的新特性，但异常与契约依然是当今 C++ 社区争论的热点。
* 使用 lambda 表达式创建函数对象，这是 C++11 中最重要的特性之一。

让我们首先来看一下自动类型推导。

## 使用 auto 关键字的自动类型推导

自从 C++11 引入 auto 关键字以来，C++ 社区对如何使用 auto 的不同形式（如 const auto&、auto&、auto&& 和 decltype(auto)）一直存在不少困惑。

### 在函数签名中使用 auto

虽然一些 C++ 程序员不推荐这样做，但根据我的经验，在函数签名中使用 auto 可以提高头文件的可读性和浏览效率。

下面展示了使用 auto 的语法与传统显式类型声明语法的对比示例：


<a>![](/img/chp/ch2/1.png)</a>

auto 语法既可以带返回类型后置说明，也可以不带。在某些情况下必须使用返回类型后置说明。例如，当我们编写虚函数，或者函数声明位于头文件而函数定义位于 .cpp 文件中时，就需要使用后置返回类型。

需要注意的是，auto 语法同样可以用于自由函数（free functions）：
 
<a>![](/img/chp/ch2/2.png)</a>

#### 使用 decltype(auto) 转发返回类型

有一种相对少见的自动类型推导形式叫作 decltype(auto)。它最常见的用途是用于从函数中精确转发返回类型。设想我们要为前面表中声明的 `val()` 和 `mref()` 编写包装函数，如下所示：

```cpp
int val_wrapper() { return val(); }   // 返回 int  
int& mref_wrapper() { return mref(); } // 返回 int&
```

如果我们希望为这些包装函数使用返回类型推导，直接使用 `auto` 关键字的话，编译器在两种情况下都会将返回类型推导为 `int`：

```cpp
auto val_wrapper() { return val(); }   // 返回 int  
auto mref_wrapper() { return mref(); } // 也返回 int
```

如果我们希望 `mref_wrapper()` 返回 `int&`，就需要写成 `auto&`。在这个示例中这样做没有问题，因为我们知道 `mref()` 的返回类型。
然而，这并非总是如此。如果我们希望编译器自动选择与被包装函数完全相同的返回类型，而不显式写出 `int&` 或 `auto&`，我们就可以使用 decltype(auto)：

```cpp
decltype(auto) val_wrapper() { return val(); }   // 返回 int  
decltype(auto) mref_wrapper() { return mref(); } // 返回 int&
```

这样，当我们不确定 `val()` 或 `mref()` 返回什么类型时，就不需要显式选择使用 `auto` 还是 `auto&`。
这种情况通常出现在泛型代码中——被包装的函数类型往往是模板参数。


### 在变量声明中使用 auto

自从 C++11 引入 auto 关键字以来，C++ 程序员之间就围绕它展开了不少争论。许多人认为它降低了代码的可读性，甚至让 C++ 看起来更像是一种动态类型语言。
我个人并不热衷于参与这些争论，但根据我的经验，我的观点是：几乎总是应该使用 auto，因为它能让代码更安全、更简洁、也更不容易出错。

当然，过度使用 auto 可能会让代码更难理解。在阅读代码时，我们通常想知道某个对象支持哪些操作。一个好的 IDE 可以帮助我们查看这些信息，但它并不会直接体现在源代码中。
C++20 的 concepts（概念） 特性在一定程度上解决了这个问题，它更关注对象的行为而不是显式的类型。
关于 C++ concepts 的更多内容，请参见第 8 章《编译期编程》。

我更倾向于在局部变量中使用 auto，并采用从左到右的初始化风格。也就是说，将变量名写在左边，等号在中间，右边是初始化表达式，如下所示：

```cpp
auto i = 0;
auto x = Foo{};
auto y = create_object();
auto z = std::mutex{}; // 从 C++17 起合法
```

随着 C++17 引入的强制性复制省略（guaranteed copy elision），语句 `auto x = Foo{}` 与 `Foo x{}` 完全等价。
也就是说，语言保证不会创建需要移动或复制的临时对象。
这意味着我们现在可以放心地使用这种从左到右的初始化风格，而不用担心性能问题，甚至可以将它用于不可移动/不可复制类型，例如 `std::atomic` 或 `std::mutex`。

使用 auto 声明变量的一个重大优势是：你永远不会忘记初始化变量，因为 `auto x;` 是无法编译的。
未初始化变量是导致未定义行为的常见来源之一，而按照这里建议的风格可以完全避免这一问题。

使用 auto 还能帮助你自动推导出变量的正确类型。不过，你仍然需要显式表达变量的使用意图：你是否需要一个引用或副本？是否需要修改变量，还是仅仅读取它？


#### const 引用（const reference）

const 引用（记作 `const auto&`）具有一个重要特性：它可以绑定到任何对象。
通过这种引用，原始对象永远无法被修改。
我认为，对于那些复制代价较高的对象，`const` 引用应当作为默认选择。

如果 `const` 引用绑定到一个临时对象，那么该临时对象的生命周期将被延长至引用的生命周期。
下面的例子展示了这一行为：

```cpp
void some_func(const std::string& a, const std::string& b) {
    const auto& str = a + b; // a + b 返回一个临时对象
    // ...
} // str 作用域结束，临时对象被销毁
```

我们还可以通过 `auto&` 得到一个 `const` 引用，例如：

```cpp
auto foo = Foo{};
auto& cref = foo.cref(); // cref 是一个 const 引用
auto& mref = foo.mref(); // mref 是一个可变引用
```

虽然这种写法在语法上完全合法，但更好的做法是：
当我们确实在处理 `const` 引用时，应当显式使用 `const auto&`；
而 `auto&` 则应当仅用于表示可变引用（mutable reference）。


#### 可变引用（mutable reference）

与 `const` 引用相反，可变引用不能绑定到临时对象。
如前所述，我们使用 `auto&` 表示可变引用。
只有当你确实打算修改被引用的对象时，才应使用可变引用。


#### 转发引用（forwarding reference）

`auto&&` 被称为转发引用（forwarding reference，也称为万能引用 universal reference）。
它可以绑定到任何对象，这使得它在某些场景中非常有用。

转发引用与 `const` 引用类似——它们也会延长临时对象的生命周期。
但不同的是，`auto&&` 允许修改它所引用的对象（包括临时对象）。

当你只是将变量转发（forward）给其他代码时，应使用 `auto&&`。
在这些转发场景中，你通常不关心变量是 const 还是可变，
而只是想把它传递给真正使用该变量的代码。


⚠️ 注意：`auto&&` 和 `T&&` 只有在函数模板中且 T 是该函数模板的模板参数时，
才被视为转发引用。如果在显式类型中使用 `&&`，例如：

```cpp
std::string&&
```

那么它表示的是一个右值引用（rvalue reference），不具备转发引用的特性。（关于右值和移动语义的内容，将在本章后面讨论。）


#### 便于使用的实践建议

虽然以下内容是我的个人观点，但我建议遵循以下规则来提高代码的可读性与安全性：

* 对于基本类型（如 `int`、`float` 等）以及体积较小的非基本类型（如 `std::pair`、`std::complex`），使用 `const auto`；
* 对于较大、复制代价高的类型，使用 `const auto&`；

以上两条规则可覆盖 C++ 代码中绝大多数变量声明。

当你需要可变引用或显式拷贝行为时，再使用 `auto&` 或 `auto`。
这能清楚地向阅读代码的人传达：这些变量很重要，因为它们要么复制对象，要么修改被引用的对象。

最后，仅在转发（forwarding）代码中使用 `auto&&`。

遵循这些规则可以让你的代码更容易阅读、调试和推理。

> 你可能会注意到，虽然我推荐在大多数变量声明中使用 `const auto` 和 `const auto&`，
> 但在本书的某些地方我仍使用了简单的 `auto`。
> 这是因为书籍篇幅有限，代码示例需保持简洁。


### 指针的 const 传播（Const propagation for pointers）

通过使用关键字 `const`，我们可以告知编译器哪些对象是不可变的（immutable）。
编译器随后会检查我们是否尝试修改不应被修改的对象，也就是所谓的const 正确性（const-correctness）。

然而，在编写 const 正确代码时，一个常见的陷阱是：
即使对象被声明为 const，它的成员指针仍可能修改其所指向的值。

例如：

```cpp
class Foo {
public:
    Foo(int* ptr) : ptr_{ptr} {}
    auto set_ptr_val(int v) const {
        *ptr_ = v; // 尽管函数声明为 const，但仍可编译！
    }
private:
    int* ptr_{};
};

int main() {
    auto i = 0;
    const auto foo = Foo{&i};
    foo.set_ptr_val(42);
}
```

虽然 `set_ptr_val()` 修改了一个 `int` 值，但它仍然可以声明为 `const`，因为被修改的是 `ptr_` 所指向的对象，而非 `ptr_` 本身。


为了解决这一问题，标准库扩展中新增了一个名为`std::experimental::propagate_const` 的包装器（目前已被最新版本的 Clang 与 GCC 支持）。通过使用 `propagate_const`，上述 `set_ptr_val()` 将无法通过编译。

需要注意：`propagate_const` 仅适用于指针和类指针类型（如 `std::shared_ptr`、`std::unique_ptr`），但不适用于 `std::function`。

示例：

```cpp
#include <experimental/propagate_const>

class Foo {
public:
    Foo(int* ptr) : ptr_{ptr} {}

    auto set_ptr(int* p) const {
        ptr_ = p; // 无法编译，符合预期
    }

    auto set_val(int v) const {
        val_ = v; // 无法编译，符合预期
    }

    auto set_ptr_val(int v) const {
        *ptr_ = v; // 无法编译，const 被传播
    }

private:
    std::experimental::propagate_const<int*> ptr_ = nullptr;
    int val_{};
};
```
 
在大型代码库中，正确使用 `const` 的重要性不可低估。而 `propagate_const` 的引入，使得 `const` 的传播与检查更加高效、可靠。

接下来，我们将讨论移动语义（move semantics），以及在类中处理资源时的一些重要规则。


## 移动语义详解（Move semantics explained）

移动语义（move semantics）是 C++11 引入的一个概念。根据我的经验，即使是有丰富经验的程序员，也常常难以真正理解它的工作原理。因此，本节将深入讲解移动语义的机制、编译器在何时使用它，以及它存在的必要性。


C++ 之所以引入“移动语义”这个概念，而多数其他语言并不需要，根本原因在于 —— C++ 是一种以值（value）为基础的语言。（这一点在第 1 章《C++ 简介》中已经讨论过。）如果 C++ 没有内建移动语义，那么“值语义”的优势在许多场景中将会丧失。程序员会被迫在以下几种折衷方案中做出选择：

* 执行冗余的深度拷贝（deep clone）操作，性能代价高昂；
* 像 Java 那样使用指针来操作对象，从而丧失值语义的稳健性；
* 执行容易出错的交换（swap）操作，以牺牲代码可读性为代价。

显然，这三种方式都不是理想的解决方案。
这正是引入 移动语义（move semantics） 的原因所在。



### 拷贝构造、交换与移动

在深入移动语义细节之前，我们先来区分三个相关操作：

* 拷贝构造（copy construction）
* 交换（swap）
* 移动构造（move construction）


#### 拷贝构造（Copy-constructing an object）

当我们拷贝一个管理资源的对象时，必须为新对象分配一个新的资源，并从源对象中复制数据，以确保两个对象彼此完全独立。

假设我们有一个类 `Widget`，它在构造时会分配某种资源。下面的代码展示了默认构造和拷贝构造的过程：

```cpp
auto a = Widget{};
auto b = a; // 拷贝构造（copy-construction）
```

<a>![](/img/chp/ch2/3.png)</a>

在这一过程中，系统会进行以下两步操作：

1. 为 `b` 分配新的资源；
2. 将 `a` 中的资源内容复制到 `b` 中。

这种“分配 + 拷贝”的操作既耗时又浪费资源。更糟的是，在许多情况下，原对象（如 `a`）在拷贝后已不再需要。

这时，移动语义 就能派上用场。编译器会检测到此类情形（即旧对象不再被引用），并执行移动操作（move operation）来代替拷贝操作，从而避免无谓的资源分配与复制。

#### 交换两个对象（Swapping two objects）

在 C++11 引入移动语义之前，交换（swap）两个对象的内容 是一种常见的方式，
用于在不进行额外分配和拷贝的情况下转移数据。

如下示例所示，两个对象只是简单地交换它们内部的内容：

```cpp
auto a = Widget{};
auto b = Widget{};
std::swap(a, b);
```

下图展示了这一过程的示意：

<a>![](/img/chp/ch2/4.png)</a>

`std::swap()` 函数是一个简单但非常有用的工具函数，
它在本章后面介绍的 “拷贝并交换（copy-and-swap）惯用法” 中发挥了重要作用。


#### 移动构造对象（Move-constructing an object）

当我们移动（move）一个对象时，目标对象会直接“窃取”源对象所持有的资源，而源对象则会被重置（reset）。

可以看到，这与交换（swap）非常相似，唯一的区别在于：被移动的源对象（moved-from object）不需要从目标对象那里获得任何资源。

例如：

```cpp
auto a = Widget{};
auto b = std::move(a); // 告诉编译器将资源从 a 移动到 b
```

下图展示了该过程：

<a>![](/img/chp/ch2/5.png)</a>

尽管源对象在移动后被重置，但它仍然处于一种有效状态（valid state）。然而，这种“重置”并不是编译器自动完成的。我们必须在移动构造函数（move constructor）中手动实现这种重置，以确保对象仍然能被安全销毁（destroyed）或重新赋值（assigned）。本章稍后会进一步讨论“有效状态”的含义。

移动对象只有在该对象类型拥有某种资源时才有意义（最常见的情况是堆内存分配的资源）。如果对象的所有数据都包含在其自身内部，那么最有效的“移动”方式其实就是普通的拷贝（copy）。


现在你已经对移动语义有了基本理解，接下来我们将深入探讨它的具体机制与实现细节。



好的，这是原文的中文翻译，去除了代码和注释，仅保留概念性描述：


### 资源的获取与五法则（The Rule of Five）

要完全理解移动语义，我们需要回顾 C++ 中类和资源获取的基础知识。C++ 中的一个基本概念是：一个类应该完全管理自身的资源。 这意味着当一个类被拷贝、被移动、被拷贝赋值、被移动赋值或被析构时，该类应确保其资源得到相应处理。实现这五种特殊函数（方法）的必要性通常被称为五法则（Rule of Five）。

我们来看一个处理已分配资源的类如何实现五法则。在 `Buffer` 类中，已分配的资源是一个由原始指针 `ptr_` 指向的 `float` 数组。

```
class Buffer {
public:
  // Constructor
  Buffer(const std::initializer_list<float>& values) : size_{values.size()} {
    ptr_ = new float[values.size()];
    std::copy(values.begin(), values.end(), ptr_);
  }

  // Iterators for accessing the data
  auto begin() const { return ptr_; }

  auto end() const { return ptr_ + size_; }

private:
  size_t size_{0};
  float* ptr_{nullptr};
};
```

在这种情况下，被处理的资源是在 `Buffer` 类的构造函数中分配的一块内存。内存可能是类需要处理的最常见的资源，但资源可以包含更多，例如互斥锁（mutex）、显卡上的纹理句柄、线程句柄等等。

五法则中提到的五个函数如下：

在引入移动语义（C++11 之前），这三个函数通常被称为三法则（Rule of Three）：

1.  拷贝构造函数： 在创建新对象作为现有对象的副本时调用。
2.  拷贝赋值运算符： 当将一个已初始化的对象赋值给另一个对象时调用。
3.  析构函数： 在对象超出作用域时自动调用，用于释放资源。

```

  // 1. Copy constructor
  Buffer(const Buffer& other) : size_{other.size_} {
    ptr_ = new float[size_];
    std::copy(other.ptr_, other.ptr_ + size_, ptr_);
  }

  // 2. Copy assignment
  auto& operator=(const Buffer& other) {
    delete[] ptr_;
    ptr_ = new float[other.size_];
    size_ = other.size_;
    std::copy(other.ptr_, other.ptr_ + size_, ptr_);
    return *this;
  }

  // 3. Destructor
  ~Buffer() {
    delete[] ptr_; // Note, it is valid to delete a nullptr
    ptr_ = nullptr;
  }

```

尽管正确实现这三个函数足以让一个类处理其内部资源，但会产生两个问题：

* 无法拷贝的资源： 比如 `std::thread`、网络连接等，拷贝没有意义或不可能。在这种情况下，我们不能传递对象。
* 不必要的拷贝： 如果我们从函数返回 `Buffer` 类对象，整个数组都需要被拷贝。（尽管编译器在某些情况下会优化掉拷贝，但我们先忽略这一点。）

移动语义是这些问题的解决方案。除了拷贝构造和拷贝赋值之外，我们还需向类中添加：

4.  移动构造函数
5.  移动赋值运算符

```

  // 4. Move constructor
  Buffer(Buffer&& other) noexcept : size_{other.size_}, ptr_{other.ptr_} {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  // 5. Move assignment
  auto& operator=(Buffer&& other) noexcept {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    return *this;
  }
```

与拷贝版本的参数是常量引用（`const Buffer&`）不同，移动版本接受一个 `Buffer&&` 对象。`&&` 限定符表示该参数是我们打算从中移动而非拷贝的对象。用 C++ 术语来说，这被称为右值（rvalue）。

拷贝函数用于复制对象，而移动函数旨在将资源从一个对象转移到另一个对象，从而解除源对象对该资源的拥有权。

移动函数的实现

移动构造函数和移动赋值运算符不进行内存分配等可能抛出异常的操作，因此应被标记为 `noexcept`：

4.  移动构造函数： 简单地从源对象中接管资源指针 (`ptr_`) 和大小，然后将源对象（`other`）的指针设为 `nullptr`，大小设为 $0$。
5.  移动赋值运算符： 类似地，接管源对象的资源，并重置源对象的状态。

现在，当编译器检测到存在一个看似拷贝的操作（例如从函数返回一个 `Buffer`），但被拷贝的值之后不再使用时，它将利用不会抛出异常的移动构造函数/移动赋值运算符来代替拷贝。

这非常高效：接口保持与拷贝一样清晰，但在底层，编译器执行的是简单的资源移动。因此，程序员无需使用特殊的指针或输出参数来避免拷贝；由于类实现了移动语义，编译器会自动处理。

重要提示： 务必将你的移动构造函数和移动赋值运算符标记为 `noexcept`（除非它们确实可能抛出异常）。如果不标记为 `noexcept`，在某些条件下，标准库容器和算法将无法使用它们，而是退回到使用常规的拷贝/赋值操作。

为了知道编译器何时允许移动而非拷贝对象，理解右值是必要的。


### 具名变量与右值

那么，编译器在什么时候允许移动对象而不是拷贝呢？简而言之，当一个对象可以被归类为右值（rvalue）时，编译器就会移动它。

“右值”这个术语听起来可能很复杂，但本质上，右值就是一个不与具名变量（Named Variable）绑定的对象，原因通常是以下之一：

  * 它直接来自于一个函数。
  * 我们通过使用 `std::move()` 将一个变量变成了右值。

下面的例子展示了这两种情况：

```cpp
// 从 make_buffer 返回的对象没有与变量绑定
x = make_buffer(); // 执行移动赋值

// 变量 "x" 通过 std::move() 传入
y = std::move(x); // 执行移动赋值
```

在本书中，我也会将 左值（lvalue） 和 具名变量 互换使用。左值对应于那些我们可以在代码中通过名字来引用的对象。

现在，我们通过在一个类中使用 `std::string` 类型的成员变量，使这个概念更进阶一些。下面的 `Button` 类将作为示例：

```cpp
class Button {
public:
    Button() {}
    auto set_title(const std::string& s) { // 接收左值引用（拷贝）
        title_ = s;
    }
    auto set_title(std::string&& s) { // 接收右值引用（移动）
        title_ = std::move(s);
    }
    std::string title_;
};
```

我们还需要一个返回标题的自由函数和一个 `Button` 变量：

```cpp
auto get_ok() {
    return std::string("OK");
}
auto button = Button{};
```

有了这些前提，我们来看几个拷贝和移动的详细案例：

  * 案例 1： `Button::title_` 执行了拷贝赋值，因为 `std::string` 对象与变量 `str` 绑定。

    ```cpp
    auto str = std::string{"OK"};
    button.set_title(str); // 拷贝赋值
    ```

  * 案例 2： `Button::title_` 执行了移动赋值，因为 `str` 通过 `std::move()` 传入。

    ```cpp
    auto str = std::string{"OK"};
    button.set_title(std::move(str)); // 移动赋值
    ```

  * 案例 3： `Button::title_` 执行了移动赋值，因为新的 `std::string` 对象直接来自一个函数（是右值）。

    ```cpp
    button.set_title(get_ok()); // 移动赋值
    ```

  * 案例 4： `Button::title_` 执行了拷贝赋值，因为 `str` 变量绑定了 `get_ok()` 返回的对象（与案例 1 相同）。

    ```cpp
    auto str = get_ok();
    button.set_title(str); // 拷贝赋值
    ```

  * 案例 5： `Button::title_` 执行了拷贝赋值，因为 `str` 被声明为 `const`，因此不允许被修改（移动操作会修改源对象）。

    ```cpp
    const auto str = get_ok();
    button.set_title(std::move(str)); // 拷贝赋值
    ```

正如你所见，判断一个对象是被移动还是被拷贝是相当简单的。如果它有一个变量名（即它是左值），它就会被拷贝；否则，它就会被移动。 如果你使用 `std::move()` 来移动一个具名对象，该对象不能被声明为 `const`。


### 默认移动语义与零法则

本节讨论自动生成的拷贝赋值运算符。重要的是要知道，自动生成的函数不提供强异常保证（strong exception guarantees）。因此，如果在拷贝赋值过程中抛出异常，对象可能会陷入只被部分拷贝的中间状态。
 

与拷贝构造函数和拷贝赋值运算符一样，移动构造函数和移动赋值运算符也可以由编译器自动生成。虽然某些编译器在特定条件下允许自动生成这些函数（后续会详细说明），但我们可以简单地使用 `= default` 关键字来强制编译器生成它们。

对于像 `Button` 这样没有手动管理任何资源的类，我们可以通过添加 `default` 关键字来扩展它：

```
class Button {
public:
  Button() {}

  // Copy-constructor/copy-assignment
  Button(const Button&) = default;
  auto operator=(const Button&) -> Button& = default;

  // Move-constructor/move-assignment
  Button(Button&&) noexcept = default;
  auto operator=(Button&&) noexcept -> Button& = default;

  // Destructor
  ~Button() = default;
}
```
 

更简单地说，如果我们不声明任何自定义的拷贝构造函数、拷贝赋值运算符或析构函数，那么移动构造函数和移动赋值运算符将隐式地被声明。这意味着最初定义的 `Button` 类（即一个空类，不包含任何五法则函数）实际上已经自动处理了所有操作。

```
class Button {
public:
  Button() {} // Same as before
  // Nothing here, the compiler generates everything automatically!
  // ...
};
```

这一原则被称为零法则（The Rule of Zero），它倡导：如果类不需要显式声明任何五法则函数（因为它们可以依赖于成员的默认行为），那么就一个也不要声明。
 

很容易忘记，只要添加了五法则中的任何一个自定义函数，就会阻止编译器自动生成其他的函数。

例如，如果一个版本的 `Button` 类包含一个自定义析构函数，那么它的移动运算符（移动构造和移动赋值）就不会被自动生成。结果是，该类在需要移动的场景下，将始终退回到执行拷贝操作。

```
class Button {
public:
  Button() {}
  ~Button(){
    std::cout << "destructed\n"
  }
  // ...
};
```

我们可以利用对这些自动生成函数的深入理解，来实现应用类。


#### 实际代码库中的零法则

在实践中，你需要亲手编写拷贝/移动构造函数、拷贝/移动赋值运算符和析构函数的情况应该非常少。

将你的类编写成不需要显式编写（或 `= default` 声明）任何这些特殊成员函数，这种实践通常被称为零法则（Rule of Zero）。这意味着，如果应用程序代码库中的某个类需要显式编写这些函数中的任何一个，那么这部分代码可能更适合放在你代码库的底层库部分。

本书后续会讨论 `std::optional`，这是一个在应用零法则时，用于处理可选成员的实用工具类。

编写一个空的析构函数可能会阻止编译器执行某些优化。

正如在代码片段中所见，对一个包含空析构函数的平凡类（trivial class）进行数组拷贝，生成的汇编代码（未优化）与手工编写 `for` 循环进行拷贝时生成的汇编代码是相同的。

  * 第一个版本使用了空析构函数和 `std::copy()`：

    ```cpp
    struct Point {
        int x_, y_;
        ~Point() {} // 空析构函数，不要使用！
    };
    auto copy(Point* src, Point* dst) {
        std::copy(src, src+64, dst);
    }
    ```

  * 第二个版本使用了没有析构函数的 `Point` 类和手工编写的 `for` 循环：

    ```cpp
    struct Point {
        int x_, y_;
    };
    auto copy(Point* src, Point* dst) {
        const auto end = src + 64;
        for (; src != end; ++src, ++dst) {
            *dst = *src;
        }
    }
    ```

这两个版本都生成了对应于简单循环的 x86 汇编代码。

```
xor eax, eax
.L2:
mov rdx, QWORD PTR [rdi+rax]
mov QWORD PTR [rsi+rax], rdx
add rax, 8
cmp rax, 512
jne .L2
rep ret
```

然而，如果我们移除析构函数，或者将析构函数声明为 `= default`，编译器会优化 `std::copy()`，转而使用`memmove()`函数而不是循环：

```cpp
struct Point {
    int x_, y_;
    ~Point() = default; // OK：使用 default 或完全不写析构函数
};
auto copy(Point* src, Point* dst) {
    std::copy(src, src+64, dst);
}
```

这段代码生成的 x86 汇编就包含了 `memmove()` 优化。

```
mov rax, rdi
mov edx, 512
mov rdi, rsi
mov rsi, rax
jmp memmove
```

总而言之，为了从应用程序中挤出更多的性能，请使用 `= default` 析构函数或完全不写析构函数，而不是使用空的析构函数。


#### 常见的陷阱——移动非资源类型

在使用默认创建的移动赋值运算符时，存在一个常见的陷阱：混合了基础类型和更高级复合类型的类。与复合类型不同，基础类型（如 `int`、`float` 和 `bool`）在被移动时，实际上是简单地被拷贝了，因为它们不管理任何资源。

当一个简单类型与一个拥有资源的类型混合在一起时，默认的移动赋值就会变成移动和拷贝的混合体。

下面是一个会导致失败的类示例：

```cpp
class Menu {
public:
  Menu(const std::initializer_list<std::string>& items)
                : items_{items} {}
  auto select(int i) {
    index_ = i;
  }
  auto selected_item() const {
    return index_ != -1 ? items_[index_] : "";
  }
  // ...
private:
  int index_{-1}; // Currently selected item
  std::vector<std::string> items_;
};
```

如果 `Menu` 类像这样被使用，将导致未定义行为：

```cpp
auto a = Menu{"New", "Open", "Close", "Save"};
a.select(2);
auto b = std::move(a);
auto selected = a.selected_item(); // crash
```

未定义行为的发生是因为：

1.  `items_` 向量（资源拥有类型）被移动了，因此在源对象 `a` 中变为空（empty）。
2.  `index_` 整数（基础类型）被拷贝了，因此在被移动后的源对象 `a` 中仍然保留着值 `2`。

当调用 `a.selected_item()` 时，该函数会尝试访问已经为空的 `items_` 向量的索引 `2`，从而导致程序崩溃。

在处理这类情况时，更好的移动构造函数/移动赋值运算符实现方式是简单地 交换（swapping） 成员，如下所示：

```cpp
Menu(Menu&& other) noexcept {
  std::swap(items_, other.items_);
  std::swap(index_, other.index_);
}

auto& operator=(Menu&& other) noexcept {
  std::swap(items_, other.items_);
  std::swap(index_, other.index_);
  return *this;
}
```

通过这种方式，`Menu` 类可以安全地被移动，同时仍然保留了 不抛出异常（no-throw） 的保证。在本书后续的第八章《编译期编程》中，你将学习如何利用 C++ 中的反射技术来自动化创建通过交换元素来实现移动构造函数/赋值运算符的过程。


 

### 将 `&&` 限定符应用于类成员函数

除了应用于对象之外，你也可以像对成员函数应用 `const` 限定符一样，对成员函数应用 `&&` 限定符。与 `const` 限定符类似，带有 `&&` 限定符的成员函数仅在对象是右值（rvalue）时才会被重载决议（overload resolution）考虑：

```cpp
struct Foo {
auto func() && {}
};

auto a = Foo{};
a.func(); // 编译失败，'a' 不是右值
std::move(a).func(); // 编译成功
Foo{}.func(); // 编译成功
```

有人可能会觉得需要这种行为很奇怪，但它确实存在用例。我们将在第十章《代理对象与惰性求值》中研究其中一个用例。


### 当拷贝省略发生时不要移动

你可能很想在函数返回值时使用 `std::move()`，像这样：

```cpp
auto func() {
  auto x = X{};
  // ...
  return std::move(x); // 不要这样做，这会阻止 RVO
}
```

然而，除非 `x` 是一个只可移动（move-only）的类型，否则你不应该这样做。这种对 `std::move()` 的使用会阻止编译器进行返回值优化（RVO, Return Value Optimization），从而无法完全省略对 `x` 的拷贝。省略拷贝比移动更高效。

因此，当你按值返回一个新创建的对象时，不要使用 `std::move()`；而是直接返回该对象：

```cpp
auto func() {
  auto x = X{};
  // ...
  return x; // OK
}
```

这种对具名对象（Named Object）进行省略拷贝的特定示例通常被称为 NRVO（Named-RVO，具名返回值优化）。RVO 和 NRVO 已被所有主流 C++ 编译器实现。如果你想阅读更多关于 RVO 和拷贝省略的内容，可以在 `https://en.cppreference.com/w/cpp/language/copy_elision` 上找到详细的总结。


### 尽量按值传递（Pass by Value）

考虑一个将 `std::string` 转换为小写字母的函数。为了在适用的情况下使用移动构造函数，在其他情况下使用拷贝构造函数，似乎需要定义两个函数：

```cpp
// 参数 s 是一个 const 引用
auto str_to_lower(const std::string& s) -> std::string {
  auto clone = s;
  for (auto& c: clone) 
    c = std::tolower(c);
  return clone;
}

// 参数 s 是一个右值引用
auto str_to_lower(std::string&& s) -> std::string {
  for (auto& c: s)
    c = std::tolower(c);
  return s;
}
```

然而，通过按值接收 `std::string` 参数，我们可以只编写一个函数来覆盖这两种情况：

```cpp
auto str_to_lower(std::string s) -> std::string {
  for (auto& c: s)
    c = std::tolower(c);
  return s;
}
```

我们来看看为什么 `str_to_lower()` 的这种实现方式可以在可能的情况下避免不必要的拷贝。

当传递一个常规变量时，如下所示，`str` 的内容在函数调用之前被拷贝构造到参数 `s` 中，然后在函数返回时再移动赋值回 `str`：

```cpp
auto str = std::string{"ABC"};
str = str_to_lower(str);
```

当传递一个右值时，如下所示，`str` 的内容在函数调用之前被移动构造到参数 `s` 中，然后在函数返回时再移动赋值回 `str`。因此，在整个函数调用过程中没有发生拷贝：

```cpp
auto str = std::string{"ABC"};
str = str_to_lower(std::move(str));
```

乍一看，这种技术似乎可以适用于所有参数。然而，如你接下来将看到的，这种模式并非总是最佳的。



#### 按值传递不适用的情况

有时候，“按值接收参数然后移动（accept-by-value-then-move）”的模式实际上是一种悲观化（pessimization，即性能负优化）。

例如，考虑下面的类，其中的 `set_data()` 函数会保留传入参数的一个拷贝：

```cpp
class Widget {
    std::vector<int> data_{};
    // ...
public:
    void set_data(std::vector<int> x) {
        data_ = std::move(x);
    }
};
```

假设我们调用 `set_data()` 并传入一个左值（lvalue），如下所示：

```cpp
auto v = std::vector<int>{1, 2, 3, 4};
widget.set_data(v); // 传递一个左值
```

由于我们传递的是一个具名对象 `v`，这段代码会：

1.  拷贝构造一个新的 `std::vector` 对象 `x`（这是函数参数）。
2.  然后，将这个对象 `x` 移动赋值给 `data_` 成员。

除非我们传递一个空的向量对象给 `set_data()`，否则 `std::vector` 的拷贝构造函数会为其内部缓冲区执行一次堆分配。

现在，将它与下面这个为左值优化过的 `set_data()` 版本进行比较：

```cpp
void set_data(const std::vector<int>& x) {
    data_ = x; // 如果可能，重用 data_ 中的内部缓冲区
}
```

在这里，只有当当前向量 `data_` 的容量小于源对象 `x` 的大小时，赋值运算符内部才会发生堆分配。换句话说，`data_` 内部预先分配的缓冲区在许多情况下可以在赋值运算符中被重用，从而避免了一次额外的堆分配。

如果我们发现有必要对 `set_data()` 进行左值和右值优化，那么在这种情况下，最好提供两个重载版本：

```cpp
void set_data(const std::vector<int>& x) {
    data_ = x;
}

void set_data(std::vector<int>&& x) noexcept {
    data_ = std::move(x);
}
```

第一个版本对左值是最佳的，第二个版本对右值是最佳的。


#### 移动构造函数参数

最后，我们来看一个可以安全地使用按值传递模式，而不用担心刚才演示的性能负优化的场景。

在构造函数中初始化类成员时，我们可以安全地使用“按值接收参数然后移动”的模式。在构造一个新对象的过程中，不存在可以被利用的预分配缓冲区来避免堆分配的情况。

下面是一个包含一个 `std::vector` 成员和一个构造函数的示例，用于演示此模式：

```cpp
class Widget {
    std::vector<int> data_;
public:
    Widget(std::vector<int> x) // 按值接收
        : data_{std::move(x)} {} // 移动构造
    // ...
};
```

现在，我们将把焦点转移到一个不能被认为是现代 C++，但即使在今天也经常被讨论的话题上。


## 设计带有错误处理的接口

错误处理是函数和类的接口中一个重要但经常被忽视的部分。错误处理在 C++ 中是一个激烈争论的话题，但争论往往倾向于异常（exceptions）与某种其他错误机制的选择。虽然这是一个有趣的领域，但在专注于错误处理的实际实现之前，理解错误处理的其他方面更为重要。显然，异常和错误码都已成功应用于无数软件项目，并且结合使用这两种方法的项目也并不少见。

 

无论使用何种编程语言，错误处理的一个基本方面是区分编程错误（也称为 Bug）和运行时错误（runtime errors）。

运行时错误可以进一步划分为可恢复的运行时错误（recoverable runtime errors）和不可恢复的运行时错误（unrecoverable runtime errors）。一个不可恢复的运行时错误的例子是栈溢出（stack overflow）。当发生不可恢复的错误时，程序通常会立即终止，因此发出这类错误信号没有意义。然而，有些错误在一种类型的应用程序中可能被认为是可恢复的，而在其他应用程序中则可能是不可恢复的。

在讨论可恢复和不可恢复错误时，一个经常出现的极端情况是 C++ 标准库在内存耗尽时的行为，这多少有些不尽人意。当你的程序内存耗尽时，这通常是不可恢复的，但标准库（试图）在这种情况下抛出 `std::bad_alloc` 异常。我们在此不花时间讨论不可恢复的错误，但 Herb Sutter 的演讲《De-fragmenting C++: Making Exceptions and RTTI More Affordable and Usable》非常值得推荐，如果你想深入挖掘这个主题的话。



在设计和实现 API 时，你应始终反思你正在处理的错误类型，因为不同类别的错误应以完全不同的方式处理。决定错误是编程错误还是运行时错误，可以通过一种称为契约式设计（Design by Contract）的方法论来完成；这是一个本身就值得一本书来介绍的主题。然而，我将在此介绍其基础知识，这足以满足我们的目的。

> C++ 中有提议增加对契约的语言支持，但目前契约尚未进入标准。然而，许多 C++ API 和指南都假设你了解契约的基础知识，因为契约所使用的术语使得讨论和文档化类和函数的接口变得更容易。

### 契约（Contracts）

契约是某个函数的调用者（caller）与函数本身（callee）之间的一组规则。C++ 允许我们使用 C++ 类型系统来明确指定一些规则。例如，考虑以下函数签名：

```cpp
int func(float x, float y)
```

它规定 `func()` 返回一个整数（除非它抛出异常），并且调用者必须传入两个浮点值。但是，它没有说明允许传入哪些浮点值。例如，我们是否可以传入 $0.0$ 或负值？此外，`x` 和 `y` 之间可能存在一些无法轻易使用 C++ 类型系统表达的必需关系。当我们谈论 C++ 中的契约时，我们通常指的是那些难以使用类型系统表达的、存在于调用者和被调用者之间的规则。

在此不拘泥于形式，将介绍一些与契约式设计相关的概念，以便为你提供一些可用于推理论证接口和错误处理的术语：

  * 前置条件（Precondition）： 指定函数调用者的责任。可能对传递给函数的参数存在约束。或者，如果它是一个成员函数，在调用该函数之前，对象可能必须处于特定的状态。例如，在调用 `std::vector` 的 `pop_back()` 时，前置条件是该向量不能是空的。确保向量不为空是 `pop_back()` 调用者的责任。

  * 后置条件（Postcondition）： 指定函数返回时的责任。如果它是一个成员函数，该函数会将对象置于何种状态？例如，`std::list::sort()` 的后置条件是列表中元素按升序排列。

  * 不变量（Invariant）： 是应该始终保持为真的条件。不变量可用于多种上下文。

      * 循环不变量（Loop Invariant）是每次循环迭代开始时必须为真的条件。
      * 此外，类不变量（Class Invariant）定义了对象的有效状态。例如，`std::vector` 的一个不变量是 `size() <= capacity()` 始终成立。

明确阐述围绕某些代码的不变量，能让我们更好地理解代码。不变量也是一种可用于证明某个算法是否达到预期目的的工具。类不变量非常重要；因此，我们将花费更多时间来讨论它们是什么以及它们如何影响类的设计。


#### 类不变量（Class Invariants）

如前所述，类不变量定义了一个对象的有效状态。它明确规定了类内部数据成员之间的关系。在一个成员函数执行期间，对象可能暂时处于无效状态。关键在于，每当函数将控制权传递给任何能够观察对象状态的其他代码时，不变量都必须保持成立。 这可能发生在函数：

  * 返回时
  * 抛出异常时
  * 调用回调函数时
  * 调用可能观察当前调用对象状态的其他函数时；常见的情况是向其他函数传递 `this` 的引用

认识到类不变量是类中每个成员函数的前置条件和后置条件的隐式部分，这一点非常重要。如果一个成员函数使对象处于无效状态，则表示后置条件未得到满足。同样地，一个成员函数可以始终假定在被调用时，对象是处于有效状态的。这条规则的例外是类的构造函数和析构函数。

如果我们想插入代码来检查类不变量是否成立，可以在以下这些点进行：

```cpp
struct Widget {
    Widget() {
        // 初始化对象…
        // 检查类不变量
    }
    ~Widget() {
        // 检查类不变量
        // 销毁对象…
    }
    auto some_func() {
        // 检查前置条件（包括类不变量）
        // 执行实际工作…
        // 检查后置条件（包括类不变量）
    }
};
```

这里省略了拷贝/移动构造函数和拷贝/移动赋值运算符，但它们分别遵循与构造函数和 `some_func()` 相同的模式。

当一个对象被移动（moved from）后，它可能处于某种空状态或重置状态。这也被认为是对象的有效状态，因此是类不变量的一部分。然而，通常只有少数几个成员函数可以在对象处于这种状态时被调用。例如，你不能对一个已被移动的 `std::vector` 调用 `push_back()`、`empty()` 或 `size()`，但你可以调用 `clear()`，这将使向量进入可以再次使用的状态。

不过，你应该意识到，这种额外的重置状态会使类不变量变得更弱且用处更小。为了完全避免这种状态，你应该以一种方式实现你的类，即被移动后的对象被重置到默认构造后的状态。我的建议是始终这样做，除非在极少数情况下，将移动源对象重置到默认状态会带来不可接受的性能损失。通过这种方式，你可以更好地推理论证被移动后的状态，并且类使用起来更安全，因为对该对象调用成员函数是没问题的。

> 如果你能确保一个对象始终处于有效状态（类不变量成立），你很可能会得到一个难以被误用的类，并且如果实现中存在 Bug，它们通常会很容易被发现。你最不希望看到的是在代码库中发现一个类，却不知道该类的某个行为究竟是 Bug 还是特性。违反契约始终是一个严重的 Bug。

为了能够编写有意义的类不变量，我们需要编写具有高内聚性和少数可能状态的类。如果你曾经为你自己编写的类写过单元测试，你可能已经注意到，在编写单元测试的过程中，类的 API 可以得到改进。单元测试迫使你使用并反思类的接口，而不是实现细节。同理，类不变量会让你思考一个对象可能处于的所有有效状态。如果你发现难以定义类不变量，通常是因为你的类承担了过多的责任并处理了过多的状态。因此，定义类不变量通常意味着你最终会得到设计良好的类。


#### 维护契约（Contracts）

契约是你设计和实现的 API 的一部分。但你如何维护契约并将其传达给使用你的 API 的客户端呢？

C++ 目前还没有内置的契约支持，不过正在进行相关工作，计划将其添加到未来的 C++ 版本中。尽管如此，你现在仍有一些可选方案：

* 使用库： 利用像 Boost.Contract 这样的库来帮助实现契约。
* 编写文档： 将契约记录在文档中。这种方法的缺点是，在程序运行时无法对契约进行检查。此外，当代码发生变化时，文档往往会变得过时。
* 使用断言： 使用 `static_assert()`（用于编译期检查）和定义在 `<cassert>` 中的 `assert()` 宏（用于运行时检查）。断言是可移植的，属于标准的 C++ 范畴。
* 构建自定义库： 构建一个自定义库，其中包含类似于断言的宏，但对契约失败的行为有更好的控制。

在本书中，我们将使用断言（Asserts），这是一种检查契约违规的最原始但非常有效的方式。尽管如此，断言仍然可以非常有效，并对代码质量产生巨大的影响。
 


**启用与禁用断言（Asserts）**

从技术上讲，我们有两种标准的 C++ 断言方式：使用 `static_assert()` 或使用 `<cassert>` 头文件中的 `assert()` 宏。

  * `static_assert()` 在代码编译期间进行验证，因此它要求一个可以在编译时而非运行时检查的表达式。一个失败的 `static_assert()` 会导致编译错误。

  * 对于只能在运行时评估的断言，你需要使用 `assert()` 宏。

`assert()` 宏是一种运行时检查，通常在调试和测试阶段处于激活状态，而在程序以 Release 模式构建时则会完全禁用。`assert()` 宏通常是这样定义的：

```
#ifdef NDEBUG
  #define assert(condition) ((void)0)
#else
  #define assert(condition) /* implementation defined */
#endif
```

这意味着，你可以通过定义 `NDEBUG` 宏来完全移除所有的 `assert` 以及检查条件的代码。

现在，你已经掌握了一些契约式设计（Design by Contract）的术语，接下来我们将把重点放在契约违规（错误）以及如何在代码中处理它们。


### 错误处理

设计具有适当错误处理的 API 时，首先要做的就是区分编程错误和运行时错误。因此，在我们深入研究错误处理策略之前，我们将使用契约式设计（Design by Contract）来定义我们正在处理的错误类型。


#### 编程错误还是运行时错误？

如果发现契约被违反，那么我们就发现了程序中的一个错误。例如，如果我们可以检测到有人正在对一个空向量调用 `pop_back()`，我们就知道源代码中至少存在一个需要修复的 Bug。只要前置条件（precondition）没有得到满足，我们就知道正在处理的是一个编程错误。

另一方面，如果一个函数从磁盘加载记录，但由于磁盘读取错误而无法返回该记录，那么我们就检测到了一个运行时错误：

```cpp
auto load_record(std::uint32_t id) {
    assert(id != 0); // 前置条件
    auto record = read(id); // 从磁盘读取，可能会抛出异常
    assert(record.is_valid()); // 后置条件
    return record;
}
```

前置条件得到了满足，但由于程序外部的某些原因（磁盘错误），后置条件无法满足。源代码中没有 Bug，但函数无法返回在磁盘上找到的记录。由于后置条件无法满足，因此必须向调用者报告运行时错误，除非调用者可以通过重试等方式自行恢复。


#### 编程错误（Bug）

通常，编写代码来发出信号并处理代码中的 Bug 是没有意义的。相反，应该使用断言（asserts）（或前面提到的其他替代方案）来让开发人员意识到代码中的问题。你只应该对可恢复的运行时错误使用异常或错误码。

**通过假设来缩小问题空间**

断言指明了你作为代码作者所做的假设。只有当代码中所有断言都成立时，你才能保证代码按预期工作。这使得编码变得容易得多，因为你可以有效地限制你需要处理的情况的数量。断言对于你的团队在使用、阅读和修改你编写的代码时也是一个巨大的帮助。所有的假设都以断言语句的形式清晰地记录下来了。

**用断言查找 Bug**

一个失败的断言总是一个严重的 Bug。在测试过程中发现断言失败时，基本上有三种选择：

  * 断言是正确的，但代码是错误的（可能是函数实现中的 Bug，也可能是调用现场的 Bug）。根据我的经验，这是最常见的情况。写对断言通常比写对围绕断言的代码更容易。修复代码并再次测试。
  * 代码是正确的，但断言是错误的。这种情况偶尔会发生，如果你查看旧代码，通常会感到不舒服。修改或删除一个失败的断言可能会很耗时，因为你需要百分之百确定代码确实工作正常，并理解为什么一个旧的断言会突然开始失败。通常，这是因为原始作者没有考虑到新的使用场景。
  * 断言和代码都是错误的。这通常需要重新设计类或函数。可能需求发生了变化，程序员所做的假设不再成立。但不要绝望；相反，你应该庆幸这些假设是用断言明确写出来的；现在你知道代码不再工作的原因了。

运行时断言需要进行测试，否则断言就不会被执行。新编写的、带有许多断言的代码通常在测试时就会崩溃。这并不意味着你是一个糟糕的程序员；这意味着你添加了有意义的断言，捕获了一些原本可能会进入生产环境的错误。同时，导致程序测试版本终止的 Bug 也更有可能被修复。

**对性能的影响**

代码中包含许多运行时断言很可能会降低测试构建的性能。然而，断言绝不应该用于最终优化的程序版本中。如果你的断言使测试构建慢到无法使用，通常很容易在性能分析器（Profiler）中跟踪到是哪一组断言拖慢了代码（有关性能分析的更多信息，请参阅第三章《分析和测量性能》）。

通过让程序的 Release 构建完全忽略所有类型的编程错误，你的程序将不会花费时间检查由 Bug 引起的错误状态。相反，你的代码运行速度会更快，只花费时间解决它原本要解决的实际问题。它只会检查需要恢复的运行时错误。

总而言之，编程错误应该在测试程序时被检测到。 没有必要使用异常或其他错误处理机制来处理编程错误。相反，编程错误最好能记录有意义的信息并终止程序，以告知程序员需要修复 Bug。遵循此指导方针可以显著减少代码中需要处理异常的地方。我们的优化构建将具有更好的性能，并且因为 Bug 已被失败的断言检测到，所以 Bug 数量有望减少。然而，在某些情况下可能会发生运行时错误，而这些错误需要由我们实现的代码来处理和恢复。

 
#### 可恢复的运行时错误

如果一个函数无法履行其契约的一部分（即后置条件），则表明发生了运行时错误，需要将此错误信号传递给代码中可以处理并恢复有效状态的地方。处理可恢复错误的目的，就是将错误从发生地传递到可以恢复有效状态的地方。实现这一点的方法有很多。这枚硬币有两面：

  * 对于信号传递部分， 我们可以选择 C++ 异常、错误码、返回 `std::optional` 或 `std::pair`，或使用 `boost::outcome` 或 `std::experimental::expected`。
  * 对于程序有效状态的维护（避免资源泄漏）， C++ 中的工具是确定性析构函数（Deterministic Destructors）和自动存储期（Automatic Storage Duration）。

工具类 `std::optional` 和 `std::pair` 将在第九章《基本实用工具》中介绍。我们现在将重点关注 C++ 异常，以及如何在从错误中恢复时避免资源泄漏。

**异常（Exceptions）**

异常是 C++ 提供的标准错误处理机制。该语言就是设计用于异常的。其中一个例子是失败的构造函数；从构造函数发出错误信号的唯一方法是使用异常。

根据我的经验，异常的使用方式多种多样。其中一个原因是不同的应用程序对运行时错误的处理要求可能截然不同。对于某些应用程序，例如起搏器或发电厂控制系统，如果崩溃可能会产生严重影响，我们可能必须处理所有可能的异常情况（例如内存耗尽），并保持应用程序处于运行状态。有些应用程序甚至完全避免使用堆内存，这可能是因为平台根本不提供堆，或者因为堆引入了不可控的不确定性，因为分配新内存的机制超出了应用程序的控制范围。

我假设你已经了解抛出和捕获异常的语法，此处将不再赘述。

一个保证不会抛出异常的函数可以标记为 `noexcept`。 重要的是要理解，编译器并不会验证这一点；相反，判断函数是否可能抛出异常取决于代码的作者。

标记为 `noexcept` 的函数使得编译器在某些情况下可以生成更快的代码。如果一个被标记为 `noexcept` 的函数抛出异常，程序将调用 `std::terminate()` 而不是展开堆栈。下面的代码演示了如何将一个函数标记为不抛出异常：

```cpp
auto add(int a, int b) noexcept {
    return a + b;
}
```

你可能会注意到，本书中的许多代码示例即使在生产代码中是适当的，也没有使用 `noexcept`（或 `const`）。这仅仅是由于书籍的格式所限；在我通常会添加 `noexcept` 和 `const` 的所有地方都加上它们，会使代码难以阅读。

**维护有效状态**

异常处理要求我们程序员思考异常安全保证（exception safety guarantees）；即，在异常发生之前和之后，程序的状态是什么？

强异常安全（Strong Exception Safety）可以视为一个事务（transaction）。一个函数要么提交所有的状态更改，要么在发生异常时执行完全回滚（complete rollback）。

为了更具体地说明这一点，我们来看下面的简单函数：

```cpp
void func(std::string& str) {
    str += f1(); // 可能会抛出异常
    str += f2(); // 可能会抛出异常
}
```

该函数将 `f1()` 和 `f2()` 的结果追加到字符串 `str` 中。现在考虑如果在调用 `f2()` 时抛出异常会发生什么；只有 `f1()` 的结果会被追加到 `str` 中。我们希望的是，如果发生异常，`str` 保持不变。

这可以通过一种称为拷贝并交换（copy-and-swap）的惯用法来修复。这意味着我们在临时副本上执行可能抛出异常的操作，然后才通过不抛出异常的 `swap()` 函数来修改应用程序的状态：

```cpp
void func(std::string& str) {
    auto tmp = std::string{str}; // 拷贝
    tmp += f1(); // 变异副本，可能抛出异常
    tmp += f2(); // 变异副本，可能抛出异常
    std::swap(tmp, str); // 交换，永不抛出异常
}
```

同样的模式可用于成员函数中，以维护对象的有效状态。假设我们有一个类，它有两个数据成员，并且类不变量要求数据成员不能相等，如下所示：

```cpp
class Number { /* ... */ };
class Widget {
public:
    Widget(const Number& x, const Number& y) : x_{x}, y_{y} {
        assert(is_valid()); // 检查类不变量
    }
private:
    Number x_{};
    Number y_{};
    bool is_valid() const { // 类不变量
        return x_ != y_; // x_ 和 y_ 必须不相等
    }
};
```

接下来，假设我们添加了一个更新这两个数据成员的成员函数，如下所示：

```cpp
void Widget::update(const Number& x, const Number& y) {
    assert(x != y && is_valid()); // 前置条件
    x_ = x;
    y_ = y;
    assert(is_valid()); // 后置条件
}
```

前置条件规定 `x` 和 `y` 必须不相等。如果 `x_` 和 `y_` 的赋值操作可能会抛出异常，那么 `x_` 可能被更新了，而 `y_` 没有。这可能导致类不变量被破坏；即，对象处于无效状态。我们希望该函数在发生错误时，能够保持对象在赋值操作之前的有效状态。

同样，一个可能的解决方案是使用拷贝并交换惯用法：

```cpp
void Widget::update(const Number& x, const Number& y) {
    assert(x != y && is_valid()); // 前置条件
    auto x_tmp = x;
    auto y_tmp = y;
    std::swap(x_tmp, x_);
    std::swap(y_tmp, y_);
    assert(is_valid()); // 后置条件
}
```

首先，创建局部副本而不修改对象的状态。然后，如果没有抛出异常，就可以使用不抛出异常的 `swap()` 来改变对象的状态。拷贝并交换惯用法也常用于实现赋值运算符，以实现强异常安全保证。

错误处理的另一个重要方面是避免在发生错误时泄漏资源。

 

**资源获取（Resource Acquisition）**

C++ 对象的销毁是可预测的（predictable），这意味着我们对所获取资源何时以及以何种顺序释放拥有完全控制权。下面的例子进一步说明了这一点：`mutex` 变量 `m` 始终会在函数退出时被解锁，因为 `scoped_lock` 会在我们退出作用域时释放它，无论我们以何种方式、在何处退出函数：

```cpp
auto func(std::mutex& m, bool x, bool y) {
    auto guard = std::scoped_lock{m}; // 锁定互斥量
    if (x) {
        // 自动退出时，guard 会自动释放互斥量
        return;
    }
    if (y) {
        // 如果抛出异常，guard 会自动释放互斥量
        throw std::exception{};
    }
    // 函数正常退出时，guard 会自动释放互斥量
}
```

对象的所有权、生命周期和资源获取是 C++ 中的基本概念，我们将在第七章《内存管理》中进行介绍。

**性能（Performance）**

不幸的是，异常在性能方面有着不好的名声。其中一些担忧是合理的，而另一些则基于编译器实现异常效率不高时的历史观察。然而，如今人们放弃异常的主要有两个原因：

  * 二进制程序大小增加： 即使没有抛出异常，二进制程序的大小也会增加。尽管这通常不是一个问题，但它不符合零开销原则（zero-overhead principle），因为我们为没有使用的功能付出了代价。
  * 抛出和捕获异常相对昂贵： 抛出和捕获异常的运行时成本不是确定性的（deterministic）。这使得异常不适用于具有硬实时（hard real-time）要求的上下文。在这种情况下，返回一个包含返回值和错误码的 `std::pair` 等替代方案可能更好。

另一方面，当没有抛出异常时（即程序遵循成功路径时），异常的表现非常出色。而像错误码这样的其他错误报告机制，即使程序运行没有错误，也需要通过 `if-else` 语句来检查返回码。

异常应该很少发生，通常在异常发生时，异常处理带来的额外性能开销在那些情况下通常不是问题。通常可以在一些对性能至关重要的代码运行之前或之后，执行那些可能抛出异常的计算。这样，我们就可以避免在程序中无法承受异常开销的地方抛出和捕获异常。

为了对异常与其他错误报告机制进行公平比较，明确比较对象很重要。有时，异常被拿来与完全没有错误处理的情况进行比较，这是不公平的；异常需要与提供相同功能的机制进行比较。在测量过异常可能带来的影响之前，不要因为性能原因而放弃异常。你可以在下一章阅读更多关于分析和测量性能的内容。

现在我们将从错误处理转向探索如何使用 Lambda 表达式来创建函数对象。


 
## 函数对象与 Lambda 表达式

Lambda 表达式在 C++11 中引入，并在之后的每一个 C++ 版本中得到增强，是现代 C++ 中最实用的特性之一。它们的通用性不仅在于能够轻松地将函数传递给算法，还在于在许多需要传递代码的场景中的应用，特别是你可以将一个 Lambda 存储在一个 `std::function` 中。

尽管 Lambda 使这些编程技术的使用变得更加简单，但本节中提到的所有功能在没有 Lambda 的情况下也是可以实现的。

Lambda——更正式地说是 Lambda 表达式（lambda expression）——是构造函数对象（function object）的一种便捷方式。但是，我们也可以不使用 Lambda 表达式，而是实现带有重载了 `operator()` 的类，然后实例化这些类来创建函数对象。

我们稍后将探讨 Lambda 与这类类之间的相似之处，但首先我将在一个简单的用例中介绍 Lambda 表达式。

### C++ Lambda 的基本语法

简而言之，Lambda 使程序员能够像传递变量一样轻松地将函数传递给其他函数。

我们来比较一下将 Lambda 传递给算法与传递变量的区别：

```cpp
// 前提条件
auto v = std::vector{1, 3, 2, 5, 4};

// 查找数字 three
auto three = 3;
auto num_threes = std::count(v.begin(), v.end(), three);
// num_threes 是 1

// 查找大于 three 的数字
auto is_above_3 = [](int v) { return v > 3; };
auto num_above_3 = std::count_if(v.begin(), v.end(), is_above_3);
// num_above_3 是 2
```

在第一个例子中，我们向 `std::count()` 传递了一个变量；在后一个例子中，我们向 `std::count_if()` 传递了一个函数对象。这是 Lambda 的一个典型用例：我们传递一个函数，让它被另一个函数（本例中是 `std::count_if()`）多次评估。

此外，Lambda 不需要绑定到一个变量；就像我们可以将一个变量直接放入表达式中一样，我们也可以对 Lambda 执行同样的操作：

```cpp
auto num_3 = std::count(v.begin(), v.end(), 3);
auto num_above_3 = std::count_if(v.begin(), v.end(), [](int i) {
    return i > 3;
});
```

你到目前为止所看到的 Lambda 被称为无状态 Lambda（stateless lambdas）；它们不拷贝或引用 Lambda 外部的任何变量，因此不需要任何内部状态。接下来，我们通过使用捕获块（capture blocks）来引入有状态 Lambda（stateful lambdas），使内容更进阶一些。

### 捕获子句（The Capture Clause）

在前面的示例中，我们将值 `3` 硬编码在 Lambda 内部，以便我们总是统计大于三的数字。如果我们想在 Lambda 内部使用外部变量怎么办？

我们通过将外部变量放入捕获子句（capture clause）中来实现捕获；即 Lambda 中的 `[]` 部分：

```cpp
auto count_value_above(const std::vector<int>& v, int x) {
    auto is_above = [x](int i) { return i > x; };
    return std::count_if(v.begin(), v.end(), is_above);
}
```

在这个例子中，我们通过拷贝变量 `x` 将其捕获到 Lambda 中。如果我们想将 `x` 声明为引用，我们可以在开头放一个 `&`，如下所示：

```cpp
auto is_above = [&x](int i) { return i > x; };
```

该变量现在仅仅是对外部变量 `x` 的引用，就像 C++ 中的常规引用变量一样。当然，我们需要对通过引用传递给 Lambda 的对象的生命周期非常谨慎，因为 Lambda 可能会在一个引用的对象已不存在的上下文中执行。因此，按值捕获更安全。

 

#### 按引用捕获与按值捕获

使用捕获子句来引用和拷贝变量的工作方式与常规变量完全一样。请看下面这两个例子，看看你能否发现其中的区别：

<a>![](/img/chp/ch2/6.png)</a> 

在第一个示例中，`x` 被拷贝到 Lambda 内部，因此当外部的 `x` 发生变化时，Lambda 内部的 `x` 不受影响；结果 `std::count_if()` 统计的是大于 $3$ 的值的数量。

在第二个示例中，`x` 是按引用捕获的，因此 `std::count_if()` 统计的是大于 $4$ 的值的数量。


#### Lambda 与类之间的相似性

我之前提到，Lambda 表达式会生成函数对象（function objects）。函数对象是定义了调用运算符 `operator()()` 的类的一个实例。

要理解 Lambda 表达式由什么组成，你可以将其视为一个具有限制的常规类：

  * 该类只包含一个成员函数（即 `operator()`）。
  * 捕获子句（Capture Clause）是该类的成员变量及其构造函数的组合。

下表展示了 Lambda 表达式及其对应的类结构。左列是按值捕获，右列是按引用捕获：

<a>![](/img/chp/ch2/7.png)</a> 

多亏了 Lambda 表达式，我们不必手动将这些函数对象类型实现为类。
 
 

#### 在捕获中初始化变量

正如前面的例子所示，捕获子句会初始化对应类中的成员变量。这意味着我们也可以在 Lambda 内部初始化成员变量。这些变量将只在 Lambda 内部可见。

下面是一个在捕获中初始化一个名为 `numbers` 的变量的 Lambda 示例：

```cpp
auto some_func = [numbers = std::list<int>{4,2}]() {
    for (auto i : numbers)
        std::cout << i;
};

some_func(); // Output: 42
```

对应的类结构看起来像这样：

```cpp
class SomeFunc {
public:
    SomeFunc() : numbers{4, 2} {}
    void operator()() const {
        for (auto i : numbers)
            std::cout << i;
    }
private:
    std::list<int> numbers;
};

auto some_func = SomeFunc{};
some_func(); // Output: 42
```

当你在捕获中初始化一个变量时，你可以想象变量名之前有一个隐藏的 `auto` 关键字。在这种情况下，你可以认为 `numbers` 被定义为 `auto numbers = std::list<int>{4, 2}`。如果你想初始化一个引用，你可以在名字前使用 `&` 符号，它对应于 `auto&`。下面是一个示例：

```cpp
auto x = 1;
auto some_func = [&y = x]() {
    // y is a reference to x
};
```

再次强调，当引用（而不是拷贝）Lambda 外部的对象时，你必须对生命周期非常谨慎。

也可以将一个对象移动到 Lambda 内部，这在使用像 `std::unique_ptr` 这种只可移动类型时是必要的。方法如下：

```cpp
auto x = std::make_unique<int>();
auto some_func = [x = std::move(x)]() {
    // Use x here..
};
```

这也表明可以对变量使用相同的名称 (`x`)。这不是必需的。相反，我们可以在 Lambda 内部使用其他名称，例如 `[y = std::move(x)]`。

#### 可改变 Lambda 成员变量

由于 Lambda 的工作方式就像一个带有成员变量的类，它也可以改变这些变量。然而，Lambda 的函数调用运算符默认是 `const` 的，因此我们需要通过使用 `mutable` 关键字来显式指定 Lambda 可以改变其成员。在下面的示例中，Lambda 每次调用时都会改变 `counter` 变量：

```cpp
auto counter_func = [counter = 1]() mutable {
    std::cout << counter++;
};

counter_func(); // Output: 1
counter_func(); // Output: 2
counter_func(); // Output: 3
```

如果一个 Lambda 只通过引用捕获变量，我们不必在声明中添加 `mutable` 修饰符，因为 Lambda 本身并没有变异。

`mutable` Lambda 和非 `mutable` Lambda 的区别在下面的代码片段中得到体现：

 <a>![](/img/chp/ch2/8.png)</a> 

在右边的示例中，`v` 是按引用捕获的，Lambda 会变异由 `some_func()` 作用域所拥有的变量 `v`。左边列中可变异的 Lambda 只会变异它自己拥有的 `v` 的拷贝。这就是为什么这两个版本会产生不同的输出结果。

**从编译器的角度看变异成员变量**

为了理解前面示例中发生的情况，我们来看看编译器是如何看待这些 Lambda 对象的：
 
 <a>![](/img/chp/ch2/9.png)</a> 

正如你所看到的，第一种情况对应于一个带有常规成员的类，而按引用捕获的情况则对应于一个成员变量是引用的类。

> 你可能已经注意到，我们给按引用捕获类的 `operator()` 成员函数添加了 `const` 修饰符，并且我们也没有在对应的 Lambda 上指定 `mutable`。这个类仍被视为 `const` 的原因是：我们没有改变实际的类/Lambda 内部的任何东西；实际的改变应用于引用的值，因此该函数仍然被认为是 `const` 的。
  


#### 捕获全部（Capture All）

除了逐个捕获变量之外，我们还可以通过简单地写入 `[=]` 或 `[&]` 来捕获作用域中的所有变量。

  * 使用 `[=]` 意味着所有变量都将按值捕获（by value）。
  * 使用 `[&]` 意味着所有变量都将按引用捕获（by reference）。

如果在成员函数内部使用 Lambda，还可以通过 `[this]` 按引用捕获整个对象，或者通过 `[*this]` 按拷贝捕获整个对象：

```cpp
class Foo {
public:
    auto member_function() {
        auto a = 0;
        auto b = 1.0f;
        
        // 按拷贝捕获作用域内的所有变量 (a, b)
        auto lambda_0 = [=]() { std::cout << a << b; };
        
        // 按引用捕获作用域内的所有变量 (a, b)
        auto lambda_1 = [&]() { std::cout << a << b; };
        
        // 按引用捕获整个对象（即捕获 this 指针，可访问 m_）
        auto lambda_2 = [this]() { std::cout << m_; };
        
        // 按拷贝捕获整个对象（即拷贝 *this，可访问 m_ 的副本）
        auto lambda_3 = [*this]() { std::cout << m_; };
    }
private:
    int m_{};
};
```

请注意，使用 `[=]` 并不意味着作用域中的所有变量都会被拷贝到 Lambda 中；只有实际在 Lambda 内部使用的变量才会被拷贝。

当按值捕获所有变量时，你也可以指定某些变量按引用捕获（反之亦然）。下表展示了捕获块中不同组合的结果：

 <a>![](/img/chp/ch2/10.png)</a> 

尽管使用 `[&]` 或 `[=]` 捕获所有变量很方便，但我建议逐个捕获变量，因为它能清楚地表明 Lambda 作用域内部使用了哪些变量，从而提高了代码的可读性。

好的，这是内容的中文翻译，保留了所有代码并进行了格式化。
 

### 将 C 函数指针赋值给 Lambda

没有捕获（无状态）的 Lambda 可以隐式地转换为函数指针。

假设你正在使用一个 C 库或一个较旧的 C++ 库，它使用一个回调函数作为参数，如下所示：

```cpp
extern void download_webpage(const char* url,
                             void (*callback)(int, const char*));
```

这个回调函数会带着一个返回码和一些下载内容被调用。在调用 `download_webpage()` 时，可以传递一个 Lambda 作为参数。由于回调是一个常规的函数指针，该 Lambda 不得有任何捕获，并且你必须在 Lambda 前面使用一个加号（`+`）：

```cpp
auto lambda = +[](int result, const char* str) {
    // Process result and str
};
download_webpage("http://www.packt.com", lambda);
```

通过这种方式，Lambda 被转换成了一个常规的函数指针。请注意，要使用此功能，该 Lambda 不能有任何捕获。

### Lambda 类型

自 C++20 起，没有捕获的 Lambda 默认是可默认构造（default-constructible）和可赋值（assignable）的。

通过使用 `decltype`，现在可以轻松地构造具有相同类型的不同 Lambda 对象：

```cpp
auto x = [] {}; // 一个没有捕获的 Lambda
auto y = x; // 可赋值
decltype(y) z; // 可默认构造
static_assert(std::is_same_v<decltype(x), decltype(y)>); // 通过
static_assert(std::is_same_v<decltype(x), decltype(z)>); // 通过
```

然而，这仅适用于没有捕获的 Lambda。带有捕获的 Lambda 拥有自己独特的类型。即使两个带有捕获的 Lambda 函数看起来完全相同，它们仍然拥有自己独特的类型。因此，不可能将一个带有捕获的 Lambda 赋值给另一个 Lambda。

 

### Lambda 与 `std::function`

正如前一节所述，带有捕获的 Lambda（有状态 Lambda）不能相互赋值，因为它们具有独特的类型，即使它们看起来完全一样。为了能够存储和传递带有捕获的 Lambda，我们可以使用 `std::function` 来容纳由 Lambda 表达式构造的函数对象。

`std::function` 的签名定义如下：

```cpp
std::function< return_type ( parameter0, parameter1...) >
```

因此，一个不返回任何值且没有参数的 `std::function` 定义如下：

```cpp
auto func = std::function<void(void)>{};
```

一个返回 `bool` 且以 `int` 和 `std::string` 为参数的 `std::function` 定义如下：

```cpp
auto func = std::function<bool(int, std::string)>{};
```

共享相同签名（相同的参数和相同的返回类型）的 Lambda 函数可以被相同类型的 `std::function` 对象所容纳。`std::function` 也可以在运行时被重新赋值。

这里重要的是，Lambda 捕获的内容不影响其签名，因此带有捕获和不带捕获的 Lambda 都可以赋值给同一 `std::function` 变量。下面的代码展示了不同的 Lambda 如何赋值给同一个名为 `func` 的 `std::function` 对象：

```cpp
// 创建一个未赋值的 std::function 对象
auto func = std::function<void(int)>{};

// 将一个不带捕获的 Lambda 赋值给 std::function 对象
func = [](int v) { std::cout << v; };
func(12); // Prints 12

// 将一个带捕获的 Lambda 赋值给同一个 std::function 对象
auto forty_two = 42;
func = [forty_two](int v) { std::cout << (v + forty_two); };
func(12); // Prints 54
```

接下来，让我们将 `std::function` 应用到一个类似于真实世界场景的例子中。
 

#### 使用 `std::function` 实现一个简单的 `Button` 类

假设我们着手实现一个 `Button` 类。然后我们可以使用 `std::function` 来存储对应于点击按钮的操作，这样当我们调用 `on_click()` 成员函数时，相应的代码就会被执行。

我们可以这样声明 `Button` 类：

```cpp
class Button {
public:
    Button(std::function<void(void)> click) : handler_{click} {}
    auto on_click() const { handler_(); }
private:
    std::function<void(void)> handler_{};
};
```

然后，我们可以使用它来创建具有不同操作的多个按钮。这些按钮可以方便地存储在一个容器中，因为它们都具有相同的类型：

```cpp
auto create_buttons () {
    auto beep = Button([counter = 0]() mutable {
        std::cout << "Beep:" << counter << "! ";
        ++counter;
    });
    auto bop = Button([] { std::cout << "Bop. "; });
    auto silent = Button([] {});
    return std::vector<Button>{beep, bop, silent};
}
```

迭代列表并对每个按钮调用 `on_click()` 将执行相应的函数：

```cpp
const auto& buttons = create_buttons();
for (const auto& b: buttons) {
    b.on_click();
}
buttons.front().on_click(); // counter has been incremented
// Output: "Beep:0! Bop. Beep:1! "
```

前面关于按钮和点击处理程序的示例展示了将 `std::function` 与 Lambda 结合使用的一些好处：尽管每个有状态的 Lambda 都将拥有自己独特的类型，但单个 `std::function` 类型可以包装共享相同签名（返回类型和参数）的 Lambda。

另外值得一提的是，你可能已经注意到 `on_click()` 成员函数被声明为 `const`。然而，它正在通过增加其中一个点击处理程序中的 `counter` 变量来改变成员变量 `handler_` 的状态。这似乎违反了 `const` 正确性规则，因为 `Button` 的 `const` 成员函数被允许在其类成员上调用一个变异函数。允许这样做的原因与成员指针被允许在 `const` 上下文中变异其所指向的值的原因相同。在本章前面，我们讨论了如何为指针数据成员传播 `const` 属性。
 
 
#### `std::function` 的性能考量

与直接由 Lambda 表达式构造的函数对象相比，`std::function` 会带来一些性能损失。本节将讨论使用 `std::function` 时需要考虑的一些性能相关问题。

**阻止内联优化**

对于 Lambda 表达式，编译器有能力内联（inline）函数调用；即消除了函数调用的开销。`std::function` 灵活的设计使得编译器几乎不可能内联封装在 `std::function` 中的函数。如果包装在 `std::function` 中的小型函数被频繁调用，阻止内联优化可能会对性能产生负面影响。

**捕获变量的动态内存分配**

如果一个 `std::function` 被赋值给一个带有捕获变量/引用的 Lambda，`std::function` 在大多数情况下将使用堆分配的内存来存储捕获的变量。某些 `std::function` 的实现如果捕获变量的大小低于某个阈值，则不会分配额外的内存（即使用小对象优化，SSO）。

这意味着，不仅由于额外的动态内存分配而产生了性能损失，而且由于堆分配的内存可能增加缓存未命中（cache misses）的数量，性能会更慢（有关缓存未命中的更多信息，请阅读第四章《数据结构》）。

**额外的运行时计算**

调用一个 `std::function` 通常比执行一个 Lambda 要慢一些，因为涉及到的代码更多。对于小型且频繁调用的 `std::function`，这种开销可能会变得非常显著。

想象一下我们定义了一个非常小的 Lambda，如下所示：

```cpp
auto lambda = [](int v) { return v * 3; };
```

接下来的基准测试演示了执行 1000 万次函数调用时，一个明确的 Lambda 类型 `std::vector` 与一个相应 `std::function` 类型的 `std::vector` 之间的性能差异。我们先来看使用明确 Lambda 类型的版本：

```cpp
auto use_lambda() {
    using T = decltype(lambda);
    auto fs = std::vector<T>(10'000'000, lambda);
    auto res = 1;
    // Start clock
    for (const auto& f: fs)
        res = f(res);
    // Stop clock here
    return res;
}
```

我们只测量函数内部循环执行所需的时间。下一个版本将我们的 Lambda 包装在一个 `std::function` 中，看起来像这样：

```cpp
auto use_std_function() {
    using T = std::function<int(int)>;
    auto fs = std::vector<T>(10'000'000, T{lambda});
    auto res = 1;
    // Start clock
    for (const auto& f: fs)
        res = f(res);
    // Stop clock here
    return res;
}
```

我在我的 2018 年 MacBook Pro 上使用 Clang 编译这段代码，并开启了优化（`-O3`）。第一个版本 `use_lambda()` 执行循环大约需要 $2 \text{ ms}$，而第二个版本 `use_std_function()` 执行循环几乎需要 $36 \text{ ms}$。

 
### 泛型 Lambda (Generic Lambdas)

泛型 Lambda 是指接受 `auto` 参数的 Lambda，使其能够以任何类型调用。它的工作方式就像一个常规的 Lambda，但它的 `operator()` 被定义为一个成员函数模板。

只有参数是模板变量，捕获的值不是。换句话说，在下面的例子中，捕获的值 `v` 的类型将是 `int`，无论 `v0` 和 `v1` 的类型是什么：

```cpp
auto v = 3; // int
auto lambda = [v](auto v0, auto v1) {
    return v + v0*v1;
};
```

如果我们将上面的 Lambda 转换为一个类，它将对应于如下所示：

```cpp
class Lambda {
public:
    Lambda(int v) : v_{v} {}
    template <typename T0, typename T1>
    auto operator()(T0 v0, T1 v1) const {
        return v_ + v0*v1;
    }
private:
    int v_{};
};

auto v = 3;
auto lambda = Lambda{v};
```

就像模板版本一样，编译器直到 Lambda 被调用时才会生成实际的函数。因此，如果我们像这样调用前面的 Lambda：

```cpp
auto res_int = lambda(1, 2);
auto res_float = lambda(1.0f, 2.0f);
```

编译器将生成类似于以下 Lambda 的内容：

```cpp
auto lambda_int = [v](int v0, const int v1) { return v + v0*v1; };
auto lambda_float = [v](float v0, float v1) { return v + v0*v1; };
auto res_int = lambda_int(1, 2);
auto res_float = lambda_float(1.0f, 2.0f);
```

正如你可能已经想到的那样，这些版本将像常规 Lambda 一样被进一步处理。

C++20 的一个新特性是，我们可以使用 `typename` 而不是仅仅使用 `auto` 来作为泛型 Lambda 的参数类型。以下泛型 Lambda 是相同的：

```cpp
// 使用 auto
auto x = [](auto v) { return v + 1; };

// 使用 typename
auto y = []<typename Val>(Val v) { return v + 1; };
```

这使得可以命名类型或在 Lambda 主体内引用该类型。

 
## 总结

在本章中，你学习了将贯穿本书使用的现代 C++ 特性。自动类型推导、移动语义和 Lambda 表达式是当今每位 C++ 程序员都需要熟练掌握的基本技术。

我们还花了一些时间研究错误处理，以及如何思考 Bug、有效状态以及如何从运行时错误中恢复。错误处理是编程中一个极其重要但容易被忽视的部分。思考调用者和被调用者之间的契约是使你的代码正确并避免在程序的发布版本中进行不必要的防御性检查的一种方法。

在下一章中，我们将研究分析和测量 C++ 性能的策略。
 
