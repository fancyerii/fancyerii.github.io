---
layout:     post
title:      "Python函数的进阶知识(一)" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - python
    - 默认参数
    - 位置参数
---

本文介绍Python函数相关的进阶知识。
<!--more-->

**目录**
* TOC
{:toc}

函数是Python的一等公民，除了像其它语言那样封装一组代码逻辑被调用。它还和变量一样可以作为另外一个函数的参数或者返回值，并且随时在运行时修改它的代码。在Python里，用一种特殊的对象来表示函数，这个对象和普通的对象一样可以被复制，它的特殊之处在于可以用()来调用这个对象(函数)，比如下面的代码：

```
def example():
    print("func called")

print(type(example))
print(example)
example()
```

输出类似：
```
<class 'function'>
<function example at 0x7fa62a8bae18>
func called
```

example是类function的一个实例，函数对象有个"特殊"的能力(其实也不特殊，没有这个功能还能叫函数吗？)，那就是可以用()语法调用它。

## 参数

### Parameter和Argument的区别
Parameter和Argument都被翻译成中文的"参数"，但是它们的准确含义是不同的。根据[SO问题](https://stackoverflow.com/questions/1788923/parameter-vs-argument)，它们的区别为：在函数原型或者内部定义使用的是parameter；而调用时传入的是argument。比如：

```
void Foo(int i, float f)
{
    // Do things
}

void Bar()
{
    int anInt = 1;
    Foo(anInt, 2.0);
}
```
则i和f是parameter；而anInt和2.0是argument。

### 默认参数

在C/Java等语言中，函数是没有默认参数的。如果一个函数的参数很多，那么传递参数会很麻烦。因为设计函数的时候通常要考虑灵活性，这个时候很多可以由调用者决定的变量会作为参数。大部分用户不需要这种灵活性，但是也需要为这种灵活性付出代价——需要记住不用的参数的默认值并且传入。如果默认值发生变化，那就更是灾难性的后果。当然Java提供了函数重载，比较好的设计是有一个包含所有参数的函数负责实现真正的逻辑，而那些重载的版本只是调用这个函数并且提供默认值。比如：

```
void fun_with_many_params(int a, int b, int c, int d, int e){
    // Do things
}


void fun_with_many_params(int a, int b){
    fun_with_many_params(a, b, 100, 200, 300);
}

void fun_with_many_params(int a, int b, int c){
    fun_with_many_params(a, b, c, 200, 300);
}

void fun_with_many_params(int a, int b, int c, int d){
    fun_with_many_params(a, b, c, d, 300);
}
```

但是如果有默认值的参数很多，那么它们的组合方式也会非常多，写起来非常啰嗦。而且由于函数前面的限制，有些组合是无法实现的。比如上面的例子，我们如果只想传入a、b和d，那么也许我们想这样：

```
void fun_with_many_params(int a, int b, int d){
    fun_with_many_params(a, b, 100, d, 300);
}
```
但这是不行的！因为函数重载不能通过参数名字来区别，在执行fun_with_many_params(1,2,3)时不可能知道3是传给c还是d，所以为了避免这种情况就会在编译时就不允许出现这样的代码，从而给出编译错误。

而C++和Python提供了默认参数，也就是说如果调用者不传入，则使用默认值：

```
def fun_with_many_params(a, b, c=100, d=200, e=300):
    pass
```
注意有默认值的参数后面不能出现必选参数(没有默认值的参数)，比如下面的定义会出现编译错误：
```
def fun_with_many_params(a, b, c=100, d=200, e=300, f):
    pass
```
为什么呢？因为假设允许这样，则fun_with_many_params(1,2,3)的参数3是传给c还是f呢？也许读者会说传给c，那么fun_with_many_params(1,2,3,c=4)中的3传给谁呢？另外如果想传如f，那么就必须得传入c、d和e，这也让默认参数失去的意义。因此Python规定默认参数后面不能出现必选参数。

注意：Python是没有函数重载的概念的，一个函数名只能对于一个函数的定义。如果用同一个名称定义多次，即使参数个数(Python不是静态语言，所以也就不可能根据参数类型区分函数)不同也是会覆盖原来的定义的，比如：

```
def func(a, b):
    print("a={}, b={}".format(a, b))
def func(e, f, g):
    print("e={}, f={}".format(e, f))
func(1, 3) # 缺少参数g
```
### 可变长的位置参数

这里出现了位置(Positional)参数的概念，对于C/C++/Java等语言来说，所有的参数都是位置参数，因此不用特意区别。而Python有在调用的时候可以使用变量名进行调用。比如：

```
def func(a, b, c):
    pass

func(1, 2, 3)
func(b=3, a=1, c=2)
```
其它语言通常只能按照函数定义的顺序一个一个传入参数，但是Python可以使用参数(parameter)的名称进行传递，而且可以乱序，这就是keyword传递参数的方式。同时也可以混合这两种方式，比如：

```
func(1, b=2, c=3)
```
但是不能反过来：
```
func(b=2, c=3, 1)
```
因为和默认参数类似，keyword参数后面不运行出现位置参数。编译错误为："SyntaxError: positional argument follows keyword argument"。



可变位置参数是为了解决变长参数的问题，比如假设有个ShoppingCart类，有个方法是往购物车增加商品：

```
class ShoppingCart:
    def add_to_cart(items):
        self.items.extend(items)
```

如果只有一个商品，我们调用时也得构造一个list：
```
cart.add_to_cart([item])
```

这样就很不方便。其实相对其它语言来说还好了，设想一下Java语言：
```
List<Item> items = new ArrayList<Item>(1);
items.add(item);
cart.add_to_cart(items);
```

这时可以使用可变的位置参数来定义函数，它的语法是在参数前面加一个*：

```
    def add_to_cart(*items):
        self.items.extend(items)
```
调用方法为：

```
cart.add_to_cart(item)
cart.add_to_cart(item1, item2)
cart.add_to_cart(item1, item2, item3, item4, item5)
```
在函数里拿到的items是一个tuple。读者可能会问，如果我有一个item的list呢？那我总不能展开成item[0],item[1], ....传入吧。这时我们可以在调用的时候在list前加一个*，Python会把它展开。

```
items = ["item"+str(i) for in range(10)]
cart.add_to_cart(*items)
```

不过注意：上面的代码Python会把items转换成一个tuple然后传入，因此不适合用这种方式传入大量的数据。


### 可变长keyword参数(parameter)

有的时候某个函数的参数个数是不确定的，比如这个函数只是把这些参数传递给另外一个函数。如果显式的定义这些参数，那么当另一个函数修改参数时这个函数也得跟着改。一种解决办法是把这些参数放到一个dict里，比如：

```
class ShoppingCart:
    def __init__(self, options):
        self.options = options
```

如果要不这些参数传给另外某个真正要使用的函数呢，则可以使用\*\*展开成keyword参数调用：
```
    def call_another_func(self):
        another_func(**self.options)
```
假设options={"a":1, "b":2}，则相当于调用another_func(a=1, b=2)。


但是ShoppingCart的调用者还得手动构造一个dict：

```
options = {'currency': 'USD'}
cart = ShoppingCart(options)
```
或者把两个语句合并成一个：
```
cart = ShoppingCart({'currency': 'USD'})
```

但是这样还是不像传递参数。我们这是可以使用与变长位置参数类似的方法——变长keyword参数：

```
    def __init__(self, **options):
        self.options = options
```

也就是在参数前面加两个*，这个时候就可以这样调用：
```
cart = ShoppingCart(currency='USD', usertype='VIP')
```
而__init__方法得到的options则是一个dict。

注意：变长位置参数拿到的是一个不能修改的tuple；而变长keyword参数拿到的是一个dict，它是可以修改的。


### 四种类型的参数

通过前文的介绍，函数总共有4种类型的参数：

* 必选(位置)参数
* 可选参数
* 变长位置参数
* 变长keyword参数

那怎么把这4种类型的参数放到一个函数的定义中呢，最自然常见的做法是：
```
def create_element(name, editable=True, *children, **attributes):
```

首先是必选参数，然后是可选参数，变长位置参数，最后是变长keyword参数。但是这样定义有一个问题——我们如果想要传入变长位置参数时一定得传入可选参数。这样一来可选参数就失去了实际意义。读者可能会问能不能这样传入参数：
```
create_element("name", child1, child2)
```

但是程序并不知道child1代表什么含义，因此它只能按照顺序把child1传给editable。为了解决这个问题，我们需要把变长位置参数放到可选参数之前：

```
def create_element(name, *children, editable=True, **attributes):
```

不过这个时候如果需要传递可选参数就只能通过keyword的方法传入了，比如：

```
create_element("name", "c1", "c2", False, attr1="value", attr2="value2")
```

读者是不是期望children是("c1", "c2")而editable是False呢？很可惜，实际的children是("c1", "c2", False)，而editable是默认的True。为什么呢？因为Python是动态语言，它不能通过类型来匹配，比如：
```
create_element("name", "c1", "c2", "c3", attr1="value", attr2="value2")
```

这个调用和前面的参数个数一模一样，因此传递的参数方法也是一样的。但是我们可能期望"c3"是属于children而不是editable的。这种"语义"的判断是无法实现的，所以Python规定可变位置参数会"贪婪"的匹配尽可能多的参数(keyword参数它匹配不了)。所以如果我们要传入可选参数editable的话就只能通过keyword的方法：

```
create_element("name", "c1", "c2", editable=False, attr1="value", attr2="value2")
```

因为放到可变位置参数之后的位置参数(有没有默认值都有一样)只能通过keyword的方式传入，所以这类参数叫做keyword only参数，在inspect模块中通常简称kwonlyargs。注意：kwonly的参数可以有默认值从而是可选的，也可以没有默认值从而是必须的。比如：

```
def func(required1, *args, option1=0, required2):
    pass
```
参数required2是必须的，但是只能通过keyword的方式传入。我们甚至可以不命名变长位置参数，从而目的只是让某些参数变成keyword only，比如：
```
def func(a, *, b, **kwargs):
    pass

func(1, b=4)
```


### Partial函数

对于一个函数，我们可以提前设置(preload)它的某些参数，从而得到一个参数更少的函数。比如：
```
def func(a, b, c, d):
    print(f"a={a}, b={b}, c={c}, d={d}")

import functools
f2 = functools.partial(func, 1, 2)
f2(3, 4)
# 输出：a=1, b=2, c=3, d=4
```
partial函数也支持keyword参数的传递方式：
```
f3 = functools.partial(func, c=1, d=2)
f3(3, 4)
# 输出：a=3, b=4, c=1, d=2
```


注意：Partial函数只是提前设置参数得到一个新的函数，它和currying是不同的，它们的区别参考[这里](https://stackoverflow.com/questions/218025/what-is-the-difference-between-currying-and-partial-application)。

### 函数的自省(Introspection)

函数在Python内部也是表示成一个对象，我们可以通过inspect在模块运行时检查函数的各个方面。我们可以通过getfullargspec()函数获得函数参数的详细信息，其中包括：

* args
    * 固定位置参数，不包括keyword参数、可变位置参数和可变keyword参数
* varargs
    * 可变位置参数，最多一个
* defaults
    * 固定位置参数的默认值，可能小于args的个数，它的第一个值对应args的第一个默认参数
* varkw
    * 可变keyword参数，最多一个
* kwonlyargs
    * keyword参数
* kwonlydefaults
    * keyword参数的默认值，和defautls不同，它是个dict
* annotations:
    * 后面再讲 

这些概念有些抽象，我们看一个例子：
```
import inspect

def func(a, b=2, *c, d, e=3, **f)-> str:
    print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")

func(1, 2, 3, 4, 5, d=6, attr1="v1", attr2="v2")

spec = inspect.getfullargspec(func)
print(spec)
```

输出为：
```
FullArgSpec(args=['a', 'b'], varargs='c', varkw='f', defaults=(2,), kwonlyargs=['d', 'e'], kwonlydefaults={'e': 3}, annotations={'return': <class 'str'>})
```

对于上面定义的func，各个参数的含义为：
* args
    * 固定位置参数list，包括a和b
* defaults
    * 这是个tuple，说明args中默认参数b的默认值为2。这里需要注意：2为什么是b的默认值而不是a的呢？因为根据语法，默认值参数总是出现在最后面，所以defaults的最后一个值对应args的最后一个参数；defaults的倒数第二个对应args的倒数第二个；……。
* varargs
    * 可变位置参数c
* varkw
    * 可变keyword参数f
* kwonlyargs
    * keyword参数list，这里包括d和e
* kwonlydefaults
    * keyword参数的默认值，这里是个dict，key就是keyword参数名，value就是默认值

注意：args和defaults的对应关系是比较复杂的，是需要倒过来对应；而kwonlydefaults是个dict，所以比较简单。

### 例子：确定参数(Argument)的值

比如我们想在log里记录函数调用的参数，我们当然可以在函数的开头手工的打印，就像前面的代码：
```
def func(a, b=2, *c, d, e=3, **f)-> str:
    print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")
```

但是这有一个问题，首先是很麻烦，每个函数开头都得加上一段代码，而且增加或者删除参数后都得改这行代码。我们可以通过inspect模块在运行时获取这些信息，从而实现一个通用的函数来打印所有的参数。

我们下面一步一步来实现一个get_arguments函数，它的原型为：
```
def get_arguments(func, args, kwargs):
    pass
```
其中func就是函数，而args和kwargs是传给它的位置参数和keyword参数。可能有读者会问：我调用的是"func(1,2,3,4,d=5,attr1="v1")"，这么多参数怎么变成两个参数args和kwargs的呢？后面我们介绍Decorator时会解释，总之现在我们知道这样的调用最终args=(1,2,3,4)而kwargs={"d":5, "attr1":"v1"}就行了。

现在的问题就是，我们知道func对象，从而可以通过inspect得到的FullArgSpec信息结合传入的args和kwargs推出真正的参数，并且需要小心处理参数值和名称的对应关系以及默认值。我们采取迭代的方法逐步完善这个函数。

#### keyword参数

我们首先处理传入的kwargs，这是一个dict，因此参数名和值的对应关系是一目了然的：
```
def example(a, b=2, *c, d, e=3, **f)-> str:
    print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")


def get_arguments(func, args, kwargs):
    arguments = kwargs.copy()
    return arguments

args = (1, )
varargs = {'attr1': "v1", "attr2": "v2", 'd': 5}
print(get_arguments(example, args, varargs))
example(*args, **varargs)
```

输出：
```
{'attr1': 'v1', 'attr2': 'v2', 'd': 5}
a=1, b=2, c=(), d=5, e=3, f={'attr1': 'v1', 'attr2': 'v2'}
```

我们发现可变的keyword参数包括attr1和attr2，而固定的keyword参数d为5，但是get_argument的输出还缺了e，因为e是有默认值的，调用这并没有传入，所以我们需要从默认值里获取它。不过在处理默认值之前，我们先处理固定的位置参数。

#### 固定的位置参数

这部分也非常简单，spec.args里就是所有的位置参数(包括有默认值的)，而传入的args就是位置参数。

```
def get_arguments(func, args, kwargs):
    arguments = kwargs.copy()
    spec = inspect.getfullargspec(func)
    arguments.update(zip(spec.args, args))
    return arguments
```

修改后执行的结果如下：
```
{'attr1': 'v1', 'attr2': 'v2', 'd': 5, 'a': 1}
a=1, b=2, c=(), d=5, e=3, f={'attr1': 'v1', 'attr2': 'v2'}
```

注意：和keyword参数一样，由于有的参数有默认值，spec.args=["a", "b"]，而args=(1, )，所以b取的是默认值，这也需要后面处理。


#### 处理默认值

对于上面的函数，spec.args=["a", "b"]，spec.defaults=[2]，所以参数b的默认值都是1。

```
    if spec.defaults:
        for i, name in enumerate(spec.args[-len(spec.defaults):]):
            if name not in arguments:
                arguments[name] = spec.defaults[i]
```

这段代码有点复杂，我们首先看spec.args[-len(spec.defaults):]，结合前面的具体例子，spec.args[-len(spec.defaults):]就是spec.args[-1:]，所以就是["b"]。用自然语言描述就是：默认参数的长度为l=len(spec.defaults)，然后spec.arg[-l:]就是与之对应的有默认值的参数名。

这里还要判断这个变量命是否已经通过args或者varargs传入了，如果没有传入则使用默认值，否则使用传入的值，这就是"if name not in arguments"语句的作用。


这是执行get_arguments返回的就是：
```
{'attr1': 'v1', 'attr2': 'v2', 'd': 5, 'a': 1, 'b': 2}
```

我们看到b是默认值2，而a是传入的1。我们也可以测试一下传入b：

```
args = (1, 10)
varargs = {'attr1': "v1", "attr2": "v2", 'd': 5}
print(get_arguments(example, args, varargs))
```

这个时候输出的是：
```
{'attr1': 'v1', 'attr2': 'v2', 'd': 5, 'a': 1, 'b': 10}
```

当然我们也可以通过keyword传入b：

```
args = (1, )
varargs = {'attr1': "v1", "attr2": "v2", 'd': 5, 'b': 10}
print(get_arguments(example, args, varargs))
```

结果和上面是一样的！


#### 处理keyword only的参数的默认值

处理keyword参数比较简单，因为spec.kwonlydefaults是个dict：
```
    if spec.kwonlydefaults:
        for name, value in spec.kwonlydefaults.items():
            if name not in arguments:
                arguments[name] = value
```

执行测试代码：
```
args = (1, )
varargs = {'attr1': "v1", "attr2": "v2", 'd': 5}
print(get_arguments(example, args, varargs))
```

输出为：
```
{'attr1': 'v1', 'attr2': 'v2', 'd': 5, 'a': 1, 'b': 2, 'e': 3}
```

#### 处理变长位置参数

```
    if spec.varargs:
        arguments[spec.varargs] = args[len(spec.args):]
```

执行测试代码：
```
args = (1, 2, 3, 4)
varargs = {'attr1': "v1", "attr2": "v2", 'd': 5}
print(get_arguments(example, args, varargs))
example(*args, **varargs)
```

结果为：
```
{'attr1': 'v1', 'attr2': 'v2', 'd': 5, 'a': 1, 'b': 2, 'e': 3, 'c': (3, 4)}
```

变成参数c是(3, 4)。


#### 重构

对于默认参数，我们的理解是这样的：默认参数的值初始化为默认值，如果传入时有这个参数，则更新为传入的值。基于这个理解，我们可以简化上面的代码：

```
    arguments = {}
    spec = inspect.getfullargspec(func)
    if spec.defaults:
        arguments.update(zip(reversed(spec.args), reversed(spec.defaults)))
```

我们首先构造一个空的arguments的dict，然后对必选位置参数设置默认值。这里比前面简单，因为spec.defaults是和spec.args倒过来对齐的，因此使用reversed倒置后zip就行。

接着处理kwonlydefaults，这个更简单：

```
    if spec.kwonlydefaults:
        arguments.update(spec.kwonlydefaults)
```

接着设置传入的位置参数和keyword参数：
```
    arguments.update(zip(spec.args, args))
    arguments.update(kwargs)
```

完整代码如下：
```
def get_arguments(func, args, kwargs):
    arguments = {}
    spec = inspect.getfullargspec(func)
    if spec.defaults:
        arguments.update(zip(reversed(spec.args), reversed(spec.defaults)))
    if spec.kwonlydefaults:
        arguments.update(spec.kwonlydefaults)
    arguments.update(zip(spec.args, args))
    arguments.update(kwargs)
    return arguments
```

和前面比简单了很多！

 

