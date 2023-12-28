---
layout:     post
title:      "Python函数的进阶知识(二)" 
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
 
## Decorator

在一个大型项目中，在每个函数的执行前后都会有一些相同的任务，常见的包括：

* 访问控制
* 清楚临时对象
* 错误处理
* 缓存
* 记日志

在上面这些任务中，通常会存在大量重复性的代码。为了避免每个函数都复制代码，我们可以使用Decorator。从技术的角度来讲，Decorator是一个函数，它的输入是一个函数，输出是另外一个函数。通常它会做一些预处理工作，然后调用真正的函数，最后做一些后处理工作。当然它也可以完全实现一个与原来没有任何关系的函数。

我们首先来看Decorator的用法，这里使用的是suppress_errors，它的实现我们后面再讨论。这里我们只需要知道它的作用是捕获函数的异常并且继续执行就行了。我们先看最最近的方法，这也是Python 3之前能用的方法：

```
import datetime
from myapp import suppress_errors
def log_error(message, log_file='errors.log'):
    log = open(log_file, 'w')
    log.write('%s\t%s\n' % (datetime.datetime.now(), message))
log_error = suppress_errors(log_error)
```

open和write函数都可能抛出异常，suppress_errors会调用被decorated的函数并且捕获其抛出的异常。这种方法比较容易理解，但是有一个问题就是它不是函数的一部分，需要放到函数的后面，而且理论上它可以放到后面的任意位置，中间隔着其它的代码也是可以的(当然要在用之前)。但是如果函数的代码较长，过了一阵之后可能都忘了它的存在。

为了避免这个问题，Python 3增加了一个@的语法：

```
@suppress_errors
def log_error(message, log_file='errors.log'):
    """Log an error message to a file."""
    log = open(log_file, 'w')
    log.write('%s\t%s\n' % (datetime.datetime.now(), message))
```

@语法是在函数定义的前面增加Decorator，它等价于前面的定义，但是好处在于它放到了函数的前面，这样一个函数有那些Decorator一目了然，修改维护都会简单很多。不过它也有一个缺陷，那就是只有函数的定义者可以对其进行Decorator，如果我们需要对第三方开发者提供的函数进行Decorate，那么只能用前面的那种方法。

## 闭包(Closure)

闭包是在一个函数A里定义的另一个函数B，这个函数会被作为返回值从而在函数A之外被使用。比较特殊的一点是B可以使用函数A里的变量(包括参数)。文字比较绕，我们先来看一个例子：

```
def multiply_by(factor):
    def multiply(value):
        return value * factor
    return multiply

times2 = multiply_by(2)
print(times2(2))
```

multiply_by这个函数的作用是产生另外一个函数multiply，比如multiply_by(2)就会返回一个函数，它的作用是对输入乘以2。multiply_by(2)的参数2是传给factor，这个参数在里面的multiply里也是可以使用的，而调用times2(2)的参数2是传给value的，最终执行的结果是value*factor=4。

## Wrapper

闭包通常用于构建Wrapper，这是Decorator的常用模式。Wrapper用于包含另外一个函数A，并且Wrapper的参数是函数B，通过闭包的方式，A能够对B进行一些预处理或者后处理工作(当然理论上也可以完全实现更加复杂的功能)，并且返回一个新的增强版的函数A。这么描述起来有些抽象，我们来看一个例子：

```
def suppress_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass

    return wrapper
```

suppress_errors函数就是一个Wrapper，它包含的A函数就是wrapper，wrapper会作为suppress_errors的返回值。被wrapper的函数B是func，因为闭包，所以在wrapper函数里可以访问到func。这里wrapper的作用就是调用func，并且捕获和忽略其异常。

这里值得注意的是参数的处理。我们先来看一下Wrapper的用法，从而了解这种参数处理方式。我们首先定义一个有异常的函数：

```
def func(a, b=1, *c, **d):
    print(f"a={a}, b={b}, c={c}, d={d}")
    1/0
```

如果直接调用它则会抛出异常"ZeroDivisionError: division by zero"。我们现在希望通过suppress_errors来捕获并忽略异常：

```
func = suppress_errors(func)
func(3, 4, 5, 6, attr1="v1")
```

我们调用被wrapper之后的func后传入的参数是(3, 4, 5, 6, attr1="v1")，最终给到wrapper的是args=(3, 4, 5, 6), kwargs={"attr1": "v1"}，然后wrapper调用真正的func又会用(*args, **kwargs)展开，所以真正的func收到的参数还是(3, 4, 5, 6, attr1="v1")。


另外一个值得注意的就是调用func会return它的返回值。因为虽然我们这里的func没有返回值(返回None)，但是作为一个通用的wrapper函数，我们需要能够调用任意类型的函数。上面的例子虽然很简单，但是它展示里wrapper函数的基本结构，它通常在调用真正的函数前做一些预处理的工作，并且在调用后做一些后处理的工作。不过wrapper有一些问题，那就是原始的信息，包括docstring会丢失。我们来验证一下：

```
def func(a, b=1, *c, **d):
    """
    测试函数
    :param a: 必选参数a
    :param b: 可选参数b
    :param c: 可变位置参数c
    :param d: 可变keyword参数d
    :return: None
    """
    print(f"a={a}, b={b}, c={c}, d={d}")
    1/0
print(func.__doc__)
```
上面的函数会正常的输出docstring，再测试一下wrapper后：
```
func = suppress_errors(func)
print(func.__doc__)
print(func.__name__)
# 输出None和wrapper
```

这个新的func以及是那个内部的wrapper函数了，我们也不可能把func的docstring写到wrapper里，因为wrapper的输入函数func是未知的。为了(一定程度的)解决这个问题，我们可以使用functools.wraps()，它会把被wrapper的函数的名字和docstring等信息复制到返回的wrapper函数里。我们来测试一下：

```
wrappered = suppress_errors(func)
import functools
func = functools.wraps(func)(wrappered)
print(func.__name__)
print(func.__doc__)
```
这里的调用有一些不寻常，我们首先调用functools.wraps(func)，它返回的是一个wrapper，用这个wrapper对wrappered再进行wrapper。我们可以把这个语句分解一下，从而便于理解：

```
doc_wrapper = functools.wraps(func)
func = doc_wrapper(wrappered)
```


因为functools.warps(func)也是通用的语句，所以我们可以把它放到wrapper里：

```
def suppress_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    wrapper = functools.wraps(func)(wrapper)
    return wrapper
```

当然更简单非方法是把functools.wraps(func)也用@语法：
```
def suppress_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    return wrapper
```

## 带参数的Decorator

在前面的例子里我们使用了functools.wraps()，这个Decorator的比较特别的是它会输入一个参数。因为functools.wraps的作用是复制某个函数的名字和docstring等信息，所以functools.wraps(func)返回的是一个Decorator，这个wrapper会把func的名字和docstring复制到被wrapper的函数里。听起来有点绕，我们在来review一下：

```
def suppress_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    return wrapper
```
@functools.wraps(func)首先会复制func的名字和docstring，然后返回一个Decorator，这个Decorator会应用到wrapper函数上，所以它的作用是把func的名字和docstring复制到wrapper上。或者不用@语法可能更加清晰：

```
wrapper = functools.wraps(func)(wrapper)
```

现在我们的目标是给前面的suppress_errors增加一个参数log_func，这是一个函数，用于在捕获异常时打印一些信息。我们需要修改原来的代码，代码如下：

```
import functools

def suppress_errors(log_func=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_func is not None:
                    log_func(str(e))
        return wrapper
    
    return decorator
```

为了清晰对比，我们把修改前的代码再重复一遍：
```
def suppress_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    return wrapper
```

它们有如下区别：
* 原来的suppress_errors的输入是func，也就是要被decorated的函数，而在新的实现里变成了log_func
* 原理的suppress_errors直接用于func进行decorate；而新的实现里suppress_errors返回的是原来的decorator，然后再调用一次才是真正进行decorate

我们来对比一下两者的调用方式，首先看原来的实现：

```
func = suppress_errors(func)
```

新的实现：
```
decorator = suppress_errors(log_func)
func = decorator(func)
```
或者合并成一个语句：
```
func = suppress_errors(log_func)(func)
```

因此我们发现，原来suppress_errors返回的是可以直接对func进行decorate的decorator；而现在suppress_errors返回的是一个decorator。这两者的区分请读者一定理解清楚了再继续下面的阅读，如果还不清楚，可以反复阅读确保理解。

不过这样实现有一个问题：如果我不想传入log_func也需要进行一次函数调用，也就是加一个括号：

```
def func(a, b):
    print(f"a={a}, b={b}")
    1/0

func = suppress_errors()(func)
# 或者 
# func = suppress_errors(None)(func)
func(1, 2)
```

或者用@的语法：
```
@suppress_errors()
def func(a, b):
    print(f"a={a}, b={b}")
    1/0
```

这和我们的前面的习惯不同，前面我们用Decorator时通常就是@加函数名就行。如果不加()会出错，因为它相对于：

```
decorator = suppress_errors(func)
```

它把被decorate的函数当成了log_func传进去，这显然是不行的。


## 同时支持带参数和不带参数的Decorator

对于像log_func可选的参数，我们希望既可以通过函数的方式调用，也可以不通过括号()调用。因此最外层的函数能够处理两种情况：输入log_func这个函数作为参数；输入func作为参数。我们可以这样实现：

```
import functools

def suppress_errors(func=None, *, log_func=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_func is not None:
                    log_func(str(e))
        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)

```

它的用法为：

```

def log_message(s):
    print(s)

@suppress_errors(log_func=log_message)
def func(a, b):
    print(f"a={a}, b={b}")
    1/0

func(1, 2)
```
或者不传递log_func参数：
```
@suppress_errors
def func(a, b):
    print(f"a={a}, b={b}")
    1/0
```


针对两种用法，我们分别来分析详细的调用过程。我们先看第一种带参数的调用方法，它等价于：
```
decorator = suppress_errors(log_func=log_message)
func = decorator(func)
```

它进入suppress_errors是，参数func是默认的None，而log_func是传入的log_message。因为func是None，所以它走的分支是直接返回decorator。decorator的代码和没有参数的wrapper是完全一样的：

```
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_func is not None:
                    log_func(str(e))
        return wrapper
```

接下来调用decorator(func)，则返回其中的wrapper函数，这个函数就是捕获异常并且打印错误日志。这是符合我们预期的，下面我们来看第二种用法：

```
@suppress_errors
def func(a, b):
    print(f"a={a}, b={b}")
    1/0
```

如果把它写出函数调用的形势，它等价于：

```
func = suppress_errors(func)
```

这个时候func是传给第一个参数func，而log_func是None，这个时候走的是else分支代码"decorator(func)"。它返回的是wrapper函数，这也是没有问题的。

所以我们再来总结一下suppress_errors：

```
def suppress_errors(func=None, *, log_func=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_func is not None:
                    log_func(str(e))
        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
```

如果需要传入log_func，则需要使用@supress_errors(log_func=log_message)进行调用，这个时候func是None所以它返回的是decorator，然后接着调用decorator(func)返回wrapper。如果使用@suppress_errors(没有括号)，则调用的是suppress_error(func)，这是走的是else分支，也就是decorator(func)，也是返回wrapper。

那为什么在log_func前面加一个没有命名的可变位置参数(也就是参数*)呢？其实也可以不加，加的目的是把log_func变成keyword only的参数，从而提醒用户使用keyword的方式调用。如果这样定义：

```
def suppress_errors(func=None, log_func=None):
```

那么用户可能会@suppress_errors(log_message)，这样其实就把log_message传给了func，就会出错。


## 例子：实现函数的cache

下面我们使用Decorator来实现一个函数的cache。


```
import functools

def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        print('Calling %s()' % func.__name__)
        result = func(*args)
        cache[args] = result
        return result
    return wrapper
```

这个函数使用闭包返回一个wrapper函数，wrapper函数的作用就是调用真正的函数前看看cache是否已经计算过了，如果cache存在，直接返回结果，否则调用func，并且把结果保存到cache里。

这里有两点值得注意：1. cache是定义在wrapper之外的闭包，这样多次调用都可以访问相同的cache，类似与全局变量(或者函数静态变量)的作用。2. 使用变长位置参数来接受所有参数，而不是通常的"*args, **varages"。这样做的目的是为了便于把所有参数接收为一个tuple从而作为cache的key，但是这也带来一个问题：我们无法传入keyword参数。比如下面的调用会抛出异常：

```
z = multiply(x=2, y=3)
```
异常为："TypeError: wrapper() got an unexpected keyword argument 'x'"。

## 例子：用Decorator来创建Decorator

Decorator本身的目的是为了避免重复的代码，从而简化函数。但是Decorator本身会变得非常复杂，尤其是需要传递参数的Decorator(最复杂的是参数还是可选的)。当然这就有许多重复的代码，因此我们也可以再用一个Decorator来封装它。这像创造了一个工具来解决一个问题，然后又创建另一个工具来解决前一个工具的问题？

```
import functools

def decorator(declared_decorator):
    @functools.wraps(declared_decorator)
    def final_decorator(func=None, **kwargs):
        def decorated(func):
            @functools.wraps(func)
            def wrapper(*a, **kw):
                return declared_decorator(func, a, kw, **kwargs)

            return wrapper

        if func is None:
            return decorated
        else:
            return decorated(func)

    return final_decorator
```

有了上面的代码，我们在创建一个Decorator时只需要关注其逻辑，而不用关心闭包等参数传递的细节。比如要实现supress_erros，我们只需要实现suppress_errors的逻辑：

```
@decorator
def suppress_errors(func, args, kwargs, log_func=None):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_func is not None:
            log_func(str(e))
```

这个suppress_errors函数没有wrapper等代码的存在。它有4个参数，func是被decorated的函数；args是一个tuple，表示可变位置参数；kwargs是一个dict，表示可变keyword参数；这两个参数都是传递给func的，最后一个log_func是可选的，它是真正suppress_errors函数的参数。注意：args和kwargs本身是必选的位置参数。

通过@decorator，我们的suppress_errors可以应用到真正的函数上，比如没有参数的用法：

```
@suppress_errors
def example(a, b=1, *c, **d):
    print(f"a={a}, b={b}, c={c}, d={d}")
    1/0
    return True


example(1, 2, 3, 4, attr1="v1")
```

有参数的用法：
```
def log_message(s):
    print(s)

@suppress_errors(log_func=log_message)
def example(a, b=1, *c, **d):
    print(f"a={a}, b={b}, c={c}, d={d}")
    1/0
    return True
```

我们看看没有参数用法的调用过程。它等价于：

```
suppress_errors = decorator(suppress_errors)
example = suppress_errors(example)
```

decorator(suppress_errors)返回的直接是final_decorator，因此decorator(example)就是final_decorator(example)：
```
    def final_decorator(func=None, **kwargs):
        def decorated(func):
            @functools.wraps(func)
            def wrapper(*a, **kw):
                return declared_decorator(func, a, kw, **kwargs)

            return wrapper

        if func is None:
            return decorated
        else:
            return decorated(func)
```
参数func就是example。因为func非None，因此调用decorated(example)，代码为：

```
            @functools.wraps(func)
            def wrapper(*a, **kw):
                return declared_decorator(func, a, kw, **kwargs)

            return wrapper
```
最终返回的是wrapper。

调用example(1, 2, 3, 4, attr1="v1")时就是调用wrapper，参数a是(1,2,3,4)，kw是{"attr1": "v1"}。最终调用的是declared_decorator(func, a, kw, **kwargs)，也就是：

```
suppress_errors(example, (1,2,3,4), {"attr1": "v1"}, None)

def suppress_errors(func, args, kwargs, log_func=None):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_func is not None:
            log_func(str(e))
```
此时log_func是None，捕获异常后安静的离开。

接下来我们看第二种带参数的调用路径，它等价于：

```
suppress_errors = decorator(suppress_errors)
decorator = suppress_errors(log_func=log_message)
example = decorator(example)
```

第一行代码和前面一样，还是返回final_decorator。接着就是final_decorator(log_func=log_message)：

```
    def final_decorator(func=None, **kwargs):
        def decorated(func):
            @functools.wraps(func)
            def wrapper(*a, **kw):
                return declared_decorator(func, a, kw, **kwargs)

            return wrapper

        if func is None:
            return decorated
        else:
            return decorated(func)
```
此时func是None(之前func是example)，因此直接返回decorated函数。接着调用decorated(example)：

```
    def final_decorator(func=None, **kwargs):
        def decorated(func):
            @functools.wraps(func)
            def wrapper(*a, **kw):
                return declared_decorator(func, a, kw, **kwargs)

            return wrapper
```

代码还前面基本一样，唯一的区别就是原来的kwargs为None，但是此时kwargs={"log_func": log_message函数}，返回的还是wrapper。最终调用exmaple的时候还是进入suppress_errors，和前面唯一的区别就是log_func不是None而是log_message函数：

```
def suppress_errors(func, args, kwargs, log_func=None):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_func is not None:
            log_func(str(e))
```

因此这个时候捕获异常后会调用log_func把异常信息"division by zero"打印出来。


