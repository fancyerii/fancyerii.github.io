---
layout:     post
title:      "Python类型检查" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - python
    - typing 
---

本文是[Python Type Checking (Guide)](https://realpython.com/python-type-checking/)的翻译。

<!--more-->

**目录**
* TOC
{:toc}

在本指南中，你将深入了解Python类型检查。传统上，类型由Python解释器以一种灵活但隐式的方式处理。Python的最新版本允许你指定显式的类型提示，这些提示可以被不同的工具使用，以帮助你更高效地开发代码。

在本教程中，你将学到以下内容：

* 类型注解和类型提示
* 向代码添加静态类型，包括你自己的代码和他人的代码
* 运行一个静态类型检查器
* 在运行时强制类型

这是一份全面的指南，涵盖了很多内容。如果你只想快速了解类型提示在Python中的工作原理，并查看类型检查是否适用于你的代码，你不需要全部阅读。"Hello Types"和"Pros and Cons"这两个部分将让你初步了解类型检查的工作方式，并提供关于何时使用它会有用的建议。


## 类型系统

所有编程语言都包含一种类型系统，它规范了它可以处理哪些对象的类别以及这些类别如何被处理。例如，一个类型系统可以定义一个数字类型，其中42是数字类型对象的一个示例。

### 动态类型

Python是一种动态类型语言。这意味着Python解释器只在代码运行时进行类型检查，并且变量的类型允许在其生命周期内发生变化。以下是一些虚拟示例，演示了Python具有动态类型：

```
>>> if False:
...     1 + "two"  # This line never runs, so no TypeError is raised
... else:
...     1 + 2
...
3

>>> 1 + "two"  # Now this is type checked, and a TypeError is raised
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

在第一个例子中，分支1 + "two"从未运行，因此它从未进行类型检查。第二个例子显示，当评估1 + "two"时，它会引发TypeError，因为在Python中不能将整数和字符串相加。

接下来，让我们看看变量是否可以更改类型：

```python
>>> thing = "Hello"
>>> type(thing)
<class 'str'>

>>> thing = 28.1
>>> type(thing)
<class 'float'>
```
type()返回对象的类型。这些示例证实了thing的类型允许更改，并且Python在更改时正确推断出类型。

### 静态类型

动态类型的相反是静态类型。静态类型检查在运行程序之前进行。在大多数静态类型的语言中，例如C和Java，这是在编译程序时完成的。

在静态类型中，通常不允许变量更改类型，尽管可能存在将变量转换为不同类型的机制。

让我们看一个来自静态类型语言的快速示例。考虑以下Java代码片段：
 
```java
String thing;
thing = "Hello";
```

第一行声明变量名thing在编译时绑定到String类型。该名称永远不能被重新绑定到另一种类型。在第二行中，给thing赋了一个值。它永远不能被赋予不是String对象的值。例如，如果稍后说thing = 28.1f，编译器将引发错误，因为类型不兼容。

Python将始终保持一种动态类型语言。然而，PEP 484引入了类型提示，这使得对Python代码进行静态类型检查成为可能。

与大多数其他静态类型语言中类型的工作方式不同，类型提示本身并不导致Python强制执行类型。正如名称所示，类型提示只是建议类型。还有其他工具，稍后你会看到，使用类型提示进行静态类型检查。

### 鸭子类型
在讨论Python时经常使用的另一个术语是鸭子类型。这个别名来自于短语“如果它走起来像鸭子，叫起来像鸭子，那么它一定是鸭子”（或其任何变体）。

鸭子类型是与动态类型相关的概念，其中对象的类型或类不如它定义的方法重要。使用鸭子类型时，根本不检查类型。相反，你检查给定方法或属性的存在。

例如，你可以对任何定义了.\_\_len\_\_()方法的Python对象调用len()：

```python
>>> class TheHobbit:
...     def __len__(self):
...         return 95022
...
>>> the_hobbit = TheHobbit()
>>> len(the_hobbit)
95022
```



请注意，对len()的调用返回.\_\_len\_\_()方法的返回值。事实上，len()的实现基本等效于以下内容：

```python
def len(obj):
    return obj.__len__()
```

为了调用len(obj)，obj唯一的真正约束是它必须定义一个.len()方法。否则，对象可以是不同类型的，如str、list、dict或TheHobbit。

在对Python代码进行静态类型检查时，使用结构子类型支持鸭子类型。稍后你会了解更多关于鸭子类型的内容。

## 你好，类型

在这一节中，你将看到如何给函数添加类型提示。以下函数通过添加适当的大写和装饰线将文本字符串转换为标题：

```python
def headline(text, align=True):
    if align:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f" {text.title()} ".center(50, "o")
```

默认情况下，该函数返回带有下划线的左对齐标题。通过将align标志设置为False，你可以选择让标题居中，并带有o的周围线：

```python
>>> print(headline("python type checking"))
Python Type Checking
--------------------

>>> print(headline("python type checking", align=False))
oooooooooooooo Python Type Checking oooooooooooooo
```

是时候使用我们的第一个类型提示了！要向函数添加有关类型的信息，你只需像下面这样注释其参数和返回值：

```python
def headline(text: str, align: bool = True) -> str:
    ...
```

text: str语法表示text参数应该是str类型。同样，可选的align参数应该具有默认值True的bool类型。最后，-> str表示headline()将返回一个字符串。

在风格方面，PEP 8建议以下内容：

* 对于冒号使用正常规则，即冒号前不要空格，冒号后有一个空格：text: str。
* 在将参数注释与默认值组合时，在等号前后增加空格：align: bool = True。
* 在->箭头前后使用空格：def headline(...) -> str。

像这样添加类型提示没有运行时效果：它们只是提示，并不会自行执行。例如，如果我们为（诚然命名不佳的）align参数使用了错误的类型，代码仍然可以正常运行，没有任何问题或警告：

```python
>>> print(headline("python type checking", align="left"))
Python Type Checking
--------------------
```
注意：这似乎可以工作的原因是字符串"left"被认为是真值。使用align="center"将不会产生预期的效果，因为"center"也是真值。

为了捕捉这种类型的错误，你可以使用静态类型检查器。也就是说，一种在传统意义上不运行代码但检查代码类型的工具。

你的编辑器可能已经内置了这样的类型检查器。例如，PyCharm会立即给出警告：

<a>![](/img/python-type-checking/1.png)</a>

进行类型检查的最常见工具是Mypy。你稍后将简要介绍一下Mypy，而后你可以更深入地了解它的工作原理。

如果你的系统上还没有安装Mypy，可以使用pip进行安装：

```shell
$ pip install mypy
```

将以下代码放入一个名为headlines.py的文件中：

```python
# headlines.py

def headline(text: str, align: bool = True) -> str:
    if align:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f" {text.title()} ".center(50, "o")

print(headline("python type checking"))
print(headline("use mypy", align="center"))
```







这本质上与你之前看到的代码相同：headline()的定义以及使用它的两个示例。

现在在这段代码上运行Mypy：

```shell
$ mypy headlines.py
```

基于类型提示，Mypy能够告诉我们在第10行使用了错误的类型。

为了修复代码中的问题，你应该更改传递的align参数的值。你还可以将align标志重命名为更少混淆的名称：

```python
# headlines.py

def headline(text: str, centered: bool = False) -> str:
    if not centered:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f" {text.title()} ".center(50, "o")

print(headline("python type checking"))
print(headline("use mypy", centered=True))
```

在这里，你将align更改为centered，并在调用headline()时正确使用了布尔值。现在代码通过了Mypy：

```shell
$ mypy headlines.py
Success: no issues found in 1 source file
```

成功消息确认没有检测到类型错误。Mypy的旧版本通常通过根本不输出来指示这一点。此外，当你运行代码时，你会看到预期的输出：

```shell
$ python headlines.py
Python Type Checking
--------------------
oooooooooooooooooooo Use Mypy oooooooooooooooooooo
```


第一个标题左对齐，而第二个标题居中。



## 优缺点

前面的部分让你对Python中的类型检查有了一点了解。你也看到了向代码添加类型的一个优点的示例：类型提示有助于捕获某些错误。其他优点包括：

* 类型提示有助于文档化你的代码。传统上，如果你想要记录函数参数的预期类型，你会使用文档字符串。这种方法虽然有效，但由于没有文档字符串的标准（尽管有PEP 257，但它们不能轻松用于自动检查）。
* 类型提示改进了IDE和代码检查器。它们使得静态推断代码变得更加容易。这反过来又使得IDE能够提供更好的代码补全和类似的功能。使用类型注解，PyCharm知道text是一个字符串，并且可以根据此给出具体的建议。

<a>![](/img/python-type-checking/1.png)</a>

* 类型提示有助于构建和维护更清晰的架构。编写类型提示的行为迫使你思考程序中的类型。尽管Python的动态性是其伟大的优点之一，但意识到依赖鸭子类型、重载方法或多个返回类型是一件好事。

当然，静态类型检查并不全是好处。你还应该考虑一些缺点：

* 添加类型提示需要开发者的时间和精力。即使它可能节省了调试时间，但你会花更多的时间输入代码。
* 类型提示在现代Python中效果最好。注释是在Python 3.0中引入的，而在Python 2.7中可以使用类型注解。然而，像变量注释和延迟评估类型提示这样的改进意味着你在使用Python 3.6甚至Python 3.7进行类型检查时会有更好的体验。
* 类型提示会在启动时引入轻微的性能损耗。如果你需要使用typing模块，导入时间可能会很显著，特别是在短脚本中。

那么，你应该在自己的代码中使用静态类型检查吗？嗯，这不是一个非此即彼的问题。幸运的是，Python支持渐进式类型。这意味着你可以逐渐将类型引入你的代码中。没有类型提示的代码将被静态类型检查器忽略。因此，你可以从关键组件开始添加类型，并在它对你有价值的情况下继续添加。

看一下上面列出的优缺点列表，你会注意到添加类型对你的运行程序或程序的用户没有任何影响。类型检查旨在使你作为开发者的生活更加便利。

关于是否向项目添加类型的一些经验法则是：

* 如果你刚开始学习Python，你可以等到更有经验时再添加类型提示。
* 在短期使用的一次性脚本中，类型提示的价值很小。
* 在将由他人使用的库中，特别是在PyPI上发布的库中，类型提示提供了很大的价值。其他使用你的库的代码需要这些类型提示才能得到适当的类型检查。有关使用类型提示的项目示例，请参见cursive_re、black、我们自己的Real Python Reader和Mypy本身。
* 在更大的项目中，类型提示可以帮助你了解类型在代码中的流动，并且强烈推荐使用。特别是在与他人合作的项目中。

在他出色的文章《Python类型提示的现状》中，Bernát Gábor建议“只要单元测试是值得编写的，就应该使用类型提示”。的确，类型提示在你的代码中扮演了与测试类似的角色：它们帮助你作为开发者编写更好的代码。

希望现在你对Python中的类型检查工作原理有了一些了解，并且是否在自己的项目中使用它。在本指南的其余部分，我们将更详细地介绍Python类型系统，包括如何运行静态类型检查器（特别关注Mypy）、如何对使用没有类型提示的库的代码进行类型检查，以及如何在运行时使用注解。

### 函数注解

对于函数，你可以注释参数和返回值。具体如下所示：

```python
def func(arg: arg_type, optarg: arg_type = default) -> return_type:
    ...
```

对于参数，语法是argument: annotation，而返回类型使用-> annotation进行注释。请注意，注释必须是一个有效的Python表达式。

下面的简单示例为一个计算圆周长的函数添加了注释：

```python
import math

def circumference(radius: float) -> float:
    return 2 * math.pi * radius
```

运行代码时，你还可以检查注释。它们存储在函数的特殊.__annotations__属性中：

```python
>>> circumference(1.23)
7.728317927830891

>>> circumference.__annotations__
{'radius': <class 'float'>, 'return': <class 'float'>}
```


有时，你可能会对Mypy如何解释你的类型提示感到困惑。对于这些情况，有特殊的Mypy表达式：reveal_type()和reveal_locals()。你可以在运行Mypy之前将它们添加到你的代码中，Mypy会忠实地报告它推断出的类型。例如，将以下代码保存到reveal.py中：

```python
# reveal.py

import math
reveal_type(math.pi)

radius = 1
circumference = 2 * math.pi * radius
reveal_locals()
```

然后，通过Mypy运行此代码：

```shell
$ mypy reveal.py
reveal.py:4: error: Revealed type is 'builtins.float'

reveal.py:8: error: Revealed local types are:
reveal.py:8: error: circumference: builtins.float
reveal.py:8: error: radius: builtins.int
```

即使没有任何注释，Mypy也正确地推断出了内置的math.pi的类型，以及我们的局部变量radius和circumference的类型。

注意：reveal表达式仅用作帮助你添加类型和调试你的类型提示的工具。如果尝试将reveal.py文件作为Python脚本运行，它将崩溃并显示NameError，因为reveal_type()不是Python解释器已知的函数。

如果Mypy说“Name 'reveal_locals' is not defined”，则可能需要更新Mypy安装。reveal_locals()表达式在Mypy版本0.610及更高版本中可用。

### 变量注释

在前面章节中的circumference()定义中，你只为参数和返回值添加了注释，没有在函数体内添加任何注释。在大多数情况下，这已经足够了。

然而，有时类型检查器需要帮助来确定变量的类型。变量注释在PEP 526中定义，并在Python 3.6中引入。其语法与函数参数注释相同：

```python
pi: float = 3.142

def circumference(radius: float) -> float:
    return 2 * pi * radius
```
变量pi已经被标注为float类型。

注意：静态类型检查器完全能够确定3.142是一个浮点数，因此在此示例中，对pi的注释并不是必需的。随着你对Python类型系统的了解越来越多，你会看到更相关的变量注释示例。

变量的注释存储在模块级别的__annotations__字典中：

```python
>>> circumference(1)
6.284

>>> __annotations__
{'pi': <class 'float'>}
```
你可以注释一个变量而不给它赋值。这会将注释添加到__annotations__字典中，而变量仍然未定义：

```python
>>> nothing: str
>>> nothing
NameError: name 'nothing' is not defined

>>> __annotations__
{'nothing': <class 'str'>}
```

由于没有给nothing赋值，因此名称nothing尚未定义。

### 类型注解

正如前面提到的，注释是在Python 3中引入的，而且它们没有被回溯到Python 2。这意味着如果你编写需要支持旧版本Python的代码，你不能使用注释。

相反，你可以使用类型注解。这些是特殊格式的注释，可以用于添加与旧代码兼容的类型提示。要向函数添加类型注解，你可以像这样做：

```python
import math

def circumference(radius):
    # type: (float) -> float

    return 2 * math.pi * radius
```

类型注解只是注释，因此它们可以在任何版本的Python中使用。

类型注解由类型检查器直接处理，因此这些类型不可用于__annotations__字典：

```python
>>> circumference.__annotations__
{}
```

类型注解必须以type:字面值开头，并且在函数定义的同一行或下一行上。如果要为具有多个参数的函数注释，可以使用逗号分隔每个类型：

```python
def headline(text, width=80, fill_char="-"):
    # type: (str, int, str) -> str

    return f" {text.title()} ".center(width, fill_char)

print(headline("type comments work", width=40))
```

你也可以将每个参数写在单独的行上，并带有自己的注释：

```
# headlines.py

def headline(
    text,           # type: str
    width=80,       # type: int
    fill_char="-",  # type: str
):                  # type: (...) -> str

    return f" {text.title()} ".center(width, fill_char)

print(headline("type comments work", width=40))
```

通过Python和Mypy运行示例：

```shell
$  python headlines.py
---------- Type Comments Work ----------

$ mypy headlines.py
Success: no issues found in 1 source file
```

如果有错误，例如在第10行意外地使用width="full"调用headline()，Mypy会告诉你：

```shell
$ mypy headline.py
headline.py:10: error: Argument "width" to "headline" has incompatible
                       type "str"; expected "int"
```

你也可以为变量添加类型注解。这与向参数添加类型注解的方式类似：

```python
pi = 3.142  # type: float
```

在这个例子中，pi将被类型检查为浮点变量。

### 使用注释还是类型注解呢？

在向自己的代码添加类型提示时，应该使用注释还是类型注解？简而言之：如果可以的话，使用注释；如果必须的话，使用类型注解。

注释提供了更清晰的语法，将类型信息更接近你的代码。它们也是官方推荐的编写类型提示的方式，并且将来会进一步发展和正确维护。

类型注解更加冗长，可能会与代码中的其他类型的注释（如linter指令）冲突。但是，它们可以在不支持注释的代码库中使用。

还有一个隐藏的选项是：存根文件。稍后，当我们讨论向第三方库添加类型时，你将了解到这些。

存根文件将在任何Python版本中使用，但需要维护第二组文件。通常，只有在无法更改原始源代码时才会使用存根文件。


## 玩转Python类型，第一部分

到目前为止，你只在类型提示中使用了像str、float和bool这样的基本类型。Python类型系统非常强大，支持许多更复杂的类型。这是必要的，因为它需要能够合理地模拟Python的动态鸭子类型。

在本节中，你将学习更多关于这种类型系统的知识，同时实现一个简单的纸牌游戏。你将看到如何指定：

* 类似元组、列表和字典这样的序列和映射的类型
* 让代码更易读的类型别名
* 函数和方法不返回任何东西
* 可能是任何类型的对象

在短暂的类型理论之后，你将看到更多指定Python中类型的方法。你可以在这个部分的代码示例中找到这些例子。

### 示例：一副扑克牌

下面的例子展示了一个常规（法式）扑克牌的实现：

```python
# game.py

import random

SUITS = "♠ ♡ ♢ ♣".split()
RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()

def create_deck(shuffle=False):
    """Create a new deck of 52 cards"""
    deck = [(s, r) for r in RANKS for s in SUITS]
    if shuffle:
        random.shuffle(deck)
    return deck

def deal_hands(deck):
    """Deal the cards in the deck into four hands"""
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])

def play():
    """Play a 4-player card game"""
    deck = create_deck(shuffle=True)
    names = "P1 P2 P3 P4".split()
    hands = {n: h for n, h in zip(names, deal_hands(deck))}

    for name, cards in hands.items():
        card_str = " ".join(f"{s}{r}" for (s, r) in cards)
        print(f"{name}: {card_str}")

if __name__ == "__main__":
    play()
```

每张牌都表示为一个包含花色和点数的字符串元组。牌堆表示为一组牌的列表。create_deck()创建了一个包含52张纸牌的常规牌堆，并可选择洗牌。deal_hands()将牌堆中的牌分给四位玩家。

最后，play()进行游戏。到目前为止，它只是通过构建一个洗过牌的牌堆并把牌分给每个玩家来准备一个纸牌游戏。以下是一个典型的输出：

```shell
$ python game.py
P4: ♣9 ♢9 ♡2 ♢7 ♡7 ♣A ♠6 ♡K ♡5 ♢6 ♢3 ♣3 ♣Q
P1: ♡A ♠2 ♠10 ♢J ♣10 ♣4 ♠5 ♡Q ♢5 ♣6 ♠A ♣5 ♢4
P2: ♢2 ♠7 ♡8 ♢K ♠3 ♡3 ♣K ♠J ♢A ♣7 ♡6 ♡10 ♠K
P3: ♣2 ♣8 ♠8 ♣J ♢Q ♡9 ♡J ♠4 ♢8 ♢10 ♠9 ♡4 ♠Q
```

随着我们继续前进，你将看到如何将此示例扩展为一个更有趣的游戏。

### 序列和映射

让我们给我们的纸牌游戏添加类型提示。换句话说，让我们为create_deck()、deal_hands()和play()这些函数进行注释。首先的挑战是你需要注释像用于表示牌堆的列表和用于表示牌的元组这样的复合类型。

对于像str、float和bool这样的简单类型，添加类型提示就像使用类型本身一样简单：

```python
>>> name: str = "Guido"
>>> pi: float = 3.142
>>> centered: bool = False
```

对于复合类型，你也可以做同样的事情：

```python
>>> names: list = ["Guido", "Jukka", "Ivan"]
>>> version: tuple = (3, 7, 1)
>>> options: dict = {"centered": False, "capitalize": True}
```
然而，这并没有完全讲述清楚。names[2]、version[0]和options["centered"]的类型将会是什么？在这个具体的案例中，你可以看到它们分别是str、int和bool。然而，类型提示本身并没有提供关于这一点的信息。

相反，你应该使用typing模块中定义的特殊类型。这些类型为指定复合类型的元素类型添加了语法。你可以写出以下内容：

```python 
>>> from typing import Dict, List, Tuple
>>> names: List[str] = ["Guido", "Jukka", "Ivan"]
>>> version: Tuple[int, int, int] = (3, 7, 1)
>>> options: Dict[str, bool] = {"centered": False, "capitalize": True}
```

注意，每个类型都以大写字母开头，它们都使用方括号来定义项类型：

* names是一个字符串列表
* version是一个由三个整数组成的3元组
* options是一个将字符串映射到布尔值的字典

typing模块包含许多其他的复合类型，包括Counter、Deque、FrozenSet、NamedTuple和Set。此外，该模块还包含其他类型，你将在后面的部分中看到。

让我们回到纸牌游戏。一张牌由两个字符串组成的元组表示。你可以将其写为Tuple[str, str]，因此纸牌牌堆的类型变为List[Tuple[str, str]]。因此，你可以如下注释create_deck()：

```python
def create_deck(shuffle: bool = False) -> List[Tuple[str, str]]:
    """Create a new deck of 52 cards"""
    deck = [(s, r) for r in RANKS for s in SUITS]
    if shuffle:
        random.shuffle(deck)
    return deck
```

除了返回值，你还将bool类型添加到可选的shuffle参数中。

注意：元组和列表的注释方式不同。

元组是一个不可变序列，通常由可能不同类型的固定数量的元素组成。例如，我们将一张牌表示为花色和点数的元组。一般来说，你为n元组写Tuple[t_1, t_2, ..., t_n]。

列表是一个可变序列，通常由同一类型的未知数量的元素组成，例如一组牌。无论列表中有多少元素，注释中只有一个类型：List[t]。

在许多情况下，你的函数将期望某种序列，并且不真正关心它是列表还是元组。在这些情况下，你应该在注释函数参数时使用typing.Sequence：

```python
from typing import List, Sequence

def square(elems: Sequence[float]) -> List[float]:
    return [x**2 for x in elems]
```


使用Sequence是使用鸭子类型的一个例子。序列是任何支持len()和.getitem()的东西，与其实际类型无关。





### 类型别名

当处理像纸牌牌堆这样的嵌套类型时，类型提示可能会变得相当晦涩。在理解List[Tuple[str, str]]与我们的纸牌牌堆表示匹配之前，你可能需要仔细看一下。

现在考虑如何为deal_hands()添加注释：

```python
def deal_hands(
    deck: List[Tuple[str, str]]
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """Deal the cards in the deck into four hands"""
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])
```

那太可怕了！

回想一下，类型注释是常规的Python表达式。这意味着你可以通过将它们分配给新变量来定义自己的类型别名。例如，你可以创建Card和Deck类型别名：

```python
from typing import List, Tuple

Card = Tuple[str, str]
Deck = List[Card]
```

现在，Card可以在类型提示中或在新类型别名的定义中使用，例如上面的Deck。

使用这些别名，deal_hands()的注释变得更加可读：

```python
def deal_hands(deck: Deck) -> Tuple[Deck, Deck, Deck, Deck]:
    """Deal the cards in the deck into four hands"""
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])
```

类型别名非常适合使你的代码及其意图更清晰。与此同时，这些别名可以被检查以查看它们代表什么：

```python
>>> from typing import List, Tuple
>>> Card = Tuple[str, str]
>>> Deck = List[Card]

>>> Deck
typing.List[typing.Tuple[str, str]]
```

注意，当打印Deck时，它显示为一个列表，其中包含字符串的2元组的别名。

### 没有返回值的函数

你可能知道没有明确返回值的函数仍然会返回None：

```python
def play(player_name):
    print(f"{player_name} plays")

ret_val = play("Jacob")
```

虽然这样的函数技术上是有返回值的，但这个返回值没有用。你应该通过使用None作为返回类型来添加类型提示，表明这一点：

```python
def play(player_name: str) -> None:
    print(f"{player_name} plays")

ret_val = play("Filip")
```

注释有助于捕获那些试图使用无意义返回值的微妙错误。Mypy将给出有用的警告：

```shell
$ mypy play.py
play.py:6: error: "play" does not return a value
```

请注意，明确指出函数不返回任何内容与不在返回值的类型提示中添加类型提示是不同的：

```python
def play(player_name: str):
    print(f"{player_name} plays")

ret_val = play("Henrik")
```

在后一种情况下，Mypy没有关于返回值的信息，因此它不会生成任何警告：

```shell
$ mypy play.py
Success: no issues found in 1 source file
```

作为更奇特的情况，注意你也可以为永远不会正常返回的函数添加注释。这是使用NoReturn完成的：

```python
from typing import NoReturn

def black_hole() -> NoReturn:
    raise Exception("There is no going back ...")
```

由于black_hole()总是会引发异常，因此它永远不会正确返回。

### 示例：打一些牌

让我们回到我们的纸牌游戏示例。在游戏的第二个版本中，我们像以前一样向每个玩家发一手牌。然后选择一个开始玩家，玩家轮流出牌。但实际上游戏中没有什么规则，所以玩家将只是随机出牌：

```python
# game.py

import random
from typing import List, Tuple

SUITS = "♠ ♡ ♢ ♣".split()
RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()

Card = Tuple[str, str]
Deck = List[Card]

def create_deck(shuffle: bool = False) -> Deck:
    """Create a new deck of 52 cards"""
    deck = [(s, r) for r in RANKS for s in SUITS]
    if shuffle:
        random.shuffle(deck)
    return deck

def deal_hands(deck: Deck) -> Tuple[Deck, Deck, Deck, Deck]:
    """Deal the cards in the deck into four hands"""
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])

def choose(items):
    """Choose and return a random item"""
    return random.choice(items)

def player_order(names, start=None):
    """Rotate player order so that start goes first"""
    if start is None:
        start = choose(names)
    start_idx = names.index(start)
    return names[start_idx:] + names[:start_idx]

def play() -> None:
    """Play a 4-player card game"""
    deck = create_deck(shuffle=True)
    names = "P1 P2 P3 P4".split()
    hands = {n: h for n, h in zip(names, deal_hands(deck))}
    start_player = choose(names)
    turn_order = player_order(names, start=start_player)

    # Randomly play cards from each player's hand until empty
    while hands[start_player]:
        for name in turn_order:
            card = choose(hands[name])
            hands[name].remove(card)
            print(f"{name}: {card[0] + card[1]:<3}  ", end="")
        print()

if __name__ == "__main__":
    play()
```

请注意，除了更改play()之外，我们还添加了两个新函数，它们需要类型提示：choose()和player_order()。在讨论如何向它们添加类型提示之前，让我们看一下从运行游戏中获得的一个示例输出。

```shell
$ python game.py
P3: ♢10  P4: ♣4   P1: ♡8   P2: ♡Q
P3: ♣8   P4: ♠6   P1: ♠5   P2: ♡K
P3: ♢9   P4: ♡J   P1: ♣A   P2: ♡A
P3: ♠Q   P4: ♠3   P1: ♠7   P2: ♠A
P3: ♡4   P4: ♡6   P1: ♣2   P2: ♠K
P3: ♣K   P4: ♣7   P1: ♡7   P2: ♠2
P3: ♣10  P4: ♠4   P1: ♢5   P2: ♡3
P3: ♣Q   P4: ♢K   P1: ♣J   P2: ♡9
P3: ♢2   P4: ♢4   P1: ♠9   P2: ♠10
P3: ♢A   P4: ♡5   P1: ♠J   P2: ♢Q
P3: ♠8   P4: ♢7   P1: ♢3   P2: ♢J
P3: ♣3   P4: ♡10  P1: ♣9   P2: ♡2
P3: ♢6   P4: ♣6   P1: ♣5   P2: ♢8
```


在这个例子中，玩家P3被随机选择为起始玩家。依次，每个玩家打出一张牌：首先是P3，然后是P4，然后是P1，最后是P2。只要玩家手中还有牌，他们就会继续打牌。

### Any类型

choose()适用于名称列表和牌列表（以及任何其他序列）。为此添加类型提示的一种方法是：

```python
import random
from typing import Any, Sequence

def choose(items: Sequence[Any]) -> Any:
    return random.choice(items)
```

这基本上是说：items是一个可以包含任何类型项的序列，而choose()将返回任何类型的一个项目。不幸的是，这并不是很有用。考虑以下示例：

```python
import random
from typing import Any, Sequence

def choose(items: Sequence[Any]) -> Any:
    return random.choice(items)

names = ["Guido", "Jukka", "Ivan"]
reveal_type(names)

name = choose(names)
reveal_type(name)
```

虽然Mypy将正确地推断出names是字符串列表，但由于使用了Any类型，这种信息在调用choose()后丢失了：

```shell
$ mypy choose.py
choose.py:10: error: Revealed type is 'builtins.list[builtins.str*]'
choose.py:13: error: Revealed type is 'Any'
```


很快你将看到一个更好的方法。不过，在此之前，让我们更深入地了解一下Python类型系统以及Any扮演的特殊角色。



## 类型理论

这个教程主要是一个实用指南，我们只是浅尝Python类型提示背后的理论。如果你想回到实际示例，请随意跳到下一节。

### 子类型

一个重要的概念是子类型。形式上，我们说类型T是类型U的子类型，如果以下两个条件成立：

* T类型的每个值也在U类型的值集合中。
* U类型的每个函数也在T类型的函数集合中。

这两个条件保证了即使类型T与U不同，类型T的变量也始终可以伪装成U。

举个具体的例子，考虑T = bool 和 U = int。bool类型只有两个值。通常这些值被表示为True和False，但这些名称只是整数值1和0的别名，分别是：

```python
>>> int(False)
0

>>> int(True)
1

>>> True + True
2

>>> issubclass(bool, int)
True
```

由于0和1都是整数，第一个条件成立。你可以看到布尔值可以相加，但它们也可以做任何整数可以做的事情。这是上面第二个条件的意思。换句话说，bool是int的子类型。

子类型的重要性在于子类型始终可以伪装成其超类型。例如，下面的代码类型检查正确：

```python
def double(number: int) -> int:
    return number * 2

print(double(True))  # 传递bool而不是int
```


子类型与子类有些相关。实际上，所有的子类对应于子类型，而bool是int的子类型是因为bool是int的子类。然而，还有一些不对应于子类的子类型。例如int是float的子类型，但int不是float的子类。

### 协变、逆变和不变(Covariant, Contravariant, and Invariant)
在复合类型中使用子类型会发生什么？例如，Tuple[bool]是Tuple[int]的子类型吗？答案取决于复合类型，以及该类型是协变、逆变还是不变。这很快就会变得复杂，所以让我们举几个例子：

* Tuple是协变的。这意味着它保留其项目类型的类型层次结构：Tuple[bool]是Tuple[int]的子类型，因为bool是int的子类型。

* List是不变的。不变类型不对子类型做任何保证。虽然所有的List[bool]值都是List[int]值，但你可以将一个int附加到List[int]，而不能附加到List[bool]。换句话说，子类型的第二个条件不成立，List[bool]不是List[int]的子类型。【译注：我感觉这个说反了，应该是List[int]不能append一个bool值。】

* Callable在其参数中是逆变的。这意味着它颠倒了类型层次结构。稍后你将看到Callable的工作原理，但现在可以将Callable[[T], ...]视为其唯一参数类型为T的函数。Callable[[int], ...]的一个示例是上面定义的double()函数。逆变意味着如果预期一个操作bool的函数，那么操作int的函数也是可以接受的。

一般来说，你不需要记住这些表达式。但是，你应该意识到子类型和复合类型可能不是简单和直观的。

### 渐进类型和一致类型(Gradual Typing and Consistent Types)

前面我们提到Python支持渐进类型，你可以逐步向你的Python代码中添加类型提示。渐进类型实质上是由Any类型实现的。

不知何故，Any同时位于子类型的顶部和底部。Any类型的行为就像它是Any的子类型一样，而Any的行为就像它是任何其他类型的子类型一样。看看上面子类型的定义，这实际上是不可能的。相反，我们谈论一致类型。

如果T是U的子类型，或者T或U是Any，那么类型T与类型U是一致的。

类型检查器只会抱怨不一致的类型。因此，你永远不会看到来自Any类型的类型错误。

这意味着你可以使用Any来明确地回退到动态类型，描述在Python类型系统中描述太复杂的类型，或描述复合类型中的项。例如，一个具有字符串键并且可以接受任何类型值的字典可以被注释为Dict[str, Any]。

但是请记住，如果你使用Any，静态类型检查器实际上将不会进行任何类型检查。





## 玩转Python类型，第二部分

让我们回到我们的实际例子。回想一下，你试图给一般的choose()函数加上注解：

```python
import random
from typing import Any, Sequence

def choose(items: Sequence[Any]) -> Any:
    return random.choice(items)
```

使用Any的问题在于你无谓地丢失了类型信息。你知道如果将一个字符串列表传递给choose()，它将返回一个字符串。下面你将看到如何使用类型变量来表达这一点，以及如何处理：

* 鸭子类型和协议
* 默认值为None的参数
* 类方法
* 你自己类的类型
* 可变数量的参数
* 类型变量

### 类型变量

类型变量是一个特殊的变量，可以根据情况取任何类型。

让我们创建一个类型变量，它将有效地封装choose()的行为：

```python
# choose.py

import random
from typing import Sequence, TypeVar

Choosable = TypeVar("Choosable")

def choose(items: Sequence[Choosable]) -> Choosable:
    return random.choice(items)

names = ["Guido", "Jukka", "Ivan"]
reveal_type(names)

name = choose(names)
reveal_type(name)
```

类型变量必须使用typing模块中的TypeVar定义。当使用时，类型变量覆盖所有可能的类型，并采用最具体的类型。在这个例子中，name现在是一个str：

```bash
$ mypy choose.py
choose.py:12: error: Revealed type is 'builtins.list[builtins.str*]'
choose.py:15: error: Revealed type is 'builtins.str*'
```

考虑一些其他的例子：

```python
# choose_examples.py

from choose import choose

reveal_type(choose(["Guido", "Jukka", "Ivan"]))
reveal_type(choose([1, 2, 3]))
reveal_type(choose([True, 42, 3.14]))
reveal_type(choose(["Python", 3, 7]))
```

前两个例子应该有str和int类型，但是最后两个呢？各个列表项的类型不同，在这种情况下，Choosable类型变量尽最大努力适应：

```bash
$ mypy choose_examples.py
choose_examples.py:5: error: Revealed type is 'builtins.str*'
choose_examples.py:6: error: Revealed type is 'builtins.int*'
choose_examples.py:7: error: Revealed type is 'builtins.float*'
choose_examples.py:8: error: Revealed type is 'builtins.object*'
```


正如你已经看到的那样，bool是int的子类型，而int又是float的子类型。因此，在第三个例子中，choose()的返回值保证是可以看作float的东西。在最后一个例子中，str和int之间没有子类型关系，因此对返回值的最佳描述是对象。

请注意，这些例子中没有引发类型错误。有没有办法告诉类型检查器，choose()应该接受字符串和数字，但不能同时接受？

你可以通过列出可接受的类型来约束类型变量：

```python
# choose.py

import random
from typing import Sequence, TypeVar

Choosable = TypeVar("Choosable", str, float)

def choose(items: Sequence[Choosable]) -> Choosable:
    return random.choice(items)

reveal_type(choose(["Guido", "Jukka", "Ivan"]))
reveal_type(choose([1, 2, 3]))
reveal_type(choose([True, 42, 3.14]))
reveal_type(choose(["Python", 3, 7]))
```

现在Choosable只能是str或float，Mypy会指出最后一个例子是错误的：

```bash
$ mypy choose.py
choose.py:11: error: Revealed type is 'builtins.str*'
choose.py:12: error: Revealed type is 'builtins.float*'
choose.py:13: error: Revealed type is 'builtins.float*'
choose.py:14: error: Revealed type is 'builtins.object*'
choose.py:14: error: Value of type variable "Choosable" of "choose"
                     cannot be "object"
```

还要注意，在第二个例子中，尽管输入列表只包含int对象，但类型被认为是float。这是因为Choosable被限制为字符串和浮点数，而int是float的子类型。

在我们的纸牌游戏中，我们想要限制choose()只能用于str和Card：

```python
Choosable = TypeVar("Choosable", str, Card)

def choose(items: Sequence[Choosable]) -> Choosable:
    ...
```

我们简要地提到过Sequence代表列表和元组。正如我们注意到的，Sequence可以被视为一个鸭子类型，因为它可以是任何实现了.len()和.getitem()的对象。

### 鸭子类型和协议

回想一下介绍中的以下示例：

```python
def len(obj):
    return obj.__len__()
```

len()可以返回任何实现了.\_\_len\_\_()方法的对象的长度。我们如何为len()添加类型提示，特别是对于obj参数？

答案隐藏在学术上听起来很复杂的术语结构子类型化背后。一种分类类型系统的方式是根据它们是名义的(nominal)还是结构的(structural)：

* 在名义系统中，类型之间的比较基于名称和声明。Python类型系统在大多数情况下是名义的，其中int可以用在float的位置，因为它们之间存在子类型关系。

* 在结构系统中，类型之间的比较基于结构。你可以定义一个结构类型Sized，它包括所有定义了.len()的实例，而不考虑它们的名义类型。

通过PEP 544，正在进行工作以通过协议将一个完整的结构类型系统引入Python。PEP 544的大部分内容已经在Mypy中实现。

协议指定了一个或多个必须实现的方法。例如，所有定义了.len()的类都满足typing.Sized协议。因此，我们可以这样为len()添加注释：

```python
from typing import Sized

def len(obj: Sized) -> int:
    return obj.__len__()
```


typing模块中定义的其他协议示例包括Container、Iterable、Awaitable和ContextManager。

你也可以定义自己的协议。这可以通过继承Protocol并定义协议期望的函数签名（带有空函数体）来完成。以下示例显示了len()和Sized可能如何实现：

```python
from typing_extensions import Protocol

class Sized(Protocol):
    def __len__(self) -> int: ...

def len(obj: Sized) -> int:
    return obj.__len__()
```


在撰写本文时，对自定义协议的支持仍然是实验性的，而且只能通过typing_extensions模块使用。必须通过pip install typing-extensions从PyPI明确安装此模块。

### Optional类型

Python中常见的一种模式是使用None作为参数的默认值。这通常是为了避免可变默认值的问题或者为了有一个特殊行为的标志值。

在卡片示例中，player_order()函数使用None作为start的标志值，表示如果没有给定起始玩家，则应该随机选择一个：

```python
def player_order(names, start=None):
    """Rotate player order so that start goes first"""
    if start is None:
        start = choose(names)
    start_idx = names.index(start)
    return names[start_idx:] + names[:start_idx]
```

这给类型提示带来的挑战是，通常情况下start应该是一个字符串。然而，它也可以取特殊的非字符串值None。

为了注释这样的参数，你可以使用Optional类型：

```python
from typing import Sequence, Optional

def player_order(
    names: Sequence[str], start: Optional[str] = None
) -> Sequence[str]:
    ...
```

Optional类型简单地表示一个变量要么具有指定的类型，要么为None。指定相同内容的另一种方式是使用Union类型：Union[None, str]

请注意，当使用Optional或Union时，你必须确保在操作变量时它具有正确的类型。在本例中，通过测试start是否为None来做到这一点。如果不这样做，将导致静态类型错误以及可能的运行时错误：

```python
# player_order.py

from typing import Sequence, Optional

def player_order(
    names: Sequence[str], start: Optional[str] = None
) -> Sequence[str]:
    start_idx = names.index(start)
    return names[start_idx:] + names[:start_idx]
```

Mypy会告诉你，你没有处理start为None的情况：

```bash
$ mypy player_order.py
player_order.py:8: error: Argument 1 to "index" of "list" has incompatible
                          type "Optional[str]"; expected "str"
```

请注意，如果默认参数为None，Mypy会自动处理它。Mypy假设默认参数为None表示可选参数，即使类型提示没有明确说明。你可以这样做：

```python
def player_order(names: Sequence[str], start: str = None) -> Sequence[str]:
    ...
```

如果你不希望Mypy做出这种假设，可以使用\-\-no-implicit-optional命令行选项关闭它。


### 示例：面向对象的游戏

让我们将纸牌游戏重写为更符合面向对象的形式。这将使我们能够讨论如何正确地注释类和方法。

将我们的纸牌游戏直接转换为使用Card、Deck、Player和Game类的代码，大致如下所示：

```python
# game.py

import random
import sys

class Card:
    SUITS = "♠ ♡ ♢ ♣".split()
    RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.suit}{self.rank}"

class Deck:
    def __init__(self, cards):
        self.cards = cards

    @classmethod
    def create(cls, shuffle=False):
        """Create a new deck of 52 cards"""
        cards = [Card(s, r) for r in Card.RANKS for s in Card.SUITS]
        if shuffle:
            random.shuffle(cards)
        return cls(cards)

    def deal(self, num_hands):
        """Deal the cards in the deck into a number of hands"""
        cls = self.__class__
        return tuple(cls(self.cards[i::num_hands]) for i in range(num_hands))

class Player:
    def __init__(self, name, hand):
        self.name = name
        self.hand = hand

    def play_card(self):
        """Play a card from the player's hand"""
        card = random.choice(self.hand.cards)
        self.hand.cards.remove(card)
        print(f"{self.name}: {card!r:<3}  ", end="")
        return card

class Game:
    def __init__(self, *names):
        """Set up the deck and deal cards to 4 players"""
        deck = Deck.create(shuffle=True)
        self.names = (list(names) + "P1 P2 P3 P4".split())[:4]
        self.hands = {
            n: Player(n, h) for n, h in zip(self.names, deck.deal(4))
        }

    def play(self):
        """Play a card game"""
        start_player = random.choice(self.names)
        turn_order = self.player_order(start=start_player)

        # Play cards from each player's hand until empty
        while self.hands[start_player].hand.cards:
            for name in turn_order:
                self.hands[name].play_card()
            print()

    def player_order(self, start=None):
        """Rotate player order so that start goes first"""
        if start is None:
            start = random.choice(self.names)
        start_idx = self.names.index(start)
        return self.names[start_idx:] + self.names[:start_idx]

if __name__ == "__main__":
    # Read player names from command line
    player_names = sys.argv[1:]
    game = Game(*player_names)
    game.play()
```

现在让我们为这段代码添加类型。

### 方法的类型提示

首先，方法的类型提示与函数的类型提示基本相同。唯一的区别是self参数无需注释，因为它总是一个类实例。添加Card类的类型很容易：

```python
class Card:
    SUITS = "♠ ♡ ♢ ♣".split()
    RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()

    def __init__(self, suit: str, rank: str) -> None:
        self.suit = suit
        self.rank = rank

    def __repr__(self) -> str:
        return f"{self.suit}{self.rank}"
```

请注意，.\_\_init\_\_()方法应该始终将None作为其返回类型。

### 类作为类型

类和类型之间存在对应关系。例如，Card类的所有实例一起形成Card类型。要将类用作类型，只需使用类的名称。

例如，Deck本质上由一组Card对象组成。你可以这样注释：

```python
class Deck:
    def __init__(self, cards: List[Card]) -> None:
        self.cards = cards
```

Mypy能够将你在注释中使用的Card与Card类的定义联系起来。

但是当你需要引用当前正在定义的类时，情况就不那么清晰了。例如，Deck.create()类方法返回一个类型为Deck的对象。然而，你不能简单地添加-> Deck，因为Deck类尚未完全定义。

相反，你可以在注释中使用字符串文字。这些字符串文字将仅在类型检查器稍后进行评估，因此可以包含self和前向引用。.create()方法应该使用这种字符串文字来表示其类型：

```python
class Deck:
    @classmethod
    def create(cls, shuffle: bool = False) -> "Deck":
        """Create a new deck of 52 cards"""
        cards = [Card(s, r) for r in Card.RANKS for s in Card.SUITS]
        if shuffle:
            random.shuffle(cards)
        return cls(cards)
```

请注意，Player类也会引用Deck类。然而，这并不是问题，因为Deck在Player之前定义：

```python
class Player:
    def __init__(self, name: str, hand: Deck) -> None:
        self.name = name
        self.hand = hand
```

通常情况下，类型提示不会在运行时使用。这促使人们想到推迟注释的评估。建议不要将注释评估为Python表达式并存储其值，而是将注释的字符串表示形式存储下来，仅在需要时才对其进行评估。

这样的功能计划在尚未出现的Python 4.0中成为标准。然而，在Python 3.7及更高版本中，前向引用可通过\_\_future\_\_导入获得：

```python
from __future__ import annotations

class Deck:
    @classmethod
    def create(cls, shuffle: bool = False) -> Deck:
        ...
```

通过\_\_future\_\_导入，即使在Deck定义之前，也可以使用Deck而不是"Deck"。

### 返回self或cls

如前所述，你通常不应该对self或cls参数进行注释。部分原因是不必要的，因为self指向类的一个实例，所以它将具有类的类型。在Card示例中，self具有隐式类型Card。此外，显式添加这种类型会很麻烦，因为类尚未定义。你必须使用字符串文字语法，self："Card"。

然而，在某些情况下，你可能希望注释self或cls。考虑一下，如果你有一个其他类继承的超类，并且该超类具有返回self或cls的方法：

```python
# dogs.py

from datetime import date

class Animal:
    def __init__(self, name: str, birthday: date) -> None:
        self.name = name
        self.birthday = birthday

    @classmethod
    def newborn(cls, name: str) -> "Animal":
        return cls(name, date.today())

    def twin(self, name: str) -> "Animal":
        cls = self.__class__
        return cls(name, self.birthday)

class Dog(Animal):
    def bark(self) -> None:
        print(f"{self.name} says woof!")

fido = Dog.newborn("Fido")
pluto = fido.twin("Pluto")
fido.bark()
pluto.bark()
```

虽然代码可以正常运行，但Mypy会标记一个问题：

```python
$ mypy dogs.py
dogs.py:24: error: "Animal" has no attribute "bark"
dogs.py:25: error: "Animal" has no attribute "bark"
```

问题在于，即使继承的Dog.newborn()和Dog.twin()方法将返回一个Dog，注释却表示它们返回一个Animal。

在这种情况下，你需要更谨慎地确保注释是正确的。返回类型应与self的类型或cls的实例类型匹配。这可以通过使用类型变量来实现，这些类型变量跟踪实际传递给self和cls的内容：

```python
# dogs.py

from datetime import date
from typing import Type, TypeVar

TAnimal = TypeVar("TAnimal", bound="Animal")

class Animal:
    def __init__(self, name: str, birthday: date) -> None:
        self.name = name
        self.birthday = birthday

    @classmethod
    def newborn(cls: Type[TAnimal], name: str) -> TAnimal:
        return cls(name, date.today())

    def twin(self: TAnimal, name: str) -> TAnimal:
        cls = self.__class__
        return cls(name, self.birthday)

class Dog(Animal):
    def bark(self) -> None:
        print(f"{self.name} says woof!")

fido = Dog.newborn("Fido")
pluto = fido.twin("Pluto")
fido.bark()
pluto.bark()
```

在这个示例中，有几点需要注意：

* 类型变量TAnimal用于表示返回值可能是Animal的子类的实例。
* 我们指定Animal是TAnimal的上界。指定上界意味着TAnimal只能是Animal或其子类之一。这是为了正确限制允许的类型。
* typing.Type[]构造是typing模块中type()的等价物。你需要它来指出类方法期望一个类，并返回该类的实例。

### 注释 \*args 和 \*\*kwargs

在游戏的面向对象版本中，我们添加了在命令行上为玩家命名的选项。这是通过在程序名称后列出玩家名称来实现的：

```python
$ python game.py GeirArne Dan Joanna
Dan: ♢A   Joanna: ♡9   P1: ♣A   GeirArne: ♣2
Dan: ♡A   Joanna: ♡6   P1: ♠4   GeirArne: ♢8
Dan: ♢K   Joanna: ♢Q   P1: ♣K   GeirArne: ♠5
Dan: ♡2   Joanna: ♡J   P1: ♠7   GeirArne: ♡K
Dan: ♢10  Joanna: ♣3   P1: ♢4   GeirArne: ♠8
Dan: ♣6   Joanna: ♡Q   P1: ♣Q   GeirArne: ♢J
Dan: ♢2   Joanna: ♡4   P1: ♣8   GeirArne: ♡7
Dan: ♡10  Joanna: ♢3   P1: ♡3   GeirArne: ♠2
Dan: ♠K   Joanna: ♣5   P1: ♣7   GeirArne: ♠J
Dan: ♠6   Joanna: ♢9   P1: ♣J   GeirArne: ♣10
Dan: ♠3   Joanna: ♡5   P1: ♣9   GeirArne: ♠Q
Dan: ♠A   Joanna: ♠9   P1: ♠10  GeirArne: ♡8
Dan: ♢6   Joanna: ♢5   P1: ♢7   GeirArne: ♣4
```

这是通过将 sys.argv 解包并传递给 Game() 来实现的，当它被实例化时。 .init() 方法使用 *names 将给定的名称打包到一个元组中。

关于类型注释：即使 names 将是一个字符串元组，您也应该只注释每个名称的类型。换句话说，您应该使用 str 而不是 Tuple[str]：

```python
class Game:
    def __init__(self, *names: str) -> None:
        """Set up the deck and deal cards to 4 players"""
        deck = Deck.create(shuffle=True)
        self.names = (list(names) + "P1 P2 P3 P4".split())[:4]
        self.hands = {
            n: Player(n, h) for n, h in zip(self.names, deck.deal(4))
        }
```


同样地，如果您有一个接受 **kwargs 的函数或方法，那么您应该只注释每个可能的关键字参数的类型。

### 可调用对象(Callable)

在 Python 中，函数是一等对象。这意味着您可以将函数作为参数传递给其他函数。这也意味着您需要能够添加表示函数的类型提示。

函数，以及 lambda 函数、方法和类，都由 typing.Callable 表示。参数的类型和返回值类型通常也被表示出来。例如，Callable[[A1, A2, A3], Rt] 表示具有三个参数，类型分别为 A1、A2 和 A3 的函数。函数的返回类型是 Rt。

在以下示例中，函数 do_twice() 调用给定的函数两次并打印返回值：

```python
# do_twice.py

from typing import Callable

def do_twice(func: Callable[[str], str], argument: str) -> None:
    print(func(argument))
    print(func(argument))

def create_greeting(name: str) -> str:
    return f"Hello {name}"

do_twice(create_greeting, "Jekyll")
```


请注意第 5 行对 do_twice() 的 func 参数的注释。它表示 func 应该是一个带有一个字符串参数的可调用对象，也返回一个字符串。这样的可调用对象的一个例子是在第 9 行定义的 create_greeting()。

大多数可调用对象类型都可以以类似的方式进行注释。但是，如果您需要更灵活的选项，请查看回调协议和扩展的可调用对象类型。


### 示例：红心游戏
让我们以一个完整的红心游戏示例结束。您可能已经从其他计算机模拟游戏中了解到这个游戏。以下是规则的快速回顾：

* 四名玩家每人手持 13 张牌。
* 持有 ♣2 的玩家开始第一轮，并且必须打出 ♣2。
* 玩家跟随打牌，如果有相同花色的话，花色一致。
* 打出这个花色中最大的牌的玩家赢得这一轮，并成为下一轮的起始玩家。同时把这一轮所有的牌收归己有，并且扣起来不让人看到。
* 如果之前没有出过♡，这一轮的起始玩家不能以 ♡ 开始，除非之前的轮次中有玩家跟随过♡。
* 在所有牌都打出后，玩家如果获得某些牌，则会获得得分：
	** ♠Q 得 13 分
	** 每张 ♡ 得 1 分
        ** 如果某个玩家拿到所有13张的♡和♠Q，那么这个玩家得0分，其他3个玩家的26分

* 游戏持续数轮，直到一名玩家获得 100 分或更多。分数最少的玩家获胜。

更多详细信息可以在线找到。

在这个示例中，没有太多新的类型概念，您已经见过了。因此，我们不会详细介绍此代码，而是将其作为带有注释的代码示例。完整的代码请参考[github](https://github.com/realpython/materials/blob/master/python-type-checking/hearts.py)。以下是代码中需要注意的几点：

* 对于使用 Union 或类型变量难以表达的类型关系，可以使用 @overload 装饰器。有关示例，请参阅 Deck.getitem()，并查看文档以获取更多信息。

* 子类对应于子类型，因此可以在期望 Player 的任何地方使用 HumanPlayer。

* 当子类重新实现超类的方法时，类型注释必须匹配。有关示例，请参阅 HumanPlayer.play_card()。

在开始游戏时，您控制第一个玩家。输入数字以选择要打出的牌。以下是游戏进行的示例，高亮显示的行显示了玩家做出选择的位置：


```python
$ python hearts.py GeirArne Aldren Joanna Brad

Starting new round:
Brad -> ♣2
  0: ♣5  1: ♣Q  2: ♣K  (Rest: ♢6 ♡10 ♡6 ♠J ♡3 ♡9 ♢10 ♠7 ♠K ♠4)
  GeirArne, choose card: 2
GeirArne => ♣K
Aldren -> ♣10
Joanna -> ♣9
GeirArne wins the trick

  0: ♠4  1: ♣5  2: ♢6  3: ♠7  4: ♢10  5: ♠J  6: ♣Q  7: ♠K  (Rest: ♡10 ♡6 ♡3 ♡9)
  GeirArne, choose card: 0
GeirArne => ♠4
Aldren -> ♠5
Joanna -> ♠3
Brad -> ♠2
Aldren wins the trick

...

Joanna -> ♡J
Brad -> ♡2
  0: ♡6  1: ♡9  (Rest: )
  GeirArne, choose card: 1
GeirArne => ♡9
Aldren -> ♡A
Aldren wins the trick

Aldren -> ♣A
Joanna -> ♡Q
Brad -> ♣J
  0: ♡6  (Rest: )
  GeirArne, choose card: 0
GeirArne => ♡6
Aldren wins the trick

Scores:
Brad             14  14
Aldren           10  10
GeirArne          1   1
Joanna            1   1
```


## 静态类型检查

到目前为止，您已经了解了如何向您的代码添加类型提示。在本节中，您将学习更多关于如何实际执行 Python 代码的静态类型检查。

### Mypy 项目

Mypy 是由 Jukka Lehtosalo 在他在剑桥大学攻读博士学位期间于 2012 年左右开始的。最初，Mypy 被构想为一种具有无缝动态和静态类型的 Python 变种。请参阅 Jukka 在 2012 年 PyCon Finland 的幻灯片，了解 Mypy 最初的愿景的示例。

大部分最初的想法仍然在 Mypy 项目中扮演着重要角色。事实上，“无缝动态和静态类型”这一口号仍然明显地显示在 Mypy 的主页上，并且很好地描述了在 Python 中使用类型提示的动机。

自 2012 年以来的最大变化是，Mypy 不再是 Python 的变种。在其最初的版本中，Mypy 是一个独立的语言，与 Python 兼容，除了其类型声明。在 Guido van Rossum 的建议下，Mypy 被重写为使用注释。如今，Mypy 是一个用于常规 Python 代码的静态类型检查器。

### 运行 Mypy

在第一次运行 Mypy 之前，您必须安装该程序。这最容易通过 pip 安装：

```bash
pip install mypy
```

安装了 Mypy 之后，您可以将其作为常规命令行程序运行：

```bash
$ mypy my_program.py
```

在您的 my_program.py Python 文件上运行 Mypy 将检查其中的类型错误，而不会实际执行代码。

在对代码进行类型检查时，有许多可用的选项。由于 Mypy 仍然在非常活跃的开发中，命令行选项可能会在版本之间发生变化。您应该参考 Mypy 的帮助以查看您的版本上的默认设置：

```bash
$ mypy --help
usage: mypy [-h] [-v] [-V] [more options; see below]
            [-m MODULE] [-p PACKAGE] [-c PROGRAM_TEXT] [files ...]

Mypy is a program that will type check your Python code.

[... The rest of the help hidden for brevity ...]
```
 
另外，Mypy 的在线命令行文档提供了大量信息。

让我们看一些最常见的选项。首先，如果您使用没有类型提示的第三方包，您可能希望消除 Mypy 对这些包的警告。这可以通过 --ignore-missing-imports 选项来实现。

以下示例使用 Numpy 来计算并打印几个数字的余弦值：

```python
# cosine.py

import numpy as np

def print_cosine(x: np.ndarray) -> None:
    with np.printoptions(precision=3, suppress=True):
        print(np.cos(x))

x = np.linspace(0, 2 * np.pi, 9)
print_cosine(x)
```

请注意，np.printoptions() 仅在 Numpy 的 1.15 及更高版本中可用。运行此示例会将一些数字打印到控制台：

```bash
$ python cosine.py
[ 1.     0.707  0.    -0.707 -1.    -0.707 -0.     0.707  1.   ]
```

此示例的实际输出并不重要。但是，您应该注意到参数 x 在第 5 行被注释为 np.ndarray，因为我们要打印一个完整数组的余弦值。

您可以像往常一样在此文件上运行 Mypy：

```bash
$ mypy cosine.py 
cosine.py:3: error: No library stub file for module 'numpy'
cosine.py:3: note: (Stub files are from https://github.com/python/typeshed)
```

这些警告可能一开始对您来说并不太有意义，但是您很快就会了解 stubs 和 typeshed。您基本上可以将这些警告视为 Mypy 表示 Numpy 包不包含类型提示。

在大多数情况下，第三方包中缺少类型提示并不是您想要关心的问题，因此您可以消除这些消息：

```bash
$ mypy --ignore-missing-imports cosine.py 
Success: no issues found in 1 source file
```

如果您使用 \-\-ignore-missing-import 命令行选项，则 Mypy 不会尝试跟踪或警告任何丢失的导入。不过，这可能有点过度，因为它还会忽略实际的错误，比如拼写包名称时的错误。

处理第三方包的两种不太侵入性的方法是使用类型注释或配置文件。

在简单的示例中，您可以通过在包含 import 的行上添加类型注释来消除 numpy 警告：

```python
import numpy as np  # type: ignore
```

直接使用 # type: ignore 字面量告诉 Mypy 忽略 Numpy 的导入。

如果您有多个文件，将导入忽略的内容放入配置文件可能更容易跟踪。如果存在名为 mypy.ini 的文件，则 Mypy 会在当前目录中读取该文件。该配置文件必须包含一个名为 [mypy] 的部分，并且可能包含类似 [mypy-module] 的模块特定部分。

以下配置文件将忽略 Numpy 缺少类型提示：

```
# mypy.ini

[mypy]

[mypy-numpy]
ignore_missing_imports = True
```


在配置文件中可以指定许多选项。还可以指定全局配置文件。有关更多信息，请参阅文档。

### 添加 Stubs

Python 标准库中的所有包都可以使用类型提示。但是，如果您使用的是第三方包，则已经看到情况可能会有所不同。

以下示例使用 Parse 包来进行简单的文本解析。要跟进，请先安装 Parse：

```bash
$ pip install parse
```

Parse 可以用于识别简单的模式。以下是一个尝试根据列在行 7-11 中列出的模式之一找出您的姓名的小程序：

```python
# parse_name.py

import parse

def parse_name(text: str) -> str:
    patterns = (
        "my name is {name}",
        "i'm {name}",
        "i am {name}",
        "call me {name}",
        "{name}",
    )
    for pattern in patterns:
        result = parse.parse(pattern, text)
        if result:
            return result["name"]
    return ""

answer = input("What is your name? ")
name = parse_name(answer)
print(f"Hi {name}, nice to meet you!")
```

主要流程定义在最后三行：询问您的姓名、解析答案并打印问候语。在第 14 行调用 parse 包，以尝试根据列在第 7-11 行的模式之一找出姓名。

该程序可如下使用：

```bash
$ python parse_name.py
What is your name? I am Geir Arne
Hi Geir Arne, nice to meet you!
```

请注意，即使我回答 I am Geir Arne，该程序也能够确定 I am 不是我的名字的一部分。

让我们向程序添加一个小错误，然后看看 Mypy 是否能够帮助我们检测到它。将第 16 行从 return result["name"] 更改为 return result。这将返回一个 parse.Result 对象，而不是包含名称的字符串。

接下来在程序上运行 Mypy：

```bash
$ mypy parse_name.py 
parse_name.py:3: error: Cannot find module named 'parse'
parse_name.py:3: note: (Perhaps setting MYPYPATH or using the
                       "--ignore-missing-imports" flag would help)
```

Mypy 打印了与您在上一节中看到的类似的错误：它不知道 parse 包。您可以尝试忽略导入：

```bash
$ mypy parse_name.py --ignore-missing-imports
Success: no issues found in 1 source file
```

不幸的是，忽略导入意味着 Mypy 无法发现我们程序中的错误。更好的解决方案是为 Parse 包本身添加类型提示。由于 Parse 是开源的，您实际上可以在源代码中添加类型并发送拉取请求。

或者，您可以在 stub 文件中添加类型。存根文件是一个文本文件，其中包含方法和函数的签名，但不包含它们的实现。它们的主要功能是为您无法更改的代码添加类型提示。为了展示这是如何工作的，我们将为 Parse 包添加一些存根。

首先，您应该将所有存根文件放在一个公共目录中，并将 MYPYPATH 环境变量设置为指向该目录。在 Mac 和 Linux 上，您可以设置 MYPYPATH 如下：

```bash
$ export MYPYPATH=/home/gahjelle/python/stubs
```

您可以通过将该行添加到您的 .bashrc 文件来永久设置该变量。在 Windows 上，您可以点击开始菜单并搜索环境变量以设置 MYPYPATH。

接下来，在存根目录中创建一个名为 parse.pyi 的文件。它必须以您要添加类型提示的包的名称命名，并带有 .pyi 后缀。目前让这个文件保持空白。然后再次运行 Mypy：

```bash
$ mypy parse_name.py
parse_name.py:14: error: Module has no attribute "parse"
```

如果您已经正确设置了所有内容，那么您应该会看到这个新的错误消息。Mypy 使用新的 parse.pyi 文件来确定 parse 包中可用的函数。由于存根文件为空，Mypy 假设 parse.parse() 不存在，然后给出您在上面看到的错误。

以下示例并未为整个 parse 包添加类型。相反，它显示了您需要添加的类型提示，以便 Mypy 可以检查您对 parse.parse() 的使用：

```python
# parse.pyi

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

class Result:
    def __init__(
        self,
        fixed: Sequence[str],
        named: Mapping[str, str],
        spans: Mapping[int, Tuple[int, int]],
    ) -> None: ...
    def __getitem__(self, item: Union[int, str]) -> str: ...
    def __repr__(self) -> str: ...

def parse(
    format: str,
    string: str,
    evaluate_result: bool = ...,
    case_sensitive: bool = ...,
) -> Optional[Result]: ...
```

省略号 ... 是文件的一部分，应按照上面的确切方式进行编写。存根文件应仅包含变量、属性、函数和方法的类型提示，因此实现应该被省略并替换为省略号标记。

最后，Mypy 能够发现我们引入的错误：

```bash
$ mypy parse_name.py
parse_name.py:16: error: Incompatible return value type (got
                         "Result", expected "str")
```

这直接指向第 16 行，指出我们返回的是一个 Result 对象，而不是名称字符串。将 return result 更改回 return result["name"]，然后再次运行 Mypy，看到它是快乐的。

### Typeshed

您已经了解了如何使用存根文件为源代码本身添加类型提示。在前一节中，我们为第三方 Parse 包添加了一些类型提示。现在，如果每个人都需要为他们正在使用的所有第三方包创建自己的存根文件，那么这将不是很有效。

Typeshed 是一个包含 Python 标准库以及许多第三方包类型提示的 Github 存储库。Typeshed 已包含在 Mypy 中，因此如果您正在使用已在 Typeshed 中定义了类型提示的包，则类型检查将正常工作。

您还可以为 Typeshed 贡献类型提示。请确保首先获得软件包所有者的许可，特别是因为他们可能正在努力将类型提示添加到源代码本身——这是首选方法。

### 其他静态类型检查器

在本教程中，我们主要关注了使用 Mypy 进行类型检查。但是，在 Python 生态系统中还有其他静态类型检查器。

PyCharm 编辑器带有自己的类型检查器。如果您使用 PyCharm 编写 Python 代码，则会自动进行类型检查。

Facebook 开发了 Pyre。其声明之一是快速和高性能。虽然存在一些差异，但 Pyre 大部分功能与 Mypy 类似。如果您有兴趣尝试 Pyre，请参阅文档。

此外，Google 创建了 Pytype。这个类型检查器的工作方式也与 Mypy 大致相同。除了检查带有注释的代码外，Pytype 还支持在未注释的代码上运行类型检查，甚至自动为代码添加注释。有关更多信息，请参阅快速入门文档。

### 在运行时使用类型

最后一点需要注意的是，您可以在执行 Python 程序时在运行时使用类型提示。运行时类型检查可能永远不会在 Python 中原生支持。

但是，类型提示在运行时在 \_\_annotations\_\_ 字典中是可用的，您可以使用它们进行类型检查。在开始编写自己的用于强制类型的包之前，您应该知道已经有几个包为您完成了这个工作。查看 Enforce、Pydantic 或 Pytypes 以获取一些示例。

类型提示的另一个用途是将您的 Python 代码转换为 C 并编译它以进行优化。流行的 Cython 项目使用混合 C/Python 语言编写静态类型的 Python 代码。但是，自版本 0.27 以来，Cython 也支持类型注释。最近，Mypyc 项目已经推出。虽然还不适合一般用途，但它可以将一些带有类型注释的 Python 代码编译成 C 扩展。

