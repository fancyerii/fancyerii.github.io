---
layout:     post
title:      "第五章：变量"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - cmake 
---



 <!--more-->

前面的章节展示了如何定义基本目标并生成构建输出。这已经很有用了，但CMake还配备了许多其他功能，带来了巨大的灵活性和便利性。本章涵盖了CMake最基本的部分之一，即变量的使用。

## 5.1 变量基本概念

 
和任何计算语言一样，变量是在CMake中完成任务的基石。定义变量的最基本方式是使用set()命令。可以在CMakeLists.txt文件中定义普通变量，如下所示：

```
set(varName value... [PARENT_SCOPE])
```

变量的名称varName可以包含字母、数字和下划线，字母区分大小写。名称还可以包含字符./-+，但在实践中很少见到这些。其他字符也可能通过间接方式出现，但同样，在正常使用中很少见到这些。

在CMake中，变量具有特定的作用域，类似于其他语言中变量的作用域限制在特定的函数、文件等。变量不能在其自身作用域之外读取或修改。与其他语言相比，在CMake中，变量的作用域稍微更灵活，但暂时将变量的作用域视为定义它的文件。第5.4节“作用域块”讨论了如何定义本地作用域并将信息传递回封闭作用域。第7章“使用子目录”和第8章“函数和宏”介绍了本地作用域出现的更多情况。

CMake将所有变量视为字符串。在各种上下文中，变量可能被解释为不同的类型，但最终它们只是字符串。在设置变量的值时，CMake不要求这些值带引号，除非值包含空格。如果给出多个值，这些值将用分号连接在一起。这样得到的字符串是CMake表示列表的方式。以下示例应有助于演示这种行为。

```
set(myVar a b c)
# myVar = "a;b;c"
set(myVar a;b;c)
# myVar = "a;b;c"
set(myVar "a b c")
# myVar = "a b c"
set(myVar a b;c)
# myVar = "a;b;c"
set(myVar a "b c")
# myVar = "a;b c"
```

使用${myVar}形式获取变量的值，可以在期望字符串或变量的任何地方使用。CMake特别灵活，也可以递归使用这种形式或指定设置另一个变量的名称。此外，CMake不要求在使用变量之前定义它们。对未定义变量的使用只会导致将空字符串替换，类似于Unix shell脚本的行为。默认情况下，对未定义变量的使用不会发出警告，但可以通过给cmake命令添加--warn-uninitialized选项来启用此类警告。但是请注意，这种用法非常普遍，不一定是问题的症状，因此该选项的实用性可能有限。

```
set(foo ab)
# foo = "ab"
set(bar ${foo}cd)
# bar = "abcd"
set(baz ${foo} cd)
# baz = "ab;cd"
set(myVar ba)
# myVar = "ba"
set(big "${${myVar}r}ef")
# big = "${bar}ef" = "abcdef"
set(${foo} xyz)
# ab = "xyz"
set(bar ${notSetVar})
# bar = ""
```

字符串不限制为单行，它们可以包含嵌入的换行符。它们还可以包含引号，需要用反斜杠转义。

```
set(myVar "goes here")
set(multiLine "First line ${myVar}
Second line with a \"quoted\" word")
```

如果使用CMake 3.0或更高版本，替代引号的方法是使用受Lua启发的括号语法，其中内容的开始由 [=[ 标记，结束由 ]=] 标记。方括号之间可以包含任意数量的=字符，包括零个，但在开始和结束处必须使用相同数量的=字符。如果开头的括号紧跟着换行符，那么第一个换行符将被忽略，但后续的换行符不会。此外，不会对括号内的内容进行进一步的转换（即，不会进行变量替换或转义）。

```
# 使用括号语法的简单多行内容，方括号标记之间不需要=
set(multiLine [[
First line
Second line
]])
# 括号语法防止不必要的替换
set(shellScript [=[
#!/bin/bash
[[ -n "${USER}" ]] && echo "Have USER"
]=])
# 没有括号语法的等效代码
set(shellScript
"#!/bin/bash
[[ -n \"\${USER}\" ]] && echo \"Have USER\"
")
```

正如上面的例子所示，括号语法特别适用于定义类似Unix shell脚本的内容。这样的内容使用${...}语法用于自己的目的，并经常包含引号，但使用括号语法意味着这些内容不必被转义，不同于传统的定义CMake内容的引号样式。在[和]标记之间使用任意数量的=字符的灵活性也意味着嵌套的方括号不会被误解为标记。第20章“使用文件”中还包括进一步的示例，突出显示了括号语法可以是更好选择的情况。

可以通过调用unset()或调用set()为命名变量不提供值来取消变量。以下两种方式是等效的，如果myVar不存在，则不会出现错误或警告：

```
set(myVar)
unset(myVar)
```

除了由项目为自身使用定义的变量外，许多CMake命令的行为也可以由在调用命令时特定变量的值来影响。这是CMake用于定制命令行为或修改默认值的常见模式，以便不必为每个命令、目标定义等都重复。每个命令的CMake参考文档通常列出了可以影响该命令行为的任何变量。本书的后续章节还强调了一些有用的变量以及它们如何影响或提供有关构建的信息。

## 5.2 环境变量

CMake还允许使用修改后的CMake变量表示法检索和设置环境变量的值。使用特殊形式$ENV{varName}可以获取环境变量的值，并且它可以在任何可以使用常规${varName}形式的地方使用。设置环境变量可以通过类似于CMake变量的方式完成，只不过使用ENV{varName}而不是仅仅varName作为要设置的变量。例如：

```
set(ENV{PATH} "$ENV{PATH}:/opt/myDir")
```

但是请注意，像这样设置环境变量只影响当前运行的CMake实例。一旦CMake运行结束，对环境变量的更改就会丢失。特别是，在构建时看不到对环境变量的更改。因此，在CMakeLists.txt文件中像这样设置环境变量很少有用。


## 5.3 缓存变量


除了上面讨论的普通变量之外，CMake还支持缓存变量。与普通变量的生命周期限定在处理CMakeLists.txt文件时不同，缓存变量存储在名为CMakeCache.txt的特殊文件中，位于构建目录中，并在CMake运行之间持久存在。一旦设置，缓存变量会一直保持设置，直到有事物明确将它们从缓存中删除。缓存变量的值以与普通变量相同的方式检索（即使用${myVar}形式），但是当用于设置缓存变量时，set()命令是不同的：

```
set(varName value... CACHE type "docstring" [FORCE])
```

当存在CACHE关键字时，set()命令将应用于名为varName的缓存变量，而不是普通变量。缓存变量比普通变量附有更多的信息，包括名义类型和文档字符串。在设置缓存变量时，必须提供两者，尽管文档字符串可以为空。文档字符串不影响CMake对变量的处理方式，它仅由GUI工具使用，以提供诸如帮助详细信息、工具提示等的内容。

在处理期间，CMake始终将变量视为字符串。类型主要用于在GUI工具中改善用户体验，其中有一些重要的例外情况，这些例外情况在第5.5节“变量的潜在令人惊讶的行为”中讨论。类型必须是以下之一：

**BOOL** 

缓存变量是一个布尔值。GUI工具使用复选框或类似物来表示变量。变量的基础字符串值将符合CMake将布尔表示为字符串的方式之一（ON/OFF、TRUE/FALSE、1/0等——有关详细信息，请参见第6.1.1节“基本表达式”）。

**FILEPATH**

缓存变量表示磁盘上文件的路径。GUI工具向用户显示文件对话框，以修改变量的值。

**PATH**

与FILEPATH相同，但GUI工具显示的对话框选择目录而不是文件。

**STRING**

变量被视为任意字符串。默认情况下，GUI工具使用单行文本编辑窗口来操作变量的值。项目可以使用缓存变量属性为GUI工具提供预定义的值集，以将其显示为下拉框或类似工具（请参阅第9.6节“缓存变量属性”）。

**INTERNAL**

不打算向用户公开的变量。内部缓存变量有时用于由项目持久记录内部信息，例如缓存对查询或计算结果的结果。GUI工具不显示INTERNAL变量。INTERNAL还意味着FORCE（后文进一步讨论）。
GUI工具通常使用文档字符串作为缓存变量的工具提示，或在选择变量时作为短的一行描述。文档字符串应该简短并由纯文本组成（即无HTML标记等）。

设置布尔缓存变量是一个常见的需求，CMake提供了一个单独的命令来实现。开发人员可以使用option()命令而不是稍显冗长的set()命令。如果省略了initialValue，将使用默认值OFF。如果提供了initialValue，则它必须符合set()命令接受的布尔值之一。可以将上述option()命令视为更多或更少等效于：

```
set(optVar initialValue CACHE BOOL helpString)
```

与set()相比，option()命令更清晰地表达了布尔缓存变量的行为，因此通常会更倾向于使用该命令。然而，请注意，在某些情况下，这两个命令的效果可能不同（请参阅第5.5节“变量的潜在令人惊讶的行为”）。

普通变量和缓存变量之间的一个重要区别是，当FORCE关键字存在时，set()命令只会覆盖缓存变量，而不像普通变量那样，set()命令总是会覆盖预先存在的值。在定义缓存变量时，set()命令的行为更像是set-if-not-set，option()命令（其没有FORCE功能）也是如此。这样做的主要原因是缓存变量主要用作开发人员的定制点。与将值硬编码到CMakeLists.txt文件中作为普通变量的方式不同，可以使用缓存变量，以便开发人员可以在无需编辑CMakeLists.txt文件的情况下覆盖值。变量可以由交互式GUI工具或脚本修改，而无需更改项目本身。


## 5.4. 作用域块

如第5.1节“变量基础”中所述，变量具有作用域。缓存变量具有全局作用域，因此它们始终是可访问的。到目前为止提供的材料中，非缓存变量的作用域是定义变量的CMakeLists.txt文件。这通常称为目录作用域。子目录和函数从其父作用域继承变量（在第7.1.2节“作用域”和第8.4节“返回值”中分别讨论）。

在CMake 3.25或更高版本中，可以使用block()和endblock()命令定义本地变量作用域。进入块时，它会接收在那个时刻在周围范围内定义的所有变量的副本。对块中变量的任何更改都是在块的副本上执行的，从而保持周围范围的变量不变。退出块时，所有复制到块中或在块中创建的变量都将被丢弃。这可以是将特定一组命令与主逻辑隔离的一种有用方式。

```
set(x 1)
block()
 set(x 2) # 屏蔽外部的 "x"
 set(y 3) # 局部变量，在块外不可见
endblock()
# 在这里，x仍然等于1，y未定义
```

一个块可能并不总是希望与其调用者完全隔离。它可能希望有选择地修改周围范围中的一些变量。set() 和 unset() 命令的 PARENT_SCOPE 可以用于修改封闭作用域的变量而不是当前作用域：

```
set(x 1)
set(y 3)
block()
 set(x 2 PARENT_SCOPE) # 在这里，x仍然具有值1
 unset(y PARENT_SCOPE) # y仍然存在且具有值3
endblock()

# 在这里，x的值为2，y不再定义
```

当使用 PARENT_SCOPE 时，被设置或取消设置的变量是父作用域中的变量，而不是当前作用域中的变量。重要的是，这并不意味着在父作用域和当前作用域中都设置或取消设置变量。这使得 PARENT_SCOPE 使用起来可能有些麻烦，因为它通常意味着在需要影响两者时在两个不同的作用域中重复相同的命令。block() 命令支持 PROPAGATE 关键字，可以以更健壮和简洁的方式提供相同的行为。当控制流离开块时，PROPAGATE 关键字后列出的每个变量的值都从块传播到其周围的作用域。如果在块内取消设置传播的变量，则在退出块时在周围的作用域中取消设置该变量。

```
set(x 1)
set(z 5)
block(PROPAGATE x z)
 set(x 2) # 在这里x为2，同时传播回到外部的 "x" 
 set(y 3) # 局部变量，在块外不可见
 unset(z) # 也取消设置了外部的 "z"
endblock()

# 在这里，x 等于 2，y 和 z 未定义
```

block() 命令不仅可以用于控制变量作用域，还可以用于控制其他内容。该命令的完整签名如下：

```
block([SCOPE_FOR [VARIABLES] [POLICIES]] [PROPAGATE var...])
```

SCOPE_FOR 关键字可用于指定块应创建哪种类型的作用域。当省略 SCOPE_FOR 时，block() 为变量和策略（见第12.2节“策略作用域”讨论）均创建一个新的局部作用域。以下示例具有与上一个示例相同的效果，但它仅创建变量作用域，不更改策略作用域：

```
set(x 1)
set(z 5)
block(SCOPE_FOR VARIABLES PROPAGATE x z)
 set(x 2) # 传播回到外部的 "x"
 set(y 3) # 局部变量，在块外不可见
 unset(z) # 也取消设置了外部的 "z"
endblock()

# 在这里，x 等于 2，y 和 z 未定义
```

虽然 SCOPE_FOR VARIABLES 可能是大多数时候项目所需的，但允许创建新的策略作用域也通常是无害的。与 block(SCOPE_FOR VARIABLES) 相比，使用 block() 可能稍微效率较低，但可能仍然因其简洁性而更受欢迎。请参见第6.2.3节“中断循环”，第7.4节“提前结束处理”和第8.4节“返回值”了解 block() 命令与其他控制流结构的交互。

## 5.5. 变量的潜在令人惊讶的行为

一个经常不太容易理解的观点是，普通变量和缓存变量是两个不同的东西。可以拥有同名但保存不同值的普通变量和缓存变量。在这种情况下，使用 ${myVar} 时，CMake 将检索普通变量的值而不是缓存变量的值。换句话说，普通变量优先于缓存变量。例外情况是，在以下情况（根据后面进一步讨论的策略设置）下，当设置缓存变量的值时，具有相同名称的任何普通变量将从当前作用域中删除：

* 调用 set() 或 option() 之前，缓存变量不存在。
* 调用 set() 或 option() 之前，缓存变量存在，但没有定义类型（参见第5.6.1节“在命令行上设置缓存值”中可能发生的情况）。
* 在调用 set() 时使用了 FORCE 或 INTERNAL 选项。

在上述的前两种情况中，这意味着在第一次和后续的 CMake 运行之间可能会获得不同的行为。在第一次运行中，缓存变量可能不存在或者没有定义类型，但在后续运行中就会存在。因此，在第一次运行中，普通变量将被隐藏，但在后续运行中则不会。下面的示例应该有助于说明这个问题：

```
set(myVar foo) # 本地 myVar
set(result ${myVar}) # result = foo

set(myVar bar CACHE STRING "") # 缓存 myVar

set(result ${myVar})# 第一次运行: result = bar 后续运行: result = foo
set(myVar fred)
set(result ${myVar})# result = fred
```

我们看到，第一次运行时，由于缓存变量myVar没有值，所以会设置缓存变量为bar，并且删除普通变量bar，这个时候第二次设置result的结果为缓存变量bar。而第二次运行时，缓存变量已经存在，所以普通变量不会改变，这个时候普通变量优先于缓存变量被设置到result里。最后一次不管普通变量是否存在，都是把它设置为fred，因此最后一个result总是fred。

第7章“使用子目录”和第8章“函数和宏”进一步讨论了变量的作用域如何影响 ${myVar} 返回的值。

在CMake 3.13中，option() 的行为已更改，以便如果同名的普通变量已经存在，该命令不会执行任何操作。这种更新的行为通常是开发者直觉地期望的。在CMake 3.21中，对 set() 命令进行了类似的更改，但请注意以下两个命令的新行为存在一些差异：

对于 set()，如果缓存变量之前不存在，它仍然会被设置，但对于 option() 则不会。
如果在 set() 中使用了 INTERNAL 或 FORCE，缓存变量将始终被设置或更新。


开发者应注意这些不一致性以及提供新行为的不同 CMake 版本。策略 CMP0077 和 CMP0126 控制实际行为（详见第12章“策略”以了解如何操作这些策略）。

缓存变量和非缓存变量之间的交互也可能导致其他潜在的意外行为。考虑以下三个命令：

```
unset(foo)
set(foo)
set(foo "")
```

有人可能会认为，在这三种情况下，${foo} 的评估总是会产生一个空字符串，但只有最后一种情况是有保证的。unset(foo) 和 set(foo) 都会从当前作用域中移除非缓存变量。如果还有一个名为 foo 的缓存变量，那么该缓存变量将被保留，${foo} 将提供该缓存变量的值。从这个意义上说，unset(foo) 和 set(foo) 都有效地揭示了 foo 缓存变量（如果存在的话）。另一方面，set(foo "") 并不移除非缓存变量，它明确将其设置为空值，因此 ${foo} 将始终评估为空字符串，而不管是否还有一个名为 foo 的缓存变量。因此，将变量设置为空字符串而不是将其移除可能是实现开发者意图的更可靠方法。

对于那些项目可能需要获取缓存变量的值并忽略同名的非缓存变量的罕见情况，CMake 3.13 添加了对 $CACHE{someVar} 形式的文档。项目通常不应使用此功能，除非进行临时调试，因为它破坏了长期以来的期望，即普通变量将覆盖在缓存中设置的值。

## 5.6 操纵缓存变量

使用 set() 和 option()，项目可以为其开发人员建立一组有用的定制点。构建的不同部分可以打开或关闭，可以设置到外部包的路径，可以修改编译器和链接器的标志等等。后面的章节将涵盖这些以及其他缓存变量的用途，但首先需要了解操作这些变量的方式。
有两种主要方式可以进行操作，一种是通过 cmake 命令行，另一种是使用 GUI 工具。

### 5.6.1. 在命令行上设置缓存值

CMake 允许通过传递给 cmake 的命令行选项直接操纵缓存变量。主要的工作是-D选项，用于定义缓存变量的值。

```
cmake -D myVar:type=someValue ...
```
someValue 将替换 myVar 缓存变量的任何先前值。行为基本上就像使用 set() 命令配合 CACHE 和 FORCE 选项分配变量一样。
命令行选项只需要给出一次，因为它存储在缓存中以供后续运行使用，因此不需要在每次运行 cmake 时都提供。可以提供多个 -D 选项，以一次性在 cmake 命令行上设置多个变量。
使用此方式定义的缓存变量不必在 CMakeLists.txt 文件中设置（即不需要相应的 set() 命令）。通过命令行定义的缓存变量具有空的文档字符串。类型也可以省略，此时变量将具有未定义的类型，或者更准确地说，它被赋予一种类似于 INTERNAL 的特殊类型，CMake 将其解释为未定义。以下是通过命令行设置缓存变量的各种示例。

```
cmake -D foo:BOOL=ON ...
cmake -D "bar:STRING=This contains spaces" ...
cmake -D hideMe=mysteryValue ...
cmake -D helpers:FILEPATH=subdir/helpers.txt ...
cmake -D helpDir:PATH=/opt/helpThings ...
```

请注意，如果使用 -D 选项设置包含空格的缓存变量的值，则应对 -D 选项给定的整个值加引号。
处理在 cmake 命令行上最初未带类型声明的值有一个特殊情况。如果项目的 CMakeLists.txt 文件然后尝试设置相同的缓存变量并指定类型为 FILEPATH 或 PATH，则如果该缓存变量的值是相对路径，CMake 将视其为相对于调用 cmake 的目录，并自动将其转换为绝对路径。这不够健壮，因为 cmake 可以从任何目录调用，而不仅仅是构建目录。因此，建议开发人员始终在为代表某种路径的变量在 cmake 命令行上指定变量类型。总体上，一般建议始终在命令行上指定变量的类型，这样它很可能以最合适的形式显示在 GUI 应用程序中。这还将防止在 5.5 节“变量的潜在令人惊讶的行为”中前面提到的情况之一。
还可以使用 -U 选项从缓存中删除变量，如果需要，可以重复使用该选项以删除多个变量。请注意，-U 选项支持 * 和 ? 通配符，但需要小心避免删除超出意图范围的内容并使缓存处于无法构建的状态。通常建议只删除没有通配符的特定条目，除非确定使用的通配符是安全的。

```bash
cmake -U 'help*' -U foo ...
```


### 5.6.2 CMake GUI 工具

通过命令行设置缓存变量是自动化构建脚本和通过 cmake 命令驱动 CMake 的任何其他内容的基本部分。然而，对于日常开发，CMake 提供的 GUI 工具通常提供更好的用户体验。CMake 提供了两个等效的 GUI 工具，即 cmake-gui 和 ccmake，允许开发人员以交互方式操纵缓存变量。
cmake-gui 是一个在所有主要桌面平台上都受支持的功能齐全的 GUI 应用程序，而 ccmake 使用基于 curses 的界面，可在纯文本环境中使用，例如通过 ssh 连接。cmake-gui 包含在所有平台的官方 CMake 发布包中，ccmake 包含在除 Windows 外的所有平台上。如果在 Linux 上使用系统提供的软件包而不是官方版本，则请注意许多发行版将 cmake-gui 拆分为其自己的软件包。
下图显示了 cmake-gui 用户界面。顶部部分允许定义项目的源目录和构建目录。中间部分是可以查看和编辑缓存变量的地方。底部是配置和生成按钮，后面是显示这些操作的日志区域。

<a>![](/img/procmake/ch5/1.png)</a>

源目录必须设置为包含项目源树顶层的 CMakeLists.txt 文件的目录。构建目录是 CMake 将生成所有构建输出的位置（推荐的目录布局在第2章“设置项目”中讨论过）。对于新项目，两者都必须设置，但对于现有项目，设置构建目录还将更新源目录，因为源位置存储在构建目录的缓存中。
CMake 的两阶段设置过程在第2.3节“生成项目文件”中介绍过。在第一阶段，读取 CMakeLists.txt 文件并在内存中构建项目的表示。这被称为配置阶段。如果配置阶段成功，然后可以执行生成阶段，在构建目录中创建构建工具的项目文件。当从命令行运行 cmake 时，会自动执行两个阶段，但在 GUI 应用程序中，它们是通过配置和生成按钮分别触发的。
每次启动配置步骤时，UI 中间显示的缓存变量将被更新。任何新添加或与上一次运行的值不同的变量都将以红色突出显示（当首次加载项目时，所有变量都将显示为突出显示）。良好的实践是反复运行配置阶段，直到没有更改为止。这确保对于更复杂的项目，在启用某些选项可能添加需要另一个配置步骤的其他选项的情况下，保证了健壮的行为。
一旦所有缓存变量都显示为没有红色突出显示，就可以运行生成阶段。前面截图中的示例显示了在运行配置阶段并且没有更改任何缓存变量的情况下的典型日志输出。
将鼠标悬停在任何缓存变量上将显示一个包含该变量 docstring 的工具提示。还可以使用“添加条目”按钮添加新的缓存变量，这相当于使用空 docstring 发出 set() 命令。可以使用“删除条目”按钮删除缓存变量，尽管 CMake 很可能会在下一次运行时重新创建该变量。
单击变量允许编辑其值，使用特定于变量类型的小部件。布尔值显示为复选框，文件和路径具有浏览文件系统按钮，字符串通常显示为文本行编辑。作为特例，类型为 STRING 的缓存变量可以在 CMake GUI 中显示为下拉列表而不是显示为简单的文本输入小部件。这是通过设置缓存变量的 STRINGS 属性实现的（在第9.6节“缓存变量属性”中详细介绍，但在这里为了方便起见显示）：

```
set(TRAFFIC_LIGHT Green CACHE STRING "Status of something")
set_property(CACHE TRAFFIC_LIGHT PROPERTY STRINGS Red Orange Green)
```

在上面的示例中，TRAFFIC_LIGHT 缓存变量将最初具有值 Green。当用户尝试在 cmake-gui 中修改 TRAFFIC_LIGHT 时，他们将获得一个下拉框，其中包含三个值 Red、Orange 和 Green，而不是简单的行编辑小部件，否则他们将可以输入任意文本。请注意，在变量上设置 STRINGS 属性不会阻止该变量具有分配给它的其他值，它只影响在编辑时 cmake-gui 使用的小部件。可以通过在 CMakeLists.txt 文件中使用 set() 命令或通过其他手段（如手动编辑 CMakeCache.txt 文件）仍然为变量指定其他值。


缓存变量还可以具有将它们标记为高级或非高级的属性。这也只影响变量在 cmake-gui 中的显示方式，不以任何方式影响 CMake 在处理过程中如何使用变量。默认情况下，cmake-gui 只显示非高级变量，这通常只显示开发人员可能有兴趣查看或修改的主要变量。启用高级选项会显示除了那些标记为 INTERNAL 的缓存变量之外的所有缓存变量（查看 INTERNAL 变量的唯一方法是使用文本编辑器编辑 CMakeCache.txt 文件，因为它们不应由开发人员直接操纵）。可以使用 CMakeLists.txt 文件中的 mark_as_advanced() 命令将变量标记为高级：

```
mark_as_advanced([CLEAR|FORCE] var1 [var2...])
```
CLEAR 关键字确保变量未标记为高级，而 FORCE 关键字确保变量被标记为高级。在没有任何关键字的情况下，只有在变量尚未具有高级/非高级状态设置时，才会将其标记为高级。
选择 Grouped 选项可以通过根据变量名称的起始部分（直到第一个下划线）将变量分组，使查看高级变量变得更容易。显示的变量列表的另一种过滤方式是在搜索区域输入文本，这将导致仅显示其名称或值中包含指定文本的变量。
在新项目上首次运行配置阶段时，开发人员会看到类似于下一个截图中显示的对话框：
 

<a>![](/img/procmake/ch5/2.png)</a>

此对话框用于指定 CMake 生成器和工具链。生成器的选择通常取决于开发人员的个人偏好，在组合框中提供了可用选项。根据项目，生成器的选择可能比组合框选项允许的更受限制，例如，如果项目依赖于特定于生成器的功能。一个常见的例子是，由于苹果平台的独特特性，如代码签名和 iOS/tvOS/watchOS 支持，某个项目可能需要 Xcode 生成器。一旦为项目选择了生成器，就无法在不删除缓存并重新开始的情况下更改，如果需要，可以从“文件”菜单中执行这些操作。
对于所呈现的工具链选项，每个选项都需要从开发人员获取逐渐更多的信息。对于普通桌面开发来说，使用默认的本地编译器是通常的选择，选择该选项不需要进一步的详细信息。如果需要更多控制，开发人员可以选择覆盖本地编译器，随后会在一个对话框中提供编译器的路径。如果有一个单独的工具链文件可用，那么可以使用它来定制不仅编译器，还有目标环境、编译器标志和各种其他内容。在进行交叉编译时通常使用工具链文件，这在第 23 章“工具链和交叉编译”中有详细介绍。最后，为了获得最终的控制，开发人员可以指定用于交叉编译的完整选项集，但不建议在正常使用中这样做。工具链文件可以提供相同的信息，但其优势在于可以根据需要重复使用。


ccmake 工具提供了与 cmake-gui 应用程序大部分相同的功能，但通过基于文本的界面实现：


<a>![](/img/procmake/ch5/3.png)</a>



与 cmake-gui 不同，ccmake 不是选择源目录和构建目录，而是必须在 ccmake 命令行上指定源或构建目录，就像在 cmake 命令中一样。
ccmake 界面的一个小缺点是无法过滤显示的变量。编辑变量的方法也不如 cmake-gui 丰富。然而，当完整的 cmake-gui 应用程序不切实际或不可用时，ccmake 工具是一个有用的替代方案，例如在无法支持 UI 转发的终端连接上。


## 5.7 打印变量值
随着项目变得更加复杂或在调查意外行为时，打印诊断消息和变量值通常是很有用的。这通常是通过使用 message() 命令实现的，详细内容请参阅第13章《调试和诊断》。

现在，知道 message() 命令的最简形式就足够了，它的所有作用就是将其参数打印到 CMake 的输出中。如果给定的参数不止一个，它们之间不添加分隔符，并且自动在消息末尾添加换行符。换行符也可以使用常见的 \n 符号显式包含。通过使用通常的 ${myVar} 符号，可以将变量的值包含在消息中。

```
set(myVar HiThere)
message("The value of myVar = ${myVar}\nAnd this "
        "appears on the next line")
```

这将产生以下输出：
 
```
The value of myVar = HiThere
And this appears on the next line
```

## 5.8 字符串处理
随着项目复杂性的增加，很多情况下也需要实现更复杂的逻辑来管理变量。CMake 提供的一个核心工具是 string() 命令，它提供了各种有用的字符串处理功能。该命令使项目能够执行查找和替换操作、正则表达式匹配、大小写转换、去除空白和其他常见任务。下面介绍了一些较常用的功能，但 CMake 参考文档应被视为所有可用操作及其行为的权威来源。

string() 的第一个参数定义要执行的操作，后续参数取决于所请求的操作。这些参数通常至少需要一个输入字符串，并且由于 CMake 命令无法返回值，因此需要一个用于存储操作结果的输出变量。在下面的材料中，这个输出变量通常被命名为 outVar。

```
string(FIND inputString subString outVar [REVERSE])
```

FIND 在 inputString 中搜索 subString 并将找到的 subString 的索引存储在 outVar 中（第一个字符的索引是 0）。除非指定了 REVERSE，否则将找到第一个出现的 subString，否则将找到最后一个出现的 subString。如果 subString 在 inputString 中不存在，则 outVar 将被赋值为 -1。

```
string(FIND abcdefabcdef def fwdIndex)
string(FIND abcdefabcdef def revIndex REVERSE)
message("fwdIndex = ${fwdIndex}\n"
        "revIndex = ${revIndex}")
```
这将产生以下输出：

```
fwdIndex = 3
revIndex = 9
```

替换简单子字符串遵循类似的模式：

```
string(REPLACE matchString replaceWith outVar input...)
```

REPLACE 操作将替换输入字符串中 matchString 的每个出现为 replaceWith，并将结果存储在 outVar 中。当提供多个输入字符串时，在搜索替换之前，它们将被连接在一起，没有任何分隔符。这有时可能导致意外的匹配，通常在大多数情况下开发人员会在大多数情况下只提供一个输入字符串。


正则表达式在 REGEX 操作中也得到了很好的支持，根据第二个参数的不同变体，有几种不同的操作可用：

```
string(REGEX MATCH regex outVar input...)
string(REGEX MATCHALL regex outVar input...)
string(REGEX REPLACE regex replaceWith outVar input...)
```

要匹配的正则表达式 regex 可以使用典型的基本正则表达式语法（请参阅 CMake 参考文档以获取完整规范），尽管一些常见特性（如否定）不受支持。输入字符串在替换之前被连接。MATCH 操作仅找到第一个匹配项并将其存储在 outVar 中。MATCHALL 找到所有匹配项并将它们存储在 outVar 中作为列表。REPLACE 将返回整个输入字符串，其中每个匹配项都被 replaceWith 替换。可以在 replaceWith 中使用常规的 \1、\2 等表示法引用匹配项，但注意反斜杠本身必须进行转义，除非使用括号表示法。以下示例及其输出演示了上述要点：

```
string(REGEX MATCH "[ace]" matchOne abcdefabcdef)
string(REGEX MATCHALL "[ace]" matchAll abcdefabcdef)
string(REGEX REPLACE "([de])" "X\\1Y" replVar1 abc def abcdef)
string(REGEX REPLACE "([de])" [[X\1Y]] replVar2 abcdefabcdef)
message("matchOne = ${matchOne}\n"
        "matchAll = ${matchAll}\n"
        "replVar1 = ${replVar1}\n"
        "replVar2 = ${replVar2}")
```

输出结果为：

```
matchOne = a
matchAll = a;c;e;a;c;e
replVar1 = abcXdYXeYfabcXdYXeYf
replVar2 = abcXdYXeYfabcXdYXeYf
```
提取子字符串也是可能的：

```
string(SUBSTRING input index length outVar)
```

index 是一个整数，定义从 input 中提取子字符串的起始位置。将提取 length 个字符，或者如果 length 为 -1，则返回的子字符串将包含输入字符串的所有字符直到末尾。请注意，在 CMake 3.1 及更早版本中，如果 length 指向字符串的末尾，将报告错误。

字符串长度可以轻松获取，并且字符串可以轻松转换为大写或小写。从字符串的开头和结尾删除空白也很简单。这些操作的语法都共享相同的形式：

```
string(LENGTH input outVar)
string(TOLOWER input outVar)
string(TOUPPER input outVar)
string(STRIP input outVar)
```

在 LENGTH 的情况下，由于历史原因，该命令计算字节而不是字符。对于包含多字节字符的字符串，这意味着报告的长度将不同于字符数。

CMake 还提供其他操作，如字符串比较、哈希、时间戳、JSON 处理等，但它们在日常 CMake 项目中使用较少。感兴趣的读者应查阅 CMake 参考文档，了解 string() 命令的完整详情。


## 5.9 列表

列表在CMake中被广泛使用。最终，列表只是一个由分号分隔的列表项的单个字符串（有一个例外，稍后在第5.9.1节“方括号不平衡的问题”中讨论）。这使得操作单个列表项变得不太方便。CMake提供了list()命令来简化这些任务。与string()命令一样，list()期望其执行的操作作为第一个参数。第二个参数始终是要操作的列表，它必须是一个变量（即不允许传递原始列表，如a;b;c）。

最基本的列表操作包括计算项目数量和从列表中检索一个或多个项目：

```
list(LENGTH listVar outVar)
list(GET listVar index [index...] outVar)
```
示例用法：

```
set(myList a b c)
# 创建列表 "a;b;c"
list(LENGTH myList len)
message("length = ${len}")
list(GET myList 2 1 letters)
message("letters = ${letters}")
```

上面示例的输出将是：

```
length = 3
letters = c;b
```
插入、追加和在列表前添加项目也是常见任务：

```
list(INSERT listVar index item [item...])
list(APPEND listVar item [item...])
list(PREPEND listVar item [item...])  # 需要CMake 3.15或更高版本
```

与LENGTH和GET不同，INSERT、APPEND和PREPEND直接作用于listVar 并直接修改它，如下例所示：

```
set(myList a b c)
list(INSERT myList 2 X Y Z)
message("myList (first) = ${myList}")
list(APPEND myList d e f)
message("myList (second) = ${myList}")
list(PREPEND myList P Q R)
message("myList (third) = ${myList}")
```
将输出以下内容：

```
myList (first) = a;b;X;Y;Z;c
myList (second) = a;b;X;Y;Z;c;d;e;f
myList (third) = P;Q;R;a;b;X;Y;Z;c;d;e;f
```
在列表中查找特定项目遵循预期的模式：

```
list(FIND myList value outVar)
```
示例用法：

```
set(myList a b c d e)
list(FIND myList d index)
message("index = ${index}")
```
结果输出：
```
diff
Copy code
index = 3
```
有三个用于移除项目的操作，所有这些操作都直接修改列表：

```
list(REMOVE_ITEM myList value [value...])
list(REMOVE_AT myList index [index...])
list(REMOVE_DUPLICATES myList)
```
REMOVE_ITEM 操作可用于从列表中移除一个或多个项目的所有实例。如果项目不在列表中，则不会报错。另一方面，REMOVE_AT 指定要移除的一个或多个索引，并且如果指定的任何索引超过列表的末尾，CMake 将报错。REMOVE_DUPLICATES 将确保列表仅包含唯一的项目。

CMake 3.15 添加了对从列表前端或后端弹出项目并可选择存储弹出项目的支持：

```
# 需要 CMake 3.15 或更高版本
list(POP_FRONT myList [outVar1 [outVar2...]])
list(POP_BACK myList [outVar1 [outVar2...]])
```
如果没有提供 outVar，则从前端或后端弹出一个项目并将其丢弃。如果提供了一个或多个 outVar 名称，则弹出的项目将存储在这些变量中，弹出的项目数量等于提供的变量名数量。

还可以使用 REVERSE 或 SORT 操作重新排序列表项目：

```
list(REVERSE myList)
list(SORT
 myList [COMPARE method] [CASE case] [ORDER order])
```
所有可选关键字仅适用于 CMake 3.13 或更高版本的 list(SORT)。如果存在 COMPARE 选项，则方法必须是以下之一：

**STRING**：按字母顺序排序。这是在未给出 COMPARE 选项时的默认行为。

**FILE_BASENAME**：假定每个项目都是路径，并且项目应仅根据路径的基本名称部分排序。

**NATURAL**：类似于 STRING，但项目内的连续数字按数字顺序排序。这对于排序包含嵌入版本号的字符串非常有用。排序规则与 strverscmp() C 函数相同（GNU 扩展）。此排序方法仅适用于 CMake 3.18 或更高版本。

CASE 关键字需要 SENSITIVE 或 INSENSITIVE 以获取大小写，而 ORDER 关键字需要 ASCENDING 或 DESCENDING 以获取顺序。

对于所有带有索引的列表操作，负索引表示从列表的末尾开始计数。在这种用法中，列表中的最后一项索引为 -1，倒数第二项为 -2，依此类推。

上述描述了大多数可用的 list() 子命令。除非另有说明，否则这些命令自 CMake 3.0 以来一直受支持，因此项目通常应该能够期望它们可用。有关支持的子命令的完整列表，读者应参阅 CMake 文档。


### 5.9.1. 方括号不平衡的问题

CMake通常将分号视为列表分隔符，但有一个例外。由于历史原因，如果列表项包含一个开放方括号 [，它必须还有一个匹配的闭合方括号 ]。CMake将认为在这些方括号之间的任何分号都是列表项的一部分，而不是列表分隔符。如果尝试构建一个包含不平衡方括号的列表，该列表将无法按预期解释。以下演示了这种行为：

```
set(noBrackets "a_a" "b_b")
set(withBrackets "a[a" "b]b")
list(LENGTH noBrackets lenNo)
list(LENGTH withBrackets lenWith)
list(GET noBrackets 0 firstNo)
list(GET withBrackets 0 firstWith)
message("No brackets: Length=${lenNo} --> First_element=${firstNo}")
message("With brackets: Length=${lenWith} --> First_element=${firstWith}")
```

上述代码的输出将是：

```
No brackets: Length=2 --> First_element=a_a
With brackets: Length=1 --> First_element=a[a;b]b
```

在第8.8.3节，“参数扩展的特殊情况”中讨论了这个特性的更多方面。



## 5.10 数学运算

另一种常见的变量操作形式是数学计算。CMake提供了math()命令来执行基本的数学运算：

```
math(EXPR outVar mathExpr [OUTPUT_FORMAT format])
```

第一个参数必须是关键字 EXPR，而 mathExpr 定义了要评估的表达式，结果将存储在 outVar 中。该表达式可以使用以下任何运算符，它们的含义与它们在C代码中的含义相同：+ - * / % | & ^ ~ << >>。括号也受支持，并且具有它们在数学中的通常含义。变量可以用通常的 ${myVar} 符号在 mathExpr 中引用。

如果使用的是 CMake 3.13 或更高版本，则可以使用 OUTPUT_FORMAT 关键字来控制结果在 outVar 中的存储方式。格式应为 DECIMAL，这是默认行为，或 HEXADECIMAL。

```
set(x 3)
set(y 7)
math(EXPR zDec "(${x}+${y}) * 2")
message("decimal = ${zDec}")

# 需要 CMake 3.13 或更高版本以使用 HEXADECIMAL
math(EXPR zHex "(${x}+${y}) * 2" OUTPUT_FORMAT HEXADECIMAL)
message("hexadecimal = ${zHex}")
```
上述代码生成以下输出：

```
decimal = 20
hexadecimal = 0x14
```


## 5.11 最佳实践

在开发环境允许的情况下，CMake GUI 工具是一种快速、轻松理解项目构建选项并在开发过程中根据需要进行修改的有用方式。花一点时间熟悉它将简化以后处理更复杂项目的工作。它还为开发人员提供了一个良好的基础，以便在需要实验编译器设置等方面进行工作，因为这些在 GUI 环境中很容易找到并修改。

更倾向于提供用于控制是否启用构建的可选部分的缓存变量，而不是将逻辑编码在 CMake 之外的构建脚本中。这使得在 CMake GUI 和其他了解如何使用 CMake 缓存的工具中轻松启用或禁用它们。

尽量避免依赖于定义的环境变量，除了可能是 ubiquitous 的 PATH 或类似的操作系统级变量。构建应该是可预测的、可靠的，并且易于设置，但是如果它依赖于设置环境变量才能正确工作，这可能会让新开发人员感到沮丧，因为他们在努力设置构建环境时可能会遇到问题。此外，与运行 CMake 时的环境相比，构建本身调用时的环境可能会发生变化。因此，在可能的情况下，最好通过缓存变量直接将信息传递给 CMake。

尽量早地建立一个变量命名约定。对于缓存变量，考虑将相关变量分组在一个以常见前缀开头的下划线后面，以充分利用 CMake GUI 自动根据相同前缀对变量进行分组的功能。还要考虑到项目可能有一天会成为某个更大项目的子部分，因此可能希望以项目名称开头或与项目密切相关的名称。

尽量避免在项目中定义与缓存变量同名的非缓存变量。对于新手来说，这两种类型的变量之间的交互可能出乎意料。后面的章节还强调了与缓存变量同名的常规变量的其他常见错误和误用。

CMake 提供了大量预定义变量，这些变量提供有关系统的详细信息或影响 CMake 行为的某些方面。其中一些变量被项目广泛使用，例如只在为特定平台构建时定义的变量（WIN32、APPLE、UNIX 等）。因此，建议开发人员偶尔快速浏览 CMake 文档页面中列出的预定义变量，以帮助熟悉可用的内容。












