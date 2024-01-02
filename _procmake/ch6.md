---
layout:     post
title:      "第六章：流程控制"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - cmake 
---



 <!--more-->

大多数 CMake 项目的常见需求是仅在特定情况下执行某些步骤。例如，项目可能只希望在特定编译器或构建特定平台时使用特定的编译器标志。在其他情况下，项目可能需要迭代一组值或重复执行一些步骤，直到满足某个条件为止。这些流程控制的示例在CMake中得到很好的支持，这对大多数软件开发人员应该是熟悉的。无处不在的 if() 命令提供了预期的 if-then-else 行为，而循环是通过 foreach() 和 while() 命令提供的。这三个命令提供了大多数编程语言中实现的传统行为，但它们还具有特定于CMake的附加功能。

## 6.1. if() 命令

if() 命令的现代形式如下（可以提供多个 elseif() 子句）：

cmake
Copy code
if(expression1)
 
 # 命令 ...
elseif(expression2)
 
 # 命令 ...
else()
 
 # 命令 ...
endif()
早期版本的 CMake 要求将 expression1 重复作为 else() 和 endif() 子句的参数，但自 CMake 2.8.0 以来就不再需要了。尽管在项目和示例代码中仍然可能会遇到使用旧形式的情况，但对于新项目而言，这是不鼓励的，因为它可能会让阅读代码变得有些困惑。新项目应该将 else() 和 endif() 的参数留空，如上所示。

if() 和 elseif() 命令中的表达式可以采用各种不同的形式。CMake 提供传统的布尔逻辑以及文件系统测试、版本比较和测试存在性等各种其他条件。

### 6.1.1. 基本表达式

最基本的表达式是单个常量值：

```
if(value)
```

CMake 认为什么是真值和假值的逻辑比大多数编程语言都更复杂一些。对于单个未引用的值，规则如下：

* 如果 value 是带有值 ON、YES、TRUE 或 Y 的引用或未引用常量，则将其视为真值。测试是不区分大小写的。
* 如果 value 是带有值 OFF、NO、FALSE、N、IGNORE、NOTFOUND，空字符串或以 -NOTFOUND 结尾的字符串的引用，则将其视为假值。同样，测试是不区分大小写的。
* 如果 value 是（可能是浮点数的）数字，它将按照通常的 C 规则转换为布尔值，尽管在这种情况下很少使用除 0 或 1 以外的值。
* 如果上述情况都不适用，则将其视为变量名（或可能是字符串）并根据下文描述的方式进一步评估。


在以下示例中，仅出于说明目的显示了命令的 if(...) 部分，相应的主体和 endif() 被省略：

```
# 带引号和不带引号常量的示例
if(YES)
if("True")
if(0)
if(TRUE)
# 这些也被视为不带引号的常量，因为
# 变量的评估发生在 if() 看到值之前
set(A YES)
set(B 0)
if(${A}) # 评估为 true
if(${B}) # 评估为 false


# 不匹配任何 true 或 false 常量，因此继续
# 在下面的 fall-through 情况下测试为变量名
if(someLetters)


# 带引号的值不匹配任何 true 或 false 常量，
# 因此再次继续测试为变量名或字符串
if("someLetters")
```
CMake 文档将 fall-through 情况称为以下形式：

```
if(<variable|string>)
```
在实践中，这意味着 if 表达式是：

* 不带引号的（可能未定义的）变量的名称。
* 带引号的字符串的字符串。

当使用不带引号的变量名时，变量的值将与假常量进行比较。如果这些值中没有与变量的值匹配的值，则表达式的结果为 true。未定义的变量将评估为空字符串，该字符串与假常量之一匹配，因此将导致结果为 false。

```
# 常见模式，通常与由诸如选项（enableSomething "..."）定义的变量一起使用
if(enableSomething)
 
 # ...
endif()
```
请注意，在此讨论中，环境变量不算作变量。类似 if(ENV{some_var}) 的语句将始终评估为 false，无论名为 some_var 的环境变量是否存在。

当 if 表达式是一个引号括起的字符串时，行为更加复杂：

在 CMake 3.1 或更高版本中，不匹配任何定义的 true 常量的引号字符串总是评估为 false，无论字符串的值如何（但这可以通过设置 CMP0054 策略进行覆盖，详见第12章，策略）。
在 CMake 3.1 之前，如果字符串的值与现有变量的名称匹配，那么引号字符串将有效地被该变量名称（未引用）替换，然后重复进行测试。
上述两者可能会让开发人员感到惊讶，但至少 CMake 3.1 及更高版本的行为始终是可预测的。在 3.1 之前的行为有时会导致意外的字符串替换，当字符串值恰好与变量名匹配时，可能是在项目的相当远的地方定义的变量名。对于引号值周围的潜在混淆，通常建议避免在 if(something) 形式中使用引号参数。通常有更好的比较表达式可以更稳健地处理字符串，这将在稍后的第6.1.3节“比较测试”中进行讨论。

### 6.1.2  逻辑运算符
CMake 支持通常的 AND、OR 和 NOT 逻辑运算符，以及用于控制优先级顺序的括号。

```
# 逻辑运算符
if(NOT expression)
if(expression1 AND expression2)
if(expression1 OR expression2)

# 使用括号的示例
if(NOT (expression1 AND (expression2 OR expression3)))
```
遵循通常的约定，括号内的表达式首先进行评估，从最内部的括号开始。


### 6.1.3  比较测试
CMake 将比较测试分为不同的类别：数值、字符串、版本号和路径，但语法形式都遵循相同的模式：

```
if(value1 OPERATOR value2)
```

这两个操作数，value1 和 value2，可以是变量名或（可能带引号的）值。如果一个值与已定义变量的名称相同，它将被视为变量。否则，它将直接被视为字符串或值。不过，引号引起的值在基本一元表达式中有类似的模糊行为。在 CMake 3.1 之前，具有与变量名匹配的值的带引号字符串将被替换为该变量的值。CMake 3.1 及更高版本的行为使用带引号的值而不进行替换，这是开发者直观期望的结果。
所有比较类别支持相同的一组操作，但每个类别的 OPERATOR 名称是不同的。以下表格总结了支持的运算符：

| 比较类型      | 数值              | 字符串         | 版本号            | 路径              |
| ------------ | ----------------- | -------------- | ----------------- | ----------------- |
| 小于           | LESS              | STRLESS        | VERSION_LESS      | 无               |
| 大于           | GREATER           | STRGREATER     | VERSION_GREATER   | 无               |
| 等于           | EQUAL             | STREQUAL       | VERSION_EQUAL     | PATH_EQUAL$^2$ |
| 小于等于       | LESS_EQUAL        | STRLESS_EQUAL $^1$  | VERSION_LESS_EQUAL| 无               |
| 大于等于       | GREATER_EQUAL     | STRGREATER_EQUAL| VERSION_GREATER_EQUAL| 无               |

$^1$ CMake 3.7和之后的版本可用

$^2$ `PATH_EQUAL` 仅用于路径比较。




数值比较按预期工作，将左侧的值与右侧的值进行比较。请注意，如果任一操作数不是数字，CMake通常不会引发错误，并且其行为在值包含除数字以外的其他内容时不完全符合官方文档。根据数字和非数字的混合情况，表达式的结果可能为true或false。

```
# 有效的数字表达式，所有都评估为true
if(2 GREATER 1)
if("23" EQUAL 23)
set(val 42)
if(${val} EQUAL 42)
if("${val}" EQUAL 42)
# 无效的表达式，某些CMake版本中评估为true。不要依赖此行为。
if("23a" EQUAL 23)
```
版本号比较有点像数字比较的增强形式。假定版本号的格式为 major[.minor[.patch[.tweak]]]，其中每个组件都应为非负整数。在比较两个版本号时，首先比较主要部分。只有在主要部分相等时，才会比较次要部分（如果存在），依此类推。缺少的组件被视为零。在以下所有示例中，表达式都评估为true：

```
if(1.2 VERSION_EQUAL 1.2.0)
if(1.2 VERSION_LESS 1.2.3)
if(1.2.3 VERSION_GREATER 1.2)
if(2.0.1 VERSION_GREATER 1.9.7)
if(1.8.2 VERSION_LESS 2)
```

版本号比较具有与数字比较相同的健壮性警告。每个版本组件都应为整数，但如果不符合此限制，比较结果基本上是未定义的。

对于字符串，将按词典顺序进行比较。关于字符串内容，不做任何假设，但要注意先前描述的变量/字符串替换情况可能导致的意外替换情况。字符串比较是出现此类意外替换的最常见情况之一。

PATH_EQUAL 操作符类似于 STREQUAL 的特殊情况。假定操作数是CMake本机路径形式的路径（即用于目录分隔符的正斜杠）。对于 PATH_EQUAL，一个主要的实际区别是它使用逐组件比较。多个连续的目录分隔符会被折叠成单个分隔符，这是与 STREQUAL 的主要实际区别。

以下是官方CMake文档中演示区别的示例：

```
# 比较为真
if ("/a//b/c" PATH_EQUAL "/a/b/c")
  ...
endif()

# 比较为假
if ("/a//b/c" STREQUAL "/a/b/c")
  ...
endif()
```

除了上述的操作符形式之外，还可以将字符串与正则表达式进行匹配：

```
if(value MATCHES regex)
```

值再次遵循上述定义的变量或字符串规则，并与正则表达式进行比较。如果值匹配，则表达式评估为true。虽然CMake文档没有为 if() 命令定义支持的正则表达式语法，但在其他命令的文档中定义了它（例如，请参阅 string() 命令的文档）。基本上，CMake仅支持基本的正则表达式语法。

括号可以用于捕获匹配值的部分。该命令将设置类似于 CMAKE_MATCH_<n> 形式的变量，其中 <n> 是要匹配的组。整个匹配的字符串存储在组 0 中。

```
if("Hi from ${who}" MATCHES "Hi from (Fred|Barney).*")
  message("${CMAKE_MATCH_1} says hello")
endif()
```



### 6.1.4  文件系统测试

CMake还包括一组用于查询文件系统的测试：

* 如果文件或目录存在：if(EXISTS pathToFileOrDir)
* 如果路径指向目录：if(IS_DIRECTORY pathToDir)
* 如果文件名是符号链接：if(IS_SYMLINK fileName)
* 如果路径是绝对路径：if(IS_ABSOLUTE path)
* 如果file1更新时间比file2晚：if(file1 IS_NEWER_THAN file2)

与大多数其他 if() 表达式不同，上述运算符都不执行任何变量/字符串替换，而不管有没有引号。如果想变量替换，请使用${}。

上述每个运算符都应该是不言自明的，除了 IS_NEWER_THAN。不幸的是，IS_NEWER_THAN 对该运算符的操作命名不准确。它还在文件1和文件2具有相同时间戳时返回true，而不仅仅是文件1的时间戳比文件2的新。这在仅具有一秒分辨率时间戳的文件系统上特别重要，例如macOS 10.12及更早版本上的HFS+文件系统。在这样的系统上，经常会遇到这样的情况，即文件具有相同的时间戳，即使这些文件是由不同的命令创建的。另一个不那么直观的行为是，如果任一文件丢失，它也将返回true。此外，如果未将任一文件指定为绝对路径，则行为是不确定的。因此，通常需要以否定的方式使用 IS_NEWER_THAN 以获得所需的条件。

考虑一个场景，其中 secondFile 是从 firstFile 生成的。如果更新了 firstFile 或 secondFile 丢失，则需要重新创建 secondFile。如果 firstFile 不存在，则应该是致命错误。此类逻辑需要这样表达：

```
set(firstFile "/full/path/to/somewhere")
set(secondFile "/full/path/to/another/file")

if(NOT EXISTS ${firstFile})
  message(FATAL_ERROR "${firstFile} is missing")
elseif(NOT EXISTS ${secondFile} OR NOT ${secondFile} IS_NEWER_THAN ${firstFile})
  # ... 重新创建secondFile的命令
endif()
```

有人可能天真地认为可以下面的条件来表示：

```
# 警告：很可能是错误的
if(${firstFile} IS_NEWER_THAN ${secondFile})
  # ... 重新创建secondFile的命令
endif()
```
尽管这样的语句可能表达所需的条件，但它并不执行看似可以的操作，因为它也在两个文件具有相同时间戳时返回true。如果重新创建 secondFile 的操作很快，并且文件系统仅具有秒级时间戳分辨率，那么很可能每次运行CMake时都会重新创建 secondFile。如果构建步骤依赖于 secondFile，则构建也会在每次运行CMake后重新构建这些内容。所以比较安全的方法是使用前面那个复杂一定的方法。当然前一种方法可能重复构建，但是总好过遗漏了。

### 6.1.5 存在性测试

最后一类 if() 表达式支持测试各种CMake实体是否存在。它们在更大、更复杂的项目中特别有用，其中一些部分可能存在或未存在，或者已启用或未启用。


* 如果变量 name 已定义：if(DEFINED name)
* 如果命令 name 存在：if(COMMAND name)
* 如果策略 name 存在：if(POLICY name)
* 如果目标 name 存在：if(TARGET name)
* 如果测试 name 存在：if(TEST name)
* 在CMake 3.3及更高版本中，如果值在列表 listVar 中：if(value IN_LIST listVar)

上述除最后一种形式外，都将在发出 if() 命令的位置存在指定名称的实体时返回true。

**DEFINED**

如果指定名称的变量存在，则返回true。变量的值无关紧要，只测试其存在性。该变量可以是常规的CMake变量，也可以是缓存变量。从CMake 3.14起，可以使用 CACHE{name} 形式仅检查缓存变量的存在性。所有CMake版本还支持使用 ENV{name} 形式测试环境变量的存在性，尽管这只有从CMake 3.13开始才正式记录为支持。

```
if(DEFINED SOMEVAR)
  # 检查CMake变量（常规或缓存）
if(DEFINED CACHE{SOMEVAR})
  # 检查CMake缓存变量
if(DEFINED ENV{SOMEVAR})
  # 检查环境变量
```

**COMMAND**

测试是否存在具有指定名称的CMake命令、函数或宏。这对于在尝试使用之前检查是否定义了某些内容很有用。对于CMake提供的命令，最好测试CMake版本，但对于项目提供的函数和宏（请参见第8章，“函数和宏”），这可能是一个适当的检查。

**POLICY**

测试CMake是否知道特定的策略。策略名称通常是 CMPxxxx 的形式，其中 xxxx 是一个四位数。有关此主题的详细信息，请参见第12章，“策略”。

**TARGET**

如果由 add_executable()、add_library() 或 add_custom_target() 命令之一定义了指定名称的CMake目标，则返回true。该目标可以在任何目录中定义，只要在执行 if() 测试的地方已知即可。这个测试在复杂的项目层次结构中非常有用，该层次结构引入了其他外部项目，并且这些项目可能共享常见的依赖子项目（即此类 if() 测试可用于在尝试创建目标之前检查目标是否已定义）。

**TEST**

如果由 add_test() 命令（在第26章，“测试”中详细介绍）之前定义了具有指定名称的CMake测试，则返回true。

**IN_LIST**

如果 listVar 包含指定的值，则返回true，其中值遵循通常的变量或字符串规则。listVar 必须是列表变量的名称，不能是字符串。

```
# 正确
set(things A B C)
if("B" IN_LIST things)
  # ...
endif()

# 错误：右侧必须是变量的名称
if("B" IN_LIST "A;B;C")
  # ...
endif()
```
还要注意，仅当策略 CMP0057 为 NEW 时，才能使用 IN_LIST。


### 6.1.6 常见示例

有几种对 if() 的使用非常常见，值得特别提及。其中许多依赖于其逻辑的预定义CMake变量，特别是与编译器和目标平台相关的变量。不幸的是，经常看到这些表达式基于错误的变量。例如，考虑一个具有两个C++源文件的项目，一个用于使用Visual Studio编译器或与其兼容的编译器（例如Intel）构建，另一个用于与所有其他编译器构建。这样的逻辑通常实现如下：

```
if(WIN32)
 
 set(platformImpl source_win.cpp)
else()
 
 set(platformImpl source_generic.cpp)
endif()
```

虽然这可能对大多数项目有效，但实际上并没有表达正确的约束。例如，考虑一个在Windows上构建但使用MinGW编译器的项目。对于这种情况，source_generic.cpp 可能是更合适的源文件。上述可以更准确地实现如下：

```
if(MSVC)
 
 set(platformImpl source_msvc.cpp)
else()
 
 set(platformImpl source_generic.cpp)
endif()
```

另一个例子涉及基于使用的CMake生成器的条件行为。特别是，当使用Xcode生成器构建时，CMake提供了其他生成器不支持的附加功能。项目有时会假设构建macOS意味着将使用Xcode生成器，但这并非必须的（通常也不是）。有时会使用以下不正确的逻辑：

```
if(APPLE)
 
 # 这里是一些Xcode特定的设置...
else()
 
 # 在其他平台上的设置...
endif()
```

再次，这可能看起来是正确的，但是如果开发人员尝试在macOS上使用不同的生成器（例如Ninja或Unix Makefiles），那么该逻辑将失败。使用表达式 APPLE 测试平台并不表达正确的条件，应该测试CMake生成器：

```
if(CMAKE_GENERATOR STREQUAL "Xcode")
 
 # 这里是一些Xcode特定的设置...
else()
 
 # 在其他CMake生成器上的设置...
endif()
```

上述示例都是测试平台而不是与约束实际相关的实体的情况。这是可以理解的，因为平台是最简单的要理解和测试的事物之一，但与使用它而不是更准确的约束相比，可能会不必要地限制开发人员可用的生成器选择，或者可能完全导致错误的行为。

另一个常见的例子，这次是适当使用的，是根据是否设置了特定的CMake选项来有条件地包含目标。

```
option(BUILD_MYLIB "Enable building the MyLib target")
if(BUILD_MYLIB)
 add_library(MyLib src1.cpp src2.cpp)
endif()
```

更复杂的项目通常使用上述模式根据CMake选项或缓存变量有条件地包含子目录或执行各种其他任务。然后，开发人员可以打开/关闭该选项或将变量设置为非默认值，而无需直接编辑 CMakeLists.txt 文件。这对于由持续集成系统驱动的脚本化构建特别有用，这些系统可能希望在构建的某些部分启用或禁用某些部分。


## 6.2 循环

在许多CMake项目中，另一个常见的需求是对一组项目执行某些操作或对一系列值进行操作。或者，可能需要在特定条件满足之前重复执行某个操作。CMake很好地满足了这些需求，提供了传统的行为以及一些额外的功能，使与CMake功能一起工作变得更加容易。

### 6.2.1 foreach()

CMake提供了 foreach() 命令，使项目能够对一组项目或一系列值进行迭代。 foreach() 有几种不同的形式，其中最基本的形式是：

```
foreach(loopVar arg1 arg2 ...)
 
 # ...
endforeach()
```

在上述形式中，对于每个 argN 值，loopVar 被设置为该参数，然后执行循环体。不执行变量/字符串测试，参数的使用方式与值的指定方式完全相同。而不是明确列出每个项目，还可以使用命令的更一般形式通过一个或多个列表变量指定参数：

```
foreach(loopVar IN [LISTS listVar1 ...] [ITEMS item1 ...])
 
 # ...
endforeach()
```


在这更一般的形式中，可以仍然使用 ITEMS 关键字指定单个参数，但是 LISTS 关键字允许指定一个或多个列表变量。在使用这更一般的形式时，必须提供 ITEMS 或 LISTS（或两者）。当两者都提供时，ITEMS 必须出现在 LISTS 之后。允许 listVarN 列表变量保存空列表。示例应有助于澄清这更一般的形式的用法。

```
set(list1 A B)
set(list2)
set(foo WillNotBeShown)
foreach(loopVar IN LISTS list1 list2 ITEMS foo bar)
 
 message("Iteration for: ${loopVar}")
endforeach()
```
上述的输出将是：

```
Iteration for: A
Iteration for: B
Iteration for: foo
Iteration for: bar
```

CMake 3.17 添加了一种更专业的形式，用于同时循环遍历多个列表：

```
foreach(loopVar... IN ZIP_LISTS listVar...)
 
 # ...
endforeach()
```

如果只给定一个 loopVar，那么命令将在每次迭代时设置形式为 loopVar_N 的变量，其中 N 对应于 listVarN 变量。编号从 0 开始。如果对于每个 listVar 都有一个 loopVar，则命令将它们一对一地映射，而不是创建 loopVar_N 变量。以下示例演示了这两种情况：

```
set(list0 A B)
set(list1 one two)
foreach(var0 var1 IN ZIP_LISTS list0 list1)
 
 message("Vars: ${var0} ${var1}")
endforeach()
foreach(var IN ZIP_LISTS list0 list1)
 
 message("Vars: ${var_0} ${var_1}")
endforeach()
```

两个 foreach() 循环都将输出相同的结果：

```
Vars: A one
Vars: B two
```

以这种方式zip方式的列表不必具有相同的长度。当迭代超过较短列表的末尾时，相关的迭代变量将未定义。取未定义变量的值会导致空字符串。下面的示例演示了这种行为：

```
set(long A B C)
set(short justOne)
foreach(varLong varShort IN ZIP_LISTS long short)
 
 message("Vars: ${varLong} ${varShort}")
endforeach()
```
输出结果为：

```
Vars: A justOne
Vars: B
Vars: C
```

foreach() 命令还支持更类似于C语言的迭代方式，遍历数值范围：

```
foreach(loopVar RANGE start stop [step])
```

在使用 RANGE 形式的 foreach() 时，循环会以 loopVar 设置为范围 start 到 stop（含）中的每个值而执行。如果提供了 step 选项，则在每次迭代后将该值添加到前一个值，当其结果大于 stop 时循环停止。RANGE 形式还可以只接受一个参数，如下：

```
foreach(loopVar RANGE value)
```

这相当于 foreach(loopVar RANGE 0 value)，这意味着循环体将执行（value + 1）次。这是令人遗憾的，因为更直观的期望可能是循环体执行 value 次。出于这个原因，避免使用第二种 RANGE 形式，而是明确指定起始和停止值可能更清晰。

与 if() 和 endif() 命令的情况类似，在CMake的早期版本中（即在2.8.0之前），foreach() 命令的所有形式都要求 loopVar 也在 endforeach() 的参数中指定。同样，这会损害可读性并提供很少的好处，因此对于新项目不鼓励在 endforeach() 中指定 loopVar。

### 6.2.2 while()


CMake 提供的另一种循环命令是 while()：

```
while(condition)
 
 # ...
endwhile()
```

条件将被测试，如果它评估为 true（遇到与 if() 语句中的表达式相同的规则），则执行循环体。这将重复，直到条件评估为 false 或循环被提前退出（见下一节）。同样，在CMake版本2.8.0之前，条件必须在 endwhile() 命令中重复，但这现在不再是必需的，并且对于新项目来说，这是被积极不鼓励的。



### 6.2.3. 中断循环
while() 和 foreach() 循环都支持使用 break() 提前退出循环，或者使用 continue() 跳到下一次迭代的开始。这些命令的行为与它们在 C 语言中的同名对应物相似，并且都仅在最内层的封闭循环上操作。下面的例子演示了这种行为：

```
foreach(outerVar IN ITEMS a b c)
    unset(s)

    foreach(innerVar IN ITEMS 1 2 3)
        # 一旦字符串 s 变长，就停止内部循环
        list(APPEND s "${outerVar}${innerVar}")
        string(LENGTH "${s}" length)
        if(length GREATER 5)
            # 提前结束 innerVar foreach 循环
            break()
        endif()

        # 如果 outerVar 是 "b"，则不再执行更多处理
        if(outerVar STREQUAL "b")
            # 结束当前 innerVar 迭代，继续下一个 innerVar 项目
            continue()
        endif()

        message("Processing ${outerVar}-${innerVar}")
        message("Accumulated list: ${s}")
    endforeach()
endforeach()
```

上述例子的输出将为：

```
Processing a-1
Processing a-2
Accumulated list: a1;a2;a3
Accumulated list: b1;b2;b3
Processing c-1
Processing c-2
Accumulated list: c1;c2;c3
```

break() 和 continue() 命令也可以在由 block() 和 endblock() 命令定义的块内使用（参见第5.4节，“范围块”）。通过 break() 或 continue() 离开块会结束该块的本地作用域。下面是一个构造性的例子来演示这种行为：

```
set(log "Value: ")
set(values one two skipMe three stopHere four)
set(didSkip FALSE)

while(NOT values STREQUAL "")
    list(POP_FRONT values next)

    # "log" 的修改将被丢弃
    block(PROPAGATE didSkip)
        string(APPEND log "${next}")
        if(next MATCHES "skip")
            set(didSkip TRUE)
            continue()
        elseif(next MATCHES "stop")
            break()
        elseif(next MATCHES "t")
            string(APPEND log ", has t")
        endif()
        message("${log}")
    endblock()
endwhile()

message("Did skip: ${didSkip}")
message("Remaining values: ${values}")
```

上述例子的输出将为：

```
Value: one
Value: two, has t
Value: three, has t
Did skip: TRUE
Remaining values: four
```


### 6.3 最佳实践

在 if()、foreach() 和 while() 命令中，尽量减少字符串被错误解释为变量的机会。避免带引号的一元表达式，最好使用字符串比较操作。强烈建议将 CMake 版本设置为至少 3.1，以禁用允许将带引号的字符串值隐式转换为变量名的旧行为。
在 if(xxx MATCHES regex) 命令中使用正则表达式匹配并需要组捕获变量时，通常建议尽早将 CMAKE_MATCH_<n> 匹配结果存储在普通变量中。这些变量将被下一个执行任何正则表达式操作的命令覆盖。
在使用循环命令时，最好避免模糊或误导性的代码。如果使用 foreach() 的 RANGE 形式，请始终指定开始和结束值。如果迭代项目，请考虑使用 IN LISTS 或 IN ITEMS 形式，以更清晰地传达正在进行的操作，而不是裸露的 foreach(loopVar item1 item2 ...) 形式。


 






