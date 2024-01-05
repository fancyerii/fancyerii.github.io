---
layout:     post
title:      "第八章：函数和宏"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - cmake 
---



 <!--more-->

回顾迄今为止本书所涵盖的内容，CMake的语法已经开始看起来很像一种独立的编程语言。它支持变量、if-then-else逻辑、循环和包含其他要处理的文件。毫不奇怪地，CMake还支持函数和宏这些常见的编程概念。与它们在其他编程语言中的作用类似，函数和宏是项目和开发人员扩展CMake功能、以自然的方式封装重复任务的主要机制。它们允许开发人员定义可重用的CMake代码块，可以像常规内置CMake命令一样调用。它们还是CMake自己模块系统的基石（在第11章“模块”中单独介绍）。

## 8.1 基础知识

在CMake中，函数和宏与它们在C/C++中的同名对应物具有非常相似的特性。函数引入一个新的作用域，函数参数成为在函数体内可以访问的变量。而宏则有效地将其主体粘贴到调用点，并将宏参数替换为简单的字符串替换。这些行为反映了函数和 #define 宏在C/C++中的工作方式。CMake中的函数或宏定义如下：

```
function(name [arg1 [arg2 [...]]])
    # 函数体（即命令）...
endfunction()

macro(name [arg1 [arg2 [...]]])
    # 宏体（即命令）...
endmacro()
```

一旦定义，函数或宏的调用方式与任何其他CMake命令完全相同。然后在调用点执行函数或宏的体。例如：

```
function(print_me)
    message("Hello from inside a function")
    message("All done")
endfunction()

# 如下调用：
print_me()
```

如上所示，name 参数定义了用于调用函数或宏的名称，它应只包含字母、数字和下划线。名称将以不区分大小写的方式处理，因此大小写约定更多地是一种风格问题（CMake文档遵循所有命令名称都是小写，由下划线分隔的约定）。CMake的早期版本要求将名称重复作为 endfunction() 或 endmacro() 的参数，但新项目应避免这样做，因为这只会添加不必要的混乱。

## 8.2. 参数处理基础
函数和宏的参数处理方式基本相同，除了一个非常重要的区别。对于函数，每个参数都是一个CMake变量，并具有CMake变量的所有通常行为。例如，它们可以在 if() 语句中作为变量进行测试。相比之下，宏参数是字符串替换，因此无论在宏体中的何处使用宏调用的参数，都会将其实质粘贴到该参数出现的位置。如果在 if() 语句中使用宏参数，它将被视为字符串而不是变量。以下示例及其输出演示了这一差异：

```
function(func arg)
    if(DEFINED arg)
        message("Function arg is a defined variable")
    else()
        message("Function arg is NOT a defined variable")
    endif()
endfunction()

macro(macr arg)
    if(DEFINED arg)
        message("Macro arg is a defined variable")
    else()
        message("Macro arg is NOT a defined variable")
    endif()
endmacro()


func(foobar)
macr(foobar)
```
输出结果为：

```
Function arg is a defined variable
Macro arg is NOT a defined variable
```

除此之外，函数和宏在参数处理方面的其他特性是相同的。函数定义中的每个参数都充当对应参数的区分大小写的标签。对于函数，该标签充当变量，而对于宏，它充当字符串替换。该参数的值可以在函数或宏体中使用通常的变量表示法访问，尽管宏参数在技术上不是变量。例如：

```
function(func myArg)
    message("myArg = ${myArg}")
endfunction()

macro(macr myArg)
    message("myArg = ${myArg}")
endmacro()

func(foobar)
macr(foobar)
```

func() 和 macr() 的调用都打印出相同的内容：

```
myArg = foobar
```

除了命名参数之外，函数和宏还带有一组自动定义的变量（或者在宏的情况下是类似变量的名称），允许处理除了或替代命名参数之外的其他参数：

**ARGC**：将被设置为传递给函数的参数总数。它计算命名参数加上任何给定的额外未命名参数的总数。
**ARGV**：这是一个包含传递给函数的每个参数的列表，包括命名参数和任何给定的额外未命名参数。
**ARGN**：与 ARGV 相似，不同之处在于它仅包含超出命名参数的参数（即可选的未命名参数）。

除上述之外，每个单独的参数可以通过形如 ARGVx 的名称引用，其中 x 是参数的编号（例如 ARGV0、ARGV1 等）。这包括了命名参数，因此第一个命名参数也可以通过 ARGV0 等引用。请注意，使用 ARGVx 且 x >= ARGC 应被视为未定义行为。

典型的情况下，ARG... 这些名称的使用包括支持可选参数和实现一个可以接受任意数量项进行处理的命令。考虑一个定义可执行目标、将该目标链接到某个库并为其定义测试用例的函数。这样的函数在编写测试用例时经常遇到（测试是第26章中涵盖的主题）。与其为每个测试用例重复这些步骤，该函数允许一次性定义这些步骤，然后每个测试用例都变成了一个简单的一行定义。

```
# 使用命名参数指定目标，并将所有其他（未命名）参数视为测试用例的源文件
function(add_mytest targetName)
    add_executable(${targetName} ${ARGN})
    target_link_libraries(${targetName} PRIVATE foobar)
    add_test(NAME ${targetName} COMMAND ${targetName})
endfunction()

# 使用上述函数定义一些测试用例
add_mytest(smallTest small.cpp)
add_mytest(bigTest big.cpp algo.cpp net.cpp)
```

上述示例特别展示了 ARGN 的用途。它允许函数或宏接受可变数量的参数，同时仍指定必须提供的一组命名参数。



然而，有一个特定的情况需要注意，可能导致意外的行为。因为宏将其参数视为字符串替换而不是变量，如果宏在期望变量名的地方使用 ARGN，它将引用的变量将在调用宏的作用域中，而不是宏自己的参数中的 ARGN。以下示例突显了这种情况：

```
# 警告：此宏具有误导性
macro(dangerous)
    # 是哪个 ARGN？
    foreach(arg IN LISTS ARGN)
        message("Argument: ${arg}")
    endforeach()
endmacro()

function(func)
    dangerous(1 2)
endfunction()

func(3)
```

上述示例的输出将是：

```
Argument: 3
```

在使用 foreach() 与 LISTS 关键字时，必须提供一个变量名，但是为宏提供的 ARGN 不是一个变量名。当宏从另一个函数内部调用时，该宏最终使用来自该封闭函数的 ARGN 变量，而不是宏本身的 ARGN。将宏体的内容直接粘贴到调用它的函数中时（这实际上就是 CMake 将要做的），情况变得清晰：

```
function(func)
    # 现在清楚了，ARGN 在这里将使用来自 func 的参数
    foreach(arg IN LISTS ARGN)
        message("Argument: ${arg}")
    endforeach()
endfunction()
```

在这种情况下，考虑将宏改为函数，或者如果必须保持为宏，则避免将参数视为变量。对于上述示例，dangerous() 的实现可以更改为使用 foreach(arg IN ITEMS ${ARGN})，但请参阅第8.8节“参数处理的问题”以了解一些潜在的注意事项。



## 8.3. 关键字参数

前一节说明了如何使用 ARG... 变量来处理一组可变的参数。这种功能对于只需要一个集合的可变或可选参数的简单情况足够了，但如果必须支持多个可选或可变参数集，则处理变得相当繁琐。此外，上述基本参数处理与CMake的许多内置命令相比相当刻板，后者支持基于关键字的参数和灵活的参数排序。

考虑 target_link_libraries() 命令：

```
target_link_libraries(targetName
    <PRIVATE|PUBLIC|INTERFACE> item1 [item2 ...]
    [<PRIVATE|PUBLIC|INTERFACE> item3 [item4 ...]]
    ...
)
```

targetName 是作为第一个参数必需的，但在此之后，调用者可以以任何顺序提供任意数量的 PRIVATE、PUBLIC 或 INTERFACE 部分，每个部分允许包含任意数量的项。用户定义的函数和宏可以通过使用 cmake_parse_arguments() 命令实现类似的灵活性，该命令有两种形式。第一种形式得到所有CMake版本的支持，可用于函数和宏：

```
# 仅适用于 CMake 3.4 及更早的版本
include(CMakeParseArguments)
cmake_parse_arguments(
    prefix
    valuelessKeywords singleValueKeywords multiValueKeywords
    argsToParse...
)
```

cmake_parse_arguments() 命令曾由 CMakeParseArguments 模块提供，但在 CMake 3.5 中成为内置命令。在 CMake 3.5 及更高版本中，include(CMakeParseArguments) 行不起作用，而在较早版本的 CMake 中，它将定义 cmake_parse_arguments() 命令（有关 include() 此类用法的更多信息，请参见第11章“模块”）。

第二种形式是在 CMake 3.7 中引入的，只能在函数中使用，而不能在宏中使用：

```
# 仅适用于 CMake 3.7 及更高的版本，不要在宏中使用
cmake_parse_arguments(
    PARSE_ARGV startIndex
    prefix
    valuelessKeywords singleValueKeywords multiValueKeywords
)
```

命令的两种形式相似，只是它们以解析的参数集的方式有所不同。对于第一种形式，argsToParse 通常会被给定为 \\${ARGN}，不带引号。这提供了传递给封闭函数或宏的所有参数，除了在大多数情况下不适用的一些特定角落情况（请参阅第8.8节“参数处理的问题”）。

对于第二种形式，PARSE_ARGV 选项告诉 cmake_parse_arguments() 直接从 \\${ARGVx} 变量集中读取参数，其中 x 的范围从 startIndex 到 (ARGC - 1)。因为它直接读取变量，所以不支持在宏内使用。正如在第8.2节“参数处理基础”中已经解释的那样，宏使用字符串替换而不是变量来处理其参数。第二种形式的主要优势在于，对于函数，它可以稳健地处理第一种形式不支持的边缘情况。如果封闭函数没有命名参数，那么将 \\${ARGV} 或 \\${ARGN} 传递给第一种形式等效于在第二种形式中不适用任何边缘情况时给出 PARSE_ARGV 0。

命令的两种形式的其余行为是相同的。每个 ...Keywords 都是在解析期间搜索的关键字名称列表。因为它们是列表，所以需要用引号括起来，以确保它们被正确处理。valuelessKeywords 定义了独立的关键字参数，其行为类似于布尔开关。关键字的存在意味着一种情况，它的缺席意味着另一种情况。singleValueKeywords 在使用时每个关键字后需要精确地一个附加参数，而 multiValueKeywords 在关键字后需要零个或多个附加参数。虽然不是必需的，但流行的约定是关键字通常全部大写，如果需要，用下划线分隔单词。请注意，关键字不应该太长，否则使用起来可能会很麻烦。

当 cmake_parse_arguments() 返回时，可能会定义变量，它们的名称由指定的前缀、下划线和它们关联的关键字的名称组成。例如，对于前缀为 ARG，与名为 FOO 的关键字对应的变量将是 ARG_FOO。对于每个 valuelessKeywords，如果关键字存在，则相应的变量将被定义为 TRUE，如果不存在，则为 FALSE。对于每个 singleValueKeywords 和 multiValueKeywords，只有在关键字存在并且在关键字后提供了值时，相应的变量才会被定义。


以下示例说明了如何定义和处理三种不同类型的关键字：

```
function(func)
  # 定义支持的关键字集
  set(prefix ARG)
  set(noValues ENABLE_NET COOL_STUFF)
  set(singleValues TARGET)
  set(multiValues SOURCES IMAGES)

  # 处理传递进来的参数
  include(CMakeParseArguments)
  cmake_parse_arguments(
    ${prefix}
    "${noValues}" "${singleValues}" "${multiValues}"
    ${ARGN}
  )

  # 记录每个支持的关键字的详细信息
  message("Option summary:")

  foreach(arg IN LISTS noValues)
    if(${prefix}_${arg})
      message(" ${arg} enabled")
    else()
      message(" ${arg} disabled")
    endif()
  endforeach()

  foreach(arg IN LISTS singleValues multiValues)
    # 单一参数值将打印为字符串
    # 多个参数值将打印为列表
    message(" ${arg} = ${${prefix}_${arg}}")
  endforeach()
endfunction()

# 使用不同组合的关键字参数调用的示例
func(SOURCES foo.cpp bar.cpp TARGET MyApp ENABLE_NET)
func(COOL_STUFF TARGET dummy IMAGES here.png there.png gone.png)
```

相应的输出将如下所示：

```
Option summary:
ENABLE_NET enabled
COOL_STUFF disabled
TARGET = MyApp
SOURCES = foo.cpp;bar.cpp
IMAGES =
Option summary:
ENABLE_NET disabled
COOL_STUFF enabled
TARGET = dummy
SOURCES =
IMAGES = here.png;there.png;gone.png
```

上面示例中的 cmake_parse_arguments() 调用也可以使用第二种形式编写，如下所示：

```
cmake_parse_arguments(
  PARSE_ARGV 0
  ${prefix}
  "${noValues}" "${singleValues}" "${multiValues}"
)
```

可以向命令传递参数，以便有一些未关联任何关键字的剩余参数。cmake_parse_arguments() 命令将所有剩余参数作为列表提供在变量 <prefix>_UNPARSED_ARGUMENTS 中。PARSE_ARGV 形式的一个优势是，如果任何未解析的参数本身是一个列表，则它们的嵌入分号将被转义。这样可以保留参数的原始结构，而其他形式的命令则不会。下面的简化示例更清晰地演示了这一点：



```
function(demoArgs)
  set(noValues "")
  set(singleValues SPECIAL)
  set(multiValues ORDINARY EXTRAS)
  
  cmake_parse_arguments(
    PARSE_ARGV 0
    ARG
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  
  message("Keywords missing values: ${ARG_KEYWORDS_MISSING_VALUES}")
endfunction()

demoArgs(burger fries SPECIAL ORDINARY EXTRAS high low)
```

在demoArgs()函数中，对cmake_parse_arguments()的调用将使用值secretSauce定义变量ARG_SPECIAL。burger、fries和cheese;tomato参数不对应任何已识别的关键字，因此它们被视为剩余参数。如上面的输出所示，原始的cheese;tomato列表得到保留，因为使用了PARSE_ARGV形式。这一重要点将在第8.8.2节“转发命令参数”中重新讨论。

在上面的示例中，SPECIAL关键字期望其后跟一个单一参数。如果调用省略了值，cmake_parse_arguments()将不会引发错误。对于CMake 3.14或更早的版本，项目将无法检测到这种情况，但在更新的版本中可以。对于CMake 3.15或更高版本，<prefix>_KEYWORDS_MISSING_VALUES变量将被填充为一个列表，其中包含所有已出现但未跟随任何值的单一或多值关键字。这可以通过修改前面的示例来演示：


```
function(demoArgs)
  set(noValues "")
  set(singleValues SPECIAL)
  set(multiValues ORDINARY EXTRAS)
  
  cmake_parse_arguments(
    PARSE_ARGV 0
    ARG
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  
  message("Keywords missing values: ${ARG_KEYWORDS_MISSING_VALUES}")
endfunction()


demoArgs(burger fries SPECIAL ORDINARY EXTRAS high low)
```

在上述示例中，SPECIAL和ORDINARY后面都紧跟另一个关键字，因此它们与之相关联的值为空。两者都可以或应该有值，因此它们都将出现在cmake_parse_arguments()填充的ARG_KEYWORDS_MISSING_VALUES变量中。对于SPECIAL来说，这可能是一个错误，但对于ORDINARY来说，它可能仍然是有效的，因为多值关键字可以合法地没有值。因此，项目在使用<prefix>_KEYWORDS_MISSING_VALUES时应当小心。

cmake_parse_arguments()命令提供了相当大的灵活性。虽然命令的第一种形式通常将${ARGN}作为要解析的参数集，但也可以提供其他参数。可以利用这一点来执行多级参数解析，例如：

```
function(libWithTest)
  # 第一级参数
  set(groups LIB TEST)
  cmake_parse_arguments(GRP "" "" "${groups}" ${ARGN})
  # 结果为：GRP_LIB = 
  # TARGET Algo
  # SOURCES algo.cpp algo.h
  # PUBLIC_LIBS SomeMathHelpers
  
  # GRP_TEST =
  #   TARGET AlgoTest
  #   SOURCES algoTest.cpp
  #   PRIVATE_LIBS gtest_main

  # 第二级参数
  set(args SOURCES PRIVATE_LIBS PUBLIC_LIBS)
  cmake_parse_arguments(LIB "" "TARGET" "${args}" ${GRP_LIB})
  # 结果为：
  # LIB_TARGET = Algo
  # LIB_SOURCES = algo.cpp algo.h
  # LIB_PUBLIC_LIBS = SomeMathHelpers
  # LIB_PRIVATE_LIBS = 

  cmake_parse_arguments(TEST "" "TARGET" "${args}" ${GRP_TEST})
  # 结果为：
  # TEST_TARGET = AlgoTest
  # TEST_SOURCES = algoTest.cpp
  # TEST_PUBLIC_LIBS =
  # TEST_PRIVATE_LIBS = gtest_main

  add_library(${LIB_TARGET} ${LIB_SOURCES})
  # add_library(Algo algo.cpp algo.h)

  target_link_libraries(${LIB_TARGET}
    PUBLIC ${LIB_PUBLIC_LIBS}
    PRIVATE ${LIB_PRIVATE_LIBS}
  )
  # target_link_libraries(Algo
  #  PUBLIC SomeMathHelpers
  #  PRIVATE 
  #)
  
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  #add_executable(AlgoTest algoTest.cpp)

  target_link_libraries(${TEST_TARGET}
    PUBLIC ${TEST_PUBLIC_LIBS}
    PRIVATE ${LIB_TARGET} ${TEST_PRIVATE_LIBS}
  )
  # target_link_libraries(AlgoTest
  #   PUBLIC
  #   PRIVATE Algo gtest_main
  #)
endfunction()

libWithTest(
  LIB
  TARGET Algo
  SOURCES algo.cpp algo.h
  PUBLIC_LIBS SomeMathHelpers
  
  TEST
  TARGET AlgoTest
  SOURCES algoTest.cpp
  PRIVATE_LIBS gtest_main
)
```

在上面的示例中，由cmake_parse_arguments()解析的第一级参数是通常的\\${ARGN}。在这个第一级中，唯一的关键字是两个多值关键字LIB和TEST。它们定义了接下来的子选项应该应用于哪个目标。第二级解析的输入是\\${GRP_LIB}或\\${GRP_TEST}，而不是\\${ARGN}。由于在原始的ARGN参数集中子选项可能出现多次，因此不存在冲突，因为每个目标的子选项都是分开解析的。

与使用命名参数或使用ARG...变量的基本参数处理相比，cmake_parse_arguments()的优势有很多：

由于是基于关键字的，调用方的可读性得到了改善，因为参数本质上成为自说明的。通常情况下，阅读调用方的其他开发人员通常不需要查看函数实现或其文档，就能理解每个参数的含义。

* 调用方可以选择给出参数的顺序。
* 调用方可以简单地省略不需要提供的参数。
* 由于每个支持的关键字都必须传递给cmake_parse_arguments()，并且它通常在函数的顶部附近调用，因此通常非常清楚函数支持哪些参数。
* 由于关键字基于参数的解析由cmake_parse_arguments()命令处理而不是通过临时手动编写的解析器，几乎可以消除参数解析错误。
* 尽管这些功能非常强大，但该命令仍然有一些限制。内置命令能够支持关键字的重复使用。例如，像target_link_libraries()这样的命令允许在同一命令中多次使用PRIVATE、PUBLIC和INTERFACE关键字。cmake_parse_arguments()命令不同程度上不支持这一点。它只会返回与关键字的最后一次出现关联的值，并且会丢弃先前的值。只有在使用多级关键字集且在正在处理的参数集中关键字只出现一次的情况下，关键字才能重复。


## 8.4. 返回值

函数和宏之间的一个基本区别在于，函数引入了新的变量作用域，而宏则没有。函数接收来自调用作用域的所有变量的副本。在函数内部定义或修改的变量对函数外部同名变量没有影响（除非明确传递，如下文所述）。就变量而言，函数本质上是一个独立的沙盒，类似于由block()命令创建的作用域（参见第5.4节，“作用域块”）。另一方面，宏共享与其调用者相同的变量作用域，因此可以直接修改调用者的变量。请注意，函数不引入新的策略作用域（有关此内容的更多讨论，请参见第12.3节，“推荐实践”）。

### 8.4.1. 从函数返回值

在CMake 3.25或更高版本中，函数可以通过指定要传播到调用者的变量来有效地返回值。这是通过在return()命令中使用PROPAGATE关键字实现的，类似于先前在第7.4节“提前结束处理”中描述的行为。对于PROPAGATE之后列出的每个变量名，该变量将在调用范围内更新，以具有return()调用时函数中相同的值。如果在函数作用域中未设置传播的变量，则在调用作用域中也将不设置该变量。如果使用了PROPAGATE关键字，则在定义函数时必须将CMP0140策略设置为NEW（第12章“策略”深入讨论了策略）。

```
# 确保我们有一个支持 PROPAGATE 的 CMake 版本，且 CMP0140 策略被设置为 NEW
cmake_minimum_required(VERSION 3.25)

function(doSomething outVar)
    set(${outVar} 42) # 注意这里要set ${outVar}而不是outVar。
    return(PROPAGATE ${outVar})
endfunction()

doSomething(result)
# 此时，一个名为 result 的变量现在保存着值 42
```
通常情况下，函数不应规定在调用作用域中要设置的变量的名称。相反，函数参数应该用于告诉函数在调用作用域中设置哪些变量的名称。这确保了调用方完全控制函数的操作，并且函数不会覆盖调用者不希望被覆盖的变量。CMake自己的内置命令通常遵循此模式。上述示例通过允许调用方将结果变量的名称指定为函数的第一个参数，符合此建议。

return()语句将变量传播到调用作用域。这意味着函数内的任何block()语句都不会阻止传播到函数的调用者，但它们将影响传播的变量（们）的值。可以稍微修改上述示例来演示这一点：

```
cmake_minimum_required(VERSION 3.25)

function(doSomething outVar)
    set(${outVar} 42)
    
    block()
        set(${outVar} 27)
        return(PROPAGATE ${outVar})
    endblock()

    return(PROPAGATE ${outVar})
endfunction()

doSomething(result)
# 此时，一个名为 result 的变量现在保存着值 27
```

在CMake 3.24及更早版本中，函数不支持直接返回值。由于函数引入了它们自己的变量作用域，似乎没有简单的方法将信息传递回调用方，但实际并非如此。正如在前文第5.4节“作用域块”和第7.1.2节“作用域”中讨论的，set() 和 unset() 命令支持 PARENT_SCOPE 关键字，该关键字可用于修改调用者作用域中的变量，而不是函数内的局部变量。虽然这不同于从函数返回值，但它允许将值传递回调用作用域以实现类似的效果。

```
function(func resultVar1 resultVar2)
    set(${resultVar1} "First result" PARENT_SCOPE)
    set(${resultVar2} "Second result" PARENT_SCOPE)
endfunction()

func(myVar otherVar)
message("myVar: ${myVar}")
message("otherVar: ${otherVar}")
# 输出:
# myVar: First result
# otherVar: Second result
```

### 8.4.2. 从宏中返回值

宏可以以与函数相同的方式“返回”特定变量，通过将它们作为参数传递来指定要设置的变量的名称。唯一的区别是，在宏内部调用 set() 时不应使用 PARENT_SCOPE 关键字，因为宏已经修改了调用者作用域中的变量。实际上，如果需要在调用作用域中设置许多变量，人们可能会选择使用宏而不是函数。宏将在每次 set() 或 unset() 调用时影响调用作用域，而函数只有在明确给出 PARENT_SCOPE 给 set() 或 unset() 时才会影响调用作用域。

前一节的最后一个例子可以等效地实现为如下的宏：

```
macro(func resultVar1 resultVar2)
    set(${resultVar1} "First result")
    set(${resultVar2} "Second result")
endmacro()
```

return() 在宏中的行为与函数非常不同。由于宏不引入新的作用域，return() 语句的行为取决于调用宏的位置。请回忆宏实际上是将其命令粘贴到调用点。因此，宏中的任何 return() 语句实际上将从调用宏的任何地方的作用域返回，而不是从宏本身返回。考虑以下例子：

```
macro(inner)
    message("From inner")
    return() # 在宏中通常是危险的
    message("Never printed")
endmacro()

function(outer)
    message("From outer before calling inner")
    inner()
    message("Also never printed")
endfunction()

outer()
```


上述例子的输出将是：

```
From outer before calling inner
From inner
```

下面的代码从外部调用内部来突显函数体中的第二条消息为何未被打印，将宏体的内容粘贴到调用处：

```
function(outer)
    message("From outer before calling inner")

    # === Pasted macro body ===
    message("From inner")
    return()
    message("Never printed")
    # === End of macro body ===

    message("Also never printed")
endfunction()
```

现在更清楚为什么 return() 语句导致处理离开函数，即使它最初是从宏内部调用的。这突显了在宏中使用 return() 的危险性。因为宏不创建自己的作用域，return() 语句的结果通常与预期的不同。





## 8.5. 覆盖命令

当调用 function() 或 macro() 定义一个新命令时，如果已经存在同名的命令，CMake 的未记录的行为是使用相同的名称，只是在前面加一个下划线，使旧命令仍然可用。这适用于旧名称是内建命令或自定义函数或宏的情况。有时了解这种行为的开发者会尝试利用它来尝试创建对现有命令的包装，如下所示：

```
function(someFunc)
    # Do something...
endfunction()

# 项目的后续部分...
function(someFunc)
    if(...)
        # 覆盖行为，使用其他内容...
    else
        # 警告：预期调用原始命令，但这是不安全的
        _someFunc()
    endif()
endfunction()
```

如果命令只是像这样被覆盖一次，似乎是有效的，但如果再次被覆盖，那么原始命令将不再可用。添加一个下划线来“保存”之前的命令只适用于当前名称，不会递归应用于所有先前的覆盖。这有可能导致无限递归，如以下刻意构造的示例所示：

```
function(printme)
    message("Hello from first")
endfunction()

function(printme)
    message("Hello from second")
    _printme()
endfunction()

function(printme)
    message("Hello from third")
    _printme()
endfunction()

printme()
```

天真地期望输出如下所示：

```
Hello from third
Hello from second
Hello from first
```
但实际上，永远不会调用第一个实现，因为第二个实现最终调用自身，导致无限循环。当 CMake 处理上述情况时，发生了以下情况：

1. 创建第一个 printme 实现，并将其作为该名称的命令提供。由于此名称之前不存在任何命令，因此不需要进一步操作。
2. 遇到第二个 printme 实现。CMake 找到同名的现有命令，因此定义 _printme 来指向旧命令，并将 printme 设置为指向新定义。
3. 遇到第三个 printme 实现。同样，CMake 找到同名的现有命令，因此重新定义 _printme 来指向旧命令（即第二个实现），并将 printme 设置为指向新定义。
当调用 printme() 时，执行进入第三个实现，该实现调用 _printme()。这进入第二个实现，它也调用 _printme()，但 _printme() 又指回第二个实现，导致无限递归。执行永远不会达到第一个实现。

一般来说，覆盖函数或宏是可以的，只要它不试图像上述讨论的那样调用先前的实现。项目应该简单地假设新实现替代了旧实现，认为旧实现不再可用。


## 8.6. 函数的特殊变量

CMake 3.17版本增加了对一些变量的支持，以帮助调试和实现函数。在执行函数时，将可用以下变量：

**CMAKE_CURRENT_FUNCTION**：保存当前正在执行的函数的名称。
**CMAKE_CURRENT_FUNCTION_LIST_FILE**：包含定义当前正在执行的函数的文件的完整路径。
**CMAKE_CURRENT_FUNCTION_LIST_DIR**：保存包含定义当前正在执行的函数的文件的绝对目录。
**CMAKE_CURRENT_FUNCTION_LIST_LINE**：保存当前正在执行的函数在定义它的文件中的行号。

CMAKE_CURRENT_FUNCTION_LIST_DIR变量在函数需要引用作为函数的内部实现细节的文件时特别有用。CMAKE_CURRENT_LIST_DIR的值将包含调用函数的文件的目录，而CMAKE_CURRENT_FUNCTION_LIST_DIR保存了定义函数的目录。为了了解如何使用它，考虑以下示例。它演示了一个常见模式，其中一个函数使用configure_file()命令从定义函数的文件所在的相同目录复制文件（有关详细讨论，请参见第20.2节“复制文件”）：

```
function(writeSomeFile toWhere)
  configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template.cpp.in ${toWhere} @ONLY)
endfunction()
```

在CMake 3.17之前，上述内容通常会以以下方式实现（CMake的模块在CMake 3.17之前使用了这种技术）：

```
set(__writeSomeFile_DIR ${CMAKE_CURRENT_LIST_DIR})
function(writeSomeFile toWhere)
  configure_file(${__writeSomeFile_DIR}/template.cpp.in ${toWhere} @ONLY)
endfunction()
```

这第二个示例依赖于__writeSomeFile_DIR变量在调用函数时保持可见。通常来说，这应该是一个合理的假设，但由于函数具有全局范围的可见性，项目在一个地方定义函数并在不相关的变量范围调用它是技术上合法的。尽管这在技术上是合法的，但并不是一种推荐的做法。在使用定义函数的文件中使用包含保护时，还必须特别小心处理这种技术（请参见第7.4节“提前结束处理”）。

CMAKE_CURRENT_FUNCTION...变量仅在函数中更新，不在宏内修改。在执行宏的代码时，这些变量将保持它们在调用宏时的任何值。


## 8.7. 调用 CMake 代码的其他方式

函数和宏提供了在稍后执行代码的强大方法。它们是重用相似或重复任务的常见逻辑的重要组成部分。然而，有些情况下，项目可能希望以函数和宏单独无法捕获的方式定义要执行的 CMake 代码。

CMake 3.18 添加了 cmake_language() 命令，可用于直接调用任意 CMake 代码，而无需定义函数或宏。此功能并不旨在取代函数或宏，而是通过启用更简洁的代码和以前无法实现的逻辑表达方式来补充它们。CMake 3.18 提供的两个子命令是 CALL 和 EVAL CODE：

```
cmake_language(CALL command [args...])
cmake_language(EVAL CODE code...)
```

CALL 子命令调用单个 CMake 命令，如果需要还可带有参数。它提供了在不必硬编码所有可用选择的情况下参数化要调用的命令的能力。某些内置命令不能以这种方式调用，特别是那些开始或结束块的命令，如 if()、endif()、foreach()、endforeach() 等。

以下示例演示了如何在一组函数周围定义通用包装器，其中包含它们名称中的版本号：

```
function(qt_generate_moc)
  set(cmd qt${QT_DEFAULT_MAJOR_VERSION}_generate_moc)
  cmake_language(CALL ${cmd} ${ARGV})
endfunction()
```

上述示例假设 QT_DEFAULT_MAJOR_VERSION 变量已经在之前设置。随着未来 Qt 主要版本的发布，上述代码将继续工作，只要适当的带版本的命令仍然提供。另一种方法是为每个版本单独实现一个不断扩展的 if() 测试集。

CALL 子命令在其实用性上相对有限。EVAL CODE 子命令更为强大，因为它支持执行任何有效的 CMake 脚本。其中一个优点是它不会干扰在函数调用内部更新的变量，例如 ARGV、CMAKE_CURRENT_FUNCTION 等。以下示例利用了这种行为来实现函数的调用跟踪形式：

```
set(myProjTraceCall [=[
  message("Called ${CMAKE_CURRENT_FUNCTION}")
  set(__x 0)
  while(__x LESS ${ARGC})
    message(" ARGV${__x} = ${ARGV${__x}}")
    math(EXPR __x "${__x} + 1")
  endwhile()
  unset(__x)
]=])

function(func)
  cmake_language(EVAL CODE "${myProjTraceCall}")
  # ...
endfunction()

func(one two three)
```

注意 myProjTraceCall 中存储的代码如何利用了各种 ARG* 变量和 CMAKE_CURRENT_FUNCTION 变量。括号语法 [=[ 和 ]=] 用于防止在设置 myProjTraceCall 时对这些变量进行评估。这些变量仅在调用 cmake_language() 时评估，因此它们将反映封闭函数的详细信息。由于这种延迟评估，跟踪代码在宏内部不会按预期工作，因此只能在函数内部使用。
有关 EVAL CODE 子命令的另一个特别有趣的示例，请参见第8.8.2节“转发命令参数”。


CMake 3.19引入了DEFER子命令集。这些子命令允许将命令排队以在以后的某个时间执行，并管理当前排队命令的集合。创建延迟命令可通过以下形式完成：

```
cmake_language(DEFER
  [DIRECTORY dir]
  [ID id | ID_VAR outVar]
  CALL command [args...]
)
```

command和其参数将被排队在当前目录范围结束时执行。DIRECTORY选项可用于指定不同的目录范围。在这种情况下，dir目录必须已被CMake处理过，并且不能已经完成处理。在实践中，这意味着它必须是当前目录或父目录范围之一。

```
cmake_language(DEFER
  CALL message "当前范围处理结束"
)
cmake_language(DEFER
  DIRECTORY ${CMAKE_SOURCE_DIR}
  CALL message "顶层处理结束"
)
```

每个排队的命令都与一个标识符相关联。可以将多个命令与相同的标识符关联，以便它们可以作为一组进行操作（见下文）。通常，项目会让CMake在排队新的延迟命令时自动分配一个新的标识符。可以使用ID_VAR选项来捕获分配的标识符，然后在后续调用中使用ID选项将更多命令添加到相同的标识符中。

```
cmake_language(DEFER
  ID_VAR deferredId
  CALL message "第一个延迟命令"
)
cmake_language(DEFER
  ID ${deferredId}
  CALL message "第二个延迟命令"
)
```
其他DEFER子命令可以基于标识符查询和取消排队的命令：

```
cmake_language(DEFER [DIRECTORY dir] GET_CALL_IDS outVar)
cmake_language(DEFER [DIRECTORY dir] GET_CALL id outVar)
cmake_language(DEFER [DIRECTORY dir] CANCEL_CALL ids...)
```

GET_CALL_IDS形式返回当前为指定目录范围排队的所有命令的标识符列表，如果未给出DIRECTORY选项，则为当前目录范围。GET_CALL形式返回与指定id关联的第一个命令及其参数。无法检索与给定标识符相关的第二个或更后续命令，也无法获取与标识符关联的命令数量。CANCEL_CALL形式将丢弃与任何指定标识符关联的所有延迟命令。


在这一点上，自然会开始考虑如何运用DEFER功能的不同方式。在这样做之前，请考虑以下观察：

* 在延迟命令及其参数中进行变量扩展时，会应用特殊规则（详见第8.8.3节“参数扩展的特殊情况”）。这可能导致难以追踪的微妙问题。

* 延迟命令使开发人员更难以跟踪执行流程。特别是在函数或宏内部创建延迟命令并且其创建不明显时，情况更为复杂。

* 在推迟命令时，项目可能对在推迟和执行命令之间可能发生的情况做出了一些假设。保证这些假设保持有效可能相当困难，特别是对于推迟到父范围或在可能从任何地方调用的函数或宏内创建延迟命令的情况。
* 延迟命令可能是项目的CMake API尝试在一个函数或宏中做太多事情的迹象。

鉴于以上情况，在有选择的情况下，最好选择其他技术或重构而非推迟命令。例如，一个函数可能会包装一个创建目标的命令，然后调用其他使用该目标属性的命令，最后返回（属性将在下一章中介绍）。通过将所有这些内容封装在一个函数中，调用者没有机会在使用之前修改目标属性。与其推迟使用目标的命令，以便调用者可以修改目标属性，不如考虑将函数分解为责任更少的部分。对于这个特定的例子，要求传递目标而不是创建它可能是另一种替代解决方案。




## 8.8. 参数处理的问题

CMake对命令参数的实现包含一些微妙的行为。在大多数情况下，这些行为不会导致问题，但偶尔它们可能引起混淆或产生意外结果。为了理解为什么存在这些行为以及如何安全处理它们，有助于了解CMake是如何构造和传递参数给命令的。
考虑以下等效的调用，其中someCommand可以是任何有效的命令：

```
someCommand(a b c)
someCommand(a	b	c)
```

参数由空格分隔，连续的空格被视为单个参数分隔符。分号也充当参数分隔符，因此以下调用也等效于上述调用：

```
someCommand(a b;c)
someCommand(a;;;;b;c)
```

当一个参数需要包含嵌套的空格或分号时，必须使用引号：

```
someCommand(a "b b" c)
someCommand(a "b;b" c)
someCommand(a;"b;b";c)
```

上述三个调用都会将三个参数传递给命令。第一个调用将b b作为第二个参数传递，其他两个调用将b;b作为第二个参数传递。
在涉及变量评估且未加引号参数时，空格和分号之间的处理方式有所不同：

```
set(containsSpace "b b")
set(containsSemiColon "b;b")
someCommand(a ${containsSpace} c)
someCommand(a ${containsSemiColon} c)
```

第一次调用someCommand()导致传递三个参数，而第二次调用导致传递四个参数。containsSpace中的嵌套空格不起到参数分隔符的作用，但containsSemiColon中的嵌套分号起到了这样的作用。空格只在执行任何变量评估之前充当参数分隔符。这两种不同行为之间的交互可能导致一些令人惊讶的结果：

```
set(empty "")
set(space " ")
set(semicolon ";")
set(semiSpace "; ")
set(spaceSemi " ;")
set(spaceSemiSpace " ; ")
set(spaceSemiSemi " ;;")
set(semiSemiSpace ";; ")
set(spaceSemiSemiSpace " ;; ")
someCommand(${empty}) # 第1次调用：0个参数
someCommand(${space}) # 第2次调用：1个参数
someCommand(${semicolon}) # 第3次调用：0个参数
someCommand(${semiSpace}) # 第4次调用：1个参数
someCommand(${spaceSemi}) # 第5次调用：1个参数
someCommand(${spaceSemiSpace}) # 第6次调用：2个参数
someCommand(${spaceSemiSemi}) # 第7次调用：1个参数
someCommand(${semiSemiSpace}) # 第8次调用：1个参数
someCommand(${spaceSemiSemiSpace}) # 第9次调用：2个参数
```

从上面可以得出一些重要观察：

**观察1**
在它们是变量评估的结果时，空格永远不会被丢弃，也永远不会充当参数分隔符。

**观察2**
在没有引号的参数开头或结尾的一个或多个分号将被丢弃。

**观察3**
没有引号的参数中不在开头或结尾的连续分号会被合并并充当单个参数分隔符。

通过在包含任何变量评估的情况下引用参数，可以避免许多混淆。这消除了对嵌套空格或分号的任何特殊解释。尽管这通常不会有害，但这并不总是理想的。正如下一小节将强调的那样，有些情况需要精确地将参数取消引用，因为它们依赖于上述行为。

【译注：空格是参数，分号起分格作用。知道这个原则上面的例子很容易解释。】

### 8.8.1. 鲁棒解析参数

考虑之前在第8.3节“关键字参数”中讨论的cmake_parse_arguments()命令。该命令的原始形式通常用于如下方式：

```
function(func)
  set(noValues ENABLE_A ENABLE_B)
  set(singleValues FORMAT ARCH)
  set(multiValues SOURCES IMAGES)
  
  cmake_parse_arguments(
    ARG
    "${noValues}" "${singleValues}" "${multiValues}"
    ${ARGV}
  )
endfunction()
```

请注意，在noValues、singleValues和multiValues的使用时加了引号。当评估时，这些变量的每个都产生一个包含分号的字符串。例如，\\${singleValues}将评估为FORMAT;ARCH。引号是必需的，以防止这些分号充当参数分隔符。最终结果是cmake_parse_arguments()将看到ARG作为其第一个参数，ENABLE_A;ENABLE_B作为第二个参数，FORMAT;ARCH作为第三个参数，SOURCES;IMAGES作为第四个参数。

在调用的末尾提供的\\${ARGV}没有引号。这是有意的，以利用嵌套分号将充当参数分隔符的事实。cmake_parse_arguments()命令将其接收到的第五个及后续的参数解释为要解析的参数。通过使用未加引号的\\${ARGV}，cmake_parse_arguments()看到的是与传递给func()的相同一组参数。

上述方法的问题在于使用\\${ARGV}无法在两种情况下保留原始参数。考虑以下调用：

```
func(a "" c)
func("a;b;c" "1;2;3")
```

对于第一个调用，在func()内部\\${ARGV}的评估将是a;;c。然而，如观察3所述，两个分号将被合并，cmake_parse_arguments()将只看到a和c作为要解析的参数。空参数会被静默丢弃。对于第二个调用，${ARGV}的评估将是a;b;c;1;2;3。func()的原始调用具有a;b;c作为第一个参数和1;2;3作为第二个参数，但这通过\\${ARGV}的评估被展平，cmake_parse_arguments()命令看到的是六个独立的参数而不是两个列表。这两个问题都可以通过使用cmake_parse_arguments()命令的另一种形式来避免直接评估\\${ARGV}来解决：

```
cmake_parse_arguments(
  PARSE_ARGV 0 ARG
  "${noValues}" "${singleValues}" "${multiValues}"
)
```


在实际应用中，cmake_parse_arguments()命令通常在丢弃空参数或展平列表没有实际影响的情况下使用。在这些情况下，cmake_parse_arguments()命令的任何形式都可以安全地调用。然而，在需要精确保留参数的情况下，始终应使用PARSE_ARGV形式。


### 8.8.2. 转发命令参数

一个相对常见的需求是创建一个对现有命令进行包装的某种形式。项目可能希望支持一些额外的选项或删除现有的选项，或者可能希望在调用之前或之后执行某些处理。保留参数并在不改变其结构或丢失信息的情况下进行转发可能会令人惊讶地困难。

考虑以下示例及其输出，其中涉及前一小节的一个观点：

```
function(printArgs)
  message("ARGC = ${ARGC}\n"
          "ARGN = ${ARGN}"
  )
endfunction()

printArgs("a;b;c" "d;e;f")
```
输出：

```
ARGC = 2
ARGN = a;b;c;d;e;f
```
对于printArgs()的参数是带引号的，因此该函数只看到两个参数。但在形成${ARGN}的值时，这两个列表使用分号连接，结果是一个包含六个项目的单个列表。由于此列表展平，原始参数的原始形式丢失了。考虑一下对于尝试将参数转发到被包装的命令的包装命令的影响：

```
function(inner)
  message("inner:\n"
          "ARGC = ${ARGC}\n"
          "ARGN = ${ARGN}"
  )
endfunction()

function(outer)
  message("outer:\n"
          "ARGC = ${ARGC}\n"
          "ARGN = ${ARGN}"
  )
  inner(${ARGN}) # 不健壮的朴素转发
endfunction()

outer("a;b;c" "d;e;f")
```
输出：

```
outer:
ARGC = 2
ARGN = a;b;c;d;e;f
inner:
ARGC = 6
ARGN = a;b;c;d;e;f
```

outer()函数想要包装inner()函数并精确地转发其参数，但如上述输出所示，被inner()看到的参数数量是不同的。\\${ARGN}的评估作为将参数传递给inner()的方法触发了先前描述的列表展平行为。原始参数的结构丢失了。可以使用cmake_parse_arguments()命令的PARSE_ARGV形式来避免此列表展平：

```
function(outer)
  cmake_parse_arguments(PARSE_ARGV 0 FWD "" "" "")
  inner(${FWD_UNPARSED_ARGUMENTS})
endfunction()
```

由于没有要解析的关键字，所有给定给outer()的参数将被放置在FWD_UNPARSED_ARGUMENTS中。正如在第8.3节“关键字参数”中指出的那样，当cmake_parse_arguments()命令的PARSE_ARGV形式填充FWD_UNPARSED_ARGUMENTS时，它会转义原始参数中的任何嵌套分号。因此，当将该变量传递给inner()时，转义保留了原始参数的结构，inner()将看到与outer()相同的参数。

不幸的是，上述技术仍然存在一个弱点。由于观察2和3的影响，它不会保留任何空参数。为了避免丢弃空参数，每个参数都需要逐个列出并加引号。在CMake 3.18或更高版本中，使用cmake_language(EVAL CODE)命令提供了所需的功能：

```
function(outer)
  cmake_parse_arguments(PARSE_ARGV 0 FWD "" "" "")
 
  set(quotedArgs "")
  foreach(arg IN LISTS FWD_UNPARSED_ARGUMENTS)
    string(APPEND quotedArgs " [===[${arg}]===]")
  endforeach()
 
  cmake_language(EVAL CODE "inner(${quotedArgs})")
endfunction()
```

请注意使用括号形式进行引用。这确保了任何包含嵌套引号的参数也将被强健地处理。

上述实现提供了强健的参数转发，但需要CMake版本为3.18或更高。对于较早的版本，cmake_language()命令不可用。可以通过将要执行的命令写入文件并请求CMake通过调用include()处理该文件来实现等效能力，但这非常低效，不推荐作为一般解决方案。

上述技术仅适用于函数。cmake_parse_arguments()命令的PARSE_ARGV形式不能用于宏，这意味着无法避免列表展平。然而，如果列表展平不是一个问题，至少可以保留空字符串。以下实现演示了在假设没有值需要包含分号的情况下实现这一点的一种方式：

```
# 警告：此示例不保留列表结构。
# 它确实保留空字符串参数。
macro(outer)
  string(REPLACE ";" "]===] [===[" args "[===[${ARGV}]===]")
  cmake_language(EVAL CODE "inner(${args})")
endmacro()
```

有关上述技术可能需要的情景的示例，请参见第32.2.6节“委派提供程序”。


### 8.8.3. 参数扩展的特殊情况

尽管上述技术在一般情况下表现良好，但一些内置命令以特殊方式处理它们的参数，导致它们与预期的行为有所偏差。这些异常可分为两个主要类别：cmake_language() 和布尔表达式的求值。

**用于cmake_language()的变量求值**

cmake_language(CALL)提供了执行命令的替代方式。调用的命令可以由变量提供，而无需硬编码。为了忠实地复制对要转发到的命令的参数给出的正常参数扩展和引号处理行为，这些参数需要在调用cmake_language(CALL)时与命令本身分开。以下示例演示了这个限制：

```
# 错误：命令必须是自己的参数
set(cmdWithArgs message STATUS "Hello world")
cmake_language(CALL ${cmdWithArgs})
# 正确：Command可以是变量扩展，但它必须评估为单个值。
# 参数也可以是变量扩展，它们可以评估为列表。
set(cmd message)
set(args STATUS "Hello world")
cmake_language(CALL ${cmd} ${args})
```

cmake_language(DEFER CALL)命令具有类似的限制，但它有进一步的差异。对于要执行的命令，变量评估是立即执行的，但对于命令参数，评估是延迟的。以下示例突出了此行为：

```
set(cmd message)
set(args "before deferral")
cmake_language(DEFER CALL ${cmd} ${args})
set(cmd somethingElse)
# 不影响命令
set(args "after deferral")
# 但这样做！
```
结果为：after deferral

\\${cmd}的评估立即发生，但\\${args}直到调用延迟的命令时才会评估。在目录范围的末尾，args将具有“after deferral”的值。如果需要立即执行命令参数中的变量评估，则必须将延迟包装在cmake_language(EVAL CODE)调用内。这种情况的一个示例是在函数或宏内部创建延迟，并且延迟的命令参数需要合并作为传递给延迟函数或宏的参数的信息：

```
function(endOfScopeMessage msg)
  cmake_language(EVAL CODE "cmake_language(DEFER CALL message [[${msg}]])")
endfunction()
```
请注意，在\\${msg}评估周围使用括号语法进行引用，以确保正确处理变量中的空格。

**布尔表达式**

将参数视为布尔表达式的命令在引号和参数扩展方面也具有一些特殊规则。if()命令最好演示了这一点，但这些规则也适用于while()。考虑以下示例，它展示了未加引号的参数如何被视为变量名或字符串值的微妙行为（请参见第6.1.1节“基本表达式”以深入讨论此行为）：

```
cmake_minimum_required(VERSION 3.1)
set(someVar xxxx)
set(xxxx "some other value")
# 在这里，xxxx是不加引号的的，因此它被视为变量名，并使用其值。结果：打印NO
if(someVar STREQUAL xxxx)
  message(YES)
else()
  message(NO)
endif()
# 现在使用带引号的xxxx。这样可以防止其被视为变量名，而直接使用其值。结果：打印YES
if(someVar STREQUAL "xxxx")
  message(YES)
else()
  message(NO)
endif()
```

尝试使用变量提供参数给if()命令来复制上述情况，可能会尝试类似以下方式：

```
set(noQuotes someVar STREQUAL xxxx)
set(withQuotes someVar STREQUAL [["xxxx"]])
if(${noQuotes})
  message(YES)
else()
  message(NO)
endif()
if(${withQuotes}) # 不符合预期
  message(YES)
else()
  message(NO)
endif()
```

withQuotes的值使用括号语法使引号成为存储值的一部分。这个想法是尝试让if()命令将xxxx视为带引号的参数，但它没有产生期望的效果。if()命令在展开之前检查引号，因此在这种情况下，引号被视为参数值的一部分。在通过类似上述的扩展变量评估提供参数给if()时，没有办法使参数被视为带引号。

需要注意的另一种特殊情况是方括号如何影响在列表中解释分号的方式，如前述第5.9.1节“不平衡方括号的问题”中讨论的。在评估变量时，方括号之间的不平衡分号不被视为列表分隔符。然而，这并不扩展到CMake组装命令参数的方式，如以下示例及其输出所示：

```
function(func)
  message("参数数量：${ARGC}")
  math(EXPR lastIndex "${ARGC} - 1")
  foreach(n RANGE 0 ${lastIndex})
    message("ARGV${n} = ${ARGV${n}}")
  endforeach()
  foreach(arg IN LISTS ARGV)
    message("${arg}")
  endforeach()
endfunction()
func("a[a" "b]b" "c[c]c" "d[d" "eee")
```

func()命令确实看到了五个原始参数，如第一个foreach()循环所示。由第二个foreach()命令的ARGV变量的评估引起的干扰，嵌套的不平衡的方括号影响了变量被解释为列表的方式。

实际上，很少有可能出现方括号不平衡的情况。偶尔可能会遇到平衡的方括号，但正如上面的示例中的c[c]c参数所示，它们不会干扰列表解释。


## 8.9. 最佳实践

函数和宏是在整个项目中重复使用相同的 CMake 代码的好方法。一般来说，最好使用函数而不是宏，因为函数内部使用新的变量范围更好地隔离了该函数对调用范围的影响。宏应该通常只在宏体的内容确实需要在调用方的范围内执行时使用。这些情况通常应该相对较少。为了避免意外行为，还要避免从宏内部调用 return()。

最好通过命令参数传递函数或宏需要的所有值，而不是依赖于在调用范围内设置的变量。这往往使实现对未来的变化更具鲁棒性，而且更清晰、更易于维护。

对于除了非常琐碎的函数或宏之外的所有情况，强烈建议使用由 cmake_parse_arguments() 提供的基于关键字的参数处理。这会提高可用性并改善调用代码的鲁棒性（例如，很少有机会混淆参数）。它还允许更容易地在未来扩展函数，因为不依赖于参数的顺序或总是提供所有参数的要求，即使不相关。

在解析或转发命令参数时要注意丢弃空参数和列表平坦化。在函数内部，如果项目的最低 CMake 版本允许，最好使用 cmake_parse_arguments() 的 PARSE_ARGV 形式。在转发参数时，如果需要保留空参数和列表，则使用 cmake_language(EVAL CODE) 逐个引用每个参数。

最好避免通过 cmake_language(DEFER) 延迟命令，如果有其他替代方案的话。延迟的命令引入了脆弱性，阻碍了调试项目的能力，可能表明需要重构 CMake 函数和宏。

与其将函数和宏分布在源树中，一个常见的做法是指定一个特定的目录（通常在项目的顶层下方）用于收集各种 XXX.cmake 文件。该目录充当了一个方便从项目的任何位置访问的现成功能的目录。每个文件都可以提供适当的函数、宏、变量和其他功能。使用 .cmake 文件名后缀允许 include() 命令将这些文件视为模块，这是第11章“模块”中详细讨论的主题。它还倾向于允许 IDE 工具识别文件类型并应用 CMake 语法高亮。

不要定义或调用以单个下划线开头的函数或宏的名称。特别是，不要依赖于这样的名称，其中当函数或宏重新定义现有命令时，该命令的旧实现可用。一旦命令被多次覆盖，其原始实现就不再可访问。这种未记录的行为甚至可能在未来的 CMake 版本中被删除，因此不应使用。类似的，不要覆盖任何内置的 CMake 命令。认为这些是禁区，以便项目始终能够假定内置命令按照官方文档的规定行为，不会存在原始命令不可访问的机会。

如果项目的最低 CMake 版本设置为3.17或更高，请优先使用 CMAKE_CURRENT_FUNCTION_LIST_DIR 来引用在相对于定义函数的文件的位置预期存在的任何文件或目录。


















