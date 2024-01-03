---
layout:     post
title:      "第七章：使用子目录"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - cmake 
---



 <!--more-->

将所有内容保存在一个目录中对于简单的项目来说是可以的，但是大多数真实世界的项目会将它们的文件分布在多个目录中。通常会发现不同的文件类型或单独的模块被分组到它们自己的目录下，或者属于逻辑功能组的文件被放置在项目目录层次结构的不同部分。虽然目录结构可能是由开发人员对项目的思考方式驱动的，但项目的结构方式也会影响构建系统。在任何多目录项目中，两个基本的CMake命令是add_subdirectory()和include()。这些命令将来自另一个文件或目录的内容引入构建，允许构建逻辑分布在目录层次结构中，而不是强制将所有内容定义在最顶层。这带来了许多优势：

* 构建逻辑是本地化的，这意味着构建的特性可以在它们最相关的目录中定义。
* 构建可以由独立于顶层项目的子组件组成。这在项目使用git子模块或嵌入第三方源代码树时尤为重要。
* 由于目录可以是自包含的，通过选择是否将该目录添加到构建中，轻松地打开或关闭构建的某些部分变得相对简单。
* add_subdirectory()和include()具有相当不同的特性，因此了解两者的优势和劣势是很重要的。

## 7.1. add_subdirectory()

add_subdirectory()命令允许项目将另一个目录引入构建。该目录必须有自己的CMakeLists.txt文件，在调用add_subdirectory()时将对其进行处理。在项目的构建树中将创建一个相应的目录。

```
add_subdirectory(sourceDir [binaryDir]
                 [EXCLUDE_FROM_ALL]
                 [SYSTEM] # 需要CMake 3.25或更高版本
)
```
sourceDir不必是源树中的子目录，尽管通常是这样。可以添加任何目录，其中sourceDir可以被指定为绝对或相对路径，后者相对于当前源目录。绝对路径通常仅在添加位于主源树之外的目录时才需要。通常情况下，不需要指定binaryDir。如果省略，CMake将在构建树中创建一个与sourceDir同名的目录。如果sourceDir包含任何路径组件，这些组件将在由CMake创建的binaryDir中得以镜像。或者，可以显式指定binaryDir，它可以是绝对或相对路径，后者相对于当前二进制目录进行评估（稍后将详细讨论）。如果sourceDir是源树之外的路径，CMake要求必须指定binaryDir，因为不能自动构建相应的相对路径。

可选的EXCLUDE_FROM_ALL关键字用于控制是否应默认将在添加的子目录中定义的目标包含在项目的ALL目标中。不幸的是，对于某些CMake版本和项目生成器，它并不总是按预期方式起作用，甚至可能导致构建失败。SYSTEM关键字通常不会由项目直接使用，并在第15.7.2节“系统头搜索路径”中进行了讨论。

### 7.1.1. 源和二进制(目的)目录变量


有时，开发人员需要知道与当前源目录对应的构建目录的位置，例如在运行时复制所需的文件或执行自定义构建任务时。通过add_subdirectory()，源树和构建树的目录结构可以是任意复杂的。甚至可以在相同的源树中使用多个构建树。因此，开发人员需要CMake的一些帮助来确定感兴趣的目录。

为此，CMake提供了一些变量，用于跟踪当前正在被CMake处理的CMakeLists.txt文件的源目录和二进制目录。以下是每个文件由CMake处理时自动更新的只读变量。它们始终包含绝对路径。

**CMAKE_SOURCE_DIR**：源树的最顶层目录（即最顶层的CMakeLists.txt文件所在的位置）。此变量永远不会更改其值。

**CMAKE_BINARY_DIR**：构建树的最顶层目录。此变量永远不会更改其值。

**CMAKE_CURRENT_SOURCE_DIR**：当前由CMake处理的CMakeLists.txt文件所在的目录。每次通过add_subdirectory()调用处理新文件时，它都会更新，并在处理该目录完成后恢复。

**CMAKE_CURRENT_BINARY_DIR**：当前由CMake处理的CMakeLists.txt文件对应的构建目录。每次调用add_subdirectory()时都会更改，处理完add_subdirectory()后会再次恢复。

以下是一个示例，帮助说明这种行为:

Top level CMakeLists.txt:
```
cmake_minimum_required(VERSION 3.0)
project(MyApp)

message("top: CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message("top: CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message("top: CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message("top: CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(mysub)

message("top: CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message("top: CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

```
mysub/CMakeLists.txt:
```
message("mysub: CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message("mysub: CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message("mysub: CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message("mysub: CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")
```
对于上面的示例，如果顶层的CMakeLists.txt文件位于目录/somewhere/src，构建目录为/somewhere/build，则将生成以下输出：

```
top: CMAKE_SOURCE_DIR = /somewhere/src
top: CMAKE_BINARY_DIR = /somewhere/build
top: CMAKE_CURRENT_SOURCE_DIR = /somewhere/src
top: CMAKE_CURRENT_BINARY_DIR = /somewhere/build
mysub: CMAKE_SOURCE_DIR = /somewhere/src
mysub: CMAKE_BINARY_DIR = /somewhere/build
mysub: CMAKE_CURRENT_SOURCE_DIR = /somewhere/src/mysub
mysub: CMAKE_CURRENT_BINARY_DIR = /somewhere/build/mysub
top: CMAKE_CURRENT_SOURCE_DIR = /somewhere/src
top: CMAKE_CURRENT_BINARY_DIR = /somewhere/build
```

这表明了源目录和构建目录变量的变化情况，以及在不同的CMakeLists.txt文件中它们是如何更新的。

### 7.1.2. 作用域

在第5.4节“作用域块”中，讨论了作用域的概念。调用add_subdirectory()的一个效果是，CMake为处理该子目录的CMakeLists.txt文件创建了一个新的作用域。这个新的作用域就像调用作用域的block()命令创建的本地子作用域一样。效果非常相似：

* 进入时，调用作用域中定义的所有变量都被复制到子目录的子作用域中。
* 在子目录的子作用域中创建的任何新变量对调用作用域不可见。
* 对子目录的子作用域中变量的任何更改都局限于该子作用域。
* 在子目录的子作用域中取消设置变量不会取消在调用作用域中的设置。

CMakeLists.txt文件：

```
set(myVar foo)
message("Parent (before): myVar = ${myVar}")
message("Parent (before): childVar = ${childVar}")
add_subdirectory(subdir)
message("Parent (after): myVar = ${myVar}")
message("Parent (after): childVar = ${childVar}")
```

subdir/CMakeLists.txt文件：

```
message("Child (before): myVar = ${myVar}")
message("Child (before): childVar = ${childVar}")
set(myVar bar)
set(childVar fuzz)
message("Child (after): myVar = ${myVar}")
message("Child (after): childVar = ${childVar}")
```

这将产生以下输出：

```
Parent (before): myVar = foo   ①
Parent (before): childVar =    ②
Child (before): myVar = foo    ③
Child (before): childVar =     ④
Child (after): myVar = bar     ⑤
Child (after): childVar = fuzz ⑥
Parent (after): myVar = foo    ⑦
Parent (after): childVar =     ⑧
```

* ① myVar 在父作用域定义 
* ② childVar 在父作用域没有定义，因此是空 
* ③ myVar 在子作用域依然可见
* ④ childVar 这个时候在子作用域还是没有定义
* ⑤ myVar 在子作用域被修改
* ⑥ childVar 在子作用域被定义
* ⑦ 当返回父作用域时, myVar恢复为调用add_subdirectory()之前的值。在子作用域对myVar的修改不会影响父作用域。
* ⑧ childVar在子作用域定义，对于父作用域不可见，所以是空。


上述变量的作用域行为突显了add_subdirectory()的一个重要特性。它允许被添加的目录更改任何它想要更改的变量，而不影响调用作用域中的变量。这有助于保持调用作用域与潜在不希望更改的内容隔离。


如第5.4节“作用域块”中所讨论的，PARENT_SCOPE关键字可以与set()或unset()命令一起使用，以在父作用域而不是当前作用域中更改或取消设置变量。对于由add_subdirectory()创建的子作用域，它的工作方式与创建本地子作用域的方式相同：

CMakeLists.txt文件：

```
set(myVar foo)
message("Parent (before): myVar = ${myVar}")
add_subdirectory(subdir)
message("Parent (after): myVar = ${myVar}")
```

subdir/CMakeLists.txt:

```
message("Child (before): myVar = ${myVar}")
set(myVar bar PARENT_SCOPE)
message("Child (after): myVar = ${myVar}")
```

这将产生以下输出：

```
Parent (before): myVar = foo
Child (before): myVar = foo
Child (after): myVar = foo  ①
Parent (after): myVar = bar ②
```

* ① 子作用域中的myVar不受set()调用的影响，因为PARENT_SCOPE关键字告诉CMake修改父级的myVar，而不是本地的myVar。
* ② 父作用域的myVar已经被子作用域中的set()调用修改。

由于使用PARENT_SCOPE阻止了通过命令修改同名本地变量，如果本地作用域不重用与父级相同的变量名，可能会更容易理解。在上面的示例中，更清晰的命令集将是：

subdir/CMakeLists.txt

```
set(localVar bar)
set(myVar ${localVar} PARENT_SCOPE)
```

显然，上述是一个简单的示例，但对于真实项目，可能有许多命令会在最终设置父级的myVar变量之前累积localVar的值。

不仅变量受作用域影响，策略和一些属性在这方面也具有类似的行为。在策略的情况下，每次add_subdirectory()调用都会创建一个新的作用域，其中可以在不影响父级策略设置的情况下进行策略更改。类似地，有一些目录属性可以在子目录的CMakeLists.txt文件中设置，而这不会影响父目录的目录属性。这两者在它们各自的章节中有更详细的介绍：第12章“策略”和第9章“属性”。


### 7.1.3. 何时调用 project()

有时会出现一个问题，即是否应在子目录的 CMakeLists.txt 文件中调用 project()。在大多数情况下，这既不是必要的也不是可取的，但是是允许的。唯一必须调用 project() 的地方是顶层的 CMakeLists.txt 文件。在读取顶层 CMakeLists.txt 文件时，CMake会扫描该文件的内容，寻找对 project() 的调用。如果找不到这样的调用，CMake将发出警告并插入一个使用默认的 C 和 C++ 语言启用的内部 project() 调用。项目永远不应依赖于此机制，它们应始终显式调用 project()。请注意，仅通过包装函数或通过 add_subdirectory() 或 include() 读取的文件调用 project() 是不够的，顶级 CMakeLists.txt 文件必须直接调用 project()。

在子目录中调用 project() 通常不会造成伤害，但可能导致 CMake 不得不生成额外的文件。在大多数情况下，这些额外的 project() 调用和生成的文件只是噪音，但在某些情况下可能会有用。使用 Visual Studio 项目生成器时，每个 project() 命令都会导致关联的解决方案文件的创建。通常，开发人员会加载与顶层 project() 调用相对应的解决方案文件（该解决方案文件将位于构建目录的顶部）。该顶层解决方案文件包含项目中的所有目标。为子目录内的任何 project() 调用生成的解决方案文件将包含一个更简化的视图，仅包含该目录范围及以下的目标，以及来自构建的其余部分的任何其他目标，它们依赖于这些目标。开发人员可以加载这些子解决方案，而不是顶层解决方案，以获得项目的更简化视图，从而可以专注于目标的较小子集。对于具有许多目标的非常大的项目，这可能特别有用。

Xcode 生成器的行为类似，对于每个 project() 调用都会创建一个 Xcode 项目。可以加载这些 Xcode 项目以获得类似的简化视图，但与 Visual Studio 生成器不同，它们不包括构建该目录范围或以下目标的逻辑。开发人员负责确保从该简化视图之外需要的任何内容已经构建。在实践中，这意味着可能需要首先加载和构建顶级项目，然后再切换到简化的 Xcode 项目。

## 7.2. include()

CMake提供的另一种从其他目录中引入内容的方法是include()命令，该命令有以下两种形式：

```
include(fileName [OPTIONAL] [RESULT_VARIABLE myVar] [NO_POLICY_SCOPE])
include(module [OPTIONAL] [RESULT_VARIABLE myVar] [NO_POLICY_SCOPE])
```

第一种形式在某种程度上类似于add_subdirectory()，但有重要的区别：

* include()期望读取的是一个文件的名称，而add_subdirectory()期望的是一个目录，并将在该目录内查找CMakeLists.txt文件。传递给include()的文件名通常具有.cmake扩展名，但它可以是任何名称。
* include()不引入新的变量作用域，而add_subdirectory()引入了新的变量作用域。

默认情况下，两个命令都引入了新的策略作用域，但是include()命令可以使用NO_POLICY_SCOPE选项来告诉它不这样做（add_subdirectory()没有这样的选项）。有关策略作用域处理的更多详细信息，请参见第12章“策略”。
当处理由include()命令命名的文件时，CMAKE_CURRENT_SOURCE_DIR和CMAKE_CURRENT_BINARY_DIR变量的值不会发生变化，而对于add_subdirectory()，它们会发生变化。这将在稍后详细讨论。

include()命令的第二种形式具有完全不同的目的。它用于加载指定的模块，这是第11章“模块”中深入讨论的主题。前述除第一个点外的所有点也适用于这第二种形式。

由于在调用include()时，CMAKE_CURRENT_SOURCE_DIR的值不会发生变化，因此似乎很难让被包含的文件找到它所在的目录。CMAKE_CURRENT_SOURCE_DIR将包含调用include()的文件的位置，而不是包含的文件所在的目录。此外，与add_subdirectory()不同，其中fileName始终为CMakeLists.txt，使用include()时文件的名称可以是任何内容，因此对于被包含的文件确定其自身名称可能会很困难。为了解决这类情况，CMake提供了一组额外的变量：

**CMAKE_CURRENT_LIST_DIR**：类似于CMAKE_CURRENT_SOURCE_DIR，只是在处理包含的文件时将其更新。这是在需要当前处理文件的目录的情况下使用的变量，无论它如何被添加到构建中，它始终保存绝对路径。

**CMAKE_CURRENT_LIST_FILE**：始终给出当前正在处理的文件的名称。它始终保存指向文件的绝对路径，而不仅仅是文件名。

**CMAKE_CURRENT_LIST_LINE**：保存当前正在处理的文件的行号。这个变量很少需要使用，但在某些调试场景中可能会很有用。

请注意，上述三个变量适用于CMake正在处理的任何文件，而不仅仅是由include()命令引入的文件。即使对于通过add_subdirectory()引入的CMakeLists.txt文件，它们的值也与上述描述一样。下面的示例演示了这种行为：

CMakeLists.txt:
```
add_subdirectory(subdir)
message("")
include(subdir/CMakeLists.txt)
```

subdir/CMakeLists.txt：
```
message("CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")
message("CMAKE_CURRENT_LIST_DIR = ${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_CURRENT_LIST_FILE = ${CMAKE_CURRENT_LIST_FILE}")
message("CMAKE_CURRENT_LIST_LINE = ${CMAKE_CURRENT_LIST_LINE}")
```

输出：

```
CMAKE_CURRENT_SOURCE_DIR = /somewhere/src/subdir
CMAKE_CURRENT_BINARY_DIR = /somewhere/build/subdir
CMAKE_CURRENT_LIST_DIR = /somewhere/src/subdir
CMAKE_CURRENT_LIST_FILE = /somewhere/src/subdir/CMakeLists.txt
CMAKE_CURRENT_LIST_LINE = 5

CMAKE_CURRENT_SOURCE_DIR = /somewhere/src
CMAKE_CURRENT_BINARY_DIR = /somewhere/build
CMAKE_CURRENT_LIST_DIR = /somewhere/src/subdir
CMAKE_CURRENT_LIST_FILE = /somewhere/src/subdir/CMakeLists.txt
CMAKE_CURRENT_LIST_LINE = 5
```



上述示例还突显了include()命令的另一个有趣特性。它可以用于包含先前已经包含在构建中的文件的内容。如果大型复杂项目的不同子目录都希望利用项目某个共同区域的CMake代码，它们可以分别使用include()包含该文件。


## 7.3 项目相关的变量

正如将在后面的章节中看到的，各种情况都需要相对于源目录或构建目录中的位置的路径。考虑一个这样的例子，其中一个项目需要一个路径，该路径指向其顶层源目录中的一个文件。从“7.1.1 节：源目录和构建目录变量”中可以看出，CMAKE_SOURCE_DIR似乎是一个自然的选择，允许使用${CMAKE_SOURCE_DIR}/someFile这样的路径。但是请考虑一下，如果将该项目后来合并到另一个父项目中，通过add_subdirectory()将其引入父构建，它可以作为git子模块使用，或者使用类似于第30章“FetchContent”中讨论的技术按需获取。原始项目源树的顶部现在是父项目源树中的子目录。CMAKE_SOURCE_DIR现在指向父项目的顶部，因此文件路径将指向错误的目录。对于CMAKE_BINARY_DIR也存在类似的问题。

上述情景在在线教程和较早的项目中出现得令人惊讶，但可以轻松避免。project()命令设置了一些变量，提供了一种更可靠的方式来定义相对于目录层次结构中位置的路径。在调用project()至少一次之后，将提供以下变量：

**PROJECT_SOURCE_DIR**：当前范围或任何父范围内对project()的最近调用的源目录。项目名称（即project()命令的第一个参数）不相关。

**PROJECT_BINARY_DIR**：与由PROJECT_SOURCE_DIR定义的源目录对应的构建目录。

**projectName_SOURCE_DIR**：在当前范围或任何父范围内对project(projectName)的最近调用的源目录。这与特定项目名称以及因此与对project()的特定调用相关联。

**projectName_BINARY_DIR**：与由projectName_SOURCE_DIR定义的源目录对应的构建目录。

以下示例演示了如何使用这些变量（..._BINARY_DIR变量遵循与所示的..._SOURCE_DIR变量类似的模式）：

CMakeLists.txt:
```
cmake_minimum_required(VERSION 3.0)
message("Top level:")
message(" PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")
message(" topLevel_SOURCE_DIR = ${topLevel_SOURCE_DIR}")
add_subdirectory(child)

```
child/CMakeLists.txt:
```
message("Child:")
message(" PROJECT_SOURCE_DIR (before) = ${PROJECT_SOURCE_DIR}")
project(child)
message(" PROJECT_SOURCE_DIR (after) = ${PROJECT_SOURCE_DIR}")
message(" child_SOURCE_DIR = ${child_SOURCE_DIR}")
add_subdirectory(grandchild)

```


child/grandchild/CMakeLists.txt 
```
message("Grandchild:")
message(" PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")
message(" child_SOURCE_DIR = ${child_SOURCE_DIR}")
message(" topLevel_SOURCE_DIR = ${topLevel_SOURCE_DIR}")
```

在上述示例中，如果顶层CMakeLists.txt文件位于目录/somewhere/src中，并且构建目录为/somewhere/build，则生成的输出可能如下所示。请注意，PROJECT_SOURCE_DIR和PROJECT_BINARY_DIR在整个项目范围内保持不变，而projectName_SOURCE_DIR和projectName_BINARY_DIR仅在调用project(projectName)的范围内保持不变。

```
Top level:
  PROJECT_SOURCE_DIR = /somewhere/src
  topLevel_SOURCE_DIR = /somewhere/src
Child:
  PROJECT_SOURCE_DIR (before) = /somewhere/src
  PROJECT_SOURCE_DIR (after) = /somewhere/src/child
  child_SOURCE_DIR = /somewhere/src/child
Grandchild:
  PROJECT_SOURCE_DIR = /somewhere/src/child
  child_SOURCE_DIR = /somewhere/src/child
  topLevel_SOURCE_DIR = /somewhere/src

```

上述示例展示了与项目相关变量的多功能性。它们可以从目录层次结构的任何部分使用，可靠地引用项目中的任何其他目录。对于本节开头讨论的场景，使用\\${PROJECT_SOURCE_DIR}/someFile或者\\${projectName_SOURCE_DIR}/someFile而不是\\${CMAKE_SOURCE_DIR}/someFile将确保对someFile的路径是正确的，无论项目是作为独立构建还是作为较大项目层次结构的一部分。

一些分层的构建安排允许项目既作为独立构建也作为较大的父项目的一部分进行构建（请参阅第30章“FetchContent”）。项目的某些部分可能只在其作为构建的顶部时才有意义，比如设置打包支持。项目可以通过比较CMAKE_SOURCE_DIR和CMAKE_CURRENT_SOURCE_DIR的值来检测是否为顶层。如果它们相同，则当前目录范围必须是源树的顶层。

```
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_subdirectory(packaging)
endif()
```

上述技术由所有版本的CMake支持，是一种非常常见的模式。在CMake 3.21或更高版本中，提供了一个专用的PROJECT_IS_TOP_LEVEL变量，可以实现相同的结果，但在意图上更清晰：

```
# 需要 CMake 3.21 或更高版本
if(PROJECT_IS_TOP_LEVEL)
    add_subdirectory(packaging)
endif()
```

如果当前目录范围或以上的最近一次对project()的调用是在顶层的CMakeLists.txt文件中，则PROJECT_IS_TOP_LEVEL的值将为true。对于CMake 3.21或更高版本的每个对project()的调用，还定义了一个类似的变量<projectName>_IS_TOP_LEVEL。它被创建为一个缓存变量，因此可以从任何目录范围读取。<projectName>对应于project()命令给定的项目名称。在可能存在对project()的当前范围和感兴趣项目的范围之间的调用时，这个替代变量非常有用。


## 7.4. 提前结束处理

有时候，项目可能希望停止处理当前文件的其余部分，并将控制权返回给调用方。return()命令可以用于此目的。如果未从函数内调用，return()将结束当前文件的处理，而不管它是通过include()还是add_subdirectory()引入的。在函数内调用return()的效果在第8.4节“返回值”中有所涉及，其中特别关注了一个常见的错误，可能导致意外地从当前文件返回。

在CMake 3.24及更早版本中，return()命令无法向调用方返回任何值。从CMake 3.25开始，return()接受一个PROPAGATE关键字，它与block()命令的相同关键字类似。在PROPAGATE关键字之后列出的变量将在控制返回的范围中更新。从历史上看，return()命令通常会忽略给定给它的所有参数。因此，如果使用PROPAGATE关键字，则必须将CMP0140策略设置为NEW，以表示旧行为不适用（第12章“Policies”深入讨论了策略）。

CMakeLists.txt:
```
set(x 1)
set(y 2)
add_subdirectory(subdir)
# 此处，x 将具有值 3，y 将被取消设置
```


subdir/CMakeLists.txt:
```
# 这确保我们有一个支持 PROPAGATE 并且 CMP0140 策略设置为 NEW 的 CMake 版本。
cmake_minimum_required(VERSION 3.25)
set(x 3)
unset(y)
return(PROPAGATE x y)
```

两种涉及变量传播和与 block() 命令交互的情况值得强调。这两种情况都是因为 return() 命令更新其返回范围内的变量而产生的。对于这两种情况中的第一种情况，如果返回到的范围位于 block() 内，那么该 block() 的范围将得到更新。

```
# CMakeLists.txt
set(x 1)
set(y 2)
block()
  add_subdirectory(subdir)
# 此处，x 将具有值 3，y 将被取消设置
endblock()
# 此处，x 为 1，y 为 2
```
另一种需要强调的情况更有趣。如果 return() 语句本身位于 block() 内，那么该 block() 不会影响变量传播到返回的范围。

CMakeLists.txt:
```
set(x 1)
set(y 2)
add_subdirectory(subdir)
# Here, x will have the value 3 and y will be unset
```

subdir/CMakeLists.txt:
```
cmake_minimum_required(VERSION 3.25)
# 此 block 不会影响 x 和 y 传播到
# 父级 CMakeLists.txt 文件的范围
block()
  set(x 3)
  unset(y)
  return(PROPAGATE x y)
endblock()
```

在使用 return(PROPAGATE) 与目录范围时需要注意。尽管这似乎是将信息传递回父级范围的一种吸引人的方式，但这种用法与 CMake 最佳实践的更以目标为中心的方法不一致。将变量传播到父级范围会使项目结构更像是旧式基于变量的方法。这些方法已知是脆弱的，并且缺乏以目标为中心的方法的强大性和表现力。然而，通过从函数中返回时使用 return() 命令进行变量传播可能是合适的，如第8.4节“返回值”所讨论的。

return() 命令并不是提前结束文件处理的唯一方式。如前一节所述，项目的不同部分可能从多个地方包含同一个文件。有时候，检查这一点并且仅在首次包含时才包含文件，以防止多次重新处理文件可能是可取的。这与 C 和 C++ 头文件的情况非常相似。因此，通常会看到类似于使用 include guard 的形式：

```
if(DEFINED cool_stuff_include_guard)
    return()
endif()
set(cool_stuff_include_guard 1)
# ...
```
在CMake 3.10或更高版本中，可以使用一个专用命令来更简洁且更健壮地表达，其行为类似于C和C++的 #pragma once：

```
include_guard()
```

与手动编写的 if-endif 代码相比，这更加健壮，因为它在内部处理了保护变量的名称。该命令还接受一个可选的关键字参数 DIRECTORY 或 GLOBAL，用于指定在其中检查文件之前是否已处理的不同范围。然而，在大多数情况下，不太可能需要这些关键字。如果没有指定这两个参数，将假定变量范围，其效果与上面的 if-endif 代码完全相同。GLOBAL 确保该命令在项目的任何其他地方首次处理时即结束文件处理（即变量范围被忽略）。DIRECTORY 仅在当前目录范围及其以下范围内检查之前是否已处理过。


## 7.5. 最佳实践

在使用 add_subdirectory() 或 include() 将另一个目录引入构建中时，最佳选择并不总是明显的。一方面，add_subdirectory() 更简单，并且通过创建自己的范围更好地保持目录相对独立。另一方面，一些 CMake 命令有一些限制，只允许它们在当前文件范围内操作，因此对于这些情况，include() 更有效。第15.2.6节“源文件”和第34.5.1节“跨目录构建目标”讨论了这个主题的各个方面。

作为一般指南，大多数简单的项目可能更适合使用 add_subdirectory() 而不是 include()。它促进了项目的清晰定义，并允许给定目录的 CMakeLists.txt 更专注于该目录需要定义的内容。遵循这一策略将促进项目整体信息的更好局部化，并且只会在需要并且带来有用的好处的地方引入复杂性。并不是说 include() 本身比 add_subdirectory() 更复杂，但使用 include() 通常导致需要更明确地拼写文件路径，因为 CMake 认为当前源目录并非是包含文件的目录。在更近期的 CMake 版本中，已经移除了从不同目录调用某些命令时出现的许多限制，这进一步加强了优先选择 add_subdirectory() 的论点。

无论是使用 add_subdirectory()，include() 还是两者兼而有之，CMAKE_CURRENT_LIST_DIR 变量通常都是比 CMAKE_CURRENT_SOURCE_DIR 更好的选择。通过早期养成使用 CMAKE_CURRENT_LIST_DIR 的习惯，可以更轻松地在项目变得更加复杂时在 add_subdirectory() 和 include() 之间切换，以及移动整个目录以重构项目。

在可能的情况下，避免使用 CMAKE_SOURCE_DIR 和 CMAKE_BINARY_DIR 变量，因为这些通常会破坏将项目并入更大项目层次结构的能力。在绝大多数情况下，PROJECT_SOURCE_DIR 和 PROJECT_BINARY_DIR，或者它们的项目特定等效变量 projectName_SOURCE_DIR 和 projectName_BINARY_DIR 更适合使用。

避免在每个子目录的 CMakeLists.txt 文件中随意调用 project()。仅当该子目录可以被视为一个更或多或少独立的项目时，考虑在子目录的 CMakeLists.txt 文件中放置 project() 命令。除非整个构建具有非常多的目标，否则在除了顶级 CMakeLists.txt 文件之外的任何地方调用 project() 的需求都很少。


