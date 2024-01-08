---
layout:     post
title:      "第九章：属性"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - cmake 
---



 <!--more-->

属性影响构建过程的几乎所有方面，从源文件如何编译成目标文件，一直到构建二进制文件在打包安装程序中的安装位置。它们始终附加到特定的实体，无论是目录、目标、源文件、测试用例、缓存变量，甚至整个构建过程本身。与变量不同，属性不是保存类似变量的独立值，而是提供与其附加实体相关的信息。

对于对CMake不熟悉的人来说，属性有时会与变量混淆。虽然在功能和特性方面两者最初可能看起来相似，但属性有着非常不同的目的。变量不附加到任何特定的实体，项目通常定义和使用自己的变量是很常见的。与之相比，属性通常由CMake明确定义并记录，并且始终应用于特定的实体。导致混淆的可能原因之一是属性的默认值有时是由变量提供的。CMake用于相关属性和变量的命名通常遵循相同的模式，变量名是属性名加上CMAKE_前缀。

## 9.1 9.1. 通用属性命令
CMake提供了一些用于操作属性的命令。其中最通用的是set_property()和get_property()，允许在任何类型的实体上设置和获取任何属性。

```
set_property(entitySpecific
  [APPEND | APPEND_STRING]
  PROPERTY propertyName values...
)
```
entitySpecific定义了正在设置属性的实体。它必须是以下之一：

* GLOBAL（全局）: 表示构建本身，因此不需要指定特定实体名称。
* DIRECTORY（目录）: 如果未指定dir，则使用当前源目录。
* TARGET（目标）: 目标...
* SOURCE（源）: 源文件...
* INSTALL（安装）: 文件...
* TEST（测试）: 测试...
* CACHE（缓存）: 变量...

上述每个的第一个词定义了正在设置其属性的实体类型。GLOBAL表示构建本身，因此不需要指定特定实体名称。对于DIRECTORY，如果未命名dir，则使用当前源目录。对于所有其他类型的实体，可以列出任意数量的该类型的项。使用CMake 3.18或更高版本时，SOURCE实体类型还支持一些附加选项，这些选项在第9.5节“源属性”中讨论。
PROPERTY关键字标记了所有剩余的参数，定义了属性名及其值。propertyName通常应与CMake文档中定义的属性之一匹配，后续章节将讨论其中的一些属性。值的含义是特定于属性的。

除了CMake定义的属性之外，项目还可以设置自定义属性。对于这些属性的含义及其对构建的影响，由项目自行决定。如果选择这样做，建议项目在属性名称上使用特定于项目的前缀，以避免与CMake或其他第三方包定义的属性可能发生的名称冲突。

```
set_property(TARGET MyApp1 MyApp2
  PROPERTY MYPROJ_CUSTOM_PROP val1 val2 val3
)
```

上述示例在MyApp1和MyApp2这两个target上定义了一个自定义的MYPROJ_CUSTOM_PROP属性，其值为val1; val2; val3。它还演示了如何一次为多个目标设置属性。

APPEND和APPEND_STRING关键字可用于控制如果已经有一个值，如何更新命名属性。如果没有指定任何关键字，给定的值将替换任何先前的值。APPEND关键字更改行为以将值附加到现有值中，形成一个列表，而APPEND_STRING关键字获取现有值，并通过将两个值连接为字符串而不是列表，将新值附加到现有值中（请参见下文关于继承属性的特殊说明）。以下表演示了它们之间的区别。

| Previous Value(s) | New Value(s) | No Keyword | APPEND | APPEND_STRING |
| --- | --- | --- | --- | --- |
| foo | bar | bar | foo;bar | foobar |
| a;b | c;d | c;d | a;b;c;d | a;bc;d |

get_property()命令遵循类似的形式：

```
get_property(resultVar entitySpecific
  PROPERTY propertyName
  [DEFINED | SET | BRIEF_DOCS | FULL_DOCS]
)
```

始终需要PROPERTY关键字和propertyName。entitySpecific部分类似于set_property()中的内容，并且必须是以下之一：

* GLOBAL
* DIRECTORY [dir]
* TARGET target
* SOURCE source
* INSTALL file
* TEST test
* CACHE var
* VARIABLE

与之前一样，GLOBAL指的是整个构建，因此不需要指定特定实体的名称。DIRECTORY可用于带或不带指定特定目录，如果未提供目录，则假定为当前源目录。对于大多数其他范围，必须命名该范围内的特定实体。同样，当使用CMake 3.18或更高版本时，SOURCE实体类型在第9.5节“源属性”中支持附加选项。

VARIABLE类型有点不同，变量名被指定为propertyName，而不是附加到VARIABLE关键字。这可能看起来有点不直观，但考虑一下，如果变量与VARIABLE关键字一起命名为实体，就像对其他实体类型关键字一样。在那种情况下，就没有要为属性指定的内容了。可能有助于将VARIABLE视为指定当前范围，然后感兴趣的属性是由propertyName命名的变量。从这个角度理解时，VARIABLE与其他实体类型的处理方式是一致的。

如果没有给出可选关键字中的任何一个，属性的值将存储在由resultVar命名的变量中。这是get_property()命令的典型用法。对于VARIABLE范围，变量的值可以更直接地使用${}语法获取。可选关键字可用于检索有关属性的其他详细信息：

**DEFINED**: 获取命名属性是否已定义的布尔值。对于VARIABLE范围的查询，只有在使用define_property()命令明确定义了命名变量时，结果才为true（参见下文）。
**SET**: 获取命名属性是否已设置的布尔值。与DEFINED不同之处在于，SET查询命名属性是否实际上已设置为某个值（值本身无关），而DEFINED更多地描述属性的含义。在大多数情况下，项目通常需要SET而不是DEFINED。还请注意，属性可以对DEFINED返回true，对SET返回false，或反之亦然。
**BRIEF_DOCS**: 检索命名属性的简要文档字符串。如果未为属性定义简要文档，则结果将是字符串NOTFOUND。
**FULL_DOCS**: 检索命名属性的完整文档。如果未为属性定义完整文档，则结果将是字符串NOTFOUND。

在可选关键字中，除了SET之外，除非项目显式调用define_property()来为实体填充所请求的信息，否则它们的价值不大：

```
define_property(entityType
  PROPERTY propertyName
  [INHERITED]
  # 强制要求CMake 3.22及更早版本
  [BRIEF_DOCS briefDoc [moreBriefDocs...]]
  [FULL_DOCS fullDoc [moreFullDocs...]]
)
# 需要CMake 3.23或更高版本
[INITIALIZE_FROM_VARIABLE variableName]
```
define_property()命令不设置属性的值。相反，它控制属性如何初始化或继承，并可能提供文档。CMake 3.22及更早版本需要存在BRIEF_DOCS和FULL_DOCS，但除了通过get_property()将它们返回给项目之外，CMake不使用它们。由于它们缺乏实用性，很可能在将来的CMake版本中废弃这些文档选项。

entityType必须是GLOBAL、DIRECTORY、TARGET、SOURCE、TEST、VARIABLE或CACHED_VARIABLE之一。propertyName指定正在定义的属性。没有指定实体，尽管与get_property()命令一样，在VARIABLE的情况下，变量名被指定为propertyName。

如果给出了INHERITED选项，get_property()命令将在命名范围中未设置该属性的情况下向上链到父范围。例如，如果请求DIRECTORY属性，但未为指定的目录设置该属性，则递归查询目录范围层次结构，直到找到该属性或达到源树的顶层为止。如果在顶层目录中仍未找到，则将搜索GLOBAL范围。类似地，如果请求TARGET、SOURCE或TEST属性，但未为指定的实体设置该属性，则将搜索DIRECTORY范围（包括递归地向上搜索目录层次结构，并在必要时最终到达GLOBAL范围）。对于VARIABLE或CACHE，不提供此类链式功能，因为它们已经按设计链到父变量范围。

INHERITED属性的继承行为仅适用于get_property()命令及其特定属性类型的类似get_...函数（在下面的各节中涵盖）。在调用set_property()时，只有属性的直接值被考虑（即在计算要追加的值时不会发生继承）。

CMake 3.23增加了对INITIALIZE_FROM_VARIABLE关键字的支持，它指定要用于初始化命名属性的变量。它只能与目标属性一起使用，并且仅影响在调用define_property()后创建的目标。变量名必须以属性的名称结尾，并且不能以CMAKE_或_CMAKE_开头。这个特性是为了为项目定义的自定义属性提供默认值的一种特别有用的方式。考虑到这一点，属性名称必须还包含至少一个下划线。这个限制存在是为了鼓励项目使用具有项目特定前缀的自定义属性名称。

```
# 设置变量的示例，但也可以由用户设置，甚至保持未设置以使用空值初始化属性
set(MYPROJ_SOMETOOL_OPTIONS --verbose)
define_property(TARGET PROPERTY MYPROJ_SOMETOOL_OPTIONS
  INITIALIZE_FROM_VARIABLE MYPROJ_SOMETOOL_OPTIONS
)
```

CMake有大量每种类型的预定义属性。开发者应查阅CMake参考文档以获取可用属性及其预期目的。在后续章节中，将讨论许多这些属性及其与其他CMake命令、变量和特性的关系。



## 9.2. 全局属性
全局属性涉及整个构建作为一个整体。它们通常用于修改构建工具启动方式或工具行为的其他方面，用于定义项目文件结构的方面，并提供一定程度的构建级别信息。

除了通用的set_property()和get_property()命令外，CMake还提供了get_cmake_property()用于查询全局实体。它不仅仅是get_property()的简写，尽管它可以简单地用于检索任何全局属性的值。

```
get_cmake_property(resultVar property)
```
就像对于get_property()一样，resultVar是一个变量的名称，在命令返回时将存储所请求属性的值。property参数可以是任何全局属性的名称，或以下伪属性之一：

**VARIABLES**：返回所有常规（即非缓存）变量的列表。
**CACHE_VARIABLES**：返回所有缓存变量的列表。
**COMMANDS**：返回所有定义的命令、函数和宏的列表。命令由CMake预定义，而函数和宏可以由CMake（通常通过模块）或项目自身定义。返回的一些名称可能对应于未记录或内部实体，不打算由项目直接使用。名称的大小写可能与它们最初定义的方式不同。
**MACROS**：仅返回已定义的宏列表。这将是COMMANDS伪属性返回的子集，但请注意名称的大小写可能与COMMANDS伪属性报告的不同。
**COMPONENTS**：返回由install()命令定义的所有组件的列表，该命令在第27章“安装”中讨论。

这些只读的伪属性在技术上不是全局属性（例如，不能使用get_property()检索它们），但它们在概念上非常相似。它们只能通过get_cmake_property()检索。


## 9.3. 目录属性
目录还支持其自己的一组属性。从逻辑上讲，目录属性位于全局属性（适用于所有地方）和目标属性（仅影响个别目标）之间。因此，目录属性主要用于为目标属性设置默认值，并覆盖当前目录的全局属性或默认值。一些只读的目录属性还提供了一定程度的内省，保存有关构建如何到达目录、在该点定义了哪些内容等信息。

为了方便起见，CMake提供了专用的命令来设置和获取目录属性，比它们的通用对应命令更为简洁：

```
set_directory_properties(PROPERTIES prop1 val1 [prop2 val2] ...)
get_directory_property(resultVar [DIRECTORY dir] property)
get_directory_property(resultVar [DIRECTORY dir] DEFINITION varName)
```
尽管这个目录特定的设置命令更为简洁，但它缺乏任何APPEND或APPEND_STRING选项。这意味着它只能用于设置或替换属性，不能直接用于向现有属性添加。与更通用的set_property()相比，这个命令的另一个限制是它始终应用于当前目录。项目可以选择在方便的地方使用这个更具体的形式，并在其他地方使用通用形式，或者出于一致性考虑，通用形式可能在所有地方都使用。两种方法都没有更正确的一种，更多是一种偏好的问题。

目录特定的getter命令有两种形式。第一种形式用于从特定目录或当前目录（如果未使用DIRECTORY参数）获取属性的值。第二种形式检索变量的值，这可能看起来并不那么有用，但它提供了一种从除了当前目录之外的不同目录范围获取变量值的方法（当使用DIRECTORY参数时）。在实践中，几乎不太需要使用这种第二种形式，除非用于调试构建或类似的临时任务。

对于get_directory_property()命令的任何形式，如果使用了DIRECTORY参数，则命名目录必须已由CMake处理。CMake无法了解尚未遇到的目录范围的属性。

## 9.4. 目标属性

在CMake中，很少有什么东西像目标属性那样对目标构建方式产生如此强烈和直接的影响。它们控制并提供有关从用于编译源文件的标志到构建的二进制和中间文件的类型和位置的一切信息。一些目标属性影响目标在开发者的IDE项目中的呈现方式，而其他一些影响编译/链接时使用的工具。简而言之，目标属性是关于如何实际将源文件转化为二进制文件的大部分详细信息的地方。

在CMake中，有许多用于操作目标属性的方法。除了通用的set_property()和get_property()命令外，CMake还为方便起见提供了一些特定于目标的等效命令：

```
set_target_properties(target1 [target2...]
  PROPERTIES
  propertyName1 value1
  [propertyName2 value2] ...
)
get_target_property(resultVar target propertyName)
```
与set_directory_properties()命令一样，set_target_properties()缺乏set_property()的完整灵活性，但为常见情况提供了更简单的语法。set_target_properties()命令不支持追加到现有属性值，如果需要为给定属性提供列表值，则set_target_properties()命令要求将该值以字符串形式指定，例如"this;is;a;list"。

get_target_property()命令是get_property()的简化版本。它专注于提供一种简单的方法来获取目标属性的值，基本上只是通用命令的简写。

除了通用和特定于目标的属性获取器和设置器之外，CMake还有许多其他命令用于修改目标属性。特别是target_...()命令系列是CMake的关键部分，除了最简单的项目外，通常都会使用它们。这些命令不仅为特定目标定义属性，还定义了如何将这些信息传播给链接到它的其他目标。第15章“编译器和链接器基础”深入介绍了这些命令及它们与目标属性的关系。



## 9.5. 源文件属性
CMake还支持对单个源文件设置属性。这使得可以在文件级别上进行细粒度的编译器标志操作，而不是针对目标的所有源文件。它们还允许提供关于源文件的附加信息，以修改CMake或构建工具如何处理该文件。例如，它们可以指示文件是否作为构建的一部分生成，使用哪个编译器，与文件一起使用的非编译器工具的选项等。

项目通常不需要查询或修改源文件属性，但对于需要的情况，CMake提供了专用的设置和获取命令，以使任务更加容易。这些命令遵循与其他特定属性的设置和获取命令相似的模式：

```
set_source_files_properties(file1 [file2...]
  PROPERTIES
  propertyName1 value1
  [propertyName2 value2] ...
)
get_source_file_property(resultVar sourceFile propertyName)
```

同样，setter不提供APPEND功能，而getter实际上只是通用get_property()命令的语法缩写，不提供新功能。以下示例显示了如何在源文件上设置属性，以防止将其与其他源文件合并为一个统一构建（在第35.1节“统一构建”中讨论）：

```
add_executable(MyApp small.cpp big.cpp tall.cpp thin.cpp)
set_source_files_properties(big.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION YES)
```

在CMake 3.17及更早版本中，源属性仅对在同一目录范围中定义的目标可见。如果源属性的设置发生在不同的目录范围中，目标将无法看到该属性更改，因此该源文件的编译等不会受到影响。在CMake 3.18或更高版本中，可以使用附加选项来指定应搜索或应用源文件属性的目录范围。以下显示了在CMake 3.18或更高版本中设置源文件属性的所有选项：

```
set_property(SOURCE sources...
  [DIRECTORY dirs...]
  [TARGET_DIRECTORY targets...]
  [APPEND | APPEND_STRING]
  PROPERTY propertyName values...
)
set_source_files_properties(sources...
  [DIRECTORY dirs...]
  [TARGET_DIRECTORY targets...]
  PROPERTIES
  propertyName1 value1
  [propertyName2 value2] ...
)
```

DIRECTORY选项可用于指定应设置源属性的一个或多个目录。在这些目录中创建的任何目标将了解源属性。这些目录必须已通过对add_subdirectory()的先前调用添加到构建中。任何相对路径将被视为相对于当前源目录。

TARGET_DIRECTORY选项类似，只是后面跟着目标的名称。对于列出的每个目标，将处理创建该目标的目录（即其源目录），就好像已经使用DIRECTORY选项指定了该目录。请注意，这意味着在该目录中定义的所有目标都将了解源属性，而不仅仅是指定的目标。

CMake 3.18 还添加了用于检索源文件属性的类似选项：

```
get_property(resultVar SOURCE source
  [DIRECTORY dir | TARGET_DIRECTORY target]
  PROPERTY propertyName
  [DEFINED | SET | BRIEF_DOCS | FULL_DOCS]
)
get_source_file_property(resultVar source
  [DIRECTORY dir | TARGET_DIRECTORY target]
  propertyName
)
```

在检索源文件属性时，最多只能列出一个目标或目录，以确定从中检索属性的目录范围。如果未指定DIRECTORY或TARGET_DIRECTORY，则假定当前源目录。

无论是否指定DIRECTORY或TARGET_DIRECTORY选项，要注意源文件可能被编译到多个目标中。因此，在设置源属性的每个目录范围中，这些属性应对使用这些文件的所有目标都有意义。

开发者应注意一个可能在某些情况下对其使用构成强烈威慑的实现细节。对于某些CMake生成器（尤其是Unix Makefiles生成器），源和源属性之间的依赖关系可能比预期的要强。如果源属性用于修改特定源文件而不是整个目标的编译器标志，则更改源的编译器标志仍将导致重新构建所有目标的源文件，而不仅仅是受影响的源文件。这是在Makefile中处理依赖项细节的限制，其中测试每个单独源的编译器标志是否已更改会带来难以承受的性能损失。相关的Makefile依赖关系已经在目标级别实现，以避免这个问题。

项目可能会诱使使用源属性的典型情况是将版本详细信息传递给仅一个或两个源作为编译器定义。如第21.2节“源代码访问版本详细信息”所讨论的，有比上述构建性能问题更好的源属性替代方案。设置一些源属性还可以通过阻止这些源参与统一构建（参见第35.1节“统一构建”）来降低构建性能。

Xcode生成器在支持源属性方面也存在一个限制，无法处理特定于配置的属性值。有关此限制可能重要的情景，请参见第15.6节“特定语言的编译器标志”。


## 9.6. 缓存变量属性
与其他属性类型相比，缓存变量的属性在目的上略有不同。在很大程度上，缓存变量的属性更侧重于CMake GUI和基于控制台的ccmake工具对缓存变量的处理，而不是以任何实质方式影响构建。此外，没有提供额外的命令来操纵它们，因此必须使用带有CACHE关键字的通用set_property()和get_property()命令。

在第5.3节“缓存变量”中，讨论了缓存变量的许多方面，这些方面最终体现在缓存变量的属性中。

每个缓存变量都有一个类型，必须是BOOL、FILEPATH、PATH、STRING或INTERNAL之一。可以使用get_property()与属性名称TYPE来获取此类型。类型影响CMake GUI和ccmake在UI中如何呈现该缓存变量以及用于编辑其值的小部件类型。任何类型为INTERNAL的变量将根本不显示。
可以使用mark_as_advanced()命令将缓存变量标记为高级，这实际上只是设置了布尔型高级缓存变量属性。CMake GUI和ccmake工具都提供了显示或隐藏高级缓存变量的选项。用户然后可以选择是专注于主要基本变量还是查看完整列表。
缓存变量的帮助字符串通常在调用set()命令的一部分中设置，但也可以使用HELPSTRING缓存变量属性进行修改或读取。此帮助字符串用作CMake GUI中的工具提示以及ccmake工具中的一行帮助提示。
如果缓存变量的类型为STRING，则CMake GUI将查找名为STRINGS的缓存变量属性。如果不为空，它应该是该变量的有效值列表，CMake GUI将呈现该变量作为这些值的组合框，而不是任意文本输入小部件。在ccmake的情况下，按回车键会循环显示提供的值。请注意，CMake不强制要求缓存变量必须是STRINGS属性中的值之一，这只是对CMake GUI和ccmake工具的方便。当CMake运行其配置步骤时，仍将缓存变量视为任意字符串，因此仍然可以在cmake命令行或通过项目中的set()命令中为缓存变量提供任何值。



## 9.7. 其他属性类型
CMake还支持在单独的测试上设置属性，并提供了通常用于测试的版本的属性设置和获取命令：

```
set_tests_properties(test1 [test2...]
  PROPERTIES
  propertyName1 value1
  [propertyName2 value2] ...
)
get_test_property(resultVar test propertyName)
```
与它们的通用对应命令相比，这些命令只是略微更简洁的版本，不具备APPEND功能，但在某些情况下可能更为方便。有关测试的详细讨论，请参见第26章，“测试”。

CMake支持的另一种属性类型是安装文件的属性。这些属性特定于所使用的打包类型，通常大多数项目都不需要它们。


## 9.8. 最佳实践
属性是CMake的关键组成部分。一系列命令具有设置、修改或查询各种类型属性的能力，其中一些对项目之间的依赖关系产生进一步的影响。

除了特殊的全局伪属性之外，所有属性类型均可使用通用的set_property()命令进行完全操作，使其对开发人员可预测，并在需要时提供灵活的APPEND功能。属性特定的设置命令在某些情况下可能更为方便，比如允许一次设置多个属性，但它们缺少APPEND功能可能会导致一些项目只使用set_property()。两者都没有对与其替换而不是追加属性值的常见错误提供明确的解决方案。

对于目标属性，强烈建议使用各种target_...()命令，而不是直接操纵相关联的目标属性。这些命令不仅操纵特定目标上的属性，还建立了目标之间的依赖关系，以便CMake可以自动传播一些属性。第15章，“编译器和链接器基础”讨论了一系列主题，强调了对target_...()命令的强烈偏好。

源文件属性提供了对编译器选项等级别的细粒度控制。然而，这确实可能对项目的构建行为产生不良的负面影响。特别是，一些CMake生成器在仅有少数源文件的编译选项发生更改时可能会重新构建比必要的更多内容。Xcode生成器还存在一些限制，阻止其支持特定于配置的源文件属性。项目应考虑在可能的情况下使用其他替代方案，比如第21.2节，“版本详细信息的源代码访问”中介绍的技术。






























 








