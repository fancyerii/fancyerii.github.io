---
layout:     post
title:      "第三章：一个最小的项目"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - cmake 
---



 <!--more-->

所有 CMake 项目都始于一个名为 CMakeLists.txt 的文件，并且预期将其放置在源代码树的顶部。将其视为 CMake 项目文件，它定义了构建的所有内容，从源和目标到测试、打包以及其他自定义任务。它可以非常简单，只有几行代码，也可以相当复杂，并从其他目录引入更多文件。CMakeLists.txt 只是一个普通的文本文件，通常直接进行编辑，就像项目中的任何其他源文件一样。

延续与源的类比，CMake 定义了自己的语言，其中包含许多程序员熟悉的概念，如变量、函数、宏、条件逻辑、循环、代码注释等等。这些各种概念和特性将在接下来的几章中介绍，但现在的目标只是建立一个简单的构建作为起点。以下是一个最小化、格式良好的 CMakeLists.txt 文件，生成一个基本的可执行文件。

```bash
cmake_minimum_required(VERSION 3.2)
project(MyApp)
add_executable(MyExe main.cpp)
```

上面示例中的每一行都执行一个内置的 CMake 命令。在 CMake 中，命令类似于其他语言的函数调用，不同之处在于虽然它们支持参数，但它们不会直接返回值（不过后面的章节会展示如何以其他方式将值传递回调用方）。参数之间用空格分隔，也可以跨多行分割：

```
add_executable(MyExe
 main.cpp
 src1.cpp
 src2.cpp
)
```

命令是大小写不敏感的，因此下面的三行是等价的：

```
add_executable(MyExe main.cpp)
ADD_EXECUTABLE(MyExe main.cpp)
Add_Executable(MyExe main.cpp)
```

习惯的做法是命令名全部用小写。

## 管理CMake的版本

CMake 不断更新和扩展，以添加对新工具、平台和功能的支持。CMake 的开发人员非常谨慎，每次发布新版本时都会保持向后兼容性，因此当用户升级到较新版本的 CMake 时，项目应该继续像之前一样构建。有时，需要更改特定的 CMake 行为或在较新版本中引入更严格的检查和警告。与其要求所有项目立即处理此问题，CMake 提供了策略机制，允许项目表达“按照 CMake 版本 X.Y.Z 的方式行为”。这使得 CMake 可以在内部修复错误并引入新功能，同时仍然保持任何特定过去版本的期望行为。

项目指定其预期 CMake 版本行为的主要方式是使用 cmake_minimum_required() 命令。这应该是 CMakeLists.txt 文件的第一行，以便在任何其他操作之前检查和建立项目的要求。该命令执行两项操作：

* 它指定项目所需的 CMake 的最小版本。如果使用比指定版本旧的 CMake 处理 CMakeLists.txt 文件，它将立即停止并显示错误。这确保在继续之前有一组特定的 CMake 功能是可用的。
* 它强制执行策略设置以匹配 CMake 的行为到指定的版本。

使用此命令非常重要，以至于如果 CMakeLists.txt 文件在任何其他命令之前没有调用 cmake_minimum_required()，CMake 将发出警告。它需要知道如何为所有后续处理设置策略行为。对于大多数项目，将 cmake_minimum_required() 视为简单地指定所需的最小 CMake 版本就足够了，正如其名称所示。它还意味着 CMake 应该与该特定版本的行为相同，这可以被视为一个有用的附加好处。第12章“策略”详细讨论了策略设置，并解释了如何根据需要调整此行为。

cmake_minimum_required() 命令的典型形式如下：

```
cmake_minimum_required(VERSION major.minor[.patch[.tweak]])
```

VERSION 关键字必须始终存在，提供的版本详细信息至少必须包含主版本号和次版本号。在大多数项目中，指定修补程序和调整部分通常是不必要的，因为新功能通常只会出现在次版本更新中（这是从版本3.0开始的官方CMake行为）。只有在需要特定错误修复时，项目才应指定修补程序部分。此外，由于3.x系列中的任何CMake版本都没有使用调整号，因此项目也不需要指定。

开发人员应该仔细考虑其项目应该要求的最低CMake版本是多少。版本3.2可能是任何新项目应该考虑的最旧版本，因为它为现代CMake技术提供了一个相当完整的功能集。版本2.8.12的功能覆盖较少，缺少许多有用的功能，但对于旧项目可能是可行的。在此之前的版本缺乏许多现代CMake技术所需的重要功能。如果与快速变化的平台（例如iOS）一起工作，则可能需要使用相当新的CMake版本以支持最新的操作系统发布等。

作为一般的经验法则，选择最新的CMake版本，而不会给构建项目的人造成重大问题。通常，面临最大困难的是需要支持较旧平台的项目，其中系统提供的CMake版本可能非常旧。对于这种情况，如果可能的话，开发人员应考虑安装更近期的版本，而不是限制自己使用非常老旧的CMake版本。另一方面，如果项目本身将成为其他项目的依赖项，那么选择更近期的CMake版本可能会成为采用的一道障碍。在这种情况下，要求最旧的CMake版本仍提供所需的最低CMake功能，但如果可用，可以利用后续CMake版本的功能（第12章“策略”介绍了实现这一点的技术）。这将防止其他项目被迫要求使用比其目标环境通常允许或提供的更近期版本。依赖项目始终可以要求更近期的版本，如果他们愿意的话，但不能要求较旧的版本。使用最旧可行版本的主要缺点是它可能会导致更多的弃用警告，因为更新的CMake版本会警告有关较旧行为的信息，以鼓励项目更新自己。

## project()命令

每个CMake项目都应包含一个project()命令，并且它应该在调用cmake_minimum_required()之后出现。该命令及其最常见的选项具有以下形式：

```
project(projectName
 [VERSION major[.minor[.patch[.tweak]]]]
 [LANGUAGES languageName ...]
)
```

项目名称是必需的，只能包含字母、数字、下划线（_）和连字符（-），尽管在实践中通常只使用字母和可能是下划线。由于不允许使用空格，因此项目名称不必用引号括起来。该名称用于项目的顶层（例如Xcode和Visual Studio等项目生成器）以及项目的各个其他部分，例如作为打包和文档元数据的默认值，提供项目特定的变量等。该名称是project()命令的唯一强制参数。

可选的VERSION详细信息仅在CMake 3.0及更高版本中支持。与项目名称一样，版本详细信息由CMake用于填充一些变量并作为默认的包元数据，但除此之外，版本详细信息并没有其他重要性。尽管如此，一个良好的习惯是在此处定义项目的版本，以便项目的其他部分可以引用它。第21章“指定版本详细信息”深入介绍了这一点，并解释了如何在后续的CMakeLists.txt文件中引用此版本信息。

可选的LANGUAGES参数定义了应为项目启用的编程语言。支持的值包括C、CXX、Fortran、ASM、CUDA等。如果指定多种语言，请用空格分隔每种语言。在一些特殊情况下，项目可能希望指示不使用任何语言，可以使用LANGUAGES NONE来实现。后续章节介绍的技术利用了这种特定形式。如果未提供LANGUAGES选项，CMake将默认使用C和CXX。CMake版本3.0之前不支持LANGUAGES关键字，但仍然可以使用命令的旧形式在项目名称之后指定语言，如下所示：

```
project(MyProj C CXX)
```

鼓励新项目将最低CMake版本指定为至少3.0，并使用带有LANGUAGES关键字的新形式。project()命令不仅仅是填充一些变量。它的一个重要责任是检查每种启用语言的编译器，并确保它们能够成功编译和链接。编译器和链接器设置的问题会在非常早期被发现。一旦这些检查通过，CMake会设置许多变量和属性，这些变量和属性控制了启用语言的构建。第23章“工具链和交叉编译”在更大的细节中讨论了这个领域，包括影响工具链选择和配置的各种方式。第7章“使用子目录”还讨论了影响project()命令使用的其他考虑因素和要求。

当CMake执行的编译器和链接器检查成功时，它们的结果会被缓存，以便在后续的CMake运行中不必重复。这些缓存的详细信息存储在构建目录的CMakeCache.txt文件中。有关检查的附加详细信息可以在构建区域的子目录中找到，但开发人员通常只有在使用新或不寻常的编译器或设置交叉编译的工具链文件时，才需要查看那里。

## 构建一个简单的应用程序

为了完成我们的最小示例，add_executable() 命令告诉CMake从一组源文件创建可执行文件。这个命令的基本形式是：

```bash
add_executable(targetName source1 [source2 ...])
```

这将创建一个可执行文件，可以在CMake项目中引用为targetName。这个名称可以包含字母、数字、下划线和连字符。当构建项目时，在构建目录中将创建一个可执行文件，其平台相关的名称默认基于目标名称。考虑以下简单的示例命令：

```bash
add_executable(MyApp main.cpp)
```
默认情况下，在Windows上可执行文件的名称将是MyApp.exe，而在类Unix平台上（如macOS、Linux等）将是MyApp。可通过目标属性进行自定义可执行文件名称，这是在第9章“属性”中介绍的CMake功能。还可以通过多次调用add_executable()并使用不同的目标名称在一个CMakeLists.txt文件中定义多个可执行文件。如果在多个add_executable()命令中使用相同的目标名称，CMake将失败并突出显示错误。

## 注释

在离开本章之前，演示如何向CMakeLists.txt文件添加注释将是有用的。注释在本书中被广泛使用，鼓励开发人员养成像对待普通源代码一样对项目进行注释的习惯。CMake遵循类似于Unix shell脚本的注释约定。以#字符开头的任何行都被视为注释。在CMakeLists.txt文件中，在引号字符串内以外的任何行上，#后的任何内容也被视为注释。以下显示了一些注释示例，并总结了本章介绍的概念：


```bash
cmake_minimum_required(VERSION 3.2)
# We don't use the C++ compiler, so don't let project()
# test for it in case the platform doesn't have one
project(MyApp VERSION 4.7.2 LANGUAGES C)
# Primary tool for this project
add_executable(MainTool
 
 main.c
 
 debug.c # Optimized away for release builds
)
# Helpful diagnostic tool for development and testing
add_executable(TestTool testTool.c)
```

## 推荐做法

确保每个CMake项目的顶层CMakeLists.txt文件都有一个cmake_minimum_required()命令作为第一行。在决定要指定的最低要求版本号时，请记住，版本越高，项目将能够使用的CMake功能就越多。这也意味着项目可能更有可能适应新的平台或操作系统发布，这些发布通常会引入构建系统需要处理的新内容。相反，如果创建一个旨在作为操作系统本身的一部分构建和分发的项目（在Linux中很常见），最低的CMake版本可能会由该分发提供的CMake版本决定。

如果项目可以要求CMake 3.0或更高版本，也可以在早期强制考虑项目版本号，并尽早将版本编号纳入project()命令中。在项目的生命周期后期，克服现有流程的惯性并改变版本号的处理方式可能会非常困难。在确定版本策略时，请考虑语义化版本（Semantic Versioning）等流行的做法。
