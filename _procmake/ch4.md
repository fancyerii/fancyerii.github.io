---
layout:     post
title:      "第四章：构建简单目标"
author:     "lili"
mathjax: true
sticky: true
excerpt_separator: <!--more-->
tags:
    - cmake 
---



 <!--more-->

如前一章所示，使用CMake定义一个简单的可执行文件非常简单。先前给出的简单示例需要为可执行文件定义一个目标名称，并列出要编译的源文件：

```
add_executable(MyApp main.cpp)
```

这假设开发人员想要构建一个基本的控制台可执行文件，但CMake还允许开发人员定义其他类型的可执行文件，例如在Apple平台上的应用程序包和Windows GUI应用程序。本章讨论了可以传递给add_executable()的附加选项，以指定这些细节。

除了可执行文件，开发人员还经常需要构建和链接库。CMake支持几种不同类型的库，包括静态库、共享库、模块和框架。CMake还提供了管理目标之间的依赖关系以及如何链接库的强大功能。本章的大部分内容涵盖了关于库的知识，以及如何在CMake中处理它们。本章介绍的概念在本书的其余部分广泛应用。还提供了一些关于变量和属性的基本用法，以展示这些CMake功能与库和目标的关系。

## 可执行程序

add_executable()命令的更完整形式如下：

```
add_executable(targetName [WIN32] [MACOSX_BUNDLE]
 [EXCLUDE_FROM_ALL]
 source1 [source2 ...]
)
```
与之前显示的形式唯一的区别是新增的可选关键字。

**WIN32** 

在Windows平台上构建可执行文件时，此选项指示CMake将可执行文件构建为Windows GUI应用程序。在实践中，这意味着它将使用WinMain()入口点而不仅仅是main()，并且将链接到/SUBSYSTEM:WINDOWS选项。在其他所有平台上，WIN32选项将被忽略。

**MACOSX_BUNDLE**

存在时，此选项指示CMake在构建时创建应用程序包，适用于Apple平台。与选项名称所示的相反，它不仅适用于macOS，还适用于其他Apple平台，如iOS。此选项的确切效果在平台之间略有差异。例如，在macOS上，应用程序包的布局具有非常特定的目录结构，而在iOS上，目录结构是扁平化的。CMake还将为bundles生成基本的Info.plist文件。有关这些和其他详细信息，可以在第24.2节“应用程序包”中找到。

**EXCLUDE_FROM_ALL**

有时，项目定义了许多目标，但默认情况下只应构建其中一些。当在构建时未指定目标时，将构建默认的ALL目标（取决于使用的CMake生成器，名称可能略有不同，例如对于Xcode，可能是ALL_BUILD）。如果使用EXCLUDE_FROM_ALL选项定义了一个可执行文件，它将不会包含在默认ALL目标中。然后，仅当通过构建命令显式请求它或者它是默认ALL构建的另一个目标的依赖项时，才会构建可执行文件。从ALL中排除目标的情况通常很有用，其中可执行文件是仅偶尔需要的开发人员工具。

除了上述形式外，add_executable()命令还有其他形式，它们产生对现有可执行文件或目标的引用，而不是定义要构建的新可执行文件。这些别名可执行文件在第18章“目标类型”中有详细介绍。

## 构建库

创建简单的可执行文件是任何构建系统的基本需求。对于许多较大的项目，创建和使用库的能力也是至关重要的，以保持项目的可管理性。CMake支持构建各种不同类型的库，处理许多平台差异，但仍支持每个平台的本地特殊性。库目标是使用add_library()命令定义的，该命令有多种形式。其中最基本的形式如下：

```
add_library(targetName [STATIC | SHARED | MODULE]
 [EXCLUDE_FROM_ALL] 
 source1 [source2 ...]
)
```

这种形式类似于使用add_executable()定义简单可执行文件的方式。targetName在CMakeLists.txt文件中用于引用库，文件系统上构建库的名称默认派生自此名称。EXCLUDE_FROM_ALL关键字的作用与add_executable()中的相同，即防止库包含在默认的ALL目标中。要构建的库的类型由剩下的三个关键字之一指定，即STATIC、SHARED或MODULE。

**STATIC**

指定静态库。在Windows上，默认库名称将是targetName.lib，而在类Unix平台上，它通常是libtargetName.a。

**SHARED**

指定共享或动态链接库。在Windows上，默认库名称将是targetName.dll，在Apple平台上将是libtargetName.dylib，而在其他类Unix平台上通常是libtargetName.so。在Apple平台上，共享库也可以标记为框架，这是第24.3节“框架”中涵盖的主题。

**MODULE**

指定类似于共享库的库，但旨在在运行时动态加载，而不是直接链接到库或可执行文件。这些通常是用户可以选择加载或不加载的插件或可选组件。在Windows平台上，不会为DLL创建导入库。

可以省略定义要构建的库类型的关键字。除非项目明确需要特定类型的库，否则首选做法是不指定它，而是在构建项目时将选择权留给开发人员。在这种情况下，库将是STATIC或SHARED，选择取决于名为BUILD_SHARED_LIBS的CMake变量的值。如果BUILD_SHARED_LIBS已设置为true，则库目标将是共享库，否则它将是静态库。关于如何使用变量的详细信息请参阅第5章“变量”，但目前设置此变量的一种方法是在cmake命令行上使用-D选项，如下所示：

```
cmake -DBUILD_SHARED_LIBS=YES /path/to/source
```

也可以在CMakeLists.txt文件中放置add_library()命令之前设置此变量，如下所示，但这将要求开发人员在需要更改时修改它（即它的灵活性较差）：

```
set(BUILD_SHARED_LIBS YES)
```


与可执行文件一样，库目标也可以定义为引用某个现有二进制文件或目标，而不是由项目构建。还支持另一种伪库，用于收集对象文件，而不必创建静态库。所有这些都在第18章“目标类型”中详细讨论。

## 链接目标

在考虑构成项目的目标时，开发人员通常习惯于按照库A需要库B的方式思考，因此A链接到B。这是传统的库处理方式，其中一个库需要另一个库的想法非常简单。然而，实际上，库之间可以存在几种不同类型的依赖关系：

**PRIVATE**

私有依赖关系指定库A在其自身的内部实现中使用库B。与A链接的任何其他内容都不需要知道B，因为B是A的内部实现细节。 

**PUBLIC**

PUBLIC依赖关系指定库A不仅在其内部使用库B，而且还在其接口中使用B。这意味着不能在不提供来自B的类型的参数的情况下调用A中定义的至少一个参数的函数，因此使用A的任何内容也将直接依赖于B。这在库A的函数中定义了至少一个参数的类型来自库B时是一个示例，因此不能在不提供来自B的参数的情况下调用A中的函数。
 

**INTERFACE**

接口依赖关系指定为了使用库A，还必须使用库B。这与PUBLIC依赖关系不同，因为库A在内部不需要B，它只在其接口(头文件)中使用B。这在使用add_library()的INTERFACE形式定义的库目标时是有用的，例如在使用目标表示仅包含头文件的库的依赖关系时（请参阅第18.2.4节“接口库”）。

关于这三种类型的依赖关系比较难理解，下面参考[Examples of when PUBLIC/PRIVATE/INTERFACE should be used in cmake](https://stackoverflow.com/questions/69783203/examples-of-when-public-private-interface-should-be-used-in-cmake)给出一些简单解释。如果读者还不清楚可以参考[cmake：target_** 中的 PUBLIC，PRIVATE，INTERFACE](https://zhuanlan.zhihu.com/p/82244559)和[CMake的链接选项：PRIVATE，INTERFACE，PUBLIC](https://zhuanlan.zhihu.com/p/493493849)，讲得很清楚。

简单来说，如果A库的头文件不使用B(include B的头文件)，而只在实现(c/cpp文件)里使用B（什么？A库实现文件和头文件都也不使用B？那你还依赖B干嘛！)，那么就是PRIVATE依赖。而如果A库的头文件和实现都使用B，那么就是PUBLIC。如果A库的头文件使用B，但是实现不使用B，则就是INTERFACE。

也就是说，如果我们用A库的时候不需要知道B(包含其头文件)，那么B就是A的私有库，因此就是PRIVATE的。如果用A库的时候需要B，那么就是PUBLIC。那哪来的INTERFACE呢？一般情况A库依赖B，那么A一定会在实现(c/cpp文件)里用到B。但是有的特殊情况下，比如A只需要B的头文件里定义的一些结构体，但是不需要使用B的功能。那么只需要在A的头文件里包含B的头文件，那么如果我们要用A，INTERFACE就会把B的头文件也包含进来。


CMake通过其target_link_libraries()命令捕获了这种更丰富的依赖关系集，而不仅仅是需要链接的简单想法。该命令的一般形式是：

```
target_link_libraries(targetName
 <PRIVATE|PUBLIC|INTERFACE> item1 [item2 ...]
 [<PRIVATE|PUBLIC|INTERFACE> item3 [item4 ...]]
 ...
)
```
这允许项目精确定义一个库如何依赖于其他库。然后，CMake负责在以这种方式链接的库链中管理依赖关系。例如，考虑以下示例：

```
add_library(Collector src1.cpp)
add_library(Algo src2.cpp)
add_library(Engine src3.cpp)
add_library(Ui src4.cpp)
add_executable(MyApp main.cpp)
target_link_libraries(Collector
 PUBLIC Ui
 PRIVATE Algo Engine
)
target_link_libraries(MyApp PRIVATE Collector)
```

在此示例中，Ui库与Collector库的链接是PUBLIC，因此即使MyApp只直接链接到Collector，MyApp也会由于该PUBLIC关系而链接到Ui。另一方面，Algo和Engine库与Collector的链接是PRIVATE，因此MyApp将不会直接链接到它们。第18.2节“库”讨论了为了满足依赖关系而进行进一步链接的静态库的其他行为，包括循环依赖。

后面的章节介绍了另外一些target_...()命令，进一步增强了目标之间携带的依赖信息。这些命令允许编译器/链接器标志和头文件搜索路径在它们通过target_link_libraries()连接的目标之间传递。这些功能是逐步从CMake 2.8.11添加到3.2，并导致了更简单、更健壮的CMakeLists.txt文件。

后面的章节还将讨论更复杂的源目录层次结构的使用。在这种情况下，如果使用的是CMake 3.12或更早版本，那么target_link_libraries()中使用的targetName必须由同一目录中的add_executable()或add_library()命令定义（此限制在CMake 3.13中被删除）。

## 链接非目标(non-target)

在前面的部分，所有被链接的项目都是已存在的CMake目标，但target_link_libraries()命令比那更加灵活。除了CMake目标，还可以在target_link_libraries()命令中指定以下内容作为项目：

**库文件的完整路径**

CMake将把库文件添加到链接器命令。如果库文件发生更改，CMake将检测到该更改并重新链接目标。请注意，从CMake版本3.3开始，链接器命令始终使用指定的完整路径，但在3.3版本之前，有些情况下CMake可能会要求链接器搜索库（例如，将/usr/lib/libfoo.so替换为-lfoo）。这种在3.3版本之前的行为的推理和详细信息非常复杂，主要是历史性的，但对于感兴趣的读者，可以在CMake文档的CMP0060政策下找到完整的信息。

**纯库名称**

如果只提供了库的名称而没有路径，链接器命令将搜索该库（例如，foo变成-lfoo或foo.lib，取决于平台）。这对于系统提供的库很常见。

**链接标志**

作为一个特殊情况，以连字符开头但不是-l或-framework的项目将被视为要添加到链接器命令的标志。CMake文档警告说，这些标志只应用于PRIVATE项目，因为如果定义为PUBLIC或INTERFACE，则它们将传递到其他目标，而这可能并不总是安全的。

## 老版本的用法

由于历史原因，在target_link_libraries()中指定的任何链接项之前都可以加上以下关键字之一：debug、optimized或general。这些关键字的作用是根据构建是否配置为调试构建（参见第14章“构建类型”）来进一步细化后面的项应该在何时包含。如果一个项之前有debug关键字，那么只有在构建为调试构建时才会添加它。如果一个项之前有optimized关键字，它只会在构建不是调试构建时才会添加。general关键字指定该项应该对所有构建配置都添加，这在没有使用关键字的情况下是默认行为。debug、optimized和general关键字应该在新项目中避免使用，因为在当前的CMake功能下，有更清晰、更灵活和更健壮的方法来实现相同的功能。

target_link_libraries()命令还有一些其他形式，其中一些在CMake 2.8.11版本之前就存在了。为了理解旧的CMake项目，这里讨论了这些形式，但通常不鼓励在新项目中使用它们。应优先选择之前显示的带有PRIVATE、PUBLIC和INTERFACE部分的完整形式，因为它更准确地表达了依赖关系的性质。

```
target_link_libraries(targetName item [item...])
```
上面的形式通常相当于将这些项目定义为PUBLIC，但在某些情况下，它们可能会被视为PRIVATE。特别是，如果一个项目定义了一个包含新旧两种形式命令的库依赖链，旧式形式通常会被视为PRIVATE。

```
target_link_libraries(targetName
  LINK_INTERFACE_LIBRARIES item [item...]
)
```

这是上面新形式中INTERFACE关键字的前身，但CMake文档不鼓励使用。它的行为可能会影响不同的目标属性，由策略设置来控制。使用新的INTERFACE形式可以避免开发者的混淆。

```
target_link_libraries(targetName
  <LINK_PRIVATE|LINK_PUBLIC> lib [lib...]
  [<LINK_PRIVATE|LINK_PUBLIC> lib [lib...]]
)
```

类似于以前的旧式形式，这个形式是新形式的PRIVATE和PUBLIC关键字版本的前身。同样，旧式形式存在对它影响哪个目标属性以及新项目应优先选择PRIVATE/PUBLIC关键字形式的混淆。


## 最佳实践

目标名称不一定与项目名称相关。通常在教程和示例中会看到使用一个变量表示项目名称，并重用该变量作为可执行目标的名称，这是一种不好的做法，但非常常见：

```
# 不良实践，但非常常见
set(projectName MyExample)
project(${projectName})
add_executable(${projectName} ...)
```

这仅适用于最基本的项目，并鼓励了许多不良习惯。应将项目名称直接设置，而不是通过变量设置。选择目标名称时，应根据目标的功能而不是项目的名称，并假设项目最终将需要定义多个目标。这有助于在处理更复杂的多目标项目时养成更好的习惯。

在为库命名目标时，抵制以lib开头或结尾的诱惑。在许多平台上（即除Windows之外的几乎所有平台），在构造实际库名称时，前导lib将被自动添加，以符合平台的通常约定。如果目标名称已经以lib开头，则库文件名将以liblibsomething....的形式结束，人们通常会认为这是一个错误。

除非有强烈的理由，尽量避免在了解需要之前为库指定STATIC或SHARED关键字。这允许在静态库或动态库之间进行更大的灵活性选择，作为整个项目的策略。BUILD_SHARED_LIBS变量可用于在一个地方更改默认值，而无需修改每个add_library()调用。

始终在调用target_link_libraries()命令时指定PRIVATE、PUBLIC和/或INTERFACE关键字，而不是遵循假设一切都是PUBLIC的旧式CMake语法。随着项目变得越来越复杂，这三个关键字对处理目标之间的依赖关系的影响越来越大。从项目开始使用它们还迫使开发人员考虑目标之间的依赖关系，这有助于更早地发现项目中的结构问题。








