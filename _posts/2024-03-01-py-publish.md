---
layout:     post
title:      "翻译：How to Publish an Open-Source Python Package to PyPI" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - Pre-training
    - Speech
    - ASR
---

本文是[How to Publish an Open-Source Python Package to PyPI](https://realpython.com/pypi-publish-python-package/)的翻译。

<!--more-->

**目录**
* TOC
{:toc}


Python以“库齐全”而闻名，标准库中提供了许多复杂的功能。然而，要释放语言的全部潜力，您还应该利用[PyPI](https://pypi.org/)（Python包索引）中的社区贡献。

PyPI通常被读作“派派爱”(pie-pee-eye)，是一个包含数十万个软件包的存储库。这些软件包范围从简单的“Hello, World”实现到高级的深度学习库。在本教程中，您将学习如何将自己的软件包上传到PyPI。发布您的项目比以前更容易。但是，仍然涉及一些步骤。

在本教程中，您将学习如何：

* 准备要发布的Python软件包
* 处理软件包的版本控制
* 构建软件包并将其上传到PyPI
* 理解并使用不同的构建系统

在整个教程中，您将使用一个示例项目：一个可用于在控制台中阅读Real Python教程的读者软件包。在深入介绍如何发布此软件包之前，您将快速了解该项目。点击[代码链接](https://realpython.com/bonus/pypi-publish-python-package-source-code/)访问包含读者完整源代码的GitHub存储库。


## 了解Python打包

对于新手和老手来说，Python中的打包似乎是复杂而令人困惑的。您会在互联网上找到相互矛盾的建议，曾经被认为是良好实践的方法现在可能会受到批评。

这种情况的主要原因是Python是一种相当古老的编程语言。事实上，Python的第一个版本于1991年发布，早于万维网对一般公众的开放。自然地，在Python的最早版本中并没有包括或者甚至计划包括用于分发软件包的现代化、基于网络的系统。

注意：有关Python打包的详细讨论，请查看《Real Python Podcast》的[第156集](https://realpython.com/podcasts/rpp/156/)。

相反，随着用户需求变得明确和技术提供了新的可能性，Python的打包生态系统在几十年的时间里逐渐发展壮大。第一次打包支持是在2000年秋季发布的，随着distutils库被包含在Python 1.6和2.0中。Python包索引(PyPI)于2003年上线，最初只是一个现有软件包的纯索引，没有任何存储(hosting)功能。

注意：PyPI通常被称为Python奶酪商店，参考了蒙提·派森著名的《奶酪商店》小品。直到今天，[cheeseshop.python.org](http://cheeseshop.python.org/)重定向到PyPI。

在过去的十年里，许多举措改善了打包领域，使其从“西部荒野”变成了一个相当现代和功能齐全的系统。这主要是通过由[Python Packaging Authority(PyPA)](https://www.pypa.io/)工作组审查和实施的[Python Enhancement Proposals(PEPs)](https://peps.python.org/)来完成的。

定义Python打包工作方式的最重要文档是以下PEPs：

* [PEP 427](https://peps.python.org/pep-0427/)描述了如何打包wheels。
* [PEP 440](https://peps.python.org/pep-0440/)描述了如何解析版本号。
* [PEP 508](https://peps.python.org/pep-0508/)描述了如何指定依赖关系。
* [PEP 517](https://peps.python.org/pep-0517/)描述了构建后端应该如何工作。
* [PEP 518](https://peps.python.org/pep-0518/)描述了如何指定构建系统。
* [PEP 621](https://peps.python.org/pep-0621/)描述了项目元数据应该如何编写。
* [PEP 660](https://peps.python.org/pep-0660/)描述了可编辑安装应该如何执行。

您不需要研究这些技术文档。在本教程中，您将学习这些规范如何在实践中结合起来，因为您将通过发布自己的软件包的过程。

要了解Python打包历史的概述，请查看Thomas Kluyver在PyCon UK 2019上的演讲：[Python打包：我们是如何到达这里的，又将何去何从？](https://pyvideo.org/pycon-uk-2019/what-does-pep-517-mean-for-packaging.html)您还可以在[PyPA网站](https://www.pypa.io/en/latest/presentations/)上找到更多演示文稿。

## 创建一个小的Python包

在这一部分，您将了解一个小型的Python包，您可以将其用作可以发布到PyPI的示例。如果您已经有自己要发布的包，请随时略过本节，并在下一节再次加入。

这里将看到的包名为reader。它既可以作为在您自己的代码中下载Real Python教程的库，也可以作为在控制台中阅读教程的应用程序使用。

注：在本节中显示和解释的源代码是Real Python feed reader的简化版本，但是是完全功能的。与当前发布在[PyPI](https://pypi.org/project/realpython-reader/)上的版本相比，此版本缺少一些错误处理和额外选项。

首先，看一下reader的目录结构。该包完全位于一个目录中，该目录可以命名为任何名称。在本例中，它被命名为realpython-reader/。源代码被包含在src/目录中。这并非绝对必要，但通常是一个好主意。

注：在结构化包时使用额外的src/目录在Python社区中已经讨论多年。总的来说，一个平坦的目录结构稍微更容易入门，但是src/结构在项目增长时提供了几个优势。

内部src/reader/目录包含了所有源代码：

```
realpython-reader/
│
├── src/
│   └── reader/
│       ├── __init__.py
│       ├── \_\_main\_\_.py
│       ├── config.toml
│       ├── feed.py
│       └── viewer.py
│
├── tests/
│   ├── test_feed.py
│   └── test_viewer.py
│
├── LICENSE
├── MANIFEST.in
├── README.md
└── pyproject.toml
```


包的源代码位于src/子目录中，与一个配置文件一起。有一些测试位于一个单独的tests/子目录中。本教程不会涵盖测试本身，但您将学习如何处理测试目录。您可以在《使用Python开始测试》和《使用Pytest进行高效的Python测试》中了解更多关于测试的信息。

如果您正在使用自己的包，则可能使用不同的结构或在包目录中有其他文件。Python应用程序布局讨论了几种不同的选项。以下用于发布到PyPI的步骤将独立于您使用的布局。

在本节的其余部分，您将了解reader包的工作原理。在下一节中，您将学习有关特殊文件（如LICENSE、MANIFEST.in、README.md和pyproject.toml）的更多信息，这些文件是发布包所需的。


### 使用Real Python Reader
reader是一个基本的网络订阅阅读器，可以从Real Python订阅中下载最新的Real Python教程。

在本节中，您首先将看到一些关于reader输出的示例。您目前无法自行运行这些示例，但它们应该可以给您一些关于工具如何工作的想法。

注：如果您已经下载了reader的源代码，则可以首先创建一个虚拟环境，然后在该虚拟环境中安装本地包：

```bash
(venv) $ python -m pip install -e .
```

在整个教程中，您将学习到在运行此命令时发生的底层情况。

第一个示例使用reader获取最新文章的列表：

```shell
$ python -m reader
The latest tutorials from Real Python (https://realpython.com/)
  0 How to Publish an Open-Source Python Package to PyPI
  1 The Real Python Podcast – Episode #110
  2 Build a URL Shortener With FastAPI and Python
  3 Using Python Class Constructors
  4 Linear Regression in Python
  5 The Real Python Podcast – Episode #109
  6 pandas GroupBy: Your Guide to Grouping Data in Python
  7 Deploying a Flask Application Using Heroku
  8 Python News: What's New From April 2022
  9 The Real Python Podcast – Episode #108
 10 Top Python Game Engines
 11 Testing Your Code With pytest
 12 Python's min() and max(): Find Smallest and Largest Values
 13 Real Python at PyCon US 2022
 14 Why Is It Important to Close Files in Python?
 15 Combining Data in Pandas With merge(), .join(), and concat()
 16 The Real Python Podcast – Episode #107
 17 Python 3.11 Preview: Task and Exception Groups
 18 Building a Django User Management System
 19 How to Get the Most Out of PyCon US
```

该列表显示最近的教程，因此您看到的列表可能与上面的示例不同。请注意，每篇文章都有编号。要阅读特定教程，您可以使用相同的命令，但也包括教程的编号。

注：Real Python订阅包含文章的有限预览。因此，您无法使用reader阅读完整的教程。

在这种情况下，要阅读《如何将开源Python包发布到PyPI》，您需要在命令中添加0：

```
$ python -m reader 0
How to Publish an Open-Source Python Package to PyPI

Python is famous for coming with batteries included, and many sophisticated
capabilities are available in the standard library. However, to unlock the
full potential of the language, you should also take advantage of the
community contributions at PyPI: the Python Packaging Index.

PyPI, typically pronounced pie-pee-eye, is a repository containing several
hundred thousand packages. These range from trivial Hello, World
implementations to advanced deep learning libraries. In this tutorial,
you'll learn how to upload your own package to PyPI. Publishing your
project is easier than it used to be. Yet, there are still a few
steps involved.

[...]
```

这会使用Markdown格式将文章打印到控制台上。

注：python -m用于执行模块或包。它的工作方式类似于模块和常规脚本的python。例如，python module.py和python -m module在大多数情况下是等效的。

当您使用-m运行一个包时，包中的\_\_main\_\_.py文件将被执行。有关更多信息，请参阅调用阅读器。

目前，您需要从src/目录中运行python -m reader命令。稍后，您将学习如何从任何工作目录运行该命令。

通过更改命令行上的编号，您可以阅读任何可用的教程。

###  理解Reader代码

本教程的目的并不在于了解reader如何工作的细节。但是，如果您有兴趣了解更多关于实现的信息，可以参考[Understand the Reader Code](https://realpython.com/pypi-publish-python-package/#understand-the-reader-code)

### 调用Reader

当您的项目变得更加复杂时，一个挑战是让用户知道他们如何使用您的项目。由于reader由四个不同的源代码文件组成，用户如何知道要执行哪个文件以使用该应用程序呢？

注意：单个Python文件通常被称为脚本或模块。您可以将包视为模块的集合。

通常情况下，您通过提供其文件名来运行Python脚本。例如，如果您有一个名为hello.py的脚本，那么您可以这样运行它：

```shell
$ python hello.py
Hi there!
```

当您运行此假设的脚本时，它会将“Hi there！”打印到控制台。同样，您还可以使用python解释器程序的-m选项，通过指定其模块名称而不是文件名来运行脚本：

```shell
$ python -m hello
Hi there!
```

对于当前目录中的模块，模块名称与文件名相同，只是省略了.py后缀。

使用-m的一个优点是，它允许您调用Python路径中的所有模块，包括内置于Python中的模块。一个示例是调用antigravity：

```shell
$ python -m antigravity
Created new window in existing browser session.
```

如果要在没有-m的情况下运行内置模块，则需要首先查找它在系统中的存储位置，然后使用其完整路径调用它。

使用-m的另一个优点是，它适用于包以及模块。正如您之前学到的，只要reader/目录在您的工作目录中可用，就可以使用-m调用reader包：

```shell
$ cd src/
$ python -m reader
```

因为reader是一个包，名称只是指一个目录。Python如何决定要运行该目录中的哪个代码呢？它会寻找名为\_\_main\_\_.py的文件。如果存在这样的文件，则会执行它。如果不存在，则会打印出错误消息：

```shell
$ python -m urllib
python: No module named urllib.__main__; 'urllib' is a package and
        cannot be directly executed
```


错误消息表示标准库的urllib包没有定义\_\_main\_\_.py文件。

如果您正在创建一个应该被执行的包，则应包含一个\_\_main\_\_.py文件。您也可以遵循Rich的优秀示例，使用python -m rich来演示您的包的功能。

稍后，您将看到如何创建入口点到您的包，这些入口点的行为类似于常规的命令行程序。这将更容易让最终用户使用。

## 准备您的包以发布

您有一个想要发布的包。也许您已经复制了reader，或者您有自己的包。在本节中，您将看到在将包上传到PyPI之前需要采取哪些步骤。

### 给您的包命名

第一个——也可能是最困难的——步骤是为您的包想出一个好的名字。PyPI上的所有包都需要具有唯一的名称。现在在PyPI上有数十万个包，因此您喜欢的名称很可能已经被占用了。

作为一个例子，在PyPI上已经有一个名为reader的包。使包名唯一的一种方法是在名称前添加一个可识别的前缀。在此示例中，您将使用realpython-reader作为reader包的PyPI名称。

无论您为包选择的PyPI名称是什么，这就是您在使用pip安装时要使用的名称：

```shell
$ python -m pip install realpython-reader
```

请注意，PyPI名称不需要与包名称匹配。在这里，包仍然被命名为reader，这就是您在导入包时需要使用的名称：

```python
>>> import reader
>>> reader.__version__
'1.0.0'

>>> from reader import feed
>>> feed.get_titles()
['How to Publish an Open-Source Python Package to PyPI', ...]
```

有时您需要为包使用不同的名称。但是，如果包名和PyPI名称相同，则可以为用户简化事情。

请注意，尽管包名称不需要像PyPI名称那样全局唯一，但它在您运行它的环境中确实需要是唯一的。

例如，如果您安装了两个具有相同包名的包，例如reader和realpython-reader，那么像import reader这样的语句就是模棱两可的。Python通过在导入路径中找到的第一个包来解决此问题。通常情况下，这将是按字母顺序排序时的第一个包。但是，您不应该依赖此行为。

通常情况下，您希望包名称尽可能唯一，同时又要平衡短而简洁的名称的便利性。realpython-reader是一个专门的Feed阅读器，而PyPI上的reader则更加通用。对于本教程，没有理由同时需要这两者，因此与非唯一名称的妥协可能是值得的。

### 配置您的包

为了准备将您的包发布到PyPI，您需要提供一些关于它的信息。一般来说，您需要指定两种类型的信息：

您的构建系统的配置
您的包的配置
构建系统负责创建您将上传到PyPI的实际文件，通常以wheel或源分发（sdist）格式。很长一段时间以来，这是由distutils或setuptools完成的。但是，PEP 517和PEP 518引入了一种指定自定义构建系统的方法。

注：您可以在项目中选择使用哪种构建系统。不同构建系统之间的主要区别在于您如何配置您的包以及要运行哪些命令来构建和上传您的包。

本教程将重点介绍使用setuptools作为构建系统。但是，稍后您将学习如何使用Flit和Poetry等替代方案。

每个Python项目都应使用名为pyproject.toml的文件来指定其构建系统。您可以通过添加以下内容到pyproject.toml中来使用setuptools：

```toml

[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"
```

这指定您正在使用setuptools作为构建系统，以及Python必须安装的依赖项，以便构建您的包。通常情况下，您选择的构建系统的文档将告诉您如何在pyproject.toml中编写build-system表。

您需要提供的更有趣的信息涉及您的包本身。PEP 621定义了如何在pyproject.toml中包含有关包的元数据，以使其在不同的构建系统中尽可能统一。

注意：从历史上看，Setuptools使用setup.py来配置您的包。因为这是一个在安装时运行的实际Python脚本，所以它非常强大，当构建复杂的包时可能仍然需要它。

但是，通常最好使用声明性配置文件来表达如何构建您的包，因为这样更容易推理，并且需要担心的问题更少。使用setup.cfg是配置Setuptools的最常见方法。

然而，Setuptools正朝着使用PEP 621中指定的pyproject.toml的方向发展。在本教程中，您将使用pyproject.toml进行所有包配置。

reader包的一个相当简单的配置可能如下所示：


```toml
# pyproject.toml

[build-system]
requires      = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "realpython-reader"
version = "1.0.0"
description = "Read the latest Real Python tutorials"
readme = "README.md"
authors = [{ name = "Real Python", email = "info@realpython.com" }]
license = { file = "LICENSE" }
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]
keywords = ["feed", "reader", "tutorial"]
dependencies = [
    "feedparser >= 5.2.0",
    "html2text",
    'tomli; python_version < "3.11"',
]
requires-python = ">=3.9"

[project.optional-dependencies]
dev = ["black", "bumpver", "isort", "pip-tools", "pytest"]

[project.urls]
Homepage = "https://github.com/realpython/reader"

[project.scripts]
realpython = "reader.__main__:main"
```



大部分信息都是可选的，而且还有其他设置可以使用，这些设置在本示例中没有包含。查看文档以获取所有细节。

您必须在pyproject.toml中包含的最小信息如下：

* name：指定您的包在PyPI上显示的名称。
* version：设置您的包的当前版本。

正如上面的示例所示，您可以包含更多信息。pyproject.toml中的其他一些键的解释如下：

* classifiers：使用分类器列表描述您的项目。您应该使用这些，因为它们可以使您的项目更易于搜索。
* dependencies：列出您的包对第三方库的任何依赖关系。reader依赖于feedparser、html2text和tomli，因此它们在这里列出。
* project.urls：添加链接，您可以使用这些链接向用户展示有关您的包的其他信息。您可以在这里包含多个链接。
* project.scripts：创建调用包内函数的命令行脚本。在这里，新的realpython命令调用reader.__main__模块中的main()。

project.scripts表是可以处理入口点的三个表之一。您还可以包括project.gui-scripts和project.entry-points，它们分别指定GUI应用程序和插件。

所有这些信息的目的是使您的包在PyPI上更具吸引力和可查找性。查看PyPI上的realpython-reader项目页面，并将其信息与上面的pyproject.toml进行比较：
 
<a>![](/img/pypi/1.png)</a>
*PyPI上有关realpython-reader包的信息*

PyPI上的所有信息都来自pyproject.toml和README.md。例如，版本号基于project.toml中的version = "1.0.0"行，而Read the latest Real Python tutorials则是从description中复制的。

此外，项目描述从您的README.md文件中提取。在侧边栏中，您可以在项目链接部分找到来自project.urls的信息，并在元部分中找到来自许可证和作者的信息。您在分类器中指定的值可在侧边栏底部看到。

有关所有键的详细信息，请参阅[PEP 621](https://peps.python.org/pep-0621/#specification)。在下一小节中，您将了解有关依赖关系以及project.optional-dependencies的更多信息。


### 指定您的包依赖项

您的包可能会依赖于不属于标准库的第三方库。您应该在pyproject.toml的依赖列表中指定这些库。在上面的示例中，您做了以下操作：

```toml
dependencies = [    "feedparser >= 5.2.0",    "html2text",    'tomli; python_version < "3.11"',]
```

这指定了reader依赖于feedparser、html2text和tomli。此外，它表示：

* feedparser必须是5.2.0或更高版本。
* html2text可以是任何版本。
* tomli可以是任何版本，但仅在Python 3.10或更早版本中需要。

这展示了在指定依赖项时可以使用的一些可能性，包括[版本说明符](https://peps.python.org/pep-0508/#specification)和[环境标记](https://peps.python.org/pep-0508/#environment-markers)。您可以使用后者来考虑不同的操作系统、Python版本等。

但是，请注意，您应该努力只指定您的库或应用程序工作所需的最低要求。这个列表将在每次安装您的包时由pip来解析依赖项。通过保持这个列表最小化，您确保您的包尽可能兼容。

您可能听说过应该锁定您的依赖关系。这是一个很好的建议！但是，在这种情况下，它并不适用。您锁定依赖项是为了确保您的环境是可重现的。另一方面，您的包应该希望能够在许多不同的Python环境中工作。

在添加包到依赖项时，您应该遵循以下准则：

* 只列出您的直接依赖关系。例如，reader导入feedparser、html2text和tomli，因此这些都是列出的。另一方面，feedparser依赖于sgmllib3k，但reader不直接使用这个库，因此没有指定。
* 永远不要使用==将您的依赖关系固定到一个特定的版本。
* 如果您依赖于特定版本的功能，请使用>=添加一个下限。
* 如果您担心某个依赖项可能会在主版本升级中破坏兼容性，请使用<添加一个上限。在这种情况下，您应该认真测试这些升级，并在可能的情况下删除或增加上限。

请注意，这些规则适用于配置您要向他人提供的包。如果您要部署您的包，那么您应该在虚拟环境中固定您的依赖项。

pip-tools项目是管理固定依赖项的一个很好的方法。它带有一个pip-compile命令，可以创建或更新完整的依赖项列表。

例如，假设您将reader部署到一个虚拟环境中。然后，您可以使用pip-tools创建一个可重现的环境。事实上，pip-compile可以直接使用您的pyproject.toml文件：

```shell
(venv) $ python -m pip install pip-tools
(venv) $ pip-compile pyproject.toml
feedparser==6.0.8
    via realpython-reader (pyproject.toml)
html2text==2020.1.16
    via realpython-reader (pyproject.toml)
sgmllib3k==1.0.0
    via feedparser
tomli==2.0.1 ; python_version < "3.11"
    via realpython-reader (pyproject.toml)
```

pip-compile创建了一个详细的requirements.txt文件，其中的内容与上面的输出类似。您可以使用pip install或pip-sync将这些依赖项安装到您的环境中：

```shell
(venv) $ pip-sync
Collecting feedparser==6.0.8
  ...
Installing collected packages: sgmllib3k, tomli, html2text, feedparser
Successfully installed feedparser-6.0.8 html2text-2020.1.16 sgmllib3k-1.0.0
                       tomli-2.0.1
```



请参阅[pip-tools文档](https://pip-tools.readthedocs.io/)以获取更多信息。

您还可以在名为project.optional-dependencies的单独表中指定包的可选依赖项。通常，您会在这里指定开发或测试期间使用的依赖项。但是，您还可以指定额外的依赖项，以支持包中某些功能。

在上面的示例中，您包含了以下部分：

```toml
[project.optional-dependencies]
dev = ["black", "bumpver", "isort", "pip-tools", "pytest"]
```

这添加了一个名为dev的可选依赖项组。您可以有多个这样的组，并且可以根据需要命名这些组。

默认情况下，可选依赖项在安装包时不包含在内。但是，通过在运行pip时在方括号中添加组名，您可以手动指定它们应该安装。例如，您可以通过执行以下操作安装reader的额外开发依赖项：

```shell
(venv) $ python -m pip install realpython-reader[dev]
```


您还可以在使用 pip-compile 固定依赖项时使用 --extra 命令行选项来包含可选依赖项：

```shell
(venv) $ pip-compile --extra dev pyproject.toml
attrs==21.4.0
    via pytest
black==22.3.0
    via realpython-reader (pyproject.toml)
...
tomli==2.0.1 ; python_version < "3.11"
    via
      black
      pytest
      realpython-reader (pyproject.toml)
```


这将创建一个包含常规和开发依赖项的固定 requirements.txt 文件


### 添加文档

为您的软件包[添加一些文档](https://realpython.com/documenting-python-code/)是发布前的必要步骤。根据您的项目，您的文档可能只是一个单独的 README 文件，也可能是一个包含教程、示例库和 API 参考的完整网页。

至少，您应该在项目中包含一个 README 文件。一个[好的 README](https://readme.so/) 应该快速描述您的项目，并解释如何安装和使用您的软件包。通常情况下，您希望在 pyproject.toml 中的 readme 键中引用您的 README。这样也会在 PyPI 项目页面上显示这些信息。

您可以使用 [Markdown](https://www.markdownguide.org/basic-syntax) 或 [reStructuredText](http://docutils.sourceforge.net/rst.html) 作为项目描述的格式。PyPI 会根据文件扩展名自动确定您使用的格式。如果您不需要 reStructuredText 的高级功能，那么通常最好使用 Markdown 来编写 README。它更简单，而且在 PyPI 之外有更广泛的支持。

对于更大的项目，您可能希望提供比单个文件能容纳的更多文档。在这种情况下，您可以将文档托管在 [GitHub](https://github.com/) 或 [Read the Docs](https://readthedocs.org/) 等网站上，并从 PyPI 项目页面链接到它。

您可以通过在 pyproject.toml 中的 project.urls 表中指定来链接到其他 URL。在示例中，URLs 部分用于链接到 reader GitHub 存储库。

### 测试你的Package

测试是开发软件包时很有用的，您应该包含它们。正如前面提到的，本教程不涉及测试，但您可以在 tests/ 源代码目录中查看 reader 的测试。

在准备将软件包发布时，您应该考虑测试所起的作用。它们通常只对开发人员有意义，因此不应包含在通过 PyPI 分发的软件包中。

Setuptools 的较新版本在[代码发现](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#automatic-discovery)方面非常出色，通常会将您的源代码包含在软件包分发中，但会排除您的测试、文档和类似的开发工件。

您可以通过在 pyproject.toml 中使用 find 指令来精确控制软件包中包含的内容。有关更多信息，请参阅 [Setuptools 文档](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#custom-discovery)。

### 版本


您的软件包需要有一个版本。此外，PyPI 只允许您一次上传特定版本的软件包。换句话说，如果您想要更新您在 PyPI 上的软件包，那么您需要先增加版本号。这是一个好事，因为它有助于确保可复现性：具有相同版本的两个环境应该表现相同。

有许多不同的[版本控制方案](https://en.wikipedia.org/wiki/Software_versioning)可供选择。对于 Python 项目，PEP 440 给出了一些建议。但是，为了灵活，该 PEP 中的描述比较复杂。对于简单的项目，您应该坚持使用简单的版本控制方案。

[Semantic versioning](https://semver.org/) 是一个很好的默认方案，尽管它并不完美。您将版本指定为三个数字组件，例如 1.2.3。这些组件分别称为 MAJOR、MINOR 和 PATCH。以下是关于何时递增每个组件的建议：

* 当您进行不兼容的 API 更改时，请递增 MAJOR 版本。
* 当您以向后兼容的方式添加功能时，请递增 MINOR 版本。
* 当您进行向后兼容的错误修复时，请递增 PATCH 版本。（来源）

当您递增 MINOR 时，应将 PATCH 重置为 0；当您递增 MAJOR 时，应将 PATCH 和 MINOR 都重置为 0。

[日历版本](https://calver.org/)是语义版本的一种替代方案，正在越来越受欢迎，被像 Ubuntu、Twisted、Black 和 pip 这样的项目所使用。日历版本也由几个数字组件组成，但其中一个或几个与当前年份、月份或周数相关联。

通常情况下，您希望在项目的不同文件中指定版本号。例如，版本号在 pyproject.toml 和 reader/init.py 中都有提及。为了确保版本号保持一致，您可以使用像 [BumpVer](https://pypi.org/project/bumpver/) 这样的工具。

BumpVer 允许您直接在文件中写入版本号，然后根据需要更新这些版本号。例如，您可以安装并集成 BumpVer 到您的项目中：

```shell
(venv) $ python -m pip install bumpver
(venv) $ bumpver init
WARNING - Couldn't parse pyproject.toml: Missing version_pattern
Updated pyproject.toml
```


bumpver init 命令会在您的 pyproject.toml 中创建一个部分，允许您为您的项目配置该工具。根据您的需求，您可能需要更改许多默认设置。对于 reader，您可能最终会得到类似以下内容的配置：

```toml
[tool.bumpver]
current_version = "1.0.0"
version_pattern = "MAJOR.MINOR.PATCH"
commit_message  = "Bump version {old_version} -> {new_version}"
commit          = true
tag             = true
push            = false

[tool.bumpver.file_patterns]
"pyproject.toml" = ['current_version = "{version}"', 'version = "{version}"']
"src/reader/__init__.py" = ["{version}"]
```

要使 BumpVer 正常工作，您必须在 file_patterns 子部分中指定所有包含您的版本号的文件。请注意，BumpVer 与 Git 集成良好，可以在更新版本号时自动提交、标记和推送。

注意：BumpVer 与您的版本控制系统集成。如果您的仓库中存在未提交的更改，它将拒绝更新您的文件。

完成配置后，您可以使用单个命令在所有文件中提升版本。例如，要将 reader 的 MINOR 版本号增加，您可以执行以下操作：

```shell
(venv) $ bumpver update --minor
INFO    - Old Version: 1.0.0
INFO    - New Version: 1.1.0
```


这将在 pyproject.toml 和 init.py 中将版本号从 1.0.0 更改为 1.1.0。您可以使用 --dry 标志查看 BumpVer 将要进行的更改，而不实际执行它们。


### 添加资源文件到您的包

有时，您的包中会有一些不是源代码文件的文件。例如数据文件、二进制文件、文档，以及——就像在本例中一样——配置文件。

为了确保这些文件在构建项目时被包含，您需要使用一个清单文件。对于许多项目，您不需要担心清单：默认情况下，Setuptools会将所有源代码文件和 README 文件包含在构建中。

如果您有其他资源文件并且需要更新清单，则需要在项目的基本目录中创建一个名为 MANIFEST.in 的文件，该文件位于 pyproject.toml 旁边。该文件指定了要包含哪些文件以及要排除哪些文件的规则：

```
# MANIFEST.in

include src/reader/*.toml
```

此示例将包含 src/reader/ 目录中的所有 .toml 文件。实际上，这就是配置文件。

有关设置清单的更多信息，请[参阅文档](https://packaging.python.org/en/latest/guides/using-manifest-in/)。[check-manifest 工具](https://pypi.org/project/check-manifest/)也可以用于处理 MANIFEST.in。

### 为您的包添加许可证

如果您要与他人共享您的包，则需要向您的包添加一个许可证，该许可证说明其他人如何使用您的包。例如，reader 是根据 [MIT 许可证](https://mit-license.org/)分发的。

许可证是法律文件，通常您不希望编写自己的许可证。相反，您应该选择已经可用的许可证之一。

您应该在您的项目中添加一个名为 LICENSE 的文件，其中包含您选择的许可证的文本。然后，您可以在 pyproject.toml 中引用此文件，以便在 PyPI 上显示许可证。

### 本地安装您的包

您已经完成了为您的包进行的所有必要设置和配置。在接下来的部分中，您将了解如何最终将您的包发布到 PyPI。不过，在此之前，您将学习可编辑的安装。这是一种使用 pip 在本地安装您的包的方法，使您在安装后可以编辑您的代码。

注意：通常，pip 执行常规安装，将包放入您的 site-packages/ 文件夹中。如果您安装本地项目，则源代码将复制到 site-packages/ 中。这样做的效果是，稍后对源代码所做的更改将不会生效。您需要首先重新安装您的包。

在开发过程中，这既是无效的，也是令人沮丧的。可编辑的安装通过直接链接到您的源代码来解决这个问题。

可编辑的安装已经在 PEP 660 中得到了规范。当您开发您的包时，这些功能非常有用，因为您可以测试您的包的所有功能并在无需重新安装的情况下更新源代码。

您可以通过添加 -e 或 --editable 标志来使用 pip 在可编辑模式下安装您的包：

```
(venv) $ python -m pip install -e .
```

注意命令末尾的句点（.）。这是命令的必要部分，告诉 pip 您要安装当前工作目录中的包。一般来说，这应该是包含您的 pyproject.toml 文件的目录的路径。

注意：您可能会收到一个错误消息，显示“项目文件具有‘pyproject.toml’，但其构建后端缺少‘build_editable’钩子。”这是由于 Setuptools 对 PEP 660 的支持的限制所致。您可以通过添加一个名为 setup.py 的文件来解决此问题，其内容如下：

```python
# setup.py

from setuptools import setup

setup()
```

此 shim 将可编辑的安装的任务委托给 Setuptools 的传统机制，直到原生支持 PEP 660 为止。

安装成功后，您的项目在您的环境中可用，与当前目录无关。此外，您的脚本已设置好，因此您可以运行它们。回想一下，reader 定义了一个名为 realpython 的脚本：

```shell
(venv) $ realpython
The latest tutorials from Real Python (https://realpython.com/)
  0 How to Publish an Open-Source Python Package to PyPI
  [...]
```
 

您也可以在任何目录中使用 python -m reader，或者从 REPL 或另一个脚本中导入您的包：

```python
>>> from reader import feed
>>> feed.get_titles()
['How to Publish an Open-Source Python Package to PyPI', ...]
```

在开发过程中以可编辑模式安装您的包可以使您的开发体验更加愉快。这也是定位某些错误的好方法，其中您可能在您的当前工作目录中无意中依赖于文件可用性。

这些都需要一些时间，但这些是您需要为您的包做的准备工作。在下一节中，您将学习如何实际发布它！

## 将您的包发布到 PyPI

您的包终于准备好迎接计算机外部的世界了！在本节中，您将学习如何构建您的包并将其上传到 PyPI。

如果您还没有在 PyPI 上注册帐户，那么现在就是注册 PyPI 帐户的时候了。在注册时，您还应该在 TestPyPI 上注册一个帐户。TestPyPI 非常有用！如果您搞砸了，可以尝试发布包的所有步骤，而不会产生任何后果。

要构建并将您的包上传到 PyPI，您将使用两个名为 Build 和 Twine 的工具。您可以像通常一样使用 pip 安装它们：

```shell
(venv) $ python -m pip install build twine
```

您将在接下来的子节中学习如何使用这些工具。

### 构建您的包
在 PyPI 上，包不是以纯源代码的形式分发的。相反，它们被打包成分发包。分发包的最常见格式是源存档和 [Python Wheel](https://realpython.com/python-wheels/)。

注：wheel的名称是指奶酪商店中最重要的物品，即奶酪轮。

源存档包含您的源代码和任何支持文件，这些文件被包装成一个 tar 文件。同样，wheel实质上是一个包含您的代码的 zip 存档。您应该为您的包提供源存档和wheel。对于最终用户来说，wheel通常更快、更方便，而源存档提供了一个灵活的备份选择。

要为您的包创建一个源存档和一个wheel，您可以使用 Build：

```shell
(venv) $ python -m build
[...]
Successfully built realpython-reader-1.0.0.tar.gz and
    realpython_reader-1.0.0-py3-none-any.whl
```

正如输出中的最后一行所说，这将创建一个源存档和一个wheel。您可以在一个新创建的 dist 目录中找到它们：

```
realpython-reader/
│
└── dist/
    ├── realpython_reader-1.0.0-py3-none-any.whl
    └── realpython-reader-1.0.0.tar.gz
```

.tar.gz 文件是您的源存档，而 .whl 文件是您的wheel。这些是您将上传到 PyPI 并在以后安装您的包时 pip 下载的文件。

### 确认您的包构建

在上传您新构建的分发包之前，您应该检查它们是否包含您期望的文件。wheel文件实际上是一个带有不同扩展名的 ZIP 文件。您可以解压缩它并检查其内容，如下所示：

```shell
(venv) $ cd dist/
(venv) $ unzip realpython_reader-1.0.0-py3-none-any.whl -d reader-whl
(venv) $ tree reader-whl/
reader-whl/
├── reader
│   ├── config.toml
│   ├── feed.py
│   ├── __init__.py
│   ├── __main__.py
│   └── viewer.py
└── realpython_reader-1.0.0.dist-info
    ├── entry_points.txt
    ├── LICENSE
    ├── METADATA
    ├── RECORD
    ├── top_level.txt
    └── WHEEL

2 directories, 11 files
```

首先将wheel文件重命名为具有 .zip 扩展名，以便您可以扩展它。

您应该看到列出了所有您的源代码，以及一些新创建的包含您在 pyproject.toml 中提供的信息的文件。特别是，请确保所有子包和支持文件，如 config.toml，都已包含在内。

您还可以查看源存档的内容，因为它被打包成一个 [tar 存档](https://en.wikipedia.org/wiki/Tar_(computing))。然而，如果您的wheel包含您期望的文件，那么源存档也应该是可以的。

Twine 还可以检查您的包描述在 PyPI 上是否正确渲染。您可以在 dist 中创建的文件上运行 twine check：

```shell
(venv) $ twine check dist/*
Checking distribution dist/realpython_reader-1.0.0-py3-none-any.whl: Passed
Checking distribution dist/realpython-reader-1.0.0.tar.gz: Passed
```


这不会捕捉到您可能遇到的所有问题，但它是一个很好的第一道防线。

### 上传您的包
现在您准备好将您的包实际上传到 PyPI 了。为此，您将再次使用 Twine 工具，告诉它上传您构建的分发包。

首先，您应该上传到 TestPyPI，以确保一切正常：

```
(venv) $ twine upload -r testpypi dist/*
```

Twine 会要求您输入用户名和密码。

【译注：现在pypi/testpypi不允许通过用户名和密码上传了，请在[这里](https://test.pypi.org/manage/account/token/)增加一个token，然后在$HOME/.pypirc增加：

```
[testpypi]
  username = __token__
  password = pypi-.....
```
】


注意：如果您按照使用 reader 包作为示例的教程，则前面的命令可能会失败，并显示一条消息，指出您无权上传到 realpython-reader 项目。

您可以将 pyproject.toml 中的名称更改为一些唯一的内容，例如 test-<your-username>。然后重新构建项目，并将新构建的文件上传到 TestPyPI。

如果上传成功，那么您可以快速转到 [TestPyPI](https://test.pypi.org/)，向下滚动，并查看您的项目在新发布中自豪地显示！点击您的包，并确保一切看起来都没问题。

如果您一直在使用 reader 包进行操作，那么教程到此结束！虽然您可以随心所欲地使用 TestPyPI，但您不应该仅仅为了测试而向 PyPI 上传示例包。

注意：TestPyPI 非常适用于检查您的包是否正确上传，并且您的项目页面是否与您预期的一样。您还可以尝试从 TestPyPI 安装您的包：

```shell
(venv) $ python -m pip install -i https://test.pypi.org/simple realpython-reader
```

但是，请注意，这可能会失败，因为并非所有依赖项都在 TestPyPI 上可用。这不是问题。当您将其上传到 PyPI 时，您的包仍然应该可以工作。

如果您有自己的要发布的包，那么现在终于到达了这一时刻！随着所有准备工作的完成，这最后一步非常简短：

```shell
(venv) $ twine upload dist/*
```

在请求时提供用户名和密码。就是这样！

转到 PyPI 并查找您的包。您可以通过搜索、查看“您的项目”页面或直接转到您的项目的 URL：pypi.org/project/your-package-name/。

恭喜！您的包已发布在 PyPI 上！

### 安装您的包

花点时间沉浸在 PyPI 网页的蓝色光芒中，并向您的朋友炫耀。

然后再次打开终端。还有一个更大的回报！

将您的包上传到 PyPI 后，您也可以使用 pip 安装它。首先，创建一个新的虚拟环境并激活它。然后运行以下命令：

```shell
(venv) $ python -m pip install your-package-name
```

将 your-package-name 替换为您为包选择的名称。例如，要安装 reader 包，您应该执行以下操作：

```shell
(venv) $ python -m pip install realpython-reader
```

看到您自己的代码像任何其他第三方库一样被 pip 安装，这是一种美妙的感觉！

## 探索其他构建系统

在本教程中，您使用了 Setuptools 构建您的包。Setuptools 是创建包的长期标准，它有利有弊。虽然它广泛使用且受信任，但它也具有许多可能与您无关的功能。

与 Setuptools 相比，您可以使用几种替代构建系统。在过去几年中，Python 社区已经完成了标准化 Python 打包生态系统的重要工作。这使得在不同的构建系统之间移动并使用最适合您的工作流程和包的构建系统变得更加简单。

在本节中，您将简要了解两种替代构建系统，您可以使用它们来创建和发布您的 Python 包。除了下面将要介绍的 Flit 和 Poetry 之外，您还可以查看 pbr、enscons 和 Hatchling。此外，pep517 包提供了支持创建您自己构建系统的功能。

### Flit

[Flit](https://flit.pypa.io/) 是一个非常棒的项目，旨在在打包时“使易事易”（来源）。Flit 不支持创建 C 扩展等高级包，一般情况下，它在设置包时不会给您太多选择。相反，Flit 订阅了这样一个哲学观念，即发布包应该有一个明显的工作流程。

注意：您不能同时配置 Setuptools 和 Flit。为了在本节中测试工作流程，您应该安全地将 Setuptools 配置存储在您的版本控制系统中，然后删除 pyproject.toml 中的 build-system 和 project 部分。

首先使用 pip 安装 Flit：

```shell
(venv) $ python -m pip install flit
```

尽可能多地，Flit 自动化了您需要进行的包准备工作。要开始配置一个新的包，请运行 flit init：

```shell
(venv) $ flit init
Module name [reader]:
Author: Real Python
Author email: info@realpython.com
Home page: https://github.com/realpython/reader
Choose a license (see http://choosealicense.com/ for more info)
1. MIT - simple and permissive
2. Apache - explicitly grants patent rights
3. GPL - ensures that code based on this is shared with the same terms
4. Skip - choose a license later
Enter 1-4: 1

Written pyproject.toml; edit that file to add optional extra info.
```

flit init 命令将根据您对几个问题的回答创建一个 pyproject.toml 文件。您可能需要稍微编辑此文件以在使用它之前进行配置。对于 reader 项目，Flit 的 pyproject.toml 文件最终如下所示：

```toml
# pyproject.toml

[build-system]
requires = ["flit_core >=3.2,<4"]
build-backend = "flit_core.buildapi"

[project]
name = "realpython-reader"
authors = [{ name = "Real Python", email = "info@realpython.com" }]
readme = "README.md"
license = { file = "LICENSE" }
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]
dynamic = ["version", "description"]

[project.urls]
Home = "https://github.com/realpython/reader"

[project.scripts]
realpython = "reader.__main__:main"
```

 
【译注：由于包名是reader，而pypi的名字是realpython-reader，我们还需要在pyproject.toml里增加：

```toml
[tool.flit.module]
name = "reader"
```
否则会出现ValueError: No file/folder found for module realpython_reader。

】

请注意，大多数项目条目与您原始的 pyproject.toml 相同。但是，有一个区别，即版本和描述是在一个动态字段中指定的。Flit 实际上通过使用 version 和在 init.py 文件中定义的文档字符串来自行确定这些内容。Flit 的文档解释了关于 pyproject.toml 文件的一切。

Flit 可以构建您的包并将其发布到 PyPI。您不需要使用 Build 和 Twine。要构建您的包，只需执行以下操作：

```shell
(venv) $ flit build
```

这将创建一个源存档和一个wheel，类似于您之前使用 python -m build 所做的操作。如果您愿意，您也可以仍然使用 Build。

要将您的包上传到 PyPI，您可以像之前一样使用 Twine。但是，您也可以直接使用 Flit：

```shell
(venv) $ flit publish
```

发布命令将在必要时构建您的包，然后将文件上传到 PyPI，并提示您输入用户名和密码。

要看到 Flit 的早期但可识别的版本的实际操作，请查看 Thomas Kluyver 在 EuroSciPy 2017 中的[闪电演讲](https://www.youtube.com/watch?v=qTgk2DUM6G0&t=11m50s)。演示展示了如何配置您的包、构建它并将其发布到 PyPI 的过程，这在两分钟内完成。


### Poetry

Poetry 是另一个可以用来构建和上传您的包的工具。与 Flit 相比，Poetry 具有更多功能，可以在开发包时帮助您，包括强大的依赖管理功能。

在使用 Poetry 之前，您需要安装它。可以使用 pip 安装 Poetry。然而，维护者建议您使用自定义安装脚本以避免潜在的依赖冲突。请参阅文档以获取说明。

注意：您不能同时使用 Setuptools 和 Poetry 配置您的包。要测试本节中的工作流程，您应该将 Setuptools 配置安全地存储在您的版本控制系统中，然后删除 pyproject.toml 中的 build-system 和 project 部分。

安装了 Poetry 后，您可以使用一个 init 命令开始使用它，与 Flit 类似：

```shell
(venv) $ poetry init

This command will guide you through creating your pyproject.toml config.

Package name [code]: realpython-reader
Version [0.1.0]: 1.0.0
Description []: Read the latest Real Python tutorials
...
```

这将根据您对包的相关问题的回答创建一个 pyproject.toml 文件。

注意：目前 Poetry 不支持 PEP 621，因此 pyproject.toml 中的实际规范在 Poetry 和其他工具之间有所不同。

【译注：建议使用[PDM](https://github.com/pdm-project/pdm)，它支持PEP621，并且还支持类似pnpm的cache。这对于需要在每个项目中安装pytorch这样超级大的包非常方便。】

对于 Poetry，pyproject.toml 文件最终如下所示：

```toml
# pyproject.toml

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "realpython-reader"
version = "1.0.0"
description = "Read the latest Real Python tutorials"
authors = ["Real Python <info@realpython.com>"]
readme = "README.md"
homepage = "https://github.com/realpython/reader"
license = "MIT"

[tool.poetry.dependencies]
python = ">=3.9"
feedparser = "^6.0.8"
html2text = "^2020.1.16"
tomli = "^2.0.1"

[tool.poetry.scripts]
realpython = "reader.__main__:main"
```

【译注，由于pypi的名字和package不同，我们需要在[tool.poetry]表里增加：

```toml
[tool.poetry]
packages = [
    { include = "src/reader" },
]

```
】
尽管部分名称可能有所不同，但您应该能够从前面讨论的 pyproject.toml 中认出所有这些项目。

需要注意的一点是，Poetry 将根据您指定的许可证和 Python 版本自动添加分类器。Poetry 还要求您明确指定依赖项的版本。实际上，依赖管理是 Poetry 的一个强项。

与 Flit 类似，Poetry 可以构建并将包上传到 PyPI。build 命令将创建一个源存档和一个轮子：

```shell
(venv) $ poetry build
```

这将在 dist 子目录中创建两个常规文件，您可以像之前一样使用 Twine 上传。您还可以使用 Poetry 直接发布到 PyPI：

```shell
(venv) $ poetry publish
```

【译注：如前面所说，pypi/testpypi目前只支持token发布，请参考[这个so问题](https://stackoverflow.com/questions/68882603/using-python-poetry-to-publish-to-test-pypi-org)配置token.】

这将上传您的包到 PyPI。除了帮助构建和发布外，Poetry 还可以在过程的早期阶段帮助您。Poetry 可以使用 new 命令帮助您启动一个新项目。它还支持使用虚拟环境。请参阅 Poetry 的文档以获取所有详细信息。

除了略有不同的配置文件外，Flit 和 Poetry 的工作方式非常相似。Poetry 的范围更广，因为它还旨在帮助管理依赖关系，而 Flit 已经存在了一段时间。

## 结论

您现在知道如何准备您的项目并将其上传到 PyPI，以便其他人安装和使用。虽然您需要经历一些步骤，但在 PyPI 上看到自己的包是一个很好的回报。更好的是，让其他人发现您的项目有用！

在本教程中，您学会了如何通过以下步骤发布您自己的包：

* 为您的包找到一个好的名称
* 使用 pyproject.toml 配置您的包
* 构建您的包
* 将您的包上传到 PyPI

此外，您还了解了 Python 打包社区的倡议，以标准化工具和流程。

如果您仍有疑问，请随时在下面的评论部分提问。此外，Python 打包用户指南提供了比本教程更详细的信息。

