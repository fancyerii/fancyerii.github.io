---
layout:     post
title:      "使用PDM来管理Python项目" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - pip
    - PDM
---

使用Pip+venv来管理pyton项目会有很多问题，使用conda也不尽如人意。最近尝试了很多Python包管理工具，包括[Pipenv](https://pipenv.pypa.io/en/latest/)、[Poetry](https://python-poetry.org/)和[PDM](https://pdm-project.org/latest/)。通过一番尝试之后发现PDM最符合我的需求，因此记录一下。部分内容来自[Pipenv教程](/2023/12/27/pipenv-tutor)。

<!--more-->

**目录**
* TOC
{:toc}

## 概述

### 使用requirements.txt管理依赖的问题

想象一下，你正在开发一个使用第三方软件包（如flask）的Python项目。你需要指定这个依赖，以便其他开发人员和自动化系统能够运行你的应用程序。
于是，你决定在一个requirements.txt文件中包含flask的依赖项：

```
flask
```
太好了，一切在本地都正常运行，而在对应用程序进行了一番修改之后，你决定将其部署到生产环境。这就是事情变得有点棘手的地方...

上述的requirements.txt文件没有指定使用flask的哪个版本。在这种情况下，通过pip install -r requirements.txt将默认安装最新版本。这是可以的，除非最新版本中有接口或行为更改，导致我们的应用程序出现问题。

举个例子，假设flask的作者发布了一个新版本的flask。然而，它与你在开发过程中使用的版本不兼容。

现在，假设你将应用程序部署到生产环境并运行pip install -r requirements.txt。Pip获取了最新的不向后兼容的flask版本，就这样，你的应用程序在生产环境中崩溃了。

“但嘿，它在我的机器上工作！”——我曾经也有过这种经历，感觉并不好。

在这一点上，你知道在开发过程中使用的flask版本运行良好。因此，为了解决问题，你尝试在requirements.txt中更加具体。你为flask依赖项添加了一个版本说明符。这也被称为锁定依赖项：

```
flask==0.12.1
```
将flask依赖项固定到特定版本可以确保pip install -r requirements.txt设置了在开发过程中使用的确切flask版本。但它真的能做到吗？

请记住，flask本身也有依赖关系（pip会自动安装），但flask本身不为其依赖项指定确切版本。例如，它允许任何版本的Werkzeug>=0.14。

同样，为了这个例子，假设发布了一个新版本的Werkzeug，但它为你的应用程序引入了一个致命错误。

当你在生产环境中执行pip install -r requirements.txt时，这一次你将得到flask==0.12.1，因为你已经锁定了这个依赖关系。然而，不幸的是，你将获得Werkzeug的最新且有缺陷的版本。再次，产品在生产环境中崩溃。

真正的问题在于构建不是确定性的。我的意思是，给定相同的输入（requirements.txt文件），pip并不总是产生相同的环境。目前，你无法轻松地在生产环境中复制开发机器上的确切环境。

解决这个问题的典型方法是使用pip freeze。此命令允许你获取当前安装的所有第三方库的确切版本，包括pip自动安装的子依赖项。因此，你可以在开发过程中冻结一切，以确保在生产环境中拥有相同的环境。

执行pip freeze会生成你可以添加到requirements.txt的锁定依赖项：

```
click==6.7
Flask==0.12.1
itsdangerous==0.24
Jinja2==2.10
MarkupSafe==1.0
Werkzeug==0.14.1
```

通过这些锁定的依赖项，你可以确保在生产环境中安装的软件包与开发环境中的完全匹配，以确保产品不会意外中断。然而，这个“解决方案”不幸地导致了一整套新的问题。

现在，你已经指定了每个第三方软件包的确切版本，即使它们是flask的子依赖项，你也需要负责保持这些版本的更新。如果在Werkzeug==0.14.1中发现了一个安全漏洞，并且软件包维护人员立即在Werkzeug==0.14.2中修补了该漏洞，那怎么办呢？你确实需要更新到Werkzeug==0.14.2，以避免由于较早版本的Werkzeug中存在的未修补漏洞而引起的任何安全问题。

首先，你需要意识到你拥有的版本存在问题。然后，在有人利用安全漏洞之前，你需要在生产环境中获取新版本。因此，你必须手动更改requirements.txt以指定新版本Werkzeug==0.14.2。正如在这种情况下所看到的，保持必要更新的责任落在你身上。

事实上，你并不真的关心Werkzeug的哪个版本被安装，只要它不破坏你的代码。实际上，你可能希望使用最新版本，以确保获得错误修复、安全补丁、新功能、更多优化等。

真正的问题是：“如何在不负责更新子依赖项版本的情况下，为你的Python项目实现确定性构建？”

后面我们会介绍Pipenv是如果使得构建是确定性的。

### 处理具有不同依赖关系的项目

让我们稍微改变一下方向，谈谈在处理多个项目时经常出现的另一个常见问题。想象一下，ProjectA需要django==1.9，但ProjectB需要django==1.10。

默认情况下，Python试图将所有第三方软件包存储在系统范围的位置。这意味着每次你想在ProjectA和ProjectB之间切换时，你都必须确保安装了正确版本的django。这使得在项目之间切换变得痛苦，因为你必须卸载和重新安装软件包以满足每个项目的要求。

标准解决方案是使用一个具有自己的Python可执行文件和第三方软件包存储的虚拟环境。这样，ProjectA和ProjectB被充分分隔开。现在，你可以轻松地在项目之间切换，因为它们不共享相同的软件包存储位置。PackageA可以在自己的环境中拥有它需要的django版本，而PackageB可以完全独立地拥有它所需的版本。这方面一个非常常见的工具是virtualenv（或Python 3中的venv）。

### 依赖关系解决

我所说的依赖关系解决是什么意思呢？假设你有一个类似于以下内容的requirements.txt文件：

```
package_a
package_b
```

假设package_a具有一个子依赖项package_c，并且package_a需要该包的特定版本：package_c>=1.0。反过来，package_b具有相同的子依赖项，但需要package_c<=2.0。

理想情况下，当你尝试安装package_a和package_b时，安装工具会查看package_c的要求（>=1.0且<=2.0），并选择满足这些要求的版本。你希望该工具解决依赖关系，以便最终使你的程序正常工作。这就是我所说的“依赖关系解决”。

不幸的是，目前pip本身没有真正的依赖关系解决功能，但有一个开放的问题来支持它。pip处理上述情况的方式如下：

* 它安装package_a并寻找满足第一个要求（package_c>=1.0）的package_c的版本。
* 然后，pip安装最新版本的package_c以满足该要求。假设package_c的最新版本是3.1。



如果pip选择的package_c版本不符合将来的要求（例如，package_b需要package_c<=2.0），安装将失败。

对于这个问题的“解决方案”是在requirements.txt文件中指定子依赖项（package_c）所需的范围。这样，pip可以解决这个冲突并安装符合这些要求的软件包：

```shell
package_c>=1.0,<=2.0
package_a
package_b
```

然而，就像以前一样，现在你直接关心子依赖项（package_c）。这个问题的问题在于，如果package_a在不通知你的情况下更改了它的要求，那么你指定的要求（package_c>=1.0,<=2.0）可能不再可接受，安装可能会再次失败。真正的问题是，你再次负责保持子依赖项的要求的最新状态。

理想情况下，你的安装工具应该足够智能，以安装满足所有要求的软件包，而无需显式指定子依赖项的版本。



### 与Maven/Gradle对比


Maven和Gradle都是基于项目的管理方式，而不是像venv和conda基于环境。这样的隔离会更加彻底，因为如果多个项目共用一个环境的话，到了某一天它们引用的同一个包发生冲突就不好处理了。每个项目都有自己的环境，就不会跟别人冲突，但是可能会造成磁盘空间的浪费，因为可能很多项目的一些基础依赖是相同的。 在这一点来看，Java里maven的依赖是放在统一的位置，如果多个项目依赖同一个版本，那么就不会重复下载和存储。

这就是pip和maven最大的区别：pip是基于一个虚拟环境的，而maven是基于项目的。另外一个区别就是maven中的依赖必须精确制定版本号，这类似于"pip install abc==1.2.3"。这样的好处是整个构建是确定的，每个发布出去的jar包是永远不会改变的。同样的代码，保证能够精准复现编译过程。但是pip一开始大家都是不习惯指定版本号，后来发现有问题，才慢慢加入版本号，但是通常也很少指定精确的版本号(==1.2.3)。而maven习惯指定精确的版本号，即使maven也有Version Range的功能，比如：

```java
        <dependency>
            <groupId>org.checkerframework</groupId>
            <artifactId>checker-qual</artifactId>
            <version>[3,4)</version>
        </dependency>
```

它等价于">=3,<4"。

使用精确版本号的好处是严格的可重复，这对于软件的稳定性至关重要。当然缺点就是很难升级，因为定期频繁的小部分升级是容易的，但是要把一个几年都没动过的包升级是很难的。 

## 为什么选择PDM？

### 基于项目的管理

根据前面的介绍，我们需要基于项目的虚拟环境而不是很多项目共用一个虚拟环境。根据这个需求，我们就会排查conda/miniconda和venv(或者virtualenv)。

### 自动依赖管理和冲突解决

对于这个需求，Pipenv、Poetry和PDM都可以满足需求。

### 节约空间

因为我的工作是做深度学习相关开发，它有如下特点：

* 基础框架如Pytorch和Tensorflow非常大，比如Pytorch安装后需要4GB，因为它把CUDA和CUDNN都打包进去了。
* 第三方工具升级迭代快，原因是这个领域变化很快，很多工具也不是太成熟。
* 代码兼容性差，很多开源代码来自学术界，工程做得不好，换一个python版本可能就不能运行。

因此我需要经常尝试很多开源代码，这些代码通常依赖特定(范围的)版本的Pytorch或者numpy甚至python的版本。所以最好的办法就是每一个项目使用一个隔离的环境。但这就好带来一个问题：项目几十上百个，每个项目都依赖某些版本的Pytorch/Tensorflow。有很多项目的Pytorch/Tensorflow是可以用相同的版本，但是也有很多不行。如果每个环境都需要安装pytorch，那么使用的空间非常浪费。

根据这个需求，目前只有PDM实现了类似[pnpm](https://pnpm.io/motivation#saving-disk-space-and-boosting-installation-speed)的机制。默认情况下PDM会在当前项目下创建一个virtualenv环境(.venv目录)，不同的项目不共享。但是我们可以通过它的cache机制，把真正的package安装到中央的某个共享位置，而每个项目通过符号链接链接到共享目录，这样就可以节省大量空间。

### 其它优势

除此之外，根据文档，PDM还有如下优点：

* 简单快速的依赖解析器(dependency resolver)，主要用于大型二进制发行版
* [PEP 517](https://www.python.org/dev/peps/pep-0517) 构建后端(backend)
* [PEP 621](https://www.python.org/dev/peps/pep-0621) 项目元数据(metadata)
* 灵活而强大的插件系统
* 多功能的用户脚本

## PDM教程

### 新建项目

首先我们需要创建一个项目目录并且用pdm init初始化：

```shell
mkdir my-project && cd my-project
pdm init
```

#### 选择Python Interpreter

执行pdm init后第一步就需要选择Python Interpreter。它会列出检测到的python版本，我们可以根据自己的情况选择。这个版本信息会保存到.pdm-python。后面我们可以通过pdm use来修改它。

#### 是否使用Virtualenv

选择了interpreter之后PDM会询问是否创建一个虚拟环境，如果是(默认创建)，则会在当前目录(my-project)创建.venv目录用于创建虚拟环境。项目后面就会使用虚拟环境里的Python，并且安装的第三方包也会放到.venv/lib/pythonXXX/site-packages/里。

如果上一步选择的Python Interpreter本身就是虚拟环境里的，那么PDM就不会创建新的虚拟环境，而复用这个环境，并且安装的包也会装到这个环境里。个人建议不要这么做！

如果上面两种情况都不是(选择的Python Interpreter不是虚拟环境里的，而且选择不创建新的虚拟环境)，那么会创建__pypackages__目录，并且安装的包都会放到这个目录。这种安装方法是[PEP 582](https://www.python.org/dev/peps/pep-0582/)的规范，但是这个规范被Rejected了，所以强烈建议使用虚拟环境！

#### Library还是Application

如果我们的项目会发布出去被别的开发者使用，那么就应该选择Library。后续我们通常会把它发布到[PyPI](https://pypi.org/)。

在PDM中，如果你选择创建一个库，PDM将会在pyproject.toml文件中添加一个名称、版本字段，以及一个[build-system]表格用于构建后端，这只有在你的项目需要构建和分发时才有用。因此，如果你想把项目从应用程序更改为库，你需要手动添加这些字段到pyproject.toml。另外，当你运行pdm install或pdm sync时，库项目将会被安装到环境中，除非指定了\-\-no-self。

#### 设置requires-python

你需要为你的项目设置一个合适的 requires-python 值。这是一个影响依赖解析方式的重要属性。基本上，每个包的 requires-python 必须覆盖项目的 requires-python 范围。例如，考虑以下设置：

项目: requires-python = ">=3.9"
包 foo: requires-python = ">=3.7,<3.11"

解析依赖关系将导致无法解决的情况：

```
Unable to find a resolution because the following dependencies don't work
on all Python versions defined by the project's `requires-python`
```
因为依赖的 requires-python 是 >=3.7,<3.11，它不覆盖项目的 requires-python 范围，即 >=3.9。换句话说，项目承诺在 Python 3.9、3.10、3.11（以及更高版本）上工作，但依赖关系不支持 Python 3.11（或任何更高版本）。由于 PDM 创建了一个跨平台的锁定文件，应该在 requires-python 范围内的所有 Python 版本上工作，它找不到有效的解决方案。要解决此问题，你需要为 requires-python 添加一个最大版本，如 >=3.9,<3.11。

requires-python 的值是在 [PEP 440](https://peps.python.org/pep-0440/#version-specifiers) 中定义的版本说明符。以下是一些示例：

requires-python |含义
--|--
=3.7 | Python 3.7 及以上
=3.7,<3.11 | Python 3.7、3.8、3.9 和 3.10
=3.6,!=3.8.,!=3.9. | Python 3.6 及以上，但不包括 3.8 和 3.9

#### 与较旧的 Python 版本一起工作

尽管 PDM 在 Python 3.8 及以上版本上运行，但你仍然可以在你的工作项目中使用较低的 Python 版本。但请记住，如果你的项目是一个需要构建、发布或安装的库，你要确保所使用的 PEP 517 构建后端支持你所需的最低 Python 版本。例如，默认后端 pdm-backend 只在 Python 3.7+ 上工作，因此如果你在 Python 3.6 的项目上运行 pdm build，你将会收到一个错误。大多数现代构建后端已经放弃了对 Python 3.6 及更低版本的支持，因此强烈建议将 Python 版本升级到 3.7+。以下是一些常用构建后端的支持的 Python 范围，我们只列出支持 PEP 621 的后端，否则 PDM 无法与它们一起工作。

Backend	| Supported Python	| Support PEP 621
--|--|--
pdm-backend |	>=3.7 |	Yes
setuptools>=60 |	>=3.7 |	Experimental
hatchling |	>=3.7 |	Yes
flit-core>=3.4 |	>=3.6 |	Yes
flit-core>=3.2,<3.4 |	>=3.4 |	Yes

请注意，如果你的项目是一个应用程序（即没有名称元数据），则上述后端的限制不适用。因此，如果你不需要构建后端，你可以使用任何 Python 版本 >=2.7。


#### 导入来自其他包管理器的项目

如果你已经在使用其他包管理工具，如 Pipenv 或 Poetry，那么迁移到 PDM 就很容易了。PDM 提供了导入命令，因此你不必手动初始化项目，它现在支持：

* Pipenv 的 Pipfile
* Poetry 的 pyproject.toml 中的部分
* Flit 的 pyproject.toml 中的部分
* pip 使用的 requirements.txt 格式
* setuptools 的 setup.py（它需要在项目环境中安装 setuptools。你可以通过为 venv 配置 venv.with_pip 为 true，并为 pypackages 添加 setuptools 来实现）

另外，在执行 pdm init 或 pdm install 时，如果你的 PDM 项目尚未初始化，PDM 可以自动检测可能要导入的文件。

转换 setup.py 将使用项目解释器执行文件。确保 setuptools 已经与解释器安装，并且 setup.py 可信。

#### 与版本控制的工作

必须提交 pyproject.toml 文件。你应该提交 pdm.lock 和 pdm.toml 文件。不要提交 .pdm-python 文件。

必须提交 pyproject.toml 文件，因为它包含了 PDM 所需的项目构建元数据和依赖项。它也常被其他 Python 工具用于配置。在 Pip 文档中了解更多关于 pyproject.toml 文件的信息。

你应该提交 pdm.lock 文件，这样做可以确保所有安装程序都使用相同的依赖版本。了解如何更新依赖项，请参阅更新现有依赖项。

pdm.toml 包含一些项目范围的配置，可能有助于共享。

.pdm-python 存储了当前项目使用的 Python 路径，不需要共享。

#### 显示当前 Python 环境

```
$ pdm info
PDM version:
  2.0.0
Python Interpreter:
  /opt/homebrew/opt/python@3.9/bin/python3.9 (3.9)
Project Root:
  /Users/fming/wkspace/github/test-pdm
Project Packages:
  /Users/fming/wkspace/github/test-pdm/__pypackages__/3.9

# Show environment info
$ pdm info --env
{
  "implementation_name": "cpython",
  "implementation_version": "3.8.0",
  "os_name": "nt",
  "platform_machine": "AMD64",
  "platform_release": "10",
  "platform_system": "Windows",
  "platform_version": "10.0.18362",
  "python_full_version": "3.8.0",
  "platform_python_implementation": "CPython",
  "python_version": "3.8",
  "sys_platform": "win32"
}
```

这个命令用于检查项目正在使用哪种模式：

* 如果项目包是 None，虚拟环境模式已启用。
* 否则，PEP 582 模式已启用。

现在，你已经设置了一个新的 PDM 项目并获得了一个 pyproject.toml 文件。参考[元数据部分](https://pdm-project.org/latest/reference/pep621/)了解如何正确编写 pyproject.toml。


###  管理依赖关系

PDM 提供了一系列有用的命令来帮助管理你的项目和依赖关系。以下示例在 Ubuntu 18.04 上运行，如果你使用的是 Windows，可能需要进行一些更改。

#### 添加依赖项

pdm add 可以跟随一个或多个依赖项，依赖项的规范描述在 [PEP 508](https://www.python.org/dev/peps/pep-0508/) 中。

示例：

```shell
pdm add requests   # add requests
pdm add requests==2.25.1   # add requests with version constraint
pdm add requests[socks]   # add requests with extra dependency
pdm add "flask>=1.0" flask-sqlalchemy   # add multiple dependencies with different specifiers
```

PDM 还允许通过提供 -G/\-\-group \<name> 选项来创建额外的依赖组，这些依赖将分别放在项目文件中的 [project.optional-dependencies.\<name>] 表中。

你可以在上传包之前，在可选依赖项中引用其他可选组：

```toml
[project]
name = "foo"
version = "0.1.0"

[project.optional-dependencies]
socks = ["pysocks"]
jwt = ["pyjwt"]
all = ["foo[socks,jwt]"]
```

##### 本地依赖项

可以使用其路径添加本地包。路径可以是文件或目录：

```shell
pdm add ./sub-package
pdm add ./first-1.0.0-py2.py3-none-any.whl
```

路径必须以 . 开头，否则它将被识别为普通的命名要求。本地依赖项将以 URL 格式写入到 pyproject.toml 文件中：

```toml
[project]
dependencies = [
    "sub-package @ file:///${PROJECT_ROOT}/sub-package",
    "first @ file:///${PROJECT_ROOT}/first-1.0.0-py2.py3-none-any.whl",
]
```
 

如果你使用的是 hatchling 而不是 pdm 后端，URL 将如下所示：

```
sub-package @ {root:uri}/sub-package
first @ {root:uri}/first-1.0.0-py2.py3-none-any.whl
```
其他后端不支持在 URL 中编码相对路径，而是会写入绝对路径。

##### URL 依赖项

PDM 还支持直接从网络地址下载和安装软件包。

示例：

```shell
# Install gzipped package from a plain URL
pdm add "https://github.com/numpy/numpy/releases/download/v1.20.0/numpy-1.20.0.tar.gz"
# Install wheel from a plain URL
pdm add "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.5.0/en_core_web_trf-3.5.0-py3-none-any.whl"
```


##### 版本控制系统依赖项

你也可以从 git 仓库 URL 或其他版本控制系统安装。支持以下版本控制系统：

* Git：git
* Mercurial：hg
* Subversion：svn
* Bazaar：bzr

URL 应该是这样的：{vcs}+{url}@{rev}

示例：

```shell
# Install pip repo on tag `22.0`
pdm add "git+https://github.com/pypa/pip.git@22.0"
# Provide credentials in the URL
pdm add "git+https://username:password@github.com/username/private-repo.git@master"
# Give a name to the dependency
pdm add "pip @ git+https://github.com/pypa/pip.git@22.0"
# Or use the #egg fragment
pdm add "git+https://github.com/pypa/pip.git@22.0#egg=pip"
# Install from a subdirectory
pdm add "git+https://github.com/owner/repo.git@master#egg=pkg&subdirectory=subpackage"
```

##### 隐藏 URL 中的凭据(credentials)

你可以使用 ${ENV_VAR} 变量语法来隐藏 URL 中的凭据：

```toml
[project]
dependencies = [
  "mypackage @ git+http://${VCS_USER}:${VCS_PASSWD}@test.git.com/test/mypackage.git@master"
]

```

这些变量将在安装项目时从环境变量中读取。

##### 添加仅用于开发的依赖项

从 1.5.0 版本开始支持本特性。

PDM 还支持定义一组仅用于开发的依赖项，例如一些用于测试和其他用于代码检查。通常我们不希望这些依赖项出现在分发的元数据中，因此使用 optional-dependencies 可能不是一个好主意。我们可以将它们定义为开发依赖项：

```shell
pdm add -dG test pytest
```

这将导致一个如下的 pyproject.toml 文件：

```toml
[tool.pdm.dev-dependencies]
test = ["pytest"]
```

你可以拥有几个开发依赖项组。与 optional-dependencies 不同，它们不会出现在包分发的元数据中，如 PKG-INFO 或 METADATA。包索引将不会意识到这些依赖项。模式与 optional-dependencies 相似，只是它位于 tool.pdm 表中。为了向后兼容，如果仅指定了 -d 或 --dev，依赖项将默认放在 [tool.pdm.dev-dependencies] 下的 dev 组中。

注意： 相同的组名不得同时出现在 [tool.pdm.dev-dependencies] 和 [project.optional-dependencies] 中。


##### 可编辑的依赖项

本地目录和版本控制系统依赖项可以以可编辑模式安装。如果你熟悉 pip，那就像是 pip install -e \<package>。可编辑包仅允许在开发依赖项中使用：

```shell
# A relative path to the directory
pdm add -e ./sub-package --dev
# A file URL to a local directory
pdm add -e file:///path/to/sub-package --dev
# A VCS URL
pdm add -e git+https://github.com/pallets/click.git@main#egg=click --dev
```

注意： 可编辑安装仅允许在开发依赖组中使用。其他组，包括默认组，将会失败并显示 [PdmUsageError]。



##### 保存版本规范

如果给定的包没有版本规范，例如 pdm add requests。PDM 提供了三种不同的行为来保存依赖项的版本规范，这由 --save-<strategy>（假设 2.21.0 是可以找到的依赖项的最新版本）来指定：

* minimum: 保存最低版本规范：>=2.21.0（默认）。
* compatible: 保存兼容版本规范：>=2.21.0,<3.0.0。
* exact: 保存精确版本规范：==2.21.0。
* wildcard: 不限制版本并将规范保持为通配符：*。

##### 添加预发布版本

可以给 pdm add 添加 \-\-pre/\-\-prerelease 选项，以便允许为给定的包固定预发布版本。

#### 更新现有依赖项

更新锁文件中的所有依赖项：

```shell
pdm update
```

更新指定的包：

```shell
pdm update requests
```

更新多个依赖组：

```shell
pdm update -G security -G http
```

或使用逗号分隔的列表：

```shell
pdm update -G "security,http"
```

更新指定组中的给定包：

```shell
pdm update -G security cryptography
```

如果未提供组，PDM 将在默认依赖项集中搜索要求，并在未找到时引发错误。

更新开发依赖项中的软件包：

```shell
# 更新所有默认 + dev-dependencies
pdm update -d
# 更新指定 dev-dependencies 组中的软件包
pdm update -dG test pytest
```

##### 关于更新策略

类似地，PDM 还提供了 3 种不同的更新依赖项和子依赖项的行为，通过 \-\-update-\<strategy> 选项来指定：

* reuse: 保留所有已锁定的依赖项，除了命令行中给定的那些（默认）。
* reuse-installed: 尝试重用工作集中安装的版本。这也会影响命令行中请求的软件包。
* eager: 尝试锁定命令行中的软件包及其递归子依赖项的更新版本，并保持其他依赖项不变。
* all: 更新所有依赖项和子依赖项。

##### 将软件包更新到破坏版本规范的版本

可以给 -u/\-\-unconstrained 以告诉 PDM 忽略 pyproject.toml 中的版本规范。这类似于 yarn upgrade -L/\-\-latest 命令。此外，pdm update 还支持 \-\-pre/\-\-prerelease 选项。

#### 删除现有依赖项

从项目文件和库目录中删除现有依赖项：

```shell
# 从默认依赖项中删除 requests
pdm remove requests
# 从可选依赖项的 'web' 组中删除 h11
pdm remove -G web h11
# 从 dev-dependencies 的 'test' 组中删除 pytest-cov
pdm remove -dG test pytest-cov
```

#### 安装锁定文件中固定版本的软件包

有几个类似的命令可以完成这个任务，稍有不同：

* pdm sync 从锁定文件安装软件包。
* pdm update 将更新锁定文件，然后同步。
* pdm install 将检查项目文件是否有更改，如果需要，将更新锁定文件，然后同步。

sync 还有一些选项来管理已安装的软件包：

* \-\-clean：将删除不再在锁定文件中的软件包。
* \-\-only-keep：只保留选定的软件包（使用选项如 -G 或 --prod）。

#### 指定要使用的锁定文件

可以使用 -L/\-\-lockfile <filepath> 选项或 PDM_LOCKFILE 环境变量来指定另一个锁定文件，而不是默认的 pdm lock。

#### 选择要安装或锁定的依赖组子集

假设我们有一个具有以下依赖项的项目：

```toml
[project]  # This is production dependencies
dependencies = ["requests"]

[project.optional-dependencies]  # This is optional dependencies
extra1 = ["flask"]
extra2 = ["django"]

[tool.pdm.dev-dependencies]  # This is dev dependencies
dev1 = ["pytest"]
dev2 = ["mkdocs"]

```

|Command	| What it does |	Comments |
|--|--|--|
pdm install	|install all groups locked in the lockfile	|
pdm install -G extra1	| install prod deps, dev deps, and "extra1" optional group	|
pdm install -G dev1	| install prod deps and only "dev1" dev group	|
pdm install -G:all	| install prod deps, dev deps and "extra1", "extra2" optional groups	|
pdm install -G extra1 -G dev1	| install prod deps, "extra1" optional group and only "dev1" dev group |	
pdm install --prod	| install prod only	|
pdm install --prod -G extra1	| install prod deps and "extra1" optional	|
pdm install --prod -G dev1	| Fail, --prod can't be given with dev dependencies	| Leave the --prod option


只要没有传递 \-\-prod 参数并且 -G 没有指定任何 dev 组，所有开发依赖项都会被包含进来。

另外，如果你不想安装根项目，可以添加 \-\-no-self 选项，当你希望所有软件包都以非可编辑版本安装时，可以使用 \-\-no-editable。

你也可以在这些选项下使用 pdm lock 命令，仅锁定指定的组，这将记录在锁定文件的 [metadata] 表中。如果未指定 \-\-group/\-\-prod/\-\-dev/\-\-no-default 选项，pdm sync 和 pdm update 将使用锁定文件中的组进行操作。但是，如果命令的参数中包含了任何不包含在锁定文件中的组，PDM 将会引发错误。

这个功能在管理多个锁定文件时特别有价值，每个锁定文件可能固定了同一软件包的不同版本。要在锁定文件之间进行切换，可以使用 \-\-lockfile/-L 选项。

举一个实际的例子，你的项目依赖于 werkzeug 的发布版本，而在开发时可能希望使用本地的开发版本。你可以将以下内容添加到你的 pyproject.toml 文件中：

```toml
[project]
requires-python = ">=3.7"
dependencies = ["werkzeug"]

[tool.pdm.dev-dependencies]
dev = ["werkzeug @ file:///${PROJECT_ROOT}/dev/werkzeug"]
```

然后，运行 pdm lock 使用不同的选项来为不同的目的生成锁定文件：

```shell
# 锁定 default + dev，并将本地的 werkzeug 复制固定在 pdm.lock 中。
pdm lock
# 锁定 default，并将发布版本的 werkzeug 固定在 pdm.prod.lock 中。
pdm lock --prod -L pdm.prod.lock
```

检查锁定文件中的 metadata.groups 字段，以查看包含了哪些组。


#### 锁定策略

目前，我们支持三个标志来控制锁定行为：cross_platform、static_urls 和 direct_minimal_versions，其含义如下。您可以通过 \-\-strategy/-S 选项传递一个或多个标志给 pdm lock，可以通过给出逗号分隔的列表或多次传递该选项来实现。这两个命令的功能相同：

```shell
pdm lock -S cross_platform,static_urls
pdm lock -S cross_platform -S static_urls
```

这些标志将被编码到锁定文件中，并在下次运行 pdm lock 时读取。但您可以通过在标志名称前加上 no_ 来禁用标志：

```shell
pdm lock -S no_cross_platform
```

这个命令使得锁定文件不跨平台。

##### 跨平台(Cross platform)

**自版本 2.6.0 起**

默认情况下，生成的锁定文件是跨平台的，这意味着在解决依赖关系时不考虑当前平台。结果的锁定文件将包含所有可能平台和 Python 版本的 wheels 和依赖项。然而，有时候当一个发布版不包含所有 wheels 时，这会导致一个错误的锁定文件。为了避免这种情况，您可以告诉 PDM 创建一个仅适用于当前平台的锁定文件，修剪掉对当前平台不相关的 wheels。这可以通过在 pdm lock 中传递 \-\-strategy no_cross_platform 选项来实现：

```
pdm lock --strategy no_cross_platform
```

##### 静态 URLs(Static URLs)

**自版本 2.8.0 起**


默认情况下，PDM 在锁定文件中仅存储软件包的文件名，这有利于在不同的软件包索引之间重用。然而，如果您想在锁定文件中存储软件包的静态 URL，可以通过传递 --strategy static_urls 选项给 pdm lock 来实现：

```
pdm lock --strategy static_urls
```

这些设置将被保存并记住在同一锁定文件中。您还可以通过传递 \-\-strategy no_static_urls 来禁用它。

##### 直接最小版本(Direct minimal versions)

**自版本 2.10.0 起**

当通过传递 \-\-strategy direct_minimal_versions 启用时，将解析在 pyproject.toml 中指定的依赖项到可用的最小版本，而不是最新版本。当您希望测试项目在一系列依赖项版本中的兼容性时，这是非常有用的。

例如，如果在 pyproject.toml 中指定了 flask>=2.0，如果没有其他兼容性问题，flask 将被解析为版本 2.0.0。

**注意**：软件包依赖项中的版本约束并不具有未来兼容性。如果将依赖项解析为最小版本，则可能会出现向后兼容性问题。例如，flask==2.0.0 需要 werkzeug>=2.0，但实际上，它无法与 3.0.0 版的 Werkzeug 兼容，后者在发布两年后。

##### 继承父级的元数据

**从版本2.11.0开始新增**

之前，pdm lock命令会记录包的元数据。在安装时，PDM会从顶级要求开始向下遍历依赖树的叶子节点。然后，它会评估遇到的任何标记是否符合当前环境。如果标记不满足，包将被丢弃。换句话说，我们需要一个额外的“解决”步骤来进行安装。

当启用inherit_metadata策略时，PDM将继承并合并来自包祖先的环境标记。然后，在锁定期间，这些标记被编码到锁文件中，从而实现更快的安装。从版本2.11.0开始，默认情况下启用了此策略。要在配置中禁用此策略，请使用pdm config strategy.inherit_metadata false。

#### 显示已安装的软件包

类似于pip list，您可以列出安装在packages目录中的所有软件包：

```shell
pdm list
```

##### 包含和排除组

默认情况下，将列出工作集中安装的所有软件包。您可以通过--include/--exclude选项指定要列出的组，include的优先级高于exclude。

```shell
pdm list --include dev
pdm list --exclude test
```

当包含时，有一个特殊的组:sub，将显示所有传递性依赖关系。它默认包含在内。

您还可以传递\-\-resolve给pdm list，这将显示在pdm.lock中解析的软件包，而不是在工作集中安装的软件包。

##### 更改输出字段和格式

默认情况下，列表输出将显示名称、版本和位置。您可以通过\-\-fields选项查看更多字段或指定字段的顺序：

```shell
pdm list --fields name,licenses,version
```
有关所有支持的字段，请参阅[CLI参考](https://pdm-project.org/latest/reference/cli/#list_1)。

此外，您还可以指定除默认表格输出之外的输出格式。支持的格式和选项有\-\-csv、\-\-json、\-\-markdown和\-\-freeze。

##### 显示依赖树

或者通过以下方式显示依赖树：

```shell
$ pdm list --tree
tempenv 0.0.0
└── click 7.0 [ required: <7.0.0,>=6.7 ]
black 19.10b0
├── appdirs 1.4.3 [ required: Any ]
├── attrs 19.3.0 [ required: >=18.1.0 ]
├── click 7.0 [ required: >=6.5 ]
├── pathspec 0.7.0 [ required: <1,>=0.6 ]
├── regex 2020.2.20 [ required: Any ]
├── toml 0.10.0 [ required: >=0.9.4 ]
└── typed-ast 1.4.1 [ required: >=1.4.0 ]
bump2version 1.0.0
```

请注意，\-\-fields选项与\-\-tree不兼容。

##### 通过模式过滤软件包

您还可以通过传递模式来限制要显示的软件包：

```
pdm list flask-* requests-*
```

要注意的是，在--tree模式下，只会显示匹配软件包的子树。这可用于实现与pnpm why相同的目的，即显示为什么需要特定软件包。

```
$ pdm list --tree --reverse certifi
certifi 2023.7.22
└── requests 2.31.0 [ requires: >=2017.4.17 ]
└── cachecontrol[filecache] 0.13.1 [ requires: >=2.16.0 ]
```

#### 允许安装预发布版本

要启用，请在pyproject.toml中包含以下设置：

```toml
[tool.pdm.resolution]
allow-prereleases = true
```

#### 设置锁定或安装的可接受格式

如果要控制软件包的格式（二进制/源分发），您可以设置环境变量PDM_NO_BINARY和PDM_ONLY_BINARY。

每个环境变量是一个逗号分隔的软件包名称列表。您可以将其设置为:all:以应用于所有软件包。例如：

```shell
# No binary for werkzeug will be locked nor used for installation
PDM_NO_BINARY=werkzeug pdm add flask
# Only binaries will be locked in the lock file
PDM_ONLY_BINARY=:all: pdm lock
# No binaries will be used for installation
PDM_NO_BINARY=:all: pdm install
# Prefer binary distributions and even if sdist with higher version is available
PDM_PREFER_BINARY=flask pdm install
```
 
#### 锁定失败

如果PDM无法找到满足条件的锁定方法，则会输出错误信息：

```
pdm django==3.1.4 "asgiref<3"
...
🔒 Lock failed
Unable to find a resolution for asgiref because of the following conflicts:
  asgiref<3 (from project)
  asgiref<4,>=3.2.10 (from <Candidate django 3.1.4 from https://pypi.org/simple/django/>)
To fix this, you could loosen the dependency version constraints in pyproject.toml. If that is not possible, you could also override the resolved version in `[tool.pdm.resolution.overrides]` table.

```
您可以选择将django更改为较低的版本，或者删除asgiref的上限。但如果这对您的项目不合适，您可以尝试覆盖已解析的软件包版本，甚至可以在pyproject.toml中不锁定该特定软件包。


#### 管理全局项目

有时用户可能还想跟踪全局Python解释器的依赖关系。通过PDM很容易做到这一点，通过-g/\-\-global选项，它被大多数子命令支持。

如果传递了该选项，则将使用\<CONFIG_ROOT>/global-project作为项目目录，它与普通项目几乎相同，只是会自动为您创建pyproject.toml，并且不支持构建功能。这个想法来自Haskell的stack。

但是，与stack不同的是，默认情况下，PDM不会自动使用全局项目，如果找不到本地项目，用户应该显式传递-g/\-\-global来激活它，因为如果软件包安装到错误的位置，这是不太令人愉快的。但PDM也将这个决定留给了用户，只需将配置global_project.fallback设置为true即可。

默认情况下，当pdm隐式使用全局项目时，会打印以下消息：找不到项目，回退到全局项目。要禁用此消息，请将配置global_project.fallback_verbose设置为false。

如果您希望全局项目跟踪除<CONFIG_ROOT>/global-project之外的其他项目文件，可以通过-p/\-\-project <path>选项提供项目路径。特别是如果传递了\-\-global \-\-project .，PDM将会安装当前项目的依赖项到全局Python中。

警告：在使用全局项目时，要谨慎使用remove和sync \-\-clean/\-\-pure命令，因为这可能会删除在系统Python中安装的软件包。

#### 将锁定的软件包导出为其他格式

您还可以将pdm lock导出为其他格式，以便于CI流程或镜像构建过程。目前，只支持requirements.txt格式：

```shell
pdm export -o requirements.txt
```

注意：您也可以在.pre-commit钩子中运行pdm export。

### 构建和发布

如果您正在开发一个库，在向项目添加依赖项并完成编码后，是时候构建和发布您的软件包了。只需一个命令即可轻松完成：

```shell
pdm publish
```

这将自动构建一个wheel和一个源分发（sdist），并将它们上传到PyPI索引。

要指定除PyPI之外的另一个存储库，请使用--repository选项，参数可以是上传URL或存储在配置文件中的存储库的名称。

```shell
pdm publish --repository testpypi
pdm publish --repository https://test.pypi.org/legacy/
```

#### 以可信发布者身份发布

您可以配置PyPI的受信任发布者，以便在发布工作流中不需要暴露PyPI令牌。要做到这一点，请按照添加发布者的指南，并编写如下的GitHub Actions工作流：

```yaml
jobs:
  pypi-publish:
    name: upload release to PyPI
    runs-on: ubuntu-latest
    permissions:
      # This permission is needed for private repositories.
      contents: read
      # IMPORTANT: this permission is mandatory for trusted publishing
      id-token: write
    steps:
      - uses: actions/checkout@v3

      - uses: pdm-project/setup-pdm@v3

      - name: Publish package distributions to PyPI
        run: pdm publish
```

#### 分别构建和发布

您也可以将软件包构建和上传分成两个步骤，以便在上传之前检查已构建的工件。

```shell
pdm build
```

有许多选项可控制构建过程，具体取决于所使用的后端。有关更多详细信息，请参阅构建配置部分。

构件将在dist/中创建，并可以上传到PyPI。

```shell
pdm publish --no-build
```

### 配置项目

PDM 的 config 命令的用法与 git config 相似，不过不需要 \-\-list 参数来显示配置。

显示当前的配置：

```shell
pdm config
```

获取单个配置：

```shell
pdm config pypi.url
```

更改配置值并存储在主目录的配置文件中：

```shell
pdm config pypi.url "https://test.pypi.org/simple"
```

默认情况下，配置是全局更改的，如果要使配置仅对当前项目可见，请添加 \-\-local 标志：

```shell
pdm config --local pypi.url "https://test.pypi.org/simple"
```

任何本地配置将存储在项目根目录下的 pdm.toml 中。

#### 配置文件

配置文件按以下顺序搜索：

* \<PROJECT_ROOT>/pdm.toml - 项目配置
* \<CONFIG_ROOT>/config.toml - 主目录配置
* \<SITE_CONFIG_ROOT>/config.toml - 站点配置

其中 <CONFIG_ROOT> 是：

* \$XDG_CONFIG_HOME/pdm（在大多数情况下为 ~/.config/pdm）在 Linux 中，由 [XDG 基础目录规范](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)定义
* ~/Library/Application Support/pdm 在 macOS 中，由苹果文件系统基础知识定义
* %USERPROFILE%\AppData\Local\pdm 在 Windows 中，由已知文件夹定义

而 <SITE_CONFIG_ROOT> 是：

* \$XDG_CONFIG_DIRS/pdm（在大多数情况下为 /etc/xdg/pdm）在 Linux 中，由 [XDG 基础目录规范](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)定义
* /Library/Application Support/pdm 在 macOS 中，由苹果文件系统基础知识定义
* C:\ProgramData\pdm\pdm 在 Windows 中，由已知文件夹定义

如果使用了 -g/\-\-global 选项，则第一项将由 <CONFIG_ROOT>/global-project/pdm.toml 替换。


您可以在 [Configuration Page](https://pdm-project.org/latest/reference/configuration/) 中找到所有可用的配置项。

#### 配置 Python 查找器

默认情况下，PDM 将尝试在以下来源中查找 Python 解释器：

* venv: PDM 虚拟环境位置
* path: PATH 环境变量
* pyenv: pyenv 安装根目录
* rye: rye 工具链安装根目录
* asdf: asdf python 安装根目录
* winreg: Windows 注册表

您可以取消选择其中一些或更改顺序，通过设置 python.providers 配置键：

```
pdm config python.providers rye   # 仅 Rye 源
pdm config python.providers pyenv,asdf  # pyenv 和 asdf
```

#### 允许预发布版本出现在解析结果中

默认情况下，PDM 的依赖解析器将忽略预发布版本，除非在依赖的给定版本范围内没有稳定版本。可以通过在 [tool.pdm.resolution] 表中将 allow-prereleases 设置为 true 来更改此行为：

```toml
[tool.pdm.resolution]
allow-prereleases = true
```

#### 配置软件包索引

您可以告诉 PDM 在 pyproject.toml 中指定源，也可以通过 pypi.* 配置指定。

在 pyproject.toml 中添加源：

```toml
[[tool.pdm.source]]
name = "private"
url = "https://private.pypi.org/simple"
verify_ssl = true
```

通过 pdm config 更改默认索引：

```toml
pdm config pypi.url "https://test.pypi.org/simple"
```

通过 pdm config 添加额外索引：

```toml
pdm config pypi.extra.url "https://extra.pypi.org/simple"
```

可用的配置选项包括：

* url：索引的 URL
* verify_ssl：（可选）是否验证 SSL 证书，默认为 true
* username：（可选）索引的用户名
* password：（可选）索引的密码
* type：（可选）索引(index)或 find_links，默认为 index

有关源类型的说明：默认情况下，所有来源都是符合[PEP 503标准](https://www.python.org/dev/peps/pep-0503/)的“索引”，类似于pip的--index-url和--extra-index-url，然而，你可以将类型设置为find_links，其中包含要直接查找的文件或链接。有关这两种类型之间的区别，请参阅[此答案](https://www.python.org/dev/peps/pep-0503/)。

这些配置按以下顺序读取以构建最终的源列表：

* 如果 pypi 没有出现在 pyproject.toml 中任何源的 name 字段中，则使用 pypi.url
* pyproject.toml 中的源
* PDM 配置中的 pypi.\<name>.url。

您可以将 pypi.ignore_stored_index 设置为 true 以禁用 PDM 配置中的所有索引，并仅使用在 pyproject.toml 中指定的索引。

**禁用默认的 PyPI 索引**

如果要省略默认的 PyPI 索引，只需将源名称设置为 pypi，该源将替换它。

```toml
[[tool.pdm.source]]
url = "https://private.pypi.org/simple"
verify_ssl = true
name = "pypi"
```


**pyproject.toml 或配置中的索引**

当您希望与其他人共享索引以供项目使用时，您应该将它们添加到pyproject.toml中。例如，某些包只存在于私有索引中，如果某人没有配置索引，则无法安装这些包。否则，将它们存储在本地配置中，这样其他人就看不到它们。


#### 保持源的顺序

默认情况下，所有源被视为相等，它们的软件包按版本和wheel标签排序，选择最匹配的且版本最高的源。

在某些情况下，您可能希望从首选源返回软件包，并在前者缺失时搜索其他源。PDM通过读取respect-source-order配置来支持此功能：

```toml
[tool.pdm.resolution]
respect-source-order = true
```

#### 为个别软件包指定索引

您可以使用tool.pdm.source表下的include_packages和exclude_packages配置将软件包绑定到特定的源。

```toml
[[tool.pdm.source]]
name = "private"
url = "https://private.pypi.org/simple"
include_packages = ["foo", "foo-*"]
exclude_packages = ["bar-*"]
```

在上述配置中，任何匹配foo或foo-*的软件包只会从私有索引中搜索，而匹配bar-*的软件包将从除私有索引之外的所有索引中搜索。

include_packages和exclude_packages都是可选的，并且接受一个通配符模式的列表，当模式匹配时，include_packages会独自生效。

#### 使用索引存储凭证

您可以在URL中使用${ENV_VAR}变量扩展来指定凭证，这些变量将从环境变量中读取：

```toml
[[tool.pdm.source]]
name = "private"
url = "https://${PRIVATE_PYPI_USERNAME}:${PRIVATE_PYPI_PASSWORD}@private.pypi.org/simple"
```


#### 配置HTTPS证书

您可以为HTTPS请求使用自定义CA捆绑包或客户端证书。它可以配置为索引（用于软件包下载）和仓库（用于上传）：

```shell
pdm config pypi.ca_certs /path/to/ca_bundle.pem
pdm config repository.pypi.ca_certs /path/to/ca_bundle.pem
```

此外，您可以使用系统信任存储，而不是捆绑的certifi证书来验证HTTPS证书。这种方法通常支持企业代理证书，无需额外配置。

要使用信任存储，您需要Python 3.10或更新版本，并将truststore安装到与PDM相同的环境中：

```shell
$ pdm self add truststore
```

#### 索引配置合并

索引配置与tool.pdm.source表中的name字段或配置文件中的pypi.<name>键进行合并。这使您可以将URL和凭证分开存储，以避免在源代码控制中暴露密钥。例如，如果您有以下配置：

```toml
[[tool.pdm.source]]
name = "private"
url = "https://private.pypi.org/simple"
```

您可以将凭证存储在配置文件中：

```shell
pdm config pypi.private.username "foo"
pdm config pypi.private.password "bar"
```

PDM可以从这两个地方检索私有索引的配置。

如果索引需要用户名和密码，但是它们既无法从环境变量中找到，也无法从配置文件中找到，PDM将提示您输入它们。或者，如果安装了keyring，则会将其用作凭证存储。PDM可以使用来自已安装软件包或CLI的keyring。


#### 中央安装缓存#

如果一个软件包被系统上的许多项目所需，每个项目都必须保留自己的副本。这可能会浪费磁盘空间，特别是对于数据科学和机器学习项目而言。

PDM支持通过将相同的wheel安装在集中式软件包存储库中，并在不同的项目中链接到该安装来缓存相同wheel的安装。要启用它，请运行：

```shell
pdm config install.cache on
```

可以通过在命令中添加--local选项来针对每个项目启用它。

缓存位于$(pdm config cache_dir)/packages。您可以使用pdm cache info查看缓存使用情况。请注意，缓存的安装是自动管理的——如果它们未链接到任何项目，则会被删除。手动从磁盘中删除缓存可能会破坏系统上的某些项目。

此外，支持将链接到缓存条目的几种不同方式：

* 符号链接（默认），如果父级是命名空间包，则创建到软件包目录或其子目录的符号链接。
* symlink_individual，对软件包目录中的每个单独文件，创建指向它的符号链接。
* hardlink，创建到缓存条目的软件包文件的硬链接。
* pth，不链接软件包目录，而是通过.pth文件将路径添加到sys.path（IDE可能不支持此选项）。

您可以通过运行pdm config [-l] install.cache_method \<method>在它们之间切换。

注意：只有从PyPI解析的命名要求的安装可以被缓存。

#### 配置用于上传的仓库

在使用pdm publish命令时，它会从全局配置文件（<CONFIG_ROOT>/config.toml）中读取仓库密码。配置的内容如下：

```toml
[repository.pypi]
username = "frostming"
password = "<secret>"

[repository.company]
url = "https://pypi.company.org/legacy/"
username = "frostming"
password = "<secret>"
ca_certs = "/path/to/custom-cacerts.pem"
```

或者，这些凭证可以通过环境变量提供：

```
export PDM_PUBLISH_REPO=...
export PDM_PUBLISH_USERNAME=...
export PDM_PUBLISH_PASSWORD=...
export PDM_PUBLISH_CA_CERTS=...
```

一个PEM编码的证书授权捆绑包（ca_certs）可用于本地/自定义PyPI仓库，其中服务器证书未由标准certifi CA捆绑包签名。

注意：仓库与前一节中的索引不同。仓库用于发布，而索引用于锁定和解析。它们不共享配置。

提示

您不需要为pypi和testpypi仓库配置URL，它们已填充默认值。用户名、密码和证书授权捆绑包可以通过命令行参数\-\-username、\-\-password和\-\-ca-certs传递给pdm publish。

要从命令行更改仓库配置，请使用pdm config命令：

```shell
pdm config repository.pypi.username "__token__"
pdm config repository.pypi.password "my-pypi-token"

pdm config repository.company.url "https://pypi.company.org/legacy/"
pdm config repository.company.ca_certs "/path/to/custom-cacerts.pem"
```


#### 密码管理与密钥环

当密钥环可用且受支持时，密码将存储到密钥环中，并从密钥环中检索，而不是写入到配置文件中。这支持索引和上传仓库。服务名称将对于索引是pdm-pypi-\<name>，对于仓库是pdm-repository-\<name>。

您可以通过将密钥环安装到与PDM相同的环境中，或者全局安装来启用密钥环。将密钥环添加到PDM环境中：

```shell
pdm self add keyring
```

或者，如果您已经全局安装了密钥环的副本，请确保CLI在PATH环境变量中暴露，以便PDM可以发现它：

```shell
export PATH=$PATH:path/to/keyring/bin
```

#### 覆盖已解析的软件包版本

**版本1.12.0中的新功能**

有时，由于您无法修复的上游库设置了不正确的版本范围，您无法获得依赖关系的解析。在这种情况下，您可以使用PDM的覆盖功能来强制安装软件包的特定版本。

在pyproject.toml中给出以下配置：

```toml
[tool.pdm.resolution.overrides]
asgiref = "3.2.10"  # exact version
urllib3 = ">=1.26.2"  # version range
pytz = "https://mypypi.org/packages/pytz-2020.9-py3-none-any.whl"  # absolute URL

```


该表的每个条目都是一个带有期望版本的软件包名称。在此示例中，PDM将解析以上软件包为给定的版本，而不管是否存在其他解析。

警告：通过使用[tool.pdm.resolution.overrides]设置，您自行承担该解析可能存在的任何不兼容性的风险。只有当您的要求没有有效的解析，并且您知道特定版本可用时，才可以使用此功能。大多数情况下，您只需将任何瞬态约束添加到依赖关系数组中。

#### 从锁文件中排除特定软件包及其依赖项

**版本2.12.0中的新功能**

有时，您甚至不希望将某些软件包包含在锁文件中，因为您确信它们不会被任何代码使用。在这种情况下，您可以完全跳过它们及其依赖项的解析过程：

```toml
[tool.pdm.resolution]
excludes = ["requests"]
```

使用此配置，requests将不会被锁定在锁文件中，如果没有其他软件包依赖于它，那么它的依赖项，如urllib3和idna也不会出现在解析结果中。安装程序也无法选取它们。

#### 对每个pdm调用传递常量参数

**版本2.7.0中的新功能**

您可以通过tool.pdm.options配置向每个pdm命令传递额外的选项：

```toml
[tool.pdm.options]
add = ["--no-isolation", "--no-self"]
install = ["--no-self"]
lock = ["--no-cross-platform"]
```

这些选项将在命令名称之后添加。例如，根据上述配置，pdm add requests等



#### 忽略包警告

**自2.10.0版本起新增功能**

在解析依赖项时，您可能会看到类似以下的警告：

```
PackageWarning: Skipping scipy@1.10.0 because it requires Python
<3.12,>=3.8 but the project claims to work with Python>=3.9.
Narrow down the `requires-python` range to include this version. For example, ">=3.9,<3.12" should work.
  warnings.warn(record.message, PackageWarning, stacklevel=1)
Use `-q/--quiet` to suppress these warnings, or ignore them per-package with `ignore_package_warnings` config in [tool.pdm] table.
```
这是因为包的支持的 Python 版本范围不包括在pyproject.toml中指定的 requires-python 值。您可以通过添加以下配置在每个包基础上忽略这些警告：

```toml
[tool.pdm]
ignore_package_warnings = ["scipy", "tensorflow-*"]
```

其中每个条目是一个不区分大小写的通配符模式，用于匹配包名称。




### PDM 脚本

与 npm run 类似，使用 PDM，您可以加载本地包并运行任意脚本或命令。

#### 任意脚本

```shell
pdm run flask run -p 54321
```

它将在项目环境中的包的环境中运行 flask run -p 54321。

#### 用户脚本

PDM 还支持在 pyproject.toml 的可选 [tool.pdm.scripts] 部分中自定义脚本快捷方式。

然后，您可以运行 pdm run <script_name> 以在您的 PDM 项目的上下文中调用脚本。例如：

```toml
[tool.pdm.scripts]
start = "flask run -p 54321"
```

然后在您的终端中：

```shell
$ pdm run start
Flask server started at http://127.0.0.1:54321
```
 
后续的参数将附加到命令后：

```shell
$ pdm run start -h 0.0.0.0
Flask server started at http://0.0.0.0:54321
```

**类似 Yarn 的脚本快捷方式**

有一个内置的快捷方式，只要脚本不与任何内置或插件提供的命令冲突即可作为根命令使用。换句话说，如果您有一个 start 脚本，您可以同时运行 pdm run start 和 pdm start。但是如果您有一个 install 脚本，只有 pdm run install 会运行它，pdm install 仍将运行内置的 install 命令。

PDM 支持 4 种脚本类型：

##### cmd

纯文本脚本被视为普通命令，或者您可以明确指定：

```toml
[tool.pdm.scripts]
start = {cmd = "flask run -p 54321"}
```

在某些情况下，比如想在参数之间添加注释时，将命令指定为数组而不是字符串可能更方便：

```toml
[tool.pdm.scripts]
start = {cmd = [
    "flask",
    "run",
    # Important comment here about always using port 54321
    "-p", "54321"
]}
```

##### shell

Shell 脚本可用于运行更多与 Shell 特定任务相关的任务，如管道和输出重定向。这基本上是通过 subprocess.Popen() 与 shell=True 运行的：

```toml
[tool.pdm.scripts]
filter_error = {shell = "cat error.log|grep CRITICAL > critical.log"}
```

##### call

脚本还可以定义为调用形式为 <module_name>:<func_name> 的 Python 函数：

```toml
[tool.pdm.scripts]
foobar = {call = "foo_package.bar_module:main"}
```

该函数可以提供文字参数：

```toml
[tool.pdm.scripts]
foobar = {call = "foo_package.bar_module:main('dev')"}
```

##### composite

此脚本类型执行其他已定义的脚本：

```toml
[tool.pdm.scripts]
lint = "flake8"
test = "pytest"
all = {composite = ["lint", "test"]}
```

运行 pdm run all 将首先运行 lint，然后在 lint 成功后运行 test。

您还可以为所调用的脚本提供参数：

```toml
[tool.pdm.scripts]
lint = "flake8"
test = "pytest"
all = {composite = ["lint mypackage/", "test -v tests/"]}
```

注意：传递给命令行的参数会传递给每个调用的任务。

#### 脚本选项

##### env

在当前 shell 中设置的所有环境变量都可以被 pdm run 看到，并在执行时展开。此外，您还可以在您的 pyproject.toml 中定义一些固定的环境变量：

```toml
[tool.pdm.scripts]
start.cmd = "flask run -p 54321"
start.env = {FOO = "bar", FLASK_ENV = "development"}
```

请注意，我们如何使用 TOML 的语法来定义复合字典。

**关于环境变量替换**

脚本规范中的变量可以在所有脚本类型中进行替换。在 cmd 脚本中，所有平台上只支持 ${VAR} 语法，但在 shell 脚本中，语法是依赖于平台的。例如，Windows cmd 使用 %VAR%，而 bash 使用 $VAR。

注意：在复合任务级别指定的环境变量将覆盖由调用的任务定义的环境变量。

##### env_file

您还可以将所有环境变量存储在 dotenv 文件中，并让 PDM 读取它：

```toml
[tool.pdm.scripts]
start.cmd = "flask run -p 54321"
start.env_file = ".env"
```

dotenv 文件中的变量不会覆盖任何现有的环境变量。如果您希望 dotenv 文件覆盖现有的环境变量，请使用以下方式：

```toml
[tool.pdm.scripts]
start.cmd = "flask run -p 54321"
start.env_file.override = ".env"
```

注意：在复合任务级别指定的 dotenv 文件将覆盖由调用的任务定义的 dotenv 文件。

##### site_packages

为了确保运行环境与外部 Python 解释器正确隔离，除非满足以下任何条件，否则不会将所选解释器的 site-packages 加载到 sys.path 中：

* 可执行文件来自 PATH 但不在 pypackages 文件夹内。
* -s/\-\-site-packages 标志跟在 pdm run 后面。
* 在脚本表或全局设置键 _ 中有 site_packages = true。

请注意，如果启用了 PEP 582（没有 pdm run 前缀），site-packages 将始终被加载。

##### 共享选项

如果您希望所有由 pdm run 运行的任务共享选项，您可以将它们写在 [tool.pdm.scripts] 表的一个特殊键 _ 下：

```toml
[tool.pdm.scripts]
_.env_file = ".env"
start = "flask run -p 54321"
migrate_db = "flask db upgrade"
```

此外，在任务内部，PDM_PROJECT_ROOT 环境变量将被设置为项目根目录。

##### 参数占位符

默认情况下，所有用户提供的额外参数都简单地附加到命令（或对于复合任务的所有命令）。

如果您希望对用户提供的额外参数具有更多控制权，可以使用 {args} 占位符。它适用于所有脚本类型，并且每个都将适当地进行插值：

```toml
[tool.pdm.scripts]
cmd = "echo '--before {args} --after'"
shell = {shell = "echo '--before {args} --after'"}
composite = {composite = ["cmd --something", "shell {args}"]}
```


将生成以下插值（这些不是真正的脚本，只是用来说明插值）：

```
$ pdm run cmd --user --provided
--before --user --provided --after
$ pdm run cmd
--before --after
$ pdm run shell --user --provided
--before --user --provided --after
$ pdm run shell
--before --after
$ pdm run composite --user --provided
cmd --something
shell --before --user --provided --after
$ pdm run composite
cmd --something
shell --before --after
```

您可以选择为不提供用户参数的情况提供默认值，这些值将在未提供用户参数时使用：

```toml
[tool.pdm.scripts]
test = "echo '--before {args:--default --value} --after'"
```

将生成以下：

```shell
$ pdm run test --user --provided
--before --user --provided --after
$ pdm run test
--before --default --value --after
```

注意：一旦检测到占位符，就不会再附加参数。这对于复合脚本很重要，因为如果在其中一个子任务中检测到了占位符，那么对于需要它的每个嵌套命令，都不会有参数附加，您需要明确地将占位符传递给每个嵌套命令。

注意：call 脚本不支持 {args} 占位符，因为它们直接访问 sys.argv 来处理这种复杂情况以及更多。

##### {pdm} 占位符

有时您可能有多个 PDM 安装，或者以不同的名称安装了 pdm。例如，在 CI/CD 情况下，或者在不同的存储库中使用不同的 PDM 版本时可能会发生这种情况。为了使您的脚本更加健壮，您可以使用 {pdm} 来使用执行脚本的 PDM 入口点。这将扩展为 {sys.executable} -m pdm。

```toml
[tool.pdm.scripts]
whoami = { shell = "echo \`{pdm} -V\` was called as '{pdm} -V'" }
```

将产生以下输出：

```shell
$ pdm whoami
PDM, version 0.1.dev2501+g73651b7.d20231115 was called as /usr/bin/python3 -m pdm -V

$ pdm2.8 whoami
PDM, version 2.8.0 was called as <snip>/venvs/pdm2-8/bin/python -m pdm -V
```

注意：虽然上面的示例使用了 PDM 2.8，但这个功能是在 2.10 系列中引入的，并且只针对展示进行了后续追加。

#### 显示脚本列表

使用 pdm run \-\-list/-l 来显示可用脚本快捷方式的列表：

```shell
$ pdm run --list
╭─────────────┬───────┬───────────────────────────╮
│ Name │ Type │ Description │
├─────────────┼───────┼───────────────────────────┤
│ test_cmd │ cmd │ flask db upgrade │
│ test_script │ call │ call a python function │
│ test_shell │ shell │ shell command │
╰─────────────┴───────┴───────────────────────────╯
```

您可以添加一个带有脚本描述的帮助选项，它将显示在上述输出的 Description 列中。

注意：以下划线 (_) 开头的任务被视为内部（助手...），不会显示在列表中。

#### 预/后脚本

与 npm 类似，PDM 也支持通过预和后脚本组合任务，预脚本将在给定任务之前运行，后脚本将在之后运行。

```toml
[tool.pdm.scripts]
pre_compress = "\{\{ Run BEFORE the \`compress\` script }}"
compress = "tar czvf compressed.tar.gz data/"
post_compress = "\{\{ Run AFTER the \`compress\` script }}"
```

在此示例中，pdm run compress 将按顺序运行所有这 3 个脚本。

流水线快速失败：在预 - 自身 - 后脚本的流水线中，失败将取消后续的执行。

#### 钩子脚本

在某些情况下，PDM 将寻找一些特殊的钩子脚本进行执行：

* post_init：在 pdm init 之后运行
* pre_install：在安装软件包之前运行
* post_install：在安装软件包后运行
* pre_lock：在解析依赖项之前运行
* post_lock：在解析依赖项后运行
* pre_build：在构建分发之前运行
* post_build：在构建分发后运行
* pre_publish：在发布分发之前运行
* post_publish：在发布分发后运行
* pre_script：在任何脚本之前运行
* post_script：在任何脚本之后运行

注意：预/后脚本不能接收任何参数。

避免名称冲突。如果在 [tool.pdm.scripts] 表中存在一个 install 脚本，则 pre_install 脚本可以被 pdm install 和 pdm run install 同时触发。因此，建议不要使用保留名称。

注意：复合任务也可以有预和后脚本。被调用的任务将运行其自己的预和后脚本。

#### 跳过脚本

因为有时希望运行一个脚本但不运行其钩子(hook)或预和后脚本，有一个 --skip=:all 将禁用所有钩子、预和后脚本。还有 --skip=:pre 和 --skip=:post 分别允许跳过所有 pre_\* 钩子和所有 post_\* 钩子。

还可能需要一个预脚本但不需要后脚本，或者需要来自复合任务的所有任务除了一个。对于这些用例，有一个更细粒度的 --skip 参数，接受一个要排除的任务或钩子名称的列表。

```shell
pdm run --skip pre_task1,task2 my-composite
```

此命令将运行 my-composite 任务，并跳过 pre_task1 钩子以及任务 2 及其钩子。

您还可以在 PDM_SKIP_HOOKS 环境变量中提供跳过列表，但只要提供了 \-\-skip 参数，它就会被覆盖。

有关钩子和预/后脚本行为的更多详细信息，请参阅[专用钩子页面](https://pdm-project.org/latest/usage/hooks/)。

### 生命周期和钩子

与任何 Python 交付物一样，您的项目将经历 Python 项目生命周期的不同阶段，并且 PDM 提供了执行这些阶段预期任务的命令。

它还提供了附加到这些步骤的钩子，允许：

* 插件监听相同名称的[信号](https://pdm-project.org/latest/reference/api/#pdm.signals)。
* 开发人员定义具有相同名称的自定义脚本。

此外，pre_invoke 信号在调用任何命令之前发出，允许插件在此之前修改项目或选项。

内置命令当前分为 3 组：

* 初始化阶段
* 依赖管理。
* 发布阶段。

您可能需要在安装和发布阶段之间执行一些重复性任务（清理，linting，测试等），这就是为什么 PDM 允许您使用用户脚本定义自己的任务/阶段的原因。

为了提供完全的灵活性，PDM 允许根据需要跳过一些钩子和任务。

#### 初始化

初始化阶段应仅在项目的生命周期中发生一次，通过运行 pdm init 命令来初始化现有项目（提示填写 pyproject.toml 文件）。

它们触发以下钩子：

* post_init


<a>![](/img/pdm/1.png)</a>



#### 依赖管理

依赖管理对于开发人员能够工作并执行以下操作至关重要：

* lock：从 pyproject.toml 的要求计算一个锁定文件。
* sync：从锁定文件同步（添加/删除/更新）PEP582 包，并将当前项目安装为可编辑。
* add：添加一个依赖
* remove：删除一个依赖

所有这些步骤都直接可用以下命令完成：

* pdm lock：执行锁定任务
* pdm sync：执行同步任务
* pdm install：执行同步任务，如果需要，则先执行锁定任务
* pdm add：添加一个依赖要求，然后重新锁定并同步
* pdm remove：删除一个依赖要求，然后重新锁定并同步
* pdm update：从它们的最新版本重新锁定依赖项，然后同步

它们触发以下钩子：

* pre_install
* post_install
* pre_lock
* post_lock


<a>![](/img/pdm/2.png)</a>

##### 切换 Python 版本

这是依赖管理中的一个特殊情况：您可以使用 pdm use 切换当前的 Python 版本，并且它将发出带有新 Python 解释器的 post_use 信号。

<a>![](/img/pdm/3.png)</a>


#### 发布

一旦您准备好发布您的包/库，您将需要执行以下发布任务：

* build：构建/编译需要的资产，并将所有内容打包成一个 Python 包（sdist、wheel）
* upload：将包上传/发布到远程 PyPI 索引

所有这些步骤都可以使用以下命令完成：

* pdm build
* pdm publish

它们触发以下钩子：

* pre_publish
* post_publish
* pre_build
* post_build

<a>![](/img/pdm/4.png)</a>

执行将在第一个失败处停止，包括钩子在内。

#### 用户脚本

用户脚本在其自己的部分中有详细说明，但您应该知道：

* 每个用户脚本都可以定义一个 pre_\* 和 post_\* 脚本，包括复合脚本。
* 每次运行执行都会触发 pre_run 和 post_run 钩子
* 每次脚本执行都会触发 pre_script 和 post_script 钩子

给定以下脚本定义：

```toml
[tool.pdm.scripts]
pre_script = ""
post_script = ""
pre_test = ""
post_test = ""
test = ""
pre_composite = ""
post_composite = ""
composite = {composite = ["test"]}
```

pdm run test 将具有以下生命周期

<a>![](/img/pdm/5.png)</a>

而 pdm run composite 将具有以下生命周期:

<a>![](/img/pdm/6.png)</a>

#### Skipping

可以使用 \-\-skip 选项控制任何内置命令以及自定义用户脚本的哪些任务和钩子运行。

它接受一个逗号分隔的钩子/任务名称列表，以及预定义的 :all、:pre 和 :post 分别跳过所有钩子、所有 pre_\* 钩子和所有 post_\* 钩子。您还可以在 PDM_SKIP_HOOKS 环境变量中提供跳过列表，但一旦提供了 \-\-skip 参数，它就会被覆盖。

给定上述脚本块，运行 pdm run \-\-skip=:pre,post_test composite 将导致以下简化的生命周期：

<a>![](/img/pdm/7.png)</a>


### 高级用法

#### 自动化测试

##### 使用 Tox 作为运行器

[Tox](https://tox.readthedocs.io/en/latest/) 是一个很棒的工具，可以针对多个 Python 版本或依赖集进行测试。您可以配置一个像下面这样的 tox.ini 来将您的测试与 PDM 集成起来：

```toml
[tox]
env_list = py{36,37,38},lint

[testenv]
setenv =
    PDM_IGNORE_SAVED_PYTHON="1"
deps = pdm
commands =
    pdm install --dev
    pytest tests

[testenv:lint]
deps = pdm
commands =
    pdm install -G lint
    flake8 src/
```


要使用 Tox 创建的虚拟环境，您应该确保已设置 pdm config python.use_venv true。然后，PDM 将会将依赖项从 pdm lock 安装到虚拟环境中。在专用虚拟环境中，您可以直接运行工具，例如 pytest tests/，而不是 pdm run pytest tests/。

您还应该确保在测试命令中不要运行 pdm add/pdm remove/pdm update/pdm lock，否则 pdm lock 文件将意外地被修改。可以使用 deps 配置提供额外的依赖项。此外，应将 isolated_build 和 passenv 配置设置为上面的示例以确保 PDM 正常工作。

为了摆脱这些限制，有一个名为 tox-pdm 的 Tox 插件，可以简化使用方式。您可以通过以下方式安装它：

```shell
pip install tox-pdm
```
或者，

```shell
pdm add --dev tox-pdm
```

然后，您可以将 tox.ini 设置得更加整洁，如下所示： 

```toml
[tox]
env_list = py{36,37,38},lint

[testenv]
groups = dev
commands =
    pytest tests

[testenv:lint]
groups = lint
commands =
    flake8 src/

```
有关详细指导，请参阅项目的 [README](https://github.com/pdm-project/tox-pdm)。

##### 使用 Nox 作为运行器

[Nox](https://nox.thea.codes/) 是另一个非常棒的自动化测试工具。与 tox 不同，Nox 使用标准的 Python 文件进行配置。

在 Nox 中使用 PDM 更加容易，下面是 noxfile.py 的一个示例：


```python
import os
import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

@nox.session
def tests(session):
    session.run_always('pdm', 'install', '-G', 'test', external=True)
    session.run('pytest')

@nox.session
def lint(session):
    session.run_always('pdm', 'install', '-G', 'lint', external=True)
    session.run('flake8', '--import-order-style', 'google')
```



请注意，应设置 PDM_IGNORE_SAVED_PYTHON，以便 PDM 可以正确地选择虚拟环境中的 Python。另外，请确保 pdm 在 PATH 中可用。在运行 nox 之前，您还应该确保配置项 python.use_venv 设置为 true，以启用虚拟环境复用。

##### 关于 PEP 582 \_\_pypackages\_\_ 目录

默认情况下，如果您通过 pdm run 运行工具，程序和由其创建的所有子进程都将看到 pypackages。这意味着由这些工具创建的虚拟环境也会意识到 pypackages 中的包，这在某些情况下可能会导致意外行为。对于 nox，您可以通过在 noxfile.py 中添加一行来避免这种情况：

```python
os.environ.pop("PYTHONPATH", None)
```

对于 tox，PYTHONPATH 将不会传递到测试会话，因此这不会成为问题。此外，建议将 nox 和 tox 放在它们自己的 pipx 环境中，这样您就无需为每个项目安装。在这种情况下，PEP 582 包也不会成为问题。

#### 在持续集成中使用 PDM

只需记住一件事——PDM 无法安装在 Python < 3.7 上，因此，如果您的项目需要在这些 Python 版本上进行测试，您必须确保 PDM 已安装在正确的 Python 版本上，该版本可能与特定作业/任务运行的目标 Python 版本不同。

幸运的是，如果您使用 GitHub Action，有 [pdm-project/setup-pdm](https://github.com/marketplace/actions/setup-pdm) 可以使此过程更加简单。以下是 GitHub Actions 的一个示例工作流程，您可以根据需要调整它以适应其他 CI 平台。

```
Testing:
  runs-on: ${{ matrix.os }}
  strategy:
    matrix:
      python-version: [3.7, 3.8, 3.9, '3.10', '3.11']
      os: [ubuntu-latest, macOS-latest, windows-latest]

  steps:
    - uses: actions/checkout@v3
    - name: Set up PDM
      uses: pdm-project/setup-pdm@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pdm sync -d -G testing
    - name: Run Tests
      run: |
        pdm run -v pytest tests
```

提示：对于 GitHub Action 用户，在 Ubuntu 虚拟环境上有一个已知的[兼容性问题](https://github.com/actions/virtual-environments/issues/2803)。如果在该机器上 PDM 并行安装失败，您应该设置 parallel_install 为 false，或者设置 env LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1。这已经由 pdm-project/setup-pdm 操作处理。

注意

如果您的 CI 脚本在没有正确用户设置的情况下运行，当 PDM 尝试创建其缓存目录时，可能会出现权限错误。为了解决此问题，您可以自己设置 HOME 环境变量，将其设置为一个可写目录，例如：

```shell
export HOME=/tmp/home
```

#### 在多阶段 Dockerfile 中使用 PDM

可以在多阶段 Dockerfile 中使用 PDM，首先将项目和依赖项安装到 \_\_pypackages\_\_ 中，然后将此文件夹复制到最终阶段中，并将其添加到 PYTHONPATH 中。

```docker
# build stage
FROM python:3.8 AS builder

# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# copy files
COPY pyproject.toml pdm.lock README.md /project/
COPY src/ /project/src

# install dependencies and project into the local packages directory
WORKDIR /project
RUN mkdir __pypackages__ && pdm sync --prod --no-editable


# run stage
FROM python:3.8

# retrieve packages from build stage
ENV PYTHONPATH=/project/pkgs
COPY --from=builder /project/__pypackages__/3.8/lib /project/pkgs

# retrieve executables
COPY --from=builder /project/__pypackages__/3.8/bin/* /bin/

# set command/entrypoint, adapt to fit your needs
CMD ["python", "-m", "project"]
```

#### 使用 PDM 管理 monorepo

使用 PDM，您可以在单个项目中拥有多个子包，每个子包都有自己的 pyproject.toml 文件。并且您可以创建一个 pdm.lock 文件来锁定所有依赖项。子包可以将彼此作为它们的依赖项。要实现此目标，请按照以下步骤操作：

project/pyproject.toml:

```toml
[tool.pdm.dev-dependencies]
dev = [
    "-e file:///${PROJECT_ROOT}/packages/foo-core",
    "-e file:///${PROJECT_ROOT}/packages/foo-cli",
    "-e file:///${PROJECT_ROOT}/packages/foo-app",
]
```

packages/foo-cli/pyproject.toml:

```toml
[project]
dependencies = ["foo-core"]
```

packages/foo-app/pyproject.toml:

```toml
[project]
dependencies = ["foo-core"]
```


现在，在项目根目录中运行 pdm install，您将获得一个包含所有依赖项的 pdm.lock。所有子包都将以可编辑模式安装。

请参考 🚀 [示例库](https://github.com/pdm-project/pdm-example-monorepo) 了解更多详情。

#### 预提交钩子(pre-commit hook)

pre-commit 是一个强大的框架，用于集中管理 git 钩子。PDM 已经使用 [pre-commit](https://github.com/pdm-project/pdm/blob/main/.pre-commit-config.yaml) 钩子进行其内部质量保证检查。PDM 还公开了几个本地或 CI 流水线中可以运行的钩子。

##### 导出 requirements.txt

此钩子封装了命令 pdm export 以及任何有效的参数。它可以用作一个钩子（例如，用于 CI），以确保您将要检入代码库的 requirements.txt 反映了 pdm lock 的实际内容。

```
# export python requirements
- repo: https://github.com/pdm-project/pdm
  rev: 2.x.y # a PDM release exposing the hook
  hooks:
    - id: pdm-export
      # command arguments, e.g.:
      args: ['-o', 'requirements.txt', '--without-hashes']
      files: ^pdm.lock$

```

##### 检查 pdm.lock 是否与 pyproject.toml 保持同步

此钩子封装了命令 pdm lock --check 以及任何有效的参数。它可以用作一个钩子（例如，用于 CI），以确保每当 pyproject.toml 添加/更改/删除一个依赖项时，pdm.lock 也是最新的。

```
- repo: https://github.com/pdm-project/pdm
  rev: 2.x.y # a PDM release exposing the hook
  hooks:
    - id: pdm-lock-check
```

##### 同步当前工作集与 pdm.lock

此钩子封装了命令 pdm sync 以及任何有效的参数。它可以用作一个钩子，以确保您的当前工作集在您检出或合并分支时与 pdm.lock 同步。如果您想要使用系统的凭据存储，请将 keyring 添加到 additional_dependencies 中。

```
- repo: https://github.com/pdm-project/pdm
  rev: 2.x.y # a PDM release exposing the hook
  hooks:
    - id: pdm-sync
      additional_dependencies:
        - keyring
```


### 与虚拟环境一起工作

当您运行pdm init命令时，PDM会询问要在项目中使用的Python解释器，这是安装依赖项和运行任务的基本解释器。

与PEP 582相比，虚拟环境被认为更成熟，并且在Python生态系统以及集成开发环境中得到了更好的支持。因此，默认情况下，如果没有进行配置，虚拟环境是默认模式。

如果项目解释器（存储在.pdm-python中的解释器，可以通过pdm info检查）是来自虚拟环境，那么将使用虚拟环境。

#### 自动创建虚拟环境

默认情况下，PDM偏好使用虚拟环境布局，就像其他软件包管理器一样。当您在尚未决定Python解释器的新PDM管理的项目上首次运行pdm install时，PDM将在<project_root>/.venv中创建一个虚拟环境，并将依赖项安装到其中。在pdm init的交互会话中，PDM也会询问是否为您创建虚拟环境。

您可以选择PDM用于创建虚拟环境的后端。目前它支持三个后端：

* [virtualenv（默认）](https://virtualenv.pypa.io/)
* venv
* conda

您可以通过pdm config venv.backend [virtualenv|venv|conda]来更改。

#### 自行创建虚拟环境

您可以创建任意Python版本的多个虚拟环境。

```shell
# Create a virtualenv based on 3.8 interpreter
$ pdm venv create 3.8
# Assign a different name other than the version string
$ pdm venv create --name for-test 3.8
# Use venv as the backend to create, support 3 backends: virtualenv(default), venv, conda
$ pdm venv create --with venv 3.9
```

#### 虚拟环境的位置

如果未给出\-\-name，则PDM将在<project_root>/.venv中创建虚拟环境。否则，虚拟环境将放置在由venv.location配置指定的位置。它们的命名方式为<project_name>-<path_hash>-<name_or_python_version>，以避免名称冲突。您可以通过pdm config venv.in_project false禁用项目内的虚拟环境创建，所有虚拟环境将在venv.location下创建。

#### 重用您在其他地方创建的虚拟环境

您可以告诉PDM使用您在前面步骤中创建的虚拟环境，使用pdm use：

```shell
pdm use -f /path/to/venv
```

#### 虚拟环境的自动检测


当项目配置中没有解释器存储，或者设置了PDM_IGNORE_SAVED_PYTHON环境变量时，PDM会尝试检测要使用的可能虚拟环境：

* 项目根目录中的venv、env、.venv目录
* 当前激活的虚拟环境，除非设置了PDM_IGNORE_ACTIVE_VENV

#### 列出与该项目创建的所有虚拟环境#

```shell
$ pdm venv list
```

#### 显示虚拟环境的路径或Python解释器

```shell
$ pdm venv --path for-test
$ pdm venv --python for-test
```

#### 移除虚拟环境

```shell
$ pdm venv remove for-test
```

#### 激活虚拟环境

与其像pipenv和poetry一样生成子shell不同，pdm venv不会为您创建shell，而是将激活命令打印到控制台。这样做可以避免离开当前shell。然后，您可以将输出传递给eval来激活虚拟环境：

```shell
$ eval $(pdm venv activate for-test)
(test-project-for-test) $  # 进入虚拟环境
```

注意：venv activate不会切换项目使用的Python解释器。它只是通过将虚拟环境路径注入到环境变量中来更改shell。对于上述目的，请使用pdm use命令。

更多CLI用法请参阅pdm venv文档。

**寻找pdm shell？**

PDM不提供shell命令，因为许多复杂的shell函数在子shell中可能无法完美运行，这会给支持所有边缘情况带来维护负担。然而，您仍然可以通过以下方式获得此功能：

* 使用pdm run $SHELL，这将生成一个带有正确设置环境变量的子shell。子shell可以通过exit或Ctrl+D退出。
* 添加一个shell函数来激活虚拟环境，下面是一个在BASH中也可用的示例函数：

```bash
pdm() {
  local command=$1

  if [[ "$command" == "shell" ]]; then
      eval $(pdm venv activate)
  else
      command pdm $@
  fi
}
```

复制并粘贴此函数到您的~/.bashrc文件中，并重新启动您的shell。

对于fish shell，您可以将以下内容放入您的~/fish/config.fish或~/.config/fish/config.fish中：

```shell
function pdm
    set cmd $argv[1]

    if test "$cmd" = "shell"
        eval (pdm venv activate)
    else
        command pdm $argv
    end
end
```

现在您可以运行pdm shell来激活虚拟环境。虚拟环境可以像通常一样使用deactivate命令来停用。

#### 禁用虚拟环境模式

您可以通过pdm config python.use_venv false来禁用虚拟环境的自动创建和自动检测。如果禁用了虚拟环境，即使所选的解释器来自虚拟环境，也将始终使用PEP 582模式。

#### 在虚拟环境中包含pip

默认情况下，PDM不会在虚拟环境中包含pip。这增加了隔离性，确保只有您的依赖项安装在虚拟环境中。

要安装pip一次（例如，如果您想在CI中安装任意依赖项），您可以运行：

```shell
# Install pip in the virtual environment
$ pdm run python -m ensurepip
# Install arbitrary dependencies
# These dependencies are not checked for conflicts against lockfile dependencies!
$ pdm run python -m pip install coverage
```

或者您可以使用--with-pip选项创建虚拟环境：

```shell
$ pdm venv create --with-pip 3.9
```

有关ensurepip的更多详细信息，请参阅ensurepip文档。

如果您希望永久配置PDM在虚拟环境中包含pip，您可以使用venv.with_pip配置。

### 从模板创建项目

与yarn create和npm create类似，PDM也支持从模板初始化或创建项目。模板作为pdm init的一个位置参数给出，可以采用以下形式之一：

* pdm init django - 从模板https://github.com/pdm-project/template-django 初始化项目
* pdm init https://github.com/frostming/pdm-template-django - 从Git URL初始化项目。HTTPS和SSH URL都可接受。
* pdm init django@v2 - 检出特定的分支或标签。完整的Git URL也支持。
* pdm init /path/to/template - 从本地文件系统上的模板目录初始化项目。
* pdm init minimal - 使用内置的“minimal”模板进行初始化，仅生成一个pyproject.toml。

pdm init将使用内置的默认模板。

项目将在当前目录初始化，具有相同名称的现有文件将被覆盖。您还可以使用-p \<path>选项在新路径上创建项目。

#### 贡献模板

根据模板参数的第一种形式，pdm init <name> 将引用位于https://github.com/pdm-project/template-\<name>的模板存储库。要贡献模板，您可以创建一个模板存储库，并发起请求将所有权转让给pdm-project组织（可以在存储库设置页面的底部找到）。组织的管理员将审查请求并完成后续步骤。如果接受转让，您将被添加为存储库的维护者。

#### 模板的要求

模板存储库必须是一个基于pyproject的项目，其中包含一个符合PEP-621的元数据的pyproject.toml文件。不需要其他特殊的配置文件。

#### 项目名称替换

在初始化时，模板中的项目名称将被新项目的名称替换。这通过递归全文搜索和替换完成。导入名称是由项目名称派生的，通过将所有非字母数字字符替换为下划线并小写化。

例如，如果模板中的项目名称是foo-project，并且您想初始化一个名为bar-project的新项目，则将进行以下替换：

* 所有 .md 文件和 .rst 文件中的 foo-project -> bar-project
* 所有 .py 文件中的 foo_project -> bar_project
* 目录名称中的 foo_project -> bar_project
* 文件名中的 foo_project.py -> bar_project.py

因此，如果导入名称未从项目名称派生，我们不支持名称替换。

#### 使用其他项目生成器

如果您正在寻找更强大的项目生成器，可以使用cookiecutter通过--cookiecutter选项和copier通过--copier选项。

您需要分别安装cookiecutter和copier才能使用它们。您可以通过运行pdm self add \<package>来安装它们。要使用它们：

```shell
pdm init --cookiecutter gh:cjolowicz/cookiecutter-hypermodern-python
# or
pdm init --copier gh:pawamoy/copier-pdm --UNSAFE
```


## 使用PDM安装PaddleSpeech

最后我们通过一个例子来结束。如果查看[PaddleSpeech Issues](https://github.com/PaddlePaddle/PaddleSpeech/issues)，我们会发现很多用户会碰到安装问题(我也碰到过)。请读者安装好pdm。

### 新建项目

```shell
mkdir testpaddlespeech && cd testpaddlespeech
pdm init
```

我们都使用默认值。最后得到的pyproject.yaml如下：

```toml
[project]
name = "testpaddlespeech"
version = "0.1.0"
description = "Default template for PDM package"
authors = [
    {name = "", email = ""},
]
dependencies = []
requires-python = "==3.9.*"
readme = "README.md"
license = {text = "MIT"}


[tool.pdm]
distribution = false
```

### 修改pyproject.toml

通常我们是通过pdm add来添加依赖，不过我们也可以通过直接修改pyproject.toml然后pdm lock来达到同样的结果。为了便于读者对比，我这里直接修改pyproject.toml

```toml
[project]
name = "testpaddlespeech"
version = "0.1.0"
description = "Default template for PDM package"
authors = [
    {name = "", email = ""},
]

dependencies = [
    "pytest-runner>=6.0.1",
    "setuptools>=69.1.1",
    "paddlespeech>=1.4.1",
    "numpy<=1.23.5",
    "paddlespeech-ctcdecoders>=0.2.1",
    "opencc<=1.1.6",
    "pip>=24.0",
]



requires-python = ">=3.9.0"
readme = "README.md"
license = {text = "MIT"}


[project.optional-dependencies]
gpu_cuda102 = ["paddlepaddle-gpu==2.4.2"]
gpu_cuda112 = ["paddlepaddle-gpu==2.4.2.post112"]
gpu_cuda117 = ["paddlepaddle-gpu==2.4.2.post117"]
gpu_cuda116 = ["paddlepaddle-gpu==2.4.2.post116"]
cpu = ["paddlepaddle==2.4.2"]

[tool.pdm]
distribution = false

[[tool.pdm.source]]
type = "find_links"
url = "https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html"
name = "paddlepaddle-gpu"

```


这里有几点说明：

**1.paddlepaddle**

paddlespeech依赖与paddlepaddle，但是paddlepaddle(尤其是GPU版本)的安装需要通过[find_links](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages)的方式去特定的网站下载安装。因此我们需要配置[[tool.pdm.source]]。由于paddlepaddle不像Pytorch把CUDA/CUDNN打包到了自己的安装包里。所以它需要用户自己安装CUDA/CUDNN/TENSORRT，并且需要根据不同的CUDA/CUDNN/TENSORRT版本去安装特定版本的paddlepaddle。而且目前的pip无法根据用户安装的CUDA来自己选择合适的paddlepaddle版本，这也是为什么Pytorch使用了自己打包的方式(缺点当然是Pytorch的安装包超级大，因为它把CUDA/CUDNN都打包进去了)。因为paddlespeech文档说paddlepaddle==2.4.2比较匹配，所以我这里固定了paddlepaddle的版本。关于这个话题感兴趣的读者可以参考[这个issue](https://github.com/pdm-project/pdm/issues/2658)和[Idea: selector packages](https://discuss.python.org/t/idea-selector-packages/4463/31)。更多paddlepaddle 2.4.2的安装请参考[安装文档](https://www.paddlepaddle.org.cn/install/old?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

**2.numpy**

详情参考[paddle的numpy库版本和paddlespeech冲突](https://github.com/PaddlePaddle/PaddleSpeech/issues/3618)。大概情况就是librosa(0.8.1)会用到np.complex，但是1.26.2就没有这个了。这就是常见的问题，librosa在开发0.8.1版本时指定的依赖是>=1.15.0，那个时候可能还没有numpy 1.26.2。但是到了现在numpy 1.26.2出来后，就不兼容了。所以作为库的开发者(比如librosa)，要么比较保守，在指定依赖时加上"<1.20"。但是这也有问题，因为像numpy这样的库，其它的库也会经常引用，如果另外一个库需要numpy>=1.20，那就完了。因此库的开发者需要不断测试，并且更新。对于冲突，我们通常可以使用pdm list \-\-tree来分析：

```
pdm list --tree -r numpy
numpy 1.23.5 
├── bottleneck 1.3.8 [ requires: Any ]
│   └── nara-wpe 0.0.9 [ requires: Any ]
│       └── paddlespeech 1.4.1 [ requires: Any ]

..................

│       └── paddlespeech 1.4.1 [ requires: >=1.1.0 ]
├── librosa 0.8.1 [ requires: >=1.15.0 ]
│   ├── paddleaudio 1.1.0 [ requires: ==0.8.1 ]
│   │   └── paddlespeech 1.4.1 [ requires: >=1.1.0 ]
│   └── paddlespeech 1.4.1 [ requires: ==0.8.1 ]
├── matplotlib 3.8.3 [ requires: <2,>=1.21 ]
│   ├── paddleslim 2.6.0 [ requires: Any ]
│   │   └── paddlespeech 1.4.1 [ requires: >=2.3.4 ]
│   ├── paddlespeech 1.4.1 [ requires: Any ]
│   └── visualdl 2.4.2 [ requires: Any ]
│       └── paddlenlp 2.5.2 [ requires: Any ]
│           ├── paddlespeech 1.4.1 [ requires: >=2.4.8 ]
│           └── ppdiffusers 0.14.2 [ requires: >=2.5.2 ]
│               └── paddlespeech 1.4.1 [ requires: >=0.9.0 ]
.........................
```


为了解决这个问题，我在toml里加了"numpy<=1.23.5"。

**3.opencc**

参考[这个issue](https://github.com/PaddlePaddle/PaddleSpeech/issues/3617)。paddlespeech里没有指定opecc的版本，默认安装最新的版本，但是最新的版本要求GLIBC_2.32，如果用户的GLIBC版本不对的话也会出现问题。但是千万不要升级GLIBC！！！关于这个问题，感兴趣的读者可以参考[在同一台Linux机器上安装多个版本GLIBC](/2024/03/07/multi-glibc-patchelf/)。


```
$ pdm list --tree -r opencc
opencc 1.1.6 
└── paddlespeech 1.4.1 [ requires: Any ]
```

### 安装

如果读者使用cpu版本的，请运行：

```shell
pdm install -G cpu
```

安装cuda11.7版本的paddlepaddle：

```shell
pdm install -G gpu_cuda117
```

注意：取决于网络不同，上面的过程可能时间会比较长。读者可能需要修改pypi镜像为清华的源。可以参考前面的文档。

```
pdm config --local pypi.url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 导出为requirements.txt

如果有的地方不支持pdm，那么可以导出为requirements.txt后使用pip安装：

```shell
pdm export --no-hashes > requirements.txt
```

requirements.txt输出内容为：

```
aiohttp==3.9.3
aiosignal==1.3.1
annotated-types==0.6.0
anyio==4.3.0
astor==0.8.1
async-timeout==4.0.3; python_version < "3.11"
attrs==23.2.0
audioread==3.0.1
babel==2.14.0
bce-python-sdk==0.9.5
blinker==1.7.0
bottleneck==1.3.8
braceexpand==0.1.7
certifi==2024.2.2
cffi==1.16.0
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
coloredlogs==15.0.1
colorlog==6.8.2
contourpy==1.2.0
cycler==0.12.1
cython==3.0.9
datasets==2.18.0
decorator==5.1.1
dill==0.3.4
distance==0.1.3
editdistance==0.8.1
exceptiongroup==1.2.0; python_version < "3.11"
fastapi==0.110.0
filelock==3.13.1
flask==3.0.2
flask-babel==2.0.0
flatbuffers==24.3.7
fonttools==4.49.0
frozenlist==1.4.1
fsspec==2024.2.0
ftfy==6.1.3
future==1.0.0
g2p-en==2.1.0
g2pm==0.1.2.5
h11==0.14.0
h5py==3.10.0
huggingface-hub==0.21.4
humanfriendly==10.0
hyperpyyaml==1.2.2
idna==3.6
importlib-metadata==7.0.2; python_version < "3.10"
importlib-resources==6.3.0; python_version < "3.10"
inflect==7.0.0
itsdangerous==2.1.2
jieba==0.42.1
jinja2==3.1.3
joblib==1.3.2
jsonlines==4.0.0
kaldiio==2.18.0
kiwisolver==1.4.5
librosa==0.8.1
llvmlite==0.42.0
loguru==0.7.2
lxml==5.1.0
markdown-it-py==3.0.0
markupsafe==2.1.5
matplotlib==3.8.3
mdurl==0.1.2
mock==5.1.0
mpmath==1.3.0
multidict==6.0.5
multiprocess==0.70.12.2
nara-wpe==0.0.9
nltk==3.8.1
numba==0.59.0
numpy==1.23.5
onnxruntime==1.17.1
opencc==1.1.6
opencc-python-reimplemented==0.1.7
opencv-python==4.6.0.66
opt-einsum==3.3.0
packaging==24.0
paddle-bfloat==0.1.7
paddle2onnx==1.1.0
paddleaudio==1.1.0
paddlefsl==1.1.0
paddlenlp==2.5.2
paddlepaddle==2.4.2
paddleslim==2.6.0
paddlespeech==1.4.1
paddlespeech-ctcdecoders==0.2.1
paddlespeech-feat==0.1.0
pandas==2.1.0
parameterized==0.9.0
pathos==0.2.8
pattern-singleton==1.2.0
pillow==10.2.0
pip==24.0
platformdirs==4.2.0
pooch==1.8.1
portalocker==2.8.2
pox==0.3.4
ppdiffusers==0.14.2
ppft==1.7.6.8
praatio==5.1.1
prettytable==3.10.0
protobuf==3.20.0
pyarrow==15.0.1
pyarrow-hotfix==0.6
pybind11==2.11.1
pycparser==2.21
pycryptodome==3.20.0
pydantic==2.6.4
pydantic-core==2.16.3
pygments==2.17.2
pygtrie==2.5.0
pyparsing==3.1.2
pypinyin==0.44.0
pypinyin-dict==0.8.0
pyreadline3==3.4.1; sys_platform == "win32" and python_version >= "3.8"
pytest-runner==6.0.1
python-dateutil==2.9.0.post0
pytz==2024.1
pywin32==306; platform_system == "Windows"
pyworld==0.3.4
pyyaml==6.0.1
pyzmq==25.1.2
regex==2023.12.25
requests==2.31.0
resampy==0.4.3
rich==13.7.1
ruamel-yaml==0.18.6
ruamel-yaml-clib==0.2.8; platform_python_implementation == "CPython" and python_version < "3.13"
sacrebleu==2.4.1
safetensors==0.4.2
scikit-learn==1.4.1.post1
scipy==1.12.0
sentencepiece==0.2.0
seqeval==1.2.2
setuptools==69.2.0
six==1.16.0
sniffio==1.3.1
soundfile==0.12.1
starlette==0.36.3
swig==4.2.1
sympy==1.12
tabulate==0.9.0
textgrid==1.6.1
threadpoolctl==3.3.0
timer==0.2.2
tojyutping==0.2.3
tqdm==4.66.2
typeguard==2.13.3
typer==0.9.0
typing-extensions==4.10.0
tzdata==2024.1
urllib3==2.2.1
uvicorn==0.28.0
visualdl==2.4.2
wcwidth==0.2.13
webrtcvad==2.0.10
websockets==12.0
werkzeug==3.0.1
win32-setctime==1.1.0; sys_platform == "win32"
xxhash==3.4.1
yacs==0.1.8
yarl==1.9.4
zhon==2.0.2
zipp==3.18.0; python_version < "3.10" 
--find-links https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

如果读者不想使用PDM，那么直接用pip install -r requirements.txt安装。注意：这是CPU版本的！如果要安装GPU版本，请修改上面文件，把"paddlepaddle==2.4.2"改为"paddlepaddle-gpu==2.4.2.post117"

### 使用

```shell
$ eval $(pdm venv activate)
$ wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
$ wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
$ paddlespeech asr --lang zh --input zh.wav
我认为跑步最重要的就是给我带来了身体健康
```


