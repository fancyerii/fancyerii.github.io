---
layout:     post
title:      "翻译：Building and Distributing Packages with Setuptools"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - python
    - pip
    - pipenv
    - conda
---

本文是[Building and Distributing Packages with Setuptools](https://setuptools.pypa.io/en/latest/userguide/index.html)的翻译。

 <!--more-->



分享 Python 库或程序的第一步是构建分发包(distribution package)。【在 Python 社区中，分发包也简单地称为“包”。不幸的是，对于新用户来说，这个术语可能会有点令人困惑，因为术语“包”也可以指任何用于组织模块和辅助文件的目录（或子目录）。】这包括添加一组额外的文件，其中包含元数据和配置，不仅指导 setuptools 构建分发的方式，还帮助安装程序（如 pip）在安装过程中进行操作。

本文档包含了帮助 Python 开发人员完成此过程的信息。请查看快速入门以获取工作流程的概述。

还请注意，setuptools 是社区中称为构建后端的内容，用户界面由 pip 和 build 等工具提供。要使用 setuptools，必须明确创建一个如 Build System Support 中描述的 pyproject.toml 文件。

**目录**
* TOC
{:toc} 

## 快速入门

### 安装

您可以使用pip安装最新版本的setuptools：

```shell
pip install --upgrade setuptools
```

然而，大多数情况下，您不需要安装。

相反，当创建新的Python包时，建议使用一个称为build的命令行工具。此工具将自动下载setuptools和您的项目可能具有的任何其他构建时依赖项。您只需要在包的根目录下的pyproject.toml文件中指定它们，如下一节所示。

您也可以使用pip安装build：

```shell
pip install --upgrade build
```

这将允许您运行以下命令：python -m build。

**重要提示**

请注意，一些操作系统可能配备了python3和pip3命令，而不是python和pip（但它们应该是等价的）。如果您的系统中没有pip或pip3可用，请查看pip安装文档。

每个Python包必须提供一个pyproject.toml文件，并指定它想要使用的后端（构建系统）。然后，可以使用提供类似于build sdist功能的任何工具生成分发。

### 基本用法

当创建一个Python包时，您必须提供一个包含类似于下面示例的构建系统部分的pyproject.toml文件：

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

此部分声明了您的构建系统依赖项是什么，以及将用于实际打包的库。

备注：在历史上，这份文档不必要地列出了wheel在requires列表中，并且许多项目仍然这样做。这是不推荐的。当需要时，后端会自动添加wheel依赖项，并且显式列出它会导致在源分发构建时不必要地需要它。只有在构建时需要显式访问它时（例如，如果您的项目需要一个导入wheel的setup.py脚本），才应将wheel包含在requires中。

除了指定构建系统之外，您还需要添加一些包信息，例如元数据、内容、依赖项等。这可以在同一个pyproject.toml文件中完成，也可以在分离的文件中完成：setup.cfg或setup.py 。

新项目建议在自定义构建过程中不必要时避免使用setup.py配置（除了最小的stub代码）。本文档中保留了示例，以帮助对维护或贡献给现有使用setup.py的包感兴趣的人。请注意，您仍然可以在setup.cfg或pyproject.toml中保持大部分配置为声明性，并且仅在这些文件中不支持的部分（例如C扩展）中使用setup.py。

以下示例演示了一个最小配置（假设项目依赖于requests和importlib-metadata以便运行）：


**pyproject.toml**

```toml
[project]
name = "mypackage"
version = "0.0.1"
dependencies = [
    "requests",
    'importlib-metadata; python_version<"3.10"',
]
```

详细了解使用pyproject.toml文件配置setuptools，请参阅[通过pyproject.toml配置setuptools](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)

**setup.cfg**

```
[metadata]
name = mypackage
version = 0.0.1

[options]
install_requires =
    requests
    importlib-metadata; python_version<"3.10"
```

详细了解使用setup.cfg文件配置setuptools，请参阅[通过setup.cfg配置setuptools](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html)

**setup.py**

```python
from setuptools import setup

setup(
    name='mypackage',
    version='0.0.1',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
    ],
)
```

关于setup.py的关键词请参考[Keywords](https://setuptools.pypa.io/en/latest/references/keywords.html) 。


最后，您需要组织您的Python代码，使其准备好分发到类似以下的东西中（可选文件标有＃）：

```
mypackage
├── pyproject.toml  # and/or setup.cfg/setup.py (depending on the configuration method)
|   # README.rst or README.md (a nice description of your package)
|   # LICENCE (properly chosen license information, e.g. MIT, BSD-3, GPL-3, MPL-2, etc...)
└── mypackage
    ├── __init__.py
    └── ... (other Python files)
```

在系统中安装了build后，您可以运行：

```shell
python -m build
```

现在您已经准备好了您的分发（例如，在dist目录中有一个.tar.gz文件和一个.whl文件），您可以[上传](https://twine.readthedocs.io/en/stable/index.html)到[PyPI](https://pypi.org/)！

当然，在将项目发布到PyPI之前，您会希望添加更多信息，以帮助人们找到或了解您的项目。也许到那时，您的项目已经增长，包括一些依赖项，也许还包括一些数据文件和脚本。在接下来的几节中，我们将逐步介绍您需要指定的额外但重要的信息，以正确打包您的项目。

**信息：使用setup.py**

setuptools提供了对setup.py文件的一流支持作为配置机制。

然而，重要的是要记住，强烈不推荐运行此文件作为脚本（例如 python setup.py sdist），大多数命令行界面都已经或将被弃用（例如 python setup.py install、python setup.py bdist_wininst 等）。

我们还建议用户尽可能多地通过更具声明性的方式在pyproject.toml或setup.cfg中公开配置，保持setup.py的最小化，仅包含动态部分（如果适用，甚至可以完全省略它）。

有关更多背景信息，请参阅[为什么不直接调用setup.py](https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html)。

### 概述

#### 包发现

对于遵循简单目录结构的项目，setuptools应该能够自动检测所有包和命名空间。然而，复杂的项目可能包括额外的文件夹和支持文件，这些文件不一定应该分发（或可能会混淆setuptools的自动发现算法）。

因此，setuptools提供了一种方便的方式来自定义应该分发哪些包以及它们应该在哪个目录中找到，如下面的示例所示：


pyproject.toml
```toml
# ...
[tool.setuptools.packages]
find = {}  # Scan the project directory with the default parameters

# OR
[tool.setuptools.packages.find]
# All the following settings are optional:
where = ["src"]  # ["."] by default
include = ["mypackage*"]  # ["*"] by default
exclude = ["mypackage.tests*"]  # empty by default
namespaces = false  # true by default
```

setup.cfg
```
[options]
packages = find: # OR `find_namespace:` if you want to use namespaces

[options.packages.find]  # (always `find` even if `find_namespace:` was used before)
# This section is optional as well as each of the following options:
where=src  # . by default
include=mypackage*  # * by default
exclude=mypackage.tests*  # empty by default
```

setup.py
```python
from setuptools import setup, find_packages  # or find_namespace_packages

setup(
    # ...
    packages=find_packages(
        # All keyword arguments below are optional:
        where='src',  # '.' by default
        include=['mypackage*'],  # ['*'] by default
        exclude=['mypackage.tests'],  # empty by default
    ),
    # ...
)
```

当您传递以上信息以及其他必要信息时，setuptools将遍历在where中指定的目录（默认为.），并过滤它可以找到的包，遵循包含模式（默认为*），然后删除与排除模式匹配的包（默认为空），并返回一个Python包列表。

有关更多详细信息和高级用法，请参阅[包发现和命名空间包](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#package-discovery)。

提示：从版本61.0.0开始，setuptools的自动发现能力已经得到改进，可以检测到流行的项目布局（如平面布局和src布局），而不需要任何特殊配置。查看我们的参考文档以获取更多信息。

#### 入口点和自动脚本创建

setuptools支持在安装时自动创建脚本，如果您将它们指定为入口点，则会在安装时运行包内的代码。这个功能在 pip 中的应用示例是，它允许你运行类似 pip install 的命令，而不必键入 python -m pip install。以下配置示例显示了如何实现此功能：


pyproject.toml
```toml
[project.scripts]
cli-name = "mypkg.mymodule:some_func"
```

setup.cfg
```
[options.entry_points]
console_scripts =
    cli-name = mypkg.mymodule:some_func
```

setup.py
```python
setup(
    # ...
    entry_points={
        'console_scripts': [
            'cli-name = mypkg.mymodule:some_func',
        ]
    }
)
```



当安装此项目时，将创建一个名为cli-name的可执行文件。当用户调用时，cli-name将调用mypkg/mymodule.py文件中的some_func函数。请注意，您还可以使用入口点机制在安装的包之间进行广告(advertise，广而告之)并实现插件系统。有关详细使用方法，请参阅[入口点](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)。

#### 依赖管理

使用setuptools构建的包可以指定在安装包本身时自动安装的依赖项。下面的示例显示了如何配置此类依赖项：
 
pyproject.toml
```toml
[project]
# ...
dependencies = [
    "docutils",
    "requests <= 0.4",
]
# ...
```

setup.cfg
```
[options]
install_requires =
    docutils
    requests <= 0.4
```

setup.py
```python
setup(
    # ...
    install_requires=["docutils", "requests <= 0.4"],
    # ...
)
```

每个依赖项都由一个字符串表示，该字符串可以选择包含版本要求（例如 \<、>、\<=、>=、== 或 != 中的一个运算符，后跟一个版本标识符），和/或条件环境标记，例如 sys_platform == "win32"（有关更多信息，请参阅[版本说明符](https://packaging.python.org/en/latest/specifications/version-specifiers/)）。

当安装您的项目时，将会定位所有尚未安装的依赖项（通过PyPI），下载、构建（如果需要）并安装它们。当然，这只是一个简化的场景。您还可以指定一些不严格要求您的包工作的额外依赖项组，但它们将提供额外的功能。有关更高级的用法，请参阅setuptools中的[依赖管理](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html)。

#### 包含数据文件
setuptools提供了三种指定要包含在您的包中的数据文件的方法。对于最简单的用法，您可以简单地使用include_package_data关键字：
 
pyproject.toml
```toml
[tool.setuptools]
include-package-data = true
# This is already the default behaviour if you are using
# pyproject.toml to configure your build.
# You can deactivate that with `include-package-data = false`
```

setup.cfg
```
[options]
include_package_data = True
```

setup.py
```python
setup(
    # ...
    include_package_data=True,
    # ...
)
```



这告诉setuptools安装它在您的包中找到的任何数据文件。数据文件必须通过[MANIFEST.in](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#using-manifest-in)文件指定，或者由[版本控制系统插件](https://setuptools.pypa.io/en/latest/userguide/extension.html#adding-support-for-revision-control-systems)自动添加。有关更多详细信息，请参阅[数据文件支持](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)。

#### 开发模式

setuptools允许您在不复制任何文件到解释器目录（例如site-packages目录）的情况下安装包。这允许您修改源代码，并使更改生效，而无需重新构建和重新安装。这是如何做到的：

```shell
pip install --editable .
```

有关更多信息，请参阅[开发模式（即“可编辑安装”）](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)。

**提示**

在pip v21.1之前，要与开发模式兼容，需要一个setup.py脚本。使用较新版本的pip，如果项目没有setup.py，则可能以此模式安装。

如果您的pip版本早于v21.1或使用不支持PEP 660的不同打包相关工具，则可能需要在您的存储库中保留setup.py文件，以便使用可编辑的安装。

一个简单的脚本就足够了，例如：

```python
from setuptools import setup

setup()
```

您仍然可以在pyproject.toml和/或setup.cfg中保留所有配置。

**备注**

从源代码构建（例如，通过python -m build或pip install -e .）时，可能会创建一些托管构建工件和缓存文件的目录，例如build、dist、*.egg-info [2]。您可以配置您的版本控制系统以忽略它们（参见GitHub的.gitignore模板示例）。

#### 将您的包上传到PyPI

生成分发文件后，下一步将是上传您的分发文件，以便其他人可以使用它。这个功能由[twine](https://pypi.org/project/twine)提供，在[Python打包教程](https://packaging.python.org/en/latest/tutorials/packaging-projects/)中有文档记录。

#### 从setup.py过渡到声明性配置
为了避免执行任意脚本和模板代码，我们正在从运行setup()来定义所有包信息过渡到以声明性方式——使用pyproject.toml（或较旧的setup.cfg）。

为了简化过渡的挑战，我们提供了一个快速指南，以了解setuptools如何解析[pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)（或者[setup.cfg](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html)）。

**备注**

setuptools希望最终使用单个声明性格式（pyproject.toml）而不是维护两个（pyproject.toml / setup.cfg）。然而，setup.cfg可能会继续维护很长一段时间。

### Python打包资源

在Python中进行打包可能很困难，并且在不断发展。[Python打包用户指南](https://packaging.python.org/)提供了教程和最新的参考资料，可以在分发您的工作时帮助您。

## 包发现和命名空间包

注： 关键字的完整规范可以在[关键字参考](https://setuptools.pypa.io/en/latest/references/keywords.html)中找到。

重要： 这里提供的示例仅用于演示所介绍的功能。如果要在您的系统上复制它们，需要提供更多的元数据和选项参数。如果您对 setuptools 完全不熟悉，可以从快速入门部分开始。

Setuptools 提供了强大的工具来处理包的发现，包括支持命名空间包。

通常，您会按以下方式手动指定要包含的包：

setup.cfg
```
[options]
#...
packages =
    mypkg
    mypkg.subpkg1
    mypkg.subpkg2
```

setup.py
```python
setup(
    # ...
    packages=['mypkg', 'mypkg.subpkg1', 'mypkg.subpkg2']
)
```

pyproject.toml
```toml
# ...
[tool.setuptools]
packages = ["mypkg", "mypkg.subpkg1", "mypkg.subpkg2"]
# ...
```


如果您的包不在存储库的根目录中，或者与目录结构不完全对应，您还需要配置 package_dir。这可能会很快让人感到疲倦。为了加快速度，您可以依赖 setuptools 的自动发现功能，或者使用提供的工具，如下面的部分所述。

重要提示：虽然 setuptools 允许开发人员在目录名称和包名称之间创建非常复杂的映射关系，但最好保持简单，将所需的包层次结构反映在目录结构中，保持相同的名称。

### 自动发现

默认情况下，setuptools 将考虑 2 种流行的项目布局，每种布局都有其各自的优点和缺点，如下面的部分所述。

Setuptools 将自动扫描您的项目目录，寻找这些布局，并尝试猜测 [packages](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#declarative-config) 和 [py_modules](https://setuptools.pypa.io/en/latest/references/keywords.html) 配置的正确值。

**重要提示**

只有在没有为 packages 和 py_modules 提供任何配置时，自动发现才会启用。如果其中至少有一个被明确设置，自动发现就不会发生。

注意：指定 ext_modules 也可能阻止自动发现的发生，除非您选择使用 [pyproject.toml 文件配置](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) setuptools（这将禁用向后兼容的行为）。

#### src-layout

项目应该包含一个 src 目录，位于项目根目录下，所有用于分发的模块和包都放在此目录中：

```
project_root_directory
├── pyproject.toml  # AND/OR setup.cfg, setup.py
├── ...
└── src/
    └── mypkg/
        ├── __init__.py
        ├── ...
        ├── module.py
        ├── subpkg1/
        │   ├── __init__.py
        │   ├── ...
        │   └── module1.py
        └── subpkg2/
            ├── __init__.py
            ├── ...
            └── module2.py
```

这种布局在您希望使用自动发现时非常方便，因为您不必担心项目根目录中的其他 Python 文件或文件夹被错误地分发。在某些情况下，对于测试或使用 PEP 420 格式的包来说，这种布局也可能更少出错。另一方面，您不能依赖隐式的 PYTHONPATH=. 来启动 Python REPL 并使用您的包（您需要进行可编辑安装才能这样做）。

#### flat-layout（也称为“adhoc”）

包文件夹直接放置在项目根目录下：

```
project_root_directory
├── pyproject.toml  # AND/OR setup.cfg, setup.py
├── ...
└── mypkg/
    ├── __init__.py
    ├── ...
    ├── module.py
    ├── subpkg1/
    │   ├── __init__.py
    │   ├── ...
    │   └── module1.py
    └── subpkg2/
        ├── __init__.py
        ├── ...
        └── module2.py
```

为避免混淆，在 flat-layout 的情况下，已经被流行工具使用的文件和文件夹名称（或者对应于众所周知的约定，如将文档与项目代码一起分发）会被自动过滤掉。

保留package名字：

```python
FlatLayoutPackageFinder.DEFAULT_EXCLUDE: Tuple[str, ...] = ('ci', 'ci.*', 'bin', 'bin.*', 'debian', 'debian.*', 'doc', 'doc.*', 'docs', 'docs.*', 'documentation', 'documentation.*', 'manpages', 'manpages.*', 'news', 'news.*', 'newsfragments', 'newsfragments.*', 'changelog', 'changelog.*', 'test', 'test.*', 'tests', 'tests.*', 'unit_test', 'unit_test.*', 'unit_tests', 'unit_tests.*', 'example', 'example.*', 'examples', 'examples.*', 'scripts', 'scripts.*', 'tools', 'tools.*', 'util', 'util.*', 'utils', 'utils.*', 'python', 'python.*', 'build', 'build.*', 'dist', 'dist.*', 'venv', 'venv.*', 'env', 'env.*', 'requirements', 'requirements.*', 'tasks', 'tasks.*', 'fabfile', 'fabfile.*', 'site_scons', 'site_scons.*', 'benchmark', 'benchmark.*', 'benchmarks', 'benchmarks.*', 'exercise', 'exercise.*', 'exercises', 'exercises.*', 'htmlcov', 'htmlcov.*', '[._]*', '[._]*.*')
```


保留module名字：

```python
FlatLayoutModuleFinder.DEFAULT_EXCLUDE: Tuple[str, ...] = ('setup', 'conftest', 'test', 'tests', 'example', 'examples', 'build', 'toxfile', 'noxfile', 'pavement', 'dodo', 'tasks', 'fabfile', '[Ss][Cc]onstruct', 'conanfile', 'manage', 'benchmark', 'benchmarks', 'exercise', 'exercises', '[._]*')
```


警告：如果您在 flat-layout 中使用自动发现，setuptools 将拒绝创建包含多个顶级包或模块的[分发存档](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package)。

这样做是为了防止常见的错误，例如意外发布不适合分发的代码（例如与维护相关的脚本）。建议那些有意创建多包分发的用户使用[自定义发现](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#custom-discovery)或 src-layout。

对于可以用单个 Python 文件实现的实用程序/库，也有一种方便的 flat-layout 变体：

##### 单模块分发

一个独立的模块直接放置在项目根目录下，而不是放在包文件夹内：

```
project_root_directory
├── pyproject.toml  # AND/OR setup.cfg, setup.py
├── ...
└── single_file_lib.py
```

### 自定义发现

如果自动发现对您不起作用（例如，您希望在分发中包含具有保留名称的顶级包，例如 tasks、example 或 docs，或者您希望排除否则会包含的嵌套包），您可以使用提供的包发现工具：

setup.cfg:
```
# ...
[tool.setuptools.packages]
find = {}  # Scanning implicit namespaces is active by default
# OR
find = {namespaces = false}  # Disable implicit namespaces
```

setup.py
```python
from setuptools import find_packages
# or
from setuptools import find_namespace_packages
```

pyproject.toml
```toml
# ...
[tool.setuptools.packages]
find = {}  # Scanning implicit namespaces is active by default
# OR
find = {namespaces = false}  # Disable implicit namespaces
```

#### 查找简单的包

让我们从第一个工具开始。find：（find_packages（））接受一个源目录和两个包名称模式列表（要排除和包含的），然后返回一个字符串列表，表示它找到的包。要使用它，请考虑以下目录：


```
mypkg
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    ├── pkg1
    │   └── __init__.py
    ├── pkg2
    │   └── __init__.py
    ├── additional
    │   └── __init__.py
    └── pkg
        └── namespace
            └── __init__.py
```


为了让 setuptools 自动包含以 pkg 开头且不含额外内容的 src 中找到的包：

setup.cfg
```
[options]
packages = find:
package_dir =
    =src

[options.packages.find]
where = src
include = pkg*
# alternatively: `exclude = additional*`
```

setup.py
```python
setup(
    # ...
    packages=find_packages(
        where='src',
        include=['pkg*'],  # alternatively: `exclude=['additional*']`
    ),
    package_dir={"": "src"}
    # ...
)
```

pyproject.toml
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["pkg*"]  # alternatively: `exclude = ["additional*"]`
namespaces = false
```


注意： pkg 不包含 init.py 文件，因此 pkg.namespace 被 find: 忽略（参见下面的 find_namespace:）。


重要提示：include 和 exclude 接受表示 glob 模式的字符串。这些模式应该与 Python 模块的完整名称匹配（就像在 import 语句中写的一样）。例如，如果您有 util 模式，它将匹配 util/init.py，但不会匹配 util/files/init.py。

父包是否与模式匹配不会决定子模块是否包含在分发中。如果要使模式也匹配子模块，则需要显式添加通配符（例如，util*）。


#### 查找命名空间包

setuptools 提供了 find_namespace：（find_namespace_packages（）），它的行为类似于 find:，但适用于命名空间包。

在深入之前，重要的是要对[命名空间包](https://peps.python.org/pep-0420/)有一个很好的理解。以下是一个快速回顾。

当您有两个按如下组织的包时：

```
/Users/Desktop/timmins/foo/__init__.py
/Library/timmins/bar/__init__.py
```

如果 Desktop 和 Library 都在您的 PYTHONPATH 中，那么当您调用 import 机制时，将自动为您创建一个名为 timmins 的命名空间包，允许您执行以下操作：

```python
>>>import timmins.foo
>>>import timmins.bar
```
就好像您的系统上只有一个 timmins。然后，这两个包可以分别分发和单独安装，而不会影响另一个。

现在，假设您决定将 foo 部分打包进行分发，并且首先创建一个如下组织的项目目录：

```
foo
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    └── timmins
        └── foo
            └── __init__.py
```

如果您希望 timmins.foo 被自动包含在分发中，则需要指定：

setup.cfg
```
[options]
package_dir =
    =src
packages = find_namespace:

[options.packages.find]
where = src
```

find: 不起作用，因为 timmins 不直接包含 init.py，而是必须使用 find_namespace:。

您可以将 find_namespace: 视为与 find: 相同，只是它会将一个目录视为一个包，即使它不直接包含 init.py 文件。

setup.py
```python
setup(
    # ...
    packages=find_namespace_packages(where='src'),
    package_dir={"": "src"}
    # ...
)
```

当您使用 find_packages() 时，所有没有 init.py 文件的目录都将被忽略。另一方面，find_namespace_packages() 将扫描所有目录。

pyproject.toml
```toml
[tool.setuptools.packages.find]
where = ["src"]
```

当在 pyproject.toml 中使用 tool.setuptools.packages.find，setuptools 在扫描项目目录时将默认考虑[隐式命名空间](https://peps.python.org/pep-0420/)。


在 pyproject.toml 中使用 tool.setuptools.packages.find，setuptools 将在扫描项目目录时默认考虑隐式命名空间。

安装包分发后，timmins.foo 将可供您的解释器使用。

警告：请注意，如果您使用 flat-layout，则 find_namespace：（setup.cfg），find_namespace_packages（）（setup.py）和 find（pyproject.toml）将扫描您项目目录中的所有文件夹。

如果使用不当，这可能会导致不需要的文件被添加到最终的 wheel 中。例如，如果项目目录组织如下：

```
foo
├── docs
│   └── conf.py
├── timmins
│   └── foo
│       └── __init__.py
└── tests
    └── tests_foo
        └── __init__.py
```

最终用户将安装不仅 timmins.foo，还有 docs 和 tests.tests_foo。

修复这个问题的简单方法是采用上述提到的 src-layout，或者确保正确配置 include 和/或 exclude。

提示：在[构建](https://setuptools.pypa.io/en/latest/build_meta.html#building)您的包之后，您可以通过运行以下命令来查看所有文件是否正确（没有遗漏或额外的文件）：

```shell
tar tf dist/*.tar.gz
unzip -l dist/*.whl
```

这需要在您的操作系统中安装 tar 和 unzip。在 Windows 上，您还可以使用 [7zip](https://www.7-zip.org/) 等 GUI 程序。


### 老的命名空间包

在上面可以如此轻松地创建命名空间包的事实归功于 PEP 420。过去要实现相同的结果可能会更加繁琐。历史上，有两种方法可以创建命名空间包。一种是由 setuptools 支持的 pkg_resources 风格，另一种是由 Python 的 pkgutils 模块提供的 pkgutils 风格。尽管这两种风格仍然存在于许多现有的包中，但它们现在都被视为已弃用。这两种方法在许多微妙但重要的方面有所不同，您可以在 Python [打包用户指南](https://packaging.python.org/guides/packaging-namespace-packages/)中找到更多信息。

#### pkg_resources 风格命名空间包

这是 setuptools 直接支持的方法。从相同的布局开始，您需要添加两个部分。首先，在您的命名空间包目录下直接添加一个 \_\_init\_
_.py 文件，其内容如下：

```python
__import__("pkg_resources").declare_namespace(__name__)
```


并在您的 setup.cfg 或 setup.py 中添加 namespace_packages 关键字：

setup.cfg
```
[options]
namespace_packages = timmins
```

setup.py
```python
setup(
    # ...
    namespace_packages=['timmins']
)
```

然后您的目录应该是这样的：

```
foo
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    └── timmins
        ├── __init__.py
        └── foo
            └── __init__.py
```

对其他包也重复相同的操作，您就可以达到与前一节相同的结果。

#### pkgutil 风格命名空间包

这种方法与 pkg_resources 风格几乎相同，只是省略了 namespace_packages 声明，并且 \_\_init\_
_.py 文件包含以下内容：

```python
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

项目布局保持不变，pyproject.toml/setup.cfg 也保持不变。


## 在Setuptools中依赖管理

Setuptools 提供了三种依赖类型的管理方式：1) 构建系统需求，2) 必需依赖项，3) 可选依赖项。

每个依赖项，无论是哪种类型，都需要按照 [PEP 508](https://peps.python.org/pep-0508/) 和 [PEP 440](https://peps.python.org/pep-0440/) 进行指定。这允许添加[版本范围限制](https://peps.python.org/pep-0440/#version-specifiers)和[环境标记](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#environment-markers)。

### 构建系统需求

在整理所有脚本和文件并准备打包时，需要一种方式来指定实际需要进行打包的程序和库（在我们的情况下，当然是 setuptools）。这需要在您的 pyproject.toml 文件中进行指定（如果您忘记了这是什么，请查看[快速入门](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)或[构建系统支持](https://setuptools.pypa.io/en/latest/build_meta.html)）：

```toml
[build-system]
requires = ["setuptools"]
#...
```

请注意，您还应该在这里包括任何其他 setuptools 插件（例如 [setuptools-scm](https://pypi.org/project/setuptools-scm)、[setuptools-golang](https://pypi.org/project/setuptools-golang)、[setuptools-rust](https://pypi.org/project/setuptools-rust)）或构建时依赖项（例如 [Cython](https://pypi.org/project/Cython)、[cppy](https://pypi.org/project/cppy)、[pybind11](https://pypi.org/project/pybind11)）。

注意：在以前的 setuptools 版本中，这是通过 setup_requires 关键字实现的，但现在已被视为弃用，而更倾向于上述描述的 [PEP 517](https://peps.python.org/pep-0517/) 风格。要查看如何使用这个已弃用的关键字，请参考我们[关于弃用做法的指南（WIP）](https://setuptools.pypa.io/en/latest/deprecated/index.html)。

### 声明必需依赖项

这是一个包声明其核心依赖项的地方，没有这些依赖项，包将无法运行。setuptools 支持在安装包时自动下载和安装这些依赖项。虽然这其中有更多的细节，但我们从一个简单的示例开始。

pyproject.toml
```toml
[project]
# ...
dependencies = [
    "docutils",
    "BazSpam == 1.1",
]
# ...
```

setup.cfg
```
[options]
#...
install_requires =
    docutils
    BazSpam ==1.1
```


setup.py
```python
setup(
    ...,
    install_requires=[
        'docutils',
        'BazSpam ==1.1',
    ],
)
```


当安装您的项目（例如，使用 [pip](https://pypi.org/project/pip)）时，所有尚未安装的依赖项都将被定位（通过 PyPI），下载，构建（如果需要），安装，并且 2) 项目中的任何脚本都将安装有用于在运行时验证指定依赖项的可用性的包装器。

#### 特定于平台的依赖项

Setuptools 提供了在盲目安装 install_requires 中列出的所有内容之前评估特定条件的能力。这对于特定于平台的依赖项非常有用。例如，enum 包是在 Python 3.4 中添加的，因此，依赖于它的包可以选择仅在 Python 版本早于 3.4 时安装它。要实现这一点

pyproject.toml
```toml
[project]
# ...
dependencies = [
    "enum34; python_version<'3.4'",
]
# ...
```

setup.cfg
```
[options]
#...
install_requires =
    enum34;python_version<'3.4'
```

setup.py
```python
setup(
    ...,
    install_requires=[
        "enum34;python_version<'3.4'",
    ],
)
```


同样地，如果您还希望声明 pywin32 的最小版本为 1.0，并且仅在用户使用 Windows 操作系统时安装它：

pyproject.toml
```toml
[project]
# ...
dependencies = [
    "enum34; python_version<'3.4'",
    "pywin32 >= 1.0; platform_system=='Windows'",
]
# ...
```
 
setup.cfg
```
[options]
#...
install_requires =
    enum34;python_version<'3.4'
    pywin32 >= 1.0;platform_system=='Windows'
```

setup.py
```python
setup(
    ...,
    install_requires=[
        "enum34;python_version<'3.4'",
        "pywin32 >= 1.0;platform_system=='Windows'",
    ],
)
```

可以在 [PEP 508](https://peps.python.org/pep-0508/) 中找到可用于测试平台类型的环境标记。

**参考**：作为替代方法，可以针对环境标记不足的特定用例使用[后端包装器](https://setuptools.pypa.io/en/latest/build_meta.html#backend-wrapper)。

#### 直接 URL 依赖项

注意：PyPI 和其他符合标准的软件包索引不接受使用直接 URL 声明依赖项的软件包。然而，当从本地文件系统或另一个 URL 安装软件包时，pip 将接受它们。

对于那些在软件包索引中不可用但可以以源代码仓库或存档的形式从其他地方下载的依赖项，可以使用 [PEP 440](https://peps.python.org/pep-0440/#direct-references) 的直接引用的变体来指定：

pyproject.toml
```toml
[project]
# ...
dependencies = [
    "Package-A @ git+https://example.net/package-a.git@main",
    "Package-B @ https://example.net/archives/package-b.whl",
]
```

setup.cfg
```
[options]
#...
install_requires =
    Package-A @ git+https://example.net/package-a.git@main
    Package-B @ https://example.net/archives/package-b.whl
```


setup.py
```python
setup(
    install_requires=[
       "Package-A @ git+https://example.net/package-a.git@main",
       "Package-B @ https://example.net/archives/package-b.whl",
    ],
    ...,
)
```
 
对于源代码仓库的 URL，可以在 pip 的 [VCS 支持文档](https://pip.pypa.io/en/latest/topics/vcs-support/)中找到支持的协议和 VCS 特定功能列表，例如选择特定的分支或标签。支持的存档 URL 格式为 sdists 和 wheels。

### 可选依赖项

Setuptools 允许您声明默认情况下不安装的依赖项。这实际上意味着您可以创建一个带有一组额外功能的“变体”包。

例如，让我们考虑一个提供可选 PDF 支持并需要两个其他依赖项才能运行的 Package-A：

pyproject.toml
```toml
[project]
name = "Package-A"
# ...
[project.optional-dependencies]
PDF = ["ReportLab>=1.2", "RXP"]
```

setup.cfg
```
[metadata]
name = Package-A

[options.extras_require]
PDF =
    ReportLab>=1.2
    RXP
```
setup.py
```python
setup(
    name="Package-A",
    ...,
    extras_require={
        "PDF": ["ReportLab>=1.2", "RXP"],
    },
)
```

提示：为运行测试或构建文档等辅助任务声明可选要求也很方便。

PDF 是这种依赖项列表的任意标识符，其他组件可以引用这些标识符并安装它们。

这种方法的一个用例是其他软件包可以使用这个“额外”来定义自己的依赖项。例如，如果 Package-B 需要已安装了 PDF 支持的 Package-A，它可以这样声明依赖项：

pyproject.toml
```toml
[project]
name = "Package-B"
# ...
dependencies = [
    "Package-A[PDF]"
]
```

setup.cfg
```
[metadata]
name = Package-B
#...

[options]
#...
install_requires =
    Package-A[PDF]
```


setup.py
```python
setup(
    name="Package-B",
    install_requires=["Package-A[PDF]"],
    ...,
)
```

这将导致 ReportLab 与 Package-A 一起安装，如果安装了 Package-B - 即使 Package-A 已经安装。通过这种方式，一个项目可以将可选的“下游依赖项”组合在一个特征名称下，以便依赖于它的软件包无需知道下游依赖项是什么。如果 Package-A 的后续版本构建了 PDF 支持并且不再需要 ReportLab，或者它最终需要除 ReportLab 之外的其他依赖项来提供 PDF 支持，Package-B 的设置信息不需要更改，但如果需要，正确的包仍将被安装。

**提示**

最佳实践：如果一个项目最终不再需要其他软件包来支持某个特性，则应该保留该特性的空要求列表，以防止依赖于该特性的软件包出现问题（由于无效的特性名称导致）。

警告：历史上，setuptools 还支持在控制台脚本中添加额外的依赖项，例如：

setup.cfg
```
[metadata]
name = Package-A
#...

[options]
#...
entry_points=
    [console_scripts]
    rst2pdf = project_a.tools.pdfgen [PDF]
    rst2html = project_a.tools.htmlgen
```
setup.py
```python
setup(
    name="Package-A",
    ...,
    entry_points={
        "console_scripts": [
            "rst2pdf = project_a.tools.pdfgen [PDF]",
            "rst2html = project_a.tools.htmlgen",
        ],
    },
)
```
 
这种语法表明入口点（在本例中是一个控制台脚本）只有在安装了 PDF 额外项时才有效。安装程序决定如何处理没有指定 PDF 的情况（例如，省略控制台脚本，在尝试加载入口点时提供警告，假定额外项已经存在并在后面让实现失败）。

然而，pip 和其他工具可能不支持此类额外依赖项的用例，因此这种做法被视为已弃用。请参阅[入口点规范](https://packaging.python.org/en/latest/specifications/entry-points/)。

### Python 要求

在某些情况下，您可能需要指定所需的最低 Python 版本。这可以配置如下所示：

pyproject.toml
```toml
[project]
name = "Package-B"
requires-python = ">=3.6"
# ...
```
setup.cfg
```
[metadata]
name = Package-B
#...

[options]
#...
python_requires = >=3.6
```


setup.py
```python
setup(
    name="Package-B",
    python_requires=">=3.6",
    ...,
)
```


## 开发模式（也称为“可编辑安装”）

在创建 Python 项目时，开发人员通常希望在发布和准备分发归档之前，可以迭代地实现和测试更改。

在正常情况下，这可能会非常繁琐，并需要开发人员操作 PYTHONPATH 环境变量或不断重新构建和重新安装项目。

为了促进迭代式探索和实验，setuptools 允许用户指示 Python 解释器及其导入机制直接从项目文件夹加载开发中的代码，而无需将文件复制到磁盘上的其他位置。这意味着 Python 源代码的更改可以立即生效，无需进行新的安装。

您可以通过在虚拟环境中执行可编辑安装，使用 pip 的 -e/\-\-editable 标志来进入此“开发模式”，如下所示：

```
$ cd your-python-project
$ python -m venv .venv
# Activate your environment with:
#      `source .venv/bin/activate` on Unix/macOS
# or   `.venv\Scripts\activate` on Windows

$ pip install --editable .

# Now you have access to your package
# as if it was installed in .venv
$ python -c "import your_python_project"
```

“可编辑安装”与使用 pip install . 进行常规安装非常相似，只是它仅安装您的包依赖项、元数据以及控制台和 GUI 脚本的包装器。在幕后，setuptools 将尝试在目标目录（通常是 site-packages）中创建一个特殊的 [.pth 文件](https://docs.python.org/3.11/library/site.html#module-site)，该文件扩展了 PYTHONPATH 或安装了自定义[导入钩子](https://docs.python.org/3.11/reference/import.html)。

完成给定的开发任务后，您可以像通常使用 pip uninstall \<包名\> 一样简单地卸载您的包。

请注意，默认情况下，可编辑安装将至少暴露所有通常可用的文件。但是，根据项目中的文件和目录组织，它还可能作为副作用暴露一些通常不可用的文件。这是允许您逐步创建新的 Python 模块的。如果您正在寻找不同的行为，请查看以下部分。

**虚拟环境**

您可以将虚拟环境视为“隔离的 Python 运行时部署”，允许用户在不影响系统全局行为的情况下安装不同的库和工具集。

它们是测试新项目的安全方式，并且可以使用标准库中的 venv 模块轻松创建。

但是请注意，根据您的操作系统或发行版，venv 可能不会默认随 Python 安装。对于这些情况，您可能需要使用操作系统包管理器进行安装。例如，在 Debian/Ubuntu 系统中，您可以通过以下方式获得：

```shell
sudo apt install python3-venv
```

或者，您还可以尝试安装 virtualenv。关于使用 pip 和 venv 在虚拟环境中安装包的详细信息，请参阅 [Install packages in a virtual environment using pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)。

**注意**

自版本 v64.0.0 起发生更改：根据 [PEP 660](https://peps.python.org/pep-0660/) 实现了可编辑安装钩子。[命名空间包](https://peps.python.org/pep-0420/)的支持仍处于实验阶段。


### “严格”可编辑安装

在考虑可编辑安装时，用户可能有以下期望：

* 它应该允许开发人员添加新文件（或拆分/重命名现有文件）并自动暴露它们。
* 它应该尽可能接近常规安装，并帮助用户检测问题（例如，新文件未包含在分发中）。

不幸的是，这些期望相互冲突。为解决这个问题，setuptools 允许开发人员选择更“严格”的可编辑安装模式。这可以通过通过 pip 传递特殊配置设置来实现，如下所示：

```shell
pip install -e . --config-settings editable_mode=strict
```

在这种模式下，新文件将不会被暴露，可编辑安装将尽可能模拟常规安装的行为。在幕后，setuptools 将在辅助目录（$your_project_dir/build）中创建文件链接树，并通过 .pth 文件将其添加到 PYTHONPATH 中。请小心不要错误地删除此存储库，否则您的文件可能无法访问。

**警告**

严格的可编辑安装需要将辅助文件放置在 build/\_\_editable\_\_.* 目录中（相对于项目根目录）。

请在测试项目时小心不要删除此目录，否则您的可编辑安装可能会受到损害。

您可以在卸载后删除 build/\_\_editable\_\_.* 目录。

**注意**

自版本 v64.0.0 起新增：为可编辑安装添加了新的严格模式。此模式的具体实现细节可能会有所不同。

### 限制

* "editable"一词仅用于指代包目录中的Python模块。非Python文件、外部（数据）文件、可执行脚本文件、二进制扩展、头文件和元数据可能会以安装时的版本快照形式暴露出来。

* 添加新的依赖项、入口点或更改项目的元数据需要进行全新的“editable”重新安装。

* 控制台脚本和GUI脚本必须通过入口点指定才能正常工作。

* 严格的可编辑安装需要文件系统支持符号链接或硬链接。这种安装模式可能还会在项目目录下生成辅助文件。

* 无法保证可编辑安装将使用特定的技术。根据每个项目的情况，setuptools可能会选择不同的方法来确保包在运行时可导入。

* 无法保证在可编辑安装后顶级包目录之外的文件将可访问。

* 无法保证属性如__path__或__file__将与原始文件的确切位置对应（例如，setuptools可能会使用文件链接来执行可编辑安装）。鼓励用户在尝试直接访问包文件时使用importlib.resources或importlib.metadata等工具。

* 可编辑安装可能无法与使用pkgutil或pkg_resources创建的命名空间兼容。请使用PEP 420风格的隐式命名空间。

* 对于使用平坦布局结构化的项目，对PEP 420风格的隐式命名空间包的支持仍然处于实验阶段。如果遇到问题，您可以尝试将包结构转换为src-layout。

* 当前工作目录中名称巧合与已安装包名称匹配的文件系统条目可能在Python的导入系统中优先使用。鼓励用户避免这种情况。

* Setuptools将尝试为可编辑安装的模块提供正确的优先级。然而，这并不总是一项容易的任务。如果您对sys.path中的特定顺序或特定的导入优先级有要求，请注意，setuptools支持的可编辑安装可能无法满足此要求，因此它可能不适合您的用例。

**注意**

在测试环境中，可编辑安装并不是对常规安装的完美替代品。如果有疑问，请通过常规wheel安装测试您的项目。Python生态系统中有一些工具，如tox或nox，在适当的配置下可以帮助您进行测试。

### 旧版行为

如果您的项目与新的“可编辑安装”不兼容，或者您希望复制旧版行为，则可以在兼容模式下执行安装:

```shell
pip install -e . --config-settings editable_mode=compat
```
这种安装模式将尝试模拟python setup.py develop的工作方式（仍然在[PEP 660](https://peps.python.org/pep-0660/)的上下文中）。

**警告**

兼容模式是过渡性的，将在未来版本的setuptools中删除，它仅在迁移期间提供帮助。还请注意，对此模式的支持有限：可以安全地假设兼容模式是“原样提供的”，并且不太可能实施改进。鼓励用户尝试新的可编辑安装技术并进行必要的适应。


**注意**

较新版本的pip不再在存在pyproject.toml文件时运行回退命令python setup.py develop。这意味着在使用pip安装软件包时设置环境变量SETUPTOOLS_ENABLE_FEATURES="legacy-editable"将不会产生任何效果。

### 可编辑安装的工作原理


有许多技术可以用来以类似已安装的方式暴露正在开发中的软件包。根据项目文件结构和所选模式，setuptools将选择其中一种方法进行可编辑安装 [3]。

下面是一个不完全的实施机制列表。有关更多信息，请参阅[PEP 660](https://peps.python.org/pep-0660/#what-to-put-in-the-wheel)的文本。

* 可以将静态 .pth 文件  添加到site.getsitepackages()或site.getusersitepackages()列出的目录之一，以扩展sys.path。

* 可以使用一个包含一系列文件链接的目录来模拟项目结构并指向原始文件。然后可以使用静态 .pth 文件将此目录添加到sys.path中。

* 还可以使用动态 .pth 文件 来安装一个“导入查找器”（MetaPathFinder或PathEntryFinder），它将钩入Python的导入系统机制。

**注意**

setuptools不保证使用哪种技术来执行可编辑安装。这将因项目而异，并可能根据使用的setuptools具体版本而变化。


## Entry Points

Entry Points（入口点）是一种可以在安装时由包暴露的元数据类型。它们是 Python 生态系统中非常有用的功能，在两种情况下特别方便：

* 包想要提供在终端运行的命令。这种功能被称为控制台脚本（console scripts）。该命令也可以打开 GUI，这种情况下被称为 GUI 脚本（GUI scripts）。一个控制台脚本的例子是 pip 包提供的脚本，它允许您在终端中运行像 pip install 这样的命令。

* 包想要通过插件来启用其功能的定制化。例如，测试框架 pytest 允许通过 pytest11 入口点进行定制，语法高亮工具 pygments 允许使用 pygments.styles 入口点来指定额外的样式。

### 控制台脚本

让我们从控制台脚本开始。首先考虑一个没有入口点的示例。想象一个定义如下的包：

```
project_root_directory
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins
        ├── __init__.py
        └── ...
```

其中 init.py 是这样的：

```python
def hello_world():
    print("Hello world")
```

现在，假设我们想要提供一种从命令行执行 hello_world() 函数的方法。一种方法是创建一个文件 src/timmins/\_\_main\_\_.py，提供以下钩子：

```python
from . import hello_world

if name == 'main':
    hello_world()
```

然后，在安装 timmins 包之后，我们可以通过 runpy 模块调用 hello_world() 函数，如下所示：

```shell
$ python -m timmins
Hello world
```

除了使用 main.py 这种方法，您还可以创建一个用户友好的 CLI 可执行文件，可以直接调用而无需使用 python -m。在上面的示例中，要创建一个命令 hello-world 来调用 timmins.hello_world，可以在配置中添加一个控制台脚本入口点：

```toml
[project.scripts]
hello-world = "timmins:hello_world"
```

setup.cfg
```
[options.entry_points]
console_scripts =
    hello-world = timmins:hello_world
```

setup.py
```python
from setuptools import setup

setup(
    # ...,
    entry_points={
        'console_scripts': [
            'hello-world = timmins:hello_world',
        ]
    }
)
```

安装完包之后，用户可以通过在命令行中简单地调用 hello-world 来调用该函数：

```shell
$ hello-world
Hello world
```



请注意，任何作为控制台脚本使用的函数，比如这个例子中的 hello_world()，都不应接受任何参数。如果您的函数需要来自用户的任何输入，可以在函数体内使用 argparse 等常规命令行参数解析工具来解析通过 sys.argv 给定的用户输入。

您可能已经注意到，我们使用了一种特殊的语法来指定由控制台脚本调用的函数，即我们写了 timmins:hello_world，使用冒号 : 分隔包名和函数名。有关此语法的完整规范将在本文档的最后一节中讨论，并且这可以用来指定包中任何位置的函数，而不仅仅是在 init.py 中。

### GUI 脚本

除了控制台脚本，Setuptools 还支持 gui_scripts，这将在不在终端窗口中运行的情况下启动 GUI 应用程序。

例如，如果我们有一个与之前相同的目录结构的项目，其中包含一个 \_\_init\_\_.py 文件，其中包含以下内容：

```
python
import PySimpleGUI as sg

def hello_world():
    sg.Window(title="Hello world", layout=[[]], margins=(100, 50)).read()
```

然后，我们可以添加一个 GUI 脚本入口点：

pyproject.toml
```toml
[project.gui-scripts]
hello-world = "timmins:hello_world"
```

setup.cfg
```
[options.entry_points]
gui_scripts =
    hello-world = timmins:hello_world
```

setup.py
```python
from setuptools import setup

setup(
    # ...,
    entry_points={
        'gui_scripts': [
            'hello-world = timmins:hello_world',
        ]
    }
)
```


现在运行：

```shell
$ hello-world
```

将打开一个带有标题“Hello world”的小应用程序窗口。

注意，与控制台脚本一样，任何用作GUI脚本的函数都不应接受任何参数，并且任何用户输入都可以在函数体内解析。GUI脚本也使用相同的语法（在文档的最后一节讨论过）来指定要调用的函数。

注意：控制台脚本与GUI脚本之间的区别只影响Windows系统。[1] 控制台脚本包装在控制台可执行文件中，因此它们附加到控制台，并可以使用sys.stdin、sys.stdout和sys.stderr进行输入和输出。GUI脚本包装在GUI可执行文件中，因此它们可以在没有控制台的情况下启动，但除非应用程序代码重定向它们，否则不能使用标准流。其他平台没有这种区别。

注意：控制台和GUI脚本之所以起作用，是因为在幕后，诸如pip之类的安装程序会在调用的函数周围创建包装脚本。例如，在上面两个示例中，hello-world入口点将创建一个命令hello-world，启动以下脚本： 

```python
import sys
from timmins import hello_world
sys.exit(hello_world())
```

### 广告行为

控制台/GUI脚本是更一般的入口点概念的一种用法。入口点更普遍地允许打包者为其他库和应用程序发现的行为做广告。此功能启用了“插件”-类似的功能，其中一个库征集入口点，任意数量的其他库提供这些入口点。

可以在[pytest插件](https://docs.pytest.org/en/latest/writing_plugins.html)中看到此插件行为的良好示例，其中pytest是一个测试框架，允许其他库通过pytest11入口点扩展或修改其功能。

控制台/GUI脚本的工作方式类似，库会宣传其命令和像pip这样的工具会创建调用这些命令的包装脚本。


### Entry Points for Plugins

让我们考虑一个简单的示例来理解如何实现与插件对应的入口点。假设我们有一个名为 timmins 的包，其目录结构如下：

```
timmins
├── pyproject.toml        # 或者 setup.cfg、setup.py
└── src
    └── timmins
        └── __init__.py
```

在 src/timmins/\_\_init\_\_.py 中，我们有以下代码：

```python

def hello_world():
    print('Hello world')
```

基本上，我们定义了一个 hello_world() 函数，它将打印文本 'Hello world'。现在，假设我们想以不同的方式打印文本 'Hello world'。当前函数只是按原样打印文本 - 假设我们想要另一种样式，其中文本被感叹号包围：

```
!!! Hello world !!!
```

让我们看看如何使用插件来实现这一点。首先，让我们将打印文本的样式与文本本身分开。换句话说，我们可以更改 src/timmins/\_\_init\_\_.py 中的代码如下：

```python

def display(text):
    print(text)

def hello_world():
    display('Hello world')
```

这里，display() 函数控制打印文本的样式，而 hello_world() 函数调用 display() 函数来打印文本 'Hello world'。

现在，display() 函数只是按原样打印文本。为了能够自定义它，我们可以这样做。让我们引入一个名为 timmins.display 的新入口点组，并期望实现此入口点的插件包提供类似 display() 的函数。接下来，为了能够自动发现实现此入口点的插件包，我们可以使用 importlib.metadata 模块，如下所示：

```python
from importlib.metadata import entry_points
display_eps = entry_points(group='timmins.display')
```

**注意**

每个 importlib.metadata.EntryPoint 对象都是一个包含名称、组和值的对象。例如，在描述的插件包设置完成后，上述代码中的 display_eps 将如下所示：

```
(
    EntryPoint(name='excl', value='timmins_plugin_fancy:excl_display', group='timmins.display'),
    ...,
)
```

display_eps 现在是一个 EntryPoint 对象的列表，每个对象引用由一个或多个安装的插件包定义的类似 display() 的函数。然后，要导入特定的类似 display() 的函数 - 让我们选择与第一个发现的入口点对应的函数 - 我们可以使用 load() 方法，如下所示：

```python
display = display_eps[0].load()
```

最后，一个明智的行为是，如果我们找不到任何自定义 display() 函数的插件包，我们应该退回到我们的默认实现，即按原样打印文本。包含这种行为后，src/timmins/\_\_init\_\_.py 中的代码最终变为：

```python
from importlib.metadata import entry_points
display_eps = entry_points(group='timmins.display')
try:
    display = display_eps[0].load()
except IndexError:
    def display(text):
        print(text)

def hello_world():
    display('Hello world')
```


这完成了 timmins 方面的设置。接下来，我们需要实现一个插件，该插件实现了 timmins.display 入口点。让我们将这个插件命名为 timmins-plugin-fancy，并设置如下的目录结构：

```
timmins-plugin-fancy
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins_plugin_fancy
        └── __init__.py
```

然后，在 src/timmins_plugin_fancy/\_\_init\_\_.py 中，我们可以放置一个名为 excl_display() 的函数，该函数打印给定的文本并用感叹号括起来：

```python
def excl_display(text):
    print('!!!', text, '!!!')
```

这是我们希望提供给 timmins 包的类似 display() 的函数。我们可以通过在 timmins-plugin-fancy 的配置中添加以下内容来实现：

pyproject.toml
```toml
# Note the quotes around timmins.display in order to escape the dot .
[project.entry-points."timmins.display"]
excl = "timmins_plugin_fancy:excl_display"
```

setup.cfg
```
[options.entry_points]
timmins.display =
    excl = timmins_plugin_fancy:excl_display
```

setup.py
```python
from setuptools import setup

setup(
    # ...,
    entry_points = {
        'timmins.display': [
            'excl = timmins_plugin_fancy:excl_display'
        ]
    }
)
```


基本上，此配置表示我们正在提供一个位于 timmins.display 组下的入口点。该入口点命名为 excl，并且它引用了由 timmins-plugin-fancy 包定义的 excl_display 函数。

现在，如果我们安装了 timmins 和 timmins-plugin-fancy，我们应该会得到以下结果：

```
from timmins import hello_world
hello_world()
!!! Hello world !!!
```

 
而如果我们只安装了 timmins 而没有安装 timmins-plugin-fancy，则输出应该是：

```
from timmins import hello_world
hello_world()
Hello world
```

因此，我们的插件起作用了。

我们的插件也可以在 timmins.display 组下定义多个入口点。例如，在 src/timmins_plugin_fancy/__init__.py 中，我们可以有两个类似 display() 的函数，如下所示：

```python
def excl_display(text):
    print('!!!', text, '!!!')

def lined_display(text):
    print(''.join(['-' for _ in text]))
    print(text)
    print(''.join(['-' for _ in text]))
```

timmins-plugin-fancy 的配置将会改变为：

pyproject.toml
```toml
[project.entry-points."timmins.display"]
excl = "timmins_plugin_fancy:excl_display"
lined = "timmins_plugin_fancy:lined_display"
```

setup.cfg
```
[options.entry_points]
timmins.display =
    excl = timmins_plugin_fancy:excl_display
    lined = timmins_plugin_fancy:lined_display
```

setup.py
```python
from setuptools import setup

setup(
    # ...,
    entry_points = {
        'timmins.display': [
            'excl = timmins_plugin_fancy:excl_display',
            'lined = timmins_plugin_fancy:lined_display',
        ]
    }
)
```



在 timmins 方面，我们还可以使用不同的策略来加载入口点。

```python
display_eps = entry_points(group='timmins.display')
try:
    display = display_eps['lined'].load()
except KeyError:
    # if the 'lined' display is not available, use something else
    ...
```

或者我们也可以加载给定组下的所有插件。尽管在我们当前的示例中可能没有太多用处，但在许多情况下这是有用的：

```python
display_eps = entry_points(group='timmins.display')
for ep in display_eps:
    display = ep.load()
    # do something with display
    ...
```

另一个要点是，在这个特定示例中，我们使用插件来自定义函数 display() 的行为。一般来说，我们可以使用入口点来使插件不仅可以自定义函数的行为，还可以自定义整个类和模块的行为。这与控制台/GUI脚本的情况不同，其中入口点只能引用函数。用于指定入口点的语法与控制台/GUI脚本的语法相同，并在最后一节中进行了讨论。

**提示**

加载和导入入口点的推荐方法是使用 importlib.metadata 模块，它是自 Python 3.8 起的标准库的一部分，并且自 Python 3.10 起是非试验性的。对于旧版本的 Python，应该使用其后向兼容的 importlib_metadata。在使用后向兼容版本时，唯一需要更改的是将 importlib.metadata 替换为 importlib_metadata，即：

```python
from importlib_metadata import entry_points
...
```

总而言之，入口点允许一个包为通过插件进行自定义的功能打开其功能。向插件包请求入口点的包不需要任何依赖项或先验知识，下游用户可以通过组合实现入口点的插件来组合功能。


### Entry Points 语法
入口点的语法如下所示：

```
<name> = <package_or_module>[:<object>[.<attr>[.<nested-attr>]*]]
<名称> = <包或模块>[:<对象>[.<属性>[.<嵌套属性>]*]]
```

这里，方括号 \[\] 表示可选项，星号 \* 表示重复。名称是您想要创建的脚本/入口点的名称，冒号左侧是包含您想要调用的对象的包或模块（可以将其视为您在导入语句中编写的内容），右侧是您想要调用的对象（例如，一个函数）。

为了使这个语法更清晰，考虑以下例子：

**包或模块**
如果您提供：

```
<name> = <package_or_module>

```

作为入口点，其中 <包或模块> 可以在子模块或子包的情况下包含 .，那么 Python 生态系统中的工具将粗略地解释这个值为：

```
import <package_or_module>
parsed_value = <package_or_module>
 
```

**模块级对象**
如果您提供：

```
<name> = <package_or_module>:<object> 
```

其中 <对象> 不包含任何 .，这将被粗略解释为：

```python
from <package_or_module> import <object>
parsed_value = <object>
```

**嵌套对象**
如果您提供：

```
<name> = <package_or_module>:<object>.<attr>.<nested_attr>
```

这将被粗略解释为：

```python
from <package_or_module> import <object>
parsed_value = <object>.<attr>.<nested_attr>
```

在控制台/GUI 脚本的情况下，这种语法可以用来指定一个函数，而在用于插件的入口点的一般情况下，它可以用来指定一个函数、类或模块。




## 数据文件支持

Python 生态系统中的旧包装安装方法传统上允许安装“数据文件”，这些文件被放置在特定于平台的位置。然而，分发到包中的数据文件最常见的用例是由包使用，通常是通过将数据文件包含在包目录中来实现。

Setuptools 关注于这种最常见类型的数据文件，并提供了三种指定哪些文件应该包含在您的包中的方法，如下一节所述。

### 配置选项

#### include_package_data

首先，您可以使用 include_package_data 关键字。例如，如果包树看起来像这样：

```
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
└── src
    └── mypkg
        ├── __init__.py
        ├── data1.rst
        ├── data2.rst
        ├── data1.txt
        └── data2.txt
```


并且您提供以下配置：


pyproject.toml
```toml
[tool.setuptools]
# ...
# By default, include-package-data is true in pyproject.toml, so you do
# NOT have to specify this line.
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
```


setup.cfg
```
[options]
# ...
packages = find:
package_dir =
    = src
include_package_data = True

[options.packages.find]
where = src
```


setup.py
```python
from setuptools import setup, find_packages
setup(
    # ...,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True
)
```

那么所有的 .txt 和 .rst 文件将自动与您的包一起安装，前提是：

* 这些文件通过 MANIFEST.in 文件包含，例如：

```
include src/mypkg/.txt
include src/mypkg/.rst
```

* 或者，它们正在被 Git、Mercurial 或 SVN 这样的版本控制系统跟踪，并且您已经配置了适当的插件，例如 setuptools-scm 或 setuptools-svn。（有关如何编写这样的插件的信息，请参阅下面关于为版本控制系统添加支持的部分。）

**注意**

版本 v61.0.0 中的新功能：当通过 pyproject.toml 配置项目时，tool.setuptools.include-package-data 的默认值为 True。这种行为与 setup.cfg 和 setup.py 不同（其中 include_package_data=False 默认情况下），这样做是为了确保与现有项目的向后兼容性而没有更改。


### package_data

默认情况下，include_package_data 将考虑包目录（在本例中为 src/mypkg）中找到的所有非 .py 文件作为数据文件，并将那些满足上述两个条件之一的文件包含在源分发中，因此也包含在您的包的安装中。如果您想要更细粒度地控制哪些文件被包含，那么您还可以使用 package_data 关键字。例如，如果包树看起来像这样：

```
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
└── src
    └── mypkg
        ├── __init__.py
        ├── data1.rst
        ├── data2.rst
        ├── data1.txt
        └── data2.txt
```


那么您可以使用以下配置来捕获 .txt 和 .rst 文件作为数据文件：

pyproject.toml
```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
mypkg = ["*.txt", "*.rst"]
```

setup.cfg
```
[options]
# ...
packages = find:
package_dir =
    = src

[options.packages.find]
where = src

[options.package_data]
mypkg =
    *.txt
    *.rst
```


setup.py
```python
from setuptools import setup, find_packages
setup(
    # ...,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"mypkg": ["*.txt", "*.rst"]}
)
```

package_data 参数是一个字典，将包名映射到 glob 模式列表。请注意，使用 package_data 选项指定的数据文件既不需要包含在 MANIFEST.in 文件中，也不需要通过版本控制系统插件添加。

**注意**

如果您的 glob 模式使用路径，则必须使用正斜杠 (/) 作为路径分隔符，即使您在 Windows 上也是如此。Setuptools 在构建时会自动将斜杠转换为适当的特定于平台的分隔符。

**重要**

Glob 模式不会自动匹配以点 (.) 开头的目录或文件名，即点文件。要包含这样的文件，您必须显式地从一个点开始模式，例如 .* 匹配 .gitignore。

如果您有多个顶级包并且所有这些包都有相同的数据文件模式，例如：

```
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
└── src
    ├── mypkg1
    │   ├── data1.rst
    │   ├── data1.txt
    │   └── __init__.py
    └── mypkg2
        ├── data2.txt
        └── __init__.py
```


在这里，mypkg1 和 mypkg2 都共享一个 .txt 数据文件的常见模式。然而，只有 mypkg1 有 .rst 数据文件。在这种情况下，如果您想要使用 package_data 选项，以下配置将起作用：

pyproject.toml
```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"*" = ["*.txt"]
mypkg1 = ["data1.rst"]
```


setup.cfg
```
[options]
packages = find:
package_dir =
    = src

[options.packages.find]
where = src

[options.package_data]
* =
  *.txt
mypkg1 =
  data1.rst
```


setup.py
```python
from setuptools import setup, find_packages
setup(
    # ...,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": ["*.txt"], "mypkg1": ["data1.rst"]},
)
```

请注意，如果您在 setup.py 的 package_data 下空字符串 "" 中列出模式，并在 setup.cfg 和 pyproject.toml 中使用星号 *，这些模式将用于找到每个包中的文件。例如，我们使用 "" 或 * 表示应该捕获所有包的 .txt 文件作为数据文件。这些占位符被视为一种特殊情况，setuptools 不支持在此配置中对包名称使用 glob 模式（模式仅支持文件路径）。还要注意如何继续为单个包指定模式，即我们指定 mypkg1 中的 data1.rst 也应该被捕获。

**注意**

在构建 sdist 时，数据文件也从 package_name.egg-info/SOURCES.txt 文件中获取，该文件作为一种缓存。因此，在重新构建包之前，请确保删除此文件，如果 package_data 已更新。

**注意**

在 Python 中，任何目录都被视为一个包（即使它不包含 init.py，参见 Packaging namespace packages 上的原生命名空间包）。因此，如果您不依赖自动发现，应确保所有包（包括不包含任何 Python 文件的包）都包含在 packages 配置中（有关更多信息，请参阅包发现和命名空间包）。此外，建议使用点符号表示完整的包名称，而不是嵌套路径，以避免容易出错的配置。请查看下面的子目录部分。

### exclude_package_data

有时，单独使用 include_package_data 或 package_data 选项并不能精确定义您想要包含的文件。例如，考虑一个场景，您设置了 include_package_data=True，并且正在使用具有适当插件的版本控制系统。有时开发者会添加特定于目录的标记文件（例如 .gitignore、.gitkeep、.gitattributes 或 .hgignore），这些文件可能正在被版本控制系统跟踪，因此默认情况下它们将在安装包时被包含进去。

假设您想要防止这些文件被包含在安装中（它们与 Python 或包无关），那么您可以使用 exclude_package_data 选项：

pyproject.toml
```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.exclude-package-data]
mypkg = [".gitattributes"]
```


setup.cfg
```
[options]
# ...
packages = find:
package_dir =
    = src
include_package_data = True

[options.packages.find]
where = src

[options.exclude_package_data]
mypkg =
    .gitattributes
```


setup.py
```python
from setuptools import setup, find_packages
setup(
    # ...,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    exclude_package_data={"mypkg": [".gitattributes"]},
)
```

exclude_package_data 选项是一个将包名映射到通配符模式列表的字典，就像 package_data 选项一样。而且，与该选项一样，您可以在 setup.py 中使用空字符串键 ""，在 setup.cfg 和 pyproject.toml 中使用星号 * 来匹配所有顶级包。

任何与这些模式匹配的文件都将在安装时被排除，即使它们在 package_data 中列出或由于使用 include_package_data 而被包含进去。

### 总结
总而言之，这三个选项允许您：

**include_package_data**

接受 MANIFEST.in 匹配或由插件添加的所有数据文件和目录。

**package_data**

指定额外的模式以匹配可能不被 MANIFEST.in 匹配或由插件添加的文件。

**exclude_package_data**

指定数据文件和目录的模式，在包安装时不应包含在内，即使它们由于使用上述选项而被包含进去。

**注意**

由于构建过程的工作方式，您在项目中包含然后停止包含的数据文件可能会“孤立”在项目的构建目录中，需要您手动删除它们。如果您的用户和贡献者使用 Subversion 跟踪您项目的中间版本，则这也可能很重要；确保在删除包含文件的更改时通知他们，以便他们也可以手动删除这些文件。

另请参阅[Caching and Troubleshooting](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#caching-and-troubleshooting) 中的故障排除信息。


### 子目录用于数据文件

一个常见的模式是将一些（或全部）数据文件放置在单独的子目录下。例如：

```
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
└── src
    └── mypkg
        ├── data
        │   ├── data1.rst
        │   └── data2.rst
        ├── __init__.py
        ├── data1.txt
        └── data2.txt
```

在这里，.rst 文件放置在 mypkg 内的 data 子目录下，而 .txt 文件直接放置在 mypkg 下。

在这种情况下，建议的方法是将数据视为命名空间包（参见 [PEP 420](https://peps.python.org/pep-0420/)）。这样，您可以依赖于上述介绍的相同方法，使用 package_data 或 include_package_data。为了完整起见，我们在下面包含了子目录结构的配置示例，但请参考本文档前面部分的详细信息。

使用 package_data，配置可能如下所示：

pyproject.toml
```toml
# Scanning for namespace packages in the ``src`` directory is true by
# default in pyproject.toml, so you do NOT need to include the
# `tool.setuptools.packages.find` if it looks like the following:
# [tool.setuptools.packages.find]
# namespaces = true
# where = ["src"]

[tool.setuptools.package-data]
mypkg = ["*.txt"]
"mypkg.data" = ["*.rst"]
```


setup.cfg
```
[options]
# ...
packages = find_namespace:
package_dir =
    = src

[options.packages.find]
where = src

[options.package_data]
mypkg =
    *.txt
mypkg.data =
    *.rst
```


setup.py
```python
from setuptools import setup, find_namespace_packages
setup(
    # ...,
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "mypkg": ["*.txt"],
        "mypkg.data": ["*.rst"],
    }
)
```

换句话说，我们允许 Setuptools 在 src 目录中扫描命名空间包，从而识别出数据目录，然后我们分别为根包 mypkg 和 mypkg 下的命名空间包数据指定数据文件。

另外，您也可以依赖于 include_package_data。请注意，这是在 pyproject.toml 中的默认行为，但您需要在 setup.cfg 或 setup.py 中手动启用对命名空间包的扫描：

pyproject.toml
```toml
[tool.setuptools]
# ...
# By default, include-package-data is true in pyproject.toml, so you do
# NOT have to specify this line.
include-package-data = true

[tool.setuptools.packages.find]
# scanning for namespace packages is true by default in pyproject.toml, so
# you need NOT include this configuration.
namespaces = true
where = ["src"]
```


setup.cfg
```
[options]
packages = find_namespace:
package_dir =
    = src
include_package_data = True

[options.packages.find]
where = src
```


setup.py
```python
from setuptools import setup, find_namespace_packages
setup(
    # ... ,
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
```

为了避免 include_package_data 的常见错误，请确保 MANIFEST.in 设置正确，或者使用版本控制系统插件（请参阅控制分发中的文件）。

### 在运行时访问数据文件

通常，现有程序会操作包的 file 属性以找到数据文件的位置。例如，如果您的结构如下所示：



```
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
└── src
    └── mypkg
        ├── data
        │   └── data1.txt
        ├── __init__.py
        └── foo.py
```

那么，在 mypkg/foo.py 中，您可能尝试这样做来访问 mypkg/data/data1.txt：

```python
import os
data_path = os.path.join(os.path.dirname(__file__), 'data', 'data1.txt')
with open(data_path, 'r') as data_file:
    ...
```



然而，这种操作与基于 PEP 302 的导入钩子不兼容，包括从 zip 文件和 Python Eggs 中导入。强烈建议如果您使用数据文件，应该使用 importlib.resources 来访问它们。在这种情况下，您可以这样做：

```python
from importlib.resources import files
data_text = files('mypkg.data').joinpath('data1.txt').read_text()
```


importlib.resources 是在 Python 3.7 中添加的。然而，本代码中展示的 API（使用 files()）是在 Python 3.9 中才添加的，[2] 并且通过命名空间包访问数据文件的支持是在 Python 3.10 中才添加的 [3]（data 子目录是根包 mypkg 下的一个命名空间包）。因此，您可能只会发现这段代码在 Python 3.10（及更高版本）中工作。对于其他版本的 Python，建议您使用 importlib-resources 回溯版本，该版本提供了这个库的最新版本。在这种情况下，唯一需要修改的是将 importlib.resources 替换为 importlib_resources，即

```python
from importlib_resources import files
...
```

详细的使用指南请参阅使用 [importlib_resources](https://importlib-resources.readthedocs.io/en/latest/using.html)。

**提示**

包目录中的文件应为只读，以避免一系列常见问题（例如当多个用户共享一个常见的 Python 安装时，当包从 zip 文件加载时，或者当多个 Python 应用程序实例并行运行时）。

如果您的 Python 包需要对共享数据或配置进行写入，您可以使用标准的平台/操作系统特定系统目录，例如 ~/.local/config/\$appname 或 /usr/share/\$appname/\$version（Linux 特定） 。一个常见的方法是将一个只读的模板文件添加到包目录中，然后如果没有找到预先存在的文件，将其复制到正确的系统目录。


### 数据文件来自插件和扩展

如果您希望插件和扩展到您的包中贡献包数据文件，您可以借助本地/隐式命名空间包（作为文件的容器）。这样，当使用 importlib.resources 时，在运行时将列出所有文件。请注意，尽管没有严格的保证，主流的 Python 包管理器，如 pip 和派生工具，会将属于共享同一命名空间的多个发行版的文件安装到文件系统中的同一目录中。这意味着 importlib.resources 的开销将是最小的。

### 非包数据文件
历史上，通过 easy_install 的 setuptools 会将分发中的数据文件封装到 egg 中（请参阅旧文档）。由于 eggs 已经被弃用，而基于 pip 的安装回退到特定于平台的位置来安装数据文件，因此没有可靠地检索这些资源的支持设施。

相反，PyPA 建议您希望在运行时访问的任何数据文件都应包含在包中。


## 构建扩展模块

Setuptools 可以构建 C/C++ 扩展模块。setup() 的关键字参数 ext_modules 应该是 setuptools.Extension 类的实例列表。

例如，让我们考虑一个只有一个扩展模块的简单项目：

```
<project_folder>
├── pyproject.toml
└── foo.c
```

并且所有项目元数据配置都在 pyproject.toml 文件中：

```toml
# pyproject.toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "mylib-foo"  # as it would appear on PyPI
version = "0.42"
```

为了指示 setuptools 将 foo.c 文件编译成扩展模块 mylib.foo，我们需要添加一个类似于以下的 setup.py 文件：

```python
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="mylib.foo",  # as it would be imported
                               # may include packages/namespaces separated by `.`

            sources=["foo.c"], # all sources are compiled into a single binary file
        ),
    ]
)
```

另请参阅：您可以在 Python 文档中找到有关 [C/C++ 扩展](https://docs.python.org/3/extending/extending.html)的更多信息。或者，您可能也对学习 [Cython](https://cython.readthedocs.io/en/stable/index.html) 感兴趣。

如果您计划在多个平台上分发使用扩展的包，[cibuildwheel](https://pypi.org/project/cibuildwheel) 也可能会有所帮助。

**重要**

在构建包时，所有用于编译扩展的文件都需要在系统上可用，因此请确保包含一些文档，说明对于希望从源代码构建您的包的开发人员如何获取操作系统级依赖项（例如编译器和外部二进制库/文件）。

您还需要确保包含在项目中的所有辅助文件（例如您或您的团队编写的 C 标头文件）被配置为包含在您的 sdist 中。请参阅我们关于控制分发中的文件的部分。

### 编译器和链接器选项

build_ext 命令用于构建 C/C++ 扩展模块。它通过从各种来源组合编译器和链接器选项来创建运行编译器和链接器的命令行：

* sysconfig 变量 CC、CXX、CCSHARED、LDSHARED 和 CFLAGS，

* 环境变量 CC、CPP、CXX、LDSHARED 和 CFLAGS、CPPFLAGS、LDFLAGS，

* Extension 属性 include_dirs、library_dirs、extra_compile_args、extra_link_args、runtime_library_dirs。

具体来说，如果环境变量 CC、CPP、CXX 和 LDSHARED 被设置，它们将被用于替代同名的 sysconfig 变量。

编译器选项按以下顺序出现在命令行中：

* 首先是由 sysconfig 变量 CFLAGS 提供的选项，

* 然后是由环境变量 CFLAGS 和 CPPFLAGS 提供的选项，

* 然后是由 sysconfig 变量 CCSHARED 提供的选项，

* 然后是每个 Extension.include_dirs 元素的 -I 选项，

* 最后是由 Extension.extra_compile_args 提供的选项。

链接器选项按以下顺序出现在命令行中：

* 首先是由环境变量和 sysconfig 变量提供的选项，

* 然后是每个 Extension.library_dirs 元素的 -L 选项，

* 然后是每个 Extension.runtime_library_dirs 元素的类似于 -Wl,-rpath 的链接器特定选项，

* 最后是由 Extension.extra_link_args 提供的选项。

生成的命令行然后由编译器和链接器处理。根据 GCC 手册关于目录选项和环境变量的部分，在 \#include \<file> 指令中，C/C++ 编译器按以下顺序搜索文件名：

* 首先是由 -I 选项给定的目录（按照从左到右的顺序），

* 然后是由环境变量 CPATH 给定的目录（按照从左到右的顺序），

* 然后是由 -isystem 选项给定的目录（按照从左到右的顺序），

* 然后是由环境变量 C_INCLUDE_PATH（对于 C）和 CPLUS_INCLUDE_PATH（对于 C++）给定的目录，

* 然后是标准系统目录，

最后是由 -idirafter 选项给定的目录（按照从左到右的顺序）。

链接器按以下顺序搜索库文件：

* 首先是由 -L 选项给定的目录（按照从左到右的顺序），

* 然后是由环境变量 LIBRARY_PATH 给定的目录（按照从左到右的顺序）。

### 分发使用 Cython 编译的扩展模块

当您使用 setuptools.Extension 类声明 [Cython](https://pypi.org/project/Cython) 扩展模块时，setuptools 将在构建时检测 Cython 是否已安装。

如果 Cython 存在，则 setuptools 将使用它来构建 .pyx 文件。否则，setuptools 将尝试找到并编译等效的 .c 文件（而不是 .pyx）。这些文件可以使用 [cython 命令行工具](https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html)生成。

您可以通过将其作为构建依赖项包含在 pyproject.toml 中，确保 Cython 始终自动安装到构建环境中：

```toml
[build-system]
requires = [
    # ...,
    "cython",
]
```

或者，您还可以将由 Cython 预编译的 .c 代码与原始的 .pyx 文件一起包含在源分发中（这样可以在从源代码构建时节省几秒钟时间）。为了提高版本兼容性，您可能还希望将当前的 .c 文件包含在您的版本控制系统中，并在检查 .pyx 源文件的更改时重新构建它们。这将确保跟踪您的项目的人可以在不安装 Cython 的情况下构建它，并且由于生成的 C 文件中的小差异不会产生变化。请查看我们关于[控制分发中的文件](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#controlling-files-in-the-distribution) 的文档以获取更多信息。

### Extension API 参考

#### class setuptools.Extension(name, sources, \*args, \*\*kw)

描述单个扩展模块。

这意味着所有源文件将编译成单个二进制文件 \<module path>.\<suffix>（其中 <module path> 源自 name，<suffix> 定义为 importlib.machinery.EXTENSION_SUFFIXES 中的一个值）。

如果将 .pyx 文件作为源文件传递，并且在构建环境中未安装 Cython，则 setuptools 还可能尝试查找等效的 .cpp 或 .c 文件。

参数:

* name（str）- 扩展的完整名称，包括任何包 – 即不是文件名或路径名，而是 Python 的点分隔名称

* sources（list[str]）- 源文件名列表，相对于分发根目录（设置脚本所在的位置），以 Unix 格式（斜杠分隔）表示，以实现可移植性。源文件可以是 C、C++、SWIG（.i）、特定于平台的资源文件，或任何“build_ext”命令视为 Python 扩展源的其他文件。

* include_dirs（list[str]）- 要搜索 C/C++ 头文件的目录列表（以 Unix 格式表示，以实现可移植性）

* define_macros（list[tuple[str, str\|None]]）- 要定义的宏列表；每个宏都使用 2-元组定义：第一个项目对应于宏的名称，第二个项目是它的值的字符串，或者是 None，表示不定义具体的值（相当于源文件中的 “\#define FOO” 或 Unix C 编译器命令行中的 -DFOO）

* undef_macros（list[str]）- 要明确取消定义的宏列表

* library_dirs（list[str]）- 要在链接时搜索 C/C++ 库的目录列表

* libraries（list[str]）- 要链接的库名称列表（不是文件名或路径）

* runtime_library_dirs（list[str]）- 要在运行时搜索 C/C++ 库的目录列表（对于共享扩展，当加载扩展时）。在 Windows 平台上设置这个会导致构建时抛出异常。

* extra_objects（list[str]）- 要链接的额外文件列表（例如，不是由 ‘sources’ 隐式导出的对象文件，必须明确指定的静态库，二进制资源文件等）

* extra_compile_args（list[str]）- 编译源文件中使用的任何额外平台和编译器特定信息。对于 “command line” 有意义的平台和编译器，这通常是一个命令行参数列表，但对于其他平台，它可以是任何东西。

* extra_link_args（list[str]）- 链接对象文件以创建扩展的任何额外平台和编译器特定信息（或者创建一个新的静态 Python 解释器）。与 ‘extra_compile_args’ 一样的解释。

* export_symbols（list[str]）- 从共享扩展中导出的符号列表。不是所有平台都使用，对于通常仅导出一个符号的 Python 扩展来说通常不是必要的：“init” + extension_name。

* swig_opts (list[str]) – 如果源文件具有 .i 扩展名，则传递给 SWIG 的任何额外选项。

* depends (list[str]) – 扩展依赖的文件列表

* language (str) – 扩展语言（例如“c”，“c++”，“objc”）。如果未提供，将从源扩展中检测到。

* optional (bool) – 指定扩展中的构建失败不应中止构建过程，而是简单地不安装失败的扩展。

* py_limited_api (bool) – 用于使用 Python 有限 API 的选择性标志。

RAISES： **setuptools.errors.PlatformError** – 如果在 Windows 上指定了 ‘runtime_library_dirs’。 (自 v63 起)
