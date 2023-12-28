---
layout:     post
title:      "Pipenv教程"
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

本文介绍Pipenv的用途以及为什么需要它，然后简单的介绍使用pipenv管理项目依赖的方法。

 <!--more-->

**目录**
* TOC
{:toc} 

Pipenv是Python的一个打包工具，解决了使用pip、virtualenv和传统的requirements.txt时常见的一些问题。

除了解决一些常见问题外，它还通过一个单一的命令行工具 consolidaed 并简化了开发过程。

本文将介绍Pipenv解决的问题，以及如何使用Pipenv管理Python依赖项。此外，它还将介绍Pipenv与以前的软件包分发方法的关系。 

## 为什么要用pipenv

要理解Pipenv的好处，重要的是要了解Python中当前的软件包和依赖管理方法。让我们从处理第三方软件包的典型情况开始。然后，我们将逐步构建一个完整的Python应用程序的部署过程。

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

Pipenv内置了虚拟环境管理，因此你可以使用一个单一的工具进行软件包管理。


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



### 与Maven/Gradle/Docker对比


如果读者做过Java或者Android开发，就会发现Pipenv和maven有相似之处。首先它们都是基于项目的管理方式，而不是像venv和conda基于环境。这样的隔离会更加彻底，因为如果多个项目共用一个环境的话，到了某一天它们引用的同一个包发生冲突就不好处理了。每个项目都有自己的环境，就不会跟别人冲突，但是可能会造成磁盘空间的浪费，因为可能很多项目的一些基础依赖是相同的。一种解决方法是把这些公共的依赖安装到系统的python里。在这一点来看，Java里maven的依赖是放在统一的位置，如果多个项目依赖同一个版本，那么就不会重复下载和存储。

这就是pip和maven最大的区别：pip是基于一个虚拟环境的，而maven是基于项目的。另外一个区别就是maven中的依赖必须精确制定版本号，这类似于"pip install abc==1.2.3"。这样的好处是整个构建是确定的，每个发布出去的jar包是永远不会改变的。同样的代码，保证能够精准复现编译过程。但是pip一开始大家都是不习惯指定版本号，后来发现有问题，才慢慢加入版本号，但是通常也很少指定精确的版本号(==1.2.3)。而maven习惯指定精确的版本号，即使maven也有Version Range的功能，比如：

```java
        <dependency>
            <groupId>org.checkerframework</groupId>
            <artifactId>checker-qual</artifactId>
            <version>[3,4)</version>
        </dependency>
```

它等价于">=3,<4"。

使用精确版本号的好处是严格的可重复，这对于软件的稳定性至关重要。当然缺点就是很难升级，因为定期频繁的小部分升级是容易的，但是要把一个几年都没动过的包升级是很难的。和maven类似，docker也是要保证严格的可重复构建。所以我们如果要在dockerfile里拉取github的开源代码编译时，一定要制定版本号，否则人家一升级，我们的代码可能就要挂掉。




### Pipenv的冲突解决机制

Pipenv将尝试安装满足所有核心依赖项要求的子依赖项。然而，如果存在冲突的依赖关系（package_a需要package_c>=1.0，但package_b需要package_c<1.0），Pipenv将无法创建锁定文件，并输出如下错误：

```
Warning: Your dependencies could not be resolved. You likely have a mismatch in your sub-dependencies.
  You can use $ pipenv install --skip-lock to bypass this mechanism, then run $ pipenv graph to inspect the situation.
Could not find a version that matches package_c>=1.0,package_c<1.0
```

正如警告所说，你还可以显示一个依赖关系图，以了解你的顶级依赖关系及其子依赖关系：

```bash
$ pipenv graph
```

此命令将打印出一个类似树状结构的内容，显示你的依赖关系。以下是一个示例：

```
Flask==0.12.1
  - click [required: >=2.0, installed: 6.7]
  - itsdangerous [required: >=0.21, installed: 0.24]
  - Jinja2 [required: >=2.4, installed: 2.10]
    - MarkupSafe [required: >=0.23, installed: 1.0]
  - Werkzeug [required: >=0.7, installed: 0.14.1]
numpy==1.14.1
pytest==3.4.1
  - attrs [required: >=17.2.0, installed: 17.4.0]
  - funcsigs [required: Any, installed: 1.0.2]
  - pluggy [required: <0.7,>=0.5, installed: 0.6.0]
  - py [required: >=1.5.0, installed: 1.5.2]
  - setuptools [required: Any, installed: 38.5.1]
  - six [required: >=1.10.0, installed: 1.11.0]
requests==2.18.4
  - certifi [required: >=2017.4.17, installed: 2018.1.18]
  - chardet [required: >=3.0.2,<3.1.0, installed: 3.0.4]
  - idna [required: >=2.5,<2.7, installed: 2.6]
  - urllib3 [required: <1.23,>=1.21.1, installed: 1.22]
```

从pipenv graph的输出中，你可以看到我们先前安装的顶级依赖项（Flask、numpy、pytest和requests），在它们下面你可以看到它们依赖的包。

此外，你还可以颠倒树状结构，以显示需要它的父级的子依赖项：

```
$ pipenv graph --reverse
```

当你试图解决冲突的子依赖项时，这个反向树可能更有用。它可以让我们很快找到冲突的被依赖包是谁引入的。

## Pipenv使用简介

下面我们用一个例子来介绍Pipenv的使用。我们首先假设的场景是从零创建一个项目，之后再模拟从github拉取别人项目修改的例子。当然在这之前是需要安装。最简单的安装方法是使用pip，如果不想影响别人，可以使用--user安装：

```
pip install --user pipenv
```

更多的安装方法和问题请参考[官方安装文档](https://pipenv.pypa.io/en/latest/installation.html)。

### 创建项目


我们首先创建一个项目，假设我们想简单的测试一下[Huggingface Transformers](https://huggingface.co/docs/transformers/index)。

```
$ mkdir transformers-test
$ cd transformers-test
```

当然现在我们有了一个空的项目目录，我们在创建python项目时最重要的当然是选择python的版本，我这里选择3.9。因此我们用如下命令初始化一个项目：

```
pipenv --python 3.9
输出：
Using /usr/bin/python3.9 (3.9.18) to create virtualenv...
⠼ Creating virtual environment...created virtual environment CPython3.9.18.final.0-64 in 219ms
  creator CPython3Posix(dest=/home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/ubuntu/.local/share/virtualenv)
    added seed packages: pip==23.3.2, setuptools==69.0.3, wheel==0.42.0
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

✔ Successfully created virtual environment!
Virtualenv location: /home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y
Creating a Pipfile for this project...
```

可以看到它为我们在"/home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y"位置创建了一个virtual environment环境。那它怎么知道对于我们的这个项目应该使用这个venv呢？这个映射关系藏在名字"transformers-test-K5GLQG9y"里，它是目录名字和目录绝对路径的hash用-连接起来。聪明的读者可能会问，那万一我移动这个目录怎么办？凉拌！那它就不知道这个新位置的目录对应那个venv了。当然我们可以使用后面的"pipenv sync"或者"pipenv update"重新安装一个新的venv。但是读者可能不满意，那在硬盘上不是多了一个垃圾目录吗？我要手工删掉。我怎么知道环境在哪？万一删错了怎么办。官方推荐的移动目录的方法是：
```
pipenv --rm
cd ..
mv abc edf
cd edf
pipenv install
```
也就是先删除当前项目(在项目主目录下运行)，然后移动目录，最后重新创建venv和安装依赖。

如果我们想让venv安装在当前项目下，可以设置环境变量PIPENV_VENV_IN_PROJECT=1。

如果我们想修改venv被安装的路径，可以用：

```
export WORKON_HOME=~/.venvs
```

运行命令之后，我们可以发现在当前目录下多了一个Pipfile，它的内容是：

```
$ cat Pipfile 
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]

[dev-packages]

[requires]
python_version = "3.9"
python_full_version = "3.9.18"
```

Pipfile的语法是TOML，并且文件被分为不同的部分。[dev-packages] 用于仅在开发环境中使用的包，[packages] 必需的包，而 [requires] 用于其他要求，比如特定版本的Python。

### 安装依赖
```
$ pipenv install torch torchvision torchaudio transformers
Installing torch...
Resolving torch...
Added torch to Pipfile's [packages] ...
✔ Installation Succeeded
Installing torchvision...
Resolving torchvision...
Added torchvision to Pipfile's [packages] ...
✔ Installation Succeeded
Installing torchaudio...
Resolving torchaudio...
Added torchaudio to Pipfile's [packages] ...
✔ Installation Succeeded
Installing transformers...
Resolving transformers...
Added transformers to Pipfile's [packages] ...
✔ Installation Succeeded
Pipfile.lock not found, creating...
Locking [packages] dependencies...
Building requirements...
Resolving dependencies...
✔ Success!
Locking [dev-packages] dependencies...
Updated Pipfile.lock (d7b8c69d4267d800fcd61210fba73b365b55afad47f08af65fa3c36c127bfca3)!
Installing dependencies from Pipfile.lock (7bfca3)...
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
```

我这里是测试，如果网速慢的读者可以用自己喜欢的包来替代。我们可以看一下目录多了一个Pipfile.lock。这个文件比较大，详细的列举了每个依赖的安装情况，我们来看其中一个：

```
        "torch": {
            "hashes": [
                "sha256:05b18594f60a911a0c4f023f38a8bda77131fba5fd741bda626e97dcf5a3dd0a",
                "sha256:0e13034fd5fb323cbbc29e56d0637a3791e50dd589616f40c79adfa36a5a35a1",
                "sha256:255b50bc0608db177e6a3cc118961d77de7e5105f07816585fa6f191f33a9ff3",
                "sha256:33d59cd03cb60106857f6c26b36457793637512998666ee3ce17311f217afe2b",
                "sha256:3a871edd6c02dae77ad810335c0833391c1a4ce49af21ea8cf0f6a5d2096eea8",
                "sha256:6984cd5057c0c977b3c9757254e989d3f1124f4ce9d07caa6cb637783c71d42a",
                "sha256:76d37967c31c99548ad2c4d3f2cf191db48476f2e69b35a0937137116da356a1",
                "sha256:8e221deccd0def6c2badff6be403e0c53491805ed9915e2c029adbcdb87ab6b5",
                "sha256:8f32ce591616a30304f37a7d5ea80b69ca9e1b94bba7f308184bf616fdaea155",
                "sha256:9ca96253b761e9aaf8e06fb30a66ee301aecbf15bb5a303097de1969077620b6",
                "sha256:a6ebbe517097ef289cc7952783588c72de071d4b15ce0f8b285093f0916b1162",
                "sha256:bc195d7927feabc0eb7c110e457c955ed2ab616f3c7c28439dd4188cf589699f",
                "sha256:bef6996c27d8f6e92ea4e13a772d89611da0e103b48790de78131e308cf73076",
                "sha256:d93ba70f67b08c2ae5598ee711cbc546a1bc8102cef938904b8c85c2089a51a0",
                "sha256:d9b535cad0df3d13997dbe8bd68ac33e0e3ae5377639c9881948e40794a61403",
                "sha256:e0ee6cf90c8970e05760f898d58f9ac65821c37ffe8b04269ec787aa70962b69",
                "sha256:e2d83f07b4aac983453ea5bf8f9aa9dacf2278a8d31247f5d9037f37befc60e4",
                "sha256:e3225f47d50bb66f756fe9196a768055d1c26b02154eb1f770ce47a2578d3aa7",
                "sha256:f41fe0c7ecbf903a568c73486139a75cfab287a0f6c17ed0698fdea7a1e8641d",
                "sha256:f9a55d55af02826ebfbadf4e9b682f0f27766bc33df8236b48d28d705587868f"
            ],
            "index": "pypi",
            "markers": "python_full_version >= '3.8.0'",
            "version": "==2.1.2"
```

主要内容就是"version"，说明安装的是2.1.2，另外的hashes是用于check文件一致性。

### 代码开发

依赖安装好了，接着就是我们自己的代码了，我们写一个简单的测试代码。

```
$ cat test_transformers.py 
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("We are very happy to show you the 🤗 Transformers library."))
```

很好，我们来运行一下吧。

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ python test_transformers.py 
Traceback (most recent call last):
  File "test_transformers.py", line 1, in <module>
    from transformers import pipeline
ModuleNotFoundError: No module named 'transformers'
```

怎么找不到transformers呢？刚才不是安装了吗？虽然我们安装好了依赖，但是那些依赖是安装在一个venv里的，我们还没有activate啊。熟悉venv的读者肯定着急要执行activate脚本了，但是那个venv的路径在哪里呢？前面好像说是在~/.local/share/virtualenvs/下面，赶紧去找吧。

不要着急，pipenv提供了命令帮我们激活环境，甚至我们可以不激活环境也可以用pipenv run运行我们的代码。

```
$ pipenv run python test_transformers.py
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
[{'label': 'POSITIVE', 'score': 0.9997795224189758}]
```
没问题，而且我们也没有看到它激活环境。我们也可以用pipenv shell激活venv：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ pipenv shell
Launching subshell in virtual environment...
 . /home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y/bin/activate
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$  . /home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y/bin/activate
(transformers-test)ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ python test_transformers.py 
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
[{'label': 'POSITIVE', 'score': 0.9997795224189758}]
(transformers-test)ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ exit
exit
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ 
```

我们看到，执行pipenv shell之后，它帮助我们激活了对应的venv。这样就可以直接用到安装好的包了，如果想退出输入exit就行。

如果我们实在想自己激活环境，也可以使用"pipenv --venv"找到venv的路径：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ pipenv --venv
/home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y
```
另外如果我们用一些开发工具的话，可能需要告诉它这个路径。不过像VsCode这样的工具，它自己会去~/.local/share/virtualenvs/寻找环境，我们只需要Ctrl+Shift+P就可以选择环境了，如下图所示：

<a>![](/img/pipenv/1.png)</a>

### 增加新的依赖

开发的过程中，我们可能需要安装新的包，比如我们需要accelerate包来加载更大的模型，我们可以：

```
$ pipenv install accelerate
Installing accelerate...
Resolving accelerate...
Added accelerate to Pipfile's [packages] ...
✔ Installation Succeeded
Pipfile.lock (7bfca3) out of date, updating to (40fc66)...
Locking [packages] dependencies...
Building requirements...
Resolving dependencies...
✔ Success!
Locking [dev-packages] dependencies...
Updated Pipfile.lock (fa1fad14015e1bb7b99707d25f00feeb4f38fe90c2a9440921027f6d3b40fc66)!
Installing dependencies from Pipfile.lock (40fc66)...
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
```

我们的Pipfile变成了：

```
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
torch = "*"
torchvision = "*"
torchaudio = "*"
transformers = "*"
accelerate = "*"

[dev-packages]

[requires]
python_version = "3.9"
python_full_version = "3.9.18"
```
### 修改版本

通常我们可以会升级依赖，但是降级也是有可能的。比如我们安装一个新的包，它只能在accelerate <=0.24.0下运行。通常pipenv install会帮我们解决冲突，选择合适的版本。

这里假设我们自己需要降级，安装accelerate == 0.24.0，那么可以这样：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ pipenv install accelerate==0.24.0
Installing accelerate==0.24.0...
Resolving accelerate==0.24.0...
✔ Installation Succeeded
Pipfile.lock (40fc66) out of date, updating to (d23bb3)...
Locking [packages] dependencies...
Building requirements...
Resolving dependencies...
✔ Success!
Locking [dev-packages] dependencies...
Updated Pipfile.lock (4863185e51ede195e8b48dac701b02acdaab10cf2a4cbf29e2061a2809d23bb3)!
Installing dependencies from Pipfile.lock (d23bb3)...
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
```

我们看一下：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ cat Pipfile
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
torch = "*"
torchvision = "*"
torchaudio = "*"
transformers = "*"
accelerate = "==0.24.0"

[dev-packages]

[requires]
python_version = "3.9"
python_full_version = "3.9.18"
```

另外我们去Pipfile.lock也能发现，确实安装的版本是0.24.0。如果你还不放心，可能去venv里确认一下：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ ll /home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y/lib/python3.9/site-packages/accelerate-0.24.0.dist-info/
total 68
drwxrwxr-x  2 ubuntu ubuntu  4096 Dec 28 11:28 ./
drwxrwxr-x 88 ubuntu ubuntu  4096 Dec 28 11:28 ../
-rwxrwxr-x  1 ubuntu ubuntu   238 Dec 28 11:28 entry_points.txt*
-rw-rw-r--  1 ubuntu ubuntu     4 Dec 28 11:28 INSTALLER
-rw-rw-r--  1 ubuntu ubuntu 11357 Dec 28 11:28 LICENSE
-rw-rw-r--  1 ubuntu ubuntu 18080 Dec 28 11:28 METADATA
-rw-rw-r--  1 ubuntu ubuntu 12043 Dec 28 11:28 RECORD
-rw-rw-r--  1 ubuntu ubuntu     0 Dec 28 11:28 REQUESTED
-rwxrwxr-x  1 ubuntu ubuntu    11 Dec 28 11:28 top_level.txt*
-rw-rw-r--  1 ubuntu ubuntu    92 Dec 28 11:28 WHEEL
```

### 再升级

假设我们解决了问题，我们可以在accelerate==0.25.0下工作了，那么我们可以升级：

```
$ pipenv update accelerate
$ pipenv run pip freeze|grep accelerate
accelerate==0.25.0
```
但是这种升级会升级到最新的版本，如果我们需要保守一点，那么可以让它不超过0.26.0。因为大的版本变化通常容易不兼容：

```
$ pipenv update "accelerate<0.26.0"
```

运行结束后：
```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ cat Pipfile
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
torch = "*"
torchvision = "*"
torchaudio = "*"
transformers = "*"
accelerate = "<0.26.0"

[dev-packages]

[requires]
python_version = "3.9"
python_full_version = "3.9.18"
```

实际安装的是0.25.0：

```bash
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ ls /home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y/lib/python3.9/site-packages/ |grep accelerate-
accelerate-0.25.0.dist-info
```

### upgrade

与update类似的命令是upgrade，它的作用是更新Pipfile和Pipflie.lock。我来测试一下：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ pipenv upgrade accelerate
Building requirements...
Resolving dependencies...
✔ Success!
Building requirements...
Resolving dependencies...
✔ Success!
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ ls /home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y/lib/python3.9/site-packa
ges/ |grep accelerate-
accelerate-0.24.0.dist-info
```
我们看到venv里并没有更新，它只是更新了Pipfile和Pipfile.lock：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ cat Pipfile
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
torch = "*"
torchvision = "*"
torchaudio = "*"
transformers = "*"
accelerate = "*"

[dev-packages]

[requires]
python_version = "3.9"
python_full_version = "3.9.18"

ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ cat Pipfile.lock |grep -A 8 accelerate
        "accelerate": {
            "hashes": [
                "sha256:c7bb817eb974bba0ff3ea1ba0f24d55afb86d50e3d4fe98d6922dc69cf2ccff1",
                "sha256:ecf55b0ab278a1dac8539dde0d276977aff04683f07ede73eaf02478538576a1"
            ],
            "index": "pypi",
            "markers": "python_full_version >= '3.8.0'",
            "version": "==0.25.0"
        },
```

和update一样，我们也可以指定"accelerate<0.26.0"。那怎么在venv里安装0.25.0呢？可以是pipenv sync，它的意思是按照lock文件来更新依赖：

```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ pipenv sync
Installing dependencies from Pipfile.lock (40fc66)...
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
All dependencies are now up-to-date!
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test$ ls /home/ubuntu/.local/share/virtualenvs/transformers-test-K5GLQG9y/lib/python3.9/site-packages/ |grep accelerate-
accelerate-0.25.0.dist-info
```

因此这里update等价于upgrade+sync。那为什么要把update拆分成两步呢？我设想可能的场景是：我们想升级到accelerate<0.26试试，看看有没有冲突或者看看它会安装那个版本，但是并没有打算马上

### 保存和发布代码

好的习惯是用git等工具管理我们的代码，那么我们应该把Pipfile和Pipfile.lock都纳入版本控制。然后提交到中央的仓库了。

### 使用项目

很多时候，我们并不是一个项目的owner，比如我们从github上clone一个代码库然后做一些很小的修改。这个时候就不需要初始化项目了。我这里用下面的复制来替代glone命令，表示我现在是另外一个开发者：
```bash
cp -r transformers-test transformers-test2
cd transformers-test2
```
现在这个目录只有3个文件：
```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test2$ ls
Pipfile  Pipfile.lock  test_transformers.py
```
我们直接运行是不行的：
```
ubuntu@VM-128-7-ubuntu:~/lili/transformers-test2$ pipenv run python test_transformers.py 
Traceback (most recent call last):
  File "/home/ubuntu/lili/transformers-test2/test_transformers.py", line 1, in <module>
    from transformers import pipeline
ModuleNotFoundError: No module named 'transformers'
```

因为我们还没有创建与这个目录对应的venv。这个时候我们通常使用pipenv sync来创建venv并且根据Pipfile.lock来安装依赖。为什么不使用pipenv update来根据Pipfile来安装呢？因为我们拉取代码时距离作者上传可能很久了，当时用的transformers没有指定版本，用的是当时最新的4.36.2，但是等到一年后我们再根据Pipfile安装，可能就更新到5.xxx了，很可能我们的代码就不能运行了。而Pipfile.lock里(如果没有出现前面那种pipenv upgrade的操作)的版本和作者自己环境的版本是一模一样的，这就不会出问题。

好了，我们sync之后，就可以接着修改代码，比如我们想修改自己的代码使得它兼容5.xxx，那么我们就可以用pipenv update transformers
更新到我们想要的版本然后开发。。。

### 其它命令

pipenv lock会使得Pipfile.lock和Pipfile一致。通常我们不需要使用这个命令，除非我们手动修改了Pipfile。

pipenv uninstall 用于卸载package。

## 常见问题


### 怎么“激活”一个环境
习惯了venv/virtualenv和conda的读者最先想问的可能是：怎么conda activate或者source激活一个环境。
一般情况下(如果使用vscode等工具开发)，我们不需要做这个事情。我们一般也不需要知道这个环境，如果我们想命令运行，也可以使用:
```
pipenv run python xxx.py 
```
那么我们的python程序自动的就是在这个项目对应的venv里运行了。当然我们也可以使用pipenv shell激活这个环境：

```
$ pipenv shell

$ pip freeze|grep transformers
$ exit
```
我们看到，当使用pipenv shell后它会帮我们source那个环境，我们熟悉的(myenv)就出现在命令行提示里了。我们可以像在venv里使用。但是我们千万不能在里面用pip install安装，因为直接用pip安装的包不受pipenv管理。如果我们想退出，直接输入"exit"就行了。

### 怎么导入requirements.txt

```
pipenv install -r path/to/requirements.txt
```



### 怎么导出为requirements.txt

有些云平台不支持pipenv，那么可以导出精确版本的requirements.txt：

```
pipenv requirements > requirements.txt
```
老的版本是：
```
pipenv lock -r > requirements.txt
```

### pipenv命令没有自动补全

bash，把下列内容加到.bashrc里：
```
eval "$(pipenv --completion)"
```

### 我能直接编辑Pipfile和Pipfile.lock吗

Pipfile.lock不要直接编辑，而Pipfile可以手工编辑，但是最好不要手工编辑，因为pipenv命令可以完成你所有需要的操作。而且改了Pipfile而不运行相关命令，会让lock文件不一致。如果你想做一件事情发现只能修改Pipfile，那么首先去好好找找有没有命令可以帮你做，如果没有就去github提一个需求。



### 一个项目的环境只能给一个项目用吗

是的，理论上你可以用"pipenv shell"激活一个环境后去运行别的项目程序，但绝对不推荐这么做。如果你觉得两个项目创建两个环境太浪费空间，而且它们的依赖可以用同一个，那么可以在项目根目录中创建一个名为.venv的文件，其中包含指向虚拟环境的路径，pipenv将使用该路径，而不是自动生成的路径。

你需要保持两个目录的Pipfile和Pipfile.lock同步，否则venv可能就乱了。可以把其中一个设置为另外一个的符号链接，但如果这样的话符号链接就没有办法加到版本控制系统里了。

因此如果安装包不太大的话(浪费的空间也不大)，那么每个项目的重复安装相同的包到各自的venv里是比较好的选择。但是对于我们搞深度学习的来说，安装一个Pytorch就得好几GB，很多项目都是使用相同的Pytorch版本，这个浪费的空间就不少了。其实这个问题在pip里也存在，因为一开始可能所有项目共用一个venv，但是渐渐的就容易冲突，为了省事，我们也会经常一个项目创建一个venv。

如果读者一定要两个项目共享一些大的包，可以参考下一个问题。

### 怎么避免PyTorch这样的超级大包每次都重复被安装

#### 先用一个conda来安装这些超级大包
```
conda create -n torchshare python=3.9
conda activate torchshare
pip install torch

$ pip list|grep torch
torch                    2.1.2
```

#### 安装pipenv

```
$ pip install pipenv
$ which pipenv
/home/ubuntu/anaconda3/envs/torchshare/bin/pipenv
```

#### 创建第一个项目
 
```
(torchshare) ubuntu@VM-128-7-ubuntu:~/lili$ mkdir proj1
(torchshare) ubuntu@VM-128-7-ubuntu:~/lili$ cd proj1/
(torchshare) ubuntu@VM-128-7-ubuntu:~/lili/proj1$ which python
/home/ubuntu/anaconda3/envs/torchshare/bin/python
(torchshare) ubuntu@VM-128-7-ubuntu:~/lili/proj1$ pipenv --python=`which python` --site-packages
Creating a virtualenv for this project...
Pipfile: /home/ubuntu/lili/proj1/Pipfile
Using /home/ubuntu/anaconda3/envs/torchshare/bin/python (3.9.18) to create virtualenv...
Making site-packages available...
⠸ Creating virtual environment...created virtual environment CPython3.9.18.final.0-64 in 177ms
  creator CPython3Posix(dest=/home/ubuntu/.local/share/virtualenvs/proj1-Jbp8dTW5, clear=False, no_vcs_ignore=False, global=True)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/ubuntu/.local/share/virtualenv)
    added seed packages: pip==23.3.2, setuptools==69.0.3, wheel==0.42.0
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

✔ Successfully created virtual environment!
Virtualenv location: /home/ubuntu/.local/share/virtualenvs/proj1-Jbp8dTW5
Creating a Pipfile for this project...
```

注意上面的初始化命令通过--python指定使用conda的python，并且使用--site-packages告诉它我们需要继承来自conda的torch。我们可以确认一下在这个虚拟环境下可以使用torch：

```
(torchshare) ubuntu@VM-128-7-ubuntu:~/lili/proj1$ pipenv run python -c "import torch;print(torch.__version__)"
2.1.2+cu121
```

我们也可以看一下venv里的配置：

```
(torchshare) ubuntu@VM-128-7-ubuntu:~/lili/proj1$ cat `pipenv --venv`/pyvenv.cfg
home = /home/ubuntu/anaconda3/envs/torchshare/bin
implementation = CPython
version_info = 3.9.18.final.0
virtualenv = 20.25.0
include-system-site-packages = true
base-prefix = /home/ubuntu/anaconda3/envs/torchshare
base-exec-prefix = /home/ubuntu/anaconda3/envs/torchshare
base-executable = /home/ubuntu/anaconda3/envs/torchshare/bin/python
prompt = proj1
```

可以看到，python确实指向了conda，并且include-system-site-packages为true。

这个时候我们可以安装其它依赖：

```
$ pipenv install transformers accelerator
```


#### 创建第二个项目

和之前一样，这样两个项目可以共享conda环境里的pytorch。



## 参考文献
* [Pipenv官网](https://pipenv.pypa.io/en/latest/)

* [Pipenv: A Guide to the New Python Packaging Tool](https://realpython.com/pipenv-guide/)

* [Pipenv — The Gold Standard for Virtual Environments in Python](https://medium.com/@danilo.drobac/pipenv-the-gold-standard-for-virtual-environments-in-python-204c120e9c27)

* [Python virtualenv and venv dos and don’ts](https://www.infoworld.com/article/3306656/python-virtualenv-and-venv-dos-and-donts.html)

* [Stackoverflow: Keeping the same, shared virtualenvs when switching from pyenv-virtualenv to pipenv](https://stackoverflow.com/questions/55892572/keeping-the-same-shared-virtualenvs-when-switching-from-pyenv-virtualenv-to-pip/55893055#55893055)
