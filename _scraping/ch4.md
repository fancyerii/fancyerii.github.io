---
layout:     post
title:      "第四章：Writing Your First Web Scraper"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - python
    - scraping
---

<!--more-->
**目录**
* TOC
{:toc}

这一章的内容从发送GET请求（请求获取网页内容）开始，读取该页面的HTML输出，并进行一些简单的数据提取，以便隔离你所寻找的内容。在开始之前，让我们先来欣赏一下网页浏览器为我们做的一切小事情。刚开始网页抓取时，没有HTML格式化、CSS样式、JavaScript执行和图像渲染的网络可能会显得有点令人畏惧。在这一章中，我们将开始探讨如何在没有网页浏览器的帮助下格式化和解释这些基本数据。 


## 安装和使用Jupyter。

 这门课程的代码可以在https://github.com/REMitchell/python-scraping 上找到。 在大多数情况下，代码示例是Jupyter Notebook文件的形式，扩展名为.ipynb。 如果你还没有使用过它们，Jupyter Notebook是一种极好的方式，可以组织和处理许多小但相关的Python代码片段，如图4-1所示。




<a>![](/img/scraping/ch4/1.png)</a>

每段代码都包含在一个名为单元格的框中。你可以通过输入Shift + Enter或点击页面顶部的运行按钮来运行每个单元格中的代码。

Jupyter项目始于2014年的IPython（交互式Python）项目的一个衍生项目。这些笔记本设计用于在浏览器中以可访问和交互的方式运行Python代码，适合于教学和演示。

要安装Jupyter笔记本：

```
pip install notebook
```

在安装完成后，你应该可以访问jupyter命令，该命令将允许你启动Web服务器。导航至包含本书下载的练习文件的目录，并运行：

```
jupyter notebook
```
这将在8888端口上启动Web服务器。如果你已经运行了一个Web浏览器，一个新的标签页应该会自动打开。如果没有，请将终端中显示的带有提供的令牌的URL复制到你的Web浏览器中。

## 连接

在这本书的第一部分中，我们深入探讨了互联网如何将数据包从浏览器发送到Web服务器，然后再次返回。当你打开一个浏览器，输入google.com并按下回车键时，正是这种情况发生了——数据以HTTP请求的形式从你的计算机传输出去，谷歌的Web服务器以HTML文件的形式回应，代表了google.com根目录下的数据。

但是，在这些数据包和帧的交换中，Web浏览器到底起到了什么作用呢？实际上，完全没有。事实上，ARPANET（第一个公共分组交换网络）比第一个Web浏览器Nexus要早至少20年。

是的，Web浏览器是一个有用的应用程序，用于创建这些信息包，并告诉你的操作系统发送它们，并将你收到的数据解释为漂亮的图片、声音、视频和文本。然而，Web浏览器只是代码，而代码可以被拆解，分解为其基本组件，重新编写、重复使用，并被制作成任何你想要的东西。Web浏览器可以告诉处理无线（或有线）接口的应用程序发送数据的处理器，但是你可以使用Python仅需三行代码来做同样的事情：

```
from urllib.request import urlopen
html = urlopen('http://pythonscraping.com/pages/page1.html')
print(html.read())
```


要运行这个代码，你可以使用GitHub仓库中第1章的IPython笔记本，或者你可以将其保存到本地作为scrapetest.py，并在终端中使用以下命令运行它：

```
python scrapetest.py
```


请注意，如果你的计算机上同时安装了Python 2.x，并且同时运行两个Python版本，你可能需要显式地调用Python 3.x来运行这个命令：

```
python3 scrapetest.py
```

这个命令输出了位于URL http://pythonscraping.com/pages/page1.html 的页面1的完整HTML代码。更准确地说，这输出了在位于http://pythonscraping.com 的域名的服务器上的\<web root>/pages 目录中找到的HTML文件page1.html。

为什么将这些地址看作“文件”而不是“页面”很重要？大多数现代网页都与许多资源文件相关联。这些文件可能是图片文件、JavaScript文件、CSS文件，或者与您请求的页面相关联的任何其他内容。当Web浏览器遇到像\<img src="cute Kitten.jpg">这样的标签时，浏览器知道它需要向服务器发出另一个请求，以获取位置cuteKitten.jpg处的数据，以便完全渲染页面供用户查看。

当然，你的Python脚本还没有逻辑去返回并请求多个文件（但现在）；它只能读取你直接请求的单个HTML文件。

```
from urllib.request import urlopen
```

这意味着它看起来像它的意思：它查看Python模块request（在urllib库中找到）并仅导入函数urlopen。

urllib是一个标准的Python库（这意味着你不需要安装任何额外的东西来运行这个示例），它包含用于在网络上请求数据、处理Cookie，甚至更改元数据（如头部和用户代理）的函数。我们将在整本书中广泛使用urllib，所以我建议你阅读一下该库的Python文档。

urlopen用于打开网络上的远程对象并读取它。由于它是一个相当通用的函数（它可以轻松地读取HTML文件、图像文件或任何其他文件流），我们将在整本书中经常使用它。



## 介绍 BeautifulSoup

>Beautiful Soup, so rich and green,
>Waiting in a hot tureen!
>Who for such dainties would not stoop?
>Soup of the evening, beautiful Soup!

BeautifulSoup 库的名字来源于刘易斯·卡罗尔在《爱丽丝梦游仙境》中的同名诗歌。在故事中，这首诗由一个叫假乌龟（Mock Turtle）的角色演唱（假乌龟本身是对维多利亚时期流行的假乌龟汤的双关语，这种汤不是用乌龟，而是用牛肉做的）。

像它在《仙境》中的同名角色一样，BeautifulSoup 尝试理解无意义的东西；它通过修复糟糕的 HTML 并向我们呈现易于遍历的代表 XML 结构的 Python 对象，帮助格式化和组织混乱的网页。

### 安装 BeautifulSoup

因为 BeautifulSoup 库不是默认的 Python 库，所以必须安装。如果你已经有安装 Python 库的经验，请使用你喜欢的安装程序并跳到下一节 “运行 BeautifulSoup”（第46页）。对于那些没有安装过 Python 库（或需要复习）的读者来说，本书将使用这个通用方法来安装多个库，因此你可能希望将这一节作为未来的参考。

在整本书中，我们将使用 BeautifulSoup 4 库（也称为 BS4）。BeautifulSoup 4 的完整文档以及安装说明可以在 Crummy.com 上找到。

如果你花了很多时间编写 Python 代码，你可能已经使用过 Python 的包安装器（pip）。如果没有，我强烈建议你安装 pip 以便安装 BeautifulSoup 和本书中使用的其他 Python 包。根据你使用的 Python 安装程序，pip 可能已经安装在你的计算机上。要检查，请尝试：

```bash
pip
```

这个命令应打印 pip 帮助文本到你的终端。如果命令未被识别，你可能需要安装 pip。pip 可以通过多种方式安装，例如在 Linux 上使用 apt-get 或在 macOS 上使用 brew。无论你的操作系统是什么，你也可以下载 pip 引导文件 https://bootstrap.pypa.io/get-pip.py，保存此文件为 get-pip.py，并使用 Python 运行它：

```bash
python get-pip.py
```

再次注意，如果你的机器上同时安装了 Python 2.x 和 3.x，你可能需要显式调用 python3：

```bash
python3 get-pip.py
```

最后，使用 pip 安装 BeautifulSoup：

```bash
pip install bs4
```

如果你有两个版本的 Python，以及两个版本的 pip，你可能需要调用 pip3 来安装 Python 3.x 版本的包：

```bash
pip3 install bs4
```

就这样！现在 BeautifulSoup 将被识别为你机器上的一个 Python 库。你可以通过打开一个 Python 终端并导入它来测试：

```python
from bs4 import BeautifulSoup
```

导入应无错误完成。

**使用虚拟环境保持库的清晰**

如果你打算进行多个 Python 项目，或者你需要一种简单的方式来捆绑所有相关库的项目，或者你担心已安装库之间的潜在冲突，你可以安装一个 Python 虚拟环境以保持一切分离并易于管理。

当你在没有虚拟环境的情况下安装 Python 库时，你是在全局安装它。这通常需要你是管理员，或者以 root 身份运行，并且 Python 库存在于机器上的每个用户和每个项目中。幸运的是，创建虚拟环境很简单：

```bash
virtualenv scrapingEnv
```

这会创建一个名为 scrapingEnv 的新环境，你必须激活它才能使用：

```bash
cd scrapingEnv/
source bin/activate
```

激活环境后，你会在命令行提示符中看到该环境的名称，提醒你当前正在使用它。你安装的任何库或运行的脚本都将仅在该虚拟环境中。

在新创建的 scrapingEnv 环境中工作时，你可以安装并使用 BeautifulSoup；例如：

```bash
(scrapingEnv)ryan$ pip install beautifulsoup4
(scrapingEnv)ryan$ python
> from bs4 import BeautifulSoup
>
```

你可以使用 deactivate 命令离开该环境，之后你将无法访问在虚拟环境中安装的任何库：

```bash
(scrapingEnv)ryan$ deactivate
ryan$ python
> from bs4 import BeautifulSoup
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
ImportError: No module named 'bs4'
```

通过项目分离所有库也使得将整个环境文件夹压缩并发送给其他人变得容易。只要他们在其机器上安装了相同版本的 Python，你的代码就可以在虚拟环境中工作，而不需要他们自己安装任何库。

尽管我不会明确指示你在本书的所有示例中使用虚拟环境，但请记住，你可以在任何时候简单地通过预先激活它来应用虚拟环境。

### 运行 BeautifulSoup

在 BeautifulSoup 库中最常用的对象是 BeautifulSoup 对象。让我们来看一下它的实际应用，修改本章开头的示例：

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen('http://www.pythonscraping.com/pages/page1.html')
bs = BeautifulSoup(html.read(), 'html.parser')
print(bs.h1)
```

输出如下：

```
<h1>An Interesting Title</h1>
```

注意，这只返回了页面中找到的第一个 h1 标签实例。按照惯例，一个页面上应该只有一个 h1 标签，但在网络上，惯例常常被打破，所以你应该意识到这只会检索第一个 h1 标签实例，而不一定是你想要的那个。

在前面的网页抓取示例中，你导入了 urlopen 函数并调用了 html.read() 来获取页面的 HTML 内容。除了文本字符串外，BeautifulSoup 还可以直接使用 urlopen 返回的文件对象，而不需要先调用 .read()：

```python
bs = BeautifulSoup(html, 'html.parser')
```

这些 HTML 内容随后被转换成了具有以下结构的 BeautifulSoup 对象：

```
html → <html><head>...</head><body>...</body></html>
    head → <head><title>A Useful Page<title></head>
        title → <title>A Useful Page</title>
    body → <body><h1>An Int...</h1><div>Lorem ip...</div></body>
        h1 → <h1>An Interesting Title</h1>
    div → <div>Lorem Ipsum dolor...</div>
```

注意，你从页面中提取的 h1 标签在你的 BeautifulSoup 对象结构中嵌套了两层（html → body → h1）。然而，当你实际从对象中获取它时，你直接调用 h1 标签：

```python
bs.h1
```

实际上，以下任何函数调用都会产生相同的输出：

```python
bs.html.body.h1
bs.body.h1
bs.html.h1
```

当你创建一个 BeautifulSoup 对象时，会传入两个参数：

```python
bs = BeautifulSoup(html.read(), 'html.parser')
```

第一个参数是对象基于的 HTML 字符串，第二个参数指定你希望 BeautifulSoup 使用的解析器。在大多数情况下，选择哪个解析器并没有太大区别。

html.parser 是一个包含在 Python 3 中的解析器，使用时不需要额外安装。除非另有要求，我们将在整本书中使用这个解析器。

另一个流行的解析器是 lxml，可以通过 pip 安装：

```bash
pip install lxml
```

可以通过更改提供的解析器字符串来使用 lxml 与 BeautifulSoup：

```python
bs = BeautifulSoup(html.read(), 'lxml')
```

lxml 相对于 html.parser 的优势在于它通常更擅长解析“凌乱”或格式错误的 HTML 代码。它非常宽容，能够修复诸如未闭合标签、标签嵌套错误以及缺失的头部或主体标签等问题。

lxml 也比 html.parser 快一些，尽管在网络抓取中速度并不一定是优势，因为网络本身的速度几乎总是最大的瓶颈。

**避免对网页抓取代码进行过度优化**

优雅的算法很美，但在网页抓取中可能没有实际影响。几微秒的处理时间很可能会被网络请求所需的几秒钟（有时是实际的几秒钟）网络延迟所掩盖。

良好的网页抓取代码通常关注于健壮且易读的实现，而不是巧妙的处理优化。

lxml 的一个缺点是它需要单独安装，并依赖第三方 C 库来运行。与 html.parser 相比，这可能会对可移植性和易用性造成问题。

另一个流行的 HTML 解析器是 html5lib。与 lxml 类似，html5lib 是一个极其宽容的解析器，能更积极地修复损坏的 HTML。它也依赖于外部依赖项，并且比 lxml 和 html.parser 都慢。然而，如果你在处理凌乱或手写的 HTML 站点，它可能是一个不错的选择。

可以通过安装 html5lib 并将字符串 html5lib 传递给 BeautifulSoup 对象来使用：

```python
bs = BeautifulSoup(html.read(), 'html5lib')
```

希望这个 BeautifulSoup 的简短介绍能让你了解这个库的强大和简单。只要有一个识别标签包围或靠近它，几乎可以从任何 HTML（或 XML）文件中提取任何信息。第五章将更深入地探讨更复杂的 BeautifulSoup 函数调用，并介绍正则表达式及其在 BeautifulSoup 中的使用，以便从网站中提取信息。

### 可靠连接与异常处理

网络是混乱的。数据格式不好，网站会崩溃，关闭标签会丢失。在进行网页抓取时，最令人沮丧的经历之一是，你运行抓取器时去睡觉，梦想着第二天你的数据库中会有大量的数据——结果却发现抓取器在遇到意外的数据格式后出错，并在你停止查看屏幕后不久停止执行。

在这种情况下，你可能会忍不住诅咒创建网站的开发人员（以及格式奇怪的数据），但实际上你应该责怪的是自己，因为你没有提前预料到异常情况！

让我们来看一下抓取器的第一行代码，在 import 语句之后，并思考如何处理可能抛出的任何异常：

```python
html = urlopen('http://www.pythonscraping.com/pages/page1.html')
```

在这行代码中有两件主要的事情可能出错：

* 服务器上的页面找不到（或在检索时出现错误）。
* 根本找不到服务器。

在第一种情况下，将返回 HTTP 错误。这个 HTTP 错误可能是“404 页面未找到”，“500 内部服务器错误”等。在所有这些情况下，urlopen 函数将抛出通用异常 HTTPError。你可以这样处理这个异常：

```
from urllib.request import urlopen
from urllib.error import HTTPError

try:
    html = urlopen('http://www.pythonscraping.com/pages/page1.html')
except HTTPError as e:
    print(e)
    # return null, break, or do some other "Plan B"
else:
    # program continues. Note: If you return or break in the
    # exception catch, you do not need to use the "else" statement
```

如果返回 HTTP 错误代码，程序现在会打印错误并在 else 语句下不执行其余的程序。

如果根本找不到服务器（例如，http://www.pythonscraping.com 崩溃，或 URL 输入错误），urlopen 将抛出一个 URLError。这表示根本无法访问服务器，并且由于远程服务器负责返回 HTTP 状态码，因此无法抛出 HTTPError，必须捕获更严重的 URLError。你可以添加一个检查来查看是否是这种情况：

```python
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError

try:
    html = urlopen("https://pythonscrapingthisurldoesnotexist.com")
except HTTPError as e:
    print("The server returned an HTTP error")
except URLError as e:
    print("The server could not be found!")
else:
    print(html.read())
```

当然，如果页面从服务器成功检索到，页面上的内容仍然可能不是你期望的。每次你访问 BeautifulSoup 对象中的一个标签时，最好添加一个检查以确保标签实际存在。如果你尝试访问不存在的标签，BeautifulSoup 将返回一个 None 对象。问题是，尝试在 None 对象上访问一个标签本身将导致抛出 AttributeError。

以下代码行（其中 nonExistentTag 是一个虚构的标签，而不是实际的 BeautifulSoup 函数名称）：

```python
print(bs.nonExistentTag)
```

返回一个 None 对象。这个对象是完全合理的，可以处理和检查的问题。如果你不检查它，而是继续尝试在 None 对象上调用另一个函数，则会出现问题，如下所示：

```python
print(bs.nonExistentTag.someTag)
```

这将返回一个异常：

```
AttributeError: 'NoneType' object has no attribute 'someTag'
```

那么如何防止这两种情况发生呢？最简单的方法是显式检查这两种情况：

```python
try:
    badContent = bs.nonExistingTag.anotherTag
except AttributeError as e:
    print('Tag was not found')
else:
    if badContent == None:
        print ('Tag was not found')
```


对每个错误进行检查和处理一开始看起来很费力，但通过对代码进行一些重新组织，可以使其编写起来不那么困难（更重要的是，更容易阅读）。例如，这段代码是我们用稍微不同的方式编写的相同抓取器：

```python
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup


def getTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    try:
        bsObj = BeautifulSoup(html.read(), "lxml")
        title = bsObj.body.h1
    except AttributeError as e:
        return None
    return title


title = getTitle("http://www.pythonscraping.com/pages/page1.html")
if title == None:
    print("Title could not be found")
else:
    print(title)
```    
    
在这个例子中，你创建了一个 getTitle 函数，它返回页面的标题，或者在检索时出现问题时返回一个 None 对象。在 getTitle 中，你检查 HTTPError，如前例所示，并将两个 BeautifulSoup 行封装在一个 try 语句中。如果服务器不存在，html 将是一个 None 对象，html.read() 将抛出一个 AttributeError。事实上，你可以在一个 try 语句中封装任意多行代码，或完全调用另一个函数，这在任何时候都可能抛出 AttributeError。

编写抓取器时，重要的是考虑代码的整体模式，以便同时处理异常并使其可读。你还可能需要大量重复使用代码。拥有诸如 getSiteHTML 和 getTitle 之类的通用函数（包括彻底的异常处理）使得快速且可靠地进行网页抓取变得容易。







