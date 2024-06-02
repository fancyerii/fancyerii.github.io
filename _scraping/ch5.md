---
layout:     post
title:      "第五章：Advanced HTML Parsing"
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

当被问到他是如何雕刻出像大卫这样精妙的艺术作品时，米开朗基罗据说曾著名地回答道：“很简单，你只需将那些看起来不像大卫的石头剔除掉就行了。” 尽管网页抓取在大多数其他方面与大理石雕刻不同，但在从复杂的网页中提取所需信息时，你必须采取类似的态度。在本章中，我们将探索各种技术来剔除那些看起来不像你想要的内容，直到你得到所需的信息。复杂的HTML页面起初可能看起来令人生畏，但只需不断地剔除！

## 再来一碗BeautifulSoup

在第四章中，你快速了解了如何安装和运行BeautifulSoup，以及如何一次选择一个对象。在本节中，我们将讨论如何按属性搜索标签、处理标签列表以及遍历解析树。

几乎你遇到的每个网站都包含样式表。样式表的创建是为了让网页浏览器能够将HTML呈现为色彩丰富且美观的设计，供人类欣赏。你可能认为这一样式层对于网页抓取器来说完全可以忽略——但慢着！CSS实际上对网页抓取器大有帮助，因为它要求区分HTML元素以便于不同样式的设计。

CSS为网页开发者提供了一个动机，使他们添加本来可能会留有相同标记的HTML元素的标签。一些标签可能看起来像这样：

```html
<span class="green"></span>
```

其他标签看起来像这样：

```html
<span class="red"></span>
```

网页抓取器可以轻松地基于它们的类将这两个标签分开。例如，可以使用BeautifulSoup抓取所有红色文本而不是绿色文本。由于CSS依赖这些标识属性来正确地样式化网站，你几乎可以保证在大多数现代网站上会有大量的类和id属性。

让我们创建一个示例网页抓取器，抓取位于http://www.pythonscraping.com/pages/warandpeace.html的页面。

在这个页面上，故事中人物说的话是红色的，而人物的名字是绿色的。你可以在以下页面源码示例中看到引用相应CSS类的span标签：

```html
<span class="red">Heavens! what a virulent attack!</span> replied
<span class="green">the prince</span>, not in the least disconcerted by this reception.
```

你可以通过使用类似于第四章中使用的程序抓取整个页面并创建一个BeautifulSoup对象：

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen('http://www.pythonscraping.com/pages/warandpeace.html')
bs = BeautifulSoup(html.read(), 'html.parser')
```

使用这个BeautifulSoup对象，你可以使用find_all函数通过选择仅在<span class="green"></span>标签内的文本来提取一个包含专有名词的Python列表（find_all是一个非常灵活的函数，你将在本书的后面部分经常使用）：

```python
nameList = bs.findAll('span', {'class': 'green'})
for name in nameList:
    print(name.get_text())
```

运行时，它应该按照《战争与和平》中出现的顺序列出所有专有名词。它是如何工作的？之前，你调用bs.tagName来获取页面上该标签的第一个实例。现在，你调用bs.find_all(tagName, tagAttributes)来获取页面上所有该标签的列表，而不仅仅是第一个。

在获取名字列表后，程序遍历列表中的所有名字，并打印name.get_text()以便将内容与标签分开。

**何时使用get_text()以及何时保留标签**

.get_text()会从你正在处理的文档中剥离所有标签，并返回一个只包含文本的Unicode字符串。例如，如果你正在处理包含许多超链接、段落和其他标签的大段文本，这些都会被剥离掉，你将得到一个没有标签的文本块。

请记住，在BeautifulSoup对象中寻找你所需内容比在文本块中要容易得多。调用.get_text()应该总是你在打印、存储或操作最终数据之前的最后一步。通常，你应尽量长时间保留文档的标签结构。



### 使用BeautifulSoup的find()和find_all()

BeautifulSoup的find()和find_all()是你最有可能使用的两个函数。通过它们，你可以轻松地根据各种属性筛选HTML页面，找到所需标签的列表或单个标签。

这两个函数非常相似，从BeautifulSoup文档中的定义可以看出：

```
find_all(tag, attrs, recursive, text, limit, **kwargs)
find(tag, attrs, recursive, text, **kwargs)
```

很可能，你在95%的时间里只需要使用前两个参数：tag和attrs。然而，让我们详细看看所有参数。

**tag参数** 是你之前见过的；你可以传递一个标签的字符串名称，甚至是一个字符串标签名称的Python列表。例如，以下代码返回文档中所有标题标签的列表：

```python
.find_all(['h1','h2','h3','h4','h5','h6'])
```

与tag参数不同，它可以是字符串或可迭代对象，而**attrs参数**必须是一个Python字典，包含属性和值。它匹配包含这些属性中任何一个的标签。例如，以下函数将在HTML文档中返回绿色和红色的span标签：

```python
.find_all('span', {'class': ['green', 'red']})
```

**recursive参数** 是一个布尔值。你想要在文档中深入到什么程度？如果recursive设置为True，find_all函数会在子元素、子元素的子元素等中查找匹配参数的标签。如果为False，它只会查找文档中的顶级标签。默认情况下，find_all递归工作（recursive设置为True）。通常，除非你非常清楚需要做什么且性能是一个问题，否则最好保持默认设置。

**text参数** 不同寻常，因为它基于标签的文本内容进行匹配，而不是标签本身的属性。例如，如果你想找到示例页面中“the prince”被标签包围的次数，可以用以下代码替换之前示例中的.find_all()函数：

```python
nameList = bs.find_all(text='the prince')
print(len(nameList))
```

输出为7。

**limit参数** 当然，仅在find_all方法中使用；find等同于将find_all的limit设置为1的调用。如果你只对从页面中检索前x个项目感兴趣，可以设置此参数。注意，这会按照它们在文档中出现的顺序给你页面上的前几个项目，而不一定是你想要的第一个。

**kwargs参数** 允许你向方法传递任何其他命名参数。find或find_all未识别的任何额外参数将用作标签属性匹配器。例如：

```python
title = bs.find_all(id='title', class_='text')
```

这将返回第一个在class属性中包含“text”并且id属性为“title”的标签。请注意，根据惯例，id的每个值应在页面上只使用一次。因此，实际上，这样的一行代码可能并不特别有用，应等同于使用find函数：

```python
title = bs.find(id='title')
```


**关键词参数和Class**

class是Python中的保留字，不能用作变量或参数名称。例如，如果你尝试以下调用，由于class的非标准用法，你会得到语法错误：

```python
bs.find_all(class='green')
```

因此，BeautifulSoup要求你使用关键字参数_class而不是class。

你可能已经注意到，BeautifulSoup已经有一种基于属性和值查找标签的方法：attr参数。实际上，以下两行是相同的：

```python
bs.find(id='text')
bs.find(attrs={'id':'text'})
```

然而，第一行的语法更短，对于快速过滤需要特定属性的标签时，操作起来可能更容易。当过滤器变得更复杂，或者你需要在参数中传递属性值选项列表时，你可能会希望使用attrs参数：

```python
bs.find(attrs={'class':['red', 'blue', 'green']})
```


### 其他BeautifulSoup对象

到目前为止，在本书中你已经见过BeautifulSoup库中的两种对象：

**BeautifulSoup对象**
之前代码示例中作为变量bs出现的实例

**Tag对象**
通过调用BeautifulSoup对象上的find和find_all或深入查找检索到的列表中单独检索到的对象：

```python
bs.div.h1
```

然而，库中还有两个不太常用但仍然重要的对象：

**NavigableString对象**
用于表示标签内的文本，而不是标签本身（某些函数对NavigableString进行操作并生成NavigableString，而不是标签对象）。

**Comment对象**
用于查找HTML注释中的注释标签，如<!--像这样的-->。

在撰写本文时，这些是BeautifulSoup包中唯一的四种对象。这也是2004年BeautifulSoup包发布时的唯一四种对象，因此可用对象的数量在不久的将来不太可能改变。





### 树结构导航

find_all函数负责根据标签名称和属性查找标签。但如果你需要根据标签在文档中的位置来查找标签呢？这时树导航就派上用场了。在第4章中，你学习了如何在一个方向上导航BeautifulSoup树：

```python
bs.tag.subTag.anotherSubTag
```

现在让我们看看如何在HTML树中向上、横向和斜向导航。你将使用我们那个代处理的在线购物网站（http://www.pythonscraping.com/pages/page3.html）作为抓取的示例页面，如图5-1所示。


<a>![](/img/scraping/ch5/1.png)</a>
*图5.1 http://www.pythonscraping.com/pages/page3.html的截屏*

这个页面的HTML结构，被映射成树（为了简洁省略了一些标签），看起来是这样的：

```
HTML
— body
    — div.wrapper
        — h1
        — div.content
        — table#giftList
            — tr
                — th
                — th
                — th
                — th
            — tr.gift#gift1
                — td
                — td
                — span.excitingNote
                — td
                — td
                    — img
            — ...table rows continue...
        — div.footer
```


你将在接下来的几节中使用这个相同的HTML结构作为示例。

#### 处理子元素和其他后代元素

在计算机科学和某些数学分支中，你经常听到对孩子们做的可怕事情：移动他们、存储他们、移除他们，甚至杀死他们。幸运的是，本节只关注选择他们！

在BeautifulSoup库以及许多其他库中，孩子和后代之间有一个区分：就像在一个人类家谱中一样，孩子总是紧挨着父标签的下一级标签，而后代可以是父标签下面任意层级的标签。例如，tr标签是table标签的子元素，而tr、th、td、img和span都是table标签的后代（至少在我们的示例页面中）。所有的子元素都是后代，但不是所有的后代都是子元素。

一般来说，BeautifulSoup函数总是处理当前选定标签的后代。例如，bs.body.h1选择的是body标签的第一个h1后代标签。它不会查找位于body标签外的标签。

同样，bs.div.find_all('img')会找到文档中的第一个div标签，然后检索一个包含所有img标签（这些img标签是该div标签的后代）的列表。

如果你只想找到子元素，可以使用.children标签：

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/page3.html')
bs = BeautifulSoup(html, 'html.parser')

for child in bs.find('table',{'id':'giftList'}).children:
    print(child)
```


这段代码会打印出 giftList 表中的商品行列表，包括初始的列标签行。如果你用 descendants() 函数代替 children() 函数来编写这段代码，在表格中会找到并打印出大约二十多个标签，包括 img 标签、span 标签和单独的 td 标签。区分子元素和后代元素确实非常重要！

#### 处理兄弟元素

BeautifulSoup 的 next_siblings() 函数让从表格中收集数据变得非常简单，尤其是那些有标题行的表格：

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/page3.html')
bs = BeautifulSoup(html, 'html.parser')

for sibling in bs.find('table', {'id':'giftList'}).tr.next_siblings:
    print(sibling) 
```

这段代码的输出是打印出产品表中的所有产品行，除了第一个标题行。为什么标题行会被跳过呢？对象不能和它们自己成为兄弟元素。每当你获取一个对象的兄弟元素时，对象本身不会包含在列表中。顾名思义，这个函数只调用后续的兄弟元素。如果你选择列表中间的一行，并调用 next_siblings，则只会返回后续的兄弟元素。因此，通过选择标题行并调用 next_siblings，你可以选择表格中的所有行，而不选择标题行本身。

**使选择更具体**

上述代码同样适用于选择 bs.table.tr 或甚至 bs.tr 来选择表格的第一行。然而，在代码中，我尽量用较长的形式来写出所有内容：

```python
bs.find('table', {'id': 'giftList'}).tr
```

即使看起来页面上只有一个表格（或其他目标标签），也很容易漏掉一些东西。此外，页面布局经常会发生变化。曾经是页面上第一个的标签，可能某天会变成页面上第二个或第三个同类标签。为了使抓取器更健壮，在进行标签选择时最好尽可能具体。利用标签属性（如果有的话）。

作为 next_siblings 的补充，previous_siblings 函数在你想获取一组兄弟标签列表的末尾的某个易于选择的标签时也很有帮助。

当然，还有 next_sibling 和 previous_sibling 函数，它们的功能几乎与 next_siblings 和 previous_siblings 相同，只不过返回的是单个标签，而不是它们的列表。

#### 处理父元素

在抓取页面时，你可能会发现需要查找标签的父元素的情况比查找它们的子元素或兄弟元素的情况要少。通常，当你查看 HTML 页面并希望爬取它们时，你从查看标签的顶层开始，然后弄清楚如何向下钻取到你想要的确切数据。然而，有时你会遇到一些需要 BeautifulSoup 的父元素查找函数 .parent 和 .parents 的特殊情况。例如：

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/page3.html')
bs = BeautifulSoup(html, 'html.parser')
print(bs.find('img',
              {'src':'../img/gifts/img1.jpg'})
      .parent.previous_sibling.get_text())
```

这段代码将打印出位于 ../img/gifts/img1.jpg 图片位置所代表的对象的价格（在这个例子中，价格是 \\$15.00）。

这是如何工作的呢？以下图示代表了你正在处理的 HTML 页面部分的树结构，并标示了步骤：

```
 <tr>
    — td
    — td
    — td ③
        — "$15.00" ④
    — td ②
        — <img src="../img/gifts/img1.jpg"> ①
```

* ① 首先选择 src="../img/gifts/img1.jpg" 的图片标签。
* ② 选择该标签的父标签（在这个例子中是 td 标签）。
* ③ 选择 td 标签的 previous_sibling（在这个例子中是包含产品价格的 td 标签）。
* ④ 选择该标签内的文本 "\\$15.00"。


## 正则表达式

正如老计算机科学笑话所说：“假设你有一个问题，然后你决定用正则表达式来解决它。那么现在你有两个问题了。”

不幸的是，正则表达式（通常缩写为 regex）通常通过一大堆随机符号的表格来教授，这些符号串在一起看起来像一堆无意义的东西。这往往让人望而却步，后来他们进入职场后，为了避免使用正则表达式，会编写不必要复杂的搜索和过滤函数。

在网页抓取时，正则表达式是一个非常宝贵的工具。幸运的是，正则表达式并不难快速上手，你可以通过看一些简单的例子并进行试验来学习它们。

正则表达式之所以被称为正则，是因为它们用于识别属于正则语言的字符串。这里的“语言”并不是指编程语言或自然语言（如英语或法语），而是数学意义上的“遵循某些规则的字符串集合”。

正则语言是由一组线性规则生成的一组字符串，可以简单地沿着候选字符串移动并将其与规则进行匹配。例如：

* 一个或多个字母a(至少一个)。
* 在后面精确地附加五个字母 b。
* 再附加偶数个字母 c。
* 在结尾写一个字母 d 或 e。

正则表达式可以明确地确定：“是的，你给我的这个字符串符合规则，”或“这个字符串不符合规则。”这在快速扫描大型文档以查找看起来像电话号码或电子邮件地址的字符串时非常方便。

符合上述规则的字符串有 aaaabbbbbccccd、aabbbbbcce 等。从数学上讲，符合这个模式的字符串有无限多个。

正则表达式只是表示这些规则的一种简写方式。例如，以下是刚才描述的一系列步骤的正则表达式：

```
aa*bbbbb(cc)*(d|e)
```

这个字符串乍一看可能有点令人望而生畏，但当你将其分解为各个组件时会变得更清晰：

```
aa*
```

写字母 a，后面跟一个星号（*），表示“任意数量的 a，包括 0 个。”通过这种方式，你可以保证至少写一个字母 a。

```
bbbbb
```

这里没有什么特别之处——只是连续的五个 b。

```
(cc)*
```

任何偶数个的东西都可以成对地组合，所以要强制执行关于偶数的规则，你可以写两个 c，用括号将它们括起来，并在后面写一个星号，这意味着你可以有任意数量的 c 对（注意这也可以意味着零对 c）。

```
(d|e)
```

在两个表达式中间添加一个竖线（|），表示它可以是“这个东西或那个东西。”在这种情况下，你是在说“添加一个 d 或一个 e。”通过这种方式，你可以保证有且只有一个这两个字符中的一个。

**实验正则表达式**

在学习如何编写正则表达式时，关键是要不断尝试，体会它们的工作方式。如果你不想打开代码编辑器，写几行代码并运行程序以查看正则表达式是否按预期工作，你可以访问像 [RegEx Pal](http://regexpal.com/) 这样的网站，实时测试你的正则表达式。

表 5-1 列出了常用的正则表达式符号，附有简要说明和示例。这个列表并不完整，正如之前提到的，你可能会遇到因语言不同而略有差异的符号。然而，这 12 个符号是 Python 中最常用的正则表达式，可以用来查找和收集几乎任何类型的字符串。

正则表达式的一个经典例子是用于识别电子邮件地址。虽然电子邮件地址的具体规则因邮件服务器而略有不同，但我们可以创建一些通用规则。每条规则对应的正则表达式如下表所示：
通过将所有规则连接起来，你可以得到以下正则表达式：

```
[A-Za-z0-9._+]+@[A-Za-z]+.(com|org|edu|net)
```

在尝试从头编写任何正则表达式时，最好首先列出步骤，具体说明你的目标字符串的样子。注意边界情况。例如，如果你正在识别电话号码，你是否考虑了国家代码和分机号码？

**正则表达式：并非总是“正则”！**

标准版本的正则表达式（本书中涵盖的内容，Python 和 BeautifulSoup 使用的正则表达式语法）基于 Perl 的语法。大多数现代编程语言使用这一版本或类似版本的语法。然而，请注意，如果你在其他语言中使用正则表达式，可能会遇到问题。即使是一些现代语言，如 Java，在处理正则表达式时也有细微差别。如有疑问，请查阅文档！

## 正则表达式与BeautifulSoup

如果之前关于正则表达式的部分似乎与本书的主题有些脱节，这里就是将它们联系在一起的地方。BeautifulSoup和正则表达式在网页抓取时密不可分。事实上，大多数接受字符串参数的函数（例如，find(id="aTagIdHere")）同样也可以接受正则表达式作为参数。

让我们看一些示例，抓取位于 http://www.python-scraping.com/pages/page3.html 的页面。

注意该网站有许多产品图片，其形式如下：

```html
<img src="../img/gifts/img3.jpg">
```

如果你想获取所有产品图片的URL，一开始可能看起来相当简单：只需使用 .find_all("img") 抓取所有图片标签，对吧？但这里有一个问题。除了显而易见的“额外”图片（例如，徽标），现代网站通常还有隐藏图片、用于间距和对齐元素的空白图片，以及其他你可能未意识到的随机图片标签。当然，你不能指望页面上的唯一图片都是产品图片。

假设页面布局可能会改变，或者出于某种原因，你不想依赖图片在页面中的位置来找到正确的标签。当你试图抓取散布在网站各处的特定元素或数据时，可能会遇到这种情况。例如，特色产品图片可能会出现在某些页面顶部的特殊布局中，而在其他页面中则不会。

解决方案是寻找标签本身的某些识别特征。在这种情况下，你可以查看产品图片的文件路径：

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

html = urlopen('http://www.pythonscraping.com/pages/page3.html')
bs = BeautifulSoup(html, 'html.parser')
images = bs.find_all('img', {'src':re.compile('\.\.\/img\/gifts/img.*\.jpg')})
for image in images: 
    print(image['src'])
```

这将只打印相对路径以 ../img/gifts/img 开头并以 .jpg 结尾的图片路径，其输出为：

```
../img/gifts/img1.jpg
../img/gifts/img2.jpg
../img/gifts/img3.jpg
../img/gifts/img4.jpg
../img/gifts/img6.jpg
```

正则表达式可以作为任何参数插入到BeautifulSoup表达式中，使你在查找目标元素时拥有极大的灵活性。


## 访问属性

到目前为止，你已经学会了如何访问和过滤标签以及访问它们内部的内容。然而，在网页抓取中，通常你不是在寻找标签的内容，而是在寻找它们的属性。这在像 a 这样的标签中尤其有用，其中它指向的URL包含在 href 属性中；或者像 img 标签中，目标图像包含在 src 属性中。

使用标签对象，可以通过调用以下方法自动访问属性的 Python 列表：

```python
myTag.attrs
```

请记住，这实际上返回一个Python字典对象，使得这些属性的检索和操作变得简单。例如，可以使用以下代码找到图像的源位置：

```python
myImgTag.attrs['src']
```

## Lambda 表达式

Lambda 是一个高级学术术语，在编程中简单地指的是“一种简写函数的方式”。在Python中，我们可以编写一个返回数字平方的函数：

```python
def square(n):
    return n**2
```

我们可以使用 lambda 表达式在一行中完成同样的事情：

```python
square = lambda n: n**2
```

这将变量 square 直接赋值给一个函数，它接受一个参数 n 并返回 n\*\*2。但并没有规定函数必须是“命名的”或者完全赋值给变量。我们可以直接将它们写成值：

```python
>>> lambda r: r**2
<function <lambda> at 0x7f8f88223a60>
```

实际上，lambda 表达式是一个独立存在的函数，没有命名或分配给变量。在Python中，lambda 函数不能有多行代码（这只是Python的风格和审美问题，而不是计算机科学的基本规则）。

Lambda 表达式最常见的用法是作为参数传递给其他函数。BeautifulSoup允许将某些类型的函数作为参数传递给 find_all 函数。

唯一的限制是这些函数必须以标签对象为参数并返回一个布尔值。BeautifulSoup在这个函数中评估每个标签对象，返回评估为 True 的标签，而其余的则被丢弃。

例如，以下代码检索具有两个属性的所有标签：

```python
bs.find_all(lambda tag: len(tag.attrs) == 2)
```

在这里，作为参数传递的函数是 len(tag.attrs) == 2。当这个条件为 True 时，find_all 函数将返回该标签。也就是说，它将找到具有两个属性的标签，例如：

```html
<div class="body" id="content"></div>
<span style="color:red" class="title"></span>
```

Lambda 函数非常有用，甚至可以用它们来替换现有的BeautifulSoup函数：

```python
bs.find_all(lambda tag: tag.get_text() == 'Or maybe he\'s only resting?')
```

这也可以不使用 lambda 函数来实现：

```python
bs.find_all('', text='Or maybe he\'s only resting?')
```

然而，如果你记住 lambda 函数的语法以及如何访问标签属性，也许你将再也不需要记住任何其他BeautifulSoup语法了！因为提供的 lambda 函数可以是返回 True 或 False 值的任何函数，你甚至可以将它们与正则表达式结合起来，以找到具有匹配特定字符串模式的属性的标签。




## 有时不一定需要一把锤子

当面对一团复杂的标签时，很容易陷入其中，使用多行语句尝试提取信息。然而，请记住，在这一章节中过度使用技术可能导致代码难以调试、脆弱或两者兼而有之。让我们看看一些可以完全避免需要高级 HTML 解析的方法。

假设你有一些目标内容。也许是一个姓名、统计数据或一段文本。也许它被埋在 20 层标签深度的 HTML 混乱中，找不到有用的标签或 HTML 属性。你可能决定冒险，尝试写出类似以下行的提取代码：

```python
bs.find_all('table')[4].find_all('tr')[2].find('td').find_all('div')[1].find('a')
```

看起来并不怎么样。除了行的美感外，甚至网站管理员对网站进行的最小更改都可能完全破坏你的网络爬虫。如果网站的 web 开发者决定添加另一个表格或另一列数据呢？如果开发者在页面顶部添加另一个组件（带有几个 div 标签）呢？上述代码是不稳定的，依赖于网站结构永远不会改变。

那么你有哪些选择呢？

* 寻找任何可以用于直接跳到文档中间、更接近你实际想要的内容的“地标”。方便的 CSS 属性是一个明显的地标，但你也可以创造性地使用 .find_all(text='some tag content') 通过标签内容来抓取标签。
* 如果没有简单的方法来隔离你想要的标签或其任何父级标签，你能找到一个兄弟标签吗？使用 .parent 方法然后再向下钻取到目标标签。
* 完全放弃这个文档，并寻找“打印此页”链接，或者查看站点的移动版本，它可能具有更好格式的 HTML（有关模拟成移动设备和接收移动站点版本的更多信息，请参阅第17章）。
* 不要忽略\<script>标签中或者单独加载的 JavaScript 文件中的内容。JavaScript 往往包含你要找的数据，并且格式更友好！例如，我曾通过检查嵌入的 Google Maps 应用程序的 JavaScript，在网站上收集了格式良好的街道地址。有关这种技术的更多信息，请参阅第11章。
* 信息可能在页面的 URL 中。例如，页面标题和产品 ID 通常可以在那里找到。
* 如果你寻找的信息在某种程度上是这个网站独有的，那么你就很倒霉了。如果不是，请尝试想想其他可以获取这些信息的来源。是否有另一个网站具有相同的数据？这个网站是否展示了从另一个网站爬取或聚合的数据？

特别是当面对埋藏或格式不佳的数据时，重要的是不要随意挖掘并把自己写入可能无法摆脱的困境中。深呼吸，考虑其他解决方案。

正确使用这里介绍的技术将大大提升网络爬虫的稳定性和可靠性。




