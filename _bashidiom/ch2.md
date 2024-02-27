---
layout:     post
title:      "第二章：Looping Lingo"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - bash 
---



 <!--more-->

不仅仅是C风格的for循环—bash包括其他语法和风格；其中一些对Python程序员更为熟悉，但每种风格都有其用途。有一个没有明显参数的for循环，对脚本和函数内部都很有用。还有一种类似迭代器的for循环，具有显式值和可以来自其他命令的值。

## 循环结构

循环结构在编程语言中很常见。自C语言问世以来，许多编程语言都采用了C风格的for循环。这是一个强大而可读的结构，因为它将初始化代码、终止条件和迭代代码都组合到一个地方。例如，在C（或Java等语言）中：

```c
/* 非bash */
for (i=0; i<10; i++) {
    printf("%d\n", i);
}
```
通过一些细微的语法差异，bash遵循了大致相同的方法：

```bash
for ((i=0; i<10; i++)); do
    printf '%d\n' "$i"
done
```

特别要注意双括号的使用。与大括号不同，bash使用do和done来封闭循环的语句。与C/C++一样，for循环的惯用用法是空的for循环，创建一个故意的无限循环（您还会看到while true; do）：

```bash
for ((;;)); do
    printf 'forever'
done
```

但bash中的for循环并不只有这一种。在shell脚本中，这是一个常见的习惯用法：

```bash
for value; do
    echo "$value"
    # 使用 $value 做更多的事情...
done
```

这看起来好像有些东西丢失了，是吗？value 从哪里获取它的值？在命令行上这对你来说没有任何作用，但如果你在shell脚本中使用它，那么for循环将迭代脚本的参数。也就是说，它将使用 \\$1，然后是 \\$2，然后是 \\$3等等，作为value的值。

将该for循环放入名为myloop.sh的文件中，然后您可以像这样运行它并看到三个参数（“-c”、“17”和“core”）被打印出来：

```bash
$ bash myloop.sh -c 17 core
-c
17
core
$
```

这种简化的for循环在函数定义中也经常出现：

```bash
function Listem {
    for arg; do
        echo "arg to func: '$arg'"
    done
    echo "Inside func: \$0 is still: '$0'"
}
```
在函数定义内部，\\$1、\\$2等是函数的参数，而不是封闭shell脚本的参数。因此，在函数定义内部，for循环将迭代传递给函数的参数。

这种极简的for循环迭代一个隐含的值列表——这个列表要么是传递给脚本的参数，要么是传递给函数的参数。当在脚本的主体中使用时，它将迭代传递给脚本的参数；当在shell函数内部使用时，它将迭代传递给该函数的参数。

这绝对是bash中晦涩的习惯用语之一。您需要知道如何阅读它，但我们将在后面的部分回顾如何编写它（提示：就像Python所说的，显式胜过隐式）。

我们可能希望有一个同样简单的循环，但带有我们自己选择的显式值，而不仅限于参数，bash正好有这个东西。


## 显式值

在bash中，for循环可以给出一个要循环的值列表，如下所示：

```bash
for num in 1 2 3 4 5; do
    echo "$num"
```done

由于bash处理字符串，我们不仅仅局限于数字：

```bash
for person in Sue Neil Pat Harry; do
    echo $person
done
```

当然，值列表可以包括变量以及字面值：

```bash
for person in $ME $3 Pat ${RA[2]} Sue; do
    echo $person
done
```

for循环的另一个值来源可以来自其他命令，无论是单个命令还是命令管道：

```bash
for arg in $(some cmd or other | sort -u)
```
此类命令的示例有：

```bash
for arg in $(cat /some/file)
for arg in $(< /some/file) # 比调用cat更快
for pic in $(find . -name '*.jpg')
for val in $(find . -type d | LC_ALL=C sort)
```

一个常见的用法，尤其是在较旧的脚本中，类似于：

```bash
for i in $(seq 1 10)
```

因为seq命令将生成一系列数字。这种情况可以视为等效于：

```bash
for ((i = 1; i <= 10; i++))
```
后一种for循环更有效率，而且可能更可读。请注意，但是，循环终止后，i的值在这两种形式之间会有所不同[10与11]，尽管通常在循环外部不使用该值。

还有这个变体，但由于花括号扩展是在v3.0中引入的，而扩展的数字值的零填充是在v4.0中引入的，因此它在bash版本可移植性方面存在问题：

```bash
for i in {01..10}; do echo "$i"; done
```

**前导零**

在{start..end..inc}花括号扩展中，如果前两个术语之一以零开始，它将强制每个输出值具有相同的宽度—在bash v4.0或更新版本中使用零在左侧填充它们。因此，{098..100}将导致：098 099 100，而{98..0100}将填充到四个字符，导致：0098 0099 0100。

当您希望生成的数字成为更大字符串的一部分时，此花括号扩展结构可能特别有用。您只需将花括号结构放在字符串的一部分即可。例如，如果要生成类似于log01.txt到log05.txt的五个文件名，可以编写：

```bash
for filename in log{01..5}.txt ; do
    # 在这里处理文件名
    echo $filename
done
```

**花括号与printf -v**

您还可以使用数值for循环，然后使用printf -v构建文件名，但花括号扩展似乎更简单。当您需要数字值用于除文件名之外的其他某事时，使用数值for循环和printf。

seq命令仍然对生成一系列浮点样式的数字很有用。您可以在起始和结束值之间指定增量：

```bash
for value in $(seq 2.1 0.3 3.2); do
    echo $value
done
```
将产生：

```bash
2.1
2.4
2.7
3.0
```
只需记住bash不进行浮点运算。您可能希望生成这些值以在脚本中传递给其他程序。

## 与Python对比

在bash的for循环中，常见的短语是：

```bash
for person in ${name_list[@]}; do
    echo $person
done
```

可能会产生如下输出：

```
Arthur
Ann
Henry
John
```

看到这个例子，您可能会认为这个bash for循环与Python类似，可以迭代由迭代器对象返回的值。嗯，在这个例子中，bash确实正在迭代一系列的值，但这些值并不来自迭代器对象。相反，这些名称在循环开始之前就被明确列出。

${name_list[@]} 这个构造是bash数组的语法，此后将其称为列表（请参阅第7章引言中的术语讨论）。bash在准备运行命令时进行替换。因此，for循环并不看到列表语法；替换首先发生。for循环获得的内容看起来就像我们明确键入了这些值：

```bash
for person in Arthur Ann Henry John
```

那么字典呢？Python称之为“字典”的东西，bash称之为“关联数组”，其他人称之为“键/值对”或“散列”（同样，请参阅第7章引言）。${hash[@]} 这个构造对于键/值对的值是有效的。要循环遍历散列的键（即索引），请添加一个感叹号。${!hash[@]} 这个构造可以用于此代码片段所示：

```bash
# 我们想要一个散列（即键/值对）
declare -A hash
# 读取我们的数据
while read key value; do
    hash[$key]="$value"
done
# 展示我们得到了什么，尽管它们可能不会
# 以与读取相同的顺序显示
for key in "${!hash[@]}"; do
    echo "key $key ==> value ${hash[$key]}"
done
```
这里是另一个例子：

```bash
# 我们想要一个散列（即键/值对）
declare -A hash
# 读取我们的数据：单词和出现的次数
while read word count; do
    let hash[$word]+="$count"
done
# 展示我们得到了什么，尽管顺序
# 基于散列，即我们无法控制它
for key in "${!hash[@]}";do
    echo "word $key count = ${hash[$key]}"
done
```

这一章更多地涉及循环结构，如for，但如果您想要关于列表和散列的更多详细信息和示例，请参阅第7章。




## 引号和空格

关于这个for循环，还有一个重要的方面需要考虑。你是否注意到我们在前面的例子中不一致地使用了引号？如果列表中的值包含空格（例如，如果每个条目都有名和姓），那么我们的例子for循环：

```bash
for person in ${namelist[@]}; do
    echo $person
done
```
可能会产生如下输出：

```bash
Art
Smith
Ann
Arundel
Hank
Till
John
Jakes
```
为什么？如何做到的？答案在于bash对${namelist[@]}进行的替换。它只是将这些名称放在变量表达式的位置。这样就在列表中留下了八个单词，就像这样：

```bash
for person in Art Smith Ann Arundel Hank Till John Jakes
```
for循环只是得到了一个单词列表。它不知道它们来自哪里。有bash语法来解决这个问题：在列表表达式周围加上引号，每个值都将被引用。

```bash
for person in "${namelist[@]}"
```
将被转换为：

```bash
for person in "Art Smith" "Ann Arundel" "Hank Till" "John Jakes"
```
这将产生期望的结果：

```
Art Smith
Ann Arundel
Hank Till
John Jakes
```

如果您的for循环要迭代文件名列表，那么您应该确保使用引号，因为文件名中可能包含空格。

对于所有这些，还有一个小技巧。列表语法可以使用 * 或 @ 列出列表的所有元素：${namelist[*]} 同样有效……除非放在引号内。表达式：

```bash
"${namelist[*]}"
```

将被评估为一个包含所有值的单个字符串。在这个例子中：

```bash
for person in "${namelist[*]}"; do
    echo $person
done
```
将产生一行输出，像这样：

```bash
Art Smith Ann Arundel Hank Till John Jakes
```

虽然在某些情境中单个字符串可能有用，但在for循环中这特别没有意义，因为只会有一次迭代。我们建议除非您确切地知道需要 *，否则使用 @。

另请参阅第129页的“引号”部分。


## 开发和测试for循环

事实证明，“for list do something”循环在各种情况下都非常有用。让我们看两个简单的例子：在服务器列表上运行SSH命令和重命名文件，比如for file in *.JPEG; do mv -v $file ${file/JPEG/jpg}; done。但是如何开发和测试脚本，甚至是简单的for命令呢？与开发其他内容的方式相同：从简单开始，一步一步进行。但特别要使用echo（参见示例2-1）。请注意，bash内建的echo具有许多有趣的选项，但不符合POSIX标准（参见第56页的“POSIX输出”）。最常用且最有趣的选项是-e（启用反斜杠转义的解释）和-n（抑制自动换行）。
示例2-1。文件重命名——测试版本

```bash
### 构建和测试重命名命令，注意echo
for file in *.JPEG; do echo mv -v $file ${file/JPEG/jpg}; done
### 简单的多节点SSH，注意第一个echo（可以在一行上执行，但为了书本而分开）
for node in web-server{00..09}; do
    echo ssh $node 'echo -e "$HOSTNAME\t$(date "+%F") $(uptime)"';
done
```

一旦它按预期工作，删除那个前导的echo并运行。当然，如果你在块中使用了重定向，你必须小心处理，可能将\|更改为.p.，>更改为.gt.等等，直到每个阶段都正常工作。

**在多个主机上执行相同的命令**
这远超出了本书的范围，但如果您需要在许多主机上运行相同的命令，您可能应该使用Ansible、Chef、Puppet或类似的工具。有时您可能有一个非常简单和不太正式的需求，这些工具中的一个可能会有所帮助：

* clusterssh：用Perl编写，它在窗口中打开一堆不受管理的终端。
* mssh（MultiSSH）：基于GTK+的多SSH客户端，位于单个GUI窗口中。
* mussh：多主机SSH包装脚本。
* pconsole：用于平铺窗口管理器，为每个主机生成一个终端。
* multixterm：用Expect和Tk编写，驱动多个xterms。
* PAC Manager：Linux上类似SecureCRT的Perl GUI。

## while和until
我们之前简要提到过while，并且它的工作方式符合您的期望——“在条件退出状态为零时执行块”：

```bash
while <CRITERIA>; do <BLOCK>; done
```
它经常用于读取文件；请参见第9章中的几个示例。对于参数解析，请参见第75页的“解析选项”。
与其他语言不同，在bash中，until只是! while，或者说“在条件退出状态不为零时执行块”：

```bash
until <CRITERIA>; do <BLOCK>; done
```
相同：
```bash
! while <CRITERIA>; do <BLOCK>; done
```

这对于等待节点被创建或重新启动之类的操作非常方便（示例2-2）。
示例2-2。等待重新启动

```bash
until ssh user@10.10.10.10; do sleep 3; done
```

## 风格和可读性：总结
在本章中，我们首先快速查看了C/C++风格的数值for循环。然后我们进一步深入。Bash非常以字符串为导向，并且有一些其他值得知道的for循环风格。它的极简循环for variable提供了对脚本或函数的参数的隐式（可以说是晦涩的）迭代。对于提供给for循环的明确的值列表，无论是字符串还是其他类型，都为我们提供了在列表的所有元素或在哈希的所有键上进行迭代的理想机制。

我们现在知道\\${namelist[@]}和\\${namelist[ * ]}都显示列表的所有值，但如果它们被包含在双引号中，结果将是不同的：单独的字符串与一个大字符串。对于特殊的shell变量\\$@和\\$*也是如此。它们都表示对脚本的所有参数的列表（即\\$1，\\$2等）。但是，当它们被包含在双引号中时，它们也会导致多个字符串或一个字符串。为什么现在提到这个呢？只是为了回到我们最简单的for循环：

```bash
for param
```

并说这等同于：

```bash
for param in "$@"
```
我们认为第二种形式更好，因为它更明确地显示了正在迭代的值。但是，也有反对意见，即\\$@变量名本身以及引号的必要性都是专业知识，对于天真的读者来说并不比第一种简单形式更明显。如果您真的喜欢第一种形式，只需添加一条注释：

```bash
for param
 # Iterate over all the script arguments
```

当循环遍历一系列整数值时，带有双括号的C风格for循环可能是最可读且最有效的。 （如果效率是一个重要问题，请确保在脚本的早期使用declare -i i，将变量“i”显式为整数，以避免字符串的转换。）

知道您已经随时可以使用所有这些值，您可能会对它们做些什么？在循环内部发生了什么，利用这些值？对于遇到的值必须做出决策，而决策会引导我们进入bash的另一个重要特性：其超级强大和灵活的case语句，这是下一章的主题。

for命令非常有用，但开发起来可能有些棘手。从简单开始，使用echo直到确保命令按预期工作为止。并记住“语法糖”while和until命令，以提高可读性。

















