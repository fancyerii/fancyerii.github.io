---
layout:     post
title:      "第三章：Just in CASE"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - bash 
---



 <!--more-->

许多编程语言提供了“switch”或“match”语句，这是一种n路分支，可用作一系列if/then/else子句的替代方法。在bash中有一个类似的构造：case语句。它具有强大的模式匹配功能，在脚本编写中非常有用。

## 使用Case

关键字case和in界定了您要与各种模式进行比较的值。以下是一个简单的示例：

```bash
case "$var" in
    yes ) echo "glad you agreed" ;;
    no )
        echo "sorry; good bye"
        exit
        ;;
    * ) echo "invalid answer. try again" ;;
esac
```
你可能能够理解这个例子。它检查$var中的值是否为“yes”或“no”，并执行相应的语句。它甚至有一个默认操作。case语句的结束由esac标记，它是case倒过来拼写的。这个例子相当易读，但只是简单介绍。您还会注意到，您使用了两种不同的块样式，一个是“一行式”用于yes，另一个是更典型的块（由;;闭合...稍后详细介绍）用于no。您使用哪种取决于您要做什么以及代码如何对齐以提高可读性。


**case里的(**

case语句的语法包括一个可选的“（”来匹配示例中的“）”。例如，我们可以写成("yes")而不仅仅是"yes")，其他条目同理。
不过，我们很少见到有人使用这个。毕竟，谁愿意多输入一个字符呢？
case语句的真正威力和最符合习惯的外观来自于在各种可能的值比较中使用shell的模式匹配：

```bash
case "$var" in
    [Nn][Oo]* )
        echo "Fine. Leave then."
        exit
        ;;
    [Yy]?? | [Ss]ure | [Oo][Kk]* )
        echo "OK. Glad we agree."
        ;;
    * ) echo 'Try again.'
        continue
        ;;
esac
```
这里简要回顾一下bash模式匹配，您可能熟悉它作为命令行通配符（或通配符）。有三个特殊字符需要注意：?匹配一个字符，*匹配任意数量的字符（包括零个），方括号[]匹配括号内包含的任何字符。在我们的示例中，构造[Yy]匹配大写Y或小写y。构造[Nn][Oo]*匹配大写或小写N，后跟大写或小写O，后跟任意数量的其他字符。该模式匹配以下单词（以及其他单词）：no、No、nO、NO、noway、Not Ever和nope。如果$var的值是单词never，它将不匹配。
你能猜出肯定情况的一些可能值吗？竖线分隔不同的模式，它们都将导致相同的结果。（思考“或”但不是||或的意思。）单词Yes、yes、YES、yEs、yES、yup、Sure、sure、OK、ok和OKfine以及“OK why not”都可以。但这些单词不行：ya、SURE、oook等等。
默认情况并没有特殊的语法，它只是一个模式，但这个模式将匹配任何内容。如果没有其他先前的模式产生匹配，那么我们知道这个模式会匹配任意数量的任何字符。因此，如果脚本作者希望捕捉默认情况，他们会将此模式放在列表的最后。


**不是正则表达式**
在case语句中使用的模式匹配不是正则表达式。在bash中，只有一个地方允许使用正则表达式（regex或regexp），那就是在if语句中使用=~比较运算符。如果您确实需要使用正则表达式，那么您需要使用一系列的if/then/else语句，而不是case。

## 一个实际的用例
解析命令行选项的代码是找到case语句的常见地方。让我们看一个简单但有些实际的脚本，它很好地利用了case语句。

### 动机

如果你曾经使用过Linux或Unix，你可能会经常使用ls命令列出文件名和相关信息。ls的一些选项非常方便，可以提供更多信息或以某种方式进行排序。您可能会形成某些使用ls或您最常用的选项的习惯。因此，您可能会创建一些别名，甚至是整个脚本，以使使用您喜欢的组合更容易。但是然后你最终会得到几个不同的脚本。它们在功能上都有关联，但都是独立的。我们如何将它们合并到一个脚本中呢？
考虑一个熟悉的例子，即Git这个流行的源代码控制软件。它有几个相关但不同的功能，都由一个命令名称git调用，但每个功能都由单独的第二个关键字区分，例如git clone、git add、git commit、git push等等。

### 我们的脚本

我们可以将这种“子命令”方法应用于我们的情况。让我们考虑一些我们想要使用的与ls相关的功能：按文件名长度的顺序列出文件（以及长度），仅列出最长的文件名，列出最近修改的最后几个文件，以及列出带有颜色编码指示文件类型的文件名——这是ls的一个标准特性，但需要记住一些难以记忆的选项。
我们的脚本将被命名为list，但会有一个第二个词来指定这些功能之一。该词可以是color、last、length或long。下面的示例3-1包含了一个执行此操作的脚本。

```bash
#!/usr/bin/env bash
# list.sh: A wrapper script for ls-related tools & simple `case..esac` demo
# Original Author & date: _bash Idioms_ 2022
# bash Idioms filename: examples/ch03/list.sh
#_________________________________________________________________________
VERSION='v1.2b'
function Usage_Exit {
    echo "$0 [color|last|len|long]"
    exit
}
# Show each filename preceded by the length of its name, sorted by filename
# length. Note '-' is valid but uncommon in function names, but it is not
# valid in variable names. We don't usually use it, but you can.
function Ls-Length {
    ls -1 "$@" | while read fn; do
        printf '%3d %s\n' ${#fn} ${fn}
    done | sort -n
}

(( $# < 1 )) && Usage_Exit           ①
sub=$1
shift

case $sub in
    color)
        ls -N --color=tty -T 0 "$@"
        ;;
    last | latest)                 ②
        ls -lrt | tail "-n${1:-5}" ③
        ;;
    len*)                          ④
        Ls-Length "$@"
        ;;
    long)
        Ls-Length "$@" | tail -1
        ;;
    *)
        echo "unknown command: $sub"
        Usage_Exit
        ;;
esac

```

我们不会在这里解释脚本的所有部分，尽管在本书的最后，您将了解到所有使用的功能。我们主要关注case语句：

* ① 你能认识到非if逻辑吗？如果不能，（重新）阅读第1章。
* ② 这是在两个单词之间进行简单的“或”选择。
* ③ 如果在$1中没有给定一个值，将tail -n5用作默认值；请参阅第38页的“默认值”。
* ④ 这个模式将匹配以“len”开头的任何单词，因此“len”或“length”都将匹配，但“lenny”和“lens”也将匹配。然后它调用Ls-Length函数（我们在脚本中定义的），将所有传递给此脚本的命令行参数（如果有的话）传递给它。



## 包装脚本

每个人都有很多事情要处理，很多需要记住的事情，因此当你可以自动化或编写脚本来为你记住细节时，这是一种胜利。我们在示例3-1中展示了一种进行“包装脚本”的方法，但根据你要解决的问题的复杂性或要“记住”的细节，有许多有趣的变化和技巧可以使用。在示例3-1中，我们调用了一个函数或者将代码嵌入其中。这在我们的经验中适用于非常短的代码块，在这些包装脚本中这是相当常见的。如果你有一个更复杂的解决方案，或者你正在使用现有工具，你可以调用它们或者调用子脚本，尽管你需要调整错误检查和可能的使用选项。你还可以将其与第93页的“Drop-in Directories”结合使用，并从一个目录中源化所有的“模块”，也许是将代码的某些部分的维护委托给其他人或团队。这个较大的示例实际上是我们在编写这本书时使用的脚本的简化和摘录版本。AsciiDoc很酷，但我们使用了很多标记语言，它们都混在一起，因此我们可以编写一个工具来帮助我们记住事情，就像在示例3-2中所示。

```bash
#!/usr/bin/env bash
# wrapper.sh: Simple "wrapper" script demo
# Original Author & date: _bash Idioms_ 2022
# bash Idioms filename: examples/ch03/wrapper.sh
#_________________________________________________________________________
# Trivial Sanity Checks                                    ①
[ -n "$BOOK_ASC" ] || {
    echo "FATAL: export \$BOOK_ASC to the location of the Asciidoc files!"
    exit 1
}     
\cd "$BOOK_ASC" || {
    echo "FATAL: can't cd to '$BOOK_ASC'!"
    exit 2
}
SELF="$0"                                                              ②
action="$1"                                                            ③
shift                                                                   ④
[ -x /usr/bin/xsel -a $# -lt 1 ] && {                       ⑤
    # Read/write the clipboard on Linux
    text=$(xsel -b)
    function Output {
        echo -en "$*" | xsel -bi
    }
} || {
    # Read/write STDIN/STDOUT
    text=$*
    function Output {
        echo -en "$*"
    }
}
case "$action" in                                                        ⑥
#######################################################################
# Content/Markup                                               ⑦
### Headers                                                    ⑧
h1)                                                            ⑨
    # Inside chapter heading 1 (really AsciiDoc h3)            ⑩
    Output "[[$($SELF id $text)]]\n=== $text"
    ;;
h2)
    # Inside chapter heading 2 (really AsciiDoc h4)
    Output "[[$($SELF id $text)]]\n==== $text"
    ;;
h3)
    # Inside chapter heading 3 (really AsciiDoc h5)
    Output "[[$($SELF id $text)]]\n===== $text"
    ;;
### Lists
bul | bullet)
    # Bullet list (** = level 2, + = multiline element)
    Output "* $text"
    ;;
nul | number | order*) # Numbered/ordered list (.. = level 2, + = multiline)
    Output ". $text"
    ;;
term)
    # Terms
    Output "term_here::\n $text"
    ;;
### Inline
bold)
    # Inline bold (O'Reilly prefers italics to bold)
    Output "*$text*"
    ;;
i | italic* | itl)
    # Inline italics (O'Reilly prefers italics to bold)
    Output "_${text}_"
    ;;
c | constant | cons)
    # Inline constant width (command, code, keywords, more)
    Output "+$text+"
    ;;
type | constantbold) # Inline bold constant width (user types literally)
    Output "*+$text+*"
    ;;
var | constantitalic) # Inline italic constant width (user-supplied values)
    Output "_++$text++_"
    ;;
sub | subscript)
    # Inline subscript
    Output "~$text~"
    ;;
sup | superscript)
    # Inline superscript
    Output "^$text^"
    ;;
foot)
    # Create a footnote
    Output "footnote:[$text]"
    ;;
url | link)
    # Create a URL with alternate text
    Output "link:\$\$$text\$\$[]"
    # URL[link shows as]
    ;;
esc | escape)
    # Escape a character (esp. *)
    Output "\$\$$text\$\$"
    # $$*$$
    ;;
#######################################################################
# Tools                                                                 ⑪
id)
    ## Convert a hack/recipe name to an ID
    #us_text=${text// /_} # Space to '_'
    #lc_text=${us_text,,} # Lowercase; bash 4+ only!
    # Initial `tr -s '_' ' '` to preserve _ in case we process an ID
    # twice (like from "xref")
    # Also note you can break long lines with a trailing \              ⑫
    Output $(echo $text | tr -s '_' ' ' | tr '[:upper:]' '[:lower:]' |
        tr -d '[:punct:]' | tr -s ' ' '_')
    ;;
index)
    ## Creates 'index.txt' in AsciiDoc dir
    # Like:
    # ch02.asciidoc:== The Text Utils
    # ch02.asciidoc:=== Common Text Utils and similar tools
    # ch02.asciidoc:=== Finding data
    egrep '^=== ' ch*.asciidoc | egrep -v '^ch00.asciidoc' \
        >$BOOK_ASC/index.txt && {
        echo "Updated: $BOOK_ASC/index.txt"
        exit 0
    } || {
        echo "FAILED to update: $BOOK_ASC/index.txt"
        exit 1
    }
    ;;
rerun)
    ## Run examples to re-create (existing!) output files
    # Only re-run for code that ALREADY HAS a *.out file...not ALL *.sh code
    for output in examples/*/*.out; do
        code=${output/out/sh}
        echo "Re-running code for: $code > $output"
        $code >$output
    done
    ;;
cleanup)
    ## Clean up all the xHTML/XML/PDF cruft
    rm -fv {ch??,app?}.{pdf,xml,html} book.{xml,html} docbook-xsl.css
    ;;
*)                                                       ⑬
    \cd - # UGLY cheat to revert the 'cd' above...
    (
        echo "Usage:"                                    ⑭
        egrep '\)[[:space:]]+# '  $0                     ⑮                   
        
        echo ''
        egrep '\)[[:space:]]+## ' $0                     ⑯
        echo ''
        egrep '\)[[:space:]]+### ' $0                    ⑰
    ) | grep "${1:-.}" | more
    ;;
esac

```

我们在这里有很多事情要处理，所以让我们逐一分解：

* ① 实际脚本对本书的AsciiDoc源代码进行了许多操作，因此确保我们处于正确的位置并且设置了一个方便的环境变量，这样做会更容易。
* ② 通常我们使用\\$PROGRAM来保存bash的基本名称，但在这种情况下，我们将经常递归调用此脚本，因此\\$SELF似乎更直观。
* ③ 正如我们将在第11章中更详细地讨论的，使用有意义的变量名而不是位置参数是一个好主意，所以让我们这样做。
* ④ 一旦我们捕捉到动作，我们就不再需要旧的\\$1了，但可能还有更多的选项，所以将\\$1移走。
* ⑤ 如果/usr/bin/xsel存在且可执行，并且没有更多的选项，我们就知道我们正在读写X Window剪贴板，否则我们正在从参数中获取文本并将输出发送到STDOUT。在实践中，我们从编辑器复制，切换到命令行，运行工具，然后切换回来并粘贴。

* ⑥ 这是我们实际开始执行某些操作的地方，也就是找出我们的“动作”是什么。
* ⑦ 为了代码组织和可读性，将动作分成几个部分；另请参见⑪。
* ⑧ 让我们从标题的标记开始。
* ⑨ 这一行既是代码又是文档。\\$action是我们想要一个顶级（对于书中代码）标题，即h1。我们将在后面看到这也是文档。
* ⑩ 开始工作。首先，调用自己以获取文本的AsciiDoc“ID”，然后将该ID输出在双方括号中，接着是换行，然后用===缩进文本以表示标题级别，最后调用Output函数。希望代码的其余部分容易理解。
* ⑪ 为了代码组织和可读性，将动作分成几个部分；另请参见。
* ⑫ 可以使用尾随的\来分隔长行；另请参见第130页的“布局”。
* ⑬ 在通用情况下，通过将帮助或使用与未知参数处理以及出色的“在帮助中搜索”功能结合起来，我们再次变得有趣。
* ⑭ 我们将任何输出都包装在子shell中，以便在可能很长的情况下将其传递到更多中。
* ⑮ 这是我们在callout中谈到的“代码作为文档的行”⑨。我们使用grep搜索我们的case语句中的右括号)，后面是空格，然后是单个注释标记#。这给我们提供了我们的一级“内容/标记”动作。这提取了使case语句工作的实际代码行，但也因为我们如何添加注释而显示了它的功能。
* ⑯ 对于二级“工具”部分，这做了相同的事情。
* ⑰ 对于处理Git操作的三级，它将执行相同的操作，但出于简化的原因，我们在这里省略了该代码。但它还使用了grep和\\${1:-.}来显示我们请求的帮助，比如wrapper.sh help heading，或者显示所有内容（grep "."）。对于这么短的脚本，这可能看起来不像个大问题，但当它随时间增长（而且它会）时，这变得非常方便！
先前提到的grep命令和“级别”的结果是显示一个已排序但分为“一级”和“二级”部分的帮助消息：


```bash
$ examples/ch03/wrapper.sh help
Usage:
h1 )
 # Inside chapter heading 1 (really AsciiDoc h3)
h2 )
 # Inside chapter heading 2 (really AsciiDoc h4)
h3 )
 # Inside chapter heading 3 (really AsciiDoc h5)
bul|bullet )
 # Bullet list (** = level 2, + = multiline element)
nul|number|order* ) # Numbered/ordered list (.. = level 2, + = multiline)
term )
 # Terms
bold )
 # Inline bold (ORA prefers italics to bold)
i|italic*|itl )
 # Inline italics (ORA prefers italics to bold)
c|constant|cons )
 # Inline constant width (command, code, keywords, more)
type|constantbold ) # Inline bold constant width (user types literally)
var|constantitalic ) # Inline italic constant width (user-supplied values)
sub|subscript )
 # Inline subscript
sup|superscript )
 # Inline superscript
foot )
 # Create a footnote
url|link )
 # Create a URL with alternate text
esc|escape )
 # Escape a character (esp. *)
id )
 ## Convert a hack/recipe name to an ID
index )
 ## Creates 'index.txt' in AsciiDoc dir
rerun )
 ## Run examples to re-create (existing!) output files
cleanup )
 ## Clean up all the xHTML/XML/PDF cruft


$ examples/ch03/wrapper.sh
 help heading
h1 )
 #
 Inside chapter heading 1 (really AsciiDoc h3)
h2 )
 #
 Inside chapter heading 2 (really AsciiDoc h4)
h3 )
 #
 Inside chapter heading 3 (really AsciiDoc h5)
```




## 再来一个变化

在与模式相关联的每一段代码的末尾，我们都以两个分号结束。在本章开头的第一个示例中，我们写道：
```
"yes") echo "glad you agreed" ;;
```

在echo命令之后，我们放置了两个分号，表示不应采取进一步的操作。执行将在esac关键字之后继续。
但有时你可能不希望出现这种行为。在某些情况下，你可能希望检查case语句中的其他模式，或者采取其他操作。bash中的语法允许这样做，使用;;&amp;和;&amp;表示这些变体。
以下是展示这种行为的示例，提供了有关\\$filename路径的详细信息：

```bash
case $filename in
    ./*) echo -n "local "
          ;&
    [^/]*) echo -n "relative "
          ;;&
    /*) echo -n "absolute "
          ;&
    */*) echo "pathname"
          ;;
    *) echo "filename"
          ;;
esac
```
模式将按顺序与\\$filename中的值进行比较。第一个模式是两个文字字符—一个句点和一个斜杠—后跟任何字符。如果匹配（例如，如果\\$filename的值是./this/file），那么脚本将打印“local”，但末尾没有换行符。下一行是;&amp;，告诉bash“继续执行”与下一个模式相关联的命令（甚至不检查是否匹配）。因此，它还会打印“relative”。与前一个模式不同，这段代码的结尾是;&amp;，告诉bash尝试其他模式（按顺序向前）以寻找匹配。
因此，现在它将检查下一个模式，寻找前导斜杠。如果不匹配，下一个可能匹配。它查找字符串中的任何地方都有一个斜杠（任意—零个或多个—字符，然后是一个斜杠，然后是任意字符）。如果匹配（在我们的示例中是这样），它将打印单词pathname。;;表示不需要再检查其他模式，并且完成。



## 风格和可读性：总结

在本章中，我们描述了case语句，这是执行流程中的一个n路分支。它的模式匹配特性使其在脚本编写中非常有用，尽管常见用途是对特定单词进行简单的文字匹配。
变体;;、;;&amp;和;&amp;提供了一些有用的功能，但可能会比较棘手。使用if/then/else而不是case语句可能更好地组织这样的逻辑。
这些符号之间的微妙差异可能很容易被忽视，每一步发生的事情可能会有所不同。在匹配完成后的控制流可能对每种情况都不同：继续执行更多的代码、尝试匹配另一个模式或完成。因此，我们强烈建议在脚本中对您的选择进行注释，以避免混淆或误解。










