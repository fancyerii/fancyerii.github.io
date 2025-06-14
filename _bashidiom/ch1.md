---
layout:     post
title:      "第一章：A Big “if” Idiom"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - bash 
---



 <!--more-->

为了帮助你理解Bash习语，我们将看一个Bash结构，它允许你执行通常使用if/then/else结构完成的任务，但语法更简洁。我们在本章将向你展示的典型表达式具有一些优势，主要是简洁明了，同时也有一些需要避免的陷阱。但如果你不熟悉Bash习语，你可能无法识别或理解其中的奥妙。

看一下这段代码：

```bash
[[ -n "$DIR" ]] && cd "$DIR"
```

你认为这看起来像一个if语句吗？如果你熟悉Bash，你会认识到它在功能上与if语句是相同的；尽管代码中没有出现if关键字，但你会理解它作为一个if语句。

这段代码实际上在做什么？

## 这个大的“if”

为了解释这个习语，让我们首先看一个类似但更简单的例子：

```bash
cd tmp && rm scratchfile
```

这实际上也是一个if语句。如果cd命令成功，那么（且仅当）执行rm命令。这里的“习语”是使用双“&&”符号，通常读作“and”，来分隔这两个命令。

逻辑或哲学课上教导的规则是：“表达式‘A AND B’仅当A和B都为真时才为真。”因此，如果A为假，就无需考虑B的值。例如，考虑“我有一只狗 AND 我有一只猫。”如果我没有一只狗，那么对我来说，这个复合表达式为假，而不考虑我是否有猫。

让我们将这个应用到bash。请记住，bash的基本功能是执行程序。这个语句的第一部分是执行cd命令。类似于AND的逻辑，如果第一个命令失败，bash将不会执行第二部分，即rm命令。

使用“&&”是为了提醒你AND行为。实际上，bash并没有在这两个结果上执行“AND”操作。（如果这是C/C++，情况就会有所不同，尽管条件执行是相同的。）然而，这个bash习语确实提供了第二个命令的条件执行——如果第一个命令失败，它就不会运行。

让我们回到我们之前给出的原始例子，即这个表达式：

```bash
[[ -n "$DIR" ]] && cd "$DIR"
```

现在你是否理解了？第一个表达式测试变量DIR的值的长度是否非零。如果它有一个值，即它的长度非零，那么cd命令将尝试切换到由DIR的值命名的目录。

我们也可以将其写成一个显式的if语句：

```
if [[ -n "$DIR" ]]; then
    cd "$DIR"
fi
```

对于对bash不太熟悉的人来说，后一种格式显然更可读且更容易理解。但在then子句内部没有太多的内容，只有cd命令，所以在语法上显得有点“臃肿”。你需要根据你的读者可能是谁以及then子句内可能添加其他命令的可能性来决定使用哪种格式。在接下来的章节中，我们将对这个主题提出更多观点。


**bash帮助**

bash的help命令对于任何内置命令都非常有用，但是help test提供了有关test表达式的特别有用的线索，比如-n。你可能还可以查看man bash，但如果这样做，你需要搜索“conditional expressions”（条件表达式）。bash的man页面非常长；help提供了更短、更专注的主题。如果你不确定要查询的命令是否是bash的内置命令，只需使用help命令尝试一下，或者使用type -a thing来查找。诚然，知道help test将告诉你-n的含义有点棘手，但是，既然你足够聪明购买了这本书，现在你就知道了。这里还有一个微妙的小提示；去试试这个：help [. 。


## 或者...

在bash中，还有一种类似的习惯用法，使用||字符来分隔bash列表中的两个项。发音为“or”，第二部分只有在第一部分失败时才会执行。这是为了让你想起“OR”的逻辑规则，即：A OR B。整个表达式在A为真或B为真时为真。换句话说，如果A为真，那么B是真还是假就无关紧要。例如，考虑短语“I own a dog OR I own a cat”。如果我确实拥有一只狗，那么对我来说，这个表达式是真的，不管我有没有猫。
应用到bash中：

```bash
[[ -z "$DIR" ]] || cd "$DIR"
```

你能解释这个吗？如果变量长度为零，那么第一部分是“真”，因此无需执行第二部分；不会执行cd命令。但如果$DIR的长度非零，则测试将返回“假”，只有在这种情况下才会运行cd命令。
你可以将这行bash代码理解为“要么$DIR的长度为零，要么我们尝试进入该目录”。
将其写成显式的if语句有点奇怪，因为没有then操作要执行。||后面的代码就像else子句：

```bash
if [[ -z "$DIR" ]]; then
    :
else
    cd "$DIR"
fi
```

":"是shell中的空语句，因此在这种情况下什么也不做。 

总结：由&&分隔的两个命令类似于if及其then子句；由||`分隔的两个命令类似于if及其else子句。

## 多个操作
在||后的类似else的子句中，你可能想要执行多个操作，但这里存在一个危险。可能会尝试编写如下代码：

```bash
# 警告：不是你可能想要的！
cd /tmp || echo "cd to /tmp failed." ; exit
```
“或”连接告诉我们，如果cd失败，我们将执行echo命令，告诉用户cd失败了。但是这里有个问题：无论如何都会执行exit。这不是你期望的结果，对吧？
将分号视为换行符，一切就变得更加清晰了（并且显然不是你想要的结果）：

```bash
cd /tmp || echo "cd to /tmp failed."
exit
```
如何获得我们想要的行为呢？我们可以将echo和exit组合在一起，使它们在“或”右侧的子句中，像这样：

```bash
# 如果`cd`成功则执行，否则输出消息并退出
cd /tmp || { echo "cd to /tmp failed." ; exit ; }
```
花括号是bash中用于组合多个语句的语法，即将语句组合在一起。你可能已经看到过使用圆括号类似的语法，但是使用圆括号会在子shell中执行语句，也称为子进程。这将带来我们不需要的开销，并且在子shell中执行的退出也不会有太多作用。

**复合命令的结束**
Bash 的一个怪癖要求以特定的语法关闭复合命令。它必须以分号或闭括号之前的换行符结束。如果使用分号，需要在括号之前添加空格，以便括号能够被识别为保留字（否则它会与shell变量语法的闭括号混淆，例如${VAR}）。这就是为什么前面的例子以看似多余的分号结尾的原因：{ echo "..." ; exit ; }。使用换行符，那个最后的分号就不再需要了：

```bash
# 如果`cd`成功则执行，否则输出消息并退出
cd /tmp || { echo "cd to /tmp failed." ; exit 
           }
```

但是这样的代码可能不够清晰。在左边缘，它似乎放置得很奇怪；使用空格缩进，逻辑上更合理，但看起来有点简陋。
我们建议你坚持使用额外的分号，并不要忘记它与闭括号之间的空格。


## 再谈多个
如果你需要更复杂的逻辑怎么办？多个AND和OR结构应该如何处理？以下代码会执行什么操作？

```bash
[ -n "$DIR" ] && [ -d "$DIR" ] && cd "$DIR" || exit 4
```
如果DIR变量不为空且由DIR变量命名的文件是一个目录，那么它将切换到该目录；否则，它将退出脚本，返回4。这正是你可能期望的，但原因可能不是你所想象的。
从这个例子中，你可能会认为&&比||运算符具有更高的优先级，但实际上并不是。它们只是从左到右进行分组。Bash的语法规定，&&和||运算符的优先级相等，并且是左结合的。需要说服吗？看看这些例子：

```bash
$ echo 1 && echo 2 || echo 3
1
2
$
```
但同时也有：

```
$ echo 1 || echo 2 && echo 3
1
3
$
```
请注意，它始终评估最左边的运算符，无论它是AND还是OR；它不是运算符优先级，而是简单的左结合决定了评估的顺序。


##  不要这样做
说到if语句（或者说如何不使用它们），我们在这里给出一个你可能经常在旧脚本中看到的显式if的例子。我们在这里展示它，以便解释这个习惯用法，同时敦促你永远不要模仿这种风格。以下是代码：

```bash
### 不要这样写你的if语句
if [ $VAR"X" = X ]; then
echo empty
fi
### 或者这样
if [ "x$VAR" == x ]; then
echo empty
fi
### 或其他类似的变体
```
不要这样做。这些代码在做什么？它们检查变量VAR是否为空。它们通过在VAR的值后附加一些字符（这里是X）来执行此操作。如果结果字符串仅与该字母本身匹配，那么变量为空。不要这样做。
有更好的方法来进行这个检查。以下是一个简单的替代方案：

```bash
# 变量的长度是否为零？即是否为空或为null
if [[ -z "$VAR" ]]; then
echo empty
fi
```

**单括号与双括号**

这个不应该做的示例使用了单括号 [ 和 ] 来包围它们正在测试的条件。我们要你避免的不是这个。我们希望你避免字符串的附加和比较；而是使用 -z 或 -n 来进行这些测试。那么为什么我们的例子都使用了双括号 [[ 和 ]] 来编写我们的 if（以及非 if）语句呢？它们是bash的一个补充（不在原始sh命令中），并且避免了单括号表现出的一些令人困惑的边缘情况行为（变量是否在引号内）。我们展示了这个单括号的例子，因为这种比较经常在旧脚本中看到。如果你的目标是在各种平台之间以及/或者到非bash平台（例如dash）之间保持可移植性，那么你可能需要使用单括号。值得一提的是，双括号是关键字，而左单括号是内置命令，这可能解释了一些细微差异的原因。我们的建议仍然是在可以避免的情况下使用双括号语法。



您可以检查相反的情况，即检查字符串的长度是否为非零，通过使用 -n 选项或直接引用变量：

```bash
#这检查非零长度，即非空，非null
if [[ -n "$VAR" ]]; then
echo "VAR有一个值：" $VAR
fi

#这也一样
if [[ "$VAR" ]]; then
echo 这样更简单
fi
```

因此，您看到没有必要使用以前版本的test命令（“ [”）中的另一种方法，这种方法现在很少使用了。我们认为您应该看到它，这样您就能在旧脚本中识别它。现在您还知道了更好的编写方式。


## 风格和可读性：总结

在本章中，我们深入了解了一个特定的bash习惯用语——“无if”的if语句。它看起来不像传统的if/then/else，但它的行为可以完全像一个。除非这是您认识的东西，否则您阅读的一些脚本可能仍然难以理解。

这个习惯用语还值得用来检查在执行命令之前是否满足任何必要的前提条件，或者在不中断脚本主逻辑流程的情况下进行简短的错误检查。

通过使用 && 和 || 运算符，您可以编写if/then/else逻辑，而无需使用那些熟悉的关键字。但是bash确实有if、then和else作为关键字。那么何时使用它们，何时使用简写呢？答案归结为可读性。

对于复杂的逻辑，使用熟悉的关键字是最合理的。但是，对于带有单个操作的简单测试和检查情况，使用 && 和 || 运算符非常方便，不会分散主逻辑的流程。使用help test来提醒您可以使用哪些测试，例如 -n -r，并考虑将帮助文本复制到注释中以备将来参考。

无论是对于熟悉的if语句还是惯用的“无if”语句，我们都鼓励使用双括号语法。

既然您已经深入了解了一个bash习惯用语，让我们看看其他的，真正提高您的bash水平。



