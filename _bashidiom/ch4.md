---
layout:     post
title:      "第四章：Variable Vernacular"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - bash 
---



 <!--more-->

这里经常看到一个错误消息或赋值语句，其中包含习惯用法\\${0##*/}，看起来是对 \\$0 的某种引用，但实际上更多的事情正在发生。让我们仔细研究一下变量引用以及其中的一些额外字符对我们有什么作用。我们将发现一系列字符串操作，通过其中的一些特殊字符，你可以在其中获得相当大的能力。

## 变量引用

在大多数编程语言中，引用变量的值非常简单。你可以直接使用变量的名称，或者在名称后添加一个字符以明确表示你想检索该值。在bash中也是如此：通过名称赋值给变量，例如 VAR=something，然后使用以美元符号为前缀的方式检索值： $VAR。如果你想知道为什么我们需要美元符号，请考虑到bash主要处理字符串，所以：

```bash
MSG="Error: FILE not found"
```

将给你一个简单的文字字符串，显示了四个单词，而：

```bash
MSG="Error: $FILE not found"
```

将用该变量的值（假设该值保存了它正在寻找的文件的名称）替换 \\$FILE。

**变量替换**
如果要进行字符串替换，请确保使用双引号。使用单引号将使所有字符被视为文字，不进行任何替换。

为了避免对变量名何时结束的困惑（在这个例子中，空格使得很容易理解），更完整的变量引用语法使用花括号括住变量名\\${FILE}，这在我们的示例中也可以使用。

这种带有花括号的语法是围绕变量引用的许多特殊语法的基础。例如，我们可以在变量名前面放一个#号 \\${#VAR}，不返回其值，而是返回值的字符串长度。

\\${VAR}                        | \\${#VAR}
-------------------------|-------
oneword                         | 7
/usr/bin/longpath.txt   | 21
many words in one string | 24
3                                   | 1
 1                                  | 2356
 4                                  | 1427685
 7                                  | 7

但bash可以做的不仅仅是简单地检索值或其长度。



## 参数扩展
在检索变量的值时，可以指定特定的替换或编辑，影响返回的值（尽管不影响变量中的值，除了一种情况）。语法涉及大括号内使用的特殊字符序列，例如这些大括号内的字符：${VAR##*/}。以下是一些值得知道的这样的扩展。

### 缩写basename

当调用脚本时，您可能只使用其文件名作为调用脚本的命令，但这假设脚本具有执行权限并且位于PATH变量中的一个目录中。如果脚本位于当前目录中，您可能使用./scriptname调用它。您可能会使用完整的路径名/home/smith/utilities/scriptname调用它，或者如果当前工作目录附近，则可能使用相对路径名。

无论以哪种方式调用脚本，\\$0将包含您用于调用脚本的字符序列—相对路径或绝对路径，无论您如何表示它。

当您想要在使用消息中打印该脚本的名称时，您可能只想要basename，即文件本身的名称，而不包括您到达那里的任何路径：

```bash
echo "usage: ${0##*/} namesfile datafile"
```
您可能会在使用消息中看到它，告诉用户运行脚本的正确语法，或者它可能是对变量的赋值的右侧。在后一种情况下，我们希望变量被称为PROGRAM或SCRIPT之类的名称，因为这就是该表达式返回的—正在执行的脚本的名称。

让我们更仔细地看看在$0上的这个特定的参数扩展，您可以使用它来获取basename，而不包括路径的其他部分。

### 路径或前缀删除

您可以从值的前面（前缀或左侧）或尾部（后缀或右侧）删除字符。要从字符串的左侧删除一组特定字符，您在参数引用上添加一个＃和一个shell模式，该模式匹配您想要删除的那些字符。
表达式 \\${MYVAL#img_} 将删除字符 img_，如果它们是 MYVAL 变量中字符串的第一个字符。使用更复杂的模式，我们可以写成 \\${MYVAL#\*_}。这将删除任何字符序列，直到并包括下划线（如果没有匹配的模式，它将返回未更改的完整值）。

单个＃表示它将使用可能的最短匹配（非贪婪）。双＃表示使用可能的最长匹配（贪婪）。

现在，也许你能看出表达式 \\${0##\*/} 会做什么了吗？

它将从 \\$0 的值开始，即用于调用脚本的路径名。然后，从该值的左侧，它将删除以斜杠结尾的任意数量字符的最长匹配。因此，它将删除在调用脚本时使用的路径的所有部分，仅留下脚本本身的名称。
以下是\\$0和我们讨论过的此模式的一些可能值，以查看短（＃）和长（＃＃）匹配的结果可能有何不同：





| Value in $0               | Expression    | Result returned         |
|---------------------------|---------------|-------------------------|
| ./ascript                 | ${0#*/}        | ascript                 |
| ./ascript                 | ${0##*/}       | ascript                 |
| ../bin/ascript            | ${0#*/}        | bin/ascript             |
| ../bin/ascript            | ${0##*/}       | ascript                 |
| /home/guy/bin/ascript     | ${0#*/}        | home/guy/bin/ascript    |
| /home/guy/bin/ascript     | ${0##*/}       | ascript                 |


请注意，* / 的最短匹配模式可能仅匹配斜杠本身。

**Shell模式，而不是正则表达式**
在参数扩展中使用的模式不是正则表达式。它们只是shell模式匹配，其中 * 匹配0或更多个字符，？匹配一个字符，[chars]匹配花括号内的任何一个字符。


### 后缀删除

类似于 # 会删除前缀，也就是从左侧删除，我们可以通过使用 % 删除后缀，也就是从右侧删除。双百分号表示删除可能的最长匹配。以下是一些示例，演示如何删除后缀。首先的例子展示了一个变量 \\$FN，它保存了图像文件的名称。它可能以 .jpg、.jpeg、.png 或 .gif 结尾。看看不同的模式如何删除字符串右侧的不同部分。最后几个例子展示了如何从 \\$0 参数中获取类似于 dirname 的东西：
这个对于 dirname 的参数替代并不完全复制命令的输出。在路径为 /file 的情况下有所不同，因为 dirname 会返回一个斜杠，而我们的参数替代会将其全部删除。如果希望检查这一点，可以在脚本中添加一些额外的逻辑，如果不希望看到它，可以忽略这种情况，或者可以在参数的末尾添加一个斜杠，如 \\${0%/\*}/，以便所有结果都以斜杠结尾。


| Shell变量的值           | 表达式        | 返回的结果           |
|-------------------------|---------------|----------------------|
| img.1231.jpg            | ${FN%.*}      | img.1234             |
| img.1231.jpg            | ${FN%%.*}     | img                  |
| ./ascript               | ${0%/*}       | .                    |
| ./ascript               | ${0%%/*}      | .                    |
| /home/guy/bin/ascript   | ${0%/*}       | /home/guy/bin        |
| /home/guy/bin/ascript   | ${0%%/*}      |        |

**前缀和后缀删除**
可以记住 # 删除左侧部分，% 删除右侧部分，因为至少在标准的美国键盘上，# 是 shift-3，位于 % 的左侧，是 shift-5。

###  其他修改器
除了 # 和 % 之外，还有一些其他可以通过参数扩展修改值的修改器。您可以通过 ^ 或 ^^ 将字符串的第一个字符或所有字符转换为大写，或通过 , 或 ,, 将其转换为小写，如下例所示：

| Value in shell variable | Expression | Result returned     |
|--------------------------|------------|----------------------|
| message to send          | ${TXT^}    | Message to send      |
| message to send          | ${TXT^^}   | MESSAGE TO SEND      |
| Some Words               | ${TXT,}    | some Words           |
| Do Not YELL              | ${TXT,,}   | do not yell          |

您还可以考虑使用 declare -u UPPER 和 declare -l lower，它们会声明这些shell变量，使得分配给这些变量的任何文本都会被转换为大写或小写。

最灵活的修改器是在字符串的任何位置执行替换，而不仅仅是在字符串的开头或结尾。类似于sed命令，它使用斜杠 / 表示要匹配的模式和要替换的值。一个斜杠表示单次替换（第一次出现）。使用两个斜杠表示替换每一次出现。以下是一些示例：

| Value in shell variable | Expression          | Result returned              |
|--------------------------|---------------------|------------------------------|
| FN="my filename with spaces.txt” | ${FN/ /_} | my_filename with spaces.txt |
| FN="my filename with spaces.txt” | ${FN// /_} | my_filename_with_spaces.txt |
| FN="my filename with spaces.txt” | ${FN// /}  | myfilenamewithspaces.txt    |
| FN="/usr/bin/filename”     | ${FN//\// }         |[空格]usr bin filename            |
| FN="/usr/bin/filename”     | ${FN/\// }          |[空格]usr/bin/filename            |


**没有尾随斜杠**
请注意，不像您在其他类似的命令（如sed或vi）中找到的那样，这里没有尾随斜杠。闭合大括号结束了替换。

为什么不总是使用这种替换机制呢？为什么要费心使用字符串末尾的 # 或 % 替代？考虑这个文件名：frank.gifford.gif，假设您想要使用Image Magick的convert命令（这是另一个故事）将这个文件名更改为一个jpg文件。使用 / 进行替代没有办法将搜索锚定到字符串的一端或另一端。如果您读取了文件名并尝试将 .gif 替换为 .jpg，最终得到的将是 frank.jpgford.gif。对于这种情况，从字符串末尾取出的 % 替代效果要好得多。

另一个有用的修改器将提取变量的子字符串。在变量名称之后，放置一个冒号，然后是要提取的子字符串的第一个字符的偏移量。由于这是一个偏移量，请从字符串的第一个字符开始，将偏移量设置为0。接下来，再放置一个冒号和您想要的子字符串的长度。如果省略了这个第二个冒号和长度，那么您将得到整个剩余的字符串。以下是一些示例：

| Value in shell variable | Expression        | Result returned   |
|--------------------------|-------------------|--------------------|
| /home/bin/util.sh        | ${FN:0:1}         | /                  |
| /home/bin/util.sh        | ${FN:1:1}         | h                  |
| /home/bin/util.sh        | ${FN:3:2}         | me                 |
| /home/bin/util.sh        | ${FN:10:4}        | util               |
| /home/bin/util.sh        | ${FN:10}          | util.sh            |

示例4-1展示了使用参数扩展从某个输入中解析数据，以在自动创建防火墙规则配置时创建和处理特定字段。我们还在代码中包含了一个更大的bash参数扩展表，就像在本书中经常做的那样，作为一个“真实代码可读性”的示例。输出如示例4-2所示。

```bash
#!/usr/bin/env bash
# parameter-expansion.sh: parameter expansion for parsing, and a big list
# Original Author & date: _bash Idioms_ 2022
# bash Idioms filename: examples/ch04/parameter-expansion.sh
#_________________________________________________________________________
# Does not work on Zsh 5.4.2!

customer_subnet_name='Acme Inc subnet 10.11.12.13/24'

echo ''
echo "Say we have this string: $customer_subnet_name"

customer_name=${customer_subnet_name%subnet*}  # Trim from 'subnet' to end
subnet=${customer_subnet_name##* }             # Remove leading 'space*'
ipa=${subnet%/*}                               # Remove trailing '/*'
cidr=${subnet#*/}                              # Remove up to '/*'
fw_object_name=${customer_subnet_name// /_}    # Replace space with '_-
fw_object_name=${fw_object_name////-}          # Replace '/' with '-'
fw_object_name=${fw_object_name,,}             # Lowercase

echo ''
echo 'When the code runs we get:'
echo ''
echo "Customer name: $customer_name"
echo "Subnet:        $subnet"
echo "IPA            $ipa"
echo "CIDR mask:     $cidr"
echo "FW Object:     $fw_object_name"

# bash Shell Parameter Expansion: https://oreil.ly/Af8lw

# ${var#pattern}                Remove shortest (nongreedy) leading pattern
# ${var##pattern}               Remove longest (greedy) leading pattern
# ${var%pattern}                Remove shortest (nongreedy) trailing pattern
# ${var%%pattern}               Remove longest (greedy) trailing pattern

# ${var/pattern/replacement}    Replace first +pattern+ with +replacement+
# ${var//pattern/replacement}   Replace all +pattern+ with +replacement+

# ${var^pattern}                Uppercase first matching optional pattern
# ${var^^pattern}               Uppercase all matching optional pattern
# ${var,pattern}                Lowercase first matching optional pattern
# ${var,,pattern}               Lowercase all matching optional pattern

# ${var:offset}                 Substring starting at +offset+
# ${var:offset:length}          Substring starting at +offset+ for +length+

# ${var:-default}               Var if set, otherwise +default+
# ${var:=default}               Assign +default+ to +var+ if +var+ not already set
# ${var:?error_message}         Barf with +error_message+ if +var+ not set
# ${var:+replaced}              Expand to +replaced+ if +var+ _is_ set

# ${#var}                       Length of var
# ${!var[*]}                    Expand to indexes or keys
# ${!var[@]}                    Expand to indexes or keys, quoted

# ${!prefix*}                   Expand to variable names starting with +prefix+
# ${!prefix@}                   Expand to variable names starting with +prefix+, quoted

# ${var@Q}                      Quoted
# ${var@E}                      Expanded (better than `eval`!)
# ${var@P}                      Expanded as prompt
# ${var@A}                      Assign or declare
# ${var@a}                      Return attributes
```
运行的结果为：
```

Say we have this string: Acme Inc subnet 10.11.12.13/24

When the code runs we get:

Customer name: Acme Inc 
Subnet:        10.11.12.13/24
IPA            10.11.12.13
CIDR mask:     24
FW Object:     acme_inc_subnet_10.11.12.13-24
```


## 条件替换

其中一些变量替换是有条件的，即它们仅在满足特定条件时发生。您可以通过在赋值周围使用if语句来完成相同的事情，但这些习惯用法可用于某些常见情况下编写更短的代码。这些有条件的替换在这里显示为一个冒号，然后是另一个特殊字符：减号、加号或等号。它们检查的条件是：变量是否为null或未设置？空变量是其值为null字符串的变量。未设置的变量是尚未被分配或明确使用unset命令取消分配的变量。对于位置参数（如\\$1、\\$2等），如果用户没有在该位置提供参数，则它们未设置。

如果在这些条件替换中不包括冒号，则它们只考虑未设置变量的情况；null值将原样返回。

### 默认值

一个常见的情况是一个具有单个可选参数的脚本。如果在调用脚本时未提供参数，则应使用默认值。在bash中，我们可以编写类似以下的内容：

```bash
LEN=${1:-5}
```

这将将变量LEN设置为第一个参数（\\$1）的值（如果已提供），否则设置为值5。以下是一个示例脚本：

```bash
LEN="${1:-5}"
cut -d',' -f2-3 /tmp/megaraid.out | sort | uniq -c | sort -rn | head -n "$LEN"
```

它从名为/tmp/megaraid.out的逗号分隔值文件中提取第二个和第三个字段，对这些值进行排序，提供每个值对出现次数的计数，然后显示列表中的前5个值。您可以通过将该计数指定为脚本的唯一参数来覆盖默认值5，并显示前3或前10（或您想要的任何数量）。

### 逗号分隔列表

另一种条件替换，使用加号，还检查变量是否有值，如果有，则返回不同的值。也就是说，仅当变量不为null时，它才返回指定的不同值。是的，这听起来很奇怪；如果它有一个值，为什么要返回一个不同的值呢？

这种看似奇怪逻辑的一个方便的用途是构造逗号分隔列表。通常，通过重复附加“，value”或“value，”来构造这样的列表。在这样做时，通常需要一个if语句来避免在列表的前端或末尾有额外的逗号，但当您使用这个连接习惯时就不需要了：

```bash
for fn in * ; do
  S=${LIST:+,} # S for separator
  LIST="${LIST}${S}${fn}"
done
```

另请参见示例7-1。

### 修改的值

到目前为止，这些替换中没有一个修改了变量的基础值。然而，有一个例外。如果我们写 \\${VAR:=value}，它将类似于我们之前的默认值习惯，但有一个很大的例外。如果 VAR 为空或未设置，它将将该值赋给变量（因此，等号），并返回该值。（如果 VAR 已经设置，它将简单地返回其现有值。）但请注意，这个赋值的操作不适用于位置参数（如 $1），这就是为什么你不常见到它被使用的原因。

## \\$RANDOM
Bash 提供了一个非常方便的 \\$RANDOM 变量。正如《Bash 参考手册》中的“Bash 变量”部分所述：
每次引用此参数时，都会生成一个介于 0 和 32767 之间的随机整数。给这个变量赋值将为随机数生成器设置种子。虽然这不适用于密码学功能，但对于掷骰子或在预测性操作中添加一些噪音非常有用。我们在后面的“一个简单的单词计数示例”（第69页）中使用了这个功能。
如示例4-3所示，你可以从列表中随机选择一个元素。
 

```bash
declare -a mylist
mylist=(foo bar baz one two "three four")
range=${#mylist[@]}
random=$(( $RANDOM % $range )) # 0 到列表长度的计数
echo "range = $range, random = $random, choice = ${mylist[$random]}"
# 更短，但可读性较差:
# echo "choice = ${mylist[$(( $RANDOM % ${#mylist[@]} ))]}"
```

你可能也会看到类似以下的用法：

```bash 
TEMP_DIR="$TMP/myscript.$RANDOM"
[ -d "$TEMP_DIR" ] || mkdir "$TEMP_DIR"
```

然而，这可能受到竞态条件的影响，并且显然是一个简单的模式。它也在某种程度上是可预测的，但有时你希望知道什么代码正在混淆 $TMP。不要忘记设置陷阱（参见“这是一个陷阱！”第97页）来在使用后清理。我们建议考虑使用 mktemp，虽然这是一个超出 bash 惯用法范围的大问题。

**\\$RANDOM 和 dash**

\\$RANDOM 在 dash 中不可用，dash 在一些 Linux 发行版中是 /bin/sh。值得注意的是，目前的 Debian 和 Ubuntu 版本使用 dash，因为它比 bash 更小更快，从而有助于更快地启动。但这意味着 /bin/sh，以前是指向 bash 的符号链接，现在是指向 dash 的符号链接，因此各种 bash 特定的功能将无法使用。但在 Zsh 中是可以工作的。


## 命令替换
我们在第2章中已经大量使用了命令替换，但我们还没有详细讨论过它。旧的 Bourne 方式是 ``（重音符/反引号），但我们更喜欢更易读的 POSIX $()。你会看到这两种形式很多，因为这是将输出导入变量的方式；例如：

```bash
unique_lines_in_file="$(sort -u "$my_file" | wc -l)"
```
请注意，这两者是相同的，但第二个是内部的且更快：

```bash
for arg in $(cat /some/file)
for arg in $(< /some/file)
 # 比调用 cat 更快
```

**命令替换**

命令替换对于云和其他 DevOps 自动化至关重要，因为它允许您收集和使用仅在运行时存在的所有 ID 和详细信息；例如：

```bash
instance_id=$(aws ec2 run-instances --image $base_ami_id ... \
--output text --query 'Instances[*].InstanceId')
state=$(aws ec2 describe-instances --instance-ids $instance_id \
--output text --query 'Reservations[*].Instances[*].State.Name')
```

**嵌套命令替换**

使用 `` 进行嵌套命令替换变得非常丑陋且速度很快，因为您必须在每个嵌套层中转义内部的反引号。如果可以的话，最好使用 $()，如下所示：

```bash
### 正常工作
$ echo $(echo $(echo $(echo inside)))
inside
### 损坏
$ echo `echo `echo `echo inside```
echo inside
### “正常”但非常丑陋
$ echo `echo \`echo \\\`echo inside\\\`\``
inside
```

感谢我们的评论员 Ian Miell 指出并提供示例。

## 风格和可读性：总结
在 bash 中引用变量时，您有机会在设置或检索值时对其进行编辑。在变量引用的末尾使用一些特殊字符可以从字符串值的前面或末尾删除字符，将其字符改为大写或小写，替换字符，或者仅获取原始值的子字符串。对这些方便的功能的常见用法导致了对默认值、basename 和 dirname 替代以及创建逗号分隔列表的习惯用法，而无需使用显式的 if 语句。

变量替换是 bash 中的一个很棒的功能，我们建议充分利用它们。然而，我们也强烈建议您对这些语句进行注释，以明确您尝试进行的替换类型。您代码的下一个阅读者将感激不已。








