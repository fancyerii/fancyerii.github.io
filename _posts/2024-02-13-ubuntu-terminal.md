---
layout:     post
title:      "Ubuntu18.04升级Python3.6之后terminal无法打开"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - python
    - Ubuntu
    - source code
    - build
---

 

 <!--more-->

 

最近把Ubuntu18.04的python升级到python3.9之后发现terminal无法打开了。搜索了一下，是gnome-terminal只能使用3.6(含)之前的版本。最简单的办法是用update-alternatives把python3修改成python3.6(不用修改python，这是两个不同的东西，有的程序会使用python，因为它在Ubuntu18.04里默认是2.7，另外一些程序会使用python3来明确的说明版本要求)。但是我不希望系统的python3版本太低，因为有些程序它要求比较高的版本。搜索了一下，发现[这里](https://askubuntu.com/questions/1132349/terminal-not-opening-up-after-upgrading-python-to-3-7)有比较好的解决方法，那就是修改/usr/bin/gnome-terminal，修改第一行：

```shell
#!/usr/bin/python3
```

改为：
```shell
#!/usr/bin/python3.6
```

就可以了，这样terminal使用python3.6，而系统的python3指向新版本的python。

另外一个问题就是如果输入一个不存在的命令，会出现奇怪的错误：

```
$ abcd
$ abcd
Traceback (most recent call last):
  File "/usr/lib/command-not-found", line 27, in <module>
    from CommandNotFound.util import crash_guard
ModuleNotFoundError: No module named 'CommandNotFound'
```

通过搜索，找到[这个问题](https://unix.stackexchange.com/questions/9580/why-is-this-python-error-message-generated-whenever-i-type-a-nonsense-command)。原因还是因为python3从3.6升级带来的，只需要修改/usr/lib/command-not-found，把

```
#!/usr/bin/python3
```

改成

```
#!/usr/bin/python3.6
```


看起来Ubuntu的terminal是和特点版本的python绑定的，如果升级了python的版本，可能带来很多问题。如果发现了问题，一般都可以通过shell脚本的#!来解决。当然如果把python3指向python3.6就不会有任何问题了。


