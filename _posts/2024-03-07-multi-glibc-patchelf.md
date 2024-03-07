---
layout:     post
title:      "在同一台Linux机器上安装多个版本GLIBC" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - linux
    - ldd
    - glibc
    - patchelf
---

有没有在运行一个程序时遇到"/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28' not found"？是不是打算升级GLIBC？千万不要这么做，否则你要随时面临崩溃的可能。升级Linux系统？就为了使用一个新的GLIBC而升级系统，万一又有一个程序需要更新版本的GLIBC怎么办？
本文解释通过自己编译GLIBC并且使用patchelf解决这类问题。

<!--more-->

**目录**
* TOC
{:toc}

## 概述

GLIBC是Linux最核心的库(没有之一)。我们知道Linux内核是c语言编写的，而为了运行c语言的程序，需要libc库。GLIBC是gnud的libc库，它替代了Linux的libc成为Linux事实的标准库。常见的发行版都有固定的glibc版本，比如目前常见LTS版本的glibc版本(参考[这里](https://launchpad.net/ubuntu/+source/glibc)：

Ubuntu版本|版本代号|glic版本
--|--|--
Ubuntu 16.04|Xenial Xerus|2.23
Ubuntu 18.04|Bionic Beaver|2.27
Ubuntu 20.04|Focal Fossa|2.31
Ubuntu 22.04|Jammy Jellyfish|2.35

我现在使用的是18.04，目前已经[End of Standard Support(2023/5/31)](Time to prepare for Ubuntu 18.04 LTS End of Standard Support on 31 May 2023)，也就是说官方不再支持了。因此很多软件的新版本就不再支持了。比如最近我安装Audacity，官方提供的AppImage就需要glibc 2.29。因为这不是本文的重点，感兴趣的读者可以参考[这个issue](https://github.com/audacity/audacity/issues/5927)。

另外一种解决办法就是自己编译，大部分代码其实都是可以在不同版本的glibc下编译运行的(除非使用了新版本glibc的函数)，比如我可以自己[编译Audacity](https://forum.audacityteam.org/t/undefined-reference-to-dladdr-glibc-2-34/98178/3)。但这也存在很多问题，比如编译时间很长，而且编译很可能会碰到各种奇怪的问题。


## 问题

今天我遇到的问题是安装最新版本的node，安装node可以从官方[下载](https://nodejs.org/en/download/current)，比如Linux可以下载预编译的[v21.7.0](https://nodejs.org/dist/v21.7.0/node-v21.7.0-linux-x64.tar.xz)。但是下载解压运行会出现：

```shell
lili@lili-Precision-7720:~/soft/node-v20.11.1-linux-x64$ ./bin/node 
./bin/node: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28' not found (required by ./bin/node)
```

我们可以查看一下系统的GLIBC版本：

```shell
$ ldd --version
ldd (Ubuntu GLIBC 2.27-3ubuntu1.6) 2.27
```

2.27和2.28就差了一个minor版本号，但是就是无法运行。

对于node来说，使用nvm来管理多个版本会更加方便，下面我们用nvm来安装node。不过不管用哪个方法，除非自己编译，否则都是会碰到类似的问题。

### 安装nvm

```shell
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

运行完了需要重新登入shell，或者：

```shell
source ~/.bashrc
```

我们看看nvm的版本：

```shell
$ nvm --version
0.39.7
```

没有问题，下面来安装最新版本的node：

```shell
nvm install node
```

运行一下node：

```shell
$ node
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28' not found
``` 

还是同样的问题。

## 解决方法

### 分析问题

我们找到它的具体位置：

```shell
$ which node
/home/lili/.nvm/versions/node/v21.7.0/bin/node
```

我们使用ldd看一下：

```
$ ldd node 
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28' not found (required by node)
	linux-vdso.so.1 (0x00007ffe4cbd6000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f6802109000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f6801ca9000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f680190b000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f68016e8000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f68014c9000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f68010d8000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f680230d000)

```

我们看到，node链接到系统的glibc(/lib/x86_64-linux-gnu/libc.so.6，/lib64/ld-linux-x86-64.so.2)

### 解决方法

为了解决这个问题，常规的思路是重新编译，让它链接到GLIBC_2.27。如果它真用到了GLIBC_2.28特有的函数，那么就需要在我们的机器上安装GLIBC_2.28然后链接到它。这很麻烦，我们可以安装GLIBC 2.28，但是我们不想重新编译代码。那怎么办呢？这就可以用到[patchelf](https://github.com/NixOS/patchelf)这个神器。它的原理是修改可执行文件(elf,so和obj)的rpath，感兴趣的读者可以参考[Multiple glibc on a Single Linux Machine](https://www.baeldung.com/linux/multiple-glibc)、[How can I link to a specific glibc version?](https://stackoverflow.com/questions/2856438/how-can-i-link-to-a-specific-glibc-version)和[Multiple glibc libraries on a single host](https://stackoverflow.com/questions/847179/multiple-glibc-libraries-on-a-single-host)。

### 安装patchelf

去[这里](https://github.com/NixOS/patchelf/releases/tag/0.18.0)下载最新版的patchelf，不要用系统自带的(apt install patchelf)，那个太老。

解压后放到PATH里：

```shell
$ patchelf --version
patchelf 0.18.0
```

### 安装GLIBC 2.28

可以参考[How to install GLIBC 2.29 or higher in Ubuntu 18.04](https://stackoverflow.com/questions/72513993/how-to-install-glibc-2-29-or-higher-in-ubuntu-18-04)和[struct __jmp_buf_tag问题](https://stackoverflow.com/questions/76079071/when-i-compile-glibc-2-28-with-the-make-command-on-centos-7-5-i-got-the-error-l)

```shell
wget -c https://ftp.gnu.org/gnu/glibc/glibc-2.28.tar.gz
tar -zxvf glibc-2.28.tar.gz
mkdir glibc-2.28/build
cd glibc-2.28/build
../configure --prefix=/opt/glibc228 --disable-werror
make -j8 
sudo make install
```

### 对node进行patch

我们一步一步来patch(可以一步到位，但是这里记录一下我的patch过程，这样下次遇到不同的文件就有思路了)。

首先找到node所在目录，做一个备份：

```shell
$ cd $(dirname `which node`)
$ cp node node.bak
```

第一步是添加glibc2.28到rpath：

```shell
patchelf --add-rpath /opt/glibc228/lib node
```
看一下有什么变化：

```shell
$ ldd node
	linux-vdso.so.1 (0x00007ffe0f8aa000)
	libdl.so.2 => /opt/glibc228/lib/libdl.so.2 (0x00007f4978454000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f4977ff4000)
	libm.so.6 => /opt/glibc228/lib/libm.so.6 (0x00007f4977c76000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f4977a53000)
	libpthread.so.0 => /opt/glibc228/lib/libpthread.so.0 (0x00007f4977834000)
	libc.so.6 => /opt/glibc228/lib/libc.so.6 (0x00007f497747d000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f4978658000)
```

没有/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28'这样的错误了。我们运行一下：

```shell
$ node 
Segmentation fault
```

怎么回事呢?我们用gdb调试一下：

```shell
$ gdb node
GNU gdb (Ubuntu 10.2-0ubuntu1~18.04~2) 10.2
Copyright (C) 2021 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from node.bak...
(gdb) r
Starting program: /home/lili/.nvm/versions/node/v21.7.0/bin/node 
warning: File "/opt/glibc228/lib/libthread_db-1.0.so" auto-loading has been declined by your `auto-load safe-path' set to "$debugdir:$datadir/auto-load".
To enable execution of this file add
	add-auto-load-safe-path /opt/glibc228/lib/libthread_db-1.0.so
line to your configuration file "/home/lili/.gdbinit".
To completely disable this security protection add
	set auto-load safe-path /
line to your configuration file "/home/lili/.gdbinit".
For more information about this security protection see the
"Auto-loading safe path" section in the GDB manual.  E.g., run from the shell:
	info "(gdb)Auto-loading safe path"
warning: Unable to find libthread_db matching inferior's thread library, thread debugging will not be available.

Program received signal SIGSEGV, Segmentation fault.
0x00007ffff7de3a80 in call_init (env=0x7fffffffd148, argv=0x7fffffffd138, argc=1, l=<optimized out>) at dl-init.c:72
72	dl-init.c: No such file or directory.
(gdb) bt
#0  0x00007ffff7de3a80 in call_init (env=0x7fffffffd148, argv=0x7fffffffd138, argc=1, l=<optimized out>) at dl-init.c:72
#1  _dl_init (main_map=0x7ffff7ffe170, argc=1, argv=0x7fffffffd138, env=0x7fffffffd148) at dl-init.c:86
#2  0x00007ffff7dd40ca in _dl_start_user () from /lib64/ld-linux-x86-64.so.2
#3  0x0000000000000001 in ?? ()
#4  0x00007fffffffd63e in ?? ()
#5  0x0000000000000000 in ?? ()
```

我们用bt看到load /lib64/ld-linux-x86-64.so.2时出错。根据[文档](https://linux.die.net/man/8/ld-linux)，这个库用于动态加载。从路径可以看出这是系统(默认)的glibc库，因此我们需要修改它。这个需要修改的就不是rpath了，而是需要用\-\-set-interpreter，如果把ld-linux-xxx.so比喻成python或者java的解释器，这个名字就很直观了。

```shell
patchelf --set-interpreter /opt/glibc228/lib/ld-linux-x86-64.so.2 node
```
再来看一下：

```shell
$ ldd node
	linux-vdso.so.1 (0x00007ffe35957000)
	libdl.so.2 => /opt/glibc228/lib/libdl.so.2 (0x00007f3932254000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f3931df4000)
	libm.so.6 => /opt/glibc228/lib/libm.so.6 (0x00007f3931a76000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f3931853000)
	libpthread.so.0 => /opt/glibc228/lib/libpthread.so.0 (0x00007f3931634000)
	libc.so.6 => /opt/glibc228/lib/libc.so.6 (0x00007f393127d000)
	/opt/glibc228/lib/ld-linux-x86-64.so.2 => /lib64/ld-linux-x86-64.so.2 (0x00007f3932458000)
```

和前面的区别就是最后一行从：

```
/lib64/ld-linux-x86-64.so.2 (0x00007f4978658000)
```

变成了：
```
/opt/glibc228/lib/ld-linux-x86-64.so.2 => /lib64/ld-linux-x86-64.so.2 (0x00007f3932458000)
```

我们再来运行一下：

```shell
$ node
error while loading shared libraries: libstdc++.so.6: cannot open shared object file: No such file or directory
```

这次是找不到libstdc++.so.6，我们看到链接的信息是/usr/lib/x86_64-linux-gnu/libstdc++.so.6，这个文件是存在的，但是因为我们的rpath设置成了/opt/glibc228，而没有路径/usr/lib/x86_64-linux-gnu/，我们可以确认这一点：

```shell
$ patchelf --print-rpath node
/opt/glibc228/lib
```

说明：libstdc++.so.6是[c++的运行时库](https://gcc.gnu.org/onlinedocs/libstdc++/)，读者可能会问，为什么我们需要编译GLIBC，而不需要更新GLIBCXX，nodejs的核心难道不是用c++编写的v8引擎吗？没错，nodejs的编译需要g++，然后运行时依赖libcstd++.so.6。但是C++对于linux来说并不是核心库(还记得linus是怎么怼c++的吗)，libcstd++是依赖于glibc的上层库。所以它跟系统的关系并不那么密切。

因此我们只需要把/usr/lib/x86_64-linux-gnu也加到rpath里就行：

```shell
patchelf --add-rpath /usr/lib/x86_64-linux-gnu/ node
```

再来一下：

```shell
$ node 
 error while loading shared libraries: libgcc_s.so.1: cannot open shared object file: No such file or directory
```

还是有问题，不过已经不是libstdc++.so.6，而是新的问题了，我们再用ldd看看：

```shell
$ ldd node 
	linux-vdso.so.1 (0x00007fffb8b6a000)
	libdl.so.2 => /opt/glibc228/lib/libdl.so.2 (0x00007ff8907ff000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007ff89039f000)
	libm.so.6 => /opt/glibc228/lib/libm.so.6 (0x00007ff890021000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007ff88fdfe000)
	libpthread.so.0 => /opt/glibc228/lib/libpthread.so.0 (0x00007ff88fbdf000)
	libc.so.6 => /opt/glibc228/lib/libc.so.6 (0x00007ff88f828000)
	/opt/glibc228/lib/ld-linux-x86-64.so.2 => /lib64/ld-linux-x86-64.so.2 (0x00007ff890a03000)
```

我们看到libgcc_s.so.1在/lib/x86_64-linux-gnu/目录下，我们还需要再加一个：

```shell
$ patchelf --add-rpath /lib/x86_64-linux-gnu/ node
```

再运行一下：

```shell
$ node
Welcome to Node.js v21.7.0.
Type ".help" for more information.
> 
```

成功！！！


