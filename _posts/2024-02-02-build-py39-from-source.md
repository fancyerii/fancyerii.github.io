---
layout:     post
title:      "Ubuntu18.04从源代码安装Python3.9"
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

本文解释在Ubuntu18.04下从源代码build安装Python3.9的方法。

 <!--more-->

**目录**
* TOC
{:toc} 

Ubuntu18.04通过apt能够安装最新的python版本是3.8，但是我们有的时候需要更高版本的python，比如3.9。一种方法是通过conda安装，但是有的时候我们没有安装conda，那么我们可以通过源代码编译安装python(我们这里当然说的是cpython)。

## 安装依赖

```
sudo apt-get update
sudo apt-get install gdebi-core
```

```
sudo apt-get install \
    curl \
    gcc \
    libbz2-dev \
    libev-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    make \
    tk-dev \
    wget \
    zlib1g-dev
```

还需要安装uuid-dev libgdbm-compat-dev，否则会出现：
```
sudo apt-get install uuid-dev libgdbm-compat-dev
```

```
The necessary bits to build these optional modules were not found:
_dbm                  _uuid
```

## 下载

```
export PYTHON_VERSION=3.9.18
export PYTHON_MAJOR=3
curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar -xvzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

``` 

## build

```
./configure --enable-optimizations 
make
sudo make install
```


## 补充：编译python3.12

会提示:
```The necessary bits to build these optional modules were not found:
_tkinter 
```

```shell
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev python-tk python3-tk tk-dev
```

根据[这个issue](https://github.com/python/cpython/issues/98973):

```
TCLTK_LIBS="-ltk8.6 -ltkstub8.6 -ltcl8.6" TCLTK_CFLAGS="-I/usr/include/tcl8.6" ./configure --enable-optimizations 
```

另外我们在configuration之后应该看一下config.log：我们会发现类似：

```
checking for stdlib extension module _tkinter
 missing
```

如果我们需要这些扩展模块，我们在make之前就应该修复这些问题。我仔细检查了一下，还有两个扩展模块missing：

```
  360 configure:7166: checking for --enable-wasm-dynamic-linking
  361 configure:7188: result: missing
  362 configure:7191: checking for --enable-wasm-pthreads
  363 configure:7213: result: missing
```

我搜索了一下，这两个扩展是python3.11引入的，跟[WebAssembly Dynamic Linking](https://medium.com/@arora.aashish/webassembly-dynamic-linking-1644c9f40c8c)相关，关于configure的更多信息请参考[官方文档](https://docs.python.org/3/using/configure.html)。我目前应该不会用到，所以就不解决了。


