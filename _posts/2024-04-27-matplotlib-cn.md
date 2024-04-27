---
layout:     post
title:      "Matplotlib支持中文" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - python
    - matplotlib 
---



<!--more-->

**目录**
* TOC
{:toc}

## 问题

matplotlib默认无法支持中文，搜索了一下，参考[这个issue](https://github.com/ygl-rg/matplotlib-notes/issues/1)和[这篇文章](https://hoishing.medium.com/using-chinese-characters-in-matplotlib-5c49dbb6a2f7)，解决方法如下。

## 解决方法

### 下载字体

下载simhei.ttf。

### 定位matplotlib的ttf

```python
import matplotlib
print(matplotlib.matplotlib_fname())
```

比如我的位置是venv/lib/python3.10/site-packages/matplotlib/mpl-data/matplotlibrc。

### 复制字体文件

```shell
cp simhei.ttf venv/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/
```

### 清除缓存

```shell
rm ~/.cache/matplotlib
```

### 使用字体

```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False
```


