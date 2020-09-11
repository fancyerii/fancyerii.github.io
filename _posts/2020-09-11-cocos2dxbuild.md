---
layout:     post
title:      "Linux16.04 Build Cocos2d-x 3.x" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - Cocos2d-x
---

 

<!--more-->

**目录**
* TOC
{:toc}

## 安装步骤

### 下载

建议从[这里](https://cocos2d-x.org/download)下载稳定版本，我下载的是3.17.2。

### 安装依赖

```
sudo apt-get install g++ libgdk-pixbuf2.0-dev python-pip cmake libx11-dev libxmu-dev libglu1-mesa-dev libgl2ps-dev libxi-dev libzip-dev libpng-dev libcurl4-gnutls-dev libfontconfig1-dev libsqlite3-dev libglew-dev libssl-dev libgtk-3-dev libglfw3 libglfw3-dev xorg-dev
``` 
 

### Build

```
$ cd ~/soft/cocos2d-x-3.17.2/build
$ mkdir linux-build && cd linux-build
$ cmake ..
$ make -j4
```

但是make是碰到问题：
```
[ 18%] Building CXX object engine/cocos/core/CMakeFiles/cocos2d.dir/platform/CCGLView.cpp.o
/home/lili/soft/cocos2d-x-3.17.2/cocos/platform/desktop/CCGLViewImpl-desktop.cpp: In member function ‘virtual void cocos2d::GLViewImpl::setIcon(const std::vector<std::__cxx11::basic_string<char> >&) const’:
/home/lili/soft/cocos2d-x-3.17.2/cocos/platform/desktop/CCGLViewImpl-desktop.cpp:492:49: error: ‘glfwSetWindowIcon’ was not declared in this scope
     glfwSetWindowIcon(window, iconsCount, images);

```

上网搜索找到[这篇文章](https://github.com/cocos2d/cocos2d-x/issues/17150)和[这篇文章](https://discuss.cocos2d-x.org/t/building-on-ubuntu-16-04-issues/29051/13)，好像是用apt-get安装的是libglfw3的3.1，但是cocos2d-x需要3.2。因此需要卸载然后手动安装。先卸载：

```
$ apt-get purge libglfw3 libglfw3-dev
```

再自己安装：

```
$git clone https://github.com/glfw/glfw 8
$mkdir build
$cd build
$cmake …/glfw
$make -j4
$sudo make install
 
$cd /usr/local/lib
$sudo ln -s libglfw3.a libglfw.a
```


还有运行ldconfig让它生效：
```
sudo ldconfig
```

### 测试
接下来运行测试又失败：
```
lili@lili-Precision-7720:~/soft/cocos2d-x-3.17.2/build/linux-build/bin/cpp-tests$ ./cpp-tests 
./cpp-tests: error while loading shared libraries: /home/lili/soft/cocos2d-x-3.17.2/external/linux-specific/fmod/prebuilt/64-bit/libfmod.so.6: file too short
```

搜索发现[这篇文章](https://discuss.cocos2d-x.org/t/error-while-building-for-linux-libfmod-so-6/26553/31)。

```
$ cd lili@lili-Precision-7720:~/soft/cocos2d-x-3.17.2/external/linux-specific/fmod/prebuilt/64-bit
$ rm *.so.6
$ ln -s libfmodL.so libfmodL.so.6
$ ln -s libfmod.so libfmod.so.6
```

接下来运行./cpp-tests就没有问题了！
