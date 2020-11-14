---
layout:     post
title:      "CentOS下编译Tensorflow" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - CentOS
    - TensorFlow
    - 编译
    - GPU
---

本文介绍怎么在CentOS 7.6下从源代码编译GPU版本的TensorFlow 1.15。

<!--more-->

**目录**
* TOC
{:toc}
 

有一台CentOS服务器，安装了CUDA 10.1和CUDNN 7.6.5，需要在上面安装TensorFlow 1.15，但是1.15的官方版本只能用于CUDA 10.0。当然我们可以在一台机器上安装多个CUDA版本来解决这个问题。但是感觉比较麻烦，所以尝试从源代码来编译。不过TensorFlow官方只支持Ubuntu，对CentOS并没有很好的支持，在安装的过程中遇到了很多问题，费了很大的力气才安装成功，所以把过程记录下来，供大家参考。

我这里CentOS的版本是7.6.1810，gcc的版本是4.8.5。

## 安装Python3.6
CentOS自动的是Python2.7，我希望build的是Python3的TensorFlow，所以首先需要安装Python3.6：
```
sudo yum update -y
sudo yum install -y python3
```
我这里安装后的版本是3.6.8。如果CentOS的版本过低，可以从源代码编译Python。

此外也需要安装Python3.6的dev包，否则会出现找不到Python.h的错误：

```
sudo yum install python3-devel
```

安装完了需要用ldconfig更新一下：
```
sudo ldconfig
```

## 创建一个virtualenv环境
编译Tensorflow时有一下Python的依赖，我们当然可以直接把他们安装到系统里，但是我个人不喜欢修改系统的东西，所以就创建了一个virtualenv的环境用于编译TensorFlow：
```
python3.6 -m venv ~/venv
source ~/venv/bin/activate
```

安装依赖：
```
pip install numpy wheel
pip install keras_preprocessing --no-deps
```

## 获取源代码
```
git clone https://github.com/tensorflow/tensorflow.git
git checkout v1.15.4
```

## 升级git
在编译的过程中出现提示说git不支持-C的选项，这是CentOS自带的git版本太多，需要升级，参考[这篇文章](https://serverfault.com/questions/709433/install-a-newer-version-of-git-on-centos-7)

```
yum install epel-release
# 卸载老版本
yum remove git
sudo yum -y install https://packages.endpoint.com/rhel/7/os/x86_64/endpoint-repo-1.7-1.x86_64.rpm
sudo yum install git
```

注：这个源慢得要死，装个git装了几十分钟，如果实在不行可以自己用源代码编译git。

最后git的版本：
```
$ git --version
git version 2.24.1
```

## 安装JDK8
系统已经安装好，读者自行搜索安装方法。

## 安装bazel

注意TensorFlow 1.15不能用高于0.26.1的版本编译，请在[这里](https://github.com/bazelbuild/bazel/releases/tag/0.26.1)下载。我现在的是bazel-0.26.1-installer-linux-x86_64.sh，安装很简单：
```
bash bazel-0.26.1-installer-linux-x86_64.sh --user
```
安装完了检查一下：

```
$ bazel version
Build label: 0.26.1
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Thu Jun 6 11:05:05 2019 (1559819105)
Build timestamp: 1559819105
Build timestamp as int: 1559819105
```

## configure
进入TensorFlow目录运行：
```
./configure
```

注意记得在是否要支持CUDA是输入y，同时检查Python的版本和CUDA/CUDNN的版本。如果前面的步骤没有问题的话，一路回车用默认值就行。

## 编译

```
bazel build --config=cuda -c opt //tensorflow/tools/pip_package:build_pip_package
```
因为需要从google下载很多东西，所以需要配置代理，请设置HTTP_PROXY和HTTPS_PROXY代理。

## 问题一

这是会出现类似如下的错误，参考[这个Issue](https://github.com/bazelbuild/bazel/issues/5164)。
```
undeclared inclusion(s) in rule '@com_google_protobuf//:protobuf_lite':
this rule is missing dependency declarations for the following files included by

'/usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h'
  '/usr/lib/gcc/x86_64-linux-gnu/5/include/stdarg.h'
  '/usr/lib/gcc/x86_64-linux-gnu/5/include/stdint.h'
```

除了这个Issue我还搜索了很多文章，包括[SO问题：How to resolve bazel “undeclared inclusion(s)” error?](https://stackoverflow.com/questions/43921911/how-to-resolve-bazel-undeclared-inclusions-error)、[SO问题：bazel “undeclared inclusion(s)” errors after updating gcc](https://stackoverflow.com/questions/48155976/bazel-undeclared-inclusions-errors-after-updating-gcc/48524741#48524741)、[bazel的Issue](https://github.com/bazelbuild/bazel/issues/5164)、[这个gist](https://gist.github.com/gentaiscool/a628fab5cd98953af7f46b69463394b3)、[Build TensorFlow from Source in Centos 7](https://todotrader.com/build-tensorflow-from-source-in-centos-7/)、[missing dependency declarations](https://github.com/tensorflow/tensorflow/issues/1157)和[The issue of compling from source code: undeclared inclusion(s) in rule ‘@nccl_archive//:nccl’](https://fantashit.com/the-issue-of-compling-from-source-code-undeclared-inclusion-s-in-rule-nccl-archive-nccl/)。

这里有一篇文章建议清除bazel cache，但并没有解决问题：
```
bazel clean --expunge
rm -rf ~/.cache/bazel
```

然后其它大部分文章都是说要在third_party/gpus/crosstool/CROSSTOOL文件中加上"/usr/lib/gcc/x86_64-linux-gnu/5/include"。但是我发现这些文章都是编译较早版本的TensorFlow(1.2)，1.15里根本没有这个文件。

尝试了无数次，终于找到了解决办法，那就是修改BUILD.tpl文件，在builtin_include_directories里增加缺失的头文件路径。
**注意：不同的系统可能不同，需要自己找到对于的路径。我这里缺的是/lib/gcc/x86_64-redhat-linux/4.8.5和/include。**

```
$ git diff
diff --git a/third_party/gpus/crosstool/BUILD.tpl b/third_party/gpus/crosstool/BUILD.tpl
index 9fe46bbe64..623b0b048b 100644
--- a/third_party/gpus/crosstool/BUILD.tpl
+++ b/third_party/gpus/crosstool/BUILD.tpl
@@ -57,7 +57,7 @@ cc_toolchain(
 cc_toolchain_config(
     name = "cc-compiler-local-config",
     cpu = "local",
-    builtin_include_directories = [%{cxx_builtin_include_directories}],
+    builtin_include_directories = [%{cxx_builtin_include_directories}, "/lib/gcc/x86_64-redhat-linux/4.8.5/include", "/include/"],
     extra_no_canonical_prefixes_flags = [%{extra_no_canonical_prefixes_flags}],
     host_compiler_path = "%{host_compiler_path}",
     host_compiler_prefix = "%{host_compiler_prefix}",

```

## 问题二

接下来有碰到unrecognized command line option '-std=c++14'的问题，搜索到[这个Issue](https://github.com/tensorflow/tensorflow/issues/32677)。说是需要较新的支持c++14的gcc，里面的讨论有人觉得很奇怪，编译TensorFlow 2.X都不需要c++14，但是编译1.15却需要。

问题找到了，接下来就要解决怎么安装新版本的gcc的问题了。CentOS7自带的gcc是4.8.5，有点太老。升级gcc是一个非常麻烦的事情，因为很多代码都依赖glibc，牵一发动全身。特别喜欢hack的可以自己编译新版本的gcc，我这里使用的是[Developer Toolset 7](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/)，安装方法为：

```
$ sudo yum install centos-release-scl

$ sudo yum install devtoolset-7

$ scl enable devtoolset-7 bash
```

但是执行了scl脚本后的gcc版本还是没变，搜索了半天找到[这篇文章](https://serverfault.com/questions/1002266/scl-enable-devtoolset-7-dosnt-do-anything)，运行一下下面的命令：
```
source /opt/rh/devtoolset-7/enable
```

这下ok了：
```
$ gcc --version
gcc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)
```

## 问题三

继续build又碰到编译bfloat16.cc的问题，搜索到[这个Issue](https://github.com/tensorflow/tensorflow/issues/40688)。

安装的numpy版本1.19太高，需要安装1.18：
```
pip uninstall numpy
pip install numpy==1.18.*
```

继续用bazel build前要记得bazel clean。

## 打包

经过漫长的编译，终于成功了！接下来就是打包成whl。

```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

在/tmp/tensorflow_pkg下就能找到编译好的tensorflow-1.15.4-cp36-cp36m-linux_x86_64.whl，用pip install就可以了。


