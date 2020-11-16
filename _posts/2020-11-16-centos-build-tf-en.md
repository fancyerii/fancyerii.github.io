---
layout:     post
title:      "Building Tensorflow 1.15 on CentOS 7 from Source" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - CentOS
    - TensorFlow
    - Compile
    - GPU
---

This article shows the steps and problems of building Tensorflow 1.15 on CentOS 7.

<!--more-->

**Content**
* TOC
{:toc}
 

I have a CentOS machine with CUDA 10.1 and CUDNN 7.6.5. I want to install TensorFlow 1.15, but offical pip release only comes with CUDA 10.0 support. Of course I can install multiple CUDA versions to resolve this problem. But I don't like to manage that much versions and like to build it from source. Building from source on CentOS is not officically supported as Ubuntu. I have encountered so many problems that I think it's neccessary to take a note of how I solved them.

The version of CentOS is 7.6.1810 and the version of gcc is 4.8.5. If your version is different from mine, you may ask for helps from google for certain problems.


## Install  Python 3.6
CentOS 7 ships with Python2.7，But I want use python3 with TensorFlow, so I need to install Python 3.6 first.

```
sudo yum update -y
sudo yum install -y python3
```

CentOS 7.6 ships with Python 3.6.8. If you have a older CentOs verison, Compiling Python 3.6 from source or googling some other methods to install it.


We also need to install development lib of Python 3.6, or else you will see erros like "Python.h not found".


```
sudo yum install python3-devel
```

We need to inform the changes to shell with ldconfig.

```
sudo ldconfig
```

## Create a virtualenv environment

TensorFlow depends on some python packages. We can certainly install them directly. But I don't like to install them to system that may affect all user, so I created a virtualenv environment to install the packages TensorFlow needs.


First create virtualenv environment and activate it.
```
python3.6 -m venv ~/venv
source ~/venv/bin/activate
```

Then install some packages.

```
pip install numpy wheel
pip install keras_preprocessing --no-deps
```

## Get source codes
```
git clone https://github.com/tensorflow/tensorflow.git
git checkout v1.15.4
```

## Upgrade git

When compiling, it may complain git can't recognize -C options. That's because the old git version of CentOS and we need to upgrade it. See [this article](https://serverfault.com/questions/709433/install-a-newer-version-of-git-on-centos-7) for more detials.


```
yum install epel-release
# uninstall old ones
yum remove git
sudo yum -y install https://packages.endpoint.com/rhel/7/os/x86_64/endpoint-repo-1.7-1.x86_64.rpm
sudo yum install git
```

Warning: This rpm source is very very slow. It cost me dozens of minutes. If you are not that patient, you may compile git from source yourself.

Here is my new git version:
```
$ git --version
git version 2.24.1
```

## Install JDK8
My system administrator has already installed for me. So if you don't have jdk 8, you should install it.

## Install bazel

TensorFlow 1.15 can't build with bazel newer than verion 0.26.1. Please download it from [here](https://github.com/bazelbuild/bazel/releases/tag/0.26.1). My downloaded file name is bazel-0.26.1-installer-linux-x86_64.sh. And it's very simple to install.

```
bash bazel-0.26.1-installer-linux-x86_64.sh --user
```
We need to check it after installation.

```
$ bazel version
Build label: 0.26.1
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Thu Jun 6 11:05:05 2019 (1559819105)
Build timestamp: 1559819105
Build timestamp as int: 1559819105
```

## configure
Go to the TensorFlow root directory and run:
```
./configure
```

You must type "y" when asked "Do you wish to build TensorFlow with CUDA support?". If setting properly, we should hit enter key to use default values most of the time. Take care to check whether the confiure scripts detect the correct CUDA and CUDNN.

## Compile

```
bazel build --config=cuda -c opt //tensorflow/tools/pip_package:build_pip_package
```

## Problem One

This problem can be found in many articles, such as [this issue](https://github.com/bazelbuild/bazel/issues/5164).

```
undeclared inclusion(s) in rule '@com_google_protobuf//:protobuf_lite':
this rule is missing dependency declarations for the following files included by

'/usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h'
  '/usr/lib/gcc/x86_64-linux-gnu/5/include/stdarg.h'
  '/usr/lib/gcc/x86_64-linux-gnu/5/include/stdint.h'
```

To solve this problem, I googled many articles including[SO post：How to resolve bazel “undeclared inclusion(s)” error?](https://stackoverflow.com/questions/43921911/how-to-resolve-bazel-undeclared-inclusions-error), [SO post：bazel “undeclared inclusion(s)” errors after updating gcc](https://stackoverflow.com/questions/48155976/bazel-undeclared-inclusions-errors-after-updating-gcc/48524741#48524741), [bazel github issue](https://github.com/bazelbuild/bazel/issues/5164), [this gist](https://gist.github.com/gentaiscool/a628fab5cd98953af7f46b69463394b3), [Build TensorFlow from Source in Centos 7](https://todotrader.com/build-tensorflow-from-source-in-centos-7/), [missing dependency declarations](https://github.com/tensorflow/tensorflow/issues/1157) and [The issue of compling from source code: undeclared inclusion(s) in rule ‘@nccl_archive//:nccl’](https://fantashit.com/the-issue-of-compling-from-source-code-undeclared-inclusion-s-in-rule-nccl-archive-nccl/).

One of them suggested to clear bazel cache. I tried but no luck.

```
bazel clean --expunge
rm -rf ~/.cache/bazel
```

Most of these articles suggest to add inlude path of "/usr/lib/gcc/x86_64-linux-gnu/5/include" to the third_party/gpus/crosstool/CROSSTOOL file. I found all these articles are building old versions, such as 1.2, of TensorFlow, but there not exists this file in version 1.15.

After several tries and failures, I found a solution. We can modify BUILD.tpl and add missing including header files in builtin_include_directories.

 
**Note: If you use a CentOS version different from mine, you may seek your own paths. For mine, they are /lib/gcc/x86_64-redhat-linux/4.8.5 and /include.**

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

## Problem Two

bazel script complains "unrecognized command line option '-std=c++14'". And I found [this issue](https://github.com/tensorflow/tensorflow/issues/32677). It says we should use newer gcc version supporting c++14. It's very strange that version 1.14 need c++14 while version 2.X don't.


Then we need to install a newer gcc version. CentOS 7 ships with gcc 4.8.5 that's a little bit stale. But upgrading gcc is a very complicated thing that so many softwares depend on it. For those liking to hack, they can build and upgrade gcc from source. I don't like bother myself with it so I use [Developer Toolset 7](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/) to install another gcc for me.


```
$ sudo yum install centos-release-scl

$ sudo yum install devtoolset-7

$ scl enable devtoolset-7 bash
```

The gcc version remains the same after I ran the scl command. I googled and found [this article](https://serverfault.com/questions/1002266/scl-enable-devtoolset-7-dosnt-do-anything) and solved it by this command:

```
source /opt/rh/devtoolset-7/enable
```

It works!
```
$ gcc --version
gcc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)
```

## Problem Three

Another problem when compling bfloat16.cc occured. And it's the same as [this issue](https://github.com/tensorflow/tensorflow/issues/40688).

The reason is TensorFlow 1.15 is not compatible with numpy 1.19, and we should downgrade it:

```
pip uninstall numpy
pip install numpy==1.18.*
```

Remember to run "bazel clean" after changing numpy version.

## Package

After long time of compilation, it succeeded. Now it's time for package them to whl format.

```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

You should find something like tensorflow-1.15.4-cp36-cp36m-linux_x86_64.whl in /tmp/tensorflow_pkg. We then can easily install it by pip.
