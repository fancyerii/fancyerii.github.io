---
layout:     post
title:      "使用Vscode调试cuda代码" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - vscode
    - cuda
    - cmake
    - debug
---

使用vscode调试本地和远程cuda代码，使用cmake构建项目。

<!--more-->

**目录**
* TOC
{:toc}


## 本地调试

### 安装
首先需要安装[C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)、[C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)[CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)和[Nsight Visual Studio Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)等插件，如果需要CMake的语法高亮，可以安装[CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake)插件。

另外基本的c/c++环境，CUDA Toolkit当然必须安装好，cmake要求大于3.17(可以使用find_package(CUDAToolkit)，如果用老版本的需要自己查找CUDA环境，比如头文件等)。


关于在vscode里使用CMake Tools管理cmake项目的详细步骤，可以参考[Get started with CMake Tools on Linux](https://code.visualstudio.com/docs/cpp/cmake-linux)。

首先创建一个目录，并且打开vscode：

```
mkdir cudadebug
cd cudadebug
code .
```

### 创建cmake项目

Ctrl+Shift+P然后输入cmake选择"CMake: Quick Start command"，如下图所示：

<a>![](/img/vscodecuda/1.png)</a>

输入项目名称，然后选择项目类型是"Executable"。这些以后都可以在CMakeLists.txt里修改。

<a>![](/img/vscodecuda/2.png)</a>

然后它会创建CMakeLists.txt和一个main.cpp。这里要把cpp后缀改成cu，否则cmake默认会有普通的c/c++编译器而不是nvcc来编译。当然，在CMakeLists.txt的target里也要把main.cpp改成main.cu。写一个测试代码，比如：

```c++
#include <cuda.h>

#include <iostream>

#include <vector>

using namespace std;

// Add A and B vector on the GPU. Results stored into C
__global__
void addKernel(int n, float* A, float* B, float* C)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) C[i] = A[i] + B[i];
}

// Add A and B vector. Results stored into C
int add(int n, float* h_A, float* h_B, float* h_C)
{
  int size = n*sizeof(float);

  // Allocate memory on device and copy data
  float* d_A;
  cudaMalloc((void**)&d_A, size);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  float* d_B;
  cudaMalloc((void**)&d_B, size);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  float* d_C;
  cudaMalloc((void**)&d_C, size);

  // launch Kernel
  cout << "Running 256 threads on " << ceil(n/256.0f) << " blocks -> " << 256*ceil(n/256.0f) << endl;
  addKernel<<<ceil(n/256.0f),256>>>(n, d_A, d_B, d_C);

  // Transfer results back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}

// C = A + B on a GPU, where A is a vector of 1.0f and B a vector of 2.0f
// The main function takes one argument, the size of the vectors
int main(int argc, char* argv[])
{
  int n = atoi(argv[1]);

  vector<float> h_A(n, 1.0f);
  vector<float> h_B(n, 2.0f);
  vector<float> h_C(n);

  add(n, h_A.data(), h_B.data(), h_C.data());

  for(auto& c : h_C) {
    if(fabs(c-3.0f) > 0.00001f) {
      cout << "Error!" << endl;
      return 1;
    }
  }

  cout << "The program completed successfully" << endl;

  return 0;
}
```

### 修改CMakeLists.txt

为了让cmake支持CUDA，需要增加一些内容，下面是完整的CMakeLists.txt。

```cmake
cmake_minimum_required(VERSION 3.17.0)
project(testdebug VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)


set(CMAKE_CUDA_STANDARD 11)


add_executable(testdebug hello.cu)
target_link_libraries(testdebug PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(testdebug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(testdebug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

首先当然是语言上要增加CUDA。其次需要使用find_package(CUDAToolkit)找到CUDA工具。找到之后就可以用target_link_libraries把CUDA::cudart加入依赖，这样我们的代码就能正确的找到头文件和链接到库文件了。

另外我们需要调试cuda代码，默认的cmake的在Debug模式时(-DCMAKE_BUILD_TYPE=Debug)，它会根据编译器加上调试信息，但是cuda的调试选项需要我们自己加，因此我们需要根据CMAKE_BUILD_TYPE是否Debug来增加-G选项，这可以使用target_compile_options来设置(不要再用老式的cmake_cxx_flags了)。另外我们只是在编译cuda文件时使用这个选项，因此可以用生成表达式\$\<\$\<COMPILE_LANGUAGE:CUDA>:-G>来在编译cu文件时输出-G选项。对cmake不熟悉的读者可以参考[《Professional CMake》学习](/2023/12/26/procmake/)。

### 选择kit

Ctrl+Shift+P然后选择"Select a Kit."，根据自己的环境选择合适的c/c++编译器。比如我们这里选择gcc 8.4：

<a>![](/img/vscodecuda/3.png)</a>

如果找不到你想要的编译器，可以在修改cmake-tools-kits.json。

### 选择一个variant

还是Ctrl+Shift+P然后选择"CMake: Select Variant"。

<a>![](/img/vscodecuda/4.png)</a>

我们这里选择Debug。

<a>![](/img/vscodecuda/5.png)</a>

### 配置

Ctrl+Shift+P然后选择"CMake: Configure"，它会执行类似"cmake -Bbuild ."的命令

### Build

Ctrl+Shift+P然后选择"CMake: Build"，或者选择左下角的Build按钮，它会执行类似"cmake \-\-build build"的命令：

<a>![](/img/vscodecuda/6.png)</a> 

默认build的是all这个target，我们当然也可以选择某个target进行build，这等价于于"cmake \-\-build build --target ..."

<a>![](/img/vscodecuda/7.png)</a>

### 调试

在代码里增加一个断点，然后Ctrl+Shift+P运行"CMake: Debug"。如下图：

<a>![](/img/vscodecuda/8.png)</a>

### 调试cuda代码

如果我们在下图的位置增加断点，我们会发现它会自动的下移到代码结束的部分，也就是没法加断点。

<a>![](/img/vscodecuda/9.png)</a>

原因是默认的cmake使用gdb进行调试，当然它无法调试cuda代码。我们需要打开settings.json：

```
    "cmake.debugConfig": {
        "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb",
    },
```

### 命令行参数

我们如果直接调试代码会crash，因为我们没有传入命令行参数。如果是自己运行，当然很简单。要在vscode里传入cmake debug的参数，同样需要修改settings.json：

```
    "cmake.debugConfig": {
        "args": [
            "10000"
        ],
        "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb",
    }
```

最终我们可以点击调试按钮进行调试，效果如下图：

<a>![](/img/vscodecuda/10.png)</a>

## 远程ssh调试

这个和本地调试差不多，只不过是ssh到远程机器创建编辑和运行代码，和上面差不多。关于ssh调试，读者可以参考[VSCode远程调试Python](/2023/09/25/py-remote-debug/)。

## 使用gdb-server远程调试

这个和ssh的区别是：ssh调试是在服务器上启动普通的gdb/cuda-gdb，我们只不过通过ssh控制这个gdb而已。而gdb-server在服务器上通过cuda-gdb-server启动一个gdb-server，这个gdb-server监听在tcp的端口上，然后我们可以在本地用gdb或者其它ide(vscode/clion的调试器，八成也是gdb上做个ui)通过tcp连接上gdb-server。我个人理解和ssh原理差不多。可能对于无法ssh的情况下适用，比如服服务器不能ssh登录，但是可以在上面运行程序并且开放某个tcp端口。一般公司不会这么进行限制，也许对于运行在docker里的程序适合这样调试？但是docker里也可以装个sshd啊。除非在没有vscode的年代，那时候没有ssh调试，也许只能gdb-server。gdb-server要求在本地和服务器上都有相同的可执行程序(这个有时候是不可能的，比如用windows调试linux，或者不同的linux版本/发行版)。

我这里就没有测试了，感兴趣的读者可以参考[Getting Started with the CUDA Debugger](https://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html#debugging-cuda-application-remote)



