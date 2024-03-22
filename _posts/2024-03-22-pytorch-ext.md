---
layout:     post
title:      "翻译：Custom C++ and CUDA Extensions" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - pytorch
    - c++
    - extension 
---

本文翻译[Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)。

<!--more-->

**目录**
* TOC
{:toc}


PyTorch提供了大量与神经网络、任意张量代数、数据处理等相关的操作。然而，你可能仍然需要更加定制化的操作。例如，你可能想要使用一种你在论文中发现的新型激活函数，或者实现你在研究中开发的操作。

在PyTorch中集成这样一个自定义操作的最简单方式是通过扩展Function和Module来用Python编写，如[EXTENDING PYTORCH](https://pytorch.org/docs/master/notes/extending.html)所述。这样做可以让你充分利用自动求导的功能（避免编写导数函数），同时还保留了Python的常规表达能力。然而，有时候你的操作最好用C++来实现。例如，你的代码可能需要非常快速，因为它在模型中被频繁调用或者即使只调用几次也非常昂贵。另一个可能的原因是它依赖于或与其他C或C++库进行交互。为了解决这些情况，PyTorch提供了一种非常简单的编写自定义C++扩展的方法。

C++扩展是我们开发的一种机制，允许用户（你）创建PyTorch操作符，这些操作符定义在PyTorch后端之外，即与PyTorch的后端分离。这种方法与原生PyTorch操作的实现方式不同。C++扩展旨在减少与将操作集成到PyTorch后端相关的样板代码，并为基于PyTorch的项目提供高度灵活性。然而，一旦你将操作定义为C++扩展，将其转换为原生PyTorch函数在很大程度上取决于代码组织，如果你决定向上游贡献你的操作，则可以在之后解决这个问题。

## 动机和示例

本文其余部分将通过一个实际的示例来介绍编写和使用C++（以及CUDA）扩展的过程。如果你被追赶，或者如果你不在今天结束前完成该操作就会被解雇，你可以跳过这一部分，直接进入下一节中的实现细节。

假设你想到了一种新型的循环单元，发现它与现有技术相比具有更好的性能。这个循环单元类似于LSTM，但不同之处在于它没有遗忘门，而是使用指数线性单元（ELU）作为内部激活函数。因为这个单元永远不会遗忘，我们将其称为LLTM，即长长期记忆单元。

LLTM与普通的LSTM有两个显著的不同之处，这使得我们无法配置PyTorch的LSTMCell来满足我们的需求，因此我们必须创建一个自定义单元。这种情况下最简单的方法 - 也很可能是所有情况下的第一步 - 是在纯PyTorch中用Python实现我们想要的功能。为此，我们需要子类化torch.nn.Module并实现LLTM的前向传播。代码大致如下：

```
class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell
```

我们可以按照预期使用这个方法：

```python
import torch

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

new_h, new_C = rnn(X, (h, C))
```


自然地，如果可能和可行的话，你应该使用这种方法来扩展PyTorch。由于PyTorch针对CPU和GPU高度优化了其操作的实现，这些实现由诸如[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)、[Intel MKL](https://software.intel.com/en-us/mkl)或[NNPACK](https://github.com/Maratyszcza/NNPACK)等库提供支持，像上面的PyTorch代码通常足够快。然而，在某些情况下，我们也可以看到为什么还有进一步提高性能的空间。最明显的原因是PyTorch对你正在实现的算法一无所知。它只知道你用来组合算法的各个操作。因此，PyTorch必须逐个执行你的操作。由于每次对操作的实现（或内核）进行单独调用（可能涉及启动CUDA内核）都会有一定量的开销，这个开销可能在许多函数调用中变得显著。此外，运行我们代码的Python解释器本身也可能减慢程序的运行速度。

因此，加快速度的一种明确方法是在C++（或CUDA）中重写部分代码，并将特定组的操作融合起来。融合意味着将许多函数的实现组合成一个函数，从而减少内核启动以及我们可以通过增加对全局数据流的可见性来执行的其他优化的数量。

让我们看看如何使用C++扩展来实现LLTM的融合版本。我们将首先用纯C++编写它，使用支持PyTorch后端大部分功能的[ATen库](https://github.com/zdevito/ATen)，并看看它是如何轻松地让我们转换我们的Python代码的。然后，我们将通过将模型的部分移至CUDA内核来进一步加速，以便充分利用GPU提供的大规模并行性。

## 编写C++扩展

C++扩展有两种形式：它们可以通过setuptools“提前构建”，或者通过torch.utils.cpp_extension.load()“即时构建”。我们将从第一种方法开始，并稍后讨论后者。

### 使用setuptools进行构建

对于“提前构建”版本，我们通过编写一个setup.py脚本，使用setuptools来编译我们的C++代码来构建我们的C++扩展。对于LLTM来说，它看起来就像这样简单：

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```

在这段代码中，CppExtension是setuptools.Extension的一个便利包装器，它传递了正确的包含路径并将扩展的语言设置为C++。等效的原始setuptools代码将简单地是：

```python
Extension(
   name='lltm_cpp',
   sources=['lltm.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')
```


BuildExtension执行了许多必要的配置步骤和检查，并在混合C++/CUDA扩展的情况下管理混合编译。这就是我们目前需要了解有关构建C++扩展的全部内容！现在让我们来看看我们的C++扩展的实现，它位于lltm.cpp中。

### 编写C++操作符

让我们开始用C++实现LLTM！我们在反向传播中需要的一个函数是Sigmoid的导数。这段代码足够小，可以讨论在编写C++扩展时我们可以使用的整体环境：

```python
#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}
```

\<torch/extension.h> 是一个一站式头文件，包含了编写C++扩展所需的所有PyTorch组件。它包括：

* ATen库，这是我们进行张量计算的主要API，
* [pybind11](https://github.com/pybind/pybind11)，这是我们为C++代码创建Python绑定的方式，
* 管理ATen和pybind11之间交互细节的头文件。

d_sigmoid()的实现展示了如何使用ATen API。PyTorch的张量和变量接口是从ATen库自动生成的，因此我们几乎可以将我们的Python实现直接转换为C++。我们所有计算的主要数据类型将是torch::Tensor。可以在[这里](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html)查看它的完整API。还要注意，我们可以包含 \<iostream> 或任何其他C或C++头文件 - 我们完全可以使用C++11的全部功能。

需要注意的是，CUDA-11.5 nvcc在Windows上解析torch/extension.h时会遇到内部编译器错误。为了解决这个问题，将Python绑定逻辑移到纯C++文件中。示例用法如下：

```cpp
#include <ATen/ATen.h>
at::Tensor SigmoidAlphaBlendForwardCuda(....)
```

而不是：

```cpp
#include <torch/extension.h>
torch::Tensor SigmoidAlphaBlendForwardCuda(...)
```

目前在[这里](https://github.com/pytorch/pytorch/issues/69460)有一个nvcc错误的问题。完整的解决方法代码示例在[这里](https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48)。

#### 前向计算

接下来我们可以将整个前向传播迁移到C++中：

```cpp
#include <vector>

std::vector<at::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}
```

#### 反向传播

目前C++扩展API不提供自动生成反向函数的方式。因此，我们还必须实现LLTM的反向传播，该传播计算了损失对于前向传播每个输入的导数。最终，我们将前向和反向函数都放入torch.autograd.Function中，以创建一个良好的Python绑定。反向函数稍微复杂一些，所以我们不会深入研究代码（如果你有兴趣，Alex Graves的[论文](https://www.cs.toronto.edu/~graves/phd.pdf)是更多信息的良好阅读材料）：

```cpp
// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}
```


### 绑定到Python

一旦你用C++和ATen编写好了你的操作，你可以使用pybind11以非常简单的方式将你的C++函数或类绑定到Python中。关于PyTorch C++扩展的这部分问题或问题大部分都可以在[pybind11文档](https://pybind11.readthedocs.io/en/stable/)中找到答案。

对于我们的扩展，必要的绑定代码仅涉及四行：

```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}
```

这里要注意的一点是宏TORCH_EXTENSION_NAME。torch扩展构建将其定义为你在setup.py脚本中给你的扩展取的名字。在这种情况下，TORCH_EXTENSION_NAME的值将是“lltm_cpp”。这是为了避免在两个地方（构建脚本和你的C++代码）中维护扩展的名称，因为两者之间的不匹配可能导致难以跟踪的问题。

### 使用你的扩展
现在我们可以在PyTorch中导入我们的扩展。此时，你的目录结构可能看起来像这样：


```
pytorch/
  lltm-extension/
    lltm.cpp
    setup.py
```
    
现在运行"python setup.py install"来构建和安装你的扩展。这应该看起来像这样：

```shell
running install
running bdist_egg
running egg_info
creating lltm_cpp.egg-info
writing lltm_cpp.egg-info/PKG-INFO
writing dependency_links to lltm_cpp.egg-info/dependency_links.txt
writing top-level names to lltm_cpp.egg-info/top_level.txt
writing manifest file 'lltm_cpp.egg-info/SOURCES.txt'
reading manifest file 'lltm_cpp.egg-info/SOURCES.txt'
writing manifest file 'lltm_cpp.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_ext
building 'lltm_cpp' extension
creating build
creating build/temp.linux-x86_64-3.7
gcc -pthread -B ~/local/miniconda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I~/local/miniconda/lib/python3.7/site-packages/torch/include -I~/local/miniconda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -I~/local/miniconda/lib/python3.7/site-packages/torch/include/TH -I~/local/miniconda/lib/python3.7/site-packages/torch/include/THC -I~/local/miniconda/include/python3.7m -c lltm.cpp -o build/temp.linux-x86_64-3.7/lltm.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=lltm_cpp -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
creating build/lib.linux-x86_64-3.7
g++ -pthread -shared -B ~/local/miniconda/compiler_compat -L~/local/miniconda/lib -Wl,-rpath=~/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/lltm.o -o build/lib.linux-x86_64-3.7/lltm_cpp.cpython-37m-x86_64-linux-gnu.so
creating build/bdist.linux-x86_64
creating build/bdist.linux-x86_64/egg
copying build/lib.linux-x86_64-3.7/lltm_cpp.cpython-37m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
creating stub loader for lltm_cpp.cpython-37m-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/lltm_cpp.py to lltm_cpp.cpython-37.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
__pycache__.lltm_cpp.cpython-37: module references __file__
creating 'dist/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
removing '~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg' (and everything under it)
creating ~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
Extracting lltm_cpp-0.0.0-py3.7-linux-x86_64.egg to ~/local/miniconda/lib/python3.7/site-packages
lltm-cpp 0.0.0 is already the active version in easy-install.pth

Installed ~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
Processing dependencies for lltm-cpp==0.0.0
Finished processing dependencies for lltm-cpp==0.0.0
```

编译器的一个小提示：由于ABI版本问题，你用来构建C++扩展的编译器必须与PyTorch构建时使用的编译器兼容。实际上，这意味着在Linux上你必须使用GCC版本4.9及以上。对于Ubuntu 16.04和其他更近期的Linux发行版，这应该已经是默认的编译器了。在MacOS上，你必须使用clang（它没有任何ABI版本问题）。在最坏的情况下，你可以用你的编译器从源代码构建PyTorch，然后用相同的编译器构建扩展。

一旦你的扩展构建好了，你可以在Python中简单地导入它，使用你在setup.py脚本中指定的名称。只需确保首先导入torch，因为这会解析一些动态链接器必须看到的符号：

```
In [1]: import torch
In [2]: import lltm_cpp
In [3]: lltm_cpp.forward
Out[3]: <function lltm.PyCapsule.forward>
```


如果我们对函数或模块调用help()，我们可以看到它的签名与我们的C++代码相匹配：

```
In[4] help(lltm_cpp.forward)
forward(...) method of builtins.PyCapsule instance
    forward(arg0: torch::Tensor, arg1: torch::Tensor, arg2: torch::Tensor, arg3: torch::Tensor, arg4: torch::Tensor) -> List[torch::Tensor]

    LLTM forward
```


现在我们能够从Python中调用我们的C++函数，我们可以用torch.autograd.Function和torch.nn.Module将它们包装起来，使它们成为PyTorch的一等公民：

```python
import math
import torch

# Our module!
import lltm_cpp

class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
```


#### 性能比较

既然我们能够在PyTorch中使用和调用我们的C++代码，我们可以进行一个小型基准测试，看看我们在将操作重写为C++后性能提升了多少。我们将运行LLTM的前向和后向几次，并测量持续时间：

```python
import time

import torch

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))
```


如果我们在此帖子开头用纯Python编写的原始LLTM运行此代码，我们会得到以下数字（在我的机器上）：

```
Forward: 506.480 us | Backward 444.694 us
```

使用我们的新的C++版本：

```
Forward: 349.335 us | Backward 443.523 us
```

我们已经可以看到前向函数的显著加速（超过30%）。对于反向函数，虽然没有显著的加速，但也可以看到一定程度的加速。我上面写的反向传播并没有特别优化，肯定还有改进的空间。此外，PyTorch的自动微分引擎可以自动并行化计算图，可能使用更高效的操作流程，并且也是用C++实现的，因此预计速度会很快。尽管如此，这是一个很好的开始。

#### GPU设备上的性能

关于PyTorch的ATen后端的一个很棒的事实是它抽象了您正在运行的计算设备。这意味着我们为CPU编写的相同代码也可以在GPU上运行，并且各个操作将相应地分派到经过GPU优化的实现上。对于某些操作，比如矩阵乘法（比如mm或addmm），这是一个很大的优势。让我们看看使用CUDA张量运行我们的C++代码能获得多少性能提升。我们的实现不需要进行任何更改，我们只需要从Python中将张量放入GPU内存，可以在创建时添加device=cuda_device参数，也可以在创建后使用.to(cuda_device)：

```python
import torch

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 16
input_features = 32
state_size = 128

# Note the device=cuda_device arguments here
X = torch.randn(batch_size, input_features, device=cuda_device)
h = torch.randn(batch_size, state_size, device=cuda_device)
C = torch.randn(batch_size, state_size, device=cuda_device)

rnn = LLTM(input_features, state_size).to(cuda_device)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    torch.cuda.synchronize()
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    torch.cuda.synchronize()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))
```


再次比较我们的纯PyTorch代码和我们的C++版本，现在两者都在CUDA设备上运行，我们再次看到性能提升。对于Python/PyTorch：

```
Forward: 187.719 us | Backward 410.815 us
```

而对于C++/ATen：

```
Forward: 149.802 us | Backward 393.458 us
```


与非CUDA代码相比，这是一个非常好的整体加速。然而，我们可以通过编写自定义CUDA内核从我们的C++代码中获得更多性能，我们将很快深入讨论这个问题。在此之前，让我们讨论另一种构建C++扩展的方法。


## JIT编译扩展
之前，我提到了构建C++扩展的两种方式：使用setuptools或即时（JIT）。在介绍了前者之后，让我们详细介绍一下后者。即时编译机制通过调用PyTorch API中的一个简单函数torch.utils.cpp_extension.load()来为您提供一种动态编译和加载扩展的方式。对于LLTM来说，这将非常简单，就像这样：

```python
from torch.utils.cpp_extension import load

lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])
```

在这里，我们向函数提供与setuptools相同的信息。在后台，这将执行以下操作：

* 创建临时目录 /tmp/torch_extensions/lltm，
* 将一个[Ninja](https://ninja-build.org/)构建文件输出到该临时目录，
* 编译您的源文件为一个共享库，
* 将这个共享库导入为一个Python模块。

实际上，如果将verbose=True传递给cpp_extension.load()，您将会得到关于这个过程的信息：

```
Using /tmp/torch_extensions as PyTorch extensions root...
Emitting ninja build file /tmp/torch_extensions/lltm_cpp/build.ninja...
Building extension module lltm_cpp...
Loading extension module lltm_cpp...
```

生成的Python模块与使用setuptools生成的完全相同，但是消除了必须维护单独的setup.py构建文件的要求。如果您的设置更加复杂并且确实需要setuptools的全部功能，您可以编写自己的setup.py - 但在许多情况下，这种即时编译的技术就足够了。第一次运行这行代码时，它会花费一些时间，因为扩展正在后台编译。由于我们使用Ninja构建系统来构建您的源文件，所以重新编译是增量式的，因此当您第二次运行Python模块时重新加载扩展会很快，并且如果您没有更改扩展的源文件，开销很低。


## 编写混合的C++/CUDA扩展

为了真正将我们的实现推向下一个水平，我们可以手写我们前向传播和反向传播的部分，并使用自定义的CUDA核心。对于LLTM来说，这有可能特别有效，因为有大量的顺序点操作，可以全部融合并并行化在一个CUDA核心中。让我们看看如何编写这样一个CUDA核心，并使用这个扩展机制将其与PyTorch集成。

编写CUDA扩展的一般策略是首先编写一个C++文件，该文件定义将从Python中调用的函数，并使用pybind11将这些函数绑定到Python。此外，该文件还将声明在CUDA（.cu）文件中定义的函数。然后，C++函数将进行一些检查，并最终将其调用转发到CUDA函数。在CUDA文件中，我们编写我们实际的CUDA核心。cpp_extension包将负责使用C++编译器（如gcc）编译C++源文件和使用NVIDIA的nvcc编译器编译CUDA源文件。这确保每个编译器负责编译其最擅长的文件。最终，它们将被链接成一个共享库，可以从Python代码中访问。

我们将从C++文件开始，我们将其命名为lltm_cuda.cpp，例如：

```cpp
#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);

  return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return lltm_cuda_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
  m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
```

如您所见，它主要是样板代码，检查和转发到我们将在CUDA文件中定义的函数。我们将命名此文件lltm_cuda_kernel.cu（注意扩展名为.cu！）。NVCC可以合理地编译C++11，因此我们仍然可以使用ATen和C++标准库（但不包括torch.h）。请注意，setuptools无法处理具有相同名称但扩展名不同的文件，因此如果您使用setup.py方法而不是JIT方法，则必须为CUDA文件提供与C++文件不同的名称（对于JIT方法，lltm.cpp和lltm.cu将很好地工作）。让我们稍微看一下这个文件会是什么样子：

```cpp
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}
```


这里我们看到了我刚刚描述的头文件，以及我们正在使用CUDA特定的声明如\_\_device\_\_和\_\_forceinline\_\_以及函数如exp。让我们继续写几个我们需要的帮助函数：

```cpp
template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}
```

现在要实际实现一个函数，我们再次需要两件事：一个执行我们不希望手工明确编写的操作并调用CUDA核心的函数，然后是我们想要加速的部分的实际CUDA核心。对于前向传播，第一个函数应该像这样：

```cpp
std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gates = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.data<scalar_t>(),
        old_cell.data<scalar_t>(),
        new_h.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        state_size);
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}
```

这里主要感兴趣的是AT_DISPATCH_FLOATING_TYPES宏和内核启动（由\<\<\<...>>>指示）。虽然ATen抽象了我们处理的张量的设备和数据类型，但是在运行时，张量仍然由具体类型的内存支持在具体设备上。因此，我们需要一种在运行时确定张量类型然后有选择地调用具有相应正确类型签名的函数的方法。手动完成这个过程将（概念上）看起来像这样：

```cpp
switch (tensor.type().scalarType()) {
  case torch::ScalarType::Double:
    return function<double>(tensor.data<double>());
  case torch::ScalarType::Float:
    return function<float>(tensor.data<float>());
  ...
}
```

AT_DISPATCH_FLOATING_TYPES的目的是为我们处理此分派。它接受一个类型（在我们的例子中是gates.type()），一个名称（用于错误消息）和一个lambda函数。在这个lambda函数内部，类型别名scalar_t可用，并且定义为张量在该上下文中运行时实际的类型。因此，如果我们有一个模板函数（我们的CUDA核心将是这样的），我们可以使用这个scalar_t别名实例化它，并且将调用正确的函数。在这种情况下，我们还希望将张量的数据指针作为scalar_t类型的指针来检索。如果您想要在所有类型上而不仅仅是浮点类型（Float和Double）上分派，您可以使用AT_DISPATCH_ALL_TYPES。

请注意，我们使用普通的ATen执行一些操作。这些操作仍将在GPU上运行，但使用ATen的默认实现。这是有道理的，因为ATen将使用高度优化的例程来进行矩阵乘法（例如addmm）或卷积等操作，这些例程要比自己实现和改进要困难得多。

至于内核启动本身，我们在这里指定每个CUDA块将有1024个线程，并且整个GPU网格被分割为尽可能多的1 x 1024线程块，以便用一个线程来填充我们的矩阵中的每个组件。例如，如果我们的状态大小为2048，批量大小为4，我们将启动总共4 x 2 = 8个块，每个块有1024个线程。如果您以前从未听说过CUDA“块”或“网格”，那么可以阅读关于[CUDA的简介](https://devblogs.nvidia.com/even-easier-introduction-cuda)。

实际的CUDA核心相当简单（如果您以前编程过GPU的话）：

```cpp
template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ old_cell,
    scalar_t* __restrict__ new_h,
    scalar_t* __restrict__ new_cell,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    input_gate[index] = sigmoid(gates[gates_row + column]);
    output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
    candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
    new_cell[index] =
        old_cell[index] + candidate_cell[index] * input_gate[index];
    new_h[index] = tanh(new_cell[index]) * output_gate[index];
  }
}
```


这里主要有趣的是我们能够为我们门控矩阵中的每个单独组件完全并行地计算所有这些点操作。如果您想象一下必须使用巨大的for循环逐个元素串行执行这个操作，您就会明白为什么这样做会更快了。

### 使用访问器

您可以在CUDA核心中看到，我们直接使用正确类型的指针进行操作。实际上，在CUDA核心中直接使用高级类型不可知的张量将非常低效。

然而，这样做的代价是使用和可读性的降低，特别是对于高维数据。在我们的例子中，我们知道连续的门控张量有3个维度：

* 批量，批量大小的大小和3*状态大小的步幅
* 行，大小为3和状态大小的步幅
* 索引，大小为状态大小和步幅为1

那么我们如何在核心内部访问元素gates[n][row][column]呢？事实证明，您需要通过一些简单的算术来使用步幅访问您的元素。

```
gates.data<scalar_t>()[n*3*state_size + row*state_size + column]
```

除了冗长外，这个表达式需要显式知道步幅，并在其参数中传递给内核函数。您可以看到，在接受多个具有不同大小的张量的内核函数的情况下，您将得到一个非常长的参数列表。

幸运的是，对于我们来说，ATen提供了访问器，它们通过单个动态检查Tensor是否具有类型和数量的维度来创建。然后，访问器公开了一个API，用于高效访问张量元素，而无需转换为单个指针：

```cpp
torch::Tensor foo = torch::rand({12, 12});

// assert foo is 2-dimensional and holds floats.
auto foo_a = foo.accessor<float,2>();
float trace = 0;

for(int i = 0; i < foo_a.size(0); i++) {
  // use the accessor foo_a to get tensor data.
  trace += foo_a[i][i];
}
```


访问器对象具有相对较高级的接口，具有.size()和.stride()方法以及多维索引。.accessor\<>接口旨在有效访问CPU张量上的数据。CUDA张量的等效物是packed_accessor64\<>和packed_accessor32<>，它们产生具有64位或32位整数索引的Packed Accessors。

访问器与Packed Accessor的根本区别在于，Packed Accessor将大小和步幅数据复制到其结构内部而不是指向它。这使我们能够将其传递给CUDA核心函数并在其中使用其接口。

我们可以设计一个函数，该函数接受Packed Accessors而不是指针。

```cpp
__global__ void lltm_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell)
```


让我们分解此处使用的模板。前两个参数scalar_t和2与常规Accessor相同。参数torch::RestrictPtrTraits指示必须使用\_\_restrict\_\_关键字。还请注意，我们使用了PackedAccessor32变体，它将大小和步幅存储在int32_t中。这很重要，因为使用64位变体（PackedAccessor64）可能会使内核变慢。

```cpp
template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = elu(gates[n][2][c]);
    new_cell[n][c] =
        old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
    new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
  }
}
```

函数声明变得实现更可读！然后通过在主机函数内部使用.packed_accessor32\<>方法创建Packed Accessors来调用此函数。

```cpp
std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto gates = gate_weights.reshape({batch_size, 3, state_size});
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}
```


后向传递遵循相同的模式，我不会进一步详细说明它：

```cpp
template <typename scalar_t>
__global__ void lltm_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gate_weights) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_gates.size(2)){
    const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
    const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
    const auto d_new_cell =
        d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


    d_old_cell[n][c] = d_new_cell;
    const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
    const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

    d_gates[n][0][c] =
        d_input_gate * d_sigmoid(gate_weights[n][0][c]);
    d_gates[n][1][c] =
        d_output_gate * d_sigmoid(gate_weights[n][1][c]);
    d_gates[n][2][c] =
        d_candidate_cell * d_elu(gate_weights[n][2][c]);
  }
}

std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gates,
    torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_cuda", ([&] {
    lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
```


### 将C++/CUDA操作集成到PyTorch中

将我们启用CUDA的操作与PyTorch集成再次非常简单。如果您想编写一个setup.py脚本，它可能如下所示：

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm',
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```


现在我们使用CUDAExtension()而不是CppExtension()。我们只需指定.cu文件以及.cpp文件——库会为您处理所有这些麻烦的工作。JIT机制甚至更简单：

```python
from torch.utils.cpp_extension import load

lltm = load(name='lltm', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])
```

### 性能比较

我们的希望是通过CUDA并行化和融合代码中的逐点操作会提高LLTM的性能。让我们看看这是否成立。我们可以运行我之前列出的代码来运行基准测试。我们之前最快的版本是基于CUDA的C++代码：

```
Forward: 149.802 us | Backward 393.458 us
```

现在我们使用自定义CUDA核心：

```
Forward: 129.431 us | Backward 304.641 us
```

性能进一步提升！

## 结论

现在您应该对PyTorch的C++扩展机制有了良好的概述，并且有了使用它们的动机。您可以在[这里](https://github.com/pytorch/extension-cpp)找到本文中显示的代码示例。如果您有问题，请使用[论坛](https://discuss.pytorch.org/)。同时，请务必查看我们的[FAQ](https://pytorch.org/cppdocs/notes/faq.html)，以防您遇到任何问题。

