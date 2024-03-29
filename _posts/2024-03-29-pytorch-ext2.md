---
layout:     post
title:      "翻译：EXTENDING PYTORCH" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - pytorch
    - extension 
---

本文翻译[EXTENDING PYTORCH](https://pytorch.org/docs/master/notes/extending.html)。

<!--more-->

**目录**
* TOC
{:toc}

在这份说明中，我们将介绍扩展 [torch.nn](https://pytorch.org/docs/master/nn.html#module-torch.nn)、[torch.autograd](https://pytorch.org/docs/master/torch.html#module-torch.autograd)、[torch](https://pytorch.org/docs/master/torch.html#module-torch)，并编写利用我们的 C 库的自定义 C 扩展的方法。

## 扩展 torch.autograd

向 autograd 添加操作需要为每个操作实现一个新的 [Function](https://pytorch.org/docs/master/autograd.html#torch.autograd.Function) 子类。回想一下，Function 是 autograd 用来编码操作历史并计算梯度的工具。

这份文档的第一部分侧重于反向模式自动微分，因为它是最常用的特性。文末的一节讨论了正向模式自动微分的扩展。

### 何时使用

一般来说，如果要在模型中执行不可微分的计算或依赖于非 PyTorch 库（例如 NumPy）的计算，但仍希望您的操作与其他操作链接并与自动微分引擎一起工作，就要实现一个自定义函数。

在某些情况下，自定义函数还可用于提高性能和内存使用率：如果您使用 [C++ 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)实现了前向传播和后向传播，您可以将它们包装在 Function 中以与自动微分引擎进行交互。如果您希望减少后向传播保存的缓冲区数量，可以使用自定义函数将操作组合在一起。

### 何时不使用

如果您已经可以使用 PyTorch 内置操作编写您的函数，并且其反向图（很可能）已经可以被 autograd 记录，那么您不需要自己实现反向函数。考虑使用普通的 Python 函数。

如果需要维护状态，即可训练参数，您应该（也）使用自定义模块。有关扩展 torch.nn 的更多信息，请参阅下面的部分。

如果您想在反向传播过程中修改梯度或执行副作用，请考虑注册[张量](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch.Tensor.register_hook)或[模块](https://pytorch.org/docs/stable/notes/modules.html#module-hooks)钩子。

### 如何使用

按照以下步骤进行：1. 子类化 Function 并实现 forward()、（可选的）setup_context() 和 backward() 方法。2. 在 ctx 参数上调用适当的方法。3. 声明您的函数是否支持双重反向传播。4. 使用 gradcheck 验证您的梯度是否正确。

#### 步骤 1 

在子类化 Function 后，您需要定义 3 个方法：

* forward() 是执行操作的代码。它可以接受任意数量的参数，其中一些是可选的，如果指定了默认值。这里可以接受所有类型的 Python 对象。在调用前会将跟踪历史的张量参数（即，requires_grad=True）转换为不跟踪历史的张量，并在图中注册它们的使用。请注意，此逻辑不会遍历列表/字典/任何其他数据结构，只会考虑直接作为参数传递的张量。您可以返回单个张量输出，或者如果有多个输出，则返回张量元组。此外，请参阅 Function 的文档，了解可以从 forward() 中调用的有用方法的描述。

* setup_context()（可选）。可以编写一个“组合” forward()，接受一个 ctx 对象，或者（在 PyTorch 2.0 之后）编写一个不接受 ctx 的独立 forward() 和一个 setup_context() 方法，其中 ctx 修改发生。forward() 应该包含计算，而 setup_context() 应该只负责 ctx 修改（不包含任何计算）。一般来说，独立的 forward() 和 setup_context() 更接近 PyTorch 原生操作的工作方式，因此更具有与各种 PyTorch 子系统组合的能力。有关更多详细信息，请参阅关于[组合或独立 forward() 和 setup_context()](https://pytorch.org/docs/master/notes/extending.html#combining-forward-context) 的部分。

* backward()（或 vjp()）定义梯度公式。它将获得与输出数量相同的张量参数，每个参数表示相应输出的梯度。重要的是绝对不要就地修改这些张量。它应该返回与输入数量相同的张量，每个张量包含与其对应输入的梯度。如果您的输入不需要梯度（needs_input_grad 是一个布尔值元组，指示每个输入是否需要梯度计算），或者是非张量对象，则可以返回 python:None。此外，如果您的 forward() 有可选参数，则可以返回比输入数量更多的梯度，只要它们都是 None。

#### 步骤 2

您有责任正确使用 ctx 中的函数，以确保新的 Function 与自动微分引擎正常工作。

* 必须使用 save_for_backward() 保存任何要在反向传播中使用的张量。非张量应直接存储在 ctx 上。如果保存了既不是输入也不是输出的张量用于反向传播，则您的 Function 可能不支持双重反向传播（请参阅步骤 3）。

* 必须使用 mark_dirty() 标记由前向函数就地修改的任何输入。

* 必须使用 mark_non_differentiable() 告诉引擎输出是否不可微分。默认情况下，所有可微分类型的输出张量都将设置为需要梯度。非可微分类型的张量（例如，整数类型）永远不会被标记为需要梯度。

* set_materialize_grads() 可以用来告知自动微分引擎在输出不依赖于输入的情况下优化梯度计算，方法是在调用反向传播函数时不将传递给它的 grad 张量材料化。换句话说，如果设置为 False，在调用反向传播之前，python 中的 None 对象或 C++ 中的“未定义张量”（即 x.defined() 为 False 的张量 x）将不会转换为填充为零的张量，因此您的代码需要处理这些对象，就好像它们是填充为零的张量一样。此设置的默认值为 True。

#### 步骤 3

如果您的 Function 不支持双重反向传播，则应通过使用 once_differentiable() 装饰器显式声明此功能。使用此装饰器，尝试通过您的函数执行双重反向传播将产生错误。有关双重反向传播的更多信息，请参阅我们的双重反向传播教程。

#### 步骤 4

建议使用 torch.autograd.gradcheck() 来检查您的反向传播函数是否正确计算了前向传播的梯度，方法是使用您的反向传播函数计算雅可比矩阵，并将其值逐元素与使用有限差分法计算的雅可比矩阵进行比较。

### 例子

以下是线性函数的代码，附带额外的注释：

```
# Inherit from Function
class LinearFunction(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
```


现在，为了更容易使用这些自定义操作，我们建议要么给它们取别名，要么将它们包装在一个函数中。将其包装在一个函数中可以让我们支持默认参数和关键字参数：

```python
# Option 1: alias

linear = LinearFunction.apply

# Option 2: wrap in a function, to support default args and keyword args.

def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)
```
这里，我们给出了一个由非张量参数参数化的函数的额外示例：

```
class MulConstant(Function):
    @staticmethod
    def forward(tensor, constant):
        return tensor * constant

    @staticmethod
    def setup_context(ctx, inputs, output):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor, constant = inputs
        ctx.constant = constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None
```


并且，在上面的示例中，我们通过调用 set_materialize_grads(False) 来优化它：


```
class MulConstant(Function):
    @staticmethod
    def forward(tensor, constant):
        return tensor * constant

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, constant = inputs
        ctx.set_materialize_grads(False)
        ctx.constant = constant

    @staticmethod
    def backward(ctx, grad_output):
        # Here we must handle None grad_output tensor. In this case we
        # can skip unnecessary computations and just return None.
        if grad_output is None:
            return None, None

        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None
```

如果您需要在 forward() 中计算的任何“中间”张量被保存，那么它们必须作为输出返回，或者将 forward() 与 setup_context() 结合起来（参见组合或独立 forward() 和 setup_context()）。请注意，这意味着如果您希望梯度流经这些中间值，您需要为它们定义梯度公式（还请参阅[双重反向传播教程](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)）：

```
class MyCube(torch.autograd.Function):
    @staticmethod
    def forward(x):
        # We wish to save dx for backward. In order to do so, it must
        # be returned as an output.
        dx = 3 * x ** 2
        result = x ** 3
        return result, dx

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        result, dx = output
        ctx.save_for_backward(x, dx)

    @staticmethod
    def backward(ctx, grad_output, grad_dx):
        x, dx = ctx.saved_tensors
        # In order for the autograd.Function to work with higher-order
        # gradients, we must add the gradient contribution of `dx`,
        # which is grad_dx * 6 * x.
        result = grad_output * dx + grad_dx * 6 * x
        return result

# Wrap MyCube in a function so that it is clearer what the output is
def my_cube(x):
    result, dx = MyCube.apply(x)
    return result
```

注意： 传递给 backward 的输入，即 grad_output，也可以是跟踪历史的张量。因此，如果 backward 是使用可微分操作实现的（例如调用另一个自定义 Function），高阶导数将起作用。在这种情况下，使用 save_for_backward 保存的张量也可以在 backward 中使用，并且梯度会流回来，但是在 ctx 中保存的张量不会有梯度流回来。如果您需要 ctx 中保存的张量有梯度流回来，您应该将其作为自定义 Function 的输出并使用 save_for_backward 保存。

您可能想检查您实现的 backward 方法是否实际计算了您函数的导数。可以通过与使用小的有限差分的数值近似进行比较来实现：

```
from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)
```

请参阅有关有限差分梯度比较的 [Numerical gradient checking](https://pytorch.org/docs/master/autograd.html#grad-check) 了解更多详细信息。如果您的函数用于高阶导数（对反向传播进行微分），则可以使用相同包中的 gradgradcheck 函数来检查高阶导数。

### 组合或独立 forward() 和 setup_context()

定义 Function 有两种主要方式。要么：

* 定义一个将 forward 计算逻辑与 setup_context() 结合的 forward()

* （从 PyTorch 2.0 开始）定义一个独立的 forward() 和 setup_context()

我们推荐第二种选项（独立的 forward() 和 setup_context()），因为这更接近于 PyTorch 原生操作的实现方式，并且可以与 torch.func transforms 组合。但是，我们计划在未来支持两种方法；将 forward() 与 setup_context() 结合：可以提供更多灵活性，因为您可以保存中间结果而无需将它们作为输出返回。

请参见前一节关于如何使用独立的 forward() 和 setup_context() 定义 Function 的内容。

以下是如何使用组合的 forward() 和 setup_context() 定义 Function 的示例：

```
class LinearFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight, bias=None):
        # The forward pass can use ctx.
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
```

### 正向模式自动微分
覆盖正向模式自动微分公式具有非常相似的 API，但存在一些微妙的差别。您可以实现 [jvp()](https://pytorch.org/docs/master/generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp) 函数。

它将获得与输入数量相同的张量参数，每个参数表示相应输入的梯度。它应该返回与输出数量相同的张量，每个张量包含与其对应输出的梯度。jvp() 将在 forward() 方法之后、apply() 返回之前调用。

jvp() 与 backward() 函数有一些微妙的区别：

* 您可以使用 ctx 将任何数据从 forward() 传递到 jvp() 函数。如果该状态在 backward() 中不需要，您可以在 jvp() 函数的末尾通过 del ctx.foo 明确释放它。

* jvp() 的实现必须支持反向微分，或者明确检查给定的正向模式梯度是否设置了 requires_grad。

* jvp() 函数必须匹配 forward() 的视图/就地行为。例如，如果第 i 个输入就地修改，则第 i 个梯度必须就地更新。类似地，如果第 j 个输出是第 k 个输入的视图，则返回的第 j 个输出梯度必须是给定第 k 个输入梯度的视图。

* 因为用户无法指定需要计算哪个梯度，所以 jvp() 函数应始终计算所有输出的梯度。

* 正向模式梯度会遵循由 set_materialize_grads() 设置的标志，并且当禁用此设置时，您可以获取到 None 输入梯度。

### torch.func transforms 和/或 torch.vmap()

请参阅 [Extending torch.func with autograd.Function](https://pytorch.org/docs/master/notes/extending.func.html#func-autograd-function) 了解详细信息。


## 扩展 torch.nn

nn 支持两种接口 - 模块和它们的功能版本。您可以以两种方式扩展它，但我们建议对所有类型的层都使用模块，这些层包含任何参数或缓冲区，并建议对无参数操作（如激活函数、池化等）使用功能形式。

添加操作的功能版本在上面的部分中已经完全覆盖。

### 添加模块

由于 nn 大量使用 autograd，添加新模块需要实现一个执行操作并能够计算梯度的 Function。从现在开始，我们假设我们想要实现一个 Linear 模块，并且我们已经像上面的列表中实现了该函数。添加这个功能所需的代码非常少。现在，需要实现两个函数：

* \_\_init\_\_（可选）- 接受诸如核大小、特征数量等参数，并初始化参数和缓冲区。

* forward() - 实例化一个 Function 并使用它执行操作。这与上面显示的功能包装器非常相似。

这是如何实现一个 Linear 模块的：

```
class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
```

## 扩展 torch

您可以创建自定义类型来模拟 Tensor，方法是定义一个具有与 Tensor 匹配的方法的自定义类。但是，如果您希望能够将这些类型传递给像 torch.add() 这样接受 Tensor 操作数的顶层 torch 命名空间中的函数怎么办？

如果您的自定义 python 类定义了名为 \_\_torch\_function\_\_ 的方法，那么当您的自定义类的实例被传递给 torch 命名空间中的函数时，PyTorch 将调用您的 torch_function 实现。这样可以为 torch 命名空间中的任何函数定义自定义实现，您的 torch_function 实现可以调用这些函数，使用户能够在已经为 Tensor 编写的现有 PyTorch 工作流中使用您的自定义类型。这适用于与 Tensor 无关的“鸭子”类型以及 Tensor 的用户定义子类。

### 使用类似 Tensor 的类型扩展 torch

注意：这个功能灵感来自于 NumPy 的 array_function 协议。请参阅 [NumPy 文档](https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch)和 [NEP-0018](https://numpy.org/neps/nep-0018-array-function-protocol.html) 了解更多详细信息。

为了使这个具体化，让我们从一个简单的例子开始，展示 API 调度机制。我们将创建一个自定义类型，表示一个二维标量张量，由对角线条目的顺序 N 和值 value 参数化：

```python
class ScalarTensor(object):
   def __init__(self, N, value):
       self._N = N
       self._value = value

   def __repr__(self):
       return "ScalarTensor(N={}, value={})".format(self._N, self._value)

   def tensor(self):
       return self._value * torch.eye(self._N)
```

这个设计的第一次迭代并不是很有用。ScalarTensor 的主要功能是提供比基本张量类更紧凑的标量张量的字符串表示：

```
>>> d = ScalarTensor(5, 2)
>>> d
ScalarTensor(N=5, value=2)
>>> d.tensor()
tensor([[2., 0., 0., 0., 0.],
        [0., 2., 0., 0., 0.],
        [0., 0., 2., 0., 0.],
        [0., 0., 0., 2., 0.],
        [0., 0., 0., 0., 2.]])
```

如果我们尝试将这个对象与 torch API 一起使用，会遇到问题：

```
>>> import torch
>>> torch.mean(d)
TypeError: mean(): argument 'input' (position 1) must be Tensor, not ScalarTensor
```

为 ScalarTensor 添加一个 \_\_torch\_function\_\_ 实现使得上面的操作能够成功。让我们重新设计我们的实现，这次添加一个 torch_function 实现：

```python
HANDLED_FUNCTIONS = {}
class ScalarTensor(object):
    def __init__(self, N, value):
        self._N = N
        self._value = value

    def __repr__(self):
        return "ScalarTensor(N={}, value={})".format(self._N, self._value)

    def tensor(self):
        return self._value * torch.eye(self._N)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
```

\_\_torch\_function\_\_ 方法接受四个参数：func，一个引用正在被覆盖的 torch API 函数，types，实现 torch_function 的 Tensor-likes 类型的类型列表，args，传递给函数的参数元组，和 kwargs，传递给函数的关键字参数字典。它使用一个名为 HANDLED_FUNCTIONS 的全局调度表来存储自定义实现。这个字典的键是 torch 命名空间中的函数，值是 ScalarTensor 的实现。

注意： 使用全局调度表不是 torch_function API 的强制部分，它只是一个有用的设计模式，用于构造您的覆盖实现。

这个类定义还不足以使 torch.mean 在我们传递 ScalarTensor 时执行正确的操作 - 我们还需要为 ScalarTensor 操作数定义一个 torch.mean 的实现，并将该实现添加到 HANDLED_FUNCTIONS 调度表字典中。一种方法是定义一个装饰器：

```
import functools
def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator
```

这个装饰器可以应用于我们覆盖的实现：

```python
@implements(torch.mean)
def mean(input):
    return float(input._value) / input._N
```


通过这个改变，我们现在可以使用 torch.mean 与 ScalarTensor：

```
>>> d = ScalarTensor(5, 2)
>>> torch.mean(d)
0.4
```


当然，torch.mean 是最简单类型的函数覆盖的例子，因为它只接受一个操作数。我们可以使用相同的机制来覆盖接受多个操作数的函数，其中任何一个可能是定义了 \_\_torch\_function\_\_ 的张量或张量类似物，例如 torch.add()：

```
def ensure_tensor(data):
    if isinstance(data, ScalarTensor):
        return data.tensor()
    return torch.as_tensor(data)

@implements(torch.add)
def add(input, other):
   try:
       if input._N == other._N:
           return ScalarTensor(input._N, input._value + other._value)
       else:
           raise ValueError("Shape mismatch!")
   except AttributeError:
       return torch.add(ensure_tensor(input), ensure_tensor(other))
```

这个版本对于两个操作数都是 ScalarTensor 实例的情况有一个快速路径，也有一个较慢的路径，当任何一个操作数不是 ScalarTensor 时会将数据转换为张量。这使得当任一操作数是 ScalarTensor 或常规 Tensor 时，覆盖函数都能正确执行。


```
>>> s = ScalarTensor(2, 2)
>>> torch.add(s, s)
ScalarTensor(N=2, value=4)
>>> t = torch.tensor([[1, 1,], [1, 1]])
>>> torch.add(s, t)
tensor([[3., 1.],
        [1., 3.]])
```

注意，我们的 add 实现不像 torch.add() 那样接受 alpha 或 out 作为关键字参数。

```
>>> torch.add(s, s, alpha=2)
TypeError: add() got an unexpected keyword argument 'alpha'
```

为了速度和灵活性，\_\_torch\_function\_\_ 调度机制不会检查覆盖函数的签名是否与 torch API 中被覆盖的函数的签名匹配。对于某些应用程序来说，忽略可选参数是可以接受的，但是为了确保与 Tensor 的完全兼容性，用户对 torch API 函数的实现应当确保精确地模拟被覆盖的函数的 API。

在 torch API 中没有显式覆盖的函数将从 \_\_torch\_function\_\_ 返回 NotImplemented。如果所有具有定义了 \_\_torch\_function\_\_ 的操作数都返回 NotImplemented，PyTorch 将引发 TypeError。这意味着大多数情况下，对于没有为某种类型显式覆盖的操作，当传递该类型的实例时，将引发 TypeError：

```
>>> torch.mul(s, 3)
TypeError: no implementation found for 'torch.mul' on types that
implement __torch_function__: [ScalarTensor]
```

实际上，这意味着如果您希望按照这些方式实现您的覆盖，则需要显式地实现完整的 torch API 或您关心的用例所涉及的整个 API 子集。这可能是一个很大的挑战，因为完整的 torch API 非常广泛。

另一种选择是对于没有处理的操作不返回 NotImplemented，而是在没有覆盖时将一个 Tensor 传递给原始的 torch 函数。例如，如果我们将 ScalarTensor 的 \_\_torch\_function\_\_ 实现更改为以下方式：

```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
        args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
        return func(*args, **kwargs)
    return HANDLED_FUNCTIONS[func](*args, **kwargs)
```

那么 torch.mul() 将能够正确工作，尽管返回类型总是一个 Tensor 而不是 ScalarTensor，即使两个操作数都是 ScalarTensor 实例：

```
>>> s = ScalarTensor(2, 2)
>>> torch.mul(s, s)
tensor([[4., 0.],
        [0., 4.]])
```

此外，还请查看下面的 MetadataTensor 示例，了解这种模式的另一种变体，它总是返回 MetadataTensor 来通过 [torch API](https://pytorch.org/docs/master/torch.html#module-torch) 中的操作传播元数据。

\_\_torch\_function\_\_ 协议是为了完全覆盖 API 设计的，部分覆盖可能会导致不良结果，特别是某些函数会引发 TypeError。这对于子类来说尤其重要，其中 torch.add、torch.Tensor.\_\_add\_\_ 和 torch.Tensor.add 这三个函数必须被覆盖，即使它们返回完全相同的结果。未能这样做还可能导致无限递归。如果需要从 torch.Tensor 子类中实现一个函数，必须在其实现中使用 super().torch_function。

### Subclassing torch.Tensor

自版本 1.7.0 开始，在 torch.Tensor 上的方法和公共 torch.* 命名空间中的函数应用于 torch.Tensor 子类时，将返回子类实例而不是 torch.Tensor 实例：

```
>>> class SubTensor(torch.Tensor):
...     pass
>>> type(torch.add(SubTensor([0]), SubTensor([1]))).__name__
'SubTensor'
>>> type(torch.add(SubTensor([0]), torch.tensor([1]))).__name__
'SubTensor'
```

如果存在多个子类，则默认选择层次结构中最低的子类。如果没有唯一确定的方式来确定这种情况，则会引发 TypeError：

```
>>> type(torch.add(SubTensor2([0]), SubTensor([1]))).__name__
'SubTensor2'
>>> type(torch.add(SubTensor2([0]), torch.tensor([1]))).__name__
'SubTensor2'
>>> torch.add(SubTensor([0]), OtherSubTensor([1]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: no implementation found for 'torch.add' on types that implement __torch_function__: [SubTensor, OtherSubTensor]
```


如果希望对所有张量方法进行全局覆盖，可以使用 \_\_torch\_function\_\_。以下是一个记录所有函数/方法调用的示例：

```
class LoggingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # NOTE: Logging calls Tensor.__repr__, so we can't log __repr__ without infinite recursion
        if func is not torch.Tensor.__repr__:
            logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)
```

然而，如果希望在 Tensor 子类上覆盖一个方法，可以通过直接覆盖方法（为子类定义它），或者使用 \_\_torch\_function\_\_ 并与 func 匹配来实现。

在 \_\_torch\_function\_\_ 中，对于子类来说，始终应该调用 super().\_\_torch\_function\_\_(func, ...) 而不是直接调用 func，这是在版本 1.7.0 之前的情况。如果未能这样做，可能会导致 func 递归回到 \_\_torch\_function\_\_，从而导致无限递归。


### 扩展 torch 以添加一个张量包装类型

另一个有用的情况是一种类型，它作为属性或通过子类化包装一个张量。下面我们实现了这种类型的一种特殊情况，即 MetadataTensor，它附加了一个元数据字典到一个通过 torch 操作传播的张量上。由于这是对完整 torch API 的一种通用包装，我们不需要逐个实现每个覆盖，因此可以使 \_\_torch_function\_\_ 实现更容忍对哪些操作被允许：

```python
class MetadataTensor(object):
    def __init__(self, data, metadata=None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self._metadata = metadata

    def __repr__(self):
        return "Metadata:\n{}\n\ndata:\n{}".format(self._metadata, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
        args = [a._t if hasattr(a, '_t') else a for a in args]
        assert len(metadatas) > 0
        ret = func(*args, **kwargs)
        return MetadataTensor(ret, metadata=metadatas[0])
```

这个简单的实现不一定适用于 torch API 中的每个函数，但足以捕获大多数常见操作：

```
>>> metadata = {'owner': 'Ministry of Silly Walks'}
>>> m = MetadataTensor([[1, 2], [3, 4]], metadata=metadata)
>>> t = torch.tensor([[1, 2], [1, 2]])
>>> torch.add(t, m)
Metadata:
{'owner': 'Ministry of Silly Walks'}

data:
tensor([[2, 4],
        [4, 6]])
>>> torch.mul(t, m)
Metadata:
{'owner': 'Ministry of Silly Walks'}

data:
tensor([[1, 4],
        [3, 8]])
```


### 多个定义了 \_\_torch_function\_\_ 的类型的操作

可以使用具有各自实现 \_\_torch_function\_\_ 的多个不同类型来使用 torch API，但必须特别小心。在这种情况下，规则如下：

* 调度操作收集每个操作数的所有不同 \_\_torch_function\_\_ 实现，并按顺序调用它们：子类在超类之前，否则按照操作符表达式中的左到右顺序。

* 如果返回的值不是 NotImplemented 之外的任何值，则将该值作为结果返回。实现可以通过返回 NotImplemented 来注册它们不实现某个操作。

* 如果所有的 \_\_torch_function\_\_ 实现都返回 NotImplemented，PyTorch 将引发 TypeError。

### 测试覆盖 PyTorch API 的覆盖情况

实现 \_\_torch_function\_\_ 的一个麻烦之处在于，如果某些操作有覆盖，而其他操作没有覆盖，用户最多会看到不一致的体验，或者最坏的情况下，当他们使用没有覆盖的函数时，会在运行时引发错误。为了简化这个过程，PyTorch 提供了一个面向开发者的 API，用于确保对 \_\_torch_function\_\_ 覆盖的全面支持。这个 API 是私有的，并且可能在未来发生变化而没有警告。

首先，要获取所有可覆盖函数的列表，请使用 torch.overrides._get_overridable_functions。这将返回一个字典，其键是 PyTorch Python API 中的命名空间，其值是该命名空间中可覆盖的函数列表。例如，让我们打印 torch.nn.functional 中可以覆盖的前 5 个函数的名称：

```
>>> from torch.overrides import get_overridable_functions
>>> func_dict = get_overridable_functions()
>>> nn_funcs = func_dict[torch.nn.functional]
>>> print([f.__name__ for f in nn_funcs[:5])
['adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
 'adaptive_max_pool1d', 'adaptive_max_pool1d_with_indices']
```

这个函数列表使得可以迭代所有可覆盖的函数，然而在实践中，这并不足以为所有这些函数编写测试，因为需要繁琐手动地复制每个测试中每个函数的签名。为了简化这个过程，torch.overrides._get_testing_overrides 函数返回一个字典，将 PyTorch API 中可覆盖的函数映射到具有与原始函数相同签名的虚拟 lambda 函数，但它们无条件返回 -1。这些函数最适合使用 inspect 来分析原始 PyTorch 函数的函数签名：

```
>>> import inspect
>>> from torch.overrides import get_testing_overrides
>>> override_dict = get_testing_overrides()
>>> dummy_add = override_dict[torch.add]
>>> inspect.signature(dummy_add)
<Signature (input, other, out=None)>
```


最后，torch.overrides.get_ignored_functions 返回一个显式不能通过 torch_function 覆盖的函数元组。这个列表可以用来确认通过 get_overridable_functions 返回的字典中不存在的函数不能被覆盖。

## 编写自定义的 C++ 扩展
请参阅[这个PyTorch 教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)，其中有详细的解释和示例。

文档可在 [torch.utils.cpp_extension](https://pytorch.org/docs/master/cpp_extension.html) 中找到。

## 编写自定义的 C 扩展

示例可在[此GitHub 存储库](https://github.com/pytorch/extension-ffi)中找到。

