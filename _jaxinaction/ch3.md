---
layout:     post
title:      "第三章：Working with tensors"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - Jax
---

<!--more-->
**目录**
* TOC
{:toc}

本章内容包括：

* 使用 NumPy 数组
* 在 CPU/GPU/TPU 上使用 JAX 张量
* 调整代码以适应 NumPy 数组和 JAX 之间的差异
* DeviceArray
* 使用高级和低级接口：jax.numpy 和 jax.lax

在前两章中，我们展示了 JAX 是什么以及为什么要使用它，并开发了一个简单的 JAX 神经网络。本章将深入探讨 JAX 核心，从张量开始。张量或多维数组是深度学习和科学计算框架中的基本数据结构。每个程序都依赖于某种形式的张量，无论是 1D 数组、2D 矩阵还是更高维度的数组。前一章中的手写数字图像、中间激活以及最终的网络预测——所有这些都是张量。

NumPy 数组及其 API 已成为许多其他框架尊重的事实上的行业标准。本章将介绍 JAX 中的张量及其相应的操作。我们将重点介绍 NumPy 和 JAX API 之间的差异。

## 3.1 使用 NumPy 数组进行图像处理

让我们从一个实际的图像处理任务开始。假设你有一组照片需要处理。有些照片有多余的空间需要裁剪，有些有你想去除的噪声伪影，还有许多是好的，但你想对它们应用艺术效果。为简单起见，我们仅关注去噪图像，如图 3.1 所示：

<a>![](/img/jaxinaction/ch3/1.png)</a>
*图 3.1 我们想要实现的图像处理示例*

要实现这种处理，你必须将图像加载到某种数据结构中，并编写处理它们的函数。

假设你有一张照片（我选择了埃里温卡斯凯德综合体中费尔南多·博特罗的猫雕像之一），你的相机在照片上产生了一些奇怪的噪声伪影。你想去除噪声，然后可能还想对图像应用一些艺术滤镜。当然，有很多图像处理工具，但为了演示，我们想实现自己的处理流程。你可能还想将神经网络处理添加到流程中，例如，实现超分辨率或更多艺术滤镜。使用 JAX，这非常简单。


### 3.1.1 在 NumPy 数组中加载和存储图像

首先，我们需要加载图像。图像是多维对象的很好例子。它们有两个空间维度（宽度和高度），通常还有一个颜色通道维度（通常是红色、绿色、蓝色，有时还有透明度）。因此，图像自然可以用多维数组表示，而 NumPy 数组是将图像保存在计算机内存中的合适结构。

相应的图像张量也有三个维度：宽度、高度和颜色通道。黑白或灰度图像可能没有通道维度。在 NumPy 索引中，第一个维度对应于行，第二个维度对应于列。通道维度可以放在空间维度之前或之后。技术上它也可以放在高度和宽度维度之间，但这没有意义。在 scikit-image 中，通道按 RGB 对齐（红色是第一个，然后是绿色，最后是蓝色）。其他库如 OpenCV 可能使用另一种布局，例如 BGR，颜色通道的顺序是相反的。

让我们加载图像并查看其属性。我将名为 'The_Cat.jpg' 的图像放入当前文件夹中。你可以放入任何其他图像，但不要忘记在代码中更改文件名。

我们将使用著名的 scikit-image 库来加载和保存图像。在 Google Colab 中，它已经预装好，但如果你没有这个库，请按照 https://scikit-image.org/docs/stable/install.html 上的说明安装它。

本节的代码和图像可以在 GitHub 上的书籍代码库中找到：https://github.com/che-shr-cat/JAX-in-Action/blob/main/Chapter-3/JAX_in_Action_Chapter_3_Image_Processing.ipynb。

以下代码加载图像：

```
import numpy as np
from scipy.signal import convolve2d

# The rest
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from skimage.util import img_as_float32, img_as_ubyte, random_noise

img = imread('The_Cat.jpg')
plt.figure(figsize = (6, 10))
plt.imshow(img)
```

图像将显示如图 3.2 所示。根据你的屏幕分辨率，你可能需要更改图像大小。可以随意使用 figsize 参数来实现。它是一个 (宽度, 高度) 元组，每个值的单位是英寸。

<a>![](/img/jaxinaction/ch3/2.png)</a>
*图 3.2 以高度、宽度和颜色维度表示的三维数组的彩色图像。*

我们可以检查图像张量的类型：

```
type(img)
numpy.ndarray
```


张量通常通过它们的形状来描述，这是一个元素数量等于张量秩（或维度数量）的元组。每个元组元素表示沿该维度的索引位置数量。shape 属性引用它。可以通过 ndim 属性知道维度数量。

例如，一个 1024 \* 768 的彩色图像可能表示为形状 (768, 1024, 3) 或 (3, 768, 1024)。

当你处理一批图像时，会添加一个新的批量维度，通常是第一个，索引为 0。

一批 32 张 1024*768 的彩色图像可以表示为形状 (32, 768, 1024, 3) 或 (32, 3, 768, 1024)。

```
img.ndim
3
```

```
img.shape
(667, 500, 3)
```


如你所见，图像由一个三维数组表示，第一个维度是高度，第二个是宽度，第三个是颜色通道。


图像张量在内存中通常以两种常见格式表示：NCHW 或 NHWC。这些大写字母编码了张量轴的语义，其中 N 代表批次维度，C 代表通道维度，H 代表高度，W 代表宽度。以这种方式描述的张量包含一个由 N 张图像组成的批次，每张图像有 C 个颜色通道，每个通道具有高度 H 和宽度 W。在批次中包含不同大小的对象会有问题，但存在特殊的数据结构，如不规则张量（ragged tensors）。另一种选择是用一些占位元素将不同对象填充到相同的大小。

不同的框架和库偏好不同的格式。JAX (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html) 和 PyTorch (https://discuss.pytorch.org/t/why-does-pytorch-prefer-using-nchw/83637/4) 默认使用 NCHW。TensorFlow (https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)、Flax (https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Conv.html#flax.linen.Conv) 和 Haiku (https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.Conv2D) 使用 NHWC，几乎每个库都有一些函数可以在这些格式之间进行转换，或一个参数指定传递给函数的图像张量类型。

从数学角度来看，这些表示是等效的，但从实际角度来看，可能存在差异。例如，在 NVIDIA 张量核心中实现的卷积需要 NHWC 布局，并且当输入张量按 NHWC 格式排列时工作更快 (https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout)。

在 TF 运行时中的默认图优化系统 Grappler (https://www.tensorflow.org/guide/graph_optimization) 可以在布局优化期间自动将 NHWC 转换为 NCHW (https://research.google/pubs/pub48051/)。

张量还具有与所有元素关联的数据类型，称为 dtype。图像通常用范围 [0, 1] 的浮点值或范围 [0, 255] 的无符号整数表示。我们的张量中的值是无符号 8 位整数（uint8）：


```
img.dtype
dtype('uint8')
```

还有两个与大小相关的有用属性。size 属性返回张量中的元素数量，它等于数组维度的乘积。nbytes 属性返回数组元素所占用的总字节数，不包括数组对象非元素属性所占用的内存。对于包含 uint8 值的张量，这些属性返回相同的数字，因为每个元素只需要一个字节。对于 float32 值，字节数是元素数量的四倍。

```
img.size
1000500

img.nbytes
1000500

ca = img.astype(np.float32)
ca.nbytes
4002000
```

NumPy 的一个强大功能是切片，我们可以直接用它进行裁剪。通过切片，你可以沿每个轴选择张量的特定元素。例如，你可以选择特定的矩形图像子区域或仅选择特定的颜色通道。

```
cat_face = img[80:220, 190:330, 1]
cat_face.shape
(140, 140)

plt.figure(figsize = (3,4))
plt.imshow(cat_face, cmap='gray')
```

以下代码从图像的绿色通道（索引为 1 的中间通道）中选择与猫脸相关的像素。由于图像仅包含单个颜色通道的信息，我们以灰度显示图像。如果需要，你可以选择其他颜色调色板。生成的图像如图 3.3 所示。


<a>![](/img/jaxinaction/ch3/3.png)</a>
*图 3.3 仅含单个颜色通道的裁剪图像*
 

你也可以通过数组切片轻松实现基本操作，比如图像翻转和旋转。

代码 img = img[:,::-1,:] 会逆转水平维度上像素的顺序，同时保留垂直和通道轴。你还可以使用 NumPy 的 flip() 函数。旋转可以使用 rot90() 函数，通过指定旋转次数（参数 k=2）来实现，如代码 img = np.rot90(img, k=2, axes=(0, 1))。

现在我们准备实现一些更高级的图像处理。

 
### 3.1.2 使用 NumPy API 进行基本图像处理

首先，我们将图像从 uint8 数据类型转换为 float32。处理范围在 [0.0, 1.0] 的浮点值会更容易。我们使用 scikit-image 库中的 img_as_float() 函数。


假设我们有一个带噪声的图像版本。为了模拟这种情况，我们使用高斯噪声，这种噪声在低光条件和高 ISO 光感的数码相机中经常出现。我们使用同一个 scikit-image 库中的 random_noise 函数。


```
img = img_as_float32(img)
img.dtype
dtype('float32')

img_noised = random_noise(img, mode='gaussian')

plt.figure(figsize = (6, 10))
plt.imshow(img_noised)
```

以下代码生成图像的噪声版本，如图 3.4 所示。你也可以尝试其他有趣的噪声类型，比如盐和胡椒(salt-and-pepper)噪声或脉冲噪声，这些噪声以稀疏的最小值和最大值像素形式出现。

<a>![](/img/jaxinaction/ch3/4.png)</a>
*图 3.4 图像的带噪声版本*

这种类型的噪声通常可以通过高斯模糊滤波器去除。高斯模糊滤波器属于矩阵滤波器的大家族，在数字信号处理（DSP）领域中也称为有限脉冲响应（FIR）滤波器。你可能在像 Photoshop 或 GIMP 这样的图像处理应用程序中见过矩阵滤波器。

**有限脉冲响应（FIR）滤波器和卷积。**

一个 FIR 滤波器通过其矩阵（也称为核）来描述。该矩阵包含权重，当滤波器沿图像滑动时，它们用于获取图像的像素。在每个步骤中，滤波器“看”图像部分的窗口中的所有像素都会与核的相应权重相乘。然后将乘积相加得到一个单一数字，这个数字就是输出（经过滤波的）图像中像素的结果强度。将信号和核（另一个信号）进行滤波的操作称为卷积。这个过程在图 3.5 中可视化展示了出来。

<a>![](/img/jaxinaction/ch3/5.png)</a>
**图 3.5 一个有限脉冲响应（FIR）滤波器通过卷积实现（来自Mohamed Elgendy的书《Deep Learning for Vision Systems》）**

卷积可以是任意维度的。对于 1D 输入，它是一个 1D 卷积；对于 2D 输入（如图像），它是一个 2D 卷积，依此类推。这是卷积神经网络（CNNs）中广泛使用的相同操作。我们将分别对每个图像通道应用 2D 卷积。在 CNNs 中，所有图像通道通常一次处理，使用与图像相同通道维度的核。

如果你想了解更多关于数字滤波器和卷积的知识，我推荐阅读关于数字信号处理的优秀书籍：http://dspguide.com/

例如，一个简单的模糊（非高斯模糊）滤波器由一个值相等的矩阵组成。这意味着目标像素邻域内的每个像素都以相同的权重获取。这等同于对核的感受域内的所有值进行平均。我们将使用一个大小为 5x5 像素的核。你可能听说过这样一个滤波器，称为简单移动平均滤波器。

```
kernel_blur = np.ones((5,5))
kernel_blur /= np.sum(kernel_blur)

kernel_blur
Array([[0.04, 0.04, 0.04, 0.04, 0.04],
       [0.04, 0.04, 0.04, 0.04, 0.04],
       [0.04, 0.04, 0.04, 0.04, 0.04],
       [0.04, 0.04, 0.04, 0.04, 0.04],
       [0.04, 0.04, 0.04, 0.04, 0.04]], dtype=float32)
```

高斯模糊是模糊滤波器的更复杂版本，其矩阵包含不同的值，倾向于使接近中心的值较高。高斯核是由著名的高斯函数产生的。

```
# https://github.com/TheAlgorithms/Python/blob/master/digital_image_processing/filters/gaussian_filter.py
# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
# https://en.wikipedia.org/wiki/Gaussian_blur

def gaussian_kernel(kernel_size, sigma=1.0, mu=0.0):
    """ A function to generate Gaussian 2D kernel """
    center = kernel_size // 2
    x, y = np.mgrid[-center : kernel_size - center, -center : kernel_size - center]
    d = np.sqrt(np.square(x) + np.square(y))
    koeff = 1 / (2 * np.pi * np.square(sigma))
    kernel = koeff * np.exp(-np.square(d-mu) / (2 * np.square(sigma)))
    return kernel
```

```
kernel_gauss = gaussian_kernel(5)
kernel_gauss
Array([[0.00291502, 0.01306424, 0.02153928, 0.01306424, 0.00291502],
       [0.01306424, 0.05854984, 0.09653235, 0.05854984, 0.01306424],
       [0.02153928, 0.09653235, 0.15915494, 0.09653235, 0.02153928],
       [0.01306424, 0.05854984, 0.09653235, 0.05854984, 0.01306424],
       [0.00291502, 0.01306424, 0.02153928, 0.01306424, 0.00291502]],      dtype=float32)
```


现在我们需要实现应用滤波器到图像的函数。我们将分别对每个颜色通道应用滤波器。在每个颜色通道内，函数需要对颜色通道和核进行 2D 卷积。我们还将结果值剪裁到范围 [0.0, 1.0] 以限制其范围。该函数假设通道维度是最后一个维度。


```
def color_convolution(image, kernel):
  """ A function to apply a filter to an image"""
  channels = []
  for i in range(3):
    color_channel = image[:,:,i]
    filtered_channel = convolve2d(color_channel, kernel, mode="same") #, boundary="symm")
    filtered_channel = np.clip(filtered_channel, 0.0, 1.0)
    channels.append(filtered_channel)
  final_image = np.stack(channels, axis=2) #.astype('uint8')
  return final_image
```
我们准备好将我们的滤波器应用到图像上：


```
img_blur = color_convolution(img_noised, kernel_gauss)

plt.figure(figsize = (12,10))
plt.imshow(np.hstack((img_blur, img_noised)))
```


生成的去噪图像如图 3.6 所示。最好在计算机上查看图像，但你可以看到噪声量明显减少，但图像变得模糊。尽管这是预期的（我们应用了模糊滤波器），但如果可能的话，我们希望使图像更清晰。

<a>![](/img/jaxinaction/ch3/6.png)</a>
*图 3.6 图像的去噪版本（左）和带噪声版本（右）*



不出所料，还有另一个用于图像锐化的矩阵滤波器。其核心包含一个大的正数，周围是负数。这个滤波器的想法是增强中心点与其周围的对比度。

我们通过将每个值除以所有值的总和来对核值进行归一化。这是为了限制滤波器应用后的结果值在允许的范围内。我们的滤波应用函数中有一个 clip() 函数，但那是最后的手段，它将范围之外的每个值都裁剪到其边界。更精细的归一化方法会保留信号中更多的信息。

```
kernel_sharpen = np.array(
    [[-1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1],
     [-1, -1, 50, -1, -1],
     [-1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1]], dtype=np.float32
)
kernel_sharpen /= np.sum(kernel_sharpen)

kernel_sharpen
Array([[-0.03846154, -0.03846154, -0.03846154, -0.03846154, -0.03846154],
       [-0.03846154, -0.03846154, -0.03846154, -0.03846154, -0.03846154],
       [-0.03846154, -0.03846154,  1.923077  , -0.03846154, -0.03846154],
       [-0.03846154, -0.03846154, -0.03846154, -0.03846154, -0.03846154],
       [-0.03846154, -0.03846154, -0.03846154, -0.03846154, -0.03846154]],      dtype=float32)
```

让我们将锐化滤波器应用到模糊图像上：

```
img_restored = color_convolution(img_blur, kernel_sharpen)

plt.figure(figsize = (12,20))
plt.imshow(np.vstack(
    (np.hstack((img, img_noised)),
    np.hstack((img_restored, img_blur)))
))
```


图 3.7 展示了四种不同的图像（顺时针方向）：左上角是原始图像，右上角是带噪声版本，右下角是去噪但模糊的版本，最后左下角是锐化的版本。我们成功地恢复了图像的一些清晰度，并成功地去除了一些原始噪声！

<a>![](/img/jaxinaction/ch3/7.png)</a>
*图 3.7 图像的原始版本（左上角），带噪声版本（右上角），去噪但模糊的版本（右下角），以及锐化的版本（左下角）*

最后一步是保存生成的图像：


```
image_modified = img_as_ubyte(img_restored)
imsave('The_Cat_modified.jpg', arr=image_modified)
```

还有许多其他有趣的滤波器可以尝试，例如浮雕、边缘检测或自定义滤波器。我将其中一些放入了相应的Colab笔记本中。你也可以将几个滤波器核合并成一个单独的核。但现在我们将停止并回顾我们所做的工作。

我们从加载图像开始，学习了如何使用切片实现基本的图像处理操作，例如裁剪或翻转。然后我们制作了图像的带噪声版本，并学习了矩阵滤波器。通过矩阵滤波器，我们进行了一些噪声过滤和图像锐化，并且可能还会做更多工作。

我们使用NumPy实现了所有操作，现在是时候看看JAX带来了哪些变化了。

## 3.2 JAX中的张量

现在，我们将重写我们的图像处理程序，以使用JAX而不是NumPy。

### 3.2.1 切换到JAX类似NumPy的API

美妙之处在于，你可以替换几个导入语句，所有其他代码将与JAX一起工作！自己试试吧！

```

#import numpy as np
#from scipy.signal import convolve2d
import jax.numpy as np
from jax.scipy.signal import convolve2d
```

JAX有一个类似NumPy的API，从jax.numpy模块导入。还有一些在JAX中重新实现的来自SciPy的高级函数。这个jax.scipy模块不像完整的SciPy库那样丰富，但我们使用的函数（convolve2d()函数）在其中是存在的。

有时你会发现在JAX中没有对应的函数。例如，我们可能会使用scipy.ndimage中的gaussian_filter()函数进行高斯滤波。在jax.scipy.ndimage中没有这样的函数。

在这种情况下，你可能仍然会使用NumPy函数与JAX一起，并且有两个导入，一个是来自NumPy，另一个是JAX NumPy接口。通常是这样做的：

```
import numpy as np
import jax.numpy as jnp
```


然后，你使用NumPy的函数时加上前缀np，使用JAX的函数时加上前缀jnp。这可能会阻止你在NumPy函数上使用一些JAX功能，要么是因为它们是用C++实现的，Python只提供绑定，要么是因为它们不是函数式纯粹的。

另一种选择是你可以自己实现新的函数，就像我们用高斯滤波所做的那样。

在我们上面的例子中，当我们将NumPy导入替换为JAX兼容的NumPy API时，所有代码都可以正常工作。你可能唯一会注意到的是，在创建数组的地方，numpy.ndarray类型将被jaxlib.xla_extension.DeviceArray替换，就像在创建滤波器核时一样：

```
kernel_blur = np.ones((5,5))
kernel_blur /= np.sum(kernel_blur)

type(kernel_blur)
jaxlib.xla_extension.ArrayImpl
```

我们还看到数据类型现在是float32。在大多数情况下，NumPy中的数据类型将是float64。其他一切都像往常一样。

我们将在第3.3.2节中讨论浮点数据类型，现在让我们深入了解一下DeviceArray是什么。

### 3.3.2 什么是DeviceArray

DeviceArray是JAX中表示数组的类型。它可以使用不同的后端——CPU、GPU和TPU。它相当于numpy.ndarray，支持单个设备上的内存缓冲区。通常情况下，设备是JAX用于执行计算的工具。

**硬件类型：CPU、GPU、TPU**

CPU是中央处理单元，是来自Intel、AMD或苹果（现在使用自己的ARM处理器）的典型处理器。它是一种通用的通用计算设备，但许多新处理器都具有用于提高机器学习工作负载性能的特殊指令。

GPU是图形处理单元，最初用于计算机图形任务的特殊高度并行处理器。现代GPU包含大量（数千个）简单处理器（核心），并且具有高度并行性，这使得它们在运行某些算法时非常有效。矩阵乘法，目前深度学习的核心，就是其中之一。最著名且得到最好支持的是NVIDIA GPU，但AMD和Intel也有自己的GPU。

TPU是Google的Tensor Processing Unit，是ASIC（专用集成电路）的最著名例子。ASIC是定制用于特定用途的集成电路，而不是用于像CPU那样的通用用途。ASIC比GPU更专业化，因为GPU仍然是一种具有数千个计算单元的大规模并行处理器，可以执行许多不同的算法，而ASIC是一种旨在执行一组非常小的计算（比如，只有矩阵乘法）的处理器。但它执行得非常好。还有许多其他用于深度学习和人工智能的ASIC，但目前，它们在深度学习框架中的支持非常有限。

在许多情况下，你不需要手动实例化DeviceArray对象（我们也没有这样做）。你将通过jax.numpy函数（如array()、linspace()等）来创建它们。

这里与NumPy的一个显著差异是，NumPy通常接受Python列表或元组作为其API函数的输入（不包括array()构造函数）。JAX有意选择不接受列表或元组作为其函数的输入，因为这可能会导致性能下降，而这种情况很难检测。如果你想将Python列表传递给JAX函数，你必须明确将其转换为数组。

本节和以下几节的代码位于该书的存储库中：https://github.com/che-shr-cat/JAX-in-Action/blob/main/Chapter-3/JAX_in_Action_Chapter_3_DeviceArray.ipynb。


```
import numpy as np
import jax.numpy as jnp


np.array([1, 42, 31337])
array([    1,    42, 31337])

jnp.array([1, 42, 31337])
Array([    1,    42, 31337], dtype=int32)

np.sum([1, 42, 31337])
31380

try:
  jnp.sum([1, 42, 31337])
except TypeError as e:
  print(e)
sum requires ndarray or scalar arguments, got <class 'list'> at position 0.

jnp.sum(jnp.array([1, 42, 31337]))
Array(31380, dtype=int32)
```

注意：DeviceArray把标量也打包进数组里。

DeviceArray有一系列与NumPy数组类似的属性。官方文档展示了完整的方法和属性列表（https://jax.readthedocs.io/en/latest/jax.numpy.html#jax-devicearray）。



```
arr = jnp.array([1, 42, 31337])

arr.ndim
1

arr.shape
(3,)

arr.dtype
dtype('int32')

arr.size
3

arr.nbytes
12
```

DeviceArray对象设计成与Python标准库工具无缝配合。例如，当来自内置Python copy模块的copy.copy()或copy.deepcopy()遇到一个DeviceArray时，它等效于调用copy()方法，这将在与原始数组相同的设备上创建一个缓冲区的副本。

DeviceArray也可以被序列化，或转换成可以存储在文件中的格式，通过内置的pickle模块。与numpy.ndarray对象类似，DeviceArray将通过紧凑的位表示进行序列化。当执行反向操作，即反序列化或取消pickle时，结果将是一个新的DeviceArray对象在默认设备上。

这是因为反序列化可能发生在一个具有不同设备集的不同环境中。


### 3.2.3 设备相关操作

毫不奇怪，有一系列专门用于张量设备放置的方法。可能有很多可用设备。

由于我使用带有GPU的系统，我将有额外的设备可用。以下示例将使用CPU和GPU设备。如果您的系统除了CPU之外没有任何设备，请尝试使用Google Colab。即使在免费版中，它也提供云GPU。

**本地和全局设备**

首先是主机。主机是管理多个设备的CPU。单个主机可以管理多个设备（通常最多8个），因此为了使用更多设备，需要多主机配置。

JAX区分本地设备和全局设备。

对于进程而言，本地设备是进程可以直接寻址并在其上启动计算的设备。它是直接连接到运行JAX程序的主机（或计算机）的设备，例如CPU、本地GPU或直接连接到主机的8个TPU核心。jax.local_devices()函数显示进程的本地设备。jax.local_device_count()函数返回此进程可寻址的设备数量。这两个函数接收一个用于XLA后端的参数，可以是'cpu'、'gpu'或'tpu'。默认情况下，此参数为None，表示默认后端（如果可用，则为GPU或TPU）。

全局设备是跨所有进程的设备。只要每个进程在其本地设备上启动计算，计算就可以跨进程的设备并使用通过设备之间的直接通信链路进行的集体操作（通常是云TPU或GPU之间的高速互连）。jax.devices()函数显示所有可用的全局设备，jax.device_count()返回所有进程的设备总数。

我们将在第6章后面讨论多主机和多进程环境。现在，让我们只关注单主机环境；在我们的情况下，全局设备列表将等于本地设备列表。

```
jax.devices()
[cuda(id=0)]

jax.devices('cpu')
[CpuDevice(id=0)]

jax.device_count('gpu')
1

jax.local_devices()
[cuda(id=0)]
```


对于非CPU设备（这里是GPU，稍后在本章中是TPU），您可以看到process_index属性。每个JAX进程可以使用jax.process_index()函数获取其进程索引。在大多数情况下，它将等于0，但对于多进程配置，这将有所不同。在这里，在列表3.16中，我们看到有一个本地设备附加到索引为0的进程上。

**已提交和未提交的数据**

在JAX中，计算遵循数据放置。有两种不同的放置属性：

* 设备上的数据所在位置。
* 数据是否已提交到设备上。当数据已提交时，有时会称其为粘附到设备上。

您可以通过device()方法知道数据位于哪里。

默认情况下，JAX DeviceArray对象以未提交的方式放置在默认设备上。默认设备是由jax.devices()函数调用返回的列表的第一项（jax.devices()[0]）。如果存在GPU或TPU，则为第一个GPU或TPU，否则为CPU。


```
arr = jnp.array([1, 42, 31337])
arr.devices()
{cuda(id=0)}
```

您可以使用jax.default_device()上下文管理器来临时覆盖JAX操作的默认设备。如果您愿意，您还可以使用JAX_PLATFORMS环境变量或\-\-jax_platforms命令行标志。在提供JAX_PLATFORMS变量中的平台列表时，还可以设置优先级顺序。

涉及未提交数据的计算在默认设备上执行，并且结果也未提交到默认设备上。

您可以使用jax.device_put()函数调用并带有设备参数来明确将数据放置在特定设备上。在这种情况下，数据将提交到设备上。如果将None作为设备参数传递，则操作将像身份函数一样运行（如果操作数已经位于任何设备上），否则将将数据传输到默认设备并保持未提交状态。

```
arr_cpu = jax.device_put(arr, jax.devices('cpu')[0])
arr_cpu.devices()
{CpuDevice(id=0)}

arr.devices()
{cuda(id=0)}
```


请记住JAX的功能性质。jax.device_put()函数在指定的设备上创建数据的副本并返回它。原始数据不变。

还有一个反向操作jax.device_get()，用于将数据从设备传输到主机上的Python进程。返回的数据是一个NumPy ndarray。

```
arr_host = jax.device_get(arr)
type(arr_host)
numpy.ndarray

arr_host
array([    1,    42, 31337], dtype=int32)
```

涉及已提交数据的计算在已提交的设备上执行，并且结果也将提交到同一设备上。当您对提交到不同设备的参数进行操作时，将会收到错误提示（但如果某些参数未提交，则不会出错）。

```
arr = jnp.array([1, 42, 31337])
arr.devices()
{cuda(id=0)}

arr_cpu = jax.device_put(arr, jax.devices('cpu')[0])
arr_cpu.devices()
{CpuDevice(id=0)}

arr + arr_cpu
Array([    2,    84, 62674], dtype=int32)


arr_gpu = jax.device_put(arr, jax.devices('gpu')[0])

try:
  arr_gpu + arr_cpu
except ValueError as e:
  print(e)
Received incompatible devices for jitted computation. Got argument x1 of jax.numpy.add with shape int32[3] and device ids [0] on platform GPU and argument x2 of jax.numpy.add with shape int32[3] and device ids [0] on platform CPU  
```


在数组创建方面存在一些惰性，这适用于所有常量创建操作（zeros、ones、eye等）。这意味着，如果您调用类似于jax.device_put(jnp.ones(...), jax.devices()[1])的方法，则该调用将在与jax.devices()[1]对应的设备上创建一个全为零的数组，而不是在默认设备上创建，然后将其复制到jax.devices()[1]。

您可以在相关的拉取请求（https://github.com/google/jax/pull/1668）中阅读更多关于JAX延迟子语言的DeviceArray操作的内容。

现在您有了一个在GPU上进行类似于NumPy的计算的简单方法。只需记住，这并不是提高性能的全部。第五章将讨论JIT，提供更多性能改进。

### 3.2.4 异步调度

异步调度是另一个重要的概念需要了解。

JAX使用异步调度。这意味着，当执行一个操作时，JAX不会等待操作完成，而是立即将控制权返回给Python程序。JAX返回一个DeviceArray，从技术上讲，它是一个“future”。在“future”中，值不会立即可用，并且正如其名称所示，将来会在加速器设备上生成。但它已经包含了形状和类型，并且您也可以将其传递给下一个JAX计算。

我们之前没有注意到它，因为当您通过打印或转换为NumPy数组来检查结果时，JAX会自动强制Python等待计算完成。如果您想明确等待结果，可以使用DeviceArray的block_until_ready()方法。

异步调度非常有用，因为它允许Python不必等待加速器而继续运行，帮助Python代码不要成为关键路径的一部分。如果Python代码在设备上排队执行计算比设备实际执行的速度更快，并且不需要在中间检查值，那么Python程序可以以最有效的方式使用加速器，而无需加速器等待。

不了解异步调度可能会在进行基准测试时误导您，并导致过于乐观的结果。这就是我们在第一章中使用block_until_ready()方法的原因。

```
a = jnp.array(range(1000000)).reshape((1000,1000))
a.devices()
{cuda(id=0)}

%time x = jnp.dot(a,a)
CPU times: user 14.8 ms, sys: 6.06 ms, total: 20.9 ms
Wall time: 32 ms

%time x = jnp.dot(a,a).block_until_ready()
CPU times: user 2.77 ms, sys: 0 ns, total: 2.77 ms
Wall time: 41.2 ms

a_cpu = jax.device_put(a, jax.devices('cpu')[0])
%time x = jnp.dot(a_cpu,a_cpu).block_until_ready()
CPU times: user 237 ms, sys: 0 ns, total: 237 ms
Wall time: 53 ms
```


在上面的示例中，我们强调了在有无阻塞的情况下测量时间的差异。没有阻塞时，我们只测量了分派工作的时间，而没有计算本身的时间。

此外，我们还测量了在GPU和CPU上的计算时间。我们通过将数据张量提交到相应的设备来完成这项工作。正如您所看到的，GPU计算比CPU计算快30倍（4.33毫秒对比150毫秒）。【译注：上面的结果是我本机的结果，和原书有一定差距。】

在阅读文档时，您可能会注意到一些函数被明确描述为异步的。例如，对于device_put()函数，文档中说明“此函数始终是异步的，即立即返回”。

### 3.2.5 将图像处理移至TPU

我们已经将代码更改为使用JAX，并且我们的代码已经可以在GPU上运行。我们没有特别的操作来将计算转移到GPU上，但是当我们在具有GPU的系统上运行此代码时，默认设备变为了GPU设备。太神奇了！一切都可以立即使用。

让我们再做一件事，将我们的代码运行在TPU上。在Google Colab中，您需要将运行时更改为TPU。这可以通过在“运行时”->“更改运行时类型”->“硬件加速器”部分中选择TPU来完成。更改运行时类型后，您必须重新启动运行时。

如果您选择了TPU运行时，您唯一需要的是在导入JAX之前设置好Cloud TPU。在jax.tools.colab_tpu模块中有一个特殊的代码用于此目的。在幕后，它将JAX XLA后端更改为TPU，并将JAX后端链接到TPU主机。

```
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
tpu

import jax
jax.local_devices()

import jax.numpy as jnp
a = jnp.array(range(1000000)).reshape((1000,1000))
a.device()

%time x = jnp.dot(a,a).block_until_ready()

>>> CPU times: user 1.32 ms, sys: 2.63 ms, total: 3.94 ms
>>> Wall time: 4.19 ms

```

有8个TPU设备，其ID从0到7，因为每个Cloud TPU是一个带有4个TPU芯片的TPU板，每个芯片包含两个核心。您可以看到我们的张量所在的设备现在是一个TPU。更有趣的是，它是八个核心中的一个特定的TPU核心（第一个设备是默认设备）。点积计算也发生在这个特定的核心上。

为了获取有关TPU更多的信息，如其版本和利用率，您可以使用分析器服务，该服务已经为所有TPU工作者在8466端口启动。在我们的案例中，我们使用的是旧版本的TPU v2。截至我撰写本章时，TPU v3已经普遍可用，而TPU v4则处于预览状态。

更多关于TPU的信息可以在这里找到：https://blog.inten.to/hardware-for-deep-learning-part-4-asic-96a542fe6a81

这样，我们就可以将图像处理移至TPU上。


## 3.3 与 NumPy 的区别

在某些情况下，您可能仍然希望使用纯 NumPy，特别是在不需要 JAX 提供的任何优势时，尤其是在运行小规模的一次性计算时。如果不是这种情况，并且您想要利用 JAX 提供的优势，您可以从 NumPy 切换到 JAX。然而，您可能还需要对代码进行一些更改。

尽管 JAX 的 NumPy 类似 API 尽量遵循原始 NumPy API，但仍有一些重要的区别。

一个显而易见的区别是我们已经知道的加速器支持。张量可以驻留在不同的后端（CPU、GPU、TPU）上，并且您可以精确地管理张量设备的放置。异步调度也属于这一类别，因为它被设计用来高效地使用加速计算。

我们已经提到的另一个区别是对非数组输入的行为，我们在 3.2.2 节中讨论过。记住，许多 JAX 函数不接受列表或元组作为其输入，以防止性能下降。

其他区别包括不可变性和一些与支持的数据类型和类型提升相关的更特殊的话题。让我们更深入地讨论这些话题。

### 3.3.1 不可变性

JAX 数组是不可变的。我们之前可能没有注意到，但尝试更改任何张量，您会看到一个错误。这是为什么呢？

```
a_jnp = jnp.array(range(10))
a_np  = np.array(range(10))

a_np[5], a_jnp[5]
(5, Array(5, dtype=int32))

a_np[5] = 100

try:
  a_jnp[5] = 100
except TypeError as e:
  print(e)
```

记住，JAX 是设计用于遵循函数式编程范式的。这就是为什么 JAX 转换如此强大的原因。关于函数式编程，有很多优秀的书籍（例如《Grokking Functional Programming》、《Grokking Simplicity》等），我们并不打算在本书中全面覆盖这个主题。但请记住函数式纯净的基本原则，代码不应有副作用。修改原始参数的代码不是函数式纯净的。创建修改后的张量的唯一方法是基于原始张量创建另一个张量。您可能在其他具有函数式范式的系统和语言中见过这种行为，例如在 Spark 中。

这与一些 NumPy 的编程实践相矛盾。NumPy 中一个常见的操作是索引更新，即通过索引更改张量中的值。例如，更改数组第五个元素的值。这在 NumPy 中完全没问题，但在 JAX 中会引发错误。

得益于 JAX，错误信息非常详细，并提供了解决问题的方法。

**索引更新功能**

对于所有用于更新张量元素值的典型就地表达式，JAX 中都有相应的函数式纯净等价物。表 3.1 列出了 JAX 中与 NumPy 风格的就地表达式等价的函数操作。

<a>![](/img/jaxinaction/ch3/8.png)</a>

所有这些 x.at 表达式都返回 x 的修改副本，而不更改原始副本。它可能比原始的就地修改代码效率低，但由于 JIT 编译，低级别的表达式如 x = x.at[idx].set(y) 被保证就地应用，使计算高效。因此，在使用索引更新功能时无需担心效率问题。

现在我们可以更改代码以修复错误：


```
a_jnp = a_jnp.at[5].set(100)
a_jnp[5]
Array(100, dtype=int32)
```


**越界索引**

一种常见的错误类型是对数组进行越界索引。在 NumPy 中，依靠 Python 异常来处理这种情况是相当直接的。然而，当代码在加速器上运行时，这可能会变得困难甚至不可能。因此，我们需要一些非错误行为来处理越界索引。对于越界操作的索引更新，我们希望跳过这些更新；对于越界操作的索引检索，我们希望将索引限制在数组的边界内，因为我们需要返回一些值。这类似于使用特殊值（如 NaN）处理浮点计算错误。

默认情况下，JAX 假设所有索引都在边界内。JAX 通过索引更新函数的 mode 参数提供了对越界索引访问进行更精确语义的实验性支持。可能的选项有：

* "promise_in_bounds"（默认）：用户承诺所有索引都在边界内，因此不会执行额外检查。实际上，这意味着 get() 中的所有越界索引都会被剪裁，而在 set()、add() 及其他修改函数中，越界索引会被忽略。
* "clip"：将越界索引限制在有效范围内。
* "drop"：忽略越界索引。
* "fill"：是 "drop" 的别名，但对于 get()，它将返回可选的 fill_value 参数中指定的值。

以下是使用不同选项的示例：

```
a_jnp = jnp.array(range(10))

a_jnp[42]
Array(9, dtype=int32)

a_jnp.at[42].get(mode='drop')
Array(-2147483648, dtype=int32)

a_jnp.at[42].get(mode='fill', fill_value=-1)
Array(-1, dtype=int32)

a_jnp = a_jnp.at[42].set(100)
a_jnp
Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

a_jnp = a_jnp.at[42].set(100, mode='clip')
a_jnp
Array([  0,   1,   2,   3,   4,   5,   6,   7,   8, 100], dtype=int32)
```



如你所见，越界索引不会产生错误，它总是返回某个值，你可以控制这种情况的行为。关于不可变性的话题就讲到这里了，让我们看看与 NumPy 相比的另一个大话题。

### 3.3.2 类型

关于数据类型，JAX 与 NumPy 有几处不同。这包括低精度和高精度浮点格式支持，以及类型提升语义，这决定了当操作数为特定（可能不同）类型时，操作结果将具有何种类型。

**float64 支持**

虽然 NumPy 积极将操作数提升到双精度（或 float64）类型，但 JAX 默认强制使用单精度（或 float32）数值。当你直接创建一个 float64 数组时，你可能会惊讶地发现 JAX 默默地将其转换为 float32。对于许多机器学习（尤其是深度学习）工作负载来说，这是完全可以接受的，但对于一些高精度科学计算来说，这可能不太理想。

**浮点类型：float64/float32/float16/bfloat16**

在科学计算和深度学习中有许多浮点类型。IEEE 754 浮点运算标准定义了几种不同精度的格式，这些格式被广泛使用。

科学计算的默认浮点数据类型是双精度浮点数，即 float64，因为这种浮点数的大小为 64 位。IEEE 754 双精度二进制浮点格式具有 1 位符号、11 位指数和 52 位小数部分。它的范围是 ~2.23e-308 到 ~1.80e308，具有 15–17 位十进制数字的精度。

在某些情况下，有更高精度的类型，例如长双精度或扩展精度浮点数，这通常是 x86 平台上的 80 位浮点数（不过，有很多注意事项）。NumPy 支持 np.longdouble 类型以实现扩展精度，而 JAX 不支持这种类型。

深度学习应用往往对较低精度具有鲁棒性，因此单精度浮点数或 float32 成为此类应用的默认数据类型，也是 JAX 中的默认浮点数据类型。32 位 IEEE 754 浮点数具有 1 位符号、8 位指数和 23 位小数部分。它的范围是 ~1.18e-38 到 ~3.40e38，具有 6–9 位有效十进制数字的精度。

对于许多深度学习情况来说，甚至 32 位浮点数都显得过多，因此在过去的一年里，低精度训练和推理变得流行起来。通常，低精度推理比训练更容易实现，而且存在一些混合 float16/32 精度训练的复杂方案。

在低精度浮点格式中，有两种 16 位浮点数：float16 和 bfloat16。

IEEE 754 半精度浮点数或 float16 具有 1 位符号、5 位指数和 10 位小数部分。其范围是 ~5.96e−8 到 65504，具有 4 位有效十进制数字。

另一种最初由 Google 开发的 16 位格式称为“Brain Floating Point Format”，简称 bfloat16。最初的 IEEE float16 设计时并未考虑深度学习应用，其动态范围过窄。bfloat16 类型解决了这一问题，提供了与 float32 相同的动态范围。它具有 1 位符号、8 位指数和 7 位小数部分。其范围是 ~1.18e-38 到 ~3.40e38，具有 3 位有效十进制数字。bfloat16 格式作为 IEEE 754 float32 的截断版本，允许快速转换为 IEEE 754 float32 以及从其转换。在转换为 bfloat16 格式时，指数位保留，而有效数字字段可以通过截断来减少。

还有一些其他的特殊格式，你可以在我的文章中了解更多内容：https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407

要强制进行 float64 计算，你需要在启动时设置 jax_enable_x64 配置变量。

但是，64 位数据类型并不支持所有的后端。例如，TPU 不支持它。


```
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

x = jnp.array(range(10), dtype=jnp.float64)
x.dtype
dtype('float64')

xb16 = jnp.array(range(10), dtype=jnp.bfloat16)
xb16.dtype
dtype(bfloat16)

xb16.nbytes
20

x16 = jnp.array(range(10), dtype=jnp.float16)
x16.dtype
dtype('float16')
```

再次强调，特定后端可能存在限制。

**类型提升语义**

对于二元运算，JAX 的类型提升规则与 NumPy 使用的规则略有不同。

显然，对于 bfloat16 类型而言，这种提升规则不同于 NumPy，因为 NumPy 不支持这种类型。但在其他一些情况下也有所不同。

有趣的是，当你将一个普通的 float16 和一个 bfloat16 相加时，结果是 float32 类型。


```
xb16+x16
Array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.], dtype=float32)

xb16+xb16
Array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=bfloat16)
```


你可以在这里找到二元运算中 NumPy 与 JAX NumPy 类型提升语义的区别表格。https://jax.readthedocs.io/en/latest/type_promotion.html


此外，还有一个更详尽的 NumPy、TensorFlow、PyTorch、JAX NumPy 和 JAX lax（见下一节关于 lax 的内容）之间的比较：https://jax.readthedocs.io/en/latest/jep/9407-type-
promotion.html#appendix-example-type-promotion-tables
。

**特殊张量类型**

我们目前处理的是所谓的密集张量。密集张量显式包含所有的值。然而，在许多情况下，你可能会遇到稀疏数据，这意味着有许多值为零的元素和一些（通常是数量级较少的）非零元素。

稀疏张量仅显式包含非零值，可以节省大量内存，但为了有效使用它们，你需要支持稀疏性的特殊线性代数例程。JAX 具有与稀疏张量和稀疏矩阵操作相关的实验性 API，位于 jax.experimental.sparse 模块中。当本书出版时，这些内容可能会有所变化，因此我们不会在书中讨论这个实验性模块。

另一个用例是具有不同形状的张量集合，例如，长度不同的语音记录。为了将这些张量组成一个批次，现代框架中有特殊的数据结构，例如 TensorFlow 中的ragged张量。

JAX 没有像ragged张量那样的特殊结构，但你可以自由地处理这样的数据。自动批处理功能（如 vmap()）在许多方面都能有所帮助，我们将在第 6 章中看到示例。

JAX 与原始 NumPy 之间的另一个重大区别是，JAX 中存在另一个低级 API，我们将在下节讨论。



## 3.4 高级和低级接口：jax.numpy 和 jax.lax

我们已经熟悉了 jax.numpy API，它为那些了解 NumPy 的人提供了一个熟悉的接口。对于我们的图像处理示例，NumPy API 已经足够，我们不需要其他 API。

然而，你应该知道，还有一个名为 jax.lax 的低级 API，它支持诸如 jax.numpy 之类的库。jax.numpy API 是一个高级封装器，所有操作都是以 jax.lax 原语表达的。许多 jax.lax 原语本身就是等效 XLA 操作的薄封装（https://www.tensorflow.org/xla/operation_semantics）。

在图 3.8 中，显示了 JAX API 层次的图表。高级 jax.numpy API 比 jax.lax API 更稳定，更不容易发生变化。JAX 的作者建议在可能的情况下，使用诸如 jax.numpy 之类的库，而不是直接使用 jax.lax。

<a>![](/img/jaxinaction/ch3/9.png)</a>

jax.lax 提供了大量的数学运算符。其中一些运算符更通用，提供了 jax.numpy 中没有的功能。例如，当你在 jax.numpy 中进行 1D 卷积时，你使用的是 jnp.convolve 函数（https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.convolve.html）。这个函数使用的是 jax.lax 中更通用的 conv_general_dilated 函数（https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html）。

还有其他类别的运算符：控制流运算符、自定义梯度运算符、支持并行的运算符和线性代数运算符。jax.lax 运算符的完整列表可在 https://jax.readthedocs.io/en/latest/jax.lax.html#module-jax.lax 查看。

我们将在第 4 章讨论一些梯度运算符，在第 6 章讨论并行支持。现在我们简要描述一下控制流原语。

**控制流原语**

jax.lax 中的结构化控制流原语集包括 lax.cond，它用于有条件地应用两个函数之一（类似于 if 语句）；lax.switch 用于多分支；lax.while_loop 和 lax.fori_loop 用于在循环中重复调用一个函数；lax.scan 和 lax.associative_scan 用于扫描一个函数并在数组上携带状态；以及 lax.map 函数，用于在数组的前导(leading)轴上应用一个函数。这些原语是 JIT 可执行的并且可微分，帮助避免展开大循环。

在 JAX 中，你仍然可以使用 Python 控制结构，并且在大多数情况下它是可行的。然而，这种解决方案可能不是最佳的，要么是因为有更高效的解决方案，要么是因为它们生成的微分代码效率较低。

我们将在整本书中使用许多这些和其他 jax.lax 原语，并将在首次使用时在后面的章节中描述它们。现在我们给你一个使用 lax.switch 的示例。

让我们考虑一个与图像过滤完全相反的案例。在计算机视觉的深度学习中，你常常希望使你的神经网络对不同的图像失真具有鲁棒性。例如，你希望你的神经网络对噪声具有鲁棒性。为了做到这一点，你在数据集中提供噪声版本的图像，并且你需要创建这样的噪声图像。你可能还希望使用图像增强技术，通过对原始图像的一些变换（例如，一些旋转、左右（有时上下）翻转、颜色失真等）来有效增加你的训练数据集。

假设你有一组用于图像增强的函数：

```
augmentations = [
   add_noise_func,
   horizontal_flip_func,
   rotate_func,
   adjust_colors_func
]
```

我们不会提供这些函数的代码，它们只是为了演示目的。你可以自己实现它们作为练习。

在下面的示例（代码清单 3.27）中，我们使用 lax.switch 在多个选项中选择一个随机的图像增强。第一个参数的值，这里是变量 augmentation_index，决定了将在多个选项中应用哪个特定分支。第二个参数，这里是 augmentations 列表，提供了一系列函数（或分支）供选择。我们使用一组图像增强函数列表，每个函数都执行自己的图像处理，无论是添加噪声、进行水平翻转、旋转等等。如果 augmentation_index 变量是 0，则调用 augmentations 列表的第一个元素；如果 augmentation_index 变量是 1，则调用第二个元素，依此类推。其余参数，这里只有一个称为 image，将传递给所选择的函数，并且所选择函数返回的值，这里是增强后的图像，是 lax.switch 运算符的结果值。

```
def random_augmentation(image, augmentations, rng_key):
   '''A function that applies a random transformation to an image'''
   augmentation_index = random.randint(key=rng_key, minval=0, maxval=len(augmentations), shape=())
   augmented_image = lax.switch(augmentation_index, augmentations, image)
   return augmented_image
```

在第 5 章中，我们将理解为什么使用 jax.lax 控制流原语的代码可能会导致更高效的计算。


jax.lax API 比 jax.numpy 更严格。它不会隐式提升混合数据类型操作的参数。在这种情况下，使用 jax.lax 时，你必须手动进行类型提升。

```
jnp.add(42, 42.0)
Array(84., dtype=float32, weak_type=True)

jnp.add(42.0, 42.0)
Array(84., dtype=float32, weak_type=True)

try:
   lax.add(42, 42.0)
except Exception as e:
   print(e)

Cannot lower jaxpr with verifier errors:
	'stablehlo.add' op requires compatible types for all operands and results
............


lax.add(jnp.float32(42), 42.0)
Array(84., dtype=float32)
```


在上面的示例中，你可以看到 DeviceArray 值中的 weak_type 属性。这个属性意味着存在一个没有显式用户指定类型的值，例如 Python 标量字面量。你可以在 JAX 文档中阅读更多关于弱类型值的信息：
https://jax.readthedocs.io/en/latest/type_promotion.html#weak-types。

我们在这里不会深入探讨 jax.lax API，因为它的大部分优点只有在第 4 到第 6 章之后才会变得清晰。现在重要的是强调 jax.numpy API 不是 JAX 中唯一的 API。

## 3.6 总结

* JAX 有一个类似 NumPy 的 API，从 jax.numpy 模块导入，尽量接近原始 NumPy API，但存在一些差异。
* DeviceArray 是在 JAX 中表示数组的类型。
* 你可以精确地管理 DeviceArray 数据的设备放置，将其提交到选择的设备上，无论是 CPU、GPU 还是 TPU。
* JAX 使用异步调度进行计算。
* JAX 数组是不可变的，因此你必须使用 NumPy 风格的就地表达式的功能性纯等效表达式。
* JAX 提供了不同的模式来控制越界索引的非错误行为。
* JAX 默认的浮点数据类型是 float32。你可以使用 float64/float16/bfloat16，但在某些特定后端上可能存在一些限制。
* 低级别的 jax.lax API 更严格，通常比高级别的 jax.numpy API 提供更多强大的功能。






