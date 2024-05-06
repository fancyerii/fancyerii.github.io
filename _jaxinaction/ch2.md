---
layout:     post
title:      "第二章：Your first program in JAX"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - Jax
---



这一章涵盖了以下内容：

* MNIST手写数字分类问题
* 在JAX中加载数据集
* 在JAX中创建简单的神经网络
* 使用vmap()函数自动矢量化代码
* 使用grad()函数计算梯度
* 使用jit()函数进行即时编译
* 纯函数和非纯函数
* JAX深度学习项目的高级结构

在前一章中，我们学习了JAX及其重要性。我们还描述了使其如此强大的JAX功能。这一章将带领你实际了解JAX。

JAX是一个用于Python和NumPy程序可组合转换的库，技术上并不限于深度学习研究。然而，JAX仍然被认为是一个深度学习框架，有时排在PyTorch和TensorFlow之后。因此，许多人开始学习JAX用于深度学习应用。因此，展示JAX方法解决问题的简单神经网络应用对许多人来说非常有价值。

本章是一个深度学习“hello world”风格的章节。我们在这里介绍了一个手写数字图像分类的示例项目。我们将展示三种主要的JAX转换：grad()用于计算梯度，jit()用于编译，vmap()用于自动矢量化。通过这三种转换，您已经可以构建不需要在集群上分布的自定义神经网络解决方案。您将开发一个完整的神经网络分类器，并了解JAX程序的结构。

本章提供了一个总体图片，突出了JAX的功能和基本概念。我们将在后续章节中详细解释这些细节。

本章的代码可以在书籍的代码存储库中找到：https://github.com/che-shr-cat/JAX-in-Action/blob/main/Chapter-2/JAX_in_Action_Chapter_2_MNIST_MLP_Pure_JAX.ipynb。

我在Google Colab笔记本上使用GPU运行时。这是开始使用JAX的最简单方法。Colab已经预装了JAX，并且JAX可以利用硬件加速。在本书的后面，我们还将使用Google TPU与JAX。因此，我建议在Colab笔记本中运行这段代码。

您可能在不同的系统上运行代码，因此可能需要手动安装JAX。要这样做，请查看官方文档中的安装指南（https://github.com/google/jax#installation）并选择最合适的选项。

 <!--more-->

## 2.1 一个玩具机器学习问题：手写数字分类

让我们尝试解决一个简单但非常有用的图像分类问题。图像分类的任务在计算机视觉任务中是无处不在的：您可以通过超市里的照片对食物进行分类，在天文学中确定星系的类型，确定动物物种或识别邮政编码中的数字。

想象一下，您有一组带有标签的图像，每个图像都被分配了一个标签，比如“猫”，“狗”或“人类”。或者一组带有数字0到9的图像，并带有相应的标签。这通常被称为训练集。然后，您希望编写一个程序，它将接收一个没有标签的新图像，并且您希望它能够有意义地为照片分配一个预定义的标签。您通常会在所谓的测试集上测试最终模型，该模型在训练过程中未见过该集合。有时还会有一个验证集，在训练过程中用来调整模型超参数并选择最佳参数。

在现代深度学习时代，您通常会训练一个神经网络来执行此任务。然后，您将训练好的神经网络集成到一个程序中，该程序将为网络提供数据并解释其输出。

分类是一种经典的机器学习任务，与回归一样，属于监督学习的范畴。在这两种任务中，您都有一个示例数据集（训练集），为每个示例提供了监督信号（因此被称为“监督学习”），表明每个示例的正确性。

在分类中，监督信号是一个类别标签，因此您必须区分固定数量的类别。例如，通过照片对狗的品种进行分类，通过文本对推文情绪进行分类，或者根据其特征和历史记录将特定卡交易标记为欺诈。

多类别分类的特殊情况称为二元分类，当您需要在仅两个类别之间区分一个对象时。类别可以是互斥的（比如动物物种）或不互斥的（比如为照片分配预定义的标签）。对于前者情况，它被称为多类别(multi-class)分类。对于后者，它是多标签(multi-label)分类。

在回归中，监督信号通常是一个连续的数字；您需要为新案例预测这个数字。例如，根据其他测量和因素预测某一点的室内温度，根据其特征和位置预测房屋价格，或者根据照片上的食物数量预测盘子上的食物量。

我们将使用一个众所周知的手写数字MNIST数据集（http://yann.lecun.com/exdb/mnist/）。图2.1展示了该数据集中一些示例图像。

让我们首先加载这个数据集，并准备在本章中开发的解决方案中使用它。 

<a>![](/img/jaxinaction/ch1/3.png)</a>
 
 
# 2.2 加载和准备数据集

如前所述，JAX不包含任何数据加载器，因为JAX倾向于专注于其核心优势。您可以轻松地使用TensorFlow或PyTorch的数据加载器；您更喜欢或更熟悉哪一个。JAX官方文档包含了这两种加载器的示例。我们将在此特定示例中使用TensorFlow数据集及其数据加载API。

TensorFlow数据集包含一个名为mnist的MNIST数据集版本。总共有70,000张图像。数据集提供了训练/测试分割，其中训练部分有60,000张图像，测试部分有10,000张图像。图像是灰度的，大小为28x28像素。

```
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
#tf.config.set_visible_devices([], device_type='GPU')

import tensorflow_datasets as tfds

data_dir = '/tmp/tfds'

# as_supervised=True gives us the (image, label) as a tuple instead of a dict
data, info = tfds.load(name="mnist",
                       data_dir=data_dir,
                       as_supervised=True,
                       with_info=True)

data_train = data['train']
data_test  = data['test']
```


在加载数据后，我们可以用以下代码查看数据集中的样本：

```
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

ROWS = 3
COLS = 10

i = 0
fig, ax = plt.subplots(ROWS, COLS)
for image, label in data_train.take(ROWS*COLS):
    ax[int(i/COLS), i%COLS].axis('off')
    ax[int(i/COLS), i%COLS].set_title(str(label.numpy()))
    ax[int(i/COLS), i%COLS].imshow(np.reshape(image, (28,28)), cmap='gray')
    i += 1

plt.show()
```

上面的代码生成了图2.1中的一幅图像。

<a>![](/img/jaxinaction/ch2/1.png)</a>

由于图像大小相同，我们可以类似地处理它们并将多个图像打包成批次。我们可能想要的唯一预处理是归一化。它将像素字节值（uint8）从整数值范围[0, 255]转换为浮点类型（float32），范围为[0,1]。

```
HEIGHT = 28
WIDTH  = 28
CHANNELS = 1
NUM_PIXELS = HEIGHT * WIDTH * CHANNELS
NUM_LABELS = info.features['label'].num_classes

def preprocess(img, label):
  """Resize and preprocess images."""
  return (tf.cast(img, tf.float32)/255.0), label

train_data = tfds.as_numpy(data_train.map(preprocess).batch(32).prefetch(1))
test_data  = tfds.as_numpy(data_test.map(preprocess).batch(32).prefetch(1))
```

我们要求数据加载器对每个示例应用预处理函数，将所有图像打包成大小为32的一组批次，并且还要预取一个新批次，而不必等待上一个批次在GPU上完成处理。

现在已经足够了，我们可以转而开发我们的第一个使用JAX的神经网络。

## 2.3 一个简单的JAX神经网络

我们从一个简单的前馈神经网络开始，称为多层感知机（MLP）。这是一个非常简单（而且相当不实用）的网络，选择它是为了展示重要概念而不增加太多复杂性。我们将在第11章中使用更高级的解决方案。

我们的解决方案将是一个简单的两层MLP，对神经网络来说是一个典型的“hello world”示例。

我们将开发的神经网络如图2.2所示。

<a>![](/img/jaxinaction/ch2/2.png)</a>
**图2.2 我们神经网络的结构。28x28像素的图像被展平为784像素，没有2D结构，并传递到网络的输入层。接着是一个具有512个神经元的全连接隐藏层，然后是另一个具有十个神经元的全连接输出层产生目标类别的激活。**

图像被展平成一个包含784个值的一维数组（因为一个28x28的图像包含784个像素），并且这个数组是神经网络的输入。输入层将这784个图像像素中的每一个映射到一个单独的输入单元。然后是一个具有512个神经元的全连接（或密集）层。它被称为隐藏层，因为它位于输入和输出层之间。512个神经元中的每一个都同时“查看”所有的输入元素。然后是另一个全连接层。它被称为输出层。它有十个神经元，与数据集中的类别数相同。每个输出神经元负责其类别。例如，神经元#0产生类别“0”的概率，神经元#1产生类别“1”的概率，依此类推。

每个前馈层实现了一个简单的函数y = f(x\*w+b)，由权重w组成，它将传入的数据x进行乘法运算，再加上偏置b，这个结果再加上偏置b。激活函数f()是应用于乘法和加法结果的非线性函数。

首先，我们需要初始化层参数。

### 2.3.1 神经网络初始化

在训练神经网络之前，我们需要用随机数初始化所有的b和w参数：

```
LAYER_SIZES = [28*28, 512, 10]
PARAM_SCALE = 0.01

def init_network_params(sizes, key=random.PRNGKey(0), scale=1e-2):
  """Initialize all layers for a fully-connected neural network with given sizes"""

  def random_layer_params(m, n, key, scale=1e-2):
    """A helper function to randomly initialize weights and biases of a dense layer"""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
```


在JAX中使用随机数与NumPy不同，因为JAX需要纯函数，而NumPy的随机数生成器（RNG）不是纯函数，因为它们使用了隐藏的内部状态。TensorFlow和PyTorch的RNG通常也使用内部状态。JAX实现了纯函数式的随机数生成器，我们将在第7章更深入地讨论它们。现在，了解你必须为每次随机化函数调用提供一个称为密钥的RNG状态，每次使用一个密钥，因此每次需要一个新密钥时，你将一个旧密钥拆分成所需数量的新密钥。

【译注：此部分介绍的很不清楚，请参考[官方文档：Pseudorandom numbers](https://jax.readthedocs.io/en/latest/random-numbers.html)】

### 2.3.2 神经网络前向传播

然后，你需要一个执行所有神经网络计算的函数，即前向传播。我们已经为b和w参数设置了初始值。唯一缺失的部分是激活函数。我们将使用jax.nn库中的流行的Swish激活函数。

激活函数是深度学习世界中的重要组成部分。激活函数为神经网络计算提供了非线性性。没有非线性性，多层前馈网络将等同于一个单神经元。因为简单的数学：输入的线性组合仍然是输入的线性组合，这就是单个神经元的作用。我们知道，单个神经元解决复杂分类问题的能力有限，仅限于线性可分任务（您可能听说过著名的XOR问题，线性分类器无法解决它）。因此，激活函数确保了神经网络的表达能力，并防止其崩溃到一个更简单的模型。

发现了许多不同的激活函数。该领域始于简单易懂的函数，如sigmoid或双曲正切。它们平滑且具有数学家喜爱的性质，如在每个点上可微分。

$$
\text{Sigmoid}(x)=\frac{1}{1+e^{-x}}
$$

$$
\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

然后出现了一种新类型的函数，ReLU或修正线性单元。ReLU不平滑，其在点x=0处的导数不存在。然而，实践者发现神经网络使用ReLU学习速度更快。

$$
\text{ReLU}(x) = \max(0, x)
$$

接着发现了许多其他激活函数，其中一些是通过实验发现的，另一些则是通过合理设计的。在流行的设计函数中，有高斯误差线性单元（GELUs，https://arxiv.org/abs/1606.08415）或缩放指数线性单元（SELU，https://arxiv.org/abs/1706.02515）。

在深度学习的最新趋势之一是自动发现。通常称为神经架构搜索（NAS）。该方法的思想是设计一个丰富但易于管理的搜索空间，描述感兴趣的组件。这可能是激活函数、层类型、优化器更新方程等。然后我们运行一个自动过程，智能地搜索这个空间。不同的方法也可能使用强化学习、进化计算，甚至是梯度下降。

Swish（https://arxiv.org/abs/1710.05941）函数就是通过这种方式发现的。

$$
\text{Swish}(x) = x \cdot \text{sigmoid}(\beta x)
$$

NAS是一个令人兴奋的故事，我相信JAX丰富的表达能力可以在这个领域做出重要贡献。也许我们的一些读者会在深度学习中取得一些令人兴奋的进步！

在这里，我们开发了一个前向传播函数，通常称为预测函数。它接受要分类的图像，并执行所有的前向传播计算，以产生输出层神经元上的激活。具有最高激活的神经元确定输入图像的类别（所以，如果最高激活在神经元#5上，那么根据最直接的方法，神经网络已经检测到输入图像包含手写数字5）。

```
def predict(params, image):
  """Function for per-example predictions."""
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = swish(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits
```


注意我们在这里传递参数列表的方式。这与PyTorch或TensorFlow中典型程序不同，其中这些参数通常隐藏在类中，函数使用类变量来访问它们。

注意神经网络计算的结构。在JAX中，你通常有两个函数用于神经网络：一个用于初始化参数，另一个用于将神经网络应用于某些输入数据。第一个函数将参数作为某种数据结构返回（这里是数组列表，以后将是称为PyTree的特殊数据结构）。第二个函数接受参数和数据，并返回将神经网络应用于数据的结果。这种模式将在未来多次出现，即使在高级神经网络框架中也是如此。

这就是全部。我们可以使用我们的新函数进行逐个示例的预测。在这里的清单2.6中，我们生成了一个与数据集大小相同的随机图像，并将其传递给predict()函数。您也可以使用数据集中的真实图像。由于我们的神经网络尚未训练，不要期望任何好的结果。在这里，我们只关注输出形状，我们看到我们的预测返回了十个类别的十个激活值的元组。

```
random_flattened_image = random.normal(random.PRNGKey(1), (28*28*1,))
preds = predict(params, random_flattened_image)
print(preds.shape)
```

输出：
```
(10,)
```

因此，一切看起来都还不错，但我们想要处理批量图像，而我们的函数只设计为处理单个图像。自动批处理可以在这里帮助！

## 2.4 vmap：自动向量化计算以处理批量

有趣的是，我们的predict()函数是设计用于单个项目的，如果我们传入一个图像批量，它将无法工作。在清单2.7中唯一重要的变化是我们生成了一个包含32个28x28像素图像的随机批量，并将其传递给predict()函数。我们还添加了一些异常处理，但这只是为了缩小错误消息的大小，并突出显示最关键的部分：

```
random_flattened_images = random.normal(random.PRNGKey(1), (32, 28*28*1))
try:
  preds = predict(params, random_flattened_images)
except TypeError as e:
  print(e)
```

输出：
```
dot_general requires contracting dimensions to have the same shape, got (784,) and (32,).
```

这并不奇怪，因为我们的predict()函数是矩阵计算的直接实现，这些计算假设特定的数组形状。错误消息表示函数收到了第一层权重的(512, 784)数组和传入图像的(32, 784)数组，而jnp.dot()函数无法处理它。它期望用于计算与权重数组的点积以获得512个激活的784个数字。新的批量维度（这里是32的大小）让它感到困惑。


在深度学习中，多维数组是神经网络及其层之间通信的主要数据结构。它们也被称为张量。在数学或物理学中，张量有着更严格和复杂的含义，如果您对张量感到困难，不要感到尴尬。在深度学习中，它们只是多维数组的同义词。如果您已经使用过NumPy，那么您几乎了解您所需的一切。

张量或多维数组有特定的形式。矩阵是一个具有两个维度（或称为二阶张量）的张量，向量是一个具有一个维度（一阶张量）的张量，标量（或只是一个数）是一个具有零维度（零阶张量）的张量。因此，张量是标量、向量和矩阵到任意维度（秩）的泛化。

例如，您的损失函数值是一个标量（仅一个数）。在分类神经网络输出的类别概率数组中，单个输入的向量大小为k（类别数），维度为1（不要混淆大小和秩）。一批数据（一次多个输入）的这种预测数组是大小为k*m的矩阵（其中k是类别数，m是批量大小）。RGB图像是一个三阶张量，因为它有三个维度（宽度、高度、颜色通道）。一批RGB图像是一个四阶张量（新维度是批量维度）。视频帧流也可以被视为一个四阶张量（时间是新的维度）。一批视频则是一个五阶张量，依此类推。在深度学习中，通常使用不超过4或5个维度的张量。

我们有哪些选项来解决这个问题呢？
首先，有一个简单的解决方案。我们可以编写一个循环，将批量分解成单独的图像并依次处理它们。这样可以解决问题，但效率不高，因为大多数硬件在单位时间内可以进行更多的计算。在这种情况下，硬件的利用率将会明显降低。如果您已经使用过MATLAB、NumPy或类似的工具，您就会熟悉向量化的好处。这将是问题的有效解决方案。

因此，第二个选择是重写并手动向量化predict()函数，使其能够接受批量数据作为输入。这通常意味着我们的输入张量将具有额外的批量维度，并且我们需要重写计算以使用它。对于简单的计算来说很简单，但对于复杂的函数来说可能会很复杂。

第三个选项是自动向量化。JAX提供了vmap()变换，将一个能够处理单个元素的函数转换为能够处理批量的函数。它将函数映射到参数轴上。

```
from jax import vmap
batched_predict = vmap(predict, in_axes=(None, 0))
```

这只是一个一行的代码。in_axes参数控制要映射的输入数组轴。它的长度必须等于函数的位置参数数量。None表示我们不需要映射任何轴，在我们的示例中，它对应于predict()函数的第一个参数params。这个参数在任何正向传递中保持不变，所以我们不需要对其进行批处理（然而，如果我们为每次调用使用单独的神经网络权重，我们会使用这个选项）。in_axes元组的第二个元素对应于predict()函数的第二个参数image。零的值表示我们要批处理第一个（零）维度。在假设批量维度位于张量中的其他位置的情况下，我们会将此数字更改为正确的索引。

现在我们可以将我们修改后的函数应用于一批数据，并产生正确的输出：

```
# `batched_predict` has the same call signature as `predict`
batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)
```

输出：
```
(32, 10)
```


请注意重要的一点。我们没有改变我们的原始函数。我们创建了一个新函数。

vmap是一个非常有益的转换，因为它可以解放您不必手动进行向量化。向量化可能不是一个非常直观的过程，因为您必须考虑矩阵或张量以及它们的维度。这对每个人来说都不容易，可能会是一个容易出错的过程，因此在JAX中拥有自动向量化功能非常棒。您为单个示例编写一个函数，然后借助vmap()为批处理运行它。

我们几乎完成了。我们创建了一个可以处理批处理的神经网络。现在唯一缺少的部分是训练。这里就是另一个令人兴奋的部分。我们需要对它进行训练。

## 2.5 Autodiff: 如何在不了解导数的情况下计算梯度

通常，我们使用梯度下降程序来训练神经网络。梯度下降是一种简单的迭代过程，用于寻找可微函数的局部最小值。对于可微函数，我们可以找到一个梯度，它表示函数变化最显著的方向。如果我们沿着梯度的反方向走，这就是最陡下降的方向。我们通过重复步骤来到达函数的局部最小值。

神经网络是由其参数（权重）确定的可微函数。我们希望找到一组权重的组合，使某些损失函数最小化，该函数计算模型预测与实际值之间的差异。如果差异较小，预测就更好。我们可以应用梯度下降来解决这个任务。

我们从一些随机权重、选择的损失函数和一个训练数据集开始。然后我们在训练数据集上（或数据集的某个批次上）重复计算损失函数相对于当前权重的梯度。在为神经网络中的每个权重计算梯度之后，我们可以沿着梯度的相反方向更新每个权重，从权重值中减去梯度的一部分。该过程在某个预定义的迭代次数后停止，当损失改善停止时或根据其他标准停止。

该过程在下图中可视化：

<a>![](/img/jaxinaction/ch2/3.png)</a>

这个损失函数是一个曲线，其中对于任何权重值都有相应的损失值。这个曲线也称为损失曲线或拟合景观（在更复杂的情况下，有多个维度时，这是有意义的）。在这里，我们从一些初始随机权重（$W_0$）开始，并经过一系列步骤，来到了对应于某个特定权重（$W_k$）的全局最小值。一个特殊的参数称为学习率决定了我们取多大或多小的梯度量。

重复这样的步骤，我们就会跟随通向损失函数局部最小值的轨迹。我们希望这个局部最小值与全局最小值相同，或者至少不会显著差。奇怪的是，这对神经网络很有效。为什么它如此有效是一个有趣的话题。

在上面的图像中，为了演示目的，我选择了一个“好”的起点，这样很容易到达全局最小值。从拟合景观的其他地方开始可能会导致局部最小值，图中只有几个局部最小值。

这种原始梯度下降程序有很多改进，包括带有动量的梯度下降和自适应梯度下降方法，如Adam、Adadelta、RMSProp、LAMB等。其中许多方法也有助于克服一些局部最小值。

我们将使用简单的小批量梯度下降，没有动量或自适应性，类似于深度学习框架中的原始随机梯度下降（SGD）优化器。与普通的SGD相比，我们只使用了一个改进，即指数衰减学习率。然而，在我们的情况下，这是不必要的。

要实施这样的过程，我们需要从参数空间中的某个随机点开始（在上一节初始化神经网络参数时已经完成了这一步）。然后我们需要一个损失函数来评估我们当前参数集在训练数据集上的表现。损失函数计算了模型预测值与训练数据集标签的实际值之间的差异。针对特定的机器学习任务有许多不同的损失函数，我们将使用一种简单的适用于多类别分类的损失函数，即分类交叉熵函数。


```
from jax.nn import logsumexp

def loss(params, images, targets):
  """Categorical cross entropy loss function."""
  logits = batched_predict(params, images)
  log_preds = logits - logsumexp(logits)
  return -jnp.mean(targets*log_preds)
```

这里我们使用logsumexp()函数，这是机器学习中常见的技巧，用于在不出现数值溢出或下溢问题的情况下对一组对数概率进行归一化。如果你想了解更多关于这个主题的信息，这里有一个合理的解释：https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/。

loss()函数需要真实值或目标值，以计算模型预测值与真实值之间的差异。模型的预测已经以类别激活的形式给出，其中每个输出神经元为相应的类别产生一些分数。目标值最初只是类别编号。对于类别“0”，它是数字0；对于类别“1”，它是数字1，依此类推。我们需要将这些数字转换为激活值，在这种情况下，会使用特殊的one-hot编码。类别“0”会产生一个激活值数组，其中数字1位于位置0，其它位置都是0。类别“1”会在位置1产生数字1，依此类推。这种转换将在loss()函数之外完成。

在定义好loss函数之后，我们准备实现梯度下降更新。

逻辑很简单。我们需要根据当前批次的数据计算损失函数相对于模型参数的梯度。这时就用到了grad()转换。这个转换接受一个函数（这里是loss函数），并创建一个函数来计算损失函数相对于特定参数的梯度，默认情况下是函数的第一个参数（这里是params）。

这与其他框架（如TensorFlow和PyTorch）有重要的区别。在那些框架中，通常在执行前向传播后获得梯度，并且框架会跟踪在感兴趣的张量上执行的所有操作。JAX采用了不同的方法。它转换你的函数，并生成另一个计算梯度的函数。然后，你通过向这个函数提供所有相关的参数（神经网络权重和数据）来计算梯度。

这里我们计算梯度，然后以梯度相反的方向更新所有参数（因此在权重更新公式中有负号）。所有的梯度都按照学习率参数进行缩放，学习率参数取决于epochs的数量（一个epoch是对训练集进行完整遍历）。我们采用了指数衰减的学习率，因此对于后续的epochs，学习率会低于前面的epochs：


```
from jax import grad

 
INIT_LR = 1.0
DECAY_RATE = 0.95
DECAY_STEPS = 5


def update(params, x, y, epoch_number):
  grads = grad(loss)(params, x, y)
  lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
  return [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(params, grads)]
```

在这个示例中，你不直接计算损失函数。你只计算梯度。在许多情况下，你也想跟踪损失值，JAX提供了另一个函数value_and_grad()，它可以计算函数的值和梯度。我们可以相应地修改update()函数：

```
from jax import value_and_grad
 
INIT_LR = 1.0
DECAY_RATE = 0.95
DECAY_STEPS = 5


def update(params, x, y, epoch_number):
  loss_value, grads = value_and_grad(loss)(params, x, y)
  lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
  return [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(params, grads)], loss_value
```

我们已经完成了所有需要的事情；最后一件事是为指定数量的epochs运行一个循环。为此，我们需要几个更多的实用函数来计算准确率，并添加一些日志记录以跟踪训练期间所有相关信息：

```
from jax.nn import one_hot

num_epochs = 25


def batch_accuracy(params, images, targets):
  images = jnp.reshape(images, (len(images), NUM_PIXELS))
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == targets)

def accuracy(params, data):
  accs = []
  for images, targets in data:
    accs.append(batch_accuracy(params, images, targets))
  return jnp.mean(jnp.array(accs))

import time

for epoch in range(num_epochs):
  start_time = time.time()
  losses = []
  for x, y in train_data:
    x = jnp.reshape(x, (len(x), NUM_PIXELS))
    y = one_hot(y, NUM_LABELS)
    params, loss_value = update(params, x, y, epoch)
    losses.append(loss_value)
  epoch_time = time.time() - start_time

  start_time = time.time()
  train_acc = accuracy(params, train_data)
  test_acc = accuracy(params, test_data)
  eval_time = time.time() - start_time
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Eval in {:0.2f} sec".format(eval_time))
  print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))
```

输出：

```
Epoch 0 in 36.39 sec
Eval in 8.06 sec
Training set loss 0.41040700674057007
Training set accuracy 0.9299499988555908
Test set accuracy 0.931010365486145
Epoch 1 in 32.82 sec
Eval in 6.47 sec
Training set loss 0.37730318307876587
Training set accuracy 0.9500166773796082
Test set accuracy 0.9497803449630737
Epoch 2 in 32.91 sec
Eval in 6.35 sec
Training set loss 0.3708733022212982
Training set accuracy 0.9603500366210938
Test set accuracy 0.9593650102615356
Epoch 3 in 32.88 sec
...
Epoch 23 in 32.63 sec
Eval in 6.32 sec
Training set loss 0.35422590374946594
Training set accuracy 0.9921666979789734
Test set accuracy 0.9811301827430725
Epoch 24 in 32.60 sec
Eval in 6.37 sec
Training set loss 0.354021817445755
Training set accuracy 0.9924833178520203
Test set accuracy 0.9812300205230713
```


我们用JAX训练了我们的第一个神经网络。似乎一切都运行正常，并且它可以解决MNIST数据集上手写数字分类的问题，准确率为98.12%。不错。

我们的解决方案每个epoch需要超过30秒，并且每个epoch的评估运行额外需要6秒。这算快还是慢呢？

让我们看看如何通过及时编译来进行改进。

## 2.6 JIT：将您的代码编译以加快速度

你已经实现了一个完整的用于手写数字分类的神经网络。如果你在GPU机器上运行它，它甚至可以利用GPU，因为在那种情况下，默认情况下所有张量都会放在GPU上。然而，我们可以让我们的解决方案变得更快！

之前的解决方案没有使用JIT编译和XLA提供的加速。让我们来做吧。

编译你的函数很简单。你可以使用jit()函数转换或@jit注解。我们将使用后者。

在这里，我们编译了两个最耗资源的函数，update()和batch_accuracy()函数。你只需要在函数定义之前添加@jit注解即可：

```
@jit
def update(params, x, y, epoch_number):
  loss_value, grads = value_and_grad(loss)(params, x, y)
  lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
  return [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(params, grads)], loss_value
          
          
@jit
def batch_accuracy(params, images, targets):
  images = jnp.reshape(images, (len(images), NUM_PIXELS))
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == targets)
```



重新初始化神经网络参数并重新运行训练循环后，我们得到了以下结果：

```
Epoch 0 in 2.15 sec
Eval in 2.52 sec
Training set loss 0.41040700674057007
Training set accuracy 0.9299499988555908
Test set accuracy 0.931010365486145
Epoch 1 in 1.68 sec
Eval in 2.06 sec
Training set loss 0.37730318307876587
Training set accuracy 0.9500166773796082
Test set accuracy 0.9497803449630737
Epoch 2 in 1.69 sec
Eval in 2.01 sec
Training set loss 0.3708733022212982
Training set accuracy 0.9603500366210938
Test set accuracy 0.9593650102615356
Epoch 3 in 1.67 sec
...
Epoch 23 in 1.69 sec
Eval in 2.07 sec
Training set loss 0.35422590374946594
Training set accuracy 0.9921666979789734
Test set accuracy 0.9811301827430725
Epoch 24 in 1.69 sec
Eval in 2.06 sec
Training set loss 0.3540217876434326
Training set accuracy 0.9924833178520203
Test set accuracy 0.9812300205230713

```

质量保持不变，但速度显著提高。现在一个epoch只需要近1.7秒，而不是32.6秒，评估运行需要近2秒，而不是6.3秒。这是一个相当显著的改进！

你可能也会注意到，前几次迭代比后续迭代花费的时间更长。因为编译发生在函数的第一次运行期间，所以第一次运行较慢。后续运行使用编译后的函数，速度更快。我们将在第5章更深入地探讨JIT编译。

我们已经完成了我们的机器学习问题。现在稍微多谈一些一般性的想法。

## 2.7 纯函数和可组合转换：为什么这很重要？

我们已经使用JAX创建并训练了我们的第一个神经网络。在这个过程中，我们强调了JAX与传统框架（如PyTorch和TensorFlow）之间的一些重要区别。这些区别基于JAX采用的函数式方法。

正如我们多次提到的那样，JAX函数必须是纯函数。这意味着它们的行为必须仅由它们的输入定义，并且相同的输入必须始终产生相同的输出。不允许包含影响计算的内部状态。同时，也不允许副作用。

纯函数之所以好有很多原因。其中包括易于并行化、缓存以及能够进行函数组合，例如jit(vmap(grad(some_function)))。调试也变得更容易。

我们注意到的一个关键区别与随机数相关。NumPy随机数生成器（RNG）是不纯的，因为它们包含内部状态。JAX明确将其RNG设置为纯函数。现在，需要随机性的函数会传入一个状态。因此，给定相同的状态，您将始终生成相同的“随机”数。所以请小心。我们将在第7章讨论RNG。

另一个重要的区别是，神经网络参数不隐藏在某个对象中，而是始终显式地传递。许多神经网络计算结构如下：首先，您生成或初始化参数；然后，将它们传递到使用它们进行计算的函数中。我们将在JAX顶层的高级神经网络库中看到这种模式，如Flax或Haiku。

神经网络参数成为一个独立的实体。这种结构赋予您很大的自由度。您可以实现自定义更新，轻松保存和恢复它们，并创建各种函数组合。

在使用JIT编译函数时，没有副作用尤其重要。如果忽略纯度，jit()编译和缓存一个函数可能会导致意想不到的结果。如果函数行为受某些状态的影响或产生副作用，那么编译版本可能会保存其第一次运行期间发生的计算，并在后续调用中重新生成它们，这可能不是您想要的结果。我们将在第5章更多地讨论这个问题。

最后，让我们总结一下JAX深度学习项目的外观。

## 2.8 JAX深度学习项目概述

一个典型的JAX深度学习项目如图2.4所示，包括以下几个部分：

* 用于特定任务的数据集。
* 数据加载器，用于读取数据集并将其转换为一系列批次。正如我们所说，JAX不包含自己的数据加载器，您可以使用优秀的工具从PyTorch或TensorFlow加载数据。
* 模型定义为一组模型参数和一个根据这些参数执行计算的函数（请记住，JAX需要纯函数，没有状态和副作用）。可以使用JAX原语定义该函数，或者可以使用像Flax或Haiku这样的高级神经网络库。
* 模型函数可以使用vmap()进行自动向量化，并用jit()进行编译。您还可以使用pmap()将其分发到计算机集群。
* 损失函数接收模型参数和一批数据。它计算损失值（通常是我们希望最小化的某种误差），但我们实际上需要的是损失函数相对于模型参数的梯度。因此，我们使用grad() JAX转换来获取梯度函数。
* 梯度函数在模型参数和输入数据上进行评估，并为每个模型参数生成梯度。
* 使用梯度来使用某种梯度下降过程更新模型参数。您可以直接更新模型参数，也可以使用来自独立库（例如Optax）的特殊优化器。
* 在运行了几个周期的训练循环后，您会得到一个经过训练的模型（更新后的参数集），可以用于预测或您为模型设计的任何其他任务。
* 您可以使用训练好的模型将其部署到生产环境中。有几种可用的选项。例如，您可以将模型转换为TensorFlow或TFLite，或者在Amazon SageMaker中使用模型。
* 项目的不同部分可以用生态系统中的模块替换，即使在这个典型项目中也是如此。如果您涉及更高级的机器学习主题，例如强化学习、图神经网络、元学习或进化计算，那么您将从JAX生态系统中添加更多特殊模块，并/或更改此通用方案的某些部分。

<a>![](/img/jaxinaction/ch2/4.png)</a>
*图2.4 JAX项目的高级结构，包括数据加载、训练和部署到生产*

这就是全部！我们已经准备好深入研究JAX的核心。

## 2.9 总结

* JAX没有自己的数据加载器；你可以使用来自PyTorch或TensorFlow的外部数据加载器。
* 在JAX中，神经网络的参数通常作为外部参数传递给执行所有计算的函数，而不是像在TensorFlow/PyTorch中通常做的那样存储在对象内部。
* vmap()转换将针对单个输入的函数转换为针对批处理的函数。
* 你可以使用grad()函数计算函数的梯度。
* 如果你需要函数的值和梯度，可以使用value_and_grad()函数。
* jit()转换使用XLA线性代数编译器编译你的函数，并生成能够在GPU或TPU上运行的优化代码。
* 你需要使用纯函数，没有内部状态和副作用，使你的转换能够正确工作。


