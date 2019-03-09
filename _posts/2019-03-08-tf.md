---
layout:     post
title:      "Tensorflow简明教程"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 深度学习
    - Tensorflow
---

本文介绍Tensorflow的基本概念。

 <!--more-->
 
**目录**
* TOC
{:toc}

 
本文会简要的介绍TensorFlow的基本概念，主要是Low level API中的概念。并通过一个简单的线性回归介绍这些概念的实际使用。

## 概述
TensorFlow中计算的定义和计算的执行是分开的。我们编写TensorFlow程序通常分为两步：定义计算图；使用session执行计算图。不过TensorFlow 1.5之后引入了Eager Execution，使得我们不需要定义计算图，直接就可以执行计算，从而简化代码尤其是简化调试。因为本课程不会用到Eager Execution，所以略过，有兴趣的读者可以参考Tensorflow官方文档。


## Tensor
Tensor就是一个n维数组，0维数组表示一个数(scalar)，1维数组表示一个向量(vector)，二维数字表示一个矩阵(matrix)，。。。

一个Tensor里的数据都是同一种类型的，比如tf.float32或者tf.string。数组的维度个数叫作rank，比如scalar的rank是0，矩阵的rank是2。数组每一维的大小组成的list叫作shape。比如下面是一些Tensor的例子：

```
3.0 # rank为0的tensor； 一个scalar，它的shape是[]（没有shape信息）,
[1., 2., 3.] # rank为1的tensor; 一个vector，它的shape是[3]
[[1., 2., 3.], [4., 5., 6.]] # 一个rank为2的tensor； 一个matrix，shape是[2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # rank为3的tensor，shape是[2, 1, 3]
```
TensorFlow使用numpy的ndarray来表示Tensor的值。有几种重要的特殊Tensor类型，包括：

* tf.Variable
* tf.constant
* tf.placeholder
* tf.SparseTensor

除了Variable，其它类型的Tensor都是不可修改的对象，因此在一次运算的执行时它只会有一个值。但是这并不是说每次执行是值是不变的，因为有些Tensor可能是有随机函数生成的，每次执行都会产生不同的值（但是在一次执行过程中只有一个值）。

Tensor支持常见的slice操作，比如下面的slice得到矩阵的第4列(下标从0开始)：
```
my_column_vector = my_matrix[:, 3]
```

另外Tensor经常使用的函数是reshape，用来改变它的shape，比如输入的图像可能是二维的矩阵比如[28,28]，但是我们如果使用全连接的网络需要把展开成一维的向量，那么我们可以这样：
```
# images是(batch, width, height, channel)
images = tf.random_uniform([32, 28, 28, 1], maxval=255, dtype=tf.int32)
print(images.shape) # (32, 28, 28, 1)
images = tf.reshape(images, [32, -1])
print(images.shape) # (32, 784)
```

另外一种常见的操作就是修改Tensor的数据类型，比如我们输入的图像是0-255的灰度值，我们需要把它变成(0,1)之间的浮点数：
```
images = tf.cast(images, dtype=tf.float32)
images = images/255.0
with tf.Session() as sess:
	print(sess.run(images[0]))
```

## 数据流图

TensorFlow的计算图使用数据流图来表示。图中有两种类型的对象：

* Operations(简称ops) 图中的点。
	
	Operation表示计算，它的输入和输出都是Tensor。
	
* Tensors 图中的边。
	
	Tensor在图中的“流动”表示了数据的变化和处理，这也是TensorFlow名字的由来。大部分TensorFlow函数都会返回Tensor。


需要注意的是，tf.Tensor并不存储值，它只是数据流图中的节点，它表示一个计算，这个计算会产生一个Tensor。比如下面的例子：
```
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # 也是tf.float32，通过4.0推测出来的类型。
total = a + b
print(a)
print(b)
print(total)
```
它的运行结果为：
```
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
```
print a,b和c并不会得到3,4和7。这里的a,b和c只是Graph中的Operation，执行这些Operation才会得到对应的Tensor值。每个Tensor都有一个数据类型dtype，tf.constant()函数会根据我们传入的值推测其类型，对于浮点数，默认类型是tf.float32。这和传统的编程语言有一些区别，对于c/c++/java语言来说，3.0这个字面量代表的是双精度浮点数(double或者TensorFlow的float64)，而对于Python来说只有双精度浮点数(类型叫float)。因为对于大部分机器学习算法来说，单精度浮点数以及够用了，使用双精度浮点数需要更多的内存和计算时间，而且很多GPU的双精度浮点数计算速度要比单精度慢几十倍（不同的架构差别很大，比如Nvidia的GTX系列双精度慢很多，但是Nvidia的Tesla系列差别较小），因此TensorFlow默认会把3.0推测为tf.float32。

比如我们如下最简单的代码：
```
import tensorflow as tf
a = tf.add(3, 4) 
```
我们使用TensorBoard(后面会介绍)可以看到实际的数据流图如<a href='#tf1'>下图</a>所示。在数据流中每一个点表示一个Operation，比如add，每一条边表示一个Tensor。读者可能会奇怪，哪里来的x和y呢？x和y是TensorFlow自动为我们创建了两个Tensor 3和4。因为add函数会把两个Tensor加起来，它需要两个Tensor作为参数，但是我们传入的是两个数字，因此TensorFlow会自动的帮我们创建两个Constant，并且命名为x和y。因此下面的代码和上面的是等价的：
```
x=tf.constant(3, name="x")
y=tf.constant(4, name="y")
a=tf.add(x,y)
```
注意constant不是一个Tensor，但是它内部保存了一个Tensor，当把它作为add的一个参数的时候，它们之间的边就表示把x内部的Tensor传给add。我们可以使用TensorBoard查看x的内容如下：
```
dtype {"type":"DT_INT32"}
value {"tensor":{"dtype":"DT_INT32","tensor_shape":{},"int_val":3}}
```


<a name='tf1'>![](/img/tf/tensorflow-1.png)</a>
*图：数据流图*


如果我们执行下面的代码：
```
import tensorflow as tf
a = tf.add(3, 4) 
print(a)
```
有点读者可能期望得到结果7，但是实际结果却是：
```
Tensor("Add:0", shape=(), dtype=int32)
```
原因就是add返回的就是数据流图中的一个Operation，我们只是“定义”了一个计算图，但是目前还没有“执行”它。那怎么执行它呢？我们需要创建一个Session对象，然后用这个Session对象来执行图中的某些Operation。比如下面的代码就会定义出计算的结果7。
```
import tensorflow as tf
a = tf.add(3, 4)
sess = tf.Session()
print(sess.run(a))
sess.close()
```
这有些麻烦，使用Eager Execution会简单一些，但是目前它不能完全替代这种方法。上面创建Session，然后关闭Session的写法可以使用with，这样不会忘了关闭它。
```
import tensorflow as tf
a = tf.add(3, 4)
with tf.Session() as sess:
	print(sess.run(a))
```

## Operation
Operation就是数据流图中的点，TensorFlow内置了常见的Operation，包括加减乘除等算术运算，大于小于等逻辑运算，Tensor的concat、reshape、slice等操作，矩阵的乘法、求逆操作，变量的赋值、自增等操作，Softmax、relu、conv2d等神经网络运算，用于保存模型checkpoint的save、reload等操作，队列的enqueue、dequeue等操作。大部分的TensorFlow函数返回都是一个Operation，它并不会立刻产生效果(执行)，而只是定义要做的事情。

对于常见的数学运算，比如加减乘除，为了使用方便，TensorFlow实现了Operator的重载，因此下面的代码最后两行的效果是一样的：
```
a=tf.constant(4)
b=tf.constant(5)
c=a+b
d=tf.add(a,b)
```

通常我们通过函数定义Operation之间的依赖关系，比如上面的代码，add产生的Operation会依赖a和b。但有的时候，我们需要某个Operation在其它的一些Operation之后再执行，但是它们并没有直接的函数关系，那么我们可以使用tf.Graph.control_dependencies来定义这种先后顺序关系。比如：
```
#graph g有5个ops: a, b, c, d, e
g = tf.get_default_graph()
with g.control_dependencies([a, b, c]):
	# 只有当a b c都执行和才会执行d和e。
	d = ...
	e = ...
```

需要注意的是只有在control_dependencies下创建的op才会建立这种依赖关系。比如下面的例子：
```
x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
	y = x
init = tf.initialize_all_variables()

with tf.Session() as session:
	init.run()
	for i in xrange(5):
		print(y.eval())
```
看起来计算y的时候会依赖x_plus_1操作，从而给x加一。那么最终似乎应该输出[0, 1, 2, 3, 4]。但是读者如果执行一下，会发现输出的却是5个0。原因在于y = x只是让变量y指向x一样的地址，并不会在Tensorflow的计算图里创建一个新的op。如果要实现上面的效果，我们可以使用tf.identity操作，这个操作实现赋值，它会在计算图里创建一个op用于赋值。把y = x改成就可以了。
```
y = tf.identity(x)
```

tf.identity有很多用途，其中之一是control_dependencies配合，和这个技巧在CIFAR10的多GPU例子能看到：
```
with tf.control_dependencies([loss_averages_op]):
	total_loss = tf.identity(total_loss)
```
total_loss已经定义好了，显然不能再定义一次了。但是我们又需要让它依赖loss_averages_op，那怎么办呢？tf.identity这个时候就派上用场了，它创建了一个新的op，返回的值和原来并没有不同。


## Graph
TensorFlow使用数据流图来表示计算之间的依赖关系。然后通过session来执行这个图的某个子图(当然也可以是整个图)来完成模型的训练或者预测。
### 为什么要使用数据流图
数据流图是并行编程的一种常见模型。在数据流图里，节点表示计算(operation)，而边表示数据(Tensor)。比如tf.matmul这个函数返回一个operation，这个operation的输入是两个矩阵(Tensor)，输出是这两个矩阵的乘积。使用数据流图有如下好处：


* 并行计算 通过边来显示的定义计算的依赖关系，从而让执行引擎更容易实现并行计算
* 分布式计算 同样的道理，数据流图使得分布式计算变得容易
* 编译 TensorFlow的XLA编译器能用数据流图里的信息来生成更快的代码
* 可移植性 通过数据流图定义了一种语言无关的模型从而可以实现语言和平台之间的移植。比如我们后面会介绍在生产环境常见的方式：使用Python来训练模型，然后使用SaveModel API保存模型，然后使用C++的Tensorflow Serving来提供实时的模型预测功能。



## 什么是tf.Graph
tf.Graph对象包括两部分信息：


* 图结构 
       
  包括点和边，分布代表计算和数据

* 集合 
	
	tf.add_to_collection可以把一个对象加到一个集合里。比如默认构造的变量都会放到全局变量集合GraphKeys.GLOBAL_VARIABLES里，这样我们调用tf.global_variables_initializer()时就知道哪些变量是需要初始化的。此外Optimizer默认是通过GraphKeys.TRAINABLE_VARIABLES来找到需要学习的模型参数。所有预定义的集合都在GraphKeys定义，当然我们也可以自己定义集合，注意不要和系统定义的冲突。






### tf.Graph的创建
我们一般不需要自己创建tf.Graph对象，TensorFlow会自动创建一个默认的Graph对象，我们的Operation默认会加到这个Graph里，当然我们自己创建这个对象，但一般是没有必要的。
```
# 有bug的代码！！！
import tensorflow as tf
g = tf.Graph()
with g.as_default():
	x = tf.add(3, 5)
sess = tf.Session(graph=g)
with tf.Session() as sess:
	# 会抛出异常，因为默认的Graph里没有x。
	print(sess.run(x))
```

比如上面的代码会有bug，我们自己创建一个Graph对象g，并且在里面增加了Operation x，然后我们用Session()函数创建了一个Session对象sess。因为默认的Session()函数会关联上默认的Graph，所以sess.run(x)会找不到x这个Operation。正确（但没必要这样写）的代码是：
```
import tensorflow as tf
g = tf.Graph()
with g.as_default():
	x = tf.add(3, 5)
sess = tf.Session(graph=g)
with tf.Session(graph=g) as sess:
	print(sess.run(x))
```
在构造Session时指定Graph为g，这样Session对象就会关联上我们构造的Graph对象g，从而可以找到Operation x并且执行它。我们可以构造多个Graph，在多个Graph里分别增加各自的Operation，但这通常是没有意义的，因为一个Session只能关联一个Graph。比如下面的代码很可能就是有问题的：
```
g = tf.Graph()
# a在默认的Graph里
a = tf.constant(3)
# b在我们创建的g里
with g.as_default():
	b = tf.constant(5)
```

### Graph的name space
后面的变量部分我们会详细的介绍name_scope和variable_scope的区别，这里通过例子简单的介绍name_scope。
```
c_0 = tf.constant(0, name="c")  # => 名字为"c"

# 已经有重名的对象了，Tensorflow会自动加上后缀
c_1 = tf.constant(2, name="c")  # => 名字为"c_1"

# 使用name scope，所有下面的变量会加上name scope为前缀
with tf.name_scope("outer"):
	c_2 = tf.constant(2, name="c")  # => 名字为"outer/c"
	
	# name scope的嵌套
	with tf.name_scope("inner"):
		c_3 = tf.constant(3, name="c")  # => 名字为"outer/inner/c"
	
	# outer/c已经有了，因此在变量名后面加后缀
	c_4 = tf.constant(4, name="c")  # => 名字为"outer/c_1"
	
	# name scope已经存在，会自动在name scope后面加后缀
	with tf.name_scope("inner"):
		c_5 = tf.constant(5, name="c")  # => 名字为"outer/inner_1/c"
```


## 常量
我们可以使用tf.constant()构造一个常量Operation，注意它返回的是一个Operation而不是Tensor。这个函数的原型是：
```
constant(value, dtype=None, shape=None, name="Const", verify_shape=False)
```
value是一个Tensor，表示常量的值。dtype是它的数据类型，我们可以传入shape。如果verify_shape是True，那么它会检查传入的value.shape是否shape一样，如果不一样就会抛出异常。value参数可以是Python数组或者numpy数组来，比如：
```
tf.constant([[0, 1], [2, 3]])
```
我们也可以用特定的值来构造一个常量：
```
tf.zeros([2, 3], tf.int32) ==> [[0, 0, 0], [0, 0, 0]]
tf.ones([2, 2], tf.float32) ==> [[1. 1.], [1. 1.]]
tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
```
另外我们也可以用lin_space和range来构造序列：
```
tf.lin_space(10.0, 13.0, 4) ==> [10. 11. 12. 13.]
tf.range(3, 18, 3) ==> [3 6 9 12 15]
```
我们还可以用函数来生成随机的常量Operation。常见的函数包括：


* tf.random_normal 生成正态分布的随机常量
* tf.truncated_normal 生成正态分布的常量，超出均值两倍标准差的会去掉并重新采样
* tf.random_uniform 生成某个区间均匀分布的常量
* tf.random_shuffle 随机打散一个Tensor
* tf.random_crop 随机crop一个Tensor的一部分
* tf.multinomial 多项分布的随机常量
* tf.random_gamma gamma分布的随机常量


常量的值(Tensor)是保存在图的定义之中的，我们前面也用TensorBoard看到过。这会带来一个问题——当常量很大的时候会使得图的加载变得很慢。

## placeholder和feeddict
在定义TensorFlow的计算图时，模型的参数通常都是定义为变量，此外还有一些不变的值定义为常量。但是还有一类特殊的值，那就是训练数据。它不是变量，因为它的生命周期就是一个batch，而不需要对它进行更新。同时它也不是常量，因为每个batch的值都是不同的。对于这类特殊的Tensor，我们通常用PlaceHolder来表示它。PlaceHolder顾名思义就是一个“占位符”，在定义图的时候不需要提供值（也不需要像变量那样提供初始值），只是定义它的类型和shape。但是在Session.run()的时候我们需要把值“feed”进去，从而表示这一个batch的训练数据。如果忘记feed值，TensorFlow会抛出运行时的异常。比如下面是常见的错误：
```
a = tf.placeholder(tf.float32, shape=[], name="a")
b = tf.constant(1, tf.float32, name="b")
c = a + b # short for tf.add(a, b)
with tf.Session() as sess: 
	print(sess.run(c))
	
# 会抛出异常： InvalidArgumentError (see above for traceback): 
# You must feed a value for placeholder tensor 'a' with dtype float and shape []
```
正确的代码是：
```
a = tf.placeholder(tf.float32, shape=[], name="a")
b = tf.constant(1, tf.float32, name="b")
c = a + b # short for tf.add(a, b)
with tf.Session() as sess:
	print(sess.run(c, feed_dict={a:2.0}))
```
我们定义了a是一个标量(scalar, shape是[])，因此我们需要在run的时候通过参数feed_dict传入a的实际值。上面的代码我们明确的指定了PlaceHolder的shape是一个标量，我们也可以不指定或者部分指定，比如：
```
a=tf.placeholder(tf.float32, shape=None, name="a") # 不是建议的用法
b=tf.placeholder(tf.float32, shape=[None, 3], name="b") # batch可变，这是常见用法
with tf.Session() as sess:
	sess.run(a, feed_dict={a:[1,2]})
	sess.run(a, feed_dict={a:[[1, 2],[3,4]]})
	sess.run(b, feed_dict={b:[[1,2,3]]})
	#下面的代码会抛出异常，因为b的shape是[None,3]，
	# 所以它一定是二维Tensor，但是第一维可以任意
	# sess.run(b, feed_dict={b:[1, 2, 3]})
```
使用None的好处是我们在run的时候可以随意feed任何shape的Tensor，理论上我们可以编写更加灵活的代码。但缺点是这会导致代码容易出错，并且调试变得困难。一般我们都建议指定PlaceHolder的大小，或者只让它的batch那个维度是None，从而可以传入不同大小的batch。除了PlaceHolder，我们也可以feed一个变量或者常量的值，这会使得本次run的时候这些常量或者变量的值使用我们的值，但是变量本身不会被修改。比如：
```
a=tf.placeholder(tf.float32, shape=[], name="a")
b=tf.constant(2.0, name="b")
c=tf.Variable(3.0)
d=a+b+c
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(d, feed_dict={a:1.0})) # 6.0
	print(sess.run(d, feed_dict={a:1.0, b:4.0, c:6.0})) # 11.0
	print(sess.run(c)) # 3.0
```
上面的代码，我们定义了PlaceHolder a，常量b和变量c（初始值3），d=a+b+c。初始化所有的变量后，我们可以执行d，传入a=1.0，这时可以计算出d是6.0。但是我们也可以feed进去常量b和变量c的值为4.0和6.0，这个时候计算出d是11.0。上面的run并不会改变变量c（当然更不会改变常量d）的值。



## 变量
### 基本概念
变量适合用来表示共享的持久化的Tensor，其中最常用的就是用来表示模型的参数。下面的代码可以创建变量：
```
m = tf.Variable([[0, 1], [2, 3]], name="matrix")
W = tf.Variable(tf.zeros([784,10]))
```

细心的读者可能会发现，tf.constant()的第一个字母是小写的，而tf.Variable()是大写。这是TensorFlow的开发者随意命名的结果吗？答案是否定的。tf.constant()返回的是一个Operation，而tf.Variable()返回的是一个对象Variable。Variable封装了很多Operation，比如initializer是这个变量的初始化Operation、value用于获得变量内部的Tensor、assign用于给变量赋值。除了tf.Variable()之外，我们也可以是函数tf.get_variable()来创建或者重用变量。

变量和常量不同，变量的生命周期是从创建开始一直到Session结束才结束。而且变量在使用前一定要初始化，因此下面的代码是不对的：
```
W = tf.get_variable("W", shape=(784, 10), initializer=tf.zeros_initializer())
with tf.Session() as sess:
	print(sess.run(W))
```
正确的代码为：
```
W = tf.get_variable("W", shape=(784, 10), initializer=tf.zeros_initializer())
with tf.Session() as sess:
	sess.run(W.initializer)
	print(sess.run(W))
```

通常一个图中会定义很多变量，一个个的调用很麻烦。我们可以使用tf.global_variables_initializer()，这个函数会返回图中所有的全局变量的初始化Operation，我们运行它就可以初始化所有的全局变量：
```
W = tf.get_variable("W", shape=(784, 10), initializer=tf.zeros_initializer())
b = tf.get_variable("b), shape=(10), initializer=tf.zeros_initializer())
c = tf.get_variable("c", shape=(), initializer=tf.zeros_initializer())
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(W))
```
那什么是“全局变量”呢？TensorFlow的变量可以属于一个或者多个（甚至0个）集合，所谓的集合只是为了把变量分组，使用起来方便。一个变量可以不属于任何集合，也可以同时属于多个集合。默认情况下，我们创建的变量都会加到一个叫作GLOBAL_VARIABLES的集合里，也就是“全局变量”。而tf.global_variables_initializer()返回的就是这个集合里的所有变量，我们可以查看这个函数的源代码来证实这一点：
```
@tf_export("initializers.global_variables", "global_variables_initializer")
def global_variables_initializer():
	return variables_initializer(global_variables())
	
@tf_export("global_variables")
def global_variables(scope=None):
	return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope)
```

我们也可以tf.variables_initializer初始化部分变量：
```
W = tf.get_variable("W", shape=(784, 10), initializer=tf.zeros_initializer())
b = tf.get_variable("b), shape=(10), initializer=tf.zeros_initializer())
c = tf.get_variable("c", shape=(), initializer=tf.zeros_initializer())
with tf.Session() as sess:
	sess.run(tf.variables_initializer([W, b]))
	print(sess.run(W))
```
上面的代码值初始化了变量W和b，c是没有初始化的。Variable.assign()函数可以给变量赋值(它返回一个赋值的Operation)，读者可以分析一下下面这段代码会输出什么？
```
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	print(W.eval())
```
答案是10，您答对了吗？如果您的答案是100，那么请注意：W.assgin(100)只是返回一个赋值100的Operation，但是我们并没有在Session里执行它（甚至没有把这个Operation保存下来。不过它仍然在计算图里，即使没有被执行）。如果需要赋值，正确的代码应该是：
```
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	sess.run(assign_op)
	print(W.eval())
```
变量的值是保存在Session中的，它的生命周期是超过一次Session的执行的，比如下面的代码：
``` 
my_var = tf.Variable(2, name="my_var") 
my_var_times_two = my_var.assign(2 * my_var)
with tf.Session() as sess:
	sess.run(my_var.initializer)
	sess.run(my_var_times_two) # my_var现在是4
	sess.run(my_var_times_two) # my_var现在是8
	sess.run(my_var_times_two) # my_var现在是16
```

另外变量是保存在Session里的，因此不同的Session里的变量是没有关系的。比如下面的代码：
```
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10))) # 20
print(sess2.run(W.assign_sub(2))) # 8
```

### tf.Variable()和tf.get_variable()的区别
TensorFlow的文档里推荐尽量使用tf.get_variable()。tf.Variable()必须提供一个初始值，这一般通过numpy(或者Python)的随机函数来生成，初始值在传入的时候已经确定（但还没有真正初始化变量，还需要用session.run来初始化变量)。而tf.get_variable()一般通过initializer来进行初始化，这是一个函数，它可以更加灵活的根据网络的结构来初始化。比如我们调用tf.get_variable()时不提供initializer，那么默认会使用tf.glorot_uniform_initializer，它会根据输入神经元的个数和输出神经元的个数来进行合适的初始化，从而使得模型更加容易收敛。

另外一个重要的区别就是tf.Variable()总是会(成功的)创建一个变量，如果我们提供的名字重复，那么它会自动的在后面加上下划线和一个数字，比如：
```
from __future__ import print_function
from __future__ import division

import tensorflow as tf
x1 = tf.Variable(1, name="x")
x2 = tf.Variable(2, name="x")
print(x1)
print(x2)

# tf.Variable会自动生成名字
x3 = tf.Variable(3)
print(x3)
```
输出的结果为：
```
<tf.Variable 'x_2:0' shape=() dtype=int32_ref>
<tf.Variable 'x_3:0' shape=() dtype=int32_ref>
<tf.Variable 'Variable:0' shape=() dtype=int32_ref>
```
而tf.get_variable()一定需要提供变量名(tf.Variable()可以不提供name，系统会自动生成)，而且它检测变量是否存在，默认情况下已经存在会抛出异常。因此tf.get_variable()强制我们为不同的变量提供不同的名字，从而让Graph更加清晰。除此之外，tf.get_variable()可以让我们更加容易的复用变量。

如果一个变量由tf.Variable()得到，那么共享(复用)它的唯一办法就是把这个变量传给需要用到的地方。这种方法看起来很简单自然，但是在实际的应用中却很麻烦。原因之一就是tf.layers里的layer(包括我们自己封装的类似的类)隐藏了很多细节，它创建的变量都是对象的成员变量，使用它的人甚至不知道它创建了哪些变量。比如tf.layers.Dense创建了一个矩阵kernle和一个bias(可选)，假设我们想让两个tf.layers.Dense的参数实现共享，那么tf.layers.Dense必须把所有的参数都对外暴露，而且我们在构造第二个tf.layers.Dense时需要传入。这就要求使用tf.layers.Dense的人了解代码的细节，这就破坏了封装的要求。比如哪天tf.layers.Dense的实现发生了改变，我们不用两个参数kernel和bias，而是把这两个参数合并为一个大的矩阵，那么所有用到它的地方都需要修改，这就非常麻烦。

因此TensorFlow通过get_variable()提供了另外一种通过名字来共享变量的方式，它需要和variable_scope配合。我们可以通过tf.variable_scope构造一个variable scope，然后通过variable scope来共享变量：
```
with tf.variable_scope("myscope") as scope:
	x = tf.get_variable(name="x", initializer=tf.constant([1, 2, 3]))
	print(x)
# 不加reuse=True会抛出异常，不允许创建同名的变量。
# with tf.variable_scope("myscope") as scope:
#    x = tf.get_variable(name="x", dtype=tf.int32, initializer=tf.constant([1, 2, 3]))

with tf.variable_scope("myscope", reuse=True) as scope:
	# 注意要加上dtype=tf.int32
	x = tf.get_variable(name="x", dtype=tf.int32, initializer=tf.constant([1, 2, 3]))
	# 我们可以不提供initializer
	# x = tf.get_variable(name="x", dtype=tf.int32)
	print(x)
```	
代码的输出是：
```
<tf.Variable 'myscope/x:0' shape=(3,) dtype=int32_ref>
<tf.Variable 'myscope/x:0' shape=(3,) dtype=int32_ref>
```
我们首先在名字为myscope的variable scope里创建变量x，打印它的名字发现Tensorflow自动在名字x前加上了scope的名字和/作为前缀。接着我们又在一个新的名字叫myscope的scope里创建变量x，需要注意的是这次调用tf.variable_scope是我们传入了参数reuse=True，如果不传这个参数，那么执行第二次get_variable会抛出异常，因为默认情况下变量是不共享的。另外一个需要注意的地方是我们在第二次调用tf.get_variable时传入了参数dtype=tf.int32，如果不传会怎么样呢？我们修改后运行一下会得到如下异常：
```
ValueError: Trying to share variable myscope/x, but specified dtype float32 and found dtype int32_ref.
```

为什么这样呢？因为我们第一次创建变量时传入了初始值[1,2,3]，TensorFlow会推测我们的变量的dtype是tf.int32，第二次是重用变量，它会忽略我们传入的initializer，这个时候它会任务dtype是默认的tf.float32，而之前我们创建的变量是tf.int32，这就发生运行时异常了。

除了在tf.variable_scope传入reuse=True，我们还可以用下面的方式来更加细粒度的控制变量共享：
```
with tf.variable_scope("myscope") as scope:
	x = tf.get_variable(name="x", initializer=tf.constant([1, 2, 3]))
	y = tf.get_variable(name="y1", initializer=tf.constant(1))
	print(x)
	print(y)

with tf.variable_scope("myscope") as scope:
	# y不共享
	y = tf.get_variable(name="y2", initializer=tf.constant(2))
	scope.reuse_variables()
	x = tf.get_variable(name="x", dtype=tf.int32)
	print(x)
	print(y)
```
它的输出是：
```
<tf.Variable 'myscope/x:0' shape=(3,) dtype=int32_ref>
<tf.Variable 'myscope/y1:0' shape=() dtype=int32_ref>
<tf.Variable 'myscope/x:0' shape=(3,) dtype=int32_ref>
<tf.Variable 'myscope/y2:0' shape=() dtype=int32_ref>
```
注意scope.reuse_variables()一旦调用后，这这个scope里的变量都是共享的，TensorFlow没有一个函数能够把它改成不共享的。因此我们首先需要把不共享的变量创建好，然后调用scope.reuse_variables()，接着再使用共享的变量。

另外需要提示读者的是：虽然我们示例代码的两个with tf.variable_scope("myscope") as scope隔得很近，但是在实际代码中这两行代码可以隔得很远，甚至不在一个文件里，因此这种方式的共享变量会很方便。

tf.layers里的layer都是通过这种方式来共享变量的，比如我们想共享两个Dense：
```
x1 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="x1")
x2 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="x2")
with tf.variable_scope("myscope") as scope:
	l1 = tf.layers.Dense(units=2)
	h11 = l1(x1)
with tf.variable_scope("myscope", reuse=True) as scope:
	l2 = tf.layers.Dense(units=2)
	h12 = l2(x2)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([h11, h12], feed_dict={x1: [[1, 2, 3]], x2: [[2, 4, 6]]}))
```
我们可以发现，虽然每次结果不同，但h12总是h11的两倍，这说明它们的参数确实是共享的。

### name scope和variable scope
除了variable scope，TensorFlow还有一个name scope。variable scope的目的是为了共享变量，而name scope的目的是为了便于组织变量的namespace。下面我们通过一个例子来比较这两个scope。
```
with tf.name_scope("my_scope"):
	v1 = tf.get_variable("var1", [1], dtype=tf.float32)
	v2 = tf.Variable(1, name="var2", dtype=tf.float32)
	a = tf.add(v1, v2)
	
	print(v1.name)  # var1:0
	print(v2.name)  # my_scope/var2:0
	print(a.name)   # my_scope/Add:0

with tf.variable_scope("my_scope"):
	v1 = tf.get_variable("var1", [1], dtype=tf.float32)
	v2 = tf.Variable(1, name="var2", dtype=tf.float32)
	a = tf.add(v1, v2)
	
	print(v1.name)  # my_scope/var1:0
	print(v2.name)  # my_scope_1/var2:0
	print(a.name)   # my_scope_1/Add:0
```
第一个scope是name_scope，在这里面通过tf.Variable()或者其它函数产生的变量的名字都会加上scope的名字为前缀，但是tf.get_variable()创建的变量会忽略name scope。因此v1的名字没有my_scope作为前缀。

第二个scope是variable_scope，所有的变量都会加上scope的名字。值得注意的是：name scope和variable scope可以同名。对于同名的scope，tf.Variable()或者其它函数产生的变量会自动给第二个scope加上数字的后缀以避免重名。但是对于tf.get_variable()，加的一定是scope的名字（而不会加后缀），如果有重名的而又不是reuse=True，那么就会抛异常。上面的例子中两个get_variable都是使用名字"var1"，但是第一个在name scope里，所以它的名字叫"var1:0"，而第二个在variable scope里，所以名字叫"my_scope/var1:0"。它们的名字其实是不同的。

总结一下：tf.get_varialbe()只会使用variable_scope而且变量名一定是scope名/变量名。而tf.Variable()会同时使用name_scope和variable_scope，并且在重名的时候通过后缀避免重名。而如果两个scope重名的时候(不管是name scope和variable scope重名还是两个name scope重名)，tf.Variable()发现重名变量时会给scope name加后缀(而不是给变量名加后缀)。这一点也可以从下面这个例子验证：
```

with tf.name_scope("my_scope"):
	v2 = tf.Variable(1, name="var2", dtype=tf.float32)
	print(v2.name)

with tf.name_scope("my_scope"):
	v2 = tf.Variable(1, name="var2", dtype=tf.float32)
	print(v2.name)
```
它的运行结果是：
```
my_scope/var2:0
my_scope_1/var2:0
```


## Session
TensorFlow使用tf.Session()对象表示客户端程序(通常是Python代码)和Tensorflow执行引擎(C++代码)之间的联系。实际的代码执行可能在本地的多个设备上执行(比如多个CPU和GPU)，也可能在远程的设备上原型。

我们定义好Graph之后TensorFlow并不会执行任何计算，为了进行计算，我们需要初始化一个tf.Session对象，通常我们把它叫作session。session封装了TensorFlow运行时的状态并且可以运行TensorFlow的Operation。我们可以把Graph看成Python源代码py文件，而tf.Session看成Python的可执行文件。比如下面的例子执行两个常量的相加：
```
import tensorflow as tf
a = tf.constant(3.0, name="a")
b = tf.constant(4.0, name="b")
c = tf.add(a, b)
with tf.Session() as sess:
	print(sess.run(c))
```

tf.Session()会返回一个Session对象。它封装了一个环境，在这个环境里Operation可以得到执行，Tensor可以求值。此外Session也会给变量分配空间。我们可以使用Session对象的run函数来执行想要的Operation，TensorFlow会自动执行它依赖的其它的Operation，这些要执行的Operation和它的依赖就构成了整个图的一个子图，不需要的Operation也不会被执行。比如：
```
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
z = sess.run(pow_op)
```
我们计算pow_op并不需要useless，因此TensorFlow也不会计算它的值。Session对象的run函数如下：
```
def run(fetches, feed_dict=None, options=None, run_metadata=None)
```
fetches参数表示要执行的一个或者多个Operation，feed_dict表示要feed的PlaceHolder，后面两个参数暂不介绍。在定义图的时候我们可以指定Operation放置的设备，这样可以实现分布式的计算。比如下面的代码把Operation指定到第一个GPU上：
```
with tf.device('/gpu:0'):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
	c = tf.multiply(a, b) 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
```
使用tf.device可以手动指定Operation的设备，"/gpu:0"表示第一个GPU，"/gpu:1"表示第二个GPU，"/cpu:0"表示放置在CPU上。注意没有"cpu:1"的写法，在TensorFlow里，CPU被看成“一个”设备，即使你有多个CPU，你也没有办法指定某个计算在哪个CPU上执行，因为目前的CPU架构一般是对应用层透明的，某个线程调度到哪个CPU上是由操作系统来决定的。


## 常见错误
一个常见的错误就是在run的时候“延迟”创建Operation，比如下面的代码：
```
x = tf.Variable(1, name='x')
y = tf.Variable(2, name='y')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for _ in range(10000):
		sess.run(tf.add(x, y))
```
上面的代码会在Graph里创建10000个add节点，这会造成极大的资源浪费。正确的代码是把图的定义和图的执行分开：
```
x = tf.Variable(1, name='x')
y = tf.Variable(2, name='y')
z = x + y
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for _ in range(10000):
		sess.run(z)
```

## layers
前面章节我们使用了最基本的Operation来定义全连接层，卷积层等，为了复用，我们对它做了封装。这些基本的网络结构在复杂的网络中被经常使用，如果大家都使用自己的封装，这会重复制造轮子。因此TensorFlow提供了tf.layers模块，它实现了大部分常见的网络结构，避免重复劳动。
下面我们简单的介绍tf.layers的使用，通过一个全连接的layer Dense来作为示例。后面我们会详细介绍更多layers的用法。

使用layers非常简单，和我们之前自己封装的类似，比如我们可以用tf.layers.Dense来实现一个线性模型，这个layer进行计算outputs = activation(inputs * kernel + bias)。下面是代码示例：
```
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```
layers需要输入和输出的大小来构造合适的参数。输出参数需要显示的指定，比如上面的代码输出的大小是1。而输入的大小它会从输入“推测”出来，因此如果输入是PlaceHolder，我们需要指定其大小(batch大小可能不指定)，比如上面我们指定了输入是[None, 3]，因此当执行y = linear_model(x)的时候，TensorFlow就知道需要构造全连接层的参数矩阵应该是[3, 1]的。

通过layers定义的Graph我们也需要初始化变量，layers封装的变量会放到全局变量里，因此我们可以通过global_variables_initializer来初始化变量：
```
init = tf.global_variables_initializer()
sess.run(init)
```
对于上面使用layers定义的Graph，我们可以用session来执行它：
```
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
```

如果我们多次运行上面的代码，其结果是不一样的，因为参数的初始化是随机的。我们可以在Dense的构造函数传入初始化函数，比如glorot_uniform_initializer。那么读者可能会问，上面的代码我们没有指定，那么tf.layers.Dense会使用哪个初始化函数呢？根据文档(https://www.tensorflow.org/api_docs/python/tf/layers/Dense)，bias使用的是零来初始化，而kernel使用了tf.get_variable()函数的默认初始化函数，而tf.get_variable()默认会使用 glorot_uniform_initializer。 glorot_uniform_initializer会用一个均匀分布来初始化变量，它的范围是(-limit, limit)，其中limit=sqrt(6 / (fan_in + fan_out))。

上面我们首先定义linear_model=tf.layers.Dense()，然后y=linear_model(x)，通常我们不需要用到linear_model这个Dense对象，我们也可以使用更加简化的写法：
```
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
```

## 线性回归的例子
下面我们通过一个简单的线性回归例子来学习上面的这些基本概念是怎么应用的。我们的线性回归非常简单，输入是一个浮点数x，输出是y，我们假设它们是一种简单的线性关系：y=wx+b。参数w和b是我们需要预测的值。
### 定义数据
我们首先定义输入x和真实的y：
```
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
```
\subsubsection{定义模型}
```
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))

```

我们用前面介绍的tf.layers.Dense来定义一个全连接层(没有激活函数，因此就是线性函数)。刚开始参数w和b是随机初始化的，因此预测出来的值很真实的值差别很大。

### 定义loss
为了优化模型，我们首先需要定义损失函数。我们可以使用基本的数学函数来定义损失，但是tf.losses封装了很多常见的损失函数，我们可以直接避免自己的实现错误，而且通常tf.losses提供的实现要比我们自己手写的高效和stable。这里我们使用最小均分误差损失函数：
```
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))
```

### 定义训练Op
定义了损失函数之后，我们就需要使用梯度下降算法来不断调整参数使得损失不断减少，我们当然可以自己求损失对参数的梯度。但这通常很繁琐而且容易出错，我们使用各种深度学习框架最主要目的(之一)就是使用自动梯度。而TensorFlow就是基于自动差分(auto diff)的框架，我们通过使用tf.optimizer.Optimizer对象(的子类)，然后使用这个对象的minimize()方法就可以产生一个Operation，通过session.run()就可以不断的执行梯度下降。我们只是简单的定义了Optimizer以及调用它的minimize()方法，TensorFlow在背后帮我们做了很多事情，包括在Graph中创建用于反向计算梯度的Operation。当然Optimizer也提供了一些底层的方法让我们可以计算梯度，然后自己来用梯度来修改变量。后面的一些复杂场景，我们会介绍这些底层的用法。
```
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```
上面的代码我们首先创建一个GradientDescentOptimizer对象，这是最简单的随机梯度下降方法的封装，我们需要传入learning rate，这里我们传入0.01。然后调用它的minimize(loss)方法得到一个train的Operation。如果我们用session.run来运行这个Operation，TensorFlow就会首先进行前向的计算，通过输入计算loss，然后计算loss对参数的梯度，然后用梯度和learning rate更新参数。所有这些过程都被封装好了！

### 进行训练
定义好了训练的Operation之后，接下来就要进行训练了，我们通过session.run(train)来进行训练。实际的训练我们通常是使用随机梯度下降(这里是使用了梯度下降，一次计算所有训练数据的梯度)，因此我们需要每次使用不同的batch大小的数据来训练，那就不能用tf.constant来定义训练数据了，而是要使用placeholder配合feeddict或者使用更加高效的tf.dataset API。
```
for i in range(100):
	_, loss_value = sess.run((train, loss))
	print(loss_value)
```


