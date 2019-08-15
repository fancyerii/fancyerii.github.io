---
layout:     post
title:      "Kaldi的聚类机制"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[Clustering mechanisms in Kaldi](http://kaldi-asr.org/doc/clustering.html)。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}
 
本文介绍Kaldi里通用的聚类机制和接口。

关于聚类的类和函数列表可以参考[这里](http://kaldi-asr.org/doc/group__clustering__group.html)。本文不涉及phonetic决策树的聚类(这个话题可以参考[决策树内幕](http://kaldi-asr.org/doc/tree_internals.html)和[Kaldi怎么进行决策树聚类](http://kaldi-asr.org/doc/tree_externals.html))。不过虽然不直接讨论phonectic决策树，但是它的底层代码会用到这里介绍的类和函数。

## Clusterable接口


[Clusterable](http://kaldi-asr.org/doc/classkaldi_1_1Clusterable.html)是一个纯虚类(接口)，而类[GaussClusterable](http://kaldi-asr.org/doc/classkaldi_1_1GaussClusterable.html)继承了这个接口(GaussClusterable代表高斯统计)。未来我们可能会增加其它继承自类型Clusterable的其它类型(目前只有这一种)。Clusterable类的目的是使得我们可以实现更加通用的聚类算法。


Clusterable接口的核心是累计统计信息和计算目标函数(objective function)。两个Clusterable对象的距离是通过分别计算其目标函数、把它们累加起来然后计算累加后的统计信息得到；目标函数减少量的取反就是距离的概念。
 

未来我们想增加的Clusterable实现类可能包括从一个固定的、共享的高斯混合模型的后验和离散观察的计数的集合得到的高斯混合模型统计信息(这个目标函数等价于分布的负熵乘以计数)。


下面是得到一个Clusterable(实际是GaussClusterable)对象的代码示例：
```
Vector<BaseFloat> x_stats(10), x2_stats(10);
BaseFloat count = 100.0, var_floor = 0.01;
// initialize x_stats and x2_stats e.g. as
// x_stats = 100 * mu_i, x2_stats = 100 * (mu_i*mu_i + sigma^2_i)
Clusterable *cl = new GaussClusterable(x_stats, x2_stats, var_floor, count);
```

在介绍上面的代码含义之前我们来看一下GaussClusterable的主要代码：

```
class GaussClusterable: public Clusterable {
 public:
  GaussClusterable(const Vector<BaseFloat> &x_stats,
                   const Vector<BaseFloat> &x2_stats,
                   BaseFloat var_floor, BaseFloat count);

 private:
  double count_;
  Matrix<double> stats_; // two rows: sum, then sum-squared.
  double var_floor_;  // should be common for all objects created.

  void Read(std::istream &is, bool binary);
};
```

对于一个协方差矩阵为对角矩阵的(也就是各个维度是独立的)多维高斯分布来说，它的充分统计量就是观察(样本)的个数、和以及平方和。有了这三个量就可以计算高斯分布的均值和方差，而且也可以"合并"两个观察集合的数据。举一个一维高斯分布的例子，假设第一次我们观察到了3个样本：

```
0.1
0.3 
0.5
```

则我们的三个统计量是：样本数为3；和为0.9，平方和为0.35。那么我们可以计算均值和方差分别为：$\hat{x}=\frac{x_1+x_2+x_3}{3}=0.3$；方差为：$\hat{\sigma^2}=\frac{1}{3-1}((0.1-0.3)^2+(0.3-0.3)^2+(0.5-0.3)^2)=0.04$。


类似的我们又观察到2个样本：
```
0.2 
0.4
```

我们也可以计算其均值和方差为：0.3和0.02。

如果我们把这两次观察合并成一个数量为5的观察集合：
```
0.1
0.3
0.5
0.2
0.4
```

则其均值为：0.3和0.025。

这样计算需要保存所有的训练数据，这个量是非常大的。比如我们把这两个集合分别从3和2扩大到300,000和200,000，那么我们需要保存500,000个点。但是只要我们保存了这两个集合的3个(样本数、和以及平方和)统计量，我们就可以计算合并后的均值和方差。

理解了这个之后我们再看GaussClusterable的代码就很简单了，它有3个private的成员变量。count_代表样本数；stats_是一个(2, dim)的矩阵，第一行存储观察的和(这里是多维(dim)的，前面我们举的例子是一维的)，第二行存储平方的和；而var_floor_我们暂时先忽略。

接下来我们看最主要的构造函数：

```
inline GaussClusterable::GaussClusterable(const Vector<BaseFloat> &x_stats,
                                          const Vector<BaseFloat> &x2_stats,
                                          BaseFloat var_floor, BaseFloat count):
    count_(count), stats_(2, x_stats.Dim()), var_floor_(var_floor) {
  stats_.Row(0).CopyFromVec(x_stats);
  stats_.Row(1).CopyFromVec(x2_stats);
}
```

需要传入样本数count、和x_stats以及平方和x2_stats。


## 聚类算法


我们实现了一些通用的聚类算法。所有实现的算法列表可以参考[Algorithms for clustering](http://kaldi-asr.org/doc/group__clustering__group__algo.html)。在这些算法里被经常使用的一个数据结构是一个指向Clusterable接口的指针组成的vector：

```
std::vector<Clusterable*> to_be_clustered;
```


这个vector的下标是被聚类的"点"的下标，这里点用引号的意思是它可能不只是一个点，而是多个点的集合。


### K-means以及类似K-means接口的聚类算法
 
调用聚类算法的代码典型示例为：
```
std::vector<Clusterable*> to_be_clustered;
// initialize "to_be_clustered" somehow ...
std::vector<Clusterable*> clusters;
int32 num_clust = 10; // requesting 10 clusters
ClusterKMeansOptions opts; // all default.
std::vector<int32> assignments;
ClusterKMeans(to_be_clustered, num_clust, &clusters, &assignments, opts);
``` 

聚类代码调用完成之后，变量assignments会告诉你to_be_clustered中的每一个点最终聚到哪个聚类里了。即使对于大量的数据点，[ClusterKMeans](http://kaldi-asr.org/doc/group__clustering__group__algo.html#ga69018185a19b01f73228875aa07e3135)算法也是非常的高效。


还有两个聚类算法的参数与ClusterKMeans类似，它们是[ClusterBottomUp](http://kaldi-asr.org/doc/group__clustering__group__algo.html#gae13c55f284aeed0245a059489a0f1960)和[ClusterTopDown](http://kaldi-asr.org/doc/group__clustering__group__algo.html#gae6c80e7da6b44a697295be2de67ea1e7)。可能更有用的是ClusterTopDown，当聚类数量很大的时候它应该比ClusterKMeans更快(它是类似二叉的决策树)。内部它会调用TreeCluster，请参考下面的介绍。

### 树聚类算法


函数[TreeCluster](http://kaldi-asr.org/doc/group__clustering__group__algo.html#gad2937c106b02a420e044390b465014fa)通过二叉树来聚类数据点(叶子节点不一定只有一个点，你可以指定最大的叶子数量)。这个函数在构建自适应的回归树时很有用。参考函数的文档来了解更多它的输出格式的详细解释。简洁的解释是：它的叶子和非叶子节点是拓扑排序的，叶子节点在前而跟节点在最后，它会输出一个vector告诉你每个节点的父亲节点是谁(这样就可以重构出这棵树)。
