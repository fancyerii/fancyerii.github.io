---
layout:     post
title:      "Kaldi的矩阵库"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
---

本文的内容主要是翻译文档[The Kaldi Matrix library](http://kaldi-asr.org/doc/matrix.html)，介绍Kaldi的矩阵库。更多本系列文章请点击[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})。
 <!--more-->
 
**目录**
* TOC
{:toc}
 
Kaldi的矩阵库是对标准的BLAS[http://www.netlib.org/blas/]和LAPACK[http://www.netlib.org/lapack/]线性代数库的C++封装。

本文档介绍使用这些库的概述。代码级别的文档请参考[Matrix and vector classes](http://kaldi-asr.org/doc/group__matrix__group.html)，怎么使用这些外部矩阵库的解释请参考[External matrix libraries](http://kaldi-asr.org/doc/matrixwrap.html)。

## 矩阵和向量类型

最重要的类是Matrix和Vector，它们的用法示例为：
```
Vector<float> v(10), w(9);
for(int i=0; i < 9; i++) {
   v(i) = i; 
   w(i) = i+1;
}
Matrix<float> M(10,9);
M.AddVecVec(1.0, v, w);
```
 

上面的代码首先个构造大小分别为10和9的向量v和w。我们使用括号"()"来索引向量；而矩阵则是使用类似M(i,j)的方法。M初始化成10 x 9的矩阵，最后以后是计算向量v和w的外积$vw^T$然后存放到M。函数的命名模式是根据计算公式，比如：

$$
M \leftarrow M + \alpha vw^T
$$


上式第一个运算是+，也就是add，然后是向量v，然后是另一个向量w，因此函数名叫做AddVecVec。M并没有出现在函数名中，因为这是成员函数，M就是this。函数的参数顺序也是与公式一致的，首先是$\alpha$，然后是v，然后是w。

注意：我们必须保证M的shape是和$vw^T$一致的(这里我们是通过构造函数来得到(10,9)的矩阵，我们也可以使用resize函数来把一个shape不一致的变成我们想要的)。但是如果我们的shape设置的不对，矩阵库不会尝试帮我们纠正它，它只会(运行时)crash(有assert)。比如下面我故意把M设置成(1,9)，那么运行的结果为：
```
ASSERTION_FAILED ([5.4.232~3-532f3]:AddVecVec<float>():kaldi-matrix.cc:121) : 'a.Dim() == num_rows_ && rb.Dim() == num_cols_' 

[ Stack-Trace: ]

kaldi::MessageLogger::HandleMessage(kaldi::LogMessageEnvelope const&, char const*)
kaldi::MessageLogger::~MessageLogger()
kaldi::KaldiAssertFailure_(char const*, char const*, int, char const*)
void kaldi::MatrixBase<float>::AddVecVec<float>(float, kaldi::VectorBase<float> const&, kaldi::VectorBase<float> const&)
void kaldi::BasicTest<float>()
./my-test() [0x4015f7]
main
__libc_start_main
_start

已放弃 (核心已转储)
```

接下来我们看另外一个公式：

$$
A=\alpha BC^T+\beta A
$$

读者能猜出这个函数的名字吗？这有点困难，它的用法是：
```
A.AddMatMat(alpha, B, kNoTrans, C, kTrans, beta);
```

kNoTrans表示对于B不需要进行转置，而C后面的kTrans说明需要对C进行转置。下面是示例代码：
```
  Matrix<Real> M(5, 10), N(5, 10), P(5, 5);
  // initialize M and N somehow...
  // next line: P := 1.0 * M * N^T + 0.0 * P.
  P.AddMatMat(1.0, M, kNoTrans, N, kTrans, 0.0);
  // tr(M N^T)
  float f = TraceMatMat(M, N, kTrans),
        g = P.Trace();
  KALDI_ASSERT(f == g);  // we use this macro for asserts in Kaldi 
```

对于BLAS来说，没有函数直接实现两个矩阵A和B的乘法，要实现的话就只能用上面这个函数。也就是说BLAS定义了很多更加通用的计算公式，我们如果要实现某些运算就需要把通用的公式的某些参数设置为0或1等(或者其它特殊值)来实现。


如果像理解Matrix和Vector的更多函数，请参考[MatrixBase](http://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html)和[VectorBase](http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html)。或者参考代码matrix/kaldi-vector.h里的VectorBase类或者matrix/kaldi-matrix.h里的MatrixBase类。


## 对称矩阵和三角矩阵
 

对于对称矩阵有专门的类[SpMatrix](http://kaldi-asr.org/doc/classkaldi_1_1SpMatrix.html)，三角矩阵也有专门的[TpMatrix](http://kaldi-asr.org/doc/classkaldi_1_1TpMatrix.html)。它们在内存中都是有紧凑的下三角阵的方式保存并且都继承了[PackedMatrix](http://kaldi-asr.org/doc/classkaldi_1_1PackedMatrix.html)。在Kaldi里，SpMatrix是最有用的(TpMatrix只是用于计算Cholesky分解)。使用这些类的典型示例如下面所示。注意"AddMat2"函数。这里展示了一种新的命名模式：当一个量出现两次的时候，我们在函数名的响应部分添加"2"。

```
Matrix<float> feats(1000, 39);
// ... initialize feats somehow ...
SpMatrix<float> scatter(39);
// next line does scatter = 0.001 * feats' * feats.
scatter.AddMat2(1.0/1000, feats, kTrans, 0.0);
TpMatrix<float> cholesky(39);
cholesky.Cholesky(scatter);
cholesky.Invert();
Matrix<float> whitened_feats(1000, 39);
// If scatter = C C^T, next line does:
//  whitened_feats = feats * C^{-T}
whitened_feats.AddMatTp(1.0, feats, kNoTrans, cholesky, kTrans, 0.0);
```

上面的代码必须初始化feats，如果全是零的话代码会crash。


## SubVector和SubMatrix


如果我们想使用一个向量或者矩阵的一部分，那么SubVector和SubMatrix就非常适合这种场景。和Vector与Matrix类似，SubVector与SubMatrix也分别继承了VectorBase和MatrixBase，因此它们的方法都不会处理resize(而且SubVector和SubMatrix也不能改变大小，因为它是类似于一个指向其它Vector与Matrix的"指针"。


下面是一个示例：
```
Vector<float> v(10), w(10);
Matrix<float> M(10, 10);
SubVector<float> vs(v, 1, 9), ws(w, 1, 9);
SubMatrix<float> Ms(M, 1, 9, 1, 9);
// next line would be v(2:10) += M(2:10,2:10)*w(2:10) in some
// math scripting languages.
vs.AddMatVec(1.0, Ms, kNoTrans, ws);
```

"SubVector\<float> vs(v, 1, 9)"的意思是从v的下标1开始，截取长度为9的向量。也就是从v的第二个元素一直到最后一个。

There are other ways to obtain these types. If M is a matrix, M.Row(3) will return row 3 of M, as a SubVector. There is a corresponding constructor, for example: 

除了上面的方法，还有一些方法得到它们。如果M是一个矩阵，M.Row(3)会返回M的第三行，返回值的类型就是SubVector。也可以通过构造函数得到一个矩阵的一行，比如：
```
SubVector row_of_m(M, 0); // 得到M的第一行。
```


但是我们无法通过类似的方法得到矩阵的某一列。因为向量在内存中是连续的；我们目前无法实现一个"stride"这样的成员(也就是相邻元素的间隔，虽然BLAS是支持的)。另外一种获得SubVector或者SubMatrix的方法是使用Range函数。比如：

```
// 得到v的前5个元素，然后置为零。 
v.Range(0, 5).SetZero();
// 从下标(5,5)得到一个2x2的矩阵，然后置为零。 
M.Range(5, 2, 5, 2).SetZero(); 
```


使用SubVector和SubMatrix时必须很小心。比如，你创建了一个SubVector然后destroy或者修改了它指向的Vector或者Matrix的大小。那么SubVector可能就是"野指针"了。此外，SubVector和SubMatrix也不是只读的(按理应该这样设计)，因此你修改了SubVector也会同时修改了被他们指向的对象。如果要fix这些问题会使得代码变得很复杂，因此我们只能要求使用它的人小心了。

## 向量和矩阵的调用习惯
 
一般来说，当一个函数需要一个向量或者矩阵作为参数，我们通常使用基类，也就是VectorBase\<BaseFloat>或者MatrixBase\<BaseFloat>而不是Vector\<BaseFloat>或者Matrix\<BaseFloat>。这样的话我们可以传入SubVector或者SubMatrix(它们也是VectorBase和MatrixBase的子类)。但是如果代码需要修改向量或者矩阵的大小的时候就会要求传入Vector或者Matrix的指针，比如Vector\<BaseFloat>\* 或者Matrix\<BaseFloat>\*；这里不能传入引用，因为resize是非const的操作，根据我们的coding style，非const应用的参数是不允许的。

## 拷贝向量和矩阵
 
有很多方法来拷贝向量和矩阵。最简单是使用CopyFrom函数，比如[Matrix::CopyFromMat](http://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html#a557fe407783c0d8f2e604666fd48f96d), [Vector::CopyFromVec](http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html#a959973067b5f0a4b205c9d8385fa33df), [SpMatrix::CopyFromSp](http://kaldi-asr.org/doc/classkaldi_1_1SpMatrix.html#a1477d48c6104d9223a230acdc706a349)等等。这些函数甚至可以在float类型的矩阵和double类型的矩阵之间拷贝，这种跨类型的操作在其它函数是很少允许的。但是这些函数也不会为你自动resize，因此如果shape不对的话就会crash(你需要自己使用Vector::Resize, Matrix::Resized等函数来设置合适的shape)。也可以在不同的子类之间拷贝：比如Matrix::CopyFromTp, SpMatrix::CopyFromMat, TpMatrix::CopyFromMat等等。具体内容请参考其文档。

通常也存在和上面的函数对应的构造函数。这些构造函数会复制数据并且自动帮我们设置shape，比如：

```
Matrix<double> M(10, 10);
... 初始化 M ...
// 把M拷贝到Mf 
Matrix<float> Mf(M);
```

此外还有一些特殊作用的拷贝函数。你可以使用[Vector::CopyRowsFromMat](http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html#ae07f9efcd41c79573aef065065b12f28)来把一个矩阵的行拼接成一个大的向量拷贝，类似的有[Vector::CopyColsFromMat](http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html#aef004252892d6c81e9343453474a0d2a)，也可以通过[Matrix::CopyRowsFromVec](http://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html#a0db477a6b3daf0a83142ae056256e1e3)和[Matrix::CopyColsFromVec](http://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html#aa2ffb845eb1004d74208c9c5eaa11435)反过来把向量拷贝成矩阵。

也有只拷贝一行或者一列的版本，读者可以参考[ Vector::CopyRowFromMat](http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html#af64f4a3d1595f0bd1b11ab0fc40f9578),[Vector::CopyColFromMat](http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html#a7c1e46a5355ae05c048e7c236406fe50)和[Matrix::CopyRowFromVec](http://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html#a0fb8276b2c7f0ccfd996c37df4298568),[Matrix::CopyColFromVec](http://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html#a109e99b3999f5a7d94e2df2c8ed180de)。

## 标量乘法


有一些函数返回一个(些)标量（并且它们也不修改输入参数)，这些函数不是定义为成员函数。[Matrix-vector functions returning scalars](http://kaldi-asr.org/doc/group__matrix__funcs__scalar.html)列举了所有的这样的函数，下面是一个例子：
```
float f = VecVec(v, w), // v' * w 
      g = VecMatVec(v, M, w),  // v' * M * w
      h = TraceMatMat(M, N, kNoTrans); // tr(M * N)
```
上面的三个例子分别是内积、向量乘以矩阵再乘以向量和trace函数，它们的返回值都是标量。

## Resizing
除了SubMatrix和SubVector，其它的矩阵和向量类型都可以改变大小。比如下面的例子：
```
Vector<float> v;
Matrix<float> M;
SpMatrix<float> S;
v.Resize(10);
M.Resize(5, 10);
S.Resize(10);
```


Resize函数会把所有的数据都清零，除非你提供一个可选的初始选项。可能的初始选项包括：

* kSetZero (默认值): 清零
* kUndefined： 不初始化(内容是未定义的，我们不能假设它会保留原来的值，因为它可能申请新的内存)
* kCopyData: 原有的数据保持不变，新增加(如果有的话)清零


因此代码：
```
v.Resize(v.Dim() + 1, kCopyData);
```
它会给v增加一个零，但这不是高效的做法，因为它会重新申请一块新的内存然后复制之前的数据。

构造函数也可能会自己设置大小，因此也有初始选项，它的含义和Resize是一样的。

## 矩阵I/O

矩阵I/O的风格和其它Kaldi代码一致(参考[Kaldi I/O mechanisms](http://kaldi-asr.org/doc/io.html))。一个典型的读写示例为：
```
bool binary = false;
std::ofstream os( ... );
Vector<float> v;
v.Write(os, binary);
...
std::ifstream is( ... );
Vector<float> v;
v.Read(is, binary);
```

对于文本格式的输入和输出，也可以使用 \<\<和\>\>实现一样的事情，但是它们大部分只用于debug目的。文本格式的向量类似于：
```
[ 2.77914 1.50866 1.01017 0.263783 ] 
```
而文本格式的矩阵为：
```
[ -0.825915 -0.899772 0.158694 -0.731197 
 0.141949 0.827187 0.62493 -0.67114 
-0.814922 -0.367702 -0.155766 -0.135577 
 0.286447 0.648148 -0.643773 0.724163 
] 
```

## 矩阵库的其它函数

本文档只是矩阵库的简单介绍。大部分数学运算都是类[MatrixBase](http://kaldi-asr.org/doc/classkaldi_1_1MatrixBase.html), [VectorBase](http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html), [SpMatrix](http://kaldi-asr.org/doc/classkaldi_1_1SpMatrix.html)和[TpMatrix](http://kaldi-asr.org/doc/classkaldi_1_1TpMatrix.html)的成员函数。返回值是标量的函数列表在[这里](http://kaldi-asr.org/doc/group__matrix__funcs__scalar.html)。一些其它的(misc)函数比如傅里叶变换和指数函数等等可以在[这里](http://kaldi-asr.org/doc/group__matrix__funcs__misc.html)查看。注意：我们并没有优化所有的函数。我们的方法是快速实现我们需要的功能，只有当某些函数在某些常见被大量使用时才做特殊的优化。
