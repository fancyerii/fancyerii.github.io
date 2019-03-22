---
layout:     post
title:      "Faster R-CNN代码简介"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 计算机视觉
    - Faster R-CNN
    - 代码
    - 《深度学习理论与实战：提高篇》
---

本文简单的介绍Faster R-CNN代码的使用。更多文章请点击<a href='/tags/#《深度学习理论与实战：提高篇》'>《深度学习理论与实战：提高篇》</a>。
<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>
 <!--more-->
 
**目录**
* TOC
{:toc}

Faster R-CNN有很多开源的版本，我们这里介绍[PyTorch实现](https://github.com/jwyang/faster-rcnn.pytorch.git)的用法。前面介绍过原理，这里就不分析源代码了，有兴趣的读者开源自己阅读源代码。

## 安装
建议使用virtualenv安装。

```
# 获取代码
git clone https://github.com/jwyang/faster-rcnn.pytorch.git
#或者使用作者fork的版本，保证代码和作者使用的一致
# git clone https://github.com/fancyerii/faster-rcnn.pytorch.git

# 安装virtualenv
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate

# 安装pytorch 0.4.0(注意这个实现只支持0.4.0，不能安装0.4.1或者更新版本)
# whl包需要去PyTorch的官网下载，更加自己的Python版本，GPU进行选择合适的下载安装
# 这里有老的版本下载： https://pytorch.org/get-started/previous-versions/

# 安装其它依赖
pip install -r requirements.txt
```

## 数据准备

有很多数据集可以选择，我们这里使用PASCAL VOC数据集，这是[官网](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)。

```
cd faster-rcnn.pytorch
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

解压这3个tar包，创建data目录并且建立符号链接。
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar

# mkdir data && cd data
ln -s ../VOCdevkit VOCdevkit2007
```

## 训练

使用如下脚本进行训练，比较重要的参数是lr，如果太大可能会出现nan/inf，作者使用这个参数是可以收敛的。
```
python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 1 \
    --lr 0.0004 --lr_decay_step 8 --cuda
```

## 测试
接下来是用测试集合进行测试，作者训练后得到的mAP在73.5%左右。读者可以多调调超参数，源代码作者得出的mAP是在75.2%左右。
```
python test_net.py --dataset pascal_voc --net res101 \
--checksession 1 --checkepoch 20 --checkpoint 10021 \
--cuda

Saving cached annotations to /bigdata/lili/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl
AP for aeroplane = 0.7534
AP for bicycle = 0.8044
AP for bird = 0.7760
AP for boat = 0.6076
AP for bottle = 0.5756
AP for bus = 0.8021
AP for car = 0.8283
AP for cat = 0.8664
AP for chair = 0.5332
AP for cow = 0.8147
AP for diningtable = 0.6709
AP for dog = 0.8700
AP for horse = 0.8561
AP for motorbike = 0.7939
AP for person = 0.7834
AP for pottedplant = 0.4588
AP for sheep = 0.7238
AP for sofa = 0.7499
AP for train = 0.7524
AP for tvmonitor = 0.6907
Mean AP = 0.7356

python demo.py --net res101 \
--checksession 1 --checkepoch 20 --checkpoint 10021 \
--cuda --load_dir models --image_dir testimgs
```

## 预测
我们创建一个测试目录testimgs，在里面放几张图片，看看实际检测的效果。
```
python demo.py --net res101 --checksession 1 --checkepoch 20 --checkpoint 10021 \
    --cuda --load_dir models --image_dir testimgs
```

下图是上面命令检测的实际效果，汽车都被正确的检测出来了。


<a name='img3_det'>![](/img/fastrcnncodes/img3_det.png)</a>
*图：Faster R-CNN检测效果*
