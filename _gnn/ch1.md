---
layout:     post
title:      "第一章：开始学习图学习"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - gnn
---



 

<!--more-->

**目录**
* TOC
{:toc}
 
欢迎来到我们探索图神经网络（GNNs）世界的第一章。在本章中，我们将深入了解GNNs的基础知识，并理解它们为何是现代数据分析和机器学习中至关重要的工具。为此，我们将回答三个关键问题，这将为我们提供对GNNs的全面理解。

首先，我们将探讨图作为数据表示的重要性，以及为什么它们在计算机科学、生物学和金融等各个领域被广泛使用。接下来，我们将深入研究图学习的重要性，了解图学习的不同应用以及不同的图学习技术家族。最后，我们将重点关注GNN家族，突出其独特的特点、性能以及与其他方法的比较。

通过本章的学习，您将清楚地了解GNNs为何重要以及如何用于解决实际问题。您还将具备深入研究更高级主题所需的知识和技能。那么，让我们开始吧！

在本章中，我们将涵盖以下主要主题：

为什么要使用图？
图学习的意义在哪里？
图神经网络的重要性何在？

## 为什么要使用图？

我们首先需要回答的问题是：我们为什么对图感兴趣？图论，即对图的数学研究，已经成为理解复杂系统和关系的基本工具。图是节点（也称为顶点）和连接这些节点的边的集合的可视化表示，为表示实体及其关系提供了结构（见图1.1）。

<a>![](/img/gnn/ch1/1.png)</a>
*图1.1 - 具有六个节点和五条边的图示例*

通过将复杂系统表示为实体及其相互作用的网络，我们可以分析它们之间的关系，从而更深入地理解它们的基本结构和模式。图的多功能性使其成为各个领域的热门选择，包括以下领域：

* 计算机科学，图可以用于建模计算机程序的结构，使我们更容易理解系统中不同组件之间的互动方式。
* 物理学，图可以用于模拟物理系统及其相互作用，比如粒子之间的关系及其特性之间的关系。
* 生物学，图可以用于模拟生物系统，比如作为互连实体网络的代谢(metabolic)途径。
* 社会科学，图可以用于研究和理解复杂的社会网络，包括社区中个体之间的关系。
* 金融学，图可以用于分析股市趋势以及不同金融工具之间的关系。
* 工程学，图可以用于建模和分析复杂系统，比如交通网络和电力网络。

这些领域自然地展示出关系结构。例如，图是社交网络的自然表示形式：节点是用户，边代表友谊关系。但图的多功能性使其也适用于关系结构不太自然的领域，从而获得新的见解和理解。

例如，图可以表示图像，如图1.2所示。每个像素是一个节点，边表示相邻像素之间的关系。这使得可以将基于图的算法应用于图像处理和计算机视觉任务。

<a>![](/img/gnn/ch1/2.png)</a>
*图1.2 - 左侧：原始图像；右侧：该图像的图表示*

类似地，可以将句子转换为图，其中节点是单词，边表示相邻单词之间的关系。这种方法在自然语言处理和信息检索任务中非常有用，其中单词的上下文和意义是关键因素。

与文本和图像不同，图没有固定的结构。然而，这种灵活性也使得处理图变得更加具有挑战性。缺乏固定结构意味着它们可以具有任意数量的节点和边，没有特定的顺序。此外，图可以表示动态数据，其中实体之间的连接随时间变化。例如，用户与产品之间的关系随着它们相互作用而变化。在这种情况下，节点和边会根据现实世界的变化进行更新，如新用户、新产品和新关系。

在接下来的部分中，我们将深入探讨如何将图与机器学习结合使用，创建有价值的应用程序。
 

## 为什么要进行图学习？

图学习是将机器学习技术应用于图数据的过程。这个研究领域涵盖了一系列旨在理解和处理图结构数据的任务。图学习涉及许多任务，包括以下几个：

* **节点分类**是一个任务，涉及预测图中节点的类别（分类）。例如，它可以根据其特征对在线用户或项目进行分类。在这个任务中，模型是在一组带有标签的节点及其属性上进行训练的，并利用这些信息来预测未标记节点的类别。

* **链接预测**是一个任务，涉及预测图中节点对之间缺失的连接。这在知识图谱完成中非常有用，其目标是完成实体及其关系的图。例如，它可以用于根据他们的社交网络连接来预测人与人之间的关系（好友推荐）。

* **图分类**是一个任务，涉及将不同的图分类到预定义的类别中。一个例子是在分子生物学中，分子结构可以表示为图，目标是预测它们的属性以进行药物设计。在这个任务中，模型是在一组带有标签的图及其属性上进行训练的，并利用这些信息来将未见过的图分类。

* **图生成**是一个任务，涉及根据一组所需属性生成新的图。其中一个主要应用是为药物发现生成新颖的分子结构。这通过在一组现有的分子结构上训练模型来实现，然后使用它来生成新的、未见过的结构。生成的结构可以评估其作为药物候选的潜力并进行进一步研究。

图学习还有许多其他实际应用，可以产生重大影响。其中最著名的应用之一是推荐系统，图学习算法根据用户之前的交互和与其他项目的关系向用户推荐相关项目。另一个重要的应用是交通预测，图学习可以通过考虑不同路线和交通方式之间的复杂关系来提高旅行时间预测的准确性。

图学习的多功能性和潜力使其成为一个令人兴奋的研究和开发领域。近年来，随着大型数据集、强大的计算资源以及机器学习和人工智能的进步，图的研究取得了迅速的进展。因此，我们可以列出四个主要的图学习技术家族：

* 图信号处理，将传统的信号处理方法应用于图，如图傅里叶变换和谱分析。这些技术揭示了图的固有属性，比如其连通性和结构。
* 矩阵分解，旨在找到大型矩阵的低维表示。矩阵分解的目标是识别解释原始矩阵中观察到的关系的潜在因素或模式。这种方法可以提供数据的简洁且可解释的表示。
* 随机游走，是指一种数学概念，用于模拟图中实体的移动。通过在图上模拟随机游走，可以收集关于节点之间关系的信息。这就是为什么它们经常用于为机器学习模型生成训练数据的原因。
* 深度学习，是机器学习的一个子领域，专注于具有多个层的神经网络。深度学习方法可以有效地对图数据进行编码和表示为向量。然后可以将这些向量用于各种任务，表现出色。

需要注意的是，这些技术并不是相互独立的，它们在应用中经常重叠。实际上，它们经常组合在一起形成混合模型，以利用各自的优势。例如，可以结合矩阵分解和深度学习技术来学习图结构化数据的低维表示。

当我们深入探讨图学习的世界时，了解任何机器学习技术的基本构建块非常重要：数据集。传统的表格数据集，如电子表格，将数据表示为行和列，其中每行表示一个单独的数据点。然而，在许多现实世界的情况下，数据点之间的关系和数据点本身一样重要。这就是图数据集的用武之地。图数据集将数据点表示为图中的节点，将这些数据点之间的关系表示为边。让我们以图1.3中显示的表格数据集为例。



<a>![](/img/gnn/ch1/3.png)</a>
*图1.3 - 家谱作为表格数据集与图数据集的对比*



这个数据集表示了一个家庭中五个成员的信息。每个成员都有三个特征（或属性）：姓名、年龄和性别。然而，这个数据集的表格版本并没有显示这些人之间的连接。相反，图版本用边来表示它们，这使我们能够理解这个家庭中的关系。在许多情境中，节点之间的连接对于理解数据至关重要，这就是为什么以图形式表示数据变得越来越流行的原因。

现在我们对图机器学习及其涉及的不同类型任务有了基本的理解，我们可以继续探讨解决这些任务中最重要的方法之一：图神经网络。

## 为什么要使用图神经网络？

在本书中，我们将重点关注图学习技术中的深度学习家族，通常称为图神经网络（GNNs）。GNNs是一种新型的深度学习架构，专门设计用于处理图结构化数据。与主要用于文本和图像的传统深度学习算法不同，GNNs明确设计用于处理和分析图数据集（见图1.4）。


<a>![](/img/gnn/ch1/4.png)</a>
*图1.4 - GNN管道的高级架构，以图作为输入，并输出对应于给定任务的结果*

GNNs已经成为图学习的强大工具，并在各种任务和行业中展现了出色的结果。其中一个最引人注目的例子是GNN模型如何识别了一种新的抗生素[2]。该模型在2500种分子上进行了训练，并在一个包含6000种化合物的库上进行了测试。它预测一种名为halicin的分子应该能够杀死许多对抗生素具有抗性的细菌，同时对人类细胞的毒性较低。基于这个预测，研究人员使用halicin来治疗感染抗生素耐药细菌的小鼠。他们证明了其有效性，并相信该模型可以用于设计新药。

GNNs如何工作？让我们以社交网络中的节点分类任务为例，就像之前的家谱（图1.3）。在节点分类任务中，GNNs利用来自不同来源的信息来创建图中每个节点的向量表示。这个表示不仅包括原始节点特征（如姓名、年龄和性别），还包括边特征（如节点之间关系的强度）和全局特征（如整个网络的统计信息）。

这就是为什么GNNs比传统的机器学习技术在图上更有效。GNNs不仅限于原始属性，还用邻近节点、边和全局特征丰富了原始节点特征，使得表示更加全面和有意义。新的节点表示然后用于执行特定任务，如节点分类、回归或链接预测。

具体来说，GNNs定义了图卷积操作，聚合了来自邻近节点和边的信息以更新节点表示。这个操作是迭代执行的，随着迭代次数的增加，模型可以学习到更复杂的节点之间关系。例如，图1.5展示了一个GNN如何使用邻近节点计算节点5的表示。

<a>![](/img/gnn/ch1/5.png)</a>
*图1.5 左侧：输入图；右侧：计算图表示了GNN如何基于其邻近节点计算节点5的表示*

值得注意的是，图1.5提供了计算图的简化图示。实际上，有各种类型的GNNs和GNN层，每种都有独特的结构和从邻近节点聚合信息的方式。这些不同的GNNs变体也各有优势和局限性，并且适用于特定类型的图数据和任务。在选择适合特定问题的GNN架构时，了解图数据的特性和期望的结果非常重要。

总的来说，GNNs和其他深度学习技术一样，在应用于特定问题时最为有效。这些问题具有高复杂性，意味着学习良好的表示对于解决手头的任务至关重要。例如，一个高度复杂的任务可能是在数以亿计的选项中向数百万客户推荐正确的产品。另一方面，一些问题，比如找到我们家谱中最年轻的成员，可以在不使用任何机器学习技术的情况下解决。

此外，GNNs需要大量的数据才能发挥有效作用。在数据集较小的情况下，传统的机器学习技术可能更适合，因为它们不太依赖大量数据。然而，这些技术不如GNNs具有良好的扩展性。GNNs可以处理更大的数据集，这得益于并行和分布式训练。它们还可以更有效地利用额外信息，从而产生更好的结果。 



## 总结

在本章中，我们回答了三个主要问题：为什么使用图、为什么学习图、以及为什么使用图神经网络？首先，我们探讨了图在表示各种数据类型方面的多功能性，例如社交网络和交通网络，还有文本和图像。我们讨论了图学习的不同应用，包括节点分类和图分类，并强调了四个主要的图学习技术家族。最后，我们强调了GNNs的重要性及其在处理大型复杂数据集方面的优越性，特别是相对于其他技术。通过回答这三个主要问题，我们旨在全面概述GNNs的重要性以及它们在机器学习中成为关键工具的原因。

在第二章《图神经网络的图论基础》中，我们将深入探讨图论的基础知识，这是理解GNNs的基础。本章将涵盖图论的基本概念，包括邻接矩阵和度等概念。此外，我们还将深入研究不同类型的图及其应用，例如有向图和无向图，以及加权图和非加权图。




