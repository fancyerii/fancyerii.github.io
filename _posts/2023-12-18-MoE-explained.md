---
layout:     post
title:      "Mixture of Experts Explained" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - qlora
    - quantization 
---

本文是Huggingface博客[Mixture of Experts Explained](https://huggingface.co/blog/moe)的翻译。

<!--more-->

**目录**
* TOC
{:toc}


## TL;DR
 
1. 相对密集模型，预训练速度更快
2. 与具有相同参数数量的模型相比，推理速度更快
3. 需要大量显存，因为所有专家都加载到内存中
4. 在微调方面面临许多挑战，但最近在MoE指令微调方面取得了一些进展

## 什么是Mixture of Experts (MoE)?      
模型的规模是提高模型质量的最重要因素之一。在固定的计算预算下，对一个规模较大的模型进行较少的训练步骤(steps)比对一个规模较小的模型进行更多步骤更为优越。
混合专家模型使得可以使用更少的计算资源进行预训练，这意味着在相同的计算预算下，可以大幅提升模型或数据集的规模，而与密集模型相比，MoE 模型在预训练期间应能更快地达到相同的质量水平。
那么，MoE 到底是什么？在 Transformer 模型的背景下，MoE 主要由两个要素组成：

1. 使用稀疏 MoE 层而不是密集的前馈网络（FFN）层。MoE 层具有一定数量的“专家”（例如 8 个），其中每个专家都是一个神经网络。在实践中，这些专家通常是 FFN，但它们也可以是更复杂的网络，甚至是 MoE 本身，形成层次化的 MoE！
2. 一个门网络(gate network)或路由器(router)，用于确定将哪些token发送到哪个专家。例如，在下面的图像中，token“More”被发送到第二个专家，token“Parameters”被发送到第一个网络。正如我们稍后将探讨的，我们可以将一个token发送给多个专家。如何将token路由到专家是在使用 MoE 时要做的重要决策之一 路由器由学到的参数组成，并且与网络的其余部分一起进行预训练。

<a>![](/img/moe/1.png)</a>

所以，简而言之，在 MoE 中，我们用一个由门网络和若干个专家组成的 MoE 层替换了变压器模型的每个 FFN 层。
尽管 MoE 提供了高效的预训练和与密集模型相比更快的推理等好处，但它们也面临一些挑战：
1. 训练：MoE 能够实现更高效的计算预训练，但在精调阶段通常难以进行泛化，容易导致过拟合。
2. 推理：尽管 MoE  可能有许多参数，但在推断过程中只使用其中的一部分。这导致了与具有相同数量参数的密集模型相比更快的推断速度。然而，所有参数都需要加载到 RAM  中，因此内存要求很高。例如，对于类似 Mixtral 8x7B 的 MoE 模型，我们需要足够的 VRAM 来容纳一个具有 47B  参数的密集模型。为什么是 47B 参数而不是 8 x 7B = 56B？因为在 MoE 模型中，只有 FFN  层被视为独立的专家，模型的其余参数是共享的。同时，假设每个令牌只使用了两个专家，推断速度（FLOPs）就相当于使用了一个 12B 模型（而不是  14B 模型），因为它计算了 2x7B 的矩阵乘法，但有一些层是共享的（稍后会详细介绍）。
现在我们对 MoE 有了一个大致的了解，让我们看一下它的研究进展。

## A Brief History of MoEs  

MoE 的根源可以追溯到1991年的论文Adaptive Mixture of Local Experts。这个想法类似于集成方法，其核心思想是为系统设计一个监督过程，由不同的网络组成，每个网络处理训练案例的不同子集。每个独立的网络或专家专门处理输入空间的不同区域。专家是如何选择的呢？一个门控网络确定每个专家的权重。在训练过程中，专家和门控都会被训练。
在2010年至2015年之间，两个不同的研究领域为后来的MoE进展做出了贡献：
1. 专家作为组件：在传统的MoE设置中，整个系统由一个门控网络和多个专家组成。MoE作为整个模型已经在SVMs、高斯过程和其他方法中进行了探讨。Eigen, Ranzato, 和Ilya的工作探索了MoE作为深度网络组件的可能性。这允许在多层网络中将MoE作为层，使模型能够同时变得庞大和高效。
2. 条件计算：传统网络通过每一层处理所有输入数据。在这个时期，Yoshua Bengio研究了基于输入token动态激活或停用组件的方法。

这些工作导致了在NLP背景下探索专家混合的研究。具体来说，Shazeer等人（2017年，其中“等人”包括Geoffrey Hinton和Jeff Dean，谷歌的“Chuck Norris”）通过引入稀疏性将这个想法扩展到了一个137B的LSTM（当时
最主流的NLP架构，由Schmidhuber创建），这使得在大规模情况下仍然可以保持非常快速的推理。这项工作主要关注翻译，但面临着许多挑战，如高通信成本和训练不稳定性。

<a>![](/img/moe/2.png)</a>

## What is Sparsity?

稀疏性使用条件计算的概念。在密集模型中，所有参数都用于所有输入，而稀疏性允许我们仅对整个系统的某些部分进行运行。
让我们深入探讨Shazeer关于翻译MoEs的研究。条件计算的概念（网络的某些部分是根据每个示例激活的）允许在不增加计算的情况下扩大模型的规模，因此在每个MoE层中使用了成千上万的专家。
这种设置引入了一些挑战。例如，虽然通常较大的批次对性能更有利，但在MOEs中，随着数据通过活动的专家流动，批次大小实际上会减小。例如，如果我们的批次输入包含10个token，其中五个token可能进入一个专家，另外五个token可能分布在五个不同的专家中，导致不均匀的token大小和低效利用。下面的“使MoEs go brrr”部分将讨论其他挑战和解决方案。
我们如何解决这个问题呢？一个学到的门控网络（G）决定将输入的一部分发送给哪些专家（E）：

$$y=\sum_{i=1}^nG(x)_iE_i(x)$$

在这个设置中，所有专家都针对所有输入运行 - 这是一种加权乘法。但是，如果 G 为 0  会发生什么呢？如果是这样，就没有必要计算相应的专家操作，因此我们节省了计算资源。一个典型的门控函数是什么呢？在最传统的设置中，我们只是使用一个带有  softmax 函数的简单网络。该网络将学习将输入发送给哪个专家。

$$G_\sigma(x)=\text{Softmax}(x \cdot W_g)$$

Shazeer的工作还探讨了其他门控机制，例如有噪音的 Top-K 门控。该门控方法引入了一些（可调节的）噪音，然后保留前 k 个值。即：
1. 增加一些noise：
  $$H(x)_i=(x \cdot W_g)_i + \text{StandardNormal()} \cdot \text{SoftPlus}((x \cdot )_iW_{\text{noise}})$$
2. 我们选择top-k：
  $$\text{KeepTopK}(v,k)_i=\left\{
 \begin{aligned}
v_i & , & v_i是v的前k大的值, \\
-\infty& , & 否则
 \end{aligned}
 \right.$$
3. 再进行Softmax计算：
  $$G(x)=\text{Softmax}(\text{KeepTopK}(v,k)_i)$$

这种稀疏性引入了一些有趣的特性。通过使用足够低的 k（例如一个或两个），我们可以比激活许多专家时更快地进行训练和推理。为什么不只选择顶级专家呢？最初的推测是需要将流量路由到不同的专家，以使门学会如何路由到不同的专家，因此至少必须选择两个专家。Switch Transformers 部分重新审视了这个决定。
为什么要添加噪音？这是为了负载均衡！

## Load balancing tokens for MoEs

正如前面讨论的，如果所有的token都只发送给几个热门专家，那将使训练效率低下。在正常的MoE训练中，门控网络趋向于主要激活少数几个专家。这是自我强化的，因为受欢迎的专家训练更快，因此被更频繁地选择。为了缓解这个问题，添加了一个辅助损失，以鼓励给予所有专家相等的重要性。该损失确保所有专家接收大致相等数量的训练示例。接下来的部分还将探讨专家容量的概念，该概念引入了专家可以处理多少token的阈值。在  transformers 中，通过 aux_loss 参数设置辅助损失。

## MoEs and Transformers 

Transformer 模型是一个非常明显的案例，即增加参数数量会提高性能，因此谷歌尝试使用 GShard 进行了探索，该工具探讨了将 Transformer 扩展到超过 6000 亿个参数。
GShard 将每个 FFN 层的位置都替换为一个MoE层，使用top-2 门控机制，分别在编码器和解码器中使用。下图显示了编码器部分的情况。对于大规模计算来说，这种设置非常有利：当我们扩展到多个设备时，MoE 层在设备之间共享，而所有其他层都是复制的。这将在“Making MoEs go brrr”部分进一步讨论。

<a>![](/img/moe/3.png)</a>

为了在规模上保持平衡的负载和效率，GShard 作者在辅助损失的基础上引入了一些变化，类似于前面讨论的辅助损失：
- **随机路由**：在 top-2 设置中，我们总是选择顶级专家，但这里会根据权重随机选择第2的专家【类似生成的sample】。
- **专家容量**：我们可以设置一个阈值，规定一个专家可以处理多少个token。如果两个专家都已满载，token将被视为溢出，通过残差连接发送到下一层（或在其他项目中完全丢弃）。这个概念将成为 MoE 中最重要的概念之一。为什么需要专家容量？由于所有张量形状在编译时静态确定，但我们无法提前知道每个专家将接收多少token，因此需要确定容量因子。

GShard 论文通过表达适用于MoE的并行计算模式做出了贡献，但讨论这一点超出了本博客文章的范围。
注意：在运行推断时，只有一些专家会被触发。与此同时，存在共享的计算，如对所有token应用的自注意力。这就是为什么当我们谈论一个包含 8 个专家的 47B 模型时，我们可以运行具有 12B 密集模型计算的原因。如果使用 top-2，则会使用 14B 参数。但考虑到注意力操作等是共享的（等等），实际使用的参数数量是 12B。

## Switch-Transformers
尽管 MoE 显示出很多潜力，但它们在训练和微调时存在不稳定性。《Switch  Transformers》是一项非常令人振奋的工作，深入探讨了这些问题。作者甚至在 Hugging Face 上发布了一个具有 2048  个专家、拥有 1.6 万亿参数的 MoE 模型，您可以使用 transformers 运行。Switch Transformers  在预训练速度上实现了 T5-XXL 的 4 倍提速。

<a>![](/img/moe/4.png)</a>

正如在 GShard 中一样，作者将 FFN 层替换为MoE层。Switch Transformers论文提出了一个 Switch Transformer 层，接收两个输入（两个不同的token）并具有四个专家。
与最初使用至少两个专家的想法相反，Switch Transformers 使用了简化的单专家策略。这种方法的效果包括：
- 减少了路由器计算
- 每个专家的批处理大小至少减半
- 降低了通信成本
- 保持了模型质量
Switch Transformers 还探讨了专家容量的概念。

$$\text{Expert Capacity}=(\frac{\text{tokens per batch}}{\text{number of experts}}) \times \text{capacity factor}$$
上面建议的容量均匀地将批次中的token数量分配到专家数量。如果使用大于1的容量因子，我们为token不平衡提供了缓冲区。增加容量将导致更昂贵的设备间通信，因此这是需要权衡考虑的。特别是，Switch Transformers 在低容量因子（1-1.25）下表现良好。
Switch Transformer 的作者还重新审视并简化了在前面章节提到的负载平衡损失。对于每个 Switch 层，辅助损失在训练期间添加到总模型损失中。此损失鼓励均匀路由，并可以使用超参数进行加权。
作者还尝试了选择性精度，例如使用 bfloat16 训练专家，同时对其他计算使用完整精度。较低的精度减少了处理器之间的通信成本、计算成本以及存储张量的内存。最初的实验中，专家和门控网络都以 bfloat16 进行训练，导致训练不稳定。这主要是由于路由器计算中涉及指数函数，因此具有更高精度是重要的。为了减轻不稳定性，路由器的计算也使用完整精度。

<a>![](/img/moe/5.png)</a>

此笔记本展示了将 Switch Transformers 进行微调以进行摘要生成，但我们建议首先查看微调部分。
Switch Transformers 使用了编码器-解码器设置，在该设置中，它们对 T5 进行了 MoE 版本。GLaM 论文探讨了通过训练一个与 GPT-3 质量相匹配的模型，使用 1/3 的能量（是的，由于训练 MoE 所需的计算量较低，它们可以将碳足迹降低多达一个数量级）。作者专注于解码器模型、少样本和一样本评估，而不是微调。他们使用了 Top-2 路由和更大的容量因子。此外，他们探索了容量因子作为一个度量标准，可以根据训练和评估期间想要使用的计算量进行更改。

## Stabilizing training with router-Z loss

先前讨论的平衡损失可能导致不稳定性问题。我们可以使用许多方法来稳定稀疏模型，但会牺牲质量。例如，引入dropout可以提高稳定性，但会降低模型质量。另一方面，增加更多的乘法组件可以提高质量，但降低稳定性。
在 ST-MoE 中引入的 Router z-loss 通过对进入门控网络的大 logits 进行惩罚，显著提高了训练稳定性而不降低质量。由于此损失鼓励参数的绝对幅度变小，可减少舍入误差，这对于门控等指数函数可能具有相当大的影响。我们建议查看论文以获取详细信息。

## What does an expert learn?

ST-MoE  的作者观察到编码器专家专注于一组标记或浅层概念。例如，我们可能会有一个标点符号专家、一个专有名词专家等。另一方面，解码器专家的专业化程度较低。作者还在多语言环境中进行了训练。尽管可以想象每个专家专门研究一种语言，但事实相反：由于标记路由和负载平衡，没有单一的专家专门研究任何一种语言。

<a>![](/img/moe/6.png)</a>


## How does scaling the number of experts impact pretraining? 

更多的专家会提高样本效率和更快的加速，但这些增益是递减的（特别是在 256 或 512 之后），推断时将需要更多的 VRAM。在大规模上研究的 Switch Transformers 的属性在小规模上也是一致的，即使每层只有 2、4 或 8 个专家。


## Fine-tuning MoEs

过拟合的动力学(dynamics)在密集模型和稀疏模型之间存在很大的差异。稀疏模型更容易过拟合，因此我们可以在专家自身内部探索更高的正则化（例如，dropout），例如我们可以为密集层设定一个丢弃率，为稀疏层设定另一个更高的丢弃率。
一个决策问题是是否在微调中使用辅助损失。ST-MoE的作者尝试关闭辅助损失，即使在dropout了高达11%的标记时，质量也没有受到明显影响。token dropout可能是一种正则化形式，有助于防止过拟合。
Switch Transformers观察到，在固定的预训练困惑度(PPL)下，稀疏模型在下游任务中表现比密集对应物差，特别是在重推理的任务上，如SuperGLUE。另一方面，对于知识密集型任务，如TriviaQA，稀疏模型表现出比例上好的性能。作者还观察到，在微调中较少的专家数量有助于性能。另一个说明泛化问题存在的观察是，在较小的任务中，模型表现较差，但在较大的任务中表现良好。

<a>![](/img/moe/7.png)</a>

在小任务中（左侧），我们可以看到明显的过拟合，因为稀疏模型在验证集上表现较差。在较大的任务中（右侧），MoE表现良好。上图来自ST-MoE论文。

可以尝试冻结所有非专家权重进行实验。这导致了巨大的性能下降，这是预期的，因为MoE层对应于网络的大部分。我们可以尝试相反的方法：只冻结MoE层中的参数，结果表明这几乎与更新所有参数一样有效。这有助于加速微调并减少内存使用。

<a>![](/img/moe/8.png)</a>

在微调稀疏的MoE时，还有最后一部分需要考虑，即它们具有不同的微调超参数设置，例如，稀疏模型往往更受益于较小的批次大小和更高的学习率。

<a>![](/img/moe/9.png)</a>
**稀疏模型在微调时，通过降低学习率和增大批次大小，其质量得到改善。上图来自ST-MoE论文。**

此时，您可能对人们在微调MoE时所遇到的困难感到有些沮丧。令人振奋的是，最近一篇论文《[MoEs Meets Instruction Tuning](https://arxiv.org/pdf/2305.14705.pdf)》（2023年7月）进行了以下实验：

- 单任务微调
- 多任务指令调整
- 多任务指令调整后的单任务微调

当作者微调MoE和与之对应的T5模型时，T5模型表现更好。当作者微调Flan T5（指令微调的T5模型） MoE时，MoE的性能显著提高。而且，Flan-MoE相对于MoE的改进幅度大于Flan T5相对于T5的改进，这表明MoEs可能比密集模型更受益于指令调整。MoEs更受益于更多的任务。与先前的讨论建议关闭辅助损失函数不同，这里的损失实际上有助于防止过拟合。

<a>![](/img/moe/10.png)</a>
**与密集模型相比，稀疏模型更受益于指令调整。上图来自《MoEs Meets Instruction Tuning》论文。** 


## When to use sparse MoEs vs dense models? 
在具有许多机器的高吞吐量场景中，专家对于性能很有帮助。在给定用于预训练的固定计算预算的情况下，稀疏模型将更为优化。对于具有较小VRAM的低吞吐量场景，密集模型将更为适用。
注意：不能直接比较稀疏模型和密集模型之间的参数数量，因为它们代表着显著不同的内容。

## Making MoEs go brrr 
最初的MoE工作将MoE层呈现为一个分支设置，导致计算速度缓慢，因为GPU并非为此而设计，并且由于设备需要将信息发送给其他设备，网络带宽成为瓶颈。本节将讨论一些现有工作，以使使用这些模型进行预训练和推理更加实际。这将激发MoEs的活力。

### Parallelism

让我们简要回顾一下并行：

- 数据并行：相同的权重被复制到所有核心，数据被分割到各个核心(计算设备)。
- 模型并行：模型被分割到各个核心，数据被复制到各个核心。
- 模型和数据并行：我们可以将模型和数据分割到各个核心。请注意，不同核心处理不同批次的数据。
- 专家并行：专家被放置在不同的工作节点上。如果与数据并行结合使用，每个核心有一个不同的专家，并且数据被分割到所有核心。

使用专家并行，专家被放置在不同的工作节点上，每个工作节点处理不同批次的训练样本。对于非MoE层，专家并行的行为与数据并行相同。对于MoE层，序列中的标记被发送到包含所需专家的工作节点。

<a>![](/img/moe/11.png)</a>
**来自《Switch Transformers》论文的插图展示了如何使用不同的并行技术在核心之间分割数据和模型。**

### Capacity Factor and communication costs 

增加容量因子（CF）会提高质量，但会增加通信成本和激活内存的使用。如果all-to-all通信速度较慢，使用较小的容量因子更好。一个良好的起点是使用top-2路由，容量因子为1.25，并且每个核心有一个专家。在评估过程中，可以调整容量因子以减少计算量。

### Serving techniques         

MoEs的一个巨大缺点是参数数量巨大。对于本地使用情况，可能希望使用较小的模型。让我们快速讨论一下几种在服务中有助于的技术：

- Switch Transformers的作者进行了早期的蒸馏实验。通过将MoE蒸馏回其密集对应物，他们能够保留30-40%的稀疏性收益。因此，蒸馏提供了更快的预训练和在生产中使用较小模型的好处。
- 最近的方法修改了路由方式，将完整的句子或任务路由到专家，从而允许在服务中提取子网络。
- 专家聚合（MoE）：该技术合并专家的权重，因此在推理时减少了参数数量。

### More on efficient training         

《[FasterMoE](https://dl.acm.org/doi/abs/10.1145/3503221.3508418)》（2022年3月）分析了MoEs在高效分布式系统中的性能，并分析了不同并行策略的理论极限，以及技术方案，如对专家流行度进行倾斜、细粒度的通信调度以减少延迟，以及根据最低延迟来选择专家的拓扑感知(topology-aware)门，从而实现了17倍的加速。
《[MegaBlocks](https://arxiv.org/abs/2211.15841)》（2022年11月）通过提供能够处理MoEs中存在的动态特性的新GPU内核，探索了高效的稀疏预训练。他们的提案从不丢弃token，并且能够高效映射到现代硬件，从而实现了显著的加速。这其中的技巧是什么呢？传统的MoEs使用批量矩阵乘法，假设所有专家具有相同的形状和相同数量的token。相比之下，《Megablocks》将MoE层表示为块稀疏操作，可以适应不平衡的分配。

<a>![](/img/moe/12.png)</a>
**MegaBlocks 中提到的块稀疏矩阵乘法，适用于不同大小的专家和标记数量**

## Open Source MoEs 

现在有几个开源项目用于训练MoEs：

- Megablocks: https://github.com/stanford-futuredata/megablocks
- Fairseq: https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm
- OpenMoE: https://github.com/XueFuzhao/OpenMoE

在已发布的开源MoE领域，您可以查看：

- [Switch Transformers](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f)（Google）：基于T5的MoEs集合，从8到2048个专家。最大的模型有1.6万亿个参数。
- [NLLB MoE](https://huggingface.co/facebook/nllb-moe-54b)（Meta）：NLLB翻译模型的MoE变体。
- [OpenMoE](https://huggingface.co/fuzhao)：一个由社区共同努力的项目，发布了基于Llama的MoEs。
- [Mixtral 8x7B](https://huggingface.co/mistralai)（Mistral）：一个高质量的MoE，优于Llama 2 70B，并且推理速度更快。还发布了一个指导调整的模型。在公告博客文章中可以了解更多信息。

### Exciting directions of work         

一个未来的方向是将稀疏的MoE蒸馏回一个参数更少但参数数量相似的密集模型。

另一个研究领域是对MoEs进行量化。《[QMoE](https://arxiv.org/abs/2310.16795)》（2023年10月）朝着这个方向迈出了重要的一步，通过将MoEs量化至每个参数少于1位，从而将使用3.2TB内存的1.6万亿(1.6 Trillion)参数的Switch Transformer压缩到仅160GB。

因此，简而言之，有一些有趣的探索领域：

- 将Mixtral蒸馏为一个密集模型
- 探索专家的模型合并技术及其在推理时间上的影响
- 进行Mixtral的极端量化技术实验


#### Some resources   
      
- [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
- [Learning Factored Representations in a Deep Mixture of Experts (2013)](https://arxiv.org/abs/1312.4314)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Jun 2020)](https://arxiv.org/abs/2006.16668)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (Dec 2021)](https://arxiv.org/abs/2112.06905)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Jan 2022)](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Feb 2022)](https://arxiv.org/abs/2202.08906)
- [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models(April 2022)](https://dl.acm.org/doi/10.1145/3503221.3508418)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Nov 2022)](https://arxiv.org/abs/2211.15841)
- [Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models (May 2023)](https://arxiv.org/abs/2305.14705)
- [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)


