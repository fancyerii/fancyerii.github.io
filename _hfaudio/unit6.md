---
layout:     post
title:      "6. From Text to Speech"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - deep learning
    - speech
    - asr
    - tts
    - audio
    - huggingface
---



 <!--more-->


在上一单元中，你学会了如何使用变换器将口语转换为文本。现在让我们来换个角度，看看如何将给定的文本转换成听起来像人类语音的音频输出。

我们将在本单元研究的任务称为“文本到语音”（TTS）。能够将文本转换为可听的人类语音的模型具有广泛的潜在应用：

* 辅助应用程序：想象一下可以利用这些模型的工具，让视力受损的人通过声音获取数字内容。
* 有声读物叙述：将书面书籍转换为音频形式使文学作品更容易被那些更喜欢听书或阅读困难的人访问。
* 虚拟助手：TTS 模型是虚拟助手（如 Siri、Google Assistant 或 Amazon Alexa）的基本组成部分。一旦它们使用分类模型捕获唤醒词，并使用 ASR 模型处理你的请求，它们就可以使用 TTS 模型来回答你的查询。
* 娱乐、游戏和语言学习：为 NPC 角色配音，叙述游戏事件，或者帮助语言学习者正确发音和语调的单词和短语示例。

这些只是一些例子，我相信你还可以想象出许多其他应用！然而，拥有如此强大的力量也意味着有责任，重要的是要强调，TTS 模型有可能被用于恶意目的。例如，通过足够的语音样本，恶意行为者可能会制作出令人信服的虚假音频录音，导致未经授权使用某人的声音进行欺诈或操纵。如果你计划收集数据来对自己的系统进行微调，请认真考虑隐私和知情同意。语音数据应该在明确得到个人同意的情况下获取，确保他们理解其声音被用于 TTS 系统的目的、范围和潜在风险。请负责任地使用文本到语音技术。

**你将学到什么和你将构建什么**

在本单元中，我们将讨论以下内容：

* 适用于文本到语音训练的数据集
* 用于文本到语音的预训练模型
* 在新语言上微调 SpeechT5
* 评估 TTS 模型
 
## 文本到语音数据集

文本到语音任务（也称为语音合成）面临着一系列挑战。

首先，就像在前面讨论的自动语音识别中一样，文本与语音之间的对齐可能会棘手。
然而，与 ASR 不同，TTS 是一个一对多的映射问题，即相同的文本可以以许多不同的方式合成。想想你每天听到的语音中的声音和说话风格的多样性 - 每个人说同一句话的方式都不同，但它们都是有效和正确的！甚至不同的输出（声谱图或音频波形）可以对应相同的地面真实性。模型必须学会为每个音素、单词或句子生成正确的持续时间和时间，这可能是具有挑战性的，特别是对于长而复杂的句子。

接下来，有长距离依赖问题：语言具有时间特性，理解句子的意义通常需要考虑周围单词的上下文。确保 TTS 模型在长序列上捕获和保留上下文信息对于生成连贯和自然的语音至关重要。

最后，训练 TTS 模型通常需要文本和相应语音录音的配对。除此之外，为了确保模型可以生成对各种讲话者和说话风格都自然的语音，数据应该包含来自多个说话者的多样化和代表性的语音样本。收集这样的数据是昂贵的、耗时的，对于一些语言来说是不可行的。你可能会想，为什么不只是拿一个设计用于 ASR（自动语音识别）的数据集来训练 TTS 模型？不幸的是，自动语音识别（ASR）数据集并不是最佳选择。使其对 ASR 有益的特征，例如过多的背景噪音，在 TTS 中通常是不可取的。如果你的语音助手回复你时街道嘈杂的录音里有汽车喇叭声和施工声音，那么从一个嘈杂的街道录音中提取出说话声音是很棒的，但如果你的语音助手回复你时街道嘈杂的录音里有汽车喇叭声和施工声音，那就不太好了。尽管如此，有时一些 ASR 数据集可能对微调有用，因为找到高质量、多语言和多说话者 TTS 数据集可能会非常具有挑战性。

让我们探索一些适用于 TTS 的数据集，你可以在 🤗 Hub 上找到它们。

### LJSpeech

[LJSpeech](https://huggingface.co/datasets/lj_speech) 是一个数据集，包含 13,100 个英语音频片段及其相应的转录文本。该数据集包含单个说话者朗读英语 7 本非虚构书籍的句子的录音。由于其高音频质量和多样化的语言内容，LJSpeech 通常被用作评估 TTS 模型的基准。

### 多语言 LibriSpeech

[多语言 LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) 是 LibriSpeech 数据集的多语言扩展，后者是一个大规模的英语有声读物集合。多语言 LibriSpeech 在此基础上增加了其他语言，如德语、荷兰语、西班牙语、法语、意大利语、葡萄牙语和波兰语。它提供了每种语言的音频录音以及对齐的转录文本。该数据集为开发多语言 TTS 系统和探索跨语言语音合成技术提供了宝贵的资源。

### VCTK（Voice Cloning Toolkit）

[VCTK](https://huggingface.co/datasets/vctk) 是专为文本到语音研究和开发设计的数据集。它包含 110 个具有不同口音的英语说话者的音频录音。每个说话者朗读约 400 句子，这些句子是从报纸、彩虹段落和用于语音口音存档的引诱段落中选取的。VCTK 为训练具有多样化声音和口音的 TTS 模型提供了宝贵的资源，使语音合成更加自然和多样化。

### Libri-TTS/ LibriTTS-R

[Libri-TTS/ LibriTTS-R](https://huggingface.co/datasets/cdminix/libritts-r-aligned) 是一个英文多说话者语料库，包含约 585 小时的 24kHz 采样率的英文语音录音，由 Heiga Zen 和 Google 语音以及 Google Brain 团队成员的帮助准备。LibriTTS 语料库专为 TTS 研究而设计。它源自 LibriSpeech 语料库的原始材料（来自 LibriVox 的 mp3 音频文件和来自 Project Gutenberg 的文本文件）。与 LibriSpeech 语料库的主要区别如下：

* 音频文件采样率为 24kHz。
* 语音在句子分割处分割。
* 包括原始文本和归一化文本。
* 可以提取上下文信息（例如，相邻句子）。
* 排除了具有显着背景噪音的话语。

组建一个适用于 TTS 的好数据集并不容易，因为这样的数据集必须具备几个关键特征：

* 高质量和多样化的录音，覆盖了各种语音模式、口音、语言和情感。录音应清晰、没有背景噪音，并展现出自然的语音特征。
* 转录文本：每个音频录音都应附带其相应的文本转录。
* 丰富的语言内容：数据集应包含多样化的语言内容，包括不同类型的句子、短语和单词。它应涵盖各种主题、流派和领域，以确保模型能够处理不同的语言环境。

好消息是，你不太可能需要从头开始训练一个 TTS 模型。在下一节中，我们将探讨 🤗 Hub 上可用的预训练模型。


### 文本到语音的预训练模型

与自动语音识别（ASR）和音频分类任务相比，可用的预训练模型检查点数量明显较少。在 🤗 Hub 上，你会找到接近 300 个适合的检查点。在这些预训练模型中，我们将重点关注两种架构，它们在 🤗 Transformers 库中为你提供了方便使用的 SpeechT5 和大规模多语言语音（MMS）。在本节中，我们将探讨如何在 Transformers 库中使用这些预训练模型进行 TTS。

#### SpeechT5

SpeechT5 是由微软的 Junyi Ao 等人发布的模型，能够处理一系列的语音任务。虽然在本单元中我们专注于文本到语音的方面，但该模型可以定制为语音到文本的任务（自动语音识别或说话者识别），以及语音到语音的任务（例如语音增强或在不同声音之间转换）。这是由于模型的设计和预训练方式。

在 SpeechT5 的核心是一个常规的 Transformer 编码器-解码器模型。就像任何其他 Transformer 一样，编码器-解码器网络使用隐藏表示对序列进行序列转换。这个 Transformer 骨干结构是 SpeechT5 支持的所有任务的相同的。

这个 Transformer 还配备了六个模态特定的（语音/文本）预网络和后网络。输入的语音或文本（取决于任务）通过相应的预网络进行预处理，以获取 Transformer 可以使用的隐藏表示。然后，Transformer 的输出传递给后网络，后网络将使用它来生成目标模态的输出。

这是架构的样子（来自原始论文的图片）：

<a>![](/img/hfaudio/unit6/1.jpg)</a>

SpeechT5 首先使用大规模未标记的语音和文本数据进行预训练，以获得不同模态的统一表示。在预训练阶段，所有预网络和后网络都同时使用。

预训练之后，整个编码器-解码器骨干结构会针对每个个体任务进行微调。在这一步骤中，只有与特定任务相关的预网络和后网络被使用。例如，要将 SpeechT5 用于文本到语音，您需要使用文本编码器预网络进行文本输入，以及用于语音输出的语音解码器预网络和后网络。

这种方法允许获得针对不同语音任务进行微调的几个模型，它们都受益于未标记数据的初始预训练。

即使经过微调的模型最初使用来自共享预训练模型的相同权重集，最终的版本仍然有所不同。例如，您不能拿一个经过微调的 ASR 模型并更换预网络和后网络来获得一个可工作的 TTS 模型。SpeechT5 是灵活的，但并非那么灵活 ;)

让我们看看 SpeechT5 为 TTS 任务使用的预网络和后网络有哪些：

* 文本编码器预网络(Text encoder pre-net)：一个文本嵌入层，将文本标记映射到编码器期望的隐藏表示。这类似于 NLP 模型（如 BERT）中发生的情况。

* 语音解码器预网络(Speech decoder pre-net)：它以对数梅尔频谱图为输入，并使用一系列线性层将频谱图压缩成隐藏表示。

* 语音解码器后网络(Speech decoder post-net)：它预测要添加到输出频谱图中的残差，并用于精炼结果。

当结合在一起时，SpeechT5 用于文本到语音的架构就是这样的：

<a>![](/img/hfaudio/unit6/2.jpg)</a>

你可以看到，输出是一个对数梅尔频谱图，而不是最终的波形。如果你回忆一下，我们在第三单元简要讨论过这个话题。对于生成音频的模型来说，产生对数梅尔频谱图是常见的，但需要使用一个额外的神经网络，称为声码器，将其转换为波形。

让我们看看如何做到这一点。

首先，让我们从 🤗 Hub 加载经过微调的 TTS SpeechT5 模型，以及用于标记化和特征提取的处理器对象：

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
```


接下来，对输入文本进行标记化。

```python
inputs = processor(text="Don't count the days, make the days count.", return_tensors="pt")
```


SpeechT5 TTS 模型不仅限于为单个发言人创建语音。相反，它使用所谓的说话者嵌入，捕获特定说话者的语音特征。

说话者嵌入是一种以紧凑方式表示说话者身份的方法，作为一个固定大小的向量，不考虑话语的长度。这些嵌入捕获了有关说话者的语音、口音、语调和其他唯一特征的基本信息，区分了一个说话者和另一个说话者。这样的嵌入可以用于说话者验证、说话者日程安排、说话者识别等等。生成说话者嵌入的最常见技术包括：

* I-向量（身份向量）：I-向量基于高斯混合模型（GMM）。它将说话者表示为低维固定长度的向量，这些向量从特定说话者的 GMM 统计中获得，并以无监督方式获得。

* X-向量：X-向量使用深度神经网络（DNN）得到，通过整合时间上下文来捕获帧级别的说话者信息。

X-向量是一种最先进的方法，与 I-向量相比，在评估数据集上表现更优。深度神经网络用于获取 X-向量：它训练以区分说话者，并将可变长度的话语映射到固定维度的嵌入中。您也可以加载提前计算好的 X-向量说话者嵌入，这将封装特定说话者的语音特征。

让我们从 Hub 上的一个数据集中加载这样的说话者嵌入。这些嵌入是使用这个脚本从 CMU ARCTIC 数据集中获得的，但任何 X-向量嵌入都应该可以使用。

```python
from datasets import load_dataset

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

import torch

speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
```



说话者嵌入是一个形状为（1，512）的张量。这个特定的说话者嵌入描述了一个女性的声音。

此时我们已经有足够的输入来生成一个对数梅尔频谱图作为输出，您可以这样做：

```python
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```


这将输出一个形状为（140，80）的张量，其中包含一个对数梅尔频谱图。第一个维度是序列长度，它可能在运行中变化，因为语音解码器预网络始终对输入序列应用 dropout。这给生成的语音增加了一些随机变化。

然而，如果我们想生成语音波形，我们需要指定一个声码器来将频谱图转换为波形。理论上，您可以使用任何适用于 80 个频带的梅尔频谱图的声码器。方便地，🤗 Transformers 提供了一个基于 HiFi-GAN 的声码器。它的权重是由 SpeechT5 的原始作者友好提供的。

[HiFi-GAN](https://arxiv.org/pdf/2010.05646v2.pdf) 是一个用于高保真度语音合成的最先进的生成对抗网络（GAN）。它能够从频谱图输入生成高质量和逼真的音频波形。

在高层次上，HiFi-GAN 包括一个生成器和两个鉴别器。生成器是一个全卷积神经网络，它以梅尔频谱图为输入，并学习生成原始音频波形。鉴别器的作用是区分真实和生成的音频。两个鉴别器专注于音频的不同方面。

HiFi-GAN 在大量高质量音频录音数据集上进行训练。它使用所谓的对抗训练，其中生成器和鉴别器网络相互竞争。最初，生成器产生低质量的音频，鉴别器可以轻松将其与真实音频区分开来。随着训练的进行，生成器改善其输出，旨在欺骗鉴别器。鉴别器反过来在区分真实和生成的音频方面变得更加准确。这种对抗反馈循环有助于两个网络随着时间的推移不断改进。最终，HiFi-GAN 学会生成与训练数据特征相似的高保真度音频。

加载声码器与加载任何其他 🤗 Transformers 模型一样简单。

```python
from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```


现在您所需要做的就是在生成语音时将其作为参数传递，输出将自动转换为语音波形。

```python
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
```

让我们听听结果。SpeechT5 使用的采样率始终为 16 kHz。

```python
from IPython.display import Audio

Audio(speech, rate=16000)
```

很棒！

请随意使用 SpeechT5 文本到语音演示，探索其他语音，尝试不同的输入。请注意，此预训练检查点仅支持英语语言。



### Bark

Bark 是由 Suno AI 在 [suno-ai/bark](https://github.com/suno-ai/bark) 提出的基于 Transformer 的文本到语音模型。

与 SpeechT5 不同，Bark 直接生成原始语音波形，消除了推理过程中需要单独使用声码器的需求 - 它已经集成在其中。这种效率是通过利用 [Encodec](https://huggingface.co/docs/transformers/main/en/model_doc/encodec) 实现的，它既作为编解码器又作为压缩工具。

使用 Encodec，您可以将音频压缩成轻量级格式，以减少内存使用量，随后可以将其解压缩以恢复原始音频。这个压缩过程通过 8 个码本来实现，每个码本由整数向量组成。可以将这些码本看作是音频的整数形式的表示或嵌入。重要的是要注意，每个连续的码本都会改善从前一个码本中重建的音频的质量。由于码本是整数向量，它们可以被Transformer模型学习，这对这个任务非常有效。这就是 Bark 的特定训练目标。

更具体地说，Bark 由 4 个主要模型组成：

* BarkSemanticModel（也称为“文本”模型）：一个因果自回归 Transformer 模型，以标记化的文本作为输入，并预测捕获文本含义的语义文本标记。
* BarkCoarseModel（也称为“粗糙声学”模型）：一个因果自回归Transformer，以 BarkSemanticModel 模型的结果作为输入。它旨在预测 Encodec 所需的前两个音频码本。
* BarkFineModel（“精细声学”模型）：这次是一个非因果自动编码器Transformer，根据前几个码本嵌入的总和迭代地预测最后的码本。
* 预测了 EncodecModel 的所有码本通道后，Bark 使用它来解码输出音频数组。

应该注意，前三个模块中的每一个都可以支持条件说话者嵌入，以根据特定预定义的声音调整输出声音。

Bark 是一个高度可控的文本到语音模型，意味着您可以使用各种设置，正如我们将要看到的。

首先，加载模型及其处理器。

在这里，处理器的作用是双重的：

* 用于标记化输入文本，即将其分割成模型可以理解的小块。
* 它存储说话者嵌入，即可以根据声音预设条件生成的语音。

```python
from transformers import BarkModel, BarkProcessor

model = BarkModel.from_pretrained("suno/bark-small")
processor = BarkProcessor.from_pretrained("suno/bark-small")
```


Bark 非常灵活，可以生成由[声音嵌入库](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)条件生成的语音，该库可以通过处理器加载。

```python
# add a speaker embedding
inputs = processor("This is a test!", voice_preset="v2/en_speaker_3")

speech_output = model.generate(**inputs).cpu().numpy()
```



它还可以生成即时翻译的多语种语音，例如法语和中文。您可以在此处找到支持的语言列表。与下面将要讨论的 MMS 不同，不需要指定所使用的语言，而只需将输入文本调整为相应的语言即可。

```python
# try it in French, let's also add a French speaker embedding
inputs = processor("C'est un test!", voice_preset="v2/fr_speaker_1")

speech_output = model.generate(**inputs).cpu().numpy()
```

该模型还可以生成非言语交流，例如笑、叹气和哭泣。您只需修改输入文本，加上相应的提示，如 [clears throat]、[laughter] 或 ...。

```python
inputs = processor(
    "[clears throat] This is a test ... and I just took a long pause.",
    voice_preset="v2/fr_speaker_1",
)

speech_output = model.generate(**inputs).cpu().numpy()
```


Bark 甚至可以生成音乐。您可以通过在单词周围添加 ♪ 音符 ♪ 来帮助实现这一点。

```python
inputs = processor(
    "♪ In the mighty jungle, I'm trying to generate barks.",
)

speech_output = model.generate(**inputs).cpu().numpy()
```


除了所有这些功能之外，Bark 还支持批处理处理，这意味着您可以同时处理多个文本条目，但计算负荷会更大。在一些硬件上，如 GPU 上，批处理能够实现更快的整体生成，这意味着一次性生成样本可能比逐个生成样本更快。

让我们尝试生成一些示例：

```python
input_list = [
    "[clears throat] Hello uh ..., my dog is cute [laughter]",
    "Let's try generating speech, with Bark, a text-to-speech model",
    "♪ In the jungle, the mighty jungle, the lion barks tonight ♪",
]

# also add a speaker embedding
inputs = processor(input_list, voice_preset="v2/en_speaker_3")

speech_output = model.generate(**inputs).cpu().numpy()
```

让我们逐一听一下输出。

第一个：

```python
from IPython.display import Audio

sampling_rate = model.generation_config.sample_rate
Audio(speech_output[0], rate=sampling_rate)
```

第二个：

```python
Audio(speech_output[1], rate=sampling_rate)
```

第三个：

```python
Audio(speech_output[2], rate=sampling_rate)
```

Bark，像其他 🤗 Transformers 模型一样，可以通过几行代码进行速度和内存影响的优化。要了解详情，请单击此 Colab 演示笔记本。

### Massive Multilingual Speech (MMS)

如果您正在寻找除英语以外的其他语言的预训练模型怎么办？Massive Multilingual Speech（MMS）是另一个涵盖多种语音任务的模型，但它支持大量的语言。例如，它可以在超过1,100种语言中合成语音。

MMS的文本到语音功能基于[VITS Kim等人，2021年的技术](https://arxiv.org/pdf/2106.06103.pdf)，这是一种最先进的TTS方法之一。

VITS是一个语音生成网络，将文本转换为原始语音波形。它的工作原理类似于条件变分自编码器，从输入文本中估计音频特征。首先，生成声学特征，表示为频谱图。然后使用从HiFi-GAN改编的转置卷积层解码波形。在推断过程中，文本编码被上采样，并使用流模块和HiFi-GAN解码器转换为波形。与Bark一样，无需使用声码器，因为波形是直接生成的。

MMS模型最近已添加到🤗 Transformers中，因此您需要从源代码安装库：

```shell
pip install git+https://github.com/huggingface/transformers.git
```

让我们尝试一下MMS，并看看我们如何合成其他语言（如德语）的语音。首先，我们将加载正确语言的模型检查点和分词器：

```python
from transformers import VitsModel, VitsTokenizer

model = VitsModel.from_pretrained("facebook/mms-tts-deu")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-deu")
```

您可能会注意到，要加载MMS模型，您需要使用VitsModel和VitsTokenizer。这是因为MMS的文本到语音功能基于前面提到的VITS模型。

让我们选择德语中的一个示例文本，例如儿童歌曲的前两行：

```python
text_example = (
    "Ich bin Schnappi das kleine Krokodil, komm aus Ägypten das liegt direkt am Nil."
)
```

要生成波形输出，请使用分词器对文本进行预处理，并将其传递给模型：

```python
import torch

inputs = tokenizer(text_example, return_tensors="pt")
input_ids = inputs["input_ids"]


with torch.no_grad():
    outputs = model(input_ids)

speech = outputs["waveform"]
```

让我们来听听：

```python
from IPython.display import Audio

Audio(speech, rate=16000)
```

太棒了！如果您想尝试MMS的其他语言，请在🤗 [Hub](https://huggingface.co/models?filter=vits)上找到其他合适的vits检查点。

现在让我们看看您如何自己微调TTS模型！


## 微调SpeechT5

现在您已经熟悉了文本到语音任务以及SpeechT5模型的内部工作原理，该模型是在英语数据上进行预训练的，让我们看看如何将其微调到另一种语言。

### 准备工作

确保您有一个GPU，如果您想要重现这个示例。在笔记本中，您可以使用以下命令进行检查：

```shell
nvidia-smi
```

在我们的示例中，我们将使用约40小时的训练数据。如果您想要使用Google Colab的免费GPU跟随操作，您将需要将训练数据量减少到约10-15小时，并减少训练步骤的数量。

您还需要一些额外的依赖项：

```shell
pip install transformers datasets soundfile speechbrain accelerate
```

最后，不要忘记登录您的Hugging Face账户，这样您就可以上传和与社区分享您的模型：

```python
from huggingface_hub import notebook_login

notebook_login()
```

### 数据集

在本示例中，我们将使用[VoxPopuli数据集](https://huggingface.co/datasets/facebook/voxpopuli)的荷兰语（nl）子集。VoxPopuli是一个大规模的多语言语音语料库，由2009年至2020年欧洲议会事件录音的数据构成。它包含了15种欧洲语言的带标签的音频转录数据。虽然我们将使用荷兰语子集，但您可以随意选择其他子集。

这是一个自动语音识别（ASR）数据集，所以，如前所述，它并不是训练TTS模型的最合适选项。但是，对于这个练习来说，它是足够好的。

让我们加载数据：

```python
from datasets import load_dataset, Audio

dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
len(dataset)
```

输出：
```
20968
```


20968个示例应该足够用于微调。SpeechT5期望音频数据的采样率为16 kHz，因此请确保数据集中的示例满足此要求：

```python
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```


### 数据预处理

让我们首先定义要使用的模型检查点，并加载适当的处理器，该处理器包含我们将需要为训练准备数据的分词器和特征提取器：

```python
from transformers import SpeechT5Processor

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
```

#### SpeechT5分词的文本清理

首先，为了准备文本，我们需要处理器的分词器部分，所以让我们获取它：

```python
tokenizer = processor.tokenizer
```


让我们看一个例子：

```python
dataset[0]
```

输出：
```python
{'audio_id': '20100210-0900-PLENARY-3-nl_20100210-09:06:43_4',
 'language': 9,
 'audio': {'path': '/root/.cache/huggingface/datasets/downloads/extracted/02ec6a19d5b97c03e1379250378454dbf3fa2972943504a91c7da5045aa26a89/train_part_0/20100210-0900-PLENARY-3-nl_20100210-09:06:43_4.wav',
  'array': array([ 4.27246094e-04,  1.31225586e-03,  1.03759766e-03, ...,
         -9.15527344e-05,  7.62939453e-04, -2.44140625e-04]),
  'sampling_rate': 16000},
 'raw_text': 'Dat kan naar mijn gevoel alleen met een brede meerderheid die wij samen zoeken.',
 'normalized_text': 'dat kan naar mijn gevoel alleen met een brede meerderheid die wij samen zoeken.',
 'gender': 'female',
 'speaker_id': '1122',
 'is_gold_transcript': True,
 'accent': 'None'}
```



您可能会注意到数据集示例包含原始文本和规范化文本特征。在决定使用哪个特征作为文本输入时，重要的是要知道SpeechT5分词器没有任何数字标记。在规范化文本中，数字被写成文本形式。因此，它更适合，我们应该使用规范化文本作为输入文本。

由于SpeechT5是在英语上进行训练的，它可能无法识别荷兰语数据集中的某些字符。如果保持原样，这些字符将被转换为\<unk>标记。然而，在荷兰语中，像à这样的字符用于强调音节。为了保持文本的含义，我们可以将此字符替换为常规的a。

要识别不受支持的标记，使用SpeechT5Tokenizer从数据集中提取所有唯一字符，该分词器使用字符作为标记。为此，我们将编写extract_all_chars映射函数，将所有示例的转录串联成一个字符串，并将其转换为字符集。确保在dataset.map()中设置batched=True和batch_size=-1，以便所有转录一次性对映射函数可用。

```python
def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```


现在你有两组字符：一个是数据集的词汇表，另一个是分词器的词汇表。要识别数据集中的任何不受支持的字符，您可以取这两组集合的差异。结果集将包含数据集中存在但分词器中不存在的字符。

```python
dataset_vocab - tokenizer_vocab
```

输出：

```python
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```


为了处理前一步中识别出的不受支持的字符，我们可以定义一个函数，将这些字符映射到有效的标记。请注意，分词器已经用▁替换了空格，并且不需要单独处理。

```python
replacements = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]


def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs


dataset = dataset.map(cleanup_text)
```

现在我们已经处理了文本中的特殊字符，是时候将焦点转移到音频数据了。

#### 说话人

VoxPopuli数据集包含来自多位说话人的语音，但数据集中代表了多少位说话人呢？为了确定这一点，我们可以统计唯一说话人的数量以及每位说话人对数据集的贡献。数据集中共有20,968个示例，这些信息将让我们更好地了解数据中说话人和示例的分布情况。

```python
from collections import defaultdict

speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1
```



通过绘制直方图，您可以了解每位说话人的数据量。

```python
import matplotlib.pyplot as plt

plt.figure()
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.show()
```

<a>![](/img/hfaudio/unit6/3.png)</a>

直方图显示，数据集中约有三分之一的说话人少于100个示例，而约有十位说话人拥有500个以上的示例。为了提高训练效率和平衡数据集，我们可以将数据限制在拥有100到400个示例的说话人范围内。

```python
def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400


dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```


让我们来检查一下剩下多少位说话人：

```python
len(set(dataset["speaker_id"]))
```

输出：

```
42
```


让我们看看还剩下多少个示例：

```python
len(dataset)
```

输出：

```
9973
```

您现在剩下的示例数量略少于10,000个，来自大约40位独特的说话人，这应该是足够的。

请注意，一些拥有较少示例的说话人实际上可能有更多的音频可用，如果示例很长的话。然而，确定每位说话人的总音频量需要扫描整个数据集，这是一个耗时的过程，涉及加载和解码每个音频文件。因此，在这里我们选择跳过此步骤。

#### 说话者嵌入

为了使TTS模型能够区分多个说话者，您需要为每个示例创建一个说话者嵌入。说话者嵌入是模型的额外输入，捕捉特定说话者的语音特征。要生成这些说话者嵌入，请使用SpeechBrain中的预训练[spkrec-xvect-voxceleb模型](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)。

创建一个名为create_speaker_embedding()的函数，它接受一个输入音频波形，并输出一个包含相应说话者嵌入的512元素向量。

```python
import os
import torch
from speechbrain.pretrained import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings
```


需要注意的是，speechbrain/spkrec-xvect-voxceleb模型是在VoxCeleb数据集中的英语语音上进行训练的，而本指南中的训练示例是荷兰语的。尽管我们相信这个模型仍然会为我们的荷兰数据集生成合理的说话者嵌入，但这个假设在所有情况下可能都不成立。

为了获得最佳结果，我们需要首先在目标语音上训练一个X-Vector模型。这将确保模型能够更好地捕捉荷兰语中存在的独特语音特征。如果您想要训练自己的X-Vector模型，可以使用[这个脚本](https://huggingface.co/mechanicalsea/speecht5-vc/blob/main/manifest/utils/prep_cmu_arctic_spkemb.py)作为示例。


#### 处理数据集

最后，让我们将数据处理成模型期望的格式。创建一个prepare_dataset函数，它接受一个单独的示例，并使用SpeechT5Processor对象对输入文本进行标记化，并将目标音频加载到对数梅尔频谱图中。它还应将说话者嵌入作为额外的输入添加进去。

```python
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example
```


通过查看一个单独的示例来验证处理是否正确：

```python
processed_example = prepare_dataset(dataset[0])
list(processed_example.keys())
```

输出：

```
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

说话者嵌入应该是一个512元素向量：

```python
processed_example["speaker_embeddings"].shape
```
输出：
```
(512,)
```
标签应该是一个具有80个mel bin的对数梅尔频谱图。

```python
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(processed_example["labels"].T)
plt.show()
```

<a>![](/img/hfaudio/unit6/4.png)</a>

顺便提一句：如果您觉得这个频谱图令人困惑，可能是因为您熟悉的约定是将低频放在图的底部，高频放在顶部。然而，当使用matplotlib库将频谱图绘制为图像时，y轴被翻转，频谱图会倒置显示。

现在我们需要将处理函数应用于整个数据集。这将花费5到10分钟的时间。

```python
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

您将会看到一个警告，说明数据集中的一些示例比模型能处理的最大输入长度（600个标记）要长。从数据集中删除这些示例。在这里，我们进一步去掉了超过200个标记的示例，以便允许更大的批次大小。

```python
def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
len(dataset)
```

输出：

```python
8259
```

接下来，创建一个基本的训练/测试分割：

```python
dataset = dataset.train_test_split(test_size=0.1)
```



#### 数据整理器

为了将多个示例组合成一个批次，您需要定义一个自定义数据整理器。这个整理器将用填充标记填充较短的序列，确保所有示例具有相同的长度。对于频谱图标签，填充部分将被替换为特殊值-100。这个特殊值指示模型在计算频谱图损失时忽略该部分频谱图。

```
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
```

在SpeechT5中，模型解码器部分的输入减少了一个因子2。换句话说，它会从目标序列中丢弃每隔一个时间步长。然后，解码器会预测一个长度是原始目标序列长度两倍的序列。由于原始目标序列的长度可能是奇数，数据整理器确保将批次的最大长度向下取整为2的倍数。

```python
data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## 训练模型

从与加载处理器时相同的检查点加载预训练模型：

```python
from transformers import SpeechT5ForTextToSpeech

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

use_cache=True选项与梯度检查点不兼容。在训练期间禁用它，并重新启用缓存以加快推理时间：

```
from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate, use_cache=True)
```

定义训练参数。在这里，我们在训练过程中不计算任何评估指标，我们将在本章后面讨论评估。相反，我们只关注损失：

```
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)
```


实例化Trainer对象并将模型、数据集和数据整理器传递给它。

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)
```

有了这些，我们准备开始训练了！训练将需要几个小时。根据您的GPU，当您开始训练时，可能会遇到CUDA“内存不足”的错误。在这种情况下，您可以逐渐将per_device_train_batch_size减小到2的倍数，并将gradient_accumulation_steps增加到2倍以进行补偿。

```python
trainer.train()
```

将最终模型推送到🤗 Hub：

```python
trainer.push_to_hub()
```

### 推理
一旦您对模型进行了微调，就可以用它进行推理！从🤗 Hub加载模型（确保在以下代码片段中使用您的帐户名称）：

```python
model = SpeechT5ForTextToSpeech.from_pretrained(
    "YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl"
)
```

选择一个示例，这里我们将从测试数据集中选择一个。获取一个说话者嵌入。

```python
example = dataset["test"][304]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

定义一些输入文本并对其进行标记化。

```python
text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

预处理输入文本：

```python
inputs = processor(text=text, return_tensors="pt")
```

实例化一个语音合成器并生成语音：

```python
from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
```

准备好听结果了吗？

```python
from IPython.display import Audio

Audio(speech.numpy(), rate=16000)
```

在新语言上获得令人满意的结果可能是具有挑战性的。说话者嵌入的质量可能是一个重要因素。由于SpeechT5是使用英语x-vector进行预训练的，因此在使用英语说话者嵌入时性能最佳。如果合成的语音听起来质量不佳，请尝试使用不同的说话者嵌入。

增加训练时长也有可能提高结果的质量。即便如此，这段语音明显是荷兰语而不是英语，并且它确实捕捉到了说话者的语音特征（与示例中的原始音频进行比较）。另一个要尝试的是模型的配置。例如，尝试使用config.reduction_factor = 1，看看是否会改善结果。

在下一节中，我们将讨论如何评估文本到语音模型。


## 评估文本到语音模型

在训练期间，文本到语音模型通过预测的频谱图值和生成的频谱图之间的均方误差（或平均绝对误差）来优化。MSE和MAE都鼓励模型最小化预测和目标频谱图之间的差异。然而，由于TTS是一个一对多映射问题，即给定文本的输出频谱图可以用许多不同的方式表示，因此对生成的文本到语音（TTS）模型进行评估要困难得多。

与许多其他可以使用定量指标（如准确率或精度）客观测量的计算任务不同，评估TTS主要依赖于主观的人类分析。

用于TTS系统的最常用的评估方法之一是使用平均意见得分（MOS）进行定性评估。MOS是一种主观评分系统，允许人类评估者根据从1到5的标尺评价合成语音的感知质量。这些分数通常通过听力测试收集，其中人类参与者会听取并评价合成语音样本。

开发TTS评估的客观指标具有挑战性的一个主要原因是语音感知的主观性质。人类听众对于语音的各个方面，包括发音、语调、自然度和清晰度等，有不同的偏好和敏感度。用单一的数值来捕捉这些感知细微差别是一项艰巨的任务。与此同时，人类评估的主观性使得比较和基准不同的TTS系统变得具有挑战性。

此外，这种评估方法可能忽视了语音合成的某些重要方面，如自然度、表现力和情感影响。这些品质难以客观量化，但在需要合成语音传达类人特质并引发适当情感反应的应用中非常相关。

总之，评估文本到语音模型是一个复杂的任务，因为缺乏一个真正客观的指标。最常见的评估方法，平均意见得分（MOS），依赖于主观的人类分析。虽然MOS为合成语音的质量提供了宝贵的见解，但它也引入了变异性和主观性。

