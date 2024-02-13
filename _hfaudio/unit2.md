---
layout:     post
title:      "2. A gentle introduction to audio applications"
author:     "lili"
mathjax: true
sticky: true
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

欢迎来到 Hugging Face 音频课程的第二单元！之前，我们探索了音频数据的基础知识，并学习了如何使用 🤗 Datasets和 🤗 Transformers 库处理音频数据集。我们讨论了各种概念，如采样率、振幅、位深度、波形和频谱图，并看到了如何预处理数据以准备为预训练模型使用。

此时，您可能渴望了解 🤗 Transformers 可以处理的音频任务，并且您已经具备了深入了解所需的所有基础知识！让我们看一些令人惊叹的音频任务示例：

* 音频分类(Audio Classification)：轻松将音频剪辑分类到不同的类别中。您可以识别录音是狗叫还是猫叫，或者歌曲属于哪种音乐流派。
* 自动语音识别(Automatic Speech Recognition)：将音频剪辑转换为文本，通过自动转录它们。您可以得到某人说话的录音的文本表示，例如“你今天过得怎么样？”。对于做笔记非常有用！
* 说话人辨识(Speaker Diarization)：曾经想知道录音中是谁在说话吗？通过 🤗 Transformers，您可以确定在音频剪辑中任何给定时间是哪位发言者。想象一下，在一段录音中区分“爱丽丝”和“鲍勃”的能力。
* 文本到语音(Text to Speech)：创建一个文本的朗读版本，可用于制作有声书，帮助辅助阅读，或为游戏中的非玩家角色（NPC）提供声音。使用 🤗 Transformers，您可以轻松实现！

在本单元中，您将学习如何使用 🤗 Transformers 的 pipeline() 函数为这些任务中的一些使用预训练模型。具体来说，我们将看到预训练模型如何用于音频分类、自动语音识别和音频生成。让我们开始吧！

## 使用pipeline进行音频分类

音频分类涉及根据其内容为音频录音分配一个或多个标签。这些标签可以对应于不同的声音类别，如音乐、语音或噪音，或更具体的类别，如鸟鸣或汽车引擎声音。

在深入研究最流行的音频变换器如何工作之前，也在微调自定义模型之前，让我们看看如何只使用几行代码就可以利用 🤗 Transformers 中的现成预训练模型进行音频分类。

让我们继续使用您在上一单元中探索过的相同的 MINDS-14 数据集。如果您还记得，MINDS-14 包含人们在多种语言和方言中向电子银行系统提问的录音，并且每个录音都有意图类别。我们可以按照通话的意图对录音进行分类。

就像以前一样，让我们从加载数据的 en-AU 子集开始尝试pipeline，并将其上采样到 16kHz 的采样率，这是大多数语音模型所需的。


```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```


要将音频录音分类到一组类别中，我们可以使用 🤗 Transformers 中的音频分类pipeline。在我们的情况下，我们需要一个在 MINDS-14 数据集上进行意图分类微调的模型。幸运的是，Hub 有一个正好符合我们要求的模型！让我们使用 pipeline() 函数加载它：



```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
``` 

该pipeline期望音频数据作为 NumPy 数组。原始音频数据的所有预处理将由pipeline方便地处理。让我们选择一个示例来尝试一下：

```python
example = minds[0]
```

如果您还记得数据集的结构，原始音频数据存储在 ["audio"]["array"] 下的 NumPy 数组中，让我们直接将其传递给分类器：

```python
classifier(example["audio"]["array"])
```

输出:

```python
[
    {"score": 0.9631525278091431, "label": "pay_bill"},
    {"score": 0.02819698303937912, "label": "freeze"},
    {"score": 0.0032787492964416742, "label": "card_issues"},
    {"score": 0.0019414445850998163, "label": "abroad"},
    {"score": 0.0008378693601116538, "label": "high_value_payment"},
]
```

模型非常自信地认为呼叫者的意图是了解如何支付他们的账单。让我们看看这个示例的实际标签是什么：

```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
```

输出：

```python
"pay_bill"
```



太棒了！预测的标签是正确的！在这里，我们很幸运地找到了一个可以精确分类我们所需要的标签的模型。许多时候，处理分类任务时，预训练模型的类别集合与您需要模型区分的类别不完全相同。在这种情况下，您可以对预训练模型进行微调，以“校准”它以匹配您的确切类别标签集。我们将在即将到来的单元中学习如何做到这一点。现在，让我们来看看语音处理中另一个非常常见的任务，即自动语音识别。


## 使用pipeline进行自动语音识别（ASR）

自动语音识别（ASR）是一项任务，涉及将语音音频录音转录为文本。这项任务有许多实际应用，从为视频创建闭路字幕到为虚拟助手如 Siri 和 Alexa 启用语音命令。

在本节中，我们将使用自动语音识别pipeline来转录一个人使用与之前相同的 MINDS-14 数据集询问支付账单问题的音频录音。

首先，加载数据集并将其上采样到 16kHz，就像在音频分类中使用pipeline一样，如果您尚未执行此操作。

要转录音频录音，我们可以使用 🤗 Transformers 中的自动语音识别pipeline。让我们实例化pipeline：

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition")
```

接下来，我们将从数据集中选择一个示例，并将其原始数据传递给pipeline：

```python
example = minds[0]
asr(example["audio"]["array"])
``

输出：

```python
{"text": "I WOULD LIKE TO PAY MY ELECTRICITY BILL USING MY COD CAN YOU PLEASE ASSIST"}
```

让我们将此输出与该示例的实际转录进行比较：

```
example["english_transcription"]
```

输出：

```
"I would like to pay my electricity bill using my card can you please assist"
```

模型似乎在转录音频时做得相当不错！与原始转录相比，它只有一个词错误（“card”），这相当不错，考虑到说话者有澳大利亚口音，其中字母 “r” 通常是无声的。话虽如此，我不建议尝试使用一条鱼支付您的下一个电费！

默认情况下，此pipeline使用针对英语的自动语音识别模型，这在这个示例中是可以的。如果您想尝试以不同语言的其他 MINDS-14 子集进行转录，您可以在 🤗 Hub 上找到一个预训练的 ASR 模型。您可以首先按任务筛选模型列表，然后按语言筛选。找到您喜欢的模型后，将其名称作为 model 参数传递给pipeline。

让我们尝试一下 MINDS-14 的德语子集。加载 “de-DE” 子集：

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

获取一个示例并查看转录应该是什么：

```python
example = minds[0]
example["transcription"]
```

输出：

```
"ich möchte gerne Geld auf mein Konto einzahlen"
```

在 🤗 Hub 上找到一个预训练的德语语音识别模型，实例化pipeline，并转录示例：

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="maxidl/wav2vec2-large-xlsr-german")
asr(example["audio"]["array"])
```

输出：

```python
{"text": "ich möchte gerne geld auf mein konto einzallen"}
```

没错！

在解决自己的任务时，从像我们在本单元中展示的简单pipeline开始是一个宝贵的工具，它带来了几个好处：

* 可能已经存在一个预训练模型，它已经很好地解决了您的任务，从而为您节省了大量时间。
* pipeline() 为您处理了所有的预处理/后处理工作，因此您不必担心将数据转换为模型所需的正确格式。
* 如果结果不理想，这仍然为将来进行微调提供了一个快速的基线。
* 一旦您在自定义数据上进行了微调，并将其共享到 Hub 上，整个社区就可以通过 pipeline() 方法快速轻松地使用它，从而使 AI 更加易于使用。

## 使用pipeline进行音频生成

音频生成涵盖了一系列多用途任务，涉及生成音频输出。我们在这里要研究的任务是语音生成（又称“文本转语音”）和音乐生成。在文本转语音中，模型将一段文本转换为生动的口语语言声音，为虚拟助手、视觉受损者的辅助工具以及个性化有声书等应用打开了大门。另一方面，音乐生成可以促进创造性表达，并且主要在娱乐和游戏开发行业中使用。

在 🤗 Transformers 中，您将找到一个涵盖这两项任务的pipeline。此pipeline称为 "text-to-audio"，但为了方便起见，它还有一个 "text-to-speech" 别名。在这里，我们将同时使用这两个别名，您可以根据任务选择其中一个。

让我们探索如何使用此pipeline仅通过几行代码开始生成文本的音频叙述和音乐。

这个pipeline是 🤗 Transformers 中的新功能，是版本 4.32 发布的一部分。因此，您需要将库升级到最新版本才能获得此功能：

```shell
pip install --upgrade transformers
```

### 生成语音

让我们首先探索文本转语音生成。首先，与音频分类和自动语音识别一样，我们需要定义pipeline。我们将定义一个 text-to-speech pipeline，因为它最好描述了我们的任务，并使用 suno/bark-small 检查点：

```python
from transformers import pipeline

pipe = pipeline("text-to-speech", model="suno/bark-small")
```

接下来的步骤就像将一些文本通过pipeline传递一样简单。所有预处理将在幕后为我们完成：

```python
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
output = pipe(text)
```

在notebook中，我们可以使用以下代码片段来听取结果：

```python
from IPython.display import Audio

Audio(output["audio"], rate=output["sampling_rate"])
```

我们与pipeline一起使用的模型 Bark 实际上是多语言的，因此我们可以轻松地用法语等其他语言的文本替换初始文本，并以完全相同的方式使用pipeline。它将自动识别语言：

```python
fr_text = "Contrairement à une idée répandue, le nombre de points sur les élytres d'une coccinelle ne correspond pas à son âge, ni en nombre d'années, ni en nombre de mois. "
output = pipe(fr_text)
Audio(output["audio"], rate=output["sampling_rate"])
```

这个模型不仅是多语言的，还可以生成带有非语言交流和歌唱的音频。以下是如何让它唱歌的方法：

```python
song = "♪ In the jungle, the mighty jungle, the ladybug was seen. ♪ "
output = pipe(song)
Audio(output["audio"], rate=output["sampling_rate"])
```

我们将在后续专门用于文本转语音的单元中深入了解 Bark 的特性，并展示如何使用其他模型进行此任务。现在，让我们生成一些音乐！

### 生成音乐

与之前一样，我们将首先实例化一个pipeline。对于音乐生成，我们将定义一个 text-to-audio pipeline，并使用预训练的检查点 facebook/musicgen-small 进行初始化。

```python
music_pipe = pipeline("text-to-audio", model="facebook/musicgen-small")
```

让我们创建一个音乐的文本描述：

```
text = "90s rock song with electric guitar and heavy drums"
```

我们可以通过向模型传递一个额外的 max_new_tokens 参数来控制生成输出的长度。

```python
forward_params = {"max_new_tokens": 512}

output = music_pipe(text, forward_params=forward_params)
Audio(output["audio"][0], rate=output["sampling_rate"])
```





