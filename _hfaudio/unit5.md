---
layout:     post
title:      "5. Automatic Speech Recognition"
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


## 本单元学习内容
 

在本节中，我们将看看如何使用Transformers将口语转换为文本，这是一项称为语音识别的任务。

<a>![](/img/hfaudio/unit5/1.png)</a>

语音识别，也称为自动语音识别（ASR）或语音转文本（STT），是最受欢迎和令人兴奋的口语处理任务之一。它在各种应用中被使用，包括口述、语音助手、视频字幕和会议记录。

您可能以前已经多次使用过语音识别系统，但没有意识到！考虑一下您的智能手机设备中的数字助手（Siri、Google Assistant、Alexa）。当您使用这些助手时，它们首先要做的是将您的口语转录成书面文本，准备用于任何下游任务（比如为您查找天气🌤️）。

尝试一下下面的语音识别演示。您可以使用麦克风录制自己，也可以拖放一个音频样本进行转录。

语音识别是一项具有挑战性的任务，因为它需要对音频和文本有联合的知识。输入音频可能有很多背景噪音，并由具有不同口音的讲话者说话，这使得很难辨别出口语。书面文本可能包含一些在音频中没有声音的字符，例如标点符号，这些字符仅从音频中推断是困难的。这些都是我们在构建有效的语音识别系统时必须克服的障碍！

现在我们已经定义了我们的任务，我们可以开始更详细地研究语音识别。通过本单元的学习，您将对可用的不同预训练语音识别模型有良好的基础理解，以及如何使用🤗 Transformers库与它们一起使用。您还将了解在所选择的领域或语言上对ASR模型进行微调的程序，使您能够构建一个用于遇到的任何任务的高性能系统。您将能够通过构建一个实时演示来向您的朋友和家人展示您的模型，该演示可以接收任何口语并将其转换为文本！

具体来说，我们将涵盖以下内容：

* 语音识别的预训练模型
* 选择数据集
* 语音识别的评估和度量标准
* 使用Trainer API对ASR系统进行微调
* 构建演示
* 实践练习

## 自动语音识别的预训练模型

在本节中，我们将介绍如何使用pipeline()来利用预训练模型进行语音识别。在第2单元中，我们将pipeline()介绍为一种简单的方法来运行语音识别任务，所有的预处理和后处理都在幕后处理，并且具有在Hugging Face Hub上快速实验任何预训练检查点的灵活性。在本单元中，我们将深入探讨语音识别模型的不同属性以及如何使用它们来解决一系列不同的任务。

正如第3单元详细介绍的那样，语音识别模型主要分为两类：

* 连接主义时间分类（CTC）：仅编码器模型，在顶部有一个线性分类（CTC）头部
* 序列到序列（Seq2Seq）：编码器-解码器模型，编码器和解码器之间有交叉注意力机制

在2022年之前，CTC是这两种架构中更流行的一种，仅编码器模型（如Wav2Vec2、HuBERT和XLSR）在语音的预训练/微调范式中取得了突破。大公司，如Meta和微软，使用大量未标记的音频数据对编码器进行了预训练，为了在下游语音识别任务中获得强大的性能，用户可以采用预训练检查点，只需对10分钟左右的标记语音数据进行微调即可。

然而，CTC模型也有其缺点。将一个简单的线性层附加到编码器会得到一个小而快速的整体模型，但可能容易出现语音拼写错误。我们将在下面的示例中为Wav2Vec2模型演示这一点。

#### 探究CTC模型

让我们加载[LibriSpeech ASR](https://huggingface.co/learn/audio-course/en/chapter5/hf-internal-testing/librispeech_asr_dummy)数据集的一个小节，以展示Wav2Vec2的语音转录能力：

```python
from datasets import load_dataset

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
dataset
```

输出：

```python
Dataset({
    features: ['file', 'audio', 'text', 'speaker_id', 'chapter_id', 'id'],
    num_rows: 73
})
```



我们可以选择73个音频样本中的一个，并检查音频样本以及转录内容：

```python
from IPython.display import Audio

sample = dataset[2]

print(sample["text"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

输出：

```
HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
```

【译注：大写看着别扭，下面是小写和中文翻译

```
he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind
他告诉我们，在这年终的节日季节，圣诞节和烤牛肉即将到来时，由于食物和其结果而绘制的比喻最容易浮现在脑海中。
```

】



好了！圣诞节和烤牛肉，听起来不错！🎄选择了一个数据样本后，我们现在将一个微调的检查点加载到pipeline()中。为此，我们将使用官方在100小时LibriSpeech数据上微调的[Wav2Vec2基础检查点](https://huggingface.co/learn/audio-course/en/chapter5/facebook/wav2vec2-base-100h)：

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-100h")
```


接下来，我们将从数据集中取一个示例，并将其原始数据传递给pipeline。由于pipeline消耗我们传递的任何字典（这意味着它无法被重用），我们将传递数据的副本。这样，我们可以安全地在以下示例中重用相同的音频样本：

```python
pipe(sample["audio"].copy())
```

输出：

```
{"text": "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAUS AND ROSE BEEF LOOMING BEFORE US SIMALYIS DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND"}
```


我们可以看到，Wav2Vec2模型在转录这个样本时做得相当不错-乍一看，它看起来总体上是正确的。让我们将目标和预测放在一起，突出显示它们之间的差异：

```
Target:      HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
Prediction:  HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH **CHRISTMAUS** AND **ROSE** BEEF LOOMING BEFORE US **SIMALYIS** DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
```


比较目标文本和预测的转录，我们可以看到所有单词听起来都正确，但有些单词的拼写不准确。例如：

* CHRISTMAUS与CHRISTMAS
* ROSE与ROAST
* SIMALYIS与SIMILES

这突显了CTC模型的缺点。CTC模型本质上是一个“仅声学”的模型：它由一个编码器和一个线性层组成，编码器从音频输入中形成隐藏状态表示，线性层将隐藏状态映射到字符：

这意味着系统几乎完全基于给定的声学输入（音频的语音声音）进行预测，因此有将音频以语音方式转录的倾向（例如CHRISTMAUS）。它对前后文的语言建模上下文的重要性较小，并且容易产生语音拼写错误。更智能的模型会识别CHRISTMAUS不是英语词汇中的有效单词，并在进行预测时将其更正为CHRISTMAS。此外，我们的预测还缺少两个重要特征-大小写和标点符号-这限制了模型转录在现实世界应用中的有效性。


### 进阶至Seq2Seq模型

欢迎Seq2Seq模型登场！如第3单元所述，Seq2Seq模型由一个编码器和一个解码器组成，通过交叉注意力机制连接起来。编码器扮演了与之前相同的角色，计算音频输入的隐藏状态表示，而解码器则扮演语言模型的角色。解码器处理来自编码器的整个隐藏状态表示序列，并生成相应的文本转录。有了音频输入的全局上下文，解码器能够在进行预测时使用语言模型的上下文，即时纠正拼写错误，从而规避了发音预测的问题。

Seq2Seq模型有两个缺点：

* 解码速度较慢，因为解码过程是逐步进行的，而不是一次性完成的
* 它们对数据的需求更大，需要更多的训练数据才能达到收敛

特别是，对于语音的Seq2Seq架构的进展，对大量的训练数据的需求一直是一个瓶颈。标记的语音数据很难获取，目前最大的已注释数据集仅有10,000小时。这一切都在2022年改变了，Whisper的发布改变了一切。Whisper是由OpenAI的Alec Radford等人于2022年9月发布的用于语音识别的预训练模型。与其CTC前辈完全基于未标记的音频数据进行预训练不同，Whisper是在大量标记的音频-转录数据上进行预训练的，精确到680,000小时。

这比用于训练Wav2Vec 2.0（60,000小时）的未标记音频数据大一个数量级。更重要的是，这些预训练数据中的117,000小时是多语言（或“非英语”）数据。这导致了可以应用于96种语言的检查点，其中许多被认为是低资源的，意味着该语言缺乏用于训练的大型数据语料库。

将训练数据扩展到680,000小时的标记预训练数据后，Whisper模型展现出了强大的泛化能力，适用于许多数据集和领域。预训练的检查点在测试集中的表现与最先进的管道系统相比具有竞争力，对于LibriSpeech的测试清晰子集的字错误率（WER）接近3％，在TED-LIUM上达到了新的最先进水平，WER为4.7％（参见Whisper[论文](https://cdn.openai.com/papers/whisper.pdf)的表8）。

Whisper特别重要的一点是其处理长音频样本的能力，对输入噪声的鲁棒性以及预测大小写和标点符号的能力。这使得它成为了实际语音识别系统的可行选择。

本节的其余部分将向您展示如何使用🤗 Transformers的预训练Whisper模型进行语音识别。在许多情况下，预训练的Whisper检查点非常高效，并能够获得良好的结果，因此我们鼓励您尝试使用预训练检查点作为解决任何语音识别问题的第一步。通过微调，可以将预训练检查点调整为特定数据集和语言，以进一步改善这些结果。我们将在即将介绍的微调子节中演示如何实现这一点。

Whisper检查点有五种不同大小的配置。前四种大小适用于仅英语或多语言数据。最大的检查点仅适用于多语言。所有九个预训练检查点都可以在[Hugging Face Hub](https://huggingface.co/models?search=openai/whisper)上找到。以下表格总结了这些检查点，并提供了指向Hub上模型的链接。 “VRAM”表示以最小批处理大小为1运行模型所需的GPU内存。 “Rel Speed”是检查点相对于最大模型的相对速度。根据这些信息，您可以选择最适合您硬件的检查点。


Size |	Parameters |	VRAM / GB |		Rel Speed |		English-only |		Multilingual
--|--|--|--|--
tiny |		39 M |		1.4 |		32 |		[✓](https://huggingface.co/openai/whisper-tiny.en) |		[✓](https://huggingface.co/openai/whisper-tiny)
base |		74 M |		1.5 |		16 |		[✓](https://huggingface.co/openai/whisper-base.en) |		[✓](https://huggingface.co/openai/whisper-base)
small |		244 M |		2.3 |		6 |		[✓](https://huggingface.co/openai/whisper-small.en) |		[✓](https://huggingface.co/openai/whisper-small)
medium |		769 M |		4.2 |		2 |		[✓](https://huggingface.co/openai/whisper-medium.en) |		[✓](https://huggingface.co/openai/whisper-medium)
large |		1550 M |		7.5 |		1 |		x |		[✓](https://huggingface.co/openai/whisper-large-v2)




让我们加载[Whisper基本检查点](https://huggingface.co/openai/whisper-base)，它的大小与我们之前使用的Wav2Vec2检查点相当。预先考虑到我们将进行多语言语音识别，我们将加载基本检查点的多语言变体。我们还将在GPU上加载模型（如果可用），否则加载到CPU上。接下来的pipeline()将随后负责将所有输入/输出从CPU移动到GPU（如有需要）。

```python
import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)
```


太好了！现在让我们像以前一样转录音频。我们唯一的变化是传递了一个额外的参数max_new_tokens，它告诉模型在进行预测时生成的最大token数：

```python
pipe(sample["audio"], max_new_tokens=256)
```

输出：

```
{'text': ' He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind.'}
```

很简单！您会注意到存在大小写和标点。与来自Wav2Vec2的不带大小写和标点的转录相比，这使得转录更容易阅读。让我们将转录与目标放在一起进行对比：

```
Target:     HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
Prediction: He tells us that at this festive season of the year, with **Christmas** and **roast** beef looming before us, **similarly** is drawn from eating and its results occur most readily to the mind.
```


Whisper在纠正Wav2Vec2中看到的发音错误方面做得很好 - both Christmas和roast都拼写正确。我们看到模型仍然在SIMILES方面遇到困难，被错误地转录为similarly，但这次预测是英语词汇表中的一个有效单词。使用更大的Whisper检查点可以进一步减少转录错误，但需要更多的计算和更长的转录时间。

我们已经被承诺了一个可以处理96种语言的模型，因此现在让我们将英语语音识别留给以后，全球化吧🌎！[Multilingual LibriSpeech（MLS）](https://huggingface.co/datasets/facebook/multilingual_librispeech)数据集是LibriSpeech数据集的多语言等效版本，其中包含六种语言的标记音频数据。我们将从MLS数据集的西班牙语分割中加载一个样本，并利用流模式，这样我们就不必下载整个数据集：

```python
dataset = load_dataset(
    "facebook/multilingual_librispeech", "spanish", split="validation", streaming=True
)
sample = next(iter(dataset))
```


同样，我们将检查文本转录并听一下音频片段：

```python
print(sample["text"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

输出：
```
entonces te delelitarás en jehová y yo te haré subir sobre las alturas de la tierra y te daré á comer la heredad de jacob tu padre porque la boca de jehová lo ha hablado
```


这是我们的Whisper转录的目标文本。尽管我们现在知道我们可能可以做得更好，因为我们的模型还将预测标点和大小写，但我们的参考文本中没有这些。让我们将音频样本传递给pipeline以获取我们的文本预测。需要注意的一点是pipeline会消耗我们输入的音频输入的字典，这意味着字典不能被重复使用。为了避免这种情况，我们将传递音频样本的副本，以便我们可以在后续代码示例中重复使用相同的音频样本：

```python
pipe(sample["audio"].copy(), max_new_tokens=256, generate_kwargs={"task": "transcribe"})
```

输出：

```
{'text': ' Entonces te deleitarás en Jehová y yo te haré subir sobre las alturas de la tierra y te daré a comer la heredad de Jacob tu padre porque la boca de Jehová lo ha hablado.'}
```


很好 - 这看起来与我们的参考文本非常相似（可以说比我们的参考文本更好，因为它包含标点和大小写！）。您会注意到，我们将“task”设置为generate关键字参数（generate kwarg）。将“task”设置为“transcribe”会强制Whisper执行语音识别的任务，其中音频以其所说语言的方式转录为文本。Whisper还能够执行与之密切相关的任务，即语音翻译，其中西班牙语音频可以翻译为英文文本。为此，我们将“task”设置为“translate”：

```python
pipe(sample["audio"], max_new_tokens=256, generate_kwargs={"task": "translate"})
```

输出：

```
{'text': ' So you will choose in Jehovah and I will raise you on the heights of the earth and I will give you the honor of Jacob to your father because the voice of Jehovah has spoken to you.'}
```


现在我们知道我们可以在语音识别和语音翻译之间切换，我们可以根据需要选择我们的任务。要么我们从语言X的音频识别为相同语言X的文本（例如，西班牙语音频转为西班牙语文本），要么我们将从任何语言X的音频翻译为英文文本（例如，西班牙语音频转为英文文本）。

要了解有关如何使用“task”参数控制生成文本属性的更多信息，请参阅Whisper基础模型的[模型卡](https://huggingface.co/openai/whisper-base#usage)。

### 长篇转录和时间戳

到目前为止，我们专注于转录少于30秒的短音频样本。我们提到Whisper的吸引力之一是其能够处理长音频样本。我们将在这里解决这个任务！

让我们通过连接MLS数据集中的顺序样本来创建一个长音频文件。由于MLS数据集是通过将长的有声书录音分割为较短的片段来策划的，连接样本是重构更长有声书段落的一种方法。因此，结果音频应在整个样本上是连贯的。

我们将目标音频长度设置为5分钟，并在达到此值后停止连接样本：

```python
import numpy as np

target_length_in_m = 5

# convert from minutes to seconds (* 60) to num samples (* sampling rate)
sampling_rate = pipe.feature_extractor.sampling_rate
target_length_in_samples = target_length_in_m * 60 * sampling_rate

# iterate over our streaming dataset, concatenating samples until we hit our target
long_audio = []
for sample in dataset:
    long_audio.extend(sample["audio"]["array"])
    if len(long_audio) > target_length_in_samples:
        break

long_audio = np.asarray(long_audio)

# how did we do?
seconds = len(long_audio) / 16000
minutes, seconds = divmod(seconds, 60)
print(f"Length of audio sample is {minutes} minutes {seconds:.2f} seconds")
```

输出：

```
Length of audio sample is 5.0 minutes 17.22 seconds
```



好吧！5分钟17秒的音频要转录。直接将此长音频样本传递给模型存在两个问题：

* Whisper本质上设计用于处理30秒的样本：任何短于30秒的样本都会用静音填充到30秒，任何长于30秒的样本都会通过截取额外音频来截断到30秒，因此如果直接传递音频，我们只会得到前30秒的转录
* Transformer网络中的内存随着序列长度的平方而增加：加倍输入长度会使内存需求增加四倍，因此传递超长音频文件肯定会导致内存不足（OOM）错误



在🤗 Transformers中实现长篇转录的方式是将输入音频分成更小、更易管理的段。每个段与前一个段有少量的重叠。这样，我们就能准确地在边界处拼接这些段，因为我们可以找到段之间的重叠并相应地合并转录：


<a>![](/img/hfaudio/unit5/2.png)</a>

分段样本的优点在于我们不需要分段i的结果来转录随后的分段i+1。拼接是在我们转录了所有段之后在段边界处完成的，因此以哪种顺序转录段并不重要。该算法完全是无状态的，因此我们甚至可以同时进行chunk i+1和chunk i！这使我们可以对分段进行批处理，并与顺序转录相比并行运行模型，从而提供了大幅度的计算加速。要了解有关🤗 Transformers中分块的更多信息，您可以参考[此博客文章](https://huggingface.co/blog/asr-chunking)。

要激活长篇转录，我们在调用pipeline时必须添加一个额外的参数。这个参数，chunk_length_s，控制以秒为单位的分段段的长度。对于Whisper来说，30秒的片段是最佳的，因为这与Whisper期望的输入长度相匹配。

要激活批处理，我们需要将参数batch_size传递给pipeline。将所有内容综合起来，我们可以通过分段和批处理转录长音频样本如下：

```python
pipe(
    long_audio,
    max_new_tokens=256,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=30,
    batch_size=8,
)
```

输出：

```
{'text': ' Entonces te deleitarás en Jehová, y yo te haré subir sobre las alturas de la tierra, y te daré a comer la
heredad de Jacob tu padre, porque la boca de Jehová lo ha hablado. nosotros curados. Todos nosotros nos descarriamos
como bejas, cada cual se apartó por su camino, mas Jehová cargó en él el pecado de todos nosotros...
```



我们不会在此打印整个输出，因为它非常长（共312个单词）！在16GB V100 GPU上，您可以预期上述行大约需要3.45秒才能运行，这对于317秒的音频样本来说非常不错。在CPU上，预计要接近30秒。

Whisper还能够预测音频数据的段级时间戳。这些时间戳指示了音频短段的开始和结束时间，并且特别适用于将转录与输入音频对齐。假设我们想要为视频提供闭幕字幕-我们需要这些时间戳来知道转录的哪一部分对应于视频的某个段落，以便显示该时间的正确转录。

激活时间戳预测非常简单，我们只需要设置参数return_timestamps=True。时间戳与我们之前使用的分段和批处理方法都兼容，因此我们只需将时间戳参数追加到我们之前的调用中：


```python
pipe(
    long_audio,
    max_new_tokens=256,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=30,
    batch_size=8,
    return_timestamps=True,
)["chunks"]
```


太棒了！我们有了预测的文本以及相应的时间戳。

### 总结

Whisper是强大的语音识别和翻译的预训练模型。与Wav2Vec2相比，它具有更高的转录准确性，输出包含标点和大小写。它可以用于英语以及其他96种语言的语音转录，无论是在短音频片段还是通过分块处理更长音频时。这些特性使得它成为许多语音识别和翻译任务的可行模型，而无需进行微调。pipeline()方法提供了一个通过一行API调用进行推断的简单方法，并且可以控制生成的预测结果。

虽然Whisper模型在许多高资源语言上表现出色，但在低资源语言上（即那些缺乏可用培训数据的语言）的转录和翻译准确性较低。对于某些语言的不同口音和方言，包括不同性别、种族、年龄或其他人口统计标准的说话者，性能也会有所不同（参见Whisper论文）。

为了提高低资源语言、口音或方言的性能，我们可以取预训练的Whisper模型，并在一个小型的适当选择的数据语料库上对其进行训练，这个过程称为微调。我们将展示，仅使用额外的十小时数据，我们就可以将Whisper模型的性能提高100%以上，达到低资源语言的性能。在下一节中，我们将介绍选择微调数据集的过程。

## 选择数据集
 
与任何机器学习问题一样，我们的模型的好坏取决于我们训练的数据。语音识别数据集在它们的策划方式和覆盖的领域上差异很大。为了选择正确的数据集，我们需要将我们的标准与数据集提供的特性进行匹配。

在选择数据集之前，我们首先需要了解关键的定义特性。

### 语音数据集的特性

#### 训练小时数

简而言之，训练小时数表示数据集的大小。这类似于NLP数据集中的训练示例数量。然而，并不是越大的数据集就越好。如果我们想要一个泛化能力良好的模型，我们需要一个具有许多不同说话者、领域和说话风格的多样化数据集。

#### 领域

领域涉及数据的来源，无论是有声书、播客、YouTube还是金融会议。每个领域的数据分布都不同。例如，有声书是在高质量的工作室条件下录制的（没有背景噪音），文本取自书面文学。而对于YouTube来说，音频可能包含更多的背景噪音和更不正式的说话风格。

我们需要将我们的领域与我们预期在推断时的条件相匹配。例如，如果我们在有声书上训练我们的模型，我们不能指望它在嘈杂的环境中表现良好。

#### 说话风格

说话风格分为两类：
* 叙述(Narrated)：根据脚本朗读
* 即兴(Spontaneous)：非脚本化，对话式的言语

音频和文本数据反映了说话的风格。由于叙述文本是脚本化的，因此它倾向于清晰地讲话，没有任何错误：

```
“Consider the task of training a model on a speech recognition dataset”
```
 
而对于即兴言语，我们可以期望一种更口语化的说话风格，包括重复、犹豫和错误开始：

```
“Let’s uhh let's take a look at how you'd go about training a model on uhm a sp- speech recognition dataset”
```


#### 转录风格

转录风格指的是目标文本是否有标点、大小写或两者都有。如果我们希望系统生成完全格式化的文本，可以用于出版物或会议记录，我们需要带有标点和大小写的训练数据。如果我们只需要口头单词而不需要格式化的结构，那么既不需要标点也不需要大小写。在这种情况下，我们可以选择一个没有标点或大小写的数据集，或者选择一个具有标点和大小写的数据集，然后通过预处理从目标文本中删除它们。

### Hub上数据集的摘要

以下是Hugging Face Hub上最受欢迎的英语语音识别数据集的摘要：

| Dataset                                                                                 | Train Hours | Domain                      | Speaking Style        | Casing | Punctuation | License         | Recommended Use                  |
|-----------------------------------------------------------------------------------------|-------------|-----------------------------|-----------------------|--------|-------------|-----------------|----------------------------------|
| [LibriSpeech](https://huggingface.co/datasets/librispeech_asr)                          | 960         | Audiobook                   | Narrated              | ❌      | ❌           | CC-BY-4.0       | Academic benchmarks              |
| [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) | 3000        | Wikipedia                   | Narrated              | ✅      | ✅           | CC0-1.0         | Non-native speakers              |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)                         | 540         | European Parliament         | Oratory               | ❌      | ✅           | CC0             | Non-native speakers              |
| [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium)                                | 450         | TED talks                   | Oratory               | ❌      | ❌           | CC-BY-NC-ND 3.0 | Technical topics                 |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)                    | 10000       | Audiobook, podcast, YouTube | Narrated, spontaneous | ❌      | ✅           | apache-2.0      | Robustness over multiple domains |
| [SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech)                         | 5000        | Financial meetings          | Oratory, spontaneous  | ✅      | ✅           | User Agreement  | Fully formatted transcriptions   |
| [Earnings-22](https://huggingface.co/datasets/revdotcom/earnings22)                     | 119         | Financial meetings          | Oratory, spontaneous  | ✅      | ✅           | CC-BY-SA-4.0    | Diversity of accents             |
| [AMI](https://huggingface.co/datasets/edinburghcstr/ami)                                | 100         | Meetings                    | Spontaneous           | ✅      | ✅           | CC-BY-4.0       | Noisy speech conditions          |

这个表格可作为根据您的标准选择数据集的参考。下面是一个多语言语音识别的等效表格。请注意，我们省略了训练小时列，因为每个数据集的语言不同，而是用每个数据集的语言数量来代替：
 

| Dataset                                                                                       | Languages | Domain                                | Speaking Style | Casing | Punctuation | License   | Recommended Usage       |
|-----------------------------------------------------------------------------------------------|-----------|---------------------------------------|----------------|--------|-------------|-----------|-------------------------|
| [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) | 6         | Audiobooks                            | Narrated       | ❌      | ❌           | CC-BY-4.0 | Academic benchmarks     |
| [Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)       | 108       | Wikipedia text & crowd-sourced speech | Narrated       | ✅      | ✅           | CC0-1.0   | Diverse speaker set     |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)                               | 15        | European Parliament recordings        | Spontaneous    | ❌      | ✅           | CC0       | European languages      |
| [FLEURS](https://huggingface.co/datasets/google/fleurs)                                       | 101       | European Parliament recordings        | Spontaneous    | ❌      | ❌           | CC-BY-4.0 | Multilingual evaluation |


关于两个表格中涵盖的音频数据集的详细说明，请参阅博客文章[《音频数据集完整指南》](https://huggingface.co/blog/audio-datasets#a-tour-of-audio-datasets-on-the-hub)。虽然 Hub 上有超过 180 个语音识别数据集，但可能没有一个与您的需求完全匹配。在这种情况下，也可以使用您自己的音频数据来使用 🤗 数据集。要创建自定义音频数据集，请参阅指南[《创建音频数据集》](https://huggingface.co/docs/datasets/audio_dataset)。在创建自定义音频数据集时，请考虑在 Hub 上共享最终数据集，以便社区中的其他人可以从您的努力中受益 - 音频社区是包容性和广泛的，他人会像您一样欣赏您的工作。

好了！现在我们已经了解了选择 ASR 数据集的所有标准，让我们为本教程选择一个。我们知道 Whisper 已经能够很好地转录高资源语言（例如英语和西班牙语）的数据，所以我们将把重点放在低资源的多语言转录上。我们希望保留 Whisper 预测标点和大小写的能力，因此从第二个表格中看来，Common Voice 13 是一个很好的候选数据集！

### Common Voice 13 

Common Voice 13是一个众包数据集，其中讲者用各种语言录制来自维基百科的文本。它是 Mozilla Foundation 发布的 Common Voice 系列数据集的一部分。在撰写本文时，Common Voice 13 是迄今为止最新版本的数据集，拥有最多的语言和每种语言的小时数。

我们可以通过在 Hub 上查看数据集页面来获取 Common Voice 13 数据集的完整语言列表：[mozilla-foundation/common_voice_13_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)。第一次访问此页面时，您将被要求接受使用条款。之后，您将获得对数据集的完全访问权限。

一旦我们提供了身份验证来使用数据集，就会呈现数据集预览。数据集预览向我们展示了每种语言的数据集的前 100 个样本。此外，它已加载了音频样本，可以实时播放。在本单元中，我们将选择Dhivehi（或马尔代夫语），这是一种印欧-雅利安语言，主要在南亚岛国马尔代夫使用。虽然我们在本教程中选择 Dhivehi，但这里介绍的步骤适用于 Common Voice 13 数据集中的 108 种语言中的任何一种，更一般地，适用于 Hub 上的 180 多种音频数据集中的任何一种，因此语言或方言没有限制。

我们可以通过将子集设置为 dv（dv 是 Dhivehi 的语言标识符代码）来选择 Common Voice 13 的 Dhivehi 子集。

<a>![](/img/hfaudio/unit5/3.png)</a>

如果我们点击第一个样本的播放按钮，我们可以听取音频并查看相应的文本。浏览训练集和测试集的样本，以更好地了解我们处理的音频和文本数据。您可以从语调和风格中判断这些录音是来自朗读的语音。您还可能注意到讲者和录音质量的巨大变化，这是众包数据的常见特点。

数据集预览是在承诺使用它们之前体验音频数据集的绝佳方式。您可以选择 Hub 上的任何数据集，浏览样本并听取不同子集和拆分的音频，以判断它是否适合您的需求。一旦选择了数据集，加载数据以开始使用它是非常简单的。

现在，我个人不会说 Dhivehi，我预计大多数读者也不会！为了知道我们的微调模型是否好用，我们需要一种严格的方法来评估它在未见数据上的表现，并测量其转录准确性。我们将在下一节详细介绍这一点！



## ASR的评估指标

如果你熟悉来自自然语言处理的[Levenshtein距离](https://en.wikipedia.org/wiki/Levenshtein_distance)，那么用于评估语音识别系统的指标将会很熟悉！如果你不熟悉，也不要担心，我们将逐步解释，确保你了解不同指标并理解它们的含义。

在评估语音识别系统时，我们将系统的预测与目标文本转录进行比较，并标注出现的任何错误。我们将这些错误分为三类：

* 替换（S）：我们在预测中转录了错误的单词（“sit”而不是“sat”）
* 插入（I）：我们在预测中添加了额外的单词
* 删除（D）：我们在预测中删除了一个单词

这些错误类别对于所有语音识别指标都是相同的。不同的是我们计算这些错误的级别：我们可以在单词级别或字符级别上计算这些错误。

我们将为每个指标定义一个运行示例。在这里，我们有一个基准或参考文本序列：

```python
reference = "the cat sat on the mat"
```

和来自我们正在评估的语音识别系统的预测序列：

```python
prediction = "the cat sit on the"
```


我们可以看到，预测相当接近，但有些单词不太正确。我们将针对参考文本使用三种最流行的语音识别指标评估这个预测，并看看每种指标得到的数字是什么。

### 词错误率(Word Error Rate)


词错误率（WER）指标是语音识别的‘事实上’指标。它计算单词级别上的替换、插入和删除。这意味着错误是在单词之间逐个标注的。以我们的例子为例：

| Reference:  | the | cat | sat     | on  | the | mat |
|-------------|-----|-----|---------|-----|-----|-----|
| Prediction: | the | cat | **sit** | on  | the |     |  |
| Label:      | ✅   | ✅   | S       | ✅   | ✅   | D   |

在这里，我们有：

* 1个替换（“sit”而不是“sat”）
* 0个插入
* 1个删除（缺少“mat”）

这总共给出了2个错误。为了得到我们的错误率，我们将错误数除以参考文本中的单词总数（N），对于这个例子，是6：

$$
\begin{aligned}
WER &= \frac{S + I + D}{N} \\
&= \frac{1 + 0 + 1}{6} \\
&= 0.333
\end{aligned}
$$

好的！所以我们有一个WER为0.333，或者33.3%。请注意，“sit”一词只有一个错误的字符，但整个单词被标记为不正确。这是WER的一个定义特点：拼写错误会受到严重惩罚，无论多么小。

WER的定义是较低为佳：较低的WER意味着我们的预测中有更少的错误，因此一个完美的语音识别系统的WER将为零（没有错误）。

让我们看看如何使用🤗 Evaluate计算WER。我们将需要两个软件包来计算我们的WER指标：🤗 Evaluate用于API接口，JIWER用于进行运算：

```shell
pip install --upgrade evaluate jiwer
```

太棒了！现在我们可以加载WER指标并计算我们示例的数值：

```
from evaluate import load

wer_metric = load("wer")

wer = wer_metric.compute(references=[reference], predictions=[prediction])

print(wer)
```

打印输出：

```
0.3333333333333333
```

0.33，或33.3%，和预期一样！现在我们知道这个WER计算在幕后的运行方式。

现在，这里有件事情很令人困惑……你认为WER的上限是多少？你可能期望它是1或100%对吧？哦不！由于WER是错误与单词数（N）的比率，所以WER没有上限！让我们举个例子，我们预测了10个单词，目标只有2个单词。如果我们的所有预测都是错误的（10个错误），我们将得到一个WER为10 / 2 = 5，或500%！如果你训练了一个ASR系统并且看到一个超过100%的WER，这是需要注意的事情…… 😅

### 词正确率(Word Accuracy)

我们可以将WER反过来，得到一个越高越好的指标。我们可以测量我们系统的词正确率（WAcc）而不是词错误率：


$$
\begin{equation}
WAcc = 1 - WER \nonumber
\end{equation}
$$

WAcc也是在单词级别上测量的，它只是将WER重新制定为一个准确度指标而不是一个错误度量。WAcc在语音文献中极少被引用 - 我们会根据单词错误来考虑我们系统的预测，因此更倾向于与这些错误类型标注相关的错误率指标。

### 字符错误率(Character Error Rate)

我们对整个单词“sit”标记为错误似乎有点不公平，事实上只有一个字母是错误的。这是因为我们在单词级别上评估我们的系统，从而在单词之间逐个标注错误。字符错误率（CER）在字符级别上评估系统。这意味着我们将单词分解为它们的单个字符，并且在字符之间逐个标注错误：


| Reference:  | t   | h   | e   |     | c   | a   | t   |     | s   | a     | t   |     | o   | n   |     | t   | h   | e   |     | m   | a   | t   |
|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Prediction: | t   | h   | e   |     | c   | a   | t   |     | s   | **i** | t   |     | o   | n   |     | t   | h   | e   |     |     |     |     |
| Label:      | ✅   | ✅   | ✅   |     | ✅   | ✅   | ✅   |     | ✅   | S     | ✅   |     | ✅   | ✅   |     | ✅   | ✅   | ✅   |     | D   | D   | D   |

我们现在可以看到，“sit”这个词中的“s”和“t”被标记为正确。只有“i”被标记为替换错误（S）。因此，我们对部分正确的预测进行奖励 🤝

在我们的示例中，我们有1个字符替换，0个插入和3个删除。总共有14个字符。因此，我们的CER是：

$$
\begin{aligned}
CER &= \frac{S + I + D}{N} \\
&= \frac{1 + 0 + 3}{14} \\
&= 0.286
\end{aligned}
$$

对的！我们有一个CER为0.286，或28.6%。请注意，这比我们的WER要低 - 我们对拼写错误的惩罚要少得多。

### 我应该使用哪个指标？

总的来说，WER比CER在评估语音系统时使用更多。这是因为WER要求系统对预测的上下文有更深的理解。在我们的例子中，“sit”是错误的时态。一个理解动词与句子时态关系的系统将预测正确的动词时态“sat”。我们希望鼓励我们的语音系统达到这种理解水平。因此，尽管WER不如CER宽容，但它更有助于我们想要开发的那种可理解的系统。因此，我们通常使用WER，并鼓励您也这样做！但是，在某些情况下，使用WER是不可能的。某些语言，如中文和日文，没有“单词”的概念，因此WER是无意义的。在这种情况下，我们转而使用CER。

在我们的示例中，我们只使用了一个句子来计算WER。在评估真实系统时，我们通常会使用由数千个句子组成的整个测试集。在评估多个句子时，我们会将S、I、D和N汇总到所有句子中，然后根据上述定义的公式计算WER。这会更好地估计未见数据的WER。

### 标准化(Normalisation)

如果我们在具有标点和大小写的数据上训练ASR模型，它将学习在其转录中预测大小写和标点。当我们想要将模型用于实际的语音识别应用，例如会议记录或口述时，这非常棒，因为预测的转录将完全具有标点和大小写格式，这种风格称为正字法(orthographic)。

然而，我们也有将数据集标准化以去除任何大小写和标点的选项。将数据集标准化可以使语音识别任务变得更容易：模型不再需要区分大写和小写字符，也不需要仅通过音频数据就能预测标点（例如分号发出的声音是什么？）。由于这个原因，单词错误率自然会降低（意味着结果会更好）。Whisper论文展示了对转录进行标准化对WER结果的巨大影响（参见Whisper论文的第4.4节）。虽然我们得到了较低的WER，但该模型不一定适用于生产。缺乏大小写和标点使得模型预测的文本难以阅读。就像在前一节的示例中，我们对LibriSpeech数据集中的同一音频样本运行Wav2Vec2和Whisper一样。Wav2Vec2模型既不预测标点也不预测大小写，而Whisper两者都预测。将转录并排比较，我们可以看到Whisper的转录要容易阅读得多：

```
Wav2Vec2:  HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAUS AND ROSE BEEF LOOMING BEFORE US SIMALYIS DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
Whisper:   He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind.
```

Whisper的转录是正字法的，因此可以直接使用 - 它的格式与会议记录或口述脚本所期望的格式相同，具有标点和大小写。相反，如果我们希望将Wav2Vec2用于下游应用，我们将需要使用额外的后处理来恢复标点和大小写。

在标准化和不标准化之间存在一个折中方法：我们可以在正字法转录上训练我们的系统，然后在计算WER之前将预测和目标标准化。这样，我们既训练我们的系统来预测完全格式化的文本，又从标准化转录中获得WER的改进。

Whisper模型发布时带有一个正规化器，可以有效处理大小写、标点和数字格式等的规范化。让我们将正规化器应用于Whisper的转录，以演示我们如何对其进行规范化：太好了！我们可以看到文本已经完全转换为小写，并且所有标点已删除。现在让我们定义参考转录，然后计算参考和预测之间的规范化WER：

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

prediction = " He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind."
normalized_prediction = normalizer(prediction)

normalized_prediction
```

输出：

```
' he tells us that at this festive season of the year with christmas and roast beef looming before us similarly is drawn from eating and its results occur most readily to the mind '
```

太好了！我们可以看到，文本已经完全转换为小写，并且所有标点已被移除。现在让我们定义参考文本，然后计算参考文本与预测之间的归一化WER：0.25% - 这大约是我们预期在LibriSpeech验证集上使用Whisper基础模型得到的结果。正如我们在这里看到的，我们预测了一个正字法的转录，但在计算WER之前，我们从归一化参考和预测中获得了WER提升。

如何归一化转录最终取决于您的需求。我们建议在正字法文本上进行训练，并在归一化文本上进行评估，以兼顾两者的优点。

### 将所有内容综合起来

好了！在本单元中，我们已经讨论了三个主题：预训练模型、数据集选择和评估。让我们在一个端到端的示例中将它们结合起来🚀 我们将通过在Common Voice 13 Dhivehi测试集上评估预训练的Whisper模型来为下一节的微调做准备。我们将使用我们得到的WER数字作为微调运行的基线，或者我们将尝试并超越的目标数字🥊

首先，我们将使用pipeline()类加载预训练的Whisper模型。这个过程现在应该非常熟悉！我们唯一要做的新事情是在GPU上以半精度（float16）加载模型 - 这将加速推理，几乎不会影响WER准确性。

接下来，我们将加载Common Voice 13的Dhivehi测试集。您会记得在上一节中，Common Voice 13是有限制的，这意味着我们必须同意数据集的使用条款才能访问数据集。我们现在可以将我们的Hugging Face账户链接到我们的笔记本，以便我们可以从当前使用的机器上访问数据集。

将笔记本链接到Hub很简单 - 只需要在提示时输入您的Hub身份验证令牌。在[此处](https://huggingface.co/settings/tokens)找到您的Hub身份验证令牌，并在提示时输入：

```python
from huggingface_hub import notebook_login

notebook_login()
```


链接笔记本到我们的Hugging Face账户后，我们可以继续下载Common Voice数据集。这将花费几分钟的时间来下载和预处理，从Hugging Face Hub获取数据并在您的笔记本上自动准备数据：

```python
from datasets import load_dataset

common_voice_test = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)
```


如果在加载数据集时遇到身份验证问题，请确保已经通过以下链接在Hugging Face Hub上接受了数据集的使用条款：https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0

对整个数据集进行评估可以采用与单个示例相同的方式 - 我们只需循环遍历输入音频，而不是仅推断单个样本。为此，我们首先将数据集转换为KeyDataset。这样做的目的是挑选出我们要转发到模型的特定数据集列（在我们的例子中，这是“audio”列），忽略其余的内容（比如我们不想用于推断的目标转录）。然后，我们遍历这些转换后的数据集，将模型输出附加到列表中以保存预测。如果在GPU上以半精度运行，以下代码单元将花费约五分钟，峰值内存为12GB：

```python
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

all_predictions = []

# run streamed inference
for prediction in tqdm(
    pipe(
        KeyDataset(common_voice_test, "audio"),
        max_new_tokens=128,
        generate_kwargs={"task": "transcribe"},
        batch_size=32,
    ),
    total=len(common_voice_test),
):
    all_predictions.append(prediction["text"])
```


**如果在运行上述单元时遇到CUDA内存溢出（OOM）错误，请逐步将batch_size减少2的倍数，直到找到适合您设备的批处理大小。**

最后，我们可以计算WER。让我们首先计算正字法WER，即没有任何后处理的WER：

```python
from evaluate import load

wer_metric = load("wer")

wer_ortho = 100 * wer_metric.compute(
    references=common_voice_test["sentence"], predictions=all_predictions
)
wer_ortho
```

输出：

```
167.29577268612022
```


Okay… 167%实际上意味着我们的模型输出了垃圾 😜 别担心，我们的目标是通过在Dhivehi训练集上微调模型来改进这一点！

接下来，我们将评估归一化WER，即带有归一化后处理的WER。我们必须过滤掉在归一化后会为空的样本，否则我们参考文本中的单词总数（N）将为零，这会在我们的计算中导致除以零的错误：再次看到我们通过归一化参考和预测而获得的WER显著降低：

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

# compute normalised WER
all_predictions_norm = [normalizer(pred) for pred in all_predictions]
all_references_norm = [normalizer(label) for label in common_voice_test["sentence"]]

# filtering step to only evaluate the samples that correspond to non-zero references
all_predictions_norm = [
    all_predictions_norm[i]
    for i in range(len(all_predictions_norm))
    if len(all_references_norm[i]) > 0
]
all_references_norm = [
    all_references_norm[i]
    for i in range(len(all_references_norm))
    if len(all_references_norm[i]) > 0
]

wer = 100 * wer_metric.compute(
    references=all_references_norm, predictions=all_predictions_norm
)

wer
```

输出：

```
125.69809089960707
```

基线模型在正字法测试WER为168%，而归一化WER为126%。

好了！这些数字是我们在微调模型时要努力超越的数字，以改进Dhivehi语音识别的Whisper模型。继续阅读，以获得微调示例的操作体验 🚀



## 微调语音识别模型

在本节中，我们将介绍如何逐步指导 Whisper 对 Common Voice 13 数据集上的语音识别进行微调。我们将使用模型的“小”版本和相对轻量级的数据集，使您能够在任何 16GB+ GPU 上快速运行微调，而且磁盘空间要求较低，例如 Google Colab 免费版提供的 16GB T4 GPU。

如果您有较小的 GPU 或在训练过程中遇到内存问题，您可以按照提供的减少内存使用建议进行操作。相反，如果您可以访问更大的 GPU，则可以修改训练参数以最大化吞吐量。因此，无论您的 GPU 规格如何，本指南都是可访问的！

同样，本指南概述了如何为 Dhivehi 语言微调 Whisper 模型。然而，这里介绍的步骤可以概括应用于 Common Voice 数据集中的任何语言，并且更普遍地适用于 Hugging Face Hub 上的任何 ASR 数据集。您可以调整代码以快速切换到您选择的语言，并为您的母语微调 Whisper 模型 🌍

好了！既然已经介绍完了，让我们开始并启动我们的微调流程吧！

### 准备环境

我们强烈建议您在训练时直接上传模型检查点到 Hugging Face Hub。Hub 提供了：

* 集成版本控制：您可以确保在训练过程中不会丢失任何模型检查点。
* Tensorboard 日志：跟踪训练过程中的重要指标。
* 模型卡片：记录模型的功能和预期用例。
* 社区：与社区分享和合作的简单方式！ 🤗

将笔记本链接到 Hub 非常简单 - 只需在提示时输入您的 Hub 认证令牌即可。在此处找到您的 Hub 认证令牌，并在提示时输入：

```python
from huggingface_hub import notebook_login

notebook_login()
```

输出：
```
Login successful
Your token has been saved to /root/.huggingface/token
```


### 载入数据集
[Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) 包含大约十小时的标记 Dhivehi 数据，其中三小时是保留的测试数据。这对于微调来说是极少量的数据，因此我们将依靠 Whisper 在预训练过程中获得的广泛多语言 ASR 知识来处理低资源的 Dhivehi 语言。

使用 🤗 Datasets，下载和准备数据非常简单。我们可以仅用一行代码下载和准备 Common Voice 13 的拆分数据。由于 Dhivehi 资源非常有限，我们将合并训练和验证拆分，以获得大约七小时的训练数据。我们将使用三小时的测试数据作为我们的保留测试集：

```python
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)

print(common_voice)
```

输出：

```python
DatasetDict({
    train: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 4904
    })
    test: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 2212
    })
})
```


您可以将语言标识符从 "dv" 更改为您选择的语言标识符。要查看 Common Voice 13 中所有可能的语言，请查看 Hugging Face Hub 上的数据集卡片：https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0

大多数 ASR 数据集仅提供输入音频样本（音频）和相应的转录文本（句子）。Common Voice 包含其他元数据信息，例如口音和区域设置，我们可以在 ASR 中忽略这些信息。为了使笔记本尽可能通用，我们仅考虑输入音频和转录文本进行微调，丢弃其他附加的元数据信息：

```python
common_voice = common_voice.select_columns(["audio", "sentence"])
```

### 特征提取器、分词器和处理器

ASR 管道可以分解为三个阶段：

* 特征提取器(Feature Extractor)，将原始音频输入预处理为对数梅尔频谱图
* 执行序列到序列映射的模型
* 分词器(Tokenizer)，将预测的token后处理为文本

在 🤗 Transformers 中，Whisper 模型有一个关联的特征提取器和分词器，分别称为 [WhisperFeatureExtractor](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperFeatureExtractor) 和 [WhisperTokenizer](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperTokenizer)。为了简化我们的生活，这两个对象被封装在一个称为 [WhisperProcessor](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperProcessor) 的单一类下。我们可以调用 WhisperProcessor 来执行音频预处理和文本标记后处理。这样做，我们在训练过程中只需要跟踪两个对象：处理器和模型。

在执行多语言微调时，我们需要在实例化处理器时设置 "language" 和 "task"。"language" 应设置为源音频语言，"task" 应设置为 "transcribe" 以进行语音识别或 "translate" 以进行语音翻译。这些参数修改了分词器的行为，并应正确设置以确保目标标签正确编码。

我们可以通过导入语言列表来查看 Whisper 支持的所有可能语言：

```
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

TO_LANGUAGE_CODE
```


如果您浏览此列表，您将注意到许多语言都存在，但 Dhivehi 是少数不包括的语言之一！这意味着 Whisper 没有在 Dhivehi 上进行预训练。但是，这并不意味着我们不能在其上进行微调。这样做，我们将教 Whisper 一种新的语言，一个预训练检查点不支持的语言。这相当酷，对吧！

当您在新语言上进行微调时，Whisper 在利用其对其他 96 种语言的知识时表现良好。总的来说，所有现代语言都将在语言上类似于 Whisper 已经了解的 96 种语言之一，因此我们将落入这种跨语言知识表示的范式之下。

要微调 Whisper 在新语言上，我们需要找到与 Whisper 预训练语言最相似的语言。Dhivehi 的维基百科文章表明，Dhivehi 与斯里兰卡的僧伽罗语密切相关。如果我们再次检查语言代码，我们可以看到僧伽罗语存在于 Whisper 语言集中，因此我们可以安全地将我们的语言参数设置为 "sinhalese"。

好的！我们将从预训练检查点加载我们的处理器，将语言设置为 "sinhalese"，将任务设置为 "transcribe"，如上所述：

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
"openai/whisper-small", language="sinhalese", task="transcribe"
)
```

值得重申的是，在大多数情况下，您会发现要微调的语言在预训练语言集中，因此您可以直接将语言设置为源音频语言！请注意，在仅英语微调时，这两个参数都应省略，因为语言（“英语”）和任务（“transcribe”）只有一个选项。

### 预处理数据

让我们来看看数据集的特征。特别注意 "audio" 列 - 这详细说明了我们音频输入的采样率：

```python
common_voice["train"].features
```

输出：

```python
{'audio': Audio(sampling_rate=48000, mono=True, decode=True, id=None),
 'sentence': Value(dtype='string', id=None)}
```


由于我们的输入音频采样率为 48kHz，我们需要在将其传递给 Whisper 特征提取器之前将其降采样为 16kHz，16kHz 是 Whisper 模型期望的采样率。

我们将使用数据集的 [cast_column](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.cast_column) 方法将音频输入设置为正确的采样率。该操作不会在原地更改音频，而是向数据集发出信号，以便在加载时动态重新采样音频样本：

```python
from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))
```


现在，我们可以编写一个函数来准备我们的数据，使其准备好供模型使用：

* 我们通过调用 sample["audio"] 逐个样本加载和重新采样音频数据。如上所述，🤗 Datasets 在加载时执行任何必要的重新采样操作。
* 我们使用特征提取器从我们的 1 维音频数组中计算对数梅尔频谱图输入特征。
* 我们通过分词器将转录编码为标签 id。

```python
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example
```


我们可以使用 🤗 Datasets 的 .map 方法将数据准备函数应用于所有的训练样本。我们将从原始训练数据中删除列（音频和文本），只留下 prepare_dataset 函数返回的列：

```python
common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)
```


最后，我们过滤任何音频样本长于 30 秒的训练数据。否则，这些样本将由 Whisper 特征提取器截断，可能影响训练的稳定性。我们定义一个函数，对于少于 30 秒的样本返回 True，对于长于 30 秒的样本返回 False：

```python
max_input_length = 30.0


def is_audio_in_length_range(length):
    return length < max_input_length
```



我们通过 🤗 Datasets 的 .filter 方法将我们的过滤函数应用于训练数据集的所有样本：

```python
common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)
```



让我们检查一下通过这个过滤步骤我们删除了多少训练数据：

```python
common_voice["train"]
```

输出：

```python
Dataset({
    features: ['input_features', 'labels', 'input_length'],
    num_rows: 4904
})
```


好的！在这种情况下，实际上我们删除的样本数量与之前相同，因此没有样本长于 30 秒。如果您切换语言，情况可能不同，因此最好保持这个过滤步骤以确保稳健性。有了这些，我们的数据已经完全准备好进行训练了！让我们继续看看如何使用这些数据来对 Whisper 进行微调。


### 训练与评估

现在我们已经准备好数据，可以开始深入了解训练流程了。🤗 [Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer) 将为我们承担大部分繁重的工作。我们所需要做的就是：

* 定义数据整合器(data collator)：数据整合器接收我们预处理的数据，并准备好用于模型的 PyTorch 张量。
* 评估指标：在评估过程中，我们希望使用单词错误率（WER）指标评估模型。我们需要定义一个 compute_metrics 函数来处理此计算。
* 加载预训练检查点：我们需要加载预训练检查点，并正确配置它进行训练。
* 定义训练参数：这些参数将由 🤗 Trainer 在构建训练计划时使用。

一旦我们对模型进行了微调，就会在测试数据上对其进行评估，以验证我们是否正确训练了它来转录 Dhivehi 语音。

#### 定义数据整合器

序列到序列语音模型的数据整合器在处理输入特征和标签时是独特的：输入特征必须由特征提取器处理，而标签必须由分词器处理。

输入特征已经填充到 30 秒并转换为固定维度的对数梅尔频谱图，因此我们所要做的就是将它们转换为批处理的 PyTorch 张量。我们使用特征提取器的 .pad 方法并将 return_tensors=pt。请注意，这里不会应用额外的填充，因为输入是固定维度的，输入特征只是简单地转换为 PyTorch 张量。

另一方面，标签未填充。我们首先使用分词器的 .pad 方法将序列填充到批处理中的最大长度。然后，将填充标记替换为 -100，以便在计算损失时不考虑这些标记。然后，我们从标签序列的开头剪切转录标记，因为我们在训练期间稍后会附加它。

我们可以利用之前定义的 WhisperProcessor 执行特征提取器和分词器操作：

```python
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
```


现在我们可以初始化我们刚刚定义的数据整合器：

```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

继续前进吧！

#### 评估指标

接下来，我们定义我们将在评估集上使用的评估指标。我们将使用在评估部分介绍的单词错误率（WER）指标，这是评估 ASR 系统的‘事实’指标。

我们将从 🤗 Evaluate 中加载 WER 指标：

```python
import evaluate

metric = evaluate.load("wer")
```

然后，我们只需要定义一个函数，该函数接收我们的模型预测并返回 WER 指标。这个函数叫做 compute_metrics，首先将标签 id 中的 -100 替换为 pad_token_id（撤消我们在数据整合器中应用的步骤，以正确忽略损失中的填充标记）。然后，将预测的 id 和标签 id 解码为字符串。最后，计算预测和参考标签之间的 WER。在这里，我们可以选择使用‘规范化’的转录和预测进行评估，这些转录和预测已经删除了标点符号和大小写。我们建议您遵循此步骤，以获得通过规范化转录而获得的 WER 改进。

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}
```



#### 加载预训练检查点

现在让我们加载预训练的 Whisper small 检查点。同样，通过使用 🤗 Transformers，这很容易实现！

```
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

 
我们将在训练中将 use_cache 设置为 False，因为我们使用了[梯度检查点](https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing)和这两者不兼容。我们还将覆盖两个生成参数以控制推理期间模型的行为：我们将通过设置 language 和 task 参数来强制生成期间的语言和任务标记，并重新启用推理期间的缓存以加快推理速度：

```python
from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="sinhalese", task="transcribe", use_cache=True
)
```


### 定义训练配置

在最后一步中，我们定义与训练相关的所有参数。在这里，我们将训练步数设置为 500 步。这足以看到与预训练的 Whisper 模型相比的 WER 明显改进，同时确保微调可以在 Google Colab 免费版中运行约 45 分钟。有关训练参数的更多详细信息，请参阅 Seq2SeqTrainingArguments [文档](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments)。


```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-dv",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
```


如果您不想将模型检查点上传到 Hub，请设置 push_to_hub=False。

我们可以将训练参数与我们的模型、数据集、数据整合器和 compute_metrics 函数一起传递给 🤗 Trainer：

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
```


准备好了，我们已经准备好开始训练了！

### 训练

要启动训练，只需执行：

```python
trainer.train()
```


训练大约需要 45 分钟，具体取决于您的 GPU 或分配给 Google Colab 的 GPU。根据您的 GPU，当您开始训练时，可能会遇到 CUDA "out-of-memory" 错误。在这种情况下，您可以将每个设备训练批次大小逐步减小 2 倍，并使用[梯度累积步骤](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.gradient_accumulation_steps)来补偿。

**输出:**

| Training Loss | Epoch | Step | Validation Loss | Wer Ortho | Wer     |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:-------:|
| 0.136         | 1.63  | 500  | 0.1727          | 63.8972   | 14.0661 |

我们最终的 WER 是 14.1% - 对于七小时的训练数据和只有 500 步训练的情况来说，这并不差！与预训练模型相比，这相当于 112% 的改进！这意味着我们已经拿到了一个以前对 Dhivehi 一无所知的模型，并在不到一小时内对其进行了微调，使其能够以足够的准确率识别 Dhivehi 语音 🤯

最重要的问题是，这与其他 ASR 系统相比如何。为此，我们可以查看 autoevaluate [排行榜](https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=mozilla-foundation%2Fcommon_voice_13_0&only_verified=0&task=automatic-speech-recognition&config=dv&split=test&metric=wer)，这是一个按语言和数据集对模型进行分类，并根据它们的 WER 进行排名的排行榜。

在查看排行榜时，我们看到我们为 500 步训练的模型明显击败了我们在前面的部分中评估的预训练 Whisper Small [检查点](https://huggingface.co/openai/whisper-small)。干得好 👏

我们看到有几个检查点比我们训练的检查点做得更好。Hugging Face Hub 的美妙之处在于它是一个协作平台 - 如果我们没有时间或资源来执行更长时间的训练运行，我们可以加载社区中其他人已经训练并且慷慨地分享的检查点（确保感谢他们！）。您可以使用与之前加载预训练检查点相同的方式加载这些检查点，使用 pipeline 类！所以没有什么能阻止您从排行榜中挑选出最佳模型来用于您的任务！

当我们将训练结果推送到 Hub 时，我们可以自动将我们的检查点提交到排行榜 - 我们只需设置相应的关键字参数（kwargs）。您可以更改这些值以匹配您的数据集、语言和模型名称：

```python
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_13_0",
    "dataset": "Common Voice 13",  # a 'pretty' name for the training dataset
    "language": "dv",
    "model_name": "Whisper Small Dv - Sanchit Gandhi",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}
```

现在可以将训练结果上传到 Hub。要执行此操作，请执行 push_to_hub 命令：


```python
trainer.push_to_hub(**kwargs)
```

这将在 "your-username/the-name-you-picked" 下保存训练日志和模型权重。对于此示例，请查看 sanchit-gandhi/whisper-small-dv 中的上传。

尽管在 Common Voice 13 Dhivehi 测试数据上，微调模型产生了令人满意的结果，但它并不是最佳的。本指南的目的是演示如何使用 🤗 Trainer 对多语言语音识别进行微调的 ASR 模型。

如果您有自己的 GPU，或者订阅了 Google Colab 的付费计划，您可以将 max_steps 增加到 4000 步，通过训练更多步数来进一步改进 WER。训练 4000 步将需要大约 3-5 小时，具体取决于您的 GPU，并且产生的 WER 结果大约比训练 500 步低 3%。如果您决定训练 4000 步，我们还建议将学习率调度器更改为线性调度器（设置 lr_scheduler_type="linear"），因为这将在长时间的训练运行中产生额外的性能提升。

通过优化训练超参数（例如学习率和丢失）并使用更大的预训练检查点（medium 或 large），结果可能进一步改进。我们将这留给读者作为练习。

### 分享您的模型

现在，您可以通过 Hub 上的链接与任何人分享此模型。他们可以直接将其加载到 pipeline() 对象中，标识符为 "your-username/the-name-you-picked"。例如，要加载微调检查点 [“sanchit-gandhi/whisper-small-dv”](https://huggingface.co/sanchit-gandhi/whisper-small-dv)：


```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="sanchit-gandhi/whisper-small-dv")
```

### 结论

在本节中，我们详细介绍了使用 🤗 Datasets、Transformers 和 Hugging Face Hub 对 Whisper 模型进行语音识别的微调的逐步指南。我们首先加载了 Common Voice 13 数据集的 Dhivehi 子集，并通过计算对数梅尔频谱图和标记化文本来进行了预处理。然后，我们定义了数据整合器、评估指标和训练参数，然后使用 🤗 Trainer 对我们的模型进行了训练和评估。最后，我们将微调后的模型上传到了 Hugging Face Hub，并展示了如何使用 pipeline() 类分享和使用它。

如果您一直跟着做到了这一点，那么现在您应该有一个用于语音识别的微调检查点，做得好！🥳 更重要的是，您已经具备了在任何语音识别数据集或领域上微调 Whisper 模型所需的所有工具。那么您还在等什么呢！从本节中选择的数据集中选择一个，或选择您自己的数据集，看看您是否能够获得最先进的性能！排行榜正在等待您……


## 使用 Gradio 构建演示

现在我们已经为 Dhivehi 语音识别微调了 Whisper 模型，让我们继续构建一个 Gradio 演示，向社区展示它！

首先要做的是使用 pipeline() 类加载微调后的检查点 - 这在预训练模型部分已经非常熟悉了。您可以将 model_id 更改为您在 Hugging Face Hub 上微调模型的命名空间，或者使用预训练的 Whisper 模型进行零-shot 语音识别：

```python
from transformers import pipeline

model_id = "sanchit-gandhi/whisper-small-dv"  # update with your model id
pipe = pipeline("automatic-speech-recognition", model=model_id)
```


其次，我们将定义一个函数，该函数接收音频输入的文件路径，并通过 pipeline 进行处理。在这里，pipeline 会自动处理加载音频文件、将其重采样到正确的采样率，并使用模型进行推理。然后，我们可以简单地将转录文本作为函数的输出。为了确保我们的模型能够处理任意长度的音频输入，我们将启用分块，如预训练模型部分所述：

```python
def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "sinhalese",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]
```


我们将使用 Gradio 的块功能在我们的演示中启动两个选项卡：一个用于麦克风转录，另一个用于文件上传。

```python
import gradio as gr

demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.outputs.Textbox(),
)

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.outputs.Textbox(),
)
```


最后，我们使用刚刚定义的两个块启动 Gradio 演示：

```python
with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(debug=True)
```


这将启动一个类似于在 Hugging Face Space 上运行的演示：

<a>![](/img/hfaudio/unit5/4.png)</a>

如果您希望将您的演示托管在 Hugging Face Hub 上，您可以使用此 Space 作为您微调的模型的模板。

单击链接模板演示到您的账户：https://huggingface.co/spaces/course-demos/whisper-small?duplicate=true

我们建议给您的 Space 一个类似于您微调模型的名称（例如 whisper-small-dv-demo），并将可见性设置为“公共”。

一旦您将 Space 到您的账户中，单击“文件和版本” -> “app.py” -> “编辑”。然后将模型标识符更改为您微调的模型（第 6 行）。滚动到页面底部，单击“提交更改到主分支”。演示将重新启动，这次使用您微调的模型。您可以与您的朋友和家人分享这个演示，让他们可以使用您训练过的模型！

查看我们的视频教程，以更好地理解如何 Space 👉️ [YouTube 视频](https://www.youtube.com/watch?v=VQYuvl6-9VE)

我们期待在 Hub 上看到您的演示！


