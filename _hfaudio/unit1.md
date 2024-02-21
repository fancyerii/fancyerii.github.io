---
layout:     post
title:      "1. working with audio data"
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

## 在这个单元中学习内容

每个音频或语音任务都始于一个音频文件。在我们深入解决这些任务之前，了解这些文件实际包含什么，以及如何处理它们是非常重要的。

在本单元中，你将理解与音频数据相关的基本术语，包括波形、采样率和频谱图。你还将学习如何处理音频数据集，包括加载和预处理音频数据，以及如何高效地流式处理大型数据集。

通过本单元的学习，你将对基本音频数据术语有深入了解，并掌握处理音频数据集所需的技能，以应用于各种应用场景。本单元中所获得的知识将为你理解本课程的其余部分奠定基础。

## 音频数据简介
从本质上讲，声波是一个连续信号，意味着在给定的时间内包含无限数量的信号值。这给数字设备带来了问题，因为它们期望有限的数组。为了被数字设备处理、存储和传输，连续的声波需要被转换为一系列离散值，称为数字表示。

如果你查看任何音频数据集，你会发现包含声音摘录的数字文件，比如文本叙述或音乐。你可能会遇到不同的文件格式，比如.wav（波形音频文件）、.flac（无损音频编解码器）和.mp3（MPEG-1音频层3）。这些格式主要在它们如何压缩音频信号的数字表示方面有所不同。

让我们看看我们是如何从连续信号到达这个表示的。模拟信号首先被麦克风捕捉到，将声波转换为电信号。然后电信号通过模数转换器转换为数字表示，通过采样得到数字表示。

### 采样和采样率

采样是在固定时间步长处测量连续信号的值的过程。采样后的波形是离散的，因为它包含在均匀间隔的有限数量的信号值。


<img src="/img/hfaudio/unit1/1.png" alt="drawing" width="80%"/>

<a>![]()</a>
*来自维基百科文章的插图：采样（信号处理）*

采样率（也称为采样频率）是在一秒钟内采集的样本数，以赫兹（Hz）表示。举个参考例子，CD音质的音频采样率为44,100 Hz，意味着每秒采样44,100次。相比之下，高分辨率音频的采样率为192,000 Hz或192 kHz。在训练语音模型中常用的采样率是16,000 Hz或16 kHz。

采样率的选择主要决定了可以从信号中捕获的最高频率。这也被称为奈奎斯特限制，恰好是采样率的一半。人类语音中的可听频率低于8 kHz，因此以16 kHz采样语音是足够的。使用更高的采样率不会捕获更多信息，只会增加处理这种文件的计算成本。另一方面，以太低的采样率对音频进行采样会导致信息丢失。以8 kHz采样的语音会听起来含糊不清，因为无法在此速率下捕获更高的频率。

在处理任何音频任务时，确保数据集中的所有音频示例具有相同的采样率是很重要的。如果你计划使用自定义音频数据来微调预训练模型，你的数据的采样率应与模型预训练数据的采样率相匹配。采样率确定了连续音频样本之间的时间间隔，这影响了音频数据的时间分辨率。考虑一个例子：在采样率为16,000 Hz的情况下，5秒钟的声音将被表示为一系列80,000个值，而在采样率为8,000 Hz的情况下，相同的5秒钟的声音将被表示为一系列40,000个值。解决音频任务的Transformer模型将示例视为序列，并依赖注意机制来学习音频或多模态表示。由于不同采样率下的音频示例的序列不同，模型之间的泛化将是具有挑战性的。重新采样是使采样率匹配的过程，是音频数据的预处理的一部分。

### 振幅和比特深度

采样率告诉您样本取样的频率，每个样本到底是什么？

声音是由在人类可听到的频率下的空气压力变化所产生的。声音的振幅描述了任意时刻的声压级，以分贝（dB）表示。我们对(客观物理量)振幅的感受感受为响度。举个例子，正常说话的声音低于60 dB，摇滚音乐会在125 dB左右，接近人类听觉的极限。

在数字音频中，每个音频样本记录了音频波在某个时间点的振幅。样本的比特深度确定了该振幅值可以被描述的精度。比特深度越高，数字表示越忠实地逼近原始的连续声波。

最常见的音频比特深度是16位和24位。每个都是二进制术语，表示将振幅值量化时可能的步数：16位音频有65,536个可能值(原文为step不好翻译)，24位音频有令人惊讶的16,777,216个可能值。因为量化涉及将连续值四舍五入为离散值，采样过程引入了噪声。比特深度越高，量化噪声就越小。在实践中，16位音频的量化噪声已经足够小到不可听见，一般情况下不需要使用更高的比特深度。

你也可能会遇到32位音频。这将样本存储为浮点值，而16位和24位音频使用整数样本。32位浮点值的精度为24位，具有与24位音频相同的比特深度。浮点音频样本预期在[-1.0，1.0]范围内。由于机器学习模型通常处理浮点数据，因此在使用该数据训练模型之前，音频必须首先转换为浮点格式。我们将在下一节“预处理”中看到如何做到这一点。

与连续音频信号一样，数字音频的振幅通常以分贝（dB）表示。由于人类听觉的本质是对数的——我们对轻微的声音波动更敏感，而对响亮的声音波动不那么敏感——所以如果振幅以分贝表示，声音的响度更容易解释，分贝也是对数的。实际音频的分贝刻度从0 dB开始，代表人类可以听到的最安静的声音，而更大的声音具有更大的值。然而，对于数字音频信号，0 dB是最大的振幅，而所有其他振幅都是负值。一个快速的经验法则：每-6 dB是振幅减半，而低于-60 dB的任何值通常都是听不见的，除非你真的大声放音量。

### 音频作为波形

你可能已经看到声音被可视化为波形，它将样本值随时间绘制出来，说明了声音振幅的变化。这也被称为声音的时间域表示。

这种类型的可视化对于识别音频信号的特定特征非常有用，比如单个声音事件的时间、信号的整体响度以及音频中存在的任何不规则或噪声。

要为音频信号绘制波形，我们可以使用一个名为librosa的Python库：

```shell
pip install librosa
```


```python
import librosa

array, sampling_rate = librosa.load(librosa.ex("trumpet"))
```

<img src="/img/hfaudio/unit1/2.png" alt="drawing" width="80%"/>


这将在y轴上绘制信号的振幅，而在x轴上绘制时间。换句话说，每个点对应于在采样此声音时取得的单个样本值。此外，请注意，librosa已将音频返回为浮点值，并且振幅值确实在[-1.0，1.0]范围内。

将音频可视化并同时听取它可以成为理解正在处理的数据的有用工具。您可以看到信号的形状，观察模式，学会识别噪音或失真。如果以某种方式对数据进行预处理，例如归一化、重新采样或滤波，则可以通过视觉确认已应用预处理步骤是否如预期般有效。在训练模型之后，您还可以可视化发生错误的样本（例如，在音频分类任务中），以便调试问题。

### 频谱(spectrum)
可视化音频数据的另一种方法是绘制音频信号的频谱，也称为频域表示。频谱是使用离散傅里叶变换或DFT计算的。它描述了构成信号的各个频率及其强度。

让我们通过使用numpy的rfft()函数对相同的小号声音进行离散傅里叶变换（DFT）来绘制频谱图。虽然可以绘制整个声音的频谱，但查看一个小区域会更有用。在这里，我们将对前4096个样本进行DFT，这大致是正在播放的第一个音符的长度。


```python
import numpy as np

dft_input = array[:4096]

# calculate the DFT
window = np.hanning(len(dft_input))
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)

# get the amplitude spectrum in decibels
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# get the frequency bins
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

plt.figure().set_figwidth(12)
plt.plot(frequency, amplitude_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.xscale("log")
```

<img src="/img/hfaudio/unit1/3.png" alt="drawing" width="80%"/>

这里绘制了在该音频片段中存在的各种频率成分的强度。频率值位于 x 轴上，通常以对数尺度绘制，而它们的振幅位于 y 轴上。

我们绘制的频谱显示了几个峰值。这些峰值对应于正在播放的音符的谐波，较高的谐波较安静。由于第一个峰值约在 620 Hz 处，这是 E♭ 音符的频谱。

DFT 的输出是由实部和虚部组成的复数数组。通过使用 np.abs(dft) 取幅度信息，可以从频谱图中提取幅度信息。实部和虚部之间的角度提供了所谓的相位谱，但在机器学习应用中通常会将其丢弃。

您使用了 librosa.amplitude_to_db() 将振幅值转换为分贝标度，使得更容易看到频谱中的细节。有时人们使用功率谱，它测量能量而不是振幅；这只是一个将振幅值平方的频谱。

💡 在实践中，人们将 FFT 与 DFT 交替使用，因为 FFT 或快速傅立叶变换是在计算机上计算 DFT 的唯一有效方法。
音频信号的频率谱包含与其波形相同的信息 —— 它们只是查看相同数据的两种不同方式（这里是从小号声音中提取的前 4096 个样本）。波形绘制了音频信号随时间的振幅，而频谱则可视化了固定时间点上各个频率的振幅。

### 语谱图(Spectrogram)

如果我们想要查看音频信号中的频率如何变化怎么办？小号演奏了几个音符，它们都有不同的频率。问题是频谱图只显示了给定时刻的频率的冻结快照。解决方案是进行多次 DFT，每次只覆盖很小的时间片段，并将得到的频谱堆叠在一起形成语谱图。

语谱图绘制了音频信号的频率内容随时间变化的情况。它使您可以在一个图表上看到时间、频率和振幅。执行此计算的算法是 STFT 或短时傅立叶变换。

语谱图是您可用的最具信息性的音频工具之一。例如，在处理音乐录音时，您可以看到各种乐器和声乐轨道以及它们如何对整体声音做出贡献。在语音中，您可以识别不同的元音音素，因为每个元音都由特定的频率特征化。

让我们为相同的小号声音绘制一个语谱图，使用 librosa 的 stft() 和 specshow() 函数：

```python
import numpy as np

D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()

plt.show()
```

<img src="/img/hfaudio/unit1/4.png" alt="drawing" width="80%"/>

在这个图中，x 轴表示时间，就像波形可视化中一样，但现在 y 轴表示频率，以赫兹（Hz）为单位。颜色的强度给出了每个时间点处频率成分的振幅或功率，以分贝（dB）为单位。

语谱图是通过取音频信号的短片段（通常持续几毫秒）并计算每个片段的离散傅立叶变换以获得其频率谱而创建的。然后将得到的频谱沿时间轴堆叠在一起以创建频谱图。该图像中的每个垂直切片对应于从顶部看到的单个频率谱。默认情况下，librosa.stft() 将音频信号分割为 2048 个样本的段，这在频率分辨率和时间分辨率之间取得了良好的折衷。

由于语谱图和波形是相同数据的不同视图，因此可以使用逆 STFT 将频谱图转换回原始波形。但是，这需要相位信息以及幅度信息。如果频谱图是由机器学习模型生成的，则通常仅输出振幅。在这种情况下，我们可以使用相位重建算法，如经典的 Griffin-Lim 算法，或使用称为声码器的神经网络，从频谱图重建波形。

频谱图不仅用于可视化。许多机器学习模型将频谱图作为输入 —— 而不是波形 —— 并将频谱图作为输出。

既然我们知道了什么是频谱图以及它是如何制作的，让我们来看一种广泛用于语音处理的变体：梅尔频谱图。

### 梅尔语谱图(Mel Spectrogram)

梅尔频谱图是频谱图的一种变体，通常用于语音处理和机器学习任务。它类似于频谱图，因为它显示了音频信号随时间的频率内容，但在不同的频率轴上。

在标准语谱图中，频率轴是线性的，并以赫兹（Hz）为单位测量。然而，人类听觉系统对低频变化的敏感性比对高频变化的敏感性更高，并且随着频率增加，这种敏感性以对数形式递减。梅尔刻度是一种感知刻度，近似于人耳的非线性频率响应。

为了创建梅尔频谱图，STFT 的操作方式与之前相同，将音频分成短片段以获得一系列频率谱。此外，每个频谱通过一组滤波器，即所谓的梅尔滤波器组，以将频率转换为梅尔刻度。

让我们看看如何使用 librosa 的 melspectrogram() 函数绘制梅尔频谱图，该函数为我们执行所有这些步骤：


```python
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()
```

<img src="/img/hfaudio/unit1/5.png" alt="drawing" width="80%"/>

在上面的示例中，n_mels 代表要生成的梅尔频带数。梅尔频带定义了一组频率范围，将频谱分成感知上有意义的组件，使用一组滤波器，其形状和间距被选择来模仿人耳对不同频率的响应方式。常见的 n_mels 值为 40 或 80。fmax 表示我们关心的最高频率（以赫兹为单位）。

与常规频谱图一样，通常习惯用分贝来表示梅尔频率成分的强度。这通常被称为对数梅尔频谱图，因为将其转换为分贝涉及对数运算。上面的示例使用了 librosa.power_to_db()，因为 librosa.feature.melspectrogram() 创建了一个功率谱图。

💡 并非所有的梅尔频谱图都相同！常见使用两种不同的梅尔刻度（"htk" 和 "slaney"），并且可以使用振幅谱图代替功率谱图。将其转换为对数梅尔频谱图并不总是计算真正的分贝，而可能仅仅是取对数。因此，如果一个机器学习模型期望输入梅尔频谱图，请务必确保以相同的方式进行计算。

创建梅尔频谱图是一种有损操作，因为它涉及对信号进行滤波。将梅尔频谱图转换回波形比常规频谱图更困难，因为它需要估计被丢弃的频率。这就是为什么需要像 HiFiGAN 声码器这样的机器学习模型，以从梅尔频谱图生成波形。

与标准频谱图相比，梅尔频谱图可以更好地捕捉到音频信号的有意义特征，使其在诸如语音识别、说话人识别和音乐流派分类等任务中成为一种流行选择。

现在您已经知道如何可视化音频数据示例了，尝试看看您最喜欢的声音是什么样子吧。 :)


## 加载和探索音频数据集

在本课程中，我们将使用 🤗 Datasets来处理音频数据集。 🤗 Datasets是一个开源库，用于下载和准备所有类型数据集，包括音频。该库提供了轻松访问 Hugging Face Hub 上公开可用的无与伦比的机器学习数据集的功能。此外，🤗 Datasets包括多个针对音频数据集定制的功能，简化了研究人员和从业者处理此类数据集的工作。

要开始处理音频数据集，请确保已安装了 🤗 Datasets：

```shell
pip install datasets[audio]
```

🤗 Datasets的一个关键特性是只需一行 Python 代码即可下载和准备数据集，使用 load_dataset() 函数。

让我们加载和探索一个名为 [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) 的音频数据集，其中包含人们用多种语言和方言向电子银行系统提问的录音。

要加载 MINDS-14 数据集，我们需要在 Hub 上复制数据集的标识符（PolyAI/minds14），并将其传递给 load_dataset 函数。我们还将指定我们只对数据的澳大利亚子集（en-AU）感兴趣，并将其限制为训练集分割：

```python
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds
```

输出：
```
Dataset(
    {
        features: [
            "path",
            "audio",
            "transcription",
            "english_transcription",
            "intent_class",
            "lang_id",
        ],
        num_rows: 654,
    }
)
```

该数据集包含 654 个音频文件，每个文件都附带有一份转录(transcription)、英文翻译以及指示该问题背后意图的标签。音频列包含原始音频数据。让我们仔细看看其中一个示例：你可能会注意到音频列包含几个特征。以下是它们的含义：

```python
example = minds[0]
example
```
输出：

```
{
    "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
    "audio": {
        "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
        "array": array(
            [0.0, 0.00024414, -0.00024414, ..., -0.00024414, 0.00024414, 0.0012207],
            dtype=float32,
        ),
        "sampling_rate": 8000,
    },
    "transcription": "I would like to pay my electricity bill using my card can you please assist",
    "english_transcription": "I would like to pay my electricity bill using my card can you please assist",
    "intent_class": 13,
    "lang_id": 2,
}
```

* path：音频文件的路径（在本例中为 *.wav）。
* array：解码后的音频数据，表示为一个一维 NumPy 数组。
* sampling_rate：音频文件的采样率（本示例中为 8,000 Hz）。
* intent_class 是音频录制的分类类别。为了将这个数字转换为有意义的字符串，我们可以使用 int2str() 方法：

```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
```
输出：
```
"pay_bill"
```





如果你看一下转录特征，你会看到音频文件确实记录了一个人询问如何支付账单的问题。

如果你计划在这个数据子集上训练一个音频分类器，你可能并不一定需要所有的特征。例如，lang_id 对于所有示例来说将具有相同的值，不会有用。english_transcription 可能会在这个子集中重复转录，所以我们可以安全地删除它们。

你可以使用 🤗 Datasets的 remove_columns 方法轻松地删除不相关的特征：

```python
columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
minds
```
输出：
```
Dataset({features: ["path", "audio", "transcription", "intent_class"], num_rows: 654})
```



现在我们已经加载并检查了数据集的原始内容，让我们听几个例子！我们将使用 Gradio 的 Blocks 和 Audio 特性来解码数据集中的几个随机样本：

```python
import gradio as gr


def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label(example["intent_class"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)
```

如果你愿意，你也可以可视化一些例子。让我们为第一个示例绘制波形图。

```python
import librosa
import matplotlib.pyplot as plt
import librosa.display

array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)
```

<img src="/img/hfaudio/unit1/6.png" alt="drawing" width="80%"/>

试试看！下载 MINDS-14 数据集的另一个方言或语言，听一听并可视化一些例子，以了解整个数据集的变化。你可以在[这里](https://huggingface.co/datasets/PolyAI/minds14)找到可用语言的完整列表。


## 音频数据集的预处理

使用 🤗 Datasets加载数据只是一半的乐趣。如果你计划将其用于训练模型或运行推断，你需要先对数据进行预处理。一般来说，这将涉及以下步骤：

* 重新采样音频数据
* 过滤数据集
* 将音频数据转换为模型预期的输入格式

### 重新采样音频数据

load_dataset 函数下载音频示例时使用它们发布时的采样率。这并不总是你计划训练或使用推断的模型期望的采样率。如果采样率之间有差异，你可以将音频重新采样为模型期望的采样率。

大多数可用的预训练模型在 16 kHz 的采样率下进行了预训练。当我们探索 MINDS-14 数据集时，你可能已经注意到它的采样率为 8 kHz，这意味着我们可能需要对其进行上采样。

要做到这一点，可以使用 🤗 Datasets的 cast_column 方法。该操作不会直接更改音频，而是向数据集发出信号，以在加载时动态重新采样音频示例。以下代码将将采样率设置为 16kHz：

```python
from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

重新加载 MINDS-14 数据集中的第一个音频示例，并检查它是否已被重新采样到所需的采样率：

```python
minds[0]
```

输出：

```
{
    "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
    "audio": {
        "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
        "array": array(
            [
                2.0634243e-05,
                1.9437837e-04,
                2.2419340e-04,
                ...,
                9.3852862e-04,
                1.1302452e-03,
                7.1531429e-04,
            ],
            dtype=float32,
        ),
        "sampling_rate": 16000,
    },
    "transcription": "I would like to pay my electricity bill using my card can you please assist",
    "intent_class": 13,
}
```

你可能会注意到数组值现在也不同了。这是因为现在每个振幅值的数量已经是之前的两倍了。

💡 关于重新采样的一些背景知识：如果音频信号以 8 kHz 的采样率进行采样，即每秒有 8000 个采样读数，我们知道该音频不包含任何频率超过 4 kHz 的内容。这是由奈奎斯特采样定理保证的。由于这个原因，在采样点之间，原始连续信号总是形成一个平滑曲线。上采样到更高的采样率实际上就是计算出这条曲线之间的额外的样本值，通过近似这条曲线。然而，下采样需要首先过滤掉任何高于新奈奎斯特限制的频率，然后再估计新的采样点。换句话说，你不能简单地通过删除每个其他样本来将采样率下降 2 倍 —— 这会在信号中创建称为混叠的失真。正确地进行重新采样是棘手的，最好使用经过充分测试的库，如 librosa 或 🤗 Datasets。


### 过滤数据集

你可能需要根据某些标准来过滤数据。其中一个常见情况涉及将音频示例限制在某个持续时间范围内。例如，我们可能希望过滤掉任何超过 20 秒的示例，以防在训练模型时发生内存溢出错误。

我们可以使用 🤗 Datasets的 filter 方法，并将带有过滤逻辑的函数传递给它。让我们首先编写一个指示哪些示例保留哪些丢弃的函数。这个函数 is_audio_length_in_range 如果样本小于 20 秒则返回 True，如果大于 20 秒则返回 False。

```python
MAX_DURATION_IN_SECONDS = 20.0


def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS
```

过滤函数可以应用于数据集的列，但在这个数据集中我们没有一个音轨持续时间的列。但是，我们可以创建一个，基于该列中的值进行过滤，然后将其删除。


```python
# use librosa to get example's duration from the audio file
new_column = [librosa.get_duration(path=x) for x in minds["path"]]
minds = minds.add_column("duration", new_column)

# use 🤗 Datasets' `filter` method to apply the filtering function
minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

# remove the temporary helper column
minds = minds.remove_columns(["duration"])
minds
```

我们可以验证数据集已经从 654 个示例过滤到 624 个。

```python
Dataset({features: ["path", "audio", "transcription", "intent_class"], num_rows: 624})
```

### 音频数据预处理

处理音频数据集最具挑战性的一个方面是将数据转换为模型训练所需的正确格式。正如你所见，原始音频数据是一系列样本值的数组。然而，预训练模型，无论是用于推断还是用于微调以适应你的任务，都期望将原始数据转换为输入特征。输入特征的要求可能因模型的体系结构和预训练数据而异。好消息是，对于每个支持的音频模型，🤗 Transformers 都提供了一个特征提取器类，该类可以将原始音频数据转换为模型期望的输入特征。

那么特征提取器如何处理原始音频数据呢？让我们来看看 [Whisper](https://huggingface.co/papers/2212.04356) 的特征提取器，以了解一些常见的特征提取转换。Whisper 是由 OpenAI 的 Alec Radford 等人于 2022 年 9 月发布的自动语音识别（ASR）的预训练模型。

首先，Whisper 特征提取器对一批音频示例进行填充/截断，使所有示例的输入长度为 30 秒。比这更短的示例将通过在序列末尾附加零来填充到 30 秒（音频信号中的零表示没有信号或静音）。超过 30 秒的示例将被截断为 30 秒。由于批次中的所有元素都被填充/截断到输入空间中的最大长度，因此不需要注意力掩码。在这方面，Whisper 是独一无二的，大多数其他音频模型都需要一个注意力掩码来详细说明哪些序列被填充，因此在自注意机制中应该忽略哪些部分。Whisper 被训练为在没有注意力掩码的情况下运行，并直接从语音信号中推断出在哪里忽略输入。

Whisper 特征提取器执行的第二个操作是将填充的音频数组转换为对数梅尔频谱图。你会记得，这些频谱图描述了信号的频率随时间变化的方式，以梅尔标度表示，并以分贝（对数部分）来测量，以使频率和振幅更具代表性。

所有这些转换都可以用几行代码应用到你的原始音频数据上。让我们继续从预训练的 Whisper 检查点加载特征提取器，以准备好我们的音频数据：

```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
```


接下来，你可以编写一个函数，通过特征提取器对单个音频示例进行预处理。

```python
def prepare_dataset(example):
    audio = example["audio"]
    features = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], padding=True
    )
    return features
```


我们可以使用 🤗 Datasets的 map 方法将数据准备函数应用于所有训练示例：

```python
minds = minds.map(prepare_dataset)
minds
```

输出：
```
Dataset(
    {
        features: ["path", "audio", "transcription", "intent_class", "input_features"],
        num_rows: 624,
    }
)
```

就是这么简单，现在我们的数据集中有了对数梅尔频谱图作为输入特征。

让我们对 minds 数据集中的一个示例进行可视化：


```python
import numpy as np

example = minds[0]
input_features = example["input_features"]

plt.figure().set_figwidth(12)
librosa.display.specshow(
    np.asarray(input_features[0]),
    x_axis="time",
    y_axis="mel",
    sr=feature_extractor.sampling_rate,
    hop_length=feature_extractor.hop_length,
)
plt.colorbar()
```

<img src="/img/hfaudio/unit1/7.png" alt="drawing" width="80%"/>


现在你可以看到经过预处理后传递给 Whisper 模型的音频输入是什么样子的了。

模型的特征提取器类负责将原始音频数据转换为模型期望的格式。然而，许多涉及音频的任务都是多模态的，例如语音识别。在这种情况下，🤗 Transformers 还提供了用于处理文本输入的特定于模型的分词器。有关分词器的深入了解，请参阅我们的[自然语言处理课程](https://huggingface.co/course/chapter2/4)。

你可以单独加载 Whisper 和其他多模态模型的特征提取器和分词器，也可以通过所谓的处理器同时加载两者。为了使事情变得更简单，可以使用 AutoProcessor 从检查点加载模型的特征提取器和处理器，就像这样：

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small")
```

这里我们已经说明了基本的数据准备步骤。当然，自定义数据可能需要更复杂的预处理。在这种情况下，你可以扩展 prepare_dataset 函数来执行任何类型的自定义数据转换。使用 🤗 Datasets，如果你可以将其写成 Python 函数，你就可以将其[应用于](https://huggingface.co/docs/datasets/audio_process)你的数据集！



## 流式音频数据

处理音频数据集面临的最大挑战之一是它们的庞大大小。一分钟未压缩的 CD 音质音频（44.1kHz，16位）占用超过 5 MB 的存储空间。通常，一个音频数据集会包含几个小时的录音。

在前面的章节中，我们使用了 MINDS-14 音频数据集的一个非常小的子集，然而，典型的音频数据集要大得多。例如，SpeechColab 中 GigaSpeech 的 xs（最小）配置仅包含 10 小时的训练数据，但下载和准备所需的存储空间超过了 13GB。那么当我们想要在更大的数据集上进行训练时会发生什么呢？相同数据集的完整 xl 配置包含 10,000 小时的训练数据，需要超过 1TB 的存储空间。对于大多数人来说，这远远超出了典型硬盘的规格。我们需要花钱购买额外的存储空间吗？或者我们是否有办法在没有磁盘空间限制的情况下训练这些数据集？

🤗 Datasets通过提供[流式模式(streaming mode)](https://huggingface.co/docs/datasets/stream)来解决问题。流式允许我们在迭代数据集时逐步加载数据。与一次性下载整个数据集不同，我们每次加载一个示例。我们遍历数据集，按需即时加载和准备示例。这样，我们只加载我们正在使用的示例，而不是我们不需要的示例！一旦我们完成一个示例的采样，我们继续遍历数据集并加载下一个。

与一次性下载整个数据集相比，流式模式有三个主要优点：

* 磁盘空间：示例在迭代数据集时逐个加载到内存中。由于数据没有被本地下载，因此没有磁盘空间要求，因此你可以使用任意大小的数据集。
* 下载和处理时间：音频数据集很大，需要大量时间来下载和处理。通过流式处理，加载和处理是即时进行的，这意味着你可以在第一个示例准备好后立即开始使用数据集。
* 方便的实验：你可以对一些示例进行实验，以检查你的脚本是否工作正常，而不必下载整个数据集。

流式模式有一个注意事项。当一次性下载完整数据集时，原始数据和处理后的数据都会保存到本地磁盘上。如果我们想重新使用这个数据集，我们可以直接从磁盘加载处理后的数据，跳过下载和处理步骤。因此，我们只需要执行一次下载和处理操作，之后就可以重用准备好的数据。

使用流式模式，数据不会下载到磁盘上。因此，下载和预处理后的数据都不会被缓存。如果我们想重新使用数据集，则必须重复流式步骤，再次加载和处理音频文件。因此，建议下载那些可能多次使用的数据集。

如何启用流式模式？很简单！只需在加载数据集时设置 streaming=True。其余的事情都会由程序自动处理：

```python
gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
```


就像我们对下载的 MINDS-14 子集应用预处理步骤一样，你可以以完全相同的方式对流式数据集进行相同的预处理。

唯一的区别是你不能再使用 Python 索引访问单个样本（例如 gigaspeech["train"][sample_idx]）。相反，你必须迭代整个数据集。以下是在流式数据集中访问示例的方法：

```python
next(iter(gigaspeech["train"]))
```
输出：
```
{
    "segment_id": "YOU0000000315_S0000660",
    "speaker": "N/A",
    "text": "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>",
    "audio": {
        "path": "xs_chunks_0000/YOU0000000315_S0000660.wav",
        "array": array(
            [0.0005188, 0.00085449, 0.00012207, ..., 0.00125122, 0.00076294, 0.00036621]
        ),
        "sampling_rate": 16000,
    },
    "begin_time": 2941.89,
    "end_time": 2945.07,
    "audio_id": "YOU0000000315",
    "title": "Return to Vasselheim | Critical Role: VOX MACHINA | Episode 43",
    "url": "https://www.youtube.com/watch?v=zr2n1fLVasU",
    "source": 2,
    "category": 24,
    "original_full_path": "audio/youtube/P0004/YOU0000000315.opus",
}
```

如果你想要预览大型数据集中的几个示例，可以使用 take() 获取前 n 个元素。让我们从 gigaspeech 数据集中获取前两个示例：

```python
gigaspeech_head = gigaspeech["train"].take(2)
list(gigaspeech_head)
```
输出：
```
[
    {
        "segment_id": "YOU0000000315_S0000660",
        "speaker": "N/A",
        "text": "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>",
        "audio": {
            "path": "xs_chunks_0000/YOU0000000315_S0000660.wav",
            "array": array(
                [
                    0.0005188,
                    0.00085449,
                    0.00012207,
                    ...,
                    0.00125122,
                    0.00076294,
                    0.00036621,
                ]
            ),
            "sampling_rate": 16000,
        },
        "begin_time": 2941.89,
        "end_time": 2945.07,
        "audio_id": "YOU0000000315",
        "title": "Return to Vasselheim | Critical Role: VOX MACHINA | Episode 43",
        "url": "https://www.youtube.com/watch?v=zr2n1fLVasU",
        "source": 2,
        "category": 24,
        "original_full_path": "audio/youtube/P0004/YOU0000000315.opus",
    },
    {
        "segment_id": "AUD0000001043_S0000775",
        "speaker": "N/A",
        "text": "SIX TOMATOES <PERIOD>",
        "audio": {
            "path": "xs_chunks_0000/AUD0000001043_S0000775.wav",
            "array": array(
                [
                    1.43432617e-03,
                    1.37329102e-03,
                    1.31225586e-03,
                    ...,
                    -6.10351562e-05,
                    -1.22070312e-04,
                    -1.83105469e-04,
                ]
            ),
            "sampling_rate": 16000,
        },
        "begin_time": 3673.96,
        "end_time": 3675.26,
        "audio_id": "AUD0000001043",
        "title": "Asteroid of Fear",
        "url": "http//www.archive.org/download/asteroid_of_fear_1012_librivox/asteroid_of_fear_1012_librivox_64kb_mp3.zip",
        "source": 0,
        "category": 28,
        "original_full_path": "audio/audiobook/P0011/AUD0000001043.opus",
    },
]
```


流式模式可以将你的研究提升到一个新的水平：不仅最大的数据集对你可用，而且你可以轻松地在一个步骤中评估多个数据集上的系统，而不用担心磁盘空间。与在单个数据集上进行评估相比，多数据集评估可以更好地衡量语音识别系统的泛化能力（参见端到端语音基准（ESB））。


