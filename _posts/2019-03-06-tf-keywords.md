---
layout:     post
title:      "使用Tensorflow识别语音关键词"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 人工智能
    - 语音识别
    - 深度学习
    - Tensorflow
    - CNN
    - 《深度学习理论与实战：提高篇》
---

本教程是《深度学习理论与实战》草稿的部分内容。这个教程我们使用Tensorflow来实现某些关键词的识别，这个例子可以用于移动设备的语音控制。这里识别的词包括"yes", "no", "up", "down", "left", "right", "on", "off", "stop",和"go"这10个词。这个程序最终可以在Android上运行，具体参考[这里](https://www.tensorflow.org/tutorials/sequences/audio_recognition#running_the_model_in_an_android_app)。这是一个比较简单的任务，我们假设所有的录音文件的长度是固定的，因此我们可以使用卷积网络来实现分类。更多文章请点击<a href='/tags/#《深度学习理论与实战：提高篇》'>《深度学习理论与实战：提高篇》</a>。


<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>

 <!--more-->
 
**目录**
* TOC
{:toc}

## 运行代码
完整代码在[这里](https://github.com/fancyerii/blog-codes/blob/master/tf-keywords)。

运行 python train.py即可实现训练。代码会自动下载录音文件，这个文件大小超过2GB，因此第一次运行时需要一定的下载时间。经过18000次迭代之后识别的准确率在90%以上。

## 命令行参数

下面介绍一些这个程序的命令行参数，我们可以通过类似"--data_url=xxxx"来修改它们。

* data_url 训练数据的下载url，默认值是"http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
* data_dir 下载后存放的目录，如果下载过了，第二次就不会再下载
* background_volume 背景噪声的音量，默认0.1。这是一种Data Augmentation的技术，通过给语音增加噪声来提高模型的泛化能力。
* background_frequency 多少比例的训练数据会增加噪声，默认0.8(80%)
* silence_percentage 训练数据中silence的比例，默认10(10%)。silence模拟的是没有输入的情况。
* unknown_percentage 除了我们需要识别的"yes","no"等词，我们还需要加入一些其它词，否则只要有人说话(非silence)它就一定会识别成10个词中的某一个。默认10(10%)，表示会随机加入10%的其它词(比如"one","two","three"等)
* time_shift_ms 录音都是长度1秒的文件，但是在实际预测的时候用户开始的实际是不固定的，为了模拟这种情况，我们这里会随机的把录音文件往前或者往后平移一段时间，这个参数就是指定平移的范围。默认100(ms)，说明会随机的在[-100,100]之间平移数据
* testing_percentage 用于测试的数据比例，默认10(10%)
* validation_percentage 验证集合的比例，默认10(10%)
* sample_rate 录音的采样率，默认16000。需要和提供的wav文件的采样率匹配。
* clip_duration_ms 录音文件的时长，默认1000(ms)。
* window_size_ms 帧长，默认30(ms)，含义参考MFCC特征提取部分
* window_stride_ms 帧移，默认10(ms)
* feature_bin_count DCT后保留的系数个数，默认40(MFCC是12)
* how_many_training_steps 默认"15000,3000"，因为需要调整learning_rate，所以用逗号分成多段
* learning_rate 默认"0.001,0.0001"，结合上面的参数，它的意思是用learning_rate 0.001训练15000个minibatch，然后再用0.0001训练3000个minibatch
* eval_step_interval 默认400，训练多少个batch就评估一次
* summaries_dir 默认/tmp/retrain_logs 保存用于TensorBoard可视化的Summary(Event)文件。
* wanted_words 哪些词是我们需要识别的，默认是"yes,no,up,down,left,right,on,off,stop,go"，这些词之外的词都是unknown_words
* train_dir 模型的checkpoint存放目录，默认/tmp/speech_commands_train
* save_step_interval 默认100，每隔100次迭代就保存一份checkpoint文件
* start_checkpoint 默认空字符串，如果非空，则从这里恢复checkpoint继续训练
* model_architecture 模型结构，默认"conv"，我们只会分析conv结构，但是代码还提高其它模型结构。
* check_nans 是否检查NAN，默认False
* quantize 是否量化参数以便减小模型大小，默认False
* preprocess 语音数据预处理(特征提取)方法，默认"mfcc"


## 数据目录结构

接下来我们看一下下载解压后的数据目录结构。
```
(py3-env) lili@lili-Precision-7720:~/data/speech_dataset$ ls
_background_noise_  down     go       marvin  README.md                     stop              validation_list.txt
backward            eight    happy    nine    right                         testing_list.txt  visual
bed                 five     house    no      seven                         three             wow
bird                follow   learn    off     sheila                        tree              yes
cat                 forward  left     on      six                           two               zero
dog                 four     LICENSE  one     speech_commands_v0.02.tar.gz  up
```
我们先忽略文件，所有的目录都是一个词，每个目录下都是这个词的录音文件。比如
```
(py3-env) lili@lili-Precision-7720:~/data/speech_dataset$ ls one/|head
00176480_nohash_0.wav
004ae714_nohash_0.wav
00f0204f_nohash_0.wav
0132a06d_nohash_0.wav
0132a06d_nohash_1.wav
0132a06d_nohash_2.wav
0132a06d_nohash_3.wav
0132a06d_nohash_4.wav
0135f3f2_nohash_0.wav
0135f3f2_nohash_1.wav
```
这里有一个特殊的目录_background_noise_：
```
(py3-env) lili@lili-Precision-7720:~/data/speech_dataset$ ls _background_noise_/
doing_the_dishes.wav  exercise_bike.wav  README.md        white_noise.wav
dude_miaowing.wav     pink_noise.wav     running_tap.wav
```
这个目录包含了一些背景噪声，我们会对一定比例(background_frequency)的训练数据加入噪声，加入的噪声的音量为background_volume。

## 计算模型的设置参数

接下来我们介绍models.py文件的prepare_model_settings函数，这个函数根据前面的命令行参数来计算一些模型会用到的设置参数。
这个函数的参数为：

* label_count: 识别的词的个数，这里是10
* sample_rate: 采样率 16000
* clip_duration_ms: 录音的时长，这里是1000(ms)
* window_size_ms: 30(ms)
* window_stride_ms: 10(ms)
* feature_bin_count: 40
* preprocess: "mfcc"


```
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
	  spectrogram_length = 0
  else:
	  spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'average':
	  fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
	  average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
	  fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
	  average_window_width = -1
	  fingerprint_width = feature_bin_count
  else:
	  raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or'
		  ' "average")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
```

这个函数会计算语音需要的样本点个数(desired_samples)，$$sample\_rate(16000) * clip\_duration\_ms(1000) / 1000=16000$$。window_size_samples和window_stride_samples计算一帧的样本个数以及帧移的样本个数，这里它们的值是480和160。

接下来计算声音有多少帧(spectrogram_length)，首先是前480个点构成一帧，然后每次后移160个点，因此计算方法是：

$$
1+(desired\_samples - window\_size\_samples)/window\_stride\_samples=98
$$

对于MFCC来说，每一帧都有feature_bin_count(40)个系数，因此fingerprint_width(类比图像的height)也是40。
这里我们可以把语音类比成图像，每一帧代表一个时刻，每个时刻是40个系数，因此最终是类似spectrogram_length(98) x fingerprint_width(40)的图片。而最终的fingerprint_size就是98x40=3920。

虽然我们这里使用卷积网络来进行识别，但是这里的代码结构并不会要求输入特征(fingerprint)是二维的，它只要求它是一个一维的向量。不同的模型结构会要求不同的输入，比如conv结构要求它是二维的，那么它就自己把这个一维向量reshape成它需要的二维向量。

## AudioProcessor

接下来我们分析input_data.py的AudioProcessor类，这个类的作用是下载和处理数据，建立索引(训练集合包括哪些文件，它们的label是什么，是否要求加入噪音等等信息)，然后提取特征生成一个batch的训练数据(batch x 而最终的fingerprint_size)。

### 构造函数

它的构造函数为：
```
  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
		  wanted_words, validation_percentage, testing_percentage,
		  model_settings, summaries_dir):
	  self.data_dir = data_dir
	  self.maybe_download_and_extract_dataset(data_url, data_dir)
	  self.prepare_data_index(silence_percentage, unknown_percentage,
	  wanted_words, validation_percentage,
	  testing_percentage)
	  self.prepare_background_data()
	  self.prepare_processing_graph(model_settings, summaries_dir)
```
构造函数首先调用maybe_download_and_extract_dataset函数来下载和解压数据。这个函数的细节我们跳过，有兴趣的读者可以自行阅读。

### prepare_data_index

接下来我们来分析prepare_data_index函数，这个函数读取数据目录，把一些重要信息放到self.data_index这个dict。self.data_index是一个dict，有3个key——'training'、'validation'和'testing'，分别代表训练集、验证集和测试集。而每一个key对应的value是一个list，list的每一个元素是一个对象，它代表一个wav文件。这个对象有一个label属性和path属性，比如testing的value list里有一个：
```
{'label': 'right', 'file': '/home/lili/data/speech_dataset/right/c22d3f18_nohash_2.wav'}
```
此外这个函数也会把所有的词放到self.word_list里：
```
<class 'list'>: ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
```
self.word_to_idx里存放的是word和idx的映射关系：
```
{'marvin': 1, 'tree': 1, 'learn': 1, 'dog': 1, 'sheila': 1, 'bird': 1, 'right': 7, 'off': 9, 'backward': 1, 'six': 1, 'two': 1, 'no': 3, 'yes': 2, 'one': 1, 'follow': 1, 'up': 4, 'three': 1, 'forward': 1, 'happy': 1, 'nine': 1, 'bed': 1, 'zero': 1, 'house': 1, 'visual': 1, 'five': 1, 'seven': 1, 'cat': 1, 'left': 6, 'stop': 10, 'go': 11, 'four': 1, 'on': 8, 'wow': 1, 'down': 5, 'eight': 1}
```
我们可以看到，所有我们不需要识别的词(unknown words)都映射成1(_unknown_)。

实现的代码比较简单，但是有两点需要注意一下。

第一点涉及到划分训练集、验证集和测试集的一个小技巧。通常我们的训练数据是不断增加的，如果按照随机的按比例划分训练集、验证集和测试集，那么增加一个新的数据重新划分后有可能把原来的训练集中的数据划分到测试数据里。因为我们的模型可能要求incremental的训练，因此这就相对于把测试数据也拿来训练了。因此我们需要一种“稳定”的划分方法——原来在训练集中的数据仍然在训练数据中。这里我们使用的技巧就是对于文件名进行hash，然后根据hash的结果对总量取模来划分到不同的集合里。这样就能保证同一个数据第一次如果是在训练集合里，那么它永远都会划分到训练集合里。不过它只能大致保证三个集合的比例而不能绝对的保证比例。它的核心代码是：
```
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
	  (MAX_NUM_WAVS_PER_CLASS + 1)) *
	  (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
	  result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
	  result = 'testing'
  else:
	  result = 'training'
  return result
```
这个函数首先计算根据文件名计算hash_name_hashed，这是一个16进制的数字。(int(hash_name_hashed, 16)把它转成10机制的数字，然后模(MAX_NUM_WAVS_PER_CLASS + 1)把它的范围变换到[0, MAX_NUM_WAVS_PER_CLASS]，然后除以MAX_NUM_WAVS_PER_CLASS把它的范围变换到[0, 1]，最后乘以100把它变成[0, 100]。

因为hash函数是随机划分的，因此我们可以认为这个数字在[0,100]之间是均匀分布的，因此我们选择小于validation_percentage的数划分到验证集里，这样就可以保证验证集的比例大致(而不是绝对)是validation_percentage%。同理我们[validation_percentage, testing_percentage + validation_percentage]这个区间的数据划分到测试集合里。

假设原来有100个训练数据，现在有增加了10个数据。因为hash函数和划分方式不变，因此原来的100个训练数据总是一样的划分方式——原来在训练集中，那么现在也在训练集中；原来在测试集的现在也会在测试集中。而新增的10个数据大致也是按照比例划分到三个集合里的。假设现在又增加了20个训练数据，那么原来前110个数据的划分和第二次是一样的，而新增的20个数据也会大致按照比例划分到三个集合里。

第二点就是每个集合里都要加入一定比例(silence_percentage)的silence和unknown词，代码为：
```
    # 我们需要一个文件来表示silence，因为之后会乘以0，所以它的内容不重要。我们这里选择训练数据的第一个文件。
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
	    set_size = len(self.data_index[set_index])
	    silence_size = int(math.ceil(set_size * silence_percentage / 100))
	    for _ in range(silence_size):
		    self.data_index[set_index].append({
			    'label': SILENCE_LABEL,
			    'file': silence_wav_path
		    })
	    # 随机寻找相应数量的unknown词。
	    random.shuffle(unknown_index[set_index])
	    unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
	    self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
```

因为silence文件(没加噪音前)后面会乘以零，因此我们随便给它一个wav文件路径就行。注意silence_percentage的严格意义是指silence和"yes"，"no"等十个词的数量的比例，而不是silence占总数的比例。类似的unknown_percentage也是未知词和这十个词的比例。unknown_index["training"]存放的是用于训练的未知词，因此我们先对unknown_index进行随机shuffle，然后随机选择前n个就相对于随机的从未知词里选择n个。

### prepare_background_data函数

这个函数的作用是加载_background_noise_目录下的噪音文件到内存，以便后续在产生一个batch的训练数据时给干净的语音加入一定比例的随机噪声。它的主要代码为：
```
    with tf.Session(graph=tf.Graph()) as sess:
	    wav_filename_placeholder = tf.placeholder(tf.string, [])
	    wav_loader = io_ops.read_file(wav_filename_placeholder)
	    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
	    search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
		    '*.wav')
	    for wav_path in gfile.Glob(search_path):
		    wav_data = sess.run(
			    wav_decoder,
			    feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
		    self.background_data.append(wav_data)
```
因为需要用Tensorflow的操作来读取音频文件，所以首先构造一个Session。然后构造一个string类型的placeholder，用来传递文件路径。接着使用io_ops.read_file来读取wav文件，再用contrib_audio.decode_wav来解码音频文件。返回的wav_decoder对象的audio属性就是一个Tensor里，因为这个wav是单声道的录音，因此返回的Tensor是(16000, 1)，如果是立体声的，那么可能返回(16000, 2)，在语音识别中，我们通常要求单声道的录音。

接着遍历这个目录下的所有wav文件，遍历用到了gfile.Glob函数，它允许我们用通配符来查找文件。然后使用session.run来读取wav文件到wav_data，最后把它们都加到self.background_data里。


### prepare_processing_graph函数

这个函数会创建计算图中用于处理数据的操作，包括读取录音文件，随机的平移，增加噪声，最后提取MFCC特征。
```
    with tf.get_default_graph().name_scope('data'):
	    desired_samples = model_settings['desired_samples']
	    self.wav_filename_placeholder_ = tf.placeholder(
		    tf.string, [], name='wav_filename')
	    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
	    wav_decoder = contrib_audio.decode_wav(
		    wav_loader, desired_channels=1, desired_samples=desired_samples)
	    # Allow the audio sample's volume to be adjusted.
	    self.foreground_volume_placeholder_ = tf.placeholder(
		    tf.float32, [], name='foreground_volume')
	    scaled_foreground = tf.multiply(wav_decoder.audio,
		    self.foreground_volume_placeholder_)
	    # Shift the sample's start position, and pad any gaps with zeros.
	    self.time_shift_padding_placeholder_ = tf.placeholder(
		    tf.int32, [2, 2], name='time_shift_padding')
	    self.time_shift_offset_placeholder_ = tf.placeholder(
		    tf.int32, [2], name='time_shift_offset')
	    padded_foreground = tf.pad(
		    scaled_foreground,
		    self.time_shift_padding_placeholder_,
		    mode='CONSTANT')
	    sliced_foreground = tf.slice(padded_foreground,
		    self.time_shift_offset_placeholder_,
		    [desired_samples, -1])
	    # Mix in background noise.
	    self.background_data_placeholder_ = tf.placeholder(
		    tf.float32, [desired_samples, 1], name='background_data')
	    self.background_volume_placeholder_ = tf.placeholder(
		    tf.float32, [], name='background_volume')
	    background_mul = tf.multiply(self.background_data_placeholder_,
		    self.background_volume_placeholder_)
	    background_add = tf.add(background_mul, sliced_foreground)
	    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
	    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
	    spectrogram = contrib_audio.audio_spectrogram(
		    background_clamp,
		    window_size=model_settings['window_size_samples'],
		    stride=model_settings['window_stride_samples'],
		    magnitude_squared=True)
	    tf.summary.image(
		    'spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)

	    if model_settings['preprocess'] == 'average':
		    self.output_ = tf.nn.pool(
		    tf.expand_dims(spectrogram, -1),
		    window_shape=[1, model_settings['average_window_width']],
		    strides=[1, model_settings['average_window_width']],
			    pooling_type='AVG',
			    padding='SAME')
		    tf.summary.image('shrunk_spectrogram', self.output_, max_outputs=1)
	    elif model_settings['preprocess'] == 'mfcc':
		    self.output_ = contrib_audio.mfcc(
			    spectrogram,
			    wav_decoder.sample_rate,
			    dct_coefficient_count=model_settings['fingerprint_width'])
		    tf.summary.image(
			    'mfcc', tf.expand_dims(self.output_, -1), max_outputs=1)
	    else:
		    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or'
			    ' "average")' % (model_settings['preprocess']))
	    
	    # Merge all the summaries and write them out to /tmp/retrain_logs (by
	    # default)
	    self.merged_summaries_ = tf.summary.merge_all(scope='data')
	    self.summary_writer_ = tf.summary.FileWriter(summaries_dir + '/data',
	    tf.get_default_graph())
```

代码首先创建一个placeholder wav_filename_placeholder_来存放wav文件的路径，然后用和上面类似的代码读取wav文件并且解码wav文件。然后创建一个placeholder foreground_volume_placeholder_，它用来控制wav文件的最大音量。scaled_foreground就是把wav文件解码后得到的tensor都乘以foreground_volume_placeholder_。

接下来的代码实现语音的平移，如果是右移，那么需要在左边补零；如果是左移，则要右边补零。这个padding的量也是在生成batch数据的时候动态产生的，所以也定义为一个PlaceHolder。因为语音的tensor是(16000,1)的，所有padding是一个[2,2]的tensor，不过通常只在第一个维度(时间)padding。比如右移100个点，那么传入的tensor是[[100,0],[0,0]]。如果是左移，我们除了要padding，还要把左边的部分“切掉“，因此还会传入一个time_shift_offset_placeholder_，如果是右移，那么这个值是零。比如我们要实现左移100个点，那么传入的time_shift_padding_placeholder_应该是[[0,100],[0,0]],而time_shift_offset_placeholder_应该是[100]。

最终我们通过pad函数和slice函数得到我们需要的sliced_foreground。

接着我们需要混入噪声。placeholder background_data_placeholder_表示噪声，而background_volume_placeholder_表示混入的音量(比例)，如果background_volume_placeholder_是零就表示没有噪声。把它们乘起来就得到background_mul，然后把它加到sliced_foreground就得到background_add，因为加起来音量可能超过1，所有需要把大于1的变成1，这可以使用clip_by_value函数把音量限制在[-1,1]的区间里。

然后使用contrib_audio.audio_spectrogram把时域的信号变成二维的语谱图(spectrogram)。这个函数除了需要传入参数background_clamp之外，还需要传入帧长和帧移，magnitude_squared为True则返回的是能量的平方。因为FFT返回的是一个复数，我们一般需要的是它的模$$\sqrt{\|\text{实部}\|^2 + \|\text{虚部}\|^2 \;\;\;\;}$$，这里返回的是它的平方，这样能减少开根的计算，而且后面提取MFCC也要求是平方值。返回的spectrogram是一个三维的Tensor(1, 98, 257)，第一个维度是输入语音的通道数（单声道1，立体声是2），第二维是帧数，第三维是FFT系数的个数。

再调用contrib_audio.mfcc提取MFCC特征，这个函数需要传入spectrogram，采样率，以及返回的DCT系数的个数(40)。最终得到的output_是(1, 98, 40)的Tensor。

## 构建训练的Graph

接下来train.py的main函数会构造用于训练的Graph，部分重要代码如下：

```
  input_placeholder = tf.placeholder(
	  tf.float32, [None, fingerprint_size], name='fingerprint_input')
	  
  fingerprint_input = input_placeholder
  logits, dropout_prob = models.create_model(
	  fingerprint_input,
	  model_settings,
	  FLAGS.model_architecture,
	  is_training=True)
```

输入是一个placeholder input_placeholder，它的shape是(None, 3920)，其中3920=98(帧) x 40。然后调用models.create_model创建conv模型，返回分类的logits和一个placeholder dropout_prob。再后面计算loss和梯度就非常简单了，我们略过，这里介绍create_model函数。这个函数会根据参数选择不同的模型结构，我们这里只介绍conv结构，这是通过函数create_conv_model实现的。它的代码为：
```
def create_conv_model(fingerprint_input, model_settings, is_training):
	"""构建一个标准的卷积网络
	
	这个网络结果和下面论文'Convolutional Neural Networks for Small-footprint Keyword Spotting'
        的'cnn-trad-fpool3'类似，论文的url是：
	
	http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
	
	这是graph的结构：
	
	(fingerprint_input)
		v
		[Conv2D]<-(weights)
		v
		[BiasAdd]<-(bias)
		v
		[Relu]
		v
		[MaxPool]
		v
		[Conv2D]<-(weights)
		v
		[BiasAdd]<-(bias)
		v
		[Relu]
		v
		[MaxPool]
		v
		[MatMul]<-(weights)
		v
		[BiasAdd]<-(bias)
		v
	
	这个模型的效果很不错，但是它有大量的参数(占内存)和需要大量的计算(费CPU)。
        如果需要在资源不足的环境里，建议使用'low_latency_conv'
	
	如果is_training是True，那么会返回一个dropout的PlaceHolder
        ，调用者在session.run的时候需要feed需要dropout的比例。
	
	
	参数:
		fingerprint_input: 输入Tensor，(batch, fingerprint_size) 
		model_settings: 模型的设置
		is_training: 是否训练阶段
		
	返回值:
		输出分类的logits，如果是训练，还会返回一个dropout的place holder
	"""
	if is_training:
		dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
	input_frequency_size = model_settings['fingerprint_width']
	input_time_size = model_settings['spectrogram_length']
	fingerprint_4d = tf.reshape(fingerprint_input,
		[-1, input_time_size, input_frequency_size, 1])
	first_filter_width = 8
	first_filter_height = 20
	first_filter_count = 64
	first_weights = tf.get_variable(
		name='first_weights',
		initializer=tf.truncated_normal_initializer(stddev=0.01),
		shape=[first_filter_height, first_filter_width, 1, first_filter_count])
	first_bias = tf.get_variable(
		name='first_bias',
		initializer=tf.zeros_initializer,
		shape=[first_filter_count])
	first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
		'SAME') + first_bias
	first_relu = tf.nn.relu(first_conv)
	if is_training:
		first_dropout = tf.nn.dropout(first_relu, dropout_prob)
	else:
		first_dropout = first_relu
	max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
	
	
	second_filter_width = 4
	second_filter_height = 10
	second_filter_count = 64
	second_weights = tf.get_variable(
	name='second_weights',
	initializer=tf.truncated_normal_initializer(stddev=0.01),
		shape=[
		second_filter_height, second_filter_width, first_filter_count,
		second_filter_count
		])
	second_bias = tf.get_variable(
		name='second_bias',
		initializer=tf.zeros_initializer,
		shape=[second_filter_count])
	second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
		'SAME') + second_bias
	second_relu = tf.nn.relu(second_conv)
	if is_training:
		second_dropout = tf.nn.dropout(second_relu, dropout_prob)
	else:
		second_dropout = second_relu
		
	second_conv_shape = second_dropout.get_shape()
	second_conv_output_width = second_conv_shape[2]
	second_conv_output_height = second_conv_shape[1]
	second_conv_element_count = int(
		second_conv_output_width * second_conv_output_height *
		second_filter_count)
	flattened_second_conv = tf.reshape(second_dropout,
		[-1, second_conv_element_count])
	label_count = model_settings['label_count']
	final_fc_weights = tf.get_variable(
		name='final_fc_weights',
		initializer=tf.truncated_normal_initializer(stddev=0.01),
		shape=[second_conv_element_count, label_count])
	final_fc_bias = tf.get_variable(
		name='final_fc_bias',
		initializer=tf.zeros_initializer,
		shape=[label_count])
	final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
	if is_training:
		return final_fc, dropout_prob
	else:
		return final_fc
```
代码很简单，首先把输入从(batch, 3920) reshape成(batch, 98, 40, 1)，类似于图片的4D Tensor。然后是常规的卷积，max pooling和全连接层。如果是训练阶段，那么会进行dropout，这个dropout是一个placeholder，也会作为返回值。这样调用者就可以在用session运行它的时候可以feed合适的dropout值。

## 训练一个batch的训练数据

接下来我们继续分析怎么生成一个batch的训练数据并且用它来进行一次迭代，代码在train.py的main函数里。代码为：
```
  for training_step in xrange(start_step, training_steps_max + 1):
	  # ... 更加step确定当前的learning rate ...
	  # 获取一个batch的数据
	  train_fingerprints, train_ground_truth = audio_processor.get_data(
		  FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
		  FLAGS.background_volume, time_shift_samples, 'training', sess)
	  # 运行一次训练
	  train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
		  [
			  merged_summaries,
			  evaluation_step,
			  cross_entropy_mean,
			  train_step,
			  increment_global_step,
		  ],
		  feed_dict={
			  fingerprint_input: train_fingerprints,
			  ground_truth_input: train_ground_truth,
			  learning_rate_input: learning_rate_value,
			  dropout_prob: 0.5
		  })
```

我们首先用audio_processor.get_data随机生成一个batch的训练数据，然后用session.run来运行train_step。feed进去的包括一个batch的训练数据train_fingerprints和train_ground_truth，以及learning rate和dropout概率。

接下来我们分析audio_processor.get_data随机生成一个batch的训练数据的过程。这个函数的参数为：

* how_many batch大小，如果是-1则返回所有
* offset 如果是非随机的生成数据，这个参数指定开始的offset
* model_settings 模型的设置
* background_frequency 0.0-1.0之间的值，表示需要混入噪音的数据的比例
* background_volume_range 背景噪音的音量
* time_shift 平移的范围，为[-time_shift, time_shift]
* mode 'training', 'validation'或者'testing'
* sess Tensorflow session，用于执行前面用于产生数据的Operation，参考prepare_processing_graph函数


它的主要代码为：
\begin{lstlisting}
	# data和labels是要返回的内容
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    
    for i in xrange(offset, offset + sample_count):
	    # 选择音频的下标，如果是确定的选取(比如非training阶段)
	    if how_many == -1 or pick_deterministically:
		    sample_index = i
	    else:
		    # 随机选择一个音频
		    sample_index = np.random.randint(len(candidates))
	    sample = candidates[sample_index]
	    
	    # 如果参数time_shift > 0 则随机产生一个[-time_shift, time_shift]的平移量。
	    if time_shift > 0:
		    time_shift_amount = np.random.randint(-time_shift, time_shift)
	    else:
		    time_shift_amount = 0
		
			   
	    if time_shift_amount > 0:
		    # 如果是右移，则在左边papdding 
		    time_shift_padding = [[time_shift_amount, 0], [0, 0]]
		    time_shift_offset = [0, 0]
	    else:
		    # 如果是右移，则在右边padding，并且需要设置offset为平移量
		    time_shift_padding = [[0, -time_shift_amount], [0, 0]]
		    time_shift_offset = [-time_shift_amount, 0]
	    
	    # 运行session的feed dict
	    input_dict = {
		    self.wav_filename_placeholder_: sample['file'],
		    self.time_shift_padding_placeholder_: time_shift_padding,
		    self.time_shift_offset_placeholder_: time_shift_offset,
	    }
	    # 如果需要加入背景噪音(训练阶段)或者是silence，那么需要加入噪音
	    if use_background or sample['label'] == SILENCE_LABEL:
		    background_index = np.random.randint(len(self.background_data))
		    background_samples = self.background_data[background_index]
			
			# 从噪音文件里随机选择1s的数据。
		    background_offset = np.random.randint(
			    0, len(background_samples) - model_settings['desired_samples'])
			    background_clipped = background_samples[background_offset:(
			    background_offset + desired_samples)]
			    background_reshaped = background_clipped.reshape([desired_samples, 1])
			# 如果是silence，那么噪音的音量是随机的从(0, 1)中均匀选择
		    if sample['label'] == SILENCE_LABEL:
			    background_volume = np.random.uniform(0, 1)
			# 如果是加背景噪音，那么需要给background_frequency(0.8)比例的加噪音    
		    elif np.random.uniform(0, 1) < background_frequency:
			    #背景噪音的音量从(0, background_volume_range)中均匀产生。
			    background_volume = np.random.uniform(0, background_volume_range)
			# 1- background_frequency(0.2)比例的不加噪音    
		    else:
			    background_volume = 0
	    else:
		    # 不用加噪音
		    background_reshaped = np.zeros([desired_samples, 1])
		    background_volume = 0
		
		# 设置feed dict的背景噪音和音量    
	    input_dict[self.background_data_placeholder_] = background_reshaped
	    input_dict[self.background_volume_placeholder_] = background_volume
	    
	    # 如果数据是silence，那么设置音量为0。
	    if sample['label'] == SILENCE_LABEL:
		    input_dict[self.foreground_volume_placeholder_] = 0
	    else:
		    input_dict[self.foreground_volume_placeholder_] = 1
		    
	    # 运行graph来产生audio 
	    summary, data_tensor = sess.run(
		    [self.merged_summaries_, self.output_], feed_dict=input_dict)
		    
	    self.summary_writer_.add_summary(summary)
	    data[i - offset, :] = data_tensor.flatten()
	    label_index = self.word_to_index[sample['label']]
	    labels[i - offset] = label_index
    return data, labels
\end{lstlisting}

这里有一点不一样的就是读取音频文件以及提取MFCC特征都是用Tensorflow的Operation来实现的，因此需要用session来运行相应部分的计算图，这部分计算图是前面prepare_processing_graph定义的保存在self.output_里。
