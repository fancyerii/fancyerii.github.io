---
layout:     post
title:      "Kaldi的MFCC特征提取代码分析"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 语音识别
    - Kaldi
    - 源代码
    - MFCC
---

本文介绍featbin/compute-mfcc-feats.cc的代码。对MFCC不了解的读者可以参考[MFCC特征提取]({{ site.baseurl }}/books/mfcc)和[Kaldi文档解读：Kaldi的特征提取]({{ site.baseurl }}/kaldidoc/feature)。更多本系列文章请点击[Kaldi代码分析]({{ site.baseurl }}{% post_url 2019-06-22-kaldi-codes %})。

 <!--more-->
 
**目录**
* TOC
{:toc}

## 运行

我们首先看一看这个函数的参数：
```
kaldi/src/featbin$ compute-mfcc-feats 
./compute-mfcc-feats 

Create MFCC feature files.
Usage:  compute-mfcc-feats [options...] <wav-rspecifier> <feats-wspecifier>

Options:
  --allow-downsample          : If true, allow the input waveform to have a higher frequency than the specified --sample-frequency (and we'll downsample). (bool, default = false)
  --allow-upsample            : If true, allow the input waveform to have a lower frequency than the specified --sample-frequency (and we'll upsample). (bool, default = false)
  --blackman-coeff            : Constant coefficient for generalized Blackman window. (float, default = 0.42)
  --cepstral-lifter           : Constant that controls scaling of MFCCs (float, default = 22)
  --channel                   : Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (int, default = -1)
  --debug-mel                 : Print out debugging information for mel bin computation (bool, default = false)
  --dither                    : Dithering constant (0.0 means no dither). If you turn this off, you should set the --energy-floor option, e.g. to 1.0 or 0.1 (float, default = 1)
  --energy-floor              : Floor on energy (absolute, not relative) in MFCC computation. Only makes a difference if --use-energy=true; only necessary if --dither=0.0.  Suggested values: 0.1 or 1.0 (float, default = 0)
  --frame-length              : Frame length in milliseconds (float, default = 25)
  --frame-shift               : Frame shift in milliseconds (float, default = 10)
  --high-freq                 : High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
  --htk-compat                : If true, put energy or C0 last and use a factor of sqrt(2) on C0.  Warning: not sufficient to get HTK compatible features (need to change other parameters). (bool, default = false)
  --low-freq                  : Low cutoff frequency for mel bins (float, default = 20)
  --max-feature-vectors       : Memory optimization. If larger than 0, periodically remove feature vectors so that only this number of the latest feature vectors is retained. (int, default = -1)
  --min-duration              : Minimum duration of segments to process (in seconds). (float, default = 0)
  --num-ceps                  : Number of cepstra in MFCC computation (including C0) (int, default = 13)
  --num-mel-bins              : Number of triangular mel-frequency bins (int, default = 23)
  --output-format             : Format of the output files [kaldi, htk] (string, default = "kaldi")
  --preemphasis-coefficient   : Coefficient for use in signal preemphasis (float, default = 0.97)
  --raw-energy                : If true, compute energy before preemphasis and windowing (bool, default = true)
  --remove-dc-offset          : Subtract mean from waveform on each frame (bool, default = true)
  --round-to-power-of-two     : If true, round window size to power of two by zero-padding input to FFT. (bool, default = true)
  --sample-frequency          : Waveform data sample frequency (must match the waveform file, if specified there) (float, default = 16000)
  --snip-edges                : If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (bool, default = true)
  --subtract-mean             : Subtract mean of each feature file [CMS]; not recommended to do it this way.  (bool, default = false)
  --use-energy                : Use energy (not C0) in MFCC computation (bool, default = true)
  --utt2spk                   : Utterance to speaker-id map rspecifier (if doing VTLN and you have warps per speaker) (string, default = "")
  --vtln-high                 : High inflection point in piecewise linear VTLN warping function (if negative, offset from high-mel-freq (float, default = -500)
  --vtln-low                  : Low inflection point in piecewise linear VTLN warping function (float, default = 100)
  --vtln-map                  : Map from utterance or speaker-id to vtln warp factor (rspecifier) (string, default = "")
  --vtln-warp                 : Vtln warp factor (only applicable if vtln-map not specified) (float, default = 1)
  --window-type               : Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"blackmann") (string, default = "povey")
  --write-utt2dur             : Wspecifier to write duration of each utterance in seconds, e.g. 'ark,t:utt2dur'. (string, default = "")

Standard options:
  --config                    : Configuration file to read (this option may be repeated) (string, default = "")
  --help                      : Print out usage message (bool, default = false)
  --print-args                : Print the command line arguments (to stderr) (bool, default = true)
  --verbose                   : Verbose level (higher->more logging) (int, default = 0)


```

很多参数，但是必须的参数只有两个：输入的wav-rspecifier和输出的feats-wspecifier。不了解rspecifier和wspecifier的读者请先阅读[Kaldi文档解读]({{ site.baseurl }}{% post_url 2019-05-21-kaldi-doc %})和[Kaldi I/O mechanisms ](http://kaldi-asr.org/doc/io.html)(这个文档目前还没来得及翻译)。我们首先需要准备输入的文件：

```
$ cat wav.scp 
1088-134315-0000 flac -c -d -s /home/lili/codes/kaldi/egs/mini_librispeech/s5/mini-libridata/LibriSpeech/train-clean-5/1088/134315/1088-134315-0000.flac |
1088-134315-0001 flac -c -d -s /home/lili/codes/kaldi/egs/mini_librispeech/s5/mini-libridata/LibriSpeech/train-clean-5/1088/134315/1088-134315-0001.flac |
1088-134315-0002 flac -c -d -s /home/lili/codes/kaldi/egs/mini_librispeech/s5/mini-libridata/LibriSpeech/train-clean-5/1088/134315/1088-134315-0002.flac |

```
compute-mfcc-feats只能读取WAV格式的数据，其它的格式需要转换成WAV格式。转换可以"离线"的方式提前用工具转好，也可以on-the-fly的用命令行工具实现，比如我上面的例子是mini-librispeech的数据，它是flac格式的，可以使用flac工具on-the-fly的转好后通过管道传给Kaldi。

下面我们来运行一下(为了避免刷屏，我只显示前10行)：
```
$ compute-mfcc-feats scp:wav.scp ark,t:-|head
compute-mfcc-feats scp:wav.scp ark,t:- 
1088-134315-0000  [
  15.80592 -19.65983 -44.14267 -10.36022 -21.32767 -3.718761 -16.35019 -2.041183 1.383831 -6.504814 -15.53979 4.017059 -1.488521 
  15.80938 -19.18601 -36.05476 1.974784 -16.47062 9.139902 -17.07959 -2.513923 12.35208 0.3189384 -16.05096 -10.61343 -5.74498 
  15.56385 -20.34504 -41.09424 -5.030499 -14.83514 7.03503 -8.041247 3.885094 17.55143 -2.833649 -17.50364 -7.040634 -2.903327 
  15.47452 -18.549 -31.8735 -3.966199 -19.83389 -2.875786 -18.66935 5.130469 9.293931 -11.26405 -14.64669 -5.246292 3.741609 
  15.60467 -18.61004 -35.10693 1.910538 -24.71902 0.7080836 -16.67995 9.620322 5.798228 -7.07972 -12.95449 7.126602 11.80375 
  15.79984 -17.4576 -41.98382 -7.244875 -20.32612 4.706924 -15.74146 4.090664 9.880718 2.285589 -9.086823 4.295952 11.07193 
  15.91331 -13.78769 -35.16341 -2.869465 -16.56114 6.949731 -3.440318 -0.4015923 1.338336 -13.04577 -10.98161 -0.4763947 9.889109 
  16.03437 -11.54739 -36.76908 -4.536423 -24.24585 9.95239 -15.34235 -0.2514563 19.19995 5.409452 -2.524536 4.166977 10.39466 
  16.39969 -8.784815 -41.10754 -6.428618 -22.93435 2.602599 -15.65443 -2.352367 8.833455 -6.360246 -14.28245 -1.446442 2.700707 

```

上面的输出类型是t(文本)，并且用"-"表示输出到标准输出(屏幕)，这样便于查看。

## 调试

阅读代码最好的方式就是单步执行查看变量，调试Kaldi的代码请参考[Kaldi教程(三)的代码调试部分]({{ site.baseurl }}/kaldidoc/tutorial3#matrix%E5%BA%93%E4%BF%AE%E6%94%B9%E5%92%8C%E8%B0%83%E8%AF%95%E4%BB%A3%E7%A0%81)。

注意编译时要把kaldi.mk里的-O1去掉改成"-O0 -DKALDI_PARANOID"，具体参考上面的文档。



## compute-mfcc-feats.cc

这个文件是提取mfcc的工具，包括入口的main函数，它只是做参数的parse，真正干活的代码不多，这里只保留主要的代码。

```
int main(int argc, char *argv[]) {
  try {

    ParseOptions po(usage);
    MfccOptions mfcc_opts;

    // Register the MFCC option struct.
    mfcc_opts.Register(&po);

    // Register the options.
    po.Register("output-format", &output_format, "Format of the output "
                "files [kaldi, htk]");


    std::string wav_rspecifier = po.GetArg(1);

    std::string output_wspecifier = po.GetArg(2);

    Mfcc mfcc(mfcc_opts);


    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.


    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(),
                             vtln_warp_local, &features);
      } catch (...) {
      }
      if (subtract_mean) {
        Vector<BaseFloat> mean(features.NumCols());
        mean.AddRowSumMat(1.0, features);
        mean.Scale(1.0 / features.NumRows());
        for (int32 i = 0; i < features.NumRows(); i++)
          features.Row(i).AddVec(-1.0, mean);
      }
      if (output_format == "kaldi") {
        kaldi_writer.Write(utt, features);
      } else {
      }
      if (utt2dur_writer.IsOpen()) {
        utt2dur_writer.Write(utt, wave_data.Duration());
      }

    }

  } catch(const std::exception &e) {

  }
}
```

前面的parse参数和选项我们可以略过，"Mfcc mfcc(mfcc_opts);"，这是真正干活的。

接下来"SequentialTableReader\<WaveHolder> reader(wav_rspecifier);"用于读取wav文件。然后是用for循环读取每一个wav文件。for循环里面是处理每一个wav文件的代码：
```
      const WaveData &wave_data = reader.Value();
      // 使用SubVector取wav文件的一个通道	
      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
	//使用mfcc提取特征。
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(),
                             vtln_warp_local, &features);
```

如果有subtract_mean，则同一个utterance的每一维特征都减去均值，最后使用"kaldi_writer.Write(utt, features);"输出。

## WaveHolder读取WAV文件
我们是通过下面的代码来构造WAV文件的reader：
```
SequentialTableReader<WaveHolder> reader(wav_rspecifier);
```

这里使用SequentialTableReader，它会读取前面的wav.scp文件，但是具体怎么读取"flac -c -d -s /home/lili/codes/kaldi/egs/mini_librispeech/s5/mini-libridata/LibriSpeech/train-clean-5/1088/134315/1088-134315-0000.flac"输出的WAV文件，则需要靠WaveHolder了。

它读取WAV的代码是：
```
  bool Read(std::istream &is) {
    // We don't look for the binary-mode header here [always binary]
    try {
      t_.Read(is);  // Throws exception on failure.
      return true;
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught in WaveHolder::Read(). " << e.what();
      return false;
    }
  }
```

上面的t_是成员变量(Kaldi的coding style是私有成员以下划线结尾)，在WaveHolder的开始我们可以看到"typedef WaveData T;"，因此最终调用到WaveData类的Read()：
```
void WaveData::Read(std::istream &is) {
  const uint32 kBlockSize = 1024 * 1024;

  WaveInfo header;
  header.Read(is);

  data_.Resize(0, 0);  // clear the data.
  samp_freq_ = header.SampFreq();

  std::vector<char> buffer;
  uint32 bytes_to_go = header.IsStreamed() ? kBlockSize : header.DataBytes();

  // Once in a while header.DataBytes() will report an insane value;
  // read the file to the end
  while (is && bytes_to_go > 0) {
    uint32 block_bytes = std::min(bytes_to_go, kBlockSize);
    uint32 offset = buffer.size();
    buffer.resize(offset + block_bytes);
    is.read(&buffer[offset], block_bytes);
    uint32 bytes_read = is.gcount();
    buffer.resize(offset + bytes_read);
    if (!header.IsStreamed())
      bytes_to_go -= bytes_read;
  }


  uint16 *data_ptr = reinterpret_cast<uint16*>(&buffer[0]);

  // The matrix is arranged row per channel, column per sample.
  data_.Resize(header.NumChannels(),
               buffer.size() / header.BlockAlign());
  for (uint32 i = 0; i < data_.NumCols(); ++i) {
    for (uint32 j = 0; j < data_.NumRows(); ++j) {
      int16 k = *data_ptr++;
      if (header.ReverseBytes())
        KALDI_SWAP2(k);
      data_(j, i) =  k;
    }
  }
}
```

上面的代码显示读取WAV的Header信息，确保文件的采样率，比特率等和compute-mfcc-feats要求的(我们没有设置的话都有合理的默认值，这些默认值请查看看compute-mfcc-feats的帮助)是否一样。关于WAV格式，感兴趣的读者可以参考[Digital Audio - Creating a WAV (RIFF) file](http://www.topherlee.com/software/pcm-tut-wavformat.html)和[WAVE PCM soundfile format](http://soundfile.sapp.org/doc/WaveFormat/)。读者可以对着WAV格式头部说明然后单步调试void WaveInfo::Read(std::istream &is)函数。

## Mfcc
Mfcc是OfflineFeatureTpl模板类使用MfccComputer的typedef：
```
typedef OfflineFeatureTpl<MfccComputer> Mfcc;
```

### OfflineFeatureTpl

我们先看一下OfflineFeatureTpl的文档
```
/// This templated class is intended for offline feature extraction, i.e. where
/// you have access to the entire signal at the start.  It exists mainly to be
/// drop-in replacement for the old (pre-2016) classes Mfcc, Plp and so on, for
/// use in the offline case.  In April 2016 we reorganized the online
/// feature-computation code for greater modularity and to have correct support
/// for the snip-edges=false option.
template <class F>
class OfflineFeatureTpl {
```

大意是：2016年前的代码是Mfcc和Plp这些具体的类，它们用于计算离线的特征，而之后重构了代码。

### ComputeFeatures
我们来看计算特征的函数：
```
template <class F>
void OfflineFeatureTpl<F>::ComputeFeatures(
    const VectorBase<BaseFloat> &wave,
    BaseFloat sample_freq,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  BaseFloat new_sample_freq = computer_.GetFrameOptions().samp_freq;
  if (sample_freq == new_sample_freq) {
    Compute(wave, vtln_warp, output);
  } else {
    if (new_sample_freq < sample_freq &&
        ! computer_.GetFrameOptions().allow_downsample)
        KALDI_ERR << "Waveform and config sample Frequency mismatch: "
                  << sample_freq << " .vs " << new_sample_freq
                  << " (use --allow-downsample=true to allow "
                  << " downsampling the waveform).";
    else if (new_sample_freq > sample_freq &&
             ! computer_.GetFrameOptions().allow_upsample)
      KALDI_ERR << "Waveform and config sample Frequency mismatch: "
                  << sample_freq << " .vs " << new_sample_freq
                << " (use --allow-upsample=true option to allow "
                << " upsampling the waveform).";
    // Resample the waveform.
    Vector<BaseFloat> resampled_wave(wave);
    ResampleWaveform(sample_freq, wave,
                     new_sample_freq, &resampled_wave);
    Compute(resampled_wave, vtln_warp, output);
  }
}
```

它其实只是检查从WAV头部读取的采样率和compute-mfcc-feats传入的是否一致，如果一致使用Compute函数计算，否则如果运行的话对WAV文件进行上采样或者下采样以便满足compute-mfcc-feats的要求，最终还是调用Compute函数。

### Compute

```
template <class F>
void OfflineFeatureTpl<F>::Compute(
    const VectorBase<BaseFloat> &wave,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
  //.......	
  int32 rows_out = NumFrames(wave.Dim(), computer_.GetFrameOptions()),
      cols_out = computer_.Dim();
  //.......
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    BaseFloat raw_log_energy = 0.0;
    ExtractWindow(0, wave, r, computer_.GetFrameOptions(),
                  feature_window_function_, &window,
                  (use_raw_log_energy ? &raw_log_energy : NULL));

    SubVector<BaseFloat> output_row(*output, r);
    computer_.Compute(raw_log_energy, vtln_warp, &window, &output_row);
  }
```
首先使用NumFrames计算WAV有多少帧，然后遍历每一帧：使用ExtractWindow抽取每一帧，然后使用computer_.Compute提取特征。

### NumFrames

目前默认的方式是snip_edges，也和HTK一致，也就是保证不需要padding，如果往后移动超出范围，那就不要了。代码这里就不介绍了。

### ExtractWindow
我们只看一下函数的参数和说明。
```
// ExtractWindow extracts a windowed frame of waveform with a power-of-two,
// padded size.  It does mean subtraction, pre-emphasis and dithering as
// requested.
void ExtractWindow(int64 sample_offset,
                   const VectorBase<BaseFloat> &wave,
                   int32 f,  // with 0 <= f < NumFrames(feats, opts)
                   const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   Vector<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window) {
```

因为要做FFT，所以要求采样点是2的幂，所以会在后面padding。此外如果FrameExtractionOptions需要减去均值或者pre-emphasis或者dithering(增加很小的随机噪声防止log为0），都会在这里处理，最后会把这些点乘以窗口函数FeatureWindowFunction(比如Hamming窗)。

## MfccComputer

最终到了干活的代码了，上面会调用它的Compute函数，请阅读注释：

```
void MfccComputer::Compute(BaseFloat signal_log_energy,
                           BaseFloat vtln_warp,
                           VectorBase<BaseFloat> *signal_frame,
                           VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());
  // 获取Mel FilterBank，为了复用，会把每一个VLTN的alpha作为key存在map里。
  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));
  
  if (opts_.use_energy && !opts_.raw_energy)
    // 用向量向量乘法计算能量
    signal_log_energy = Log(std::max<BaseFloat>(VecVec(*signal_frame, *signal_frame),
                                     std::numeric_limits<float>::epsilon()));
  // FFT，默认是split-radix算法
  if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two.
    RealFft(signal_frame, true);

  // FFT得到的复数计算其模得到功率谱
  ComputePowerSpectrum(signal_frame);
  // 因为是实数，FFT只需要N/2+1个点。
  SubVector<BaseFloat> power_spectrum(*signal_frame, 0,
                                      signal_frame->Dim() / 2 + 1);
  // 使用Filter bank滤波器组提取每个bin的能量
  mel_banks.Compute(power_spectrum, &mel_energies_);

  // 避免对零取log (如果有dithering那么不应该是零，但是dithering是可选的，所以保险一点还是要处理) 
  mel_energies_.ApplyFloor(std::numeric_limits<float>::epsilon());
  mel_energies_.ApplyLog();  // 取log

  feature->SetZero();  // in case there were NaNs.
  // feature = dct_matrix_ * mel_energies [which now have log]
  // 进行DCT得到倒谱
  feature->AddMatVec(1.0, dct_matrix_, kNoTrans, mel_energies_, 0.0);
  
  if (opts_.cepstral_lifter != 0.0)
    // 倒谱系数的lifting
    feature->MulElements(lifter_coeffs_);

  if (opts_.use_energy) {
    // 如果使用能量，那么把这一帧的能量替换掉倒谱的第一个系数。
    if (opts_.energy_floor > 0.0 && signal_log_energy < log_energy_floor_)
      signal_log_energy = log_energy_floor_;
    (*feature)(0) = signal_log_energy;
  }

  if (opts_.htk_compat) {
    // HTK的能量参数放在最后面，为了便于对比，MfccOptions有一个htk_compat选项。
    BaseFloat energy = (*feature)(0);
    for (int32 i = 0; i < opts_.num_ceps - 1; i++)
      (*feature)(i) = (*feature)(i+1);
    if (!opts_.use_energy)
      energy *= M_SQRT2;  // scale on C0 (actually removing a scale
    // we previously added that's part of one common definition of
    // the cosine transform.)
    (*feature)(opts_.num_ceps - 1)  = energy;
  }
}
```

代码其实很清晰，不了解的读者也可以参考[MFCC特征提取]({{ site.baseurl }}/books/mfcc)和[模块二：语音信号处理]({{ site.baseurl }}/dev287x/ssp)。这两篇文章都有Python的代码实现可以参考。


