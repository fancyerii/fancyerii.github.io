---
layout:     post
title:      "Huggingface Whisper代码阅读（一）" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - Huggingface
    - Whisper
    - code
---

本文分析阅读Huggingface Whisper的代码。

<!--more-->

**目录**
* TOC
{:toc}


## 测试代码

```
from transformers import pipeline
import torch

#model_id = "openai/whisper-large-v3"  # update with your model id
model_id = "openai/whisper-small"

device = "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model=model_id, device=device
)


def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe"
        },
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]

text = transcribe_speech("1.wav")

print(text)
```

## 模型结构

根据论文，模型结构如下图：

<a>![](/img/hfwhisper/1.png)</a>
 
whipser是encoder-decoder架构。encoder除了输入之前的cnn之外，基本是标准的Transformer Encoder。decoder也是标准的Transformer，只不过token的设计比较复杂而已。关于Whisper的原理，请参考[Robust Speech Recognition via Large-Scale Weak Supervision论文解读](/2024/02/15/whisper/)。
 
## 模型加载

 
### pipeline函数

pipeline函数的文档参考[这里](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline)。

这个函数的原型是：

```
def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map=None,
    torch_dtype=None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs,
) -> Pipeline:
```

参数说明如下：

* task (str) — The task defining which pipeline will be returned. Currently accepted tasks are:
    * "audio-classification": will return a AudioClassificationPipeline.
    * "automatic-speech-recognition": will return a AutomaticSpeechRecognitionPipeline.
    * "conversational": will return a ConversationalPipeline.
    * "depth-estimation": will return a DepthEstimationPipeline.
    * "document-question-answering": will return a DocumentQuestionAnsweringPipeline.
    * "feature-extraction": will return a FeatureExtractionPipeline.
    * "fill-mask": will return a FillMaskPipeline:.
    * "image-classification": will return a ImageClassificationPipeline.
    * "image-feature-extraction": will return an ImageFeatureExtractionPipeline.
    * "image-segmentation": will return a ImageSegmentationPipeline.
    * "image-to-image": will return a ImageToImagePipeline.
    * "image-to-text": will return a ImageToTextPipeline.
    * "mask-generation": will return a MaskGenerationPipeline.
    * "object-detection": will return a ObjectDetectionPipeline.
    * "question-answering": will return a QuestionAnsweringPipeline.
    * "summarization": will return a SummarizationPipeline.
    * "table-question-answering": will return a TableQuestionAnsweringPipeline.
    * "text2text-generation": will return a Text2TextGenerationPipeline.
    * "text-classification" (alias "sentiment-analysis" available): will return a TextClassificationPipeline.
    * "text-generation": will return a TextGenerationPipeline:.
    * "text-to-audio" (alias "text-to-speech" available): will return a TextToAudioPipeline:.
    * "token-classification" (alias "ner" available): will return a TokenClassificationPipeline.
    * "translation": will return a TranslationPipeline.
    * "translation_xx_to_yy": will return a TranslationPipeline.
    * "video-classification": will return a VideoClassificationPipeline.
    * "visual-question-answering": will return a VisualQuestionAnsweringPipeline.
    * "zero-shot-classification": will return a ZeroShotClassificationPipeline.
    * "zero-shot-image-classification": will return a ZeroShotImageClassificationPipeline.
    * "zero-shot-audio-classification": will return a ZeroShotAudioClassificationPipeline.
    * "zero-shot-object-detection": will return a ZeroShotObjectDetectionPipeline.

* model (str or PreTrainedModel or TFPreTrainedModel, optional) — The model that will be used by the pipeline to make predictions. This can be a model identifier or an actual instance of a pretrained model inheriting from PreTrainedModel (for PyTorch) or TFPreTrainedModel (for TensorFlow).
If not provided, the default for the task will be loaded.

* config (str or PretrainedConfig, optional) — The configuration that will be used by the pipeline to instantiate the model. This can be a model identifier or an actual pretrained model configuration inheriting from PretrainedConfig.
If not provided, the default configuration file for the requested model will be used. That means that if model is given, its default configuration will be used. However, if model is not supplied, this task’s default model’s config is used instead.

* tokenizer (str or PreTrainedTokenizer, optional) — The tokenizer that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained tokenizer inheriting from PreTrainedTokenizer.
If not provided, the default tokenizer for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default tokenizer for config is loaded (if it is a string). However, if config is also not given or not a string, then the default tokenizer for the given task will be loaded.

* feature_extractor (str or PreTrainedFeatureExtractor, optional) — The feature extractor that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained feature extractor inheriting from PreTrainedFeatureExtractor.
Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal models. Multi-modal models will also require a tokenizer to be passed.
If not provided, the default feature extractor for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default feature extractor for config is loaded (if it is a string). However, if config is also not given or not a string, then the default feature extractor for the given task will be loaded.

* framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.
If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

* revision (str, optional, defaults to "main") — When passing a task name or a string model identifier: The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so revision can be any identifier allowed by git.
* use_fast (bool, optional, defaults to True) — Whether or not to use a Fast tokenizer if possible (a PreTrainedTokenizerFast).
* use_auth_token (str or bool, optional) — The token to use as HTTP bearer authorization for remote files. If True, will use the token generated when running huggingface-cli login (stored in ~/.huggingface).
* device (int or str or torch.device) — Defines the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which this pipeline will be allocated.
* device_map (str or Dict[str, Union[int, str, torch.device], optional) — Sent directly as model_kwargs (just a simpler shortcut). When accelerate library is present, set device_map="auto" to compute the most optimized device_map automatically (see here for more information).
Do not use device_map AND device at the same time as they will conflict

* torch_dtype (str or torch.dtype, optional) — Sent directly as model_kwargs (just a simpler shortcut) to use the available precision for this model (torch.float16, torch.bfloat16, … or "auto").
* trust_remote_code (bool, optional, defaults to False) — Whether or not to allow for custom code defined on the Hub in their own modeling, configuration, tokenization or even pipeline files. This option should only be set to True for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.
* model_kwargs (Dict[str, Any], optional) — Additional dictionary of keyword arguments passed along to the model’s from_pretrained(..., **model_kwargs) function.
* kwargs (Dict[str, Any], optional) — Additional keyword arguments passed along to the specific pipeline init (see the documentation for the corresponding pipeline class for possible values).

参数很多，我们这里传入的是task="automatic-speech-recognition"，model="openai/whisper-small"以及device="cpu"。

这个函数的核心代码是：

```
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **hub_kwargs,
            )
            
            
            
        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )


        normalized_task, targeted_task, task_options = check_task(task)        
        
        framework, model = infer_framework_load_model(
            model,
            model_classes=model_classes,
            config=config,
            framework=framework,
            task=task,
            **hub_kwargs,
            **model_kwargs,
        )
            
            ....
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **hub_kwargs, **tokenizer_kwargs
            )
            
            ...
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                feature_extractor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )
            
    ....
    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
```

#### cached_file函数

resolved_config_file是模型的配置信息，第一次运行会下载，之后会使用cache。我这里的cache位置是'.........../.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/config.json'。这个文件保持了模型的结果，完整的内容是：

```
{
  "_name_or_path": "openai/whisper-small",
  "activation_dropout": 0.0,
  "activation_function": "gelu",
  "architectures": [
    "WhisperForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "begin_suppress_tokens": [
    220,
    50257
  ],
  "bos_token_id": 50257,
  "d_model": 768,
  "decoder_attention_heads": 12,
  "decoder_ffn_dim": 3072,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 12,
  "decoder_start_token_id": 50258,
  "dropout": 0.0,
  "encoder_attention_heads": 12,
  "encoder_ffn_dim": 3072,
  "encoder_layerdrop": 0.0,
  "encoder_layers": 12,
  "eos_token_id": 50257,
  "forced_decoder_ids": [
    [
      1,
      50259
    ],
    [
      2,
      50359
    ],
    [
      3,
      50363
    ]
  ],
  "init_std": 0.02,
  "is_encoder_decoder": true,
  "max_length": 448,
  "max_source_positions": 1500,
  "max_target_positions": 448,
  "model_type": "whisper",
  "num_hidden_layers": 12,
  "num_mel_bins": 80,
  "pad_token_id": 50257,
  "scale_embedding": false,
  "suppress_tokens": [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    359,
    503,
    522,
    542,
    873,
    893,
    902,
    918,
    922,
    931,
    1350,
    1853,
    1982,
    2460,
    2627,
    3246,
    3253,
    3268,
    3536,
    3846,
    3961,
    4183,
    4667,
    6585,
    6647,
    7273,
    9061,
    9383,
    10428,
    10929,
    11938,
    12033,
    12331,
    12562,
    13793,
    14157,
    14635,
    15265,
    15618,
    16553,
    16604,
    18362,
    18956,
    20075,
    21675,
    22520,
    26130,
    26161,
    26435,
    28279,
    29464,
    31650,
    32302,
    32470,
    36865,
    42863,
    47425,
    49870,
    50254,
    50258,
    50360,
    50361,
    50362
  ],
  "torch_dtype": "float32",
  "transformers_version": "4.27.0.dev0",
  "use_cache": true,
  "vocab_size": 51865
}
```

#### AutoConfig.from_pretrained

AutoConfig.from_pretrained函数通过这个文件的内容用AutoConfig加载特定模型(model_type=whisper)的配置。


```
        elif "model_type" in config_dict:
            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date."
                )
            return config_class.from_dict(config_dict, **unused_kwargs)
```

这里config_class是transformers.models.whisper.configuration_whisper.WhisperConfig。

config_class.from_dict(config_dict, **unused_kwargs)的代码继承自父类AutoConfig，它做一些特殊处理，最终调用子类的构造函数：

```
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        ...

        config = cls(**config_dict)
```

在这里，config_dict为：

```
"{'_name_or_path': 'openai/whisper-small', 'activation_dropout': 0.0, 'activation_function': 'gelu', 'architectures': ['WhisperForConditionalGeneration'], 'attention_dropout': 0.0, 'begin_suppress_tokens': [220, 50257], 'bos_token_id': 50257, 'd_model': 768, 'decoder_attention_heads': 12, 'decoder_ffn_dim': 3072, 'decoder_layerdrop': 0.0, 'decoder_layers': 12, 'decoder_start_token_id': 50258, 'dropout': 0.0, 'encoder_attention_heads': 12, 'encoder_ffn_dim': 3072, 'encoder_layerdrop': 0.0, 'encoder_layers': 12, 'eos_token_id': 50257, 'forced_decoder_ids': [[1, 50259], [2, 50359], [3, 50363]], 'init_std': 0.02, 'is_encoder_decoder': True, 'max_length': 448, 'max_source_positions': 1500, 'max_target_positions': 448, 'model_type': 'whisper', 'num_hidden_layers': 12, 'num_mel_bins': 80, 'pad_token_id': 50257, 'scale_embedding': False, 'suppress_tokens': [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362], 'torch_dtype': 'float32', 'transformers_version': '4.27.0.dev0', 'use_cache': True, 'vocab_size': 51865, '_commit_hash': '973afd24965f72e36ca33b3055d56a652f456b4d', 'attn_implementation': None}"
```

WhisperConfig的构造函数的参数如下(在线文档在[这里](https://huggingface.co/docs/transformers/model_doc/whisper))：

* vocab_size (int, optional, defaults to 51865) — Vocabulary size of the Whisper model. Defines the number of different tokens that can be represented by the decoder_input_ids passed when calling WhisperModel
* num_mel_bins (int, optional, defaults to 80) — Number of mel features used per input features. Should correspond to the value used in the WhisperProcessor class.
* encoder_layers (int, optional, defaults to 4) — Number of encoder layers.
* decoder_layers (int, optional, defaults to 4) — Number of decoder layers.
* encoder_attention_heads (int, optional, defaults to 6) — Number of attention heads for each attention layer in the Transformer encoder.
* decoder_attention_heads (int, optional, defaults to 6) — Number of attention heads for each attention layer in the Transformer decoder.
* encoder_ffn_dim (int, optional, defaults to 1536) — Dimensionality of the “intermediate” (often named feed-forward) layer in encoder.
* decoder_ffn_dim (int, optional, defaults to 1536) — Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
* encoder_layerdrop (float, optional, defaults to 0.0) — The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more details.
* decoder_layerdrop (float, optional, defaults to 0.0) — The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more details.
* decoder_start_token_id (int, optional, defaults to 50257) — Corresponds to the ”<|startoftranscript|>” token, which is automatically used when no decoder_input_ids are provided to the generate function. It is used to guide the model`s generation process depending on the task.
* use_cache (bool, optional, defaults to True) — Whether or not the model should return the last key/values attentions (not used by all models).
* is_encoder_decoder (bool, optional, defaults to True) — Whether the model is used as an encoder/decoder or not.
* activation_function (str, optional, defaults to "gelu") — The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu", "silu" and "gelu_new" are supported.
* d_model (int, optional, defaults to 384) — Dimensionality of the layers.
* dropout (float, optional, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* attention_dropout (float, optional, defaults to 0.0) — The dropout ratio for the attention probabilities.
* activation_dropout (float, optional, defaults to 0.0) — The dropout ratio for activations inside the fully connected layer.
* init_std (float, optional, defaults to 0.02) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
* scale_embedding (bool, optional, defaults to False) — Scale embeddings by diving by sqrt(d_model).
* max_source_positions (int, optional, defaults to 1500) — The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
* max_target_positions (int, optional, defaults to 448) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
* pad_token_id (int, optional, defaults to 50256) — Padding token id.
* bos_token_id (int, optional, defaults to 50256) — Begin of stream token id.
* eos_token_id (int, optional, defaults to 50256) — End of stream token id.
* suppress_tokens (List[int], optional) — A list containing the non-speech tokens that will be used by the logit processor in the generate function. NON_SPEECH_TOKENS and NON_SPEECH_TOKENS_MULTI each correspond to the english-only and the multilingual model.
* begin_suppress_tokens (List[int], optional, defaults to [220,50256]) — A list containing tokens that will be supressed at the beginning of the sampling process. Initialized as the token for " " (blank_token_id) and the eos_token_id
* use_weighted_layer_sum (bool, optional, defaults to False) — Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an instance of WhisperForAudioClassification.
* classifier_proj_size (int, optional, defaults to 256) — Dimensionality of the projection before token mean-pooling for classification. Only relevant when using an instance of WhisperForAudioClassification.
* apply_spec_augment (bool, optional, defaults to False) — Whether to apply SpecAugment data augmentation to the outputs of the feature encoder. For reference see SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.
* mask_time_prob (float, optional, defaults to 0.05) — Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking procecure generates mask_time_prob*len(time_axis)/mask_time_length independent masks over the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector span to be masked, mask_time_prob should be prob_vector_start*mask_time_length. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if apply_spec_augment == True.
* mask_time_length (int, optional, defaults to 10) — Length of vector span along the time axis.
* mask_time_min_masks (int, optional, defaults to 2), — The minimum number of masks of length mask_feature_length generated along the time axis, each time step, irrespectively of mask_feature_prob. Only relevant if ”mask_time_prob*len(time_axis)/mask_time_length < mask_time_min_masks”
* mask_feature_prob (float, optional, defaults to 0.0) — Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The masking procecure generates mask_feature_prob*len(feature_axis)/mask_time_length independent masks over the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector span to be masked, mask_feature_prob should be prob_vector_start*mask_feature_length. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if apply_spec_augment is True.
* mask_feature_length (int, optional, defaults to 10) — Length of vector span along the feature axis.
* mask_feature_min_masks (int, optional, defaults to 0), — The minimum number of masks of length mask_feature_length generated along the feature axis, each time step, irrespectively of mask_feature_prob. Only relevant if mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks.
* median_filter_width (int, optional, defaults to 7) — Width of the median filter used to smoothen to cross-attention outputs when computing token timestamps. Should be an odd number.


####  check_task函数

这个函数就是根据task='automatic-speech-recognition'查找对应的targeted_task，并且最终根据框架(pytorch)得到model_classes，我这里的model_classes是：

```
{'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForCTC'>, <class 'transformers.models.auto.modeling_auto.AutoModelForSpeechSeq2Seq'>)}
```

接着调用infer_framework_load_model真正执行模型的加载。因为这个函数比较复杂，我们详细来展开它。



### infer_framework_load_model

这个函数的主要代码是：

```
        for model_class in class_tuple:

            try:
                model = model_class.from_pretrained(model, **kwargs)
                if hasattr(model, "eval"):
                    model = model.eval()
                # Stop loading on the first successful load.
                break
            except (OSError, ValueError):
                all_traceback[model_class.__name__] = traceback.format_exc()
                continue
```

class_tuple来自传入参数model_classes以及配置config.architectures=WhisperForConditionalGeneration获得。最终class_tuple的值为：

```
(<class 'transformers.models.auto.modeling_auto.AutoModelForCTC'>, <class 'transformers.models.auto.modeling_auto.AutoModelForSpeechSeq2Seq'>, <class 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'>)
```

然后分别尝试用这3个类的from_pretrained函数来加载模型。我这里实际的运行情况是AutoModelForCTC抛出异常，因为whisper不是ctc架构的模型，而第二个AutoModelForSpeechSeq2Seq能够成功加载模型。它的代码是：

```
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
```

通过config，查找到对应的model_class是'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'，最终调用这个类的from_pretrained来加载模型。

### WhisperForConditionalGeneration.from_pretrained

WhisperForConditionalGeneration继承了WhisperGenerationMixin和WhisperPreTrainedModel，WhisperPreTrainedModel最终继承了ModuleUtilsMixin，而from_pretrained最终来自ModuleUtilsMixin：

```
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                ......
                
                state_dict = load_state_dict(resolved_archive_file)
                
        init_contexts = [no_init_weights(_enable=_fast_init)]
        
        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)
            
            
            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                loaded_state_dict_keys,  # XXX: rename?
                resolved_archive_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                _fast_init=_fast_init,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=torch_dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_modules=keep_in_fp32_modules,
            )
       
        .....     
        model.eval()
        
                ...
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
```

resolved_archive_file的值为：

```
'...../.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/model.safetensors'
```

加载后的state_dict为：

```
{'model.decoder.embed_positions.weight': tensor([[ 0.0043,  0.0010,  0.0108,  ..., -0.0046, -0.0061,  0.0013],
        [ 0.0039,  0.0044,  0.0034,  ..., -0.0047, -0.0080,  0.0058],
        [ 0.0086,  0.0028,  0.0057,  ..., -0.0031, -0.0086,  0.0086],
        ...,
        [ 0.0104,  0.0073,  0.0066,  ...,  0.0104,  0.0018,  0.0017],
        [ 0.0061,  0.0015,  0.0047,  ...,  0.0060,  0.0033,  0.0067],
        [ 0.0069,  0.0108,  0.0026,  ...,  0.0034, -0.0023,  0.0089]]), 'model.decoder.embed_tokens.weight': tensor([[ 0.0055, -0.0057,  0.0133,  ...,  0.0027, -0.0191, -0.0129],
        [ 0.0065, -0.0289,  0.0112,  ...,  0.0035,  0.0122, -0.0312],
        [ 0.0014,  0.0012,  0.0141,  ...,  0.0036,  0.0384, -0.0168],
        ...,
        [ 0.0081,  0.0034,  0.0134,  ...,  0.0182,  0.0291, -0.0009],
        [ 0.0068,  0.0056,  0.0108,  ...,  0.0202,  0.0307,  0.0130],
        [ 0.0050,  0.0032, -0.0041,  ...,  0.0105,  0.0403,  0.0129]]), 'model.decoder.layer_norm.bias': tensor([-6.3904e-02,  2.0984e-01,  6.5283e-01, -8.9160e-01,  1.2817e-02,
        -1.0321e-01, -5.3809e-01,  3.2471e-01, -1.2439e-01,  5.6213e-02,
        -1.2805e-01, -2.6294e-01, -9.9060e-02, -5.9082e-01, -1.3025e-01,
         9.2590e-02,  6.3538e-02,  4.1479e-01,  1.5820e-01, -5.3564e-01,

           ...........................

         7.7100e-01,  6.7139e-01,  7.1777e-01,  5.4541e-01,  7.2217e-01,
         7.7881e-01,  4.7974e-01,  5.1611e-01,  4.7266e-01,  3.2544e-01,
         4.5996e-01,  5.2832e-01,  6.5381e-01]), 'model.decoder.layers.0.fc1.bias': tensor([-0.7275, -0.1436, -0.5371,  ..., -0.3271, -0.6587, -0.5327]), ...}
```

no_init_weights的作用是构造网络时不进行参数初始化，等到后面再读取模型参数。model = cls(config, *model_args, **model_kwargs)进入到transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration的构造函数。

### WhisperForConditionalGeneration

```
class WhisperForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
····
```

这个类的构造函数主要是构造WhisperModel实例model，接着构造输出proj_out(config.d_model=768, config.vocab_size=51865)。

### WhisperModel

```
class WhisperModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()
```


### WhisperEncoder

```
class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
```

WhisperEncoder包括两个卷积层(conv1和conv2)，然后就是config.encoder_layers=12个WhisperEncoderLayer。另外就是一个layer_norm。

### WhisperEncoderLayer

```
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
```

transformers有很多Attention的实现，这里config._attn_implementation='spda'，所以对应的是WhisperSdpaAttention类。后面我们会在forward时学习其实现。其它都是比较标准的Transformer构件，包括LayerNorm和全连接层。

### WhisperDecoder

```
class WhisperDecoder(WhisperPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
```

WhisperDecoder和WhisperEncoder差不多，主要是多层WhisperDecoderLayer。

### WhisperDecoderLayer

```
class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
```

Decoder有两种Attention，一种还是Self Attention，但是注意参数is_decoder=True和is_causal=True，这会导致它的mask有所不同。另外就是encoder_attn，它是decoder对encoder输出的attention。

### PreTrainedModel._load_pretrained_model

```
        model.tie_weights()
        
        ...
        if _fast_init:
            if not ignore_mismatched_sizes:
                not_initialized_submodules = set_initialized_submodules(model, _loaded_keys)
                
                
                
                ...     
                model.apply(model._initialize_weights)
                
            ...
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
            
```


#### PreTrainedModel.tie_weights


```
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()
```

因为条件getattr(self.config, "tie_word_embeddings", True)成立，所以会调用_tie_or_clone_weights，其中：

* output_embeddings是decoder的proj_out，也就是decoder的输出embedding
* self.get_input_embeddings()是decoder.embed_tokens，也就是输入token的embedding。

```
    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
```

#### set_initialized_submodules

```
def set_initialized_submodules(model, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    not_initialized_submodules = {}
    for module_name, module in model.named_modules():
        loaded_keys = {k.replace(f"{module_name}.", "") for k in state_dict_keys if k.startswith(f"{module_name}.")}
        if loaded_keys.issuperset(module.state_dict()):
            module._is_hf_initialized = True
        else:
            not_initialized_submodules[module_name] = module
    return not_initialized_submodules
```

这个函数把遍历model的所有模块(named_modules)，如果这些模块的所以参数都在state_dict_keys里了，那么说明这些模块的参数已经加载，因此设置module._is_hf_initialized = True。

我这里运行时not_initialized_submodules为：

```
{'': WhisperForConditionalGeneration(
  (model): WhisperModel(
    (encoder): WhisperEncoder(
      (conv1): Conv1d(80, 768, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv2): Conv1d(768, 768, kernel_size=(3,), stride=(2,), padding=(1,))
      (embed_positions): Embedding(1500, 768)
      (layers): ModuleList(
        (0-11): 12 x WhisperEncoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=False)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): WhisperDecoder(
      (embed_tokens): Embedding(51865, 768, padding_idx=50257)
      (embed_positions): WhisperPositionalEmbedding(448, 768)
      (layers): ModuleList(
        (0-11): 12 x WhisperDecoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=False)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (activation_fn): GELUActivation()
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=False)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
  )
  (proj_out): Linear(in_features=768, out_features=51865, bias=False)
), 'proj_out': Linear(in_features=768, out_features=51865, bias=False)}
```

而proj_out是tie到decoder的input embedding上的，因此也会设置_is_hf_initialized为True：


```
                if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        # Still need to initialize if there is a bias term since biases are not tied.
                        if not hasattr(output_embeddings, "bias") or output_embeddings.bias is None:
                            output_embeddings._is_hf_initialized = True
```

is_hf_initialized为True，表明模型的参数已经完成初始化。

#### PreTrainedModel._initialize_weights

```
    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True
```

如果某个module的_is_hf_initialized是True，那么就说明参数以及初始化过了，不需要在这里初始化。因此我们这里唯一需要初始化的就是WhisperForConditionalGeneration，它最终会调用下面WhisperPreTrainedModel的_init_weights。【而WhisperForConditionalGeneration只是包含了WhisperModel和proj_out，这些都已经初始化过了，后面的代码并不需要初始化什么参数。】

#### WhisperPreTrainedModel._init_weights

```
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperEncoder):
            with torch.no_grad():
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))
```

我们这里传入的module是WhisperForConditionalGeneration，上面的条件都不满足，因此什么也没有做。

#### _load_state_dict_into_model函数

这个函数除了处理一下特殊的state_dict的key(把gamma和beta变成weight和bias)，最终调用的是load函数。load函数也是在这个函数里面定义的。


#### load函数

```
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")
```

这个函数最终调用的也是pytorch的_load_from_state_dict，只不过因为pytorch的_load_from_state_dict不会递归的加载子模块，因此load函数会递归的对子模块调用load。至此，模型的参数就加载完毕。

#### GenerationConfig.from_pretrained

```
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "GenerationConfig":
    

                resolved_config_file = cached_file(
                    pretrained_model_name,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                
                
```


resolved_config_file是'.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/generation_config.json'

这个json加载后的内容是：

```
{'alignment_heads': [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], 'begin_suppress_tokens': [220, 50257], 'bos_token_id': 50257, 'decoder_start_token_id': 50258, 'eos_token_id': 50257, 'forced_decoder_ids': [[...], [...]], 'is_multilingual': True, 'lang_to_id': {'<|af|>': 50327, '<|am|>': 50334, '<|ar|>': 50272, '<|as|>': 50350, '<|az|>': 50304, '<|ba|>': 50355, '<|be|>': 50330, '<|bg|>': 50292, '<|bn|>': 50302, ...}, 'max_initial_timestamp_index': 50, 'max_length': 448, 'no_timestamps_token_id': 50363, 'pad_token_id': 50257, 'prev_sot_token_id': 50361, 'return_timestamps': False, ...}
```


### Tokenizer的加载

代码在AutoTokenizer.from_pretrained：

```
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        
            ...
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)        
```

config_tokenizer_class的值是'WhisperTokenizer'。tokenizer_class是'transformers.models.whisper.tokenization_whisper_fast.WhisperTokenizerFast'。

#### get_tokenizer_config

```
    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        _commit_hash=commit_hash,
    )
```

resolved_config_file文件是'.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/tokenizer_config.json'，它的内容是：

```
{'add_bos_token': False, 'add_prefix_space': False, 'added_tokens_decoder': {'50257': {...}, '50258': {...}, '50259': {...}, '50260': {...}, '50261': {...}, '50262': {...}, '50263': {...}, '50264': {...}, '50265': {...}, ...}, 'additional_special_tokens': ['<|endoftext|>', '<|startoftranscript|>', '<|en|>', '<|zh|>', '<|de|>', '<|es|>', '<|ru|>', '<|ko|>', '<|fr|>', ...], 'bos_token': '<|endoftext|>', 'clean_up_tokenization_spaces': True, 'eos_token': '<|endoftext|>', 'errors': 'replace', 'model_max_length': 1024, 'pad_token': '<|endoftext|>', 'processor_class': 'WhisperProcessor', 'return_attention_mask': False, 'tokenizer_class': 'WhisperTokenizer', 'unk_token': '<|endoftext|>'}
```


#### PreTrainedTokenizerBase.from_pretrained 

WhisperTokenizerFast.from_pretrained最终会调用PreTrainedTokenizerBase.from_pretrained。

```
            additional_files_names = {
                "added_tokens_file": ADDED_TOKENS_FILE,  # kept only for legacy
                "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,  # kept only for legacy
                "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
                "tokenizer_file": FULL_TOKENIZER_FILE,
            }
            vocab_files = {**cls.vocab_files_names, **additional_files_names}
            if "tokenizer_file" in vocab_files:
                # Try to get the tokenizer config to see if there are versioned tokenizer files.
                fast_tokenizer_file = FULL_TOKENIZER_FILE
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    TOKENIZER_CONFIG_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
                if resolved_config_file is not None:
                    with open(resolved_config_file, encoding="utf-8") as reader:
                        tokenizer_config = json.load(reader)
                        if "fast_tokenizer_files" in tokenizer_config:
                            fast_tokenizer_file = get_fast_tokenizer_file(tokenizer_config["fast_tokenizer_files"])
                vocab_files["tokenizer_file"] = fast_tokenizer_file
                
        ...
        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=commit_hash,
            _is_local=is_local,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
```

最终调用cls(<class 'transformers.models.whisper.tokenization_whisper_fast.WhisperTokenizerFast'>)的_from_pretrained方法，并且传入的参数resolved_vocab_files为：

```
{'vocab_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/vocab.json', 'tokenizer_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/tokenizer.json', 'merges_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/merges.txt', 'normalizer_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/normalizer.json', 'added_tokens_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/added_tokens.json', 'special_tokens_map_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/special_tokens_map.json', 'tokenizer_config_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/tokenizer_config.json'}
```


#### PreTrainedTokenizerBase._from_pretrained

首先是确定各种词典文件的名称和路径：

```
            additional_files_names = {
                "added_tokens_file": ADDED_TOKENS_FILE,  # kept only for legacy
                "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,  # kept only for legacy
                "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
                "tokenizer_file": FULL_TOKENIZER_FILE,
            }
            vocab_files = {**cls.vocab_files_names, **additional_files_names}
            if "tokenizer_file" in vocab_files:
                # Try to get the tokenizer config to see if there are versioned tokenizer files.
                fast_tokenizer_file = FULL_TOKENIZER_FILE
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    TOKENIZER_CONFIG_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
                if resolved_config_file is not None:
                    with open(resolved_config_file, encoding="utf-8") as reader:
                        tokenizer_config = json.load(reader)
                        if "fast_tokenizer_files" in tokenizer_config:
                            fast_tokenizer_file = get_fast_tokenizer_file(tokenizer_config["fast_tokenizer_files"])
                vocab_files["tokenizer_file"] = fast_tokenizer_file

        # Get files from url, cache, or disk depending on the case
        resolved_vocab_files = {}
        unresolved_files = []
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            elif single_file_id == file_id:
                if os.path.isfile(file_path):
                    resolved_vocab_files[file_id] = file_path
                elif is_remote_url(file_path):
                    resolved_vocab_files[file_id] = download_url(file_path, proxies=proxies)
            else:
                resolved_vocab_files[file_id] = cached_file(
                    pretrained_model_name_or_path,
                    file_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_vocab_files[file_id], commit_hash)

```


vocab_files为：
```
{'vocab_file': 'vocab.json', 'tokenizer_file': 'tokenizer.json', 'merges_file': 'merges.txt', 'normalizer_file': 'normalizer.json', 'added_tokens_file': 'added_tokens.json', 'special_tokens_map_file': 'special_tokens_map.json', 'tokenizer_config_file': 'tokenizer_config.json'}
```

resolved_vocab_files为：
```
{'vocab_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/vocab.json', 'tokenizer_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/tokenizer.json', 'merges_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/merges.txt', 'normalizer_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/normalizer.json', 'added_tokens_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/added_tokens.json', 'special_tokens_map_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/special_tokens_map.json', 'tokenizer_config_file': '.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/tokenizer_config.json'}
```

然后调用：

```
        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=commit_hash,
            _is_local=is_local,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
```

这里的cls为<class 'transformers.models.whisper.tokenization_whisper_fast.WhisperTokenizerFast'>。但是最后还是调用到PreTrainedTokenizerBase._from_pretrained。


#### PreTrainedTokenizerBase._from_pretrained

我们逐段来看看：

```
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            # First attempt. We get tokenizer_class from tokenizer_config to check mismatch between tokenizers.
            config_tokenizer_class = init_kwargs.get("tokenizer_class")
            init_kwargs.pop("tokenizer_class", None)
            if not has_tokenizer_file:
                init_kwargs.pop("tokenizer_file", None)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
```

上面的代码根据tokenizer_config_file的配置，找到config_tokenizer_class为'WhisperTokenizer'。


```
        if "added_tokens_decoder" in init_kwargs:
            for idx, token in init_kwargs["added_tokens_decoder"].items():
                if isinstance(token, dict):
                    token = AddedToken(**token)
                if isinstance(token, AddedToken):
                    added_tokens_decoder[int(idx)] = token
                    added_tokens_map[str(token)] = token
                else:
                    raise ValueError(
                        f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                    )
```

接着除了decoder增加的token(added_tokens_decoder)。比如一个例子是：

```
AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True)
```

这个token的id是50527，token的内容是"\<\|endoftext\|\>"。


最后调用cls的构造函数：

```
            tokenizer = cls(*init_inputs, **init_kwargs)
```

#### WhisperTokenizerFast的构造函数


```
class WhisperTokenizerFast(PreTrainedTokenizerFast): 

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = WhisperTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        normalizer_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    ):
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )

        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.add_bos_token = kwargs.pop("add_bos_token", False)

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            self.english_spelling_normalizer = None

        self.add_prefix_space = add_prefix_space
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")

        self.language = language
        self.task = task
        self.predict_timestamps = predict_timestamps
```

normalizer_file是归一化的文件(.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/normalizer.json)，加载后self.english_spelling_normalizer为：

```
{'accessorise': 'accessorize', 'accessorised': 'accessorized', 'accessorises': 'accessorizes', 'accessorising': 'accessorizing', 'acclimatisation': 'acclimatization', 'acclimatise': 'acclimatize', 'acclimatised': 'acclimatized', 'acclimatises': 'acclimatizes', 'acclimatising': 'acclimatizing', 'accoutrements': 'accouterments', 'aeon': 'eon', 'aeons': 'eons', 'aerogramme': 'aerogram', 'aerogrammes': 'aerograms', ...}
```

### AutoFeatureExtractor.from_pretrained

因为asr的输入是语音，所以pipeline会用AutoFeatureExtractor来加载FeatureExtractor。

```
        config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
        ...
        if feature_extractor_class is not None:
            feature_extractor_class = feature_extractor_class_from_name(feature_extractor_class)
            
            ...
            return feature_extractor_class.from_dict(config_dict, **kwargs)
```

config_dict为：
```
{'chunk_length': 30, 'feature_extractor_type': 'WhisperFeatureExtractor', 'feature_size': 80, 'hop_length': 160, 'mel_filters': [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], 'n_fft': 400, 'n_samples': 480000, 'nb_max_frames': 3000, 'padding_side': 'right', 'padding_value': 0.0, 'processor_class': 'WhisperProcessor', 'return_attention_mask': False, 'sampling_rate': 16000}
```

从这里我们可以知道whisper每次处理的chunk长度为30秒(chunk_length)，具体的FeatureExtractor为WhisperFeatureExtractor，特征的维度为80(melfilter的个数)，窗口长度为400个点(25ms)，窗口每次滑动160个点(10ms)。这都是asr比较标准的输入处理方式。

feature_extractor_class为<class 'transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor'>。


#### FeatureExtractionMixin.from_dict

```
    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor
```

最终还是调用cls(WhisperFeatureExtractor)的构造函数：

#### WhisperFeatureExtractor构造函数

```
class WhisperFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
```

这个函数就是初始化一下参数，最后是构造mel_filter_bank。

#### mel_filter_bank


```
def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
) -> np.ndarray:
    """
    Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
    various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
    are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
    features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.

    Different banks of mel filters were introduced in the literature. The following variations are supported:

    - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech
      bandwidth of `[0, 4600]` Hz.
    - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech
      bandwidth of `[0, 8000]` Hz. This assumes sampling rate ≥ 16 kHz.
    - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz and
      speech bandwidth of `[133, 6854]` Hz. This version also includes area normalization.
    - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes a sampling rate of
      12.5 kHz and speech bandwidth of `[0, 6250]` Hz.

    This code is adapted from *torchaudio* and *librosa*. Note that the default parameters of torchaudio's
    `melscale_fbanks` implement the `"htk"` filters while librosa uses the `"slaney"` implementation.

    Args:
        num_frequency_bins (`int`):
            Number of frequencies used to compute the spectrogram (should be the same as in `stft`).
        num_mel_filters (`int`):
            Number of mel filters to generate.
        min_frequency (`float`):
            Lowest frequency of interest in Hz.
        max_frequency (`float`):
            Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
        sampling_rate (`int`):
            Sample rate of the audio waveform.
        norm (`str`, *optional*):
            If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.
        triangularize_in_mel_space (`bool`, *optional*, defaults to `False`):
            If this option is enabled, the triangular filter is applied in mel space rather than frequency space. This
            should be set to `true` in order to get the same results as `torchaudio` when computing mel filters.

    Returns:
        `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`): Triangular filter bank matrix. This is a
        projection matrix to go from a spectrogram to a mel spectrogram.
    """
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # frequencies of FFT bins in Hz, but filters triangularized in mel space
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # frequencies of FFT bins in Hz
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )

    return mel_filters
```

关于mel_filter_bank不做深入介绍，感兴趣的读者可以查找相关资料。

### AutomaticSpeechRecognitionPipeline

```
class AutomaticSpeechRecognitionPipeline(ChunkPipeline): 

    def __init__(
        self,
        model: "PreTrainedModel",
        feature_extractor: Union["SequenceFeatureExtractor", str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        decoder: Optional[Union["BeamSearchDecoderCTC", str]] = None,
        device: Union[int, "torch.device"] = None,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        # set the model type so we can check we have the right pre- and post-processing parameters
        if model.config.model_type == "whisper":
            self.type = "seq2seq_whisper"
        elif model.__class__.__name__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values():
            self.type = "seq2seq"
        elif (
            feature_extractor._processor_class
            and feature_extractor._processor_class.endswith("WithLM")
            and decoder is not None
        ):
            self.decoder = decoder
            self.type = "ctc_with_lm"
        else:
            self.type = "ctc"

        super().__init__(model, tokenizer, feature_extractor, device=device, torch_dtype=torch_dtype, **kwargs)

```
