---
layout:     post
title:      "Tensor2Tensor中英翻译代码阅读(五)" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - TensorFlow
    - Tensor2Tensor
    - t2t
    - 机器翻译
    - 代码阅读
    - 中英翻译
---

本系列文章是Tensor2Tensor的代码阅读，主要关注中英翻译的实现。本文是第五篇，继续介绍预测解码的代码。

<!--more-->

**目录**
* TOC
{:toc}

## 解码

预测解码的代码为：
```
t2t_problem = problems.problem(PROBLEM)

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

hparams = create_hparams(HPARAMS, data_dir=DATA_DIR, problem_name=PROBLEM)
translate_model = registry.model(MODEL)(hparams, Modes.PREDICT)

# Get the encoders (fixed pre-processing) from the problem
encoders = t2t_problem.feature_encoders(DATA_DIR)

def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["targets"].decode(np.squeeze(integers))

# Get the latest checkpoint
ckpt_path = tf.train.latest_checkpoint(TRAIN_DIR)
print('Latest Checkpoint: ', ckpt_path)


def translate(inputs):
  encoded_inputs = encode(inputs)
  with tfe.restore_variables_on_create(ckpt_path):
    model_output = translate_model.infer(encoded_inputs)["outputs"]
  return decode(model_output)

inputs = ["I think they will never come back to the US.",
          "Human rights is the first priority.",
          "Everyone should have health insurance.",
          "President Trump's overall approval rating dropped 7% over the past month"]

for sentence in inputs:
  output = translate(sentence)
  print("\33[34m Inputs:\33[30m %s" % sentence)
  print("\033[35m Outputs:\33[30m %s" % output)
  print()
```

首先构造模型：
```
hparams = create_hparams(HPARAMS, data_dir=DATA_DIR, problem_name=PROBLEM)
translate_model = registry.model(MODEL)(hparams, Modes.PREDICT)
```
进入Transformer的构造函数：
```
@registry.register_model
class Transformer(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(Transformer, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # For visualizing attention heads.
    self.recurrent_memory_by_layer = None  # Override to enable recurrent memory
    self._encoder_function = transformer_encoder
    self._decoder_function = transformer_decoder
    self._init_cache_fn = _init_transformer_cache
    self._prepare_encoder_fn = transformer_prepare_encoder
    self._prepare_decoder_fn = transformer_prepare_decoder
```
然后进入基类的构造函数：
```
def __init__(self,
             hparams,
             mode=tf.estimator.ModeKeys.TRAIN,
             problem_hparams=None,
             data_parallelism=None,
             decode_hparams=None,
             **kwargs):
  """Creates a T2TModel.

  Args:
    hparams: HParams, model hyperparameters.
    mode: tf.estimator.ModeKeys, the execution mode.
    problem_hparams: HParams, hyperparameters for the
      Problem. If provided here or in hparams.problem_hparams, the model will
      automatically determine bottom, top, and loss methods. If not provided,
      calling the model will only invoke body.
    data_parallelism: a expert_utils.Parallelism object,
      specifies devices for data parallelism.
    decode_hparams: a hyperparameter object with decoding parameters.
      See decoding.decode_hparams.
    **kwargs: arguments to pass to base.Layer constructor.
  """
  # Determine name first: use registered name if possible, class name else.
  default_name = registry.default_name(type(self))
  name = self.REGISTERED_NAME or default_name
  super(T2TModel, self).__init__(
      trainable=mode == tf.estimator.ModeKeys.TRAIN, name=name, **kwargs)

  if not problem_hparams and hasattr(hparams, "problem_hparams"):
    problem_hparams = hparams.problem_hparams
  self._problem_hparams = problem_hparams

  # Setup hparams
  hparams = hparams_lib.copy_hparams(hparams)
  if self._problem_hparams and hparams.shared_embedding_and_softmax_weights:
    # If vocabularies differ, unset shared_embedding_and_softmax_weights.
    input_vocab_size = self._problem_hparams.vocab_size.get("inputs")
    target_vocab_size = self._problem_hparams.vocab_size.get("targets")
    if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
    if (input_vocab_size is not None and target_vocab_size is not None and
        input_vocab_size != target_vocab_size):
      log_info("Unsetting shared_embedding_and_softmax_weights.")
      hparams.shared_embedding_and_softmax_weights = 0

    if hparams.hidden_size:
      hidden_size = hparams.hidden_size
    else:
      hidden_size = 1024
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_EMBEDDING_SHARED_WEIGHTS,
        value={
            "vocab_size": target_vocab_size,
            "hidden_size": hidden_size
        },
        hparams=hparams)

  if self._problem_hparams:
    for feature_name, modality in six.iteritems(
        self._problem_hparams.modality):
      # If prepend mode, set weights_fn to appropriately handle it.
      if (modality in (modalities.ModalityType.CTC_SYMBOL,
                       modalities.ModalityType.IDENTITY_SYMBOL,
                       modalities.ModalityType.SYMBOL,
                       modalities.ModalityType.SYMBOL_ONE_HOT)):
        if (hparams.prepend_mode == "prepend_inputs_full_attention" or
            (hparams.prepend_mode == "prepend_inputs_masked_attention" and
             mode != tf.estimator.ModeKeys.TRAIN)):
          weights_fn = common_layers.weights_prepend_inputs_to_targets
          hparams.weights_fn[feature_name] = weights_fn

  self._original_hparams = hparams
  self.set_mode(mode)

  self._decode_hparams = hparams_lib.copy_hparams(
      decode_hparams or decoding.decode_hparams())
  self._data_parallelism = data_parallelism or eu.Parallelism([""])
  self._num_datashards = self._data_parallelism.n
  self._ps_devices = self._data_parallelism.ps_devices
  self._eager_var_store = create_eager_var_store()
  if not common_layers.is_xla_compiled():
    self.summarize_hparams()
  self._variable_scopes = {}
```
代码会读取input_vocab_size(8182)、target_vocab_size(8267)和hidden_size(512)等模型超参数。
返回到Transformer的构造函数后就是设置_encoder_function、_decoder_function等函数。

接着是通过TranslateEnzhWmt8k(TranslateProblem)的子类得到encoder：
```
encoders = t2t_problem.feature_encoders(DATA_DIR)

def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }
```
就是通过两个词典文件得到source和target的SubwordTextEncoder。这部分代码前面已经分析过了，这里略过。

然后就是翻译：
```
output = translate(sentence)
```
代码为：
```
def translate(inputs):
  encoded_inputs = encode(inputs)
  with tfe.restore_variables_on_create(ckpt_path):
    model_output = translate_model.infer(encoded_inputs)["outputs"]
  return decode(model_output)
```
首先是encode函数，把字符串变成subword的id：
```
def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D
  return {"inputs": batch_inputs}
```
首先是用source的encoder进行encode，然后加上一个EOS(1)。比如输入是"I think they will never come back to the US."，encode后变成<class 'list'>: [134, 819, 47, 31, 451, 377, 323, 6, 3, 46, 4, 1]，我们可以用词典(vocab.translate_enzh_wmt8k.8192.subwords.en)检验一下：
134对应的确实是'I_'这个subword，下划线结尾表示这是到一个词的结尾，不清楚的读者可以参考前面的部分。
接着：
```
with tfe.restore_variables_on_create(ckpt_path):
  model_output = translate_model.infer(encoded_inputs)["outputs"]
```
首先是通过check point恢复模型参数。restore_variables_on_create是saver.py里的函数：
```
@contextlib.contextmanager
def restore_variables_on_create(save_path, map_func=None):
  """ContextManager that restores variables on creation.

    When save_path is None (e.g. No checkpoint), does nothing.
    Otherwise, it preloads all values from checkpoint. When the
    corresponding variable is first created, it assigns the checkpoint
    value to the variable.

    ```python
    with restore_variables_on_create(
        tf.train.latest_checkpoint(checkpoint_dir)):
    ```

  Args:
    save_path: The checkpoint file prefix.
    map_func: A function that given the variable name as argument
        and returns a variable name in checkpoint for restore. If
        None, use the variable with the same name in checkpoint to restore.
        It's an error that the mapped variable name doesn't exist in
        checkpoint.

  Yields:
    Nothing.

  Raises:
    NotFoundError: If the variable is not found in checkpoint.
    ValueError: If not used in eager mode or map_func is not callable.
  """
  if not context.executing_eagerly():
    raise ValueError(
        "Currently, restore_variables_on_create can only be used with "
        "eager execution enabled.")
  if save_path:
    if map_func is None:
      map_func_wrapper = lambda self, x: x
    else:
      if not callable(map_func):
        raise ValueError("map_func must be callable.")
      map_func_wrapper = lambda self, x: map_func(x)

    ckpt_var_cache = {}
    reader = checkpoint_utils.load_checkpoint(save_path)
    for k, _ in checkpoint_utils.list_variables(save_path):
      ckpt_var_cache[k] = reader.get_tensor(k)

    old_init = getattr(resource_variable_ops.ResourceVariable,
                       "_init_from_args", None)
    assert old_init, "ResourceVariable misses _init_from_args method."
    setattr(resource_variable_ops.ResourceVariable, "_init_from_args",
            _init_from_checkpoint)
    setattr(resource_variable_ops.ResourceVariable, "_old_init", old_init)
    setattr(resource_variable_ops.ResourceVariable, "_map_func",
            map_func_wrapper)
    setattr(resource_variable_ops.ResourceVariable, "_ckpt_var_cache",
            ckpt_var_cache)
  try:
    yield
  except Exception as e:
    raise e
  finally:
    if save_path:
      setattr(resource_variable_ops.ResourceVariable, "_init_from_args",
              old_init)
      setattr(resource_variable_ops.ResourceVariable, "_old_init", None)
      setattr(resource_variable_ops.ResourceVariable, "_map_func", None)
      setattr(resource_variable_ops.ResourceVariable, "_ckpt_var_cache", None)

```
核心代码是读取参数：
```
reader = checkpoint_utils.load_checkpoint(save_path)
for k, _ in checkpoint_utils.list_variables(save_path):
  ckpt_var_cache[k] = reader.get_tensor(k)
``` 


<a name='img10'>![](/img/t2t-code/10.png)</a>
*图：ckpt_var_cache*

然后我们来看最关键的infer函数：
```
model_output = translate_model.infer(encoded_inputs)["outputs"]

def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """A inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: bool, whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
      if slow greedy decoding is used then the dict will also contain {
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`
      }
    """
    set_custom_getter_compose(self._custom_getter)
    with self._eager_var_store.as_default():
      # TODO(rsepassi): Make decoding work with real-valued model outputs
      # (i.e. if the target modality is RealModality).
      self.prepare_features_for_infer(features)
      if not self.has_input and beam_size > 1:
        log_warn("Beam searching for a model with no inputs.")
      if not self.has_input and self.hparams.sampling_method != "random":
        log_warn("Non-random sampling for a model with no inputs.")
      self._fill_problem_hparams_features(features)

      if self._problem_hparams:
        target_modality = self._problem_hparams.modality["targets"]
      if (target_modality == modalities.ModalityType.CLASS_LABEL or
          self._problem_hparams.get("regression_targets")):
        # No use to run beam-search for classification or regression.
        beam_size = 1
      if beam_size == 1:
        log_info("Greedy Decoding")
        results = self._greedy_infer(features, decode_length, use_tpu)
      else:
        log_info("Beam Decoding with beam size %d" % beam_size)
        results = self._beam_decode(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)

      return results
```


prepare_features_for_infer什么也不干，不用看。接着是_fill_problem_hparams_features：
它主要通过problem_hparams_to_features设置input_space_id和target_space_id，这里的值都是0。target_modality是'symbol'。
因为我们默认使用的beam_size是1，所以实际调用_greedy_infer：
```
def _greedy_infer(self, features, decode_length, use_tpu=False):
  """Fast version of greedy decoding.

  Args:
    features: an map of string to `Tensor`
    decode_length: an integer.  How many additional timesteps to decode.
    use_tpu: A bool. Whether to build the inference graph for TPU.

  Returns:
    A dict of decoding results {
        "outputs": integer `Tensor` of decoded ids of shape
            [batch_size, <= decode_length] if beam_size == 1 or
            [batch_size, top_beams, <= decode_length]
        "scores": decoding log probs from the beam search,
            None if using greedy decoding (beam_size=1)
    }

  Raises:
    NotImplementedError: If there are multiple data shards.
  """
  # For real-valued modalities use the slow decode path for now.
  if (self._target_modality_is_real or
      self._hparams.self_attention_type != "dot_product"):
    return super(Transformer, self)._greedy_infer(features, decode_length)
  with tf.variable_scope(self.name):
    if use_tpu:
      return self._fast_decode_tpu(features, decode_length)
    return self._fast_decode(features, decode_length)
```

最终调用的是_fast_decode，这个函数的代码非常多：
```
def _fast_decode(self,
                 features,
                 decode_length,
                 beam_size=1,
                 top_beams=1,
                 alpha=1.0,
                 preprocess_targets_method=None):
  """Fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    features: a map of string to model  features.
    decode_length: an integer.  How many additional timesteps to decode.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    preprocess_targets_method: method used to preprocess targets. If None,
    uses method "preprocess_targets" defined inside this method.

  Returns:
    A dict of decoding results {
        "outputs": integer `Tensor` of decoded ids of shape
            [batch_size, <= decode_length] if beam_size == 1 or
            [batch_size, top_beams, <= decode_length]
        "scores": decoding log probs from the beam search,
            None if using greedy decoding (beam_size=1)
    }

  Raises:
    NotImplementedError: If there are multiple data shards.
  """
  if self._num_datashards != 1:
    raise NotImplementedError("Fast decoding only supports a single shard.")
  dp = self._data_parallelism
  hparams = self._hparams
  target_modality = self._problem_hparams.modality["targets"]
  target_vocab_size = self._problem_hparams.vocab_size["targets"]
  if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
    target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
  if "targets_segmentation" in features:
    raise NotImplementedError(
        "Decoding not supported on packed datasets "
        " If you want to decode from a dataset, use the non-packed version"
        " of the dataset when decoding.")
  if self.has_input:
    inputs_shape = common_layers.shape_list(features["inputs"])
    if (target_modality == modalities.ModalityType.CLASS_LABEL or
        self._problem_hparams.get("regression_targets")):
      decode_length = 1
    else:
      decode_length = (
          inputs_shape[1] + features.get("decode_length", decode_length))
    batch_size = inputs_shape[0]
    inputs = self._prepare_inputs_for_decode(features)
    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = dp(
          self.encode,
          inputs,
          features["target_space_id"],
          hparams,
          features=features)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
    partial_targets = features.get("partial_targets")
  else:
    # The problem has no inputs.
    encoder_output = None
    encoder_decoder_attention_bias = None

    # Prepare partial targets.
    # In either features["inputs"] or features["targets"].
    # We force the outputs to begin with these sequences.
    partial_targets = features.get("inputs")
    if partial_targets is None:
      partial_targets = features["targets"]
    assert partial_targets is not None

  if partial_targets is not None:
    partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
    partial_targets = tf.to_int64(partial_targets)
    partial_targets_shape = common_layers.shape_list(partial_targets)
    partial_targets_length = partial_targets_shape[1]
    decode_length = (
        partial_targets_length + features.get("decode_length", decode_length))
    batch_size = partial_targets_shape[0]

  if hparams.pos == "timing":
    positional_encoding = common_attention.get_timing_signal_1d(
        decode_length + 1, hparams.hidden_size)
  elif hparams.pos == "timing_from_features":
    positional_encoding = common_attention.add_timing_signals_from_features(
        tf.zeros([1, decode_length, hparams.hidden_size]), features,
        hparams.position_features)
  elif hparams.pos == "emb":
    positional_encoding = common_attention.add_positional_embedding(
        tf.zeros([1, decode_length, hparams.hidden_size]), hparams.max_length,
        "body/targets_positional_embedding", None)
  else:
    positional_encoding = None

  def preprocess_targets(targets, i):
    """Performs preprocessing steps on the targets to prepare for the decoder.

    This includes:
      - Embedding the ids.
      - Flattening to 3D tensor.
      - Optionally adding timing signals.

    Args:
      targets: inputs ids to the decoder. [batch_size, 1]
      i: scalar, Step number of the decoding loop.

    Returns:
      Processed targets [batch_size, 1, hidden_dim]
    """
    # _shard_features called to ensure that the variable names match
    targets = self._shard_features({"targets": targets})["targets"]
    modality_name = hparams.name.get(
        "targets",
        modalities.get_name(target_modality))(hparams, target_vocab_size)
    with tf.variable_scope(modality_name):
      bottom = hparams.bottom.get(
          "targets", modalities.get_targets_bottom(target_modality))
      targets = dp(bottom, targets, hparams, target_vocab_size)[0]
    targets = common_layers.flatten4d3d(targets)

    # GO embeddings are all zero, this is because transformer_prepare_decoder
    # Shifts the targets along by one for the input which pads with zeros.
    # If the modality already maps GO to the zero embeddings this is not
    # needed.
    if not self.get_decode_start_id():
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

    if positional_encoding is not None:
      targets += positional_encoding[:, i:i + 1]
    return targets

  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(decode_length))
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        decode_length)

  # Create tensors for encoder-decoder attention history
  att_cache = {"attention_history": {}}
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  if encoder_output is not None:
    att_batch_size, enc_seq_length = common_layers.shape_list(
        encoder_output)[0:2]
    for layer in range(num_layers):
      att_cache["attention_history"]["layer_%d" % layer] = tf.zeros(
          [att_batch_size, hparams.num_heads, 0, enc_seq_length])

  def update_decoder_attention_history(cache):
    """Save attention weights in cache, e.g., for vizualization."""
    for k in [x for x in self.attention_weights
              if "decoder" in x and "self" not in x and "logits" not in x]:
      idx = k.find("layer_")
      if idx < 0:
        continue
      # Get layer number from the string name.
      layer_nbr = k[idx + 6:]
      idx = 0
      while idx + 1 < len(layer_nbr) and layer_nbr[:idx + 1].isdigit():
        idx += 1
      layer_nbr = "layer_%d" % int(layer_nbr[:idx])
      if layer_nbr in cache["attention_history"]:
        cache["attention_history"][layer_nbr] = tf.concat(
            [cache["attention_history"][layer_nbr],
             self.attention_weights[k]],
            axis=2)
  if not preprocess_targets_method:
    preprocess_targets_method = preprocess_targets

  def symbols_to_logits_fn(ids, i, cache):
    """Go from ids to logits for next symbol."""
    ids = ids[:, -1:]
    targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
    targets = preprocess_targets_method(targets, i)

    bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
    with tf.variable_scope("body"):
      body_outputs = dp(
          self.decode,
          targets,
          cache.get("encoder_output"),
          cache.get("encoder_decoder_attention_bias"),
          bias,
          hparams,
          cache,
          nonpadding=features_to_nonpadding(features, "targets"))

    update_decoder_attention_history(cache)

    modality_name = hparams.name.get(
        "targets",
        modalities.get_name(target_modality))(hparams, target_vocab_size)
    with tf.variable_scope(modality_name):
      top = hparams.top.get("targets", modalities.get_top(target_modality))
      logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

    ret = tf.squeeze(logits, axis=[1, 2, 3])
    if partial_targets is not None:
      # If the position is within the given partial targets, we alter the
      # logits to always return those values.
      # A faster approach would be to process the partial targets in one
      # iteration in order to fill the corresponding parts of the cache.
      # This would require broader changes, though.
      vocab_size = tf.shape(ret)[1]

      def forced_logits():
        return tf.one_hot(
            tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
            -1e9)

      ret = tf.cond(
          tf.less(i, partial_targets_length), forced_logits, lambda: ret)
    return ret, cache

  sos_id = self.get_decode_start_id() or 0
  eos_id = self.get_decode_end_id() or beam_search.EOS_ID
  temperature = features.get("sampling_temp",
                             getattr(hparams, "sampling_temp", 0.0))
  top_k = features.get("sampling_keep_top_k",
                       getattr(hparams, "sampling_keep_top_k", -1))

  ret = fast_decode(
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      symbols_to_logits_fn=symbols_to_logits_fn,
      hparams=hparams,
      decode_length=decode_length,
      vocab_size=target_vocab_size,
      init_cache_fn=self._init_cache_fn,
      beam_size=beam_size,
      top_beams=top_beams,
      alpha=alpha,
      batch_size=batch_size,
      force_decode_length=self._decode_hparams.force_decode_length,
      sos_id=sos_id,
      eos_id=eos_id,
      sampling_temperature=temperature,
      top_k=top_k,
      cache=att_cache)
  if partial_targets is not None:
    if beam_size <= 1 or top_beams <= 1:
      ret["outputs"] = ret["outputs"][:, partial_targets_length:]
    else:
      ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
  return ret
```

首先是设置decoder的输出长度(decode_length)为输入在加上50：
```
if self.has_input:
  inputs_shape = common_layers.shape_list(features["inputs"])
  if (target_modality == modalities.ModalityType.CLASS_LABEL or
      self._problem_hparams.get("regression_targets")):
    decode_length = 1
  else:
    decode_length = (
        inputs_shape[1] + features.get("decode_length", decode_length))
  batch_size = inputs_shape[0]
```
接着调用_prepare_inputs_for_decode：
```
def _prepare_inputs_for_decode(self, features):
  """Prepare inputs for decoding.

  Args:
    features: A map of string to model features.

  Returns:
    Inputs after fixing shape and applying modality.
  """
  dp = self._data_parallelism
  hparams = self._hparams
  inputs = features["inputs"]
  # TODO(llion): Clean up this reshaping logic.
  inputs = tf.expand_dims(inputs, axis=1)
  if len(inputs.shape) < 5:
    inputs = tf.expand_dims(inputs, axis=4)
  s = common_layers.shape_list(inputs)
  inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
  # _shard_features called to ensure that the variable names match
  inputs = self._shard_features({"inputs": inputs})["inputs"]
  input_modality = self._problem_hparams.modality["inputs"]
  input_vocab_size = self._problem_hparams.vocab_size["inputs"]
  if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
    input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
  modality_name = hparams.name.get("inputs",
                                   modalities.get_name(input_modality))(
                                       hparams, input_vocab_size)
  with tf.variable_scope(modality_name):
    bottom = hparams.bottom.get("inputs",
                                modalities.get_bottom(input_modality))
    inputs = dp(bottom, inputs, hparams, input_vocab_size)
  return inputs
```
它首先把输入从(1, 12, 1)变成(1, 12, 1, 1)，然后是计算bottom：
```
with tf.variable_scope(modality_name):
  bottom = hparams.bottom.get("inputs",
                              modalities.get_bottom(input_modality))
  inputs = dp(bottom, inputs, hparams, input_vocab_size)
return inputs
```
最终调用的是symbol_bottom，这个函数前面已经介绍过了。忘了的读者可以参考前面：
```
def symbol_bottom(x, model_hparams, vocab_size):
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    return _symbol_bottom_simple(
        x, model_hparams, vocab_size, "shared", reuse=None)
  return _symbol_bottom_simple(
      x, model_hparams, vocab_size, "input_emb", reuse=None)
```
最终返回的是(1, 12, 1, 512)。接着进行encoder的计算：
```
with tf.variable_scope("body"):
  encoder_output, encoder_decoder_attention_bias = dp(
      self.encode,
      inputs,
      features["target_space_id"],
      hparams,
      features=features)
```
它调用的是encode，这和训练时一样的，这里也不赘述了。
```
def encode(self, inputs, target_space, hparams, features=None, losses=None):
  """Encode transformer inputs, see transformer_encode."""
  return transformer_encode(
      self._encoder_function, inputs, target_space, hparams,
      attention_weights=self.attention_weights,
      features=features, losses=losses,
      prepare_encoder_fn=self._prepare_encoder_fn)
```

最终encoder的输出是(1, 12, 512)。接着对decoder进行位置编码：
```
if hparams.pos == "timing":
  positional_encoding = common_attention.get_timing_signal_1d(
      decode_length + 1, hparams.hidden_size)
```
代码应该也是介绍过的。接着构造下三角阵的attention mask：
```
decoder_self_attention_bias = (
    common_attention.attention_bias_lower_triangle(decode_length))
```
下面的代码是构造attention_history的cache，每一层一个，shape是tf.Tensor([], shape=(1, 8, 0, 12), dtype=float32)
```
for layer in range(num_layers):
  att_cache["attention_history"]["layer_%d" % layer] = tf.zeros(
      [att_batch_size, hparams.num_heads, 0, enc_seq_length])
```

然后就进入fast_decode：
```
def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                init_cache_fn=_init_transformer_cache,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                sos_id=0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False,
                scope_prefix="body/",
                sampling_temperature=0.0,
                top_k=-1,
                cache=None):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple `(ids,
      step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    init_cache_fn: Function that returns the initial cache dict.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    sos_id: End-of-sequence symbol in beam search.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.
    sampling_temperature: scalar, temperature with which to sample.
    top_k: scalar, sample only top k.
    cache: cache dictionary for additional predictions.

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  cache = init_cache_fn(
      cache=cache,
      hparams=hparams,
      batch_size=batch_size,
      attention_init_length=0,
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      scope_prefix=scope_prefix)

  if beam_size > 1:  # Beam Search
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    decoded_ids, scores, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # Greedy

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = sampling_temperature
      if hparams.sampling_method == "random_per_example":
        next_id = common_layers.sample_temperature_per_example(
            logits, temperature, top_k)
      else:
        if hparams.sampling_method == "argmax":
          temperature = 0.0
        next_id = common_layers.sample_with_temperature(logits, temperature,
                                                        top_k)

      log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
                                  axis=1)
      log_prob += tf.gather_nd(
          log_probs, log_prob_indices) * (1 - tf.to_float(hit_eos))
      # Note(thangluong): we purposely update hit_eos after aggregating log_prob
      # There is a subtle detail here that we want to include log_probs up to
      # (and inclusive of) the first eos generated, but not subsequent tokens.
      hit_eos |= tf.equal(next_id, eos_id)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)

      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, cache, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
            tf.TensorShape([None]),
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores, "cache": cache}
```
首先初始化cache：
```
def _init_transformer_cache(cache, hparams, batch_size, attention_init_length,
                            encoder_output, encoder_decoder_attention_bias,
                            scope_prefix):
  """Create the initial cache for Transformer fast decoding."""
  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  if cache is None:
    cache = {}
  cache.update({
      "layer_%d" % layer: {  # pylint: disable=g-complex-comprehension
          "k":
              common_attention.split_heads(
                  tf.zeros([batch_size,
                            attention_init_length,
                            key_channels]), hparams.num_heads),
          "v":
              common_attention.split_heads(
                  tf.zeros([batch_size,
                            attention_init_length,
                            value_channels]), hparams.num_heads),
      } for layer in range(num_layers)
  })

  # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
  # cache key "f" won't be used, which means that the` shape of cache["f"]`
  # won't be changed to
  # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
  # error when applying `nest.map reshape function` on it.
  if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
    for layer in range(num_layers):
      cache["layer_%d" % layer]["f"] = tf.zeros(
          [batch_size, 0, hparams.hidden_size])

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope(
          "%sdecoder/%s/encdec_attention/multihead_attention" %
          (scope_prefix, layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output,
            key_channels,
            name="k",
            vars_3d_num_heads=vars_3d_num_heads)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
        v_encdec = common_attention.compute_attention_component(
            encoder_output,
            value_channels,
            name="v",
            vars_3d_num_heads=vars_3d_num_heads)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
  return cache
```
为什么需要cache呢？因为decode的时候只能一个时刻一个时刻的计算，所以每个时刻都需要分别计算q、k和v。q是解码的过程中动态变化的(依赖于前一个时刻)，但是k和v是不依赖时刻的，可以提前算好，避免重复计算。计算k和v的时候有对于的attention  head的变换矩阵就行。
```
def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c",
                                vars_3d_num_heads=0,
                                layer_collection=None):
  """Computes attention component (query, key or value).

  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
    vars_3d_num_heads: an optional integer (if we want to use 3d variables)
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.

  Returns:
    c : [batch, length, depth] tensor
  """
  if layer_collection is not None:
    if filter_width != 1 or vars_3d_num_heads != 0:
      raise ValueError(
          "KFAC implementation only supports filter_width=1 (actual: {}) and "
          "vars_3d_num_heads=0 (actual: {}).".format(
              filter_width, vars_3d_num_heads))
  if vars_3d_num_heads is not None and vars_3d_num_heads > 0:
    assert filter_width == 1
    input_depth = antecedent.get_shape().as_list()[-1]
    depth_per_head = total_depth // vars_3d_num_heads
    initializer_stddev = input_depth ** -0.5
    if "q" in name:
      initializer_stddev *= depth_per_head ** -0.5
    var = tf.get_variable(
        name, [input_depth,
               vars_3d_num_heads,
               total_depth // vars_3d_num_heads],
        initializer=tf.random_normal_initializer(stddev=initializer_stddev))
    var = tf.cast(var, antecedent.dtype)
    var = tf.reshape(var, [input_depth, total_depth])
    return tf.tensordot(antecedent, var, axes=1)
  if filter_width == 1:
    return common_layers.dense(
        antecedent, total_depth, use_bias=False, name=name,
        layer_collection=layer_collection)
  else:
    return common_layers.conv1d(
        antecedent, total_depth, filter_width, padding=padding, name=name)
```
每一层都会计算，k的shape是(1, 8, 12, 64)，v也是一样。

接下来就是最复杂的部分：
```
decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
hit_eos = tf.fill([batch_size], False)
next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
_, _, _, decoded_ids, cache, log_prob = tf.while_loop(
    is_not_finished,
    inner_loop, [
        tf.constant(0), hit_eos, next_id, decoded_ids, cache,
        initial_log_prob
    ],
    shape_invariants=[
        tf.TensorShape([]),
        tf.TensorShape([None]),
        tf.TensorShape([None, None]),
        tf.TensorShape([None, None]),
        nest.map_structure(beam_search.get_state_shape_invariants, cache),
        tf.TensorShape([None]),
    ])
scores = log_prob

```

我们首先学习一下tf.while_loop函数：
```
tf.while_loop(
    cond, body, loop_vars, shape_invariants=None, parallel_iterations=10,
    back_prop=True, swap_memory=False, maximum_iterations=None, name=None
)
```
简单来说，就是：
```
while(cond){
    body();
}
```
但是怎么修改变量让cond变化呢？这就是下面介绍的参数的作用了。
cond是一个callable，调用它会产生一个True/False。
body也是一个callable，它可以返回tuple、 namedtuple或者tensor的list，而且返回值的个数和loop_vars相同。
loop_vars是cond和body都会用到的一堆变量，可以是tuple、namedtuple或者tensor的list。
cond和body会用它们来计算，也会更新它们的值。

除了常规的Tensor或者IndexedSlice，body也可以接受或者返回TensorArray。TensorArray对象的流动会在循环之间以及梯度计算时正确的被传递。

注意while_loop会调用cond和body一次，并且仅一次(当然指的是while_loop被调用时，而不是Session.run的时候，否则如果只能调用一次就不是循环了)。while_loop会自动帮我们创建一个复杂的计算图，它保证body会一直运行直到条件cond返回False。

为了保证正确性，tf.while_loop严格检查loop循环变量的shape不变性。shape的(部分)不变性指的是变量的shape在多次循环中保持不变。如果迭代过程中发现shape变得比之前更加不确定(general)或者不一样，则会抛出异常。比如之前[11, 17]，之后变成[11, None]则不行，因为None比17更不确定(general)。另外[11, 17]变成[11,21]也是不行的，因为它们不一样。如果shape_invariants没有指定，则认为每次迭代的初始化shape都是相同的。而参数shape_invariants可以让调用者制定不同变量的shape不变性，因为有点时候我们需要改变其shape。我们在body里也可以使用tf.Tensor.set_shape来指定更加准确的shape。SparseTensor和IndexedSlices的shape不变性规则如下：

a) 如果一个循环变量是SparseTensor，shape不变性要求TensorShape([r])，其中r是对应的稠密向量的rank。因此它意味着SparseTensor用来表示稠密Tensor的三个内部Tensor的shape是([None], [None, r]和[r])。注意：这里要求的shape不变性指的是对应稠密向量而言的。

b) 如果一般循环变量是IndexedSlices，则它的shape不变性要求这个IndexedSlices对应的Tensor的shape不变。这就意味着IndexedSlices内部的三个Tensor的shape是(shape, [shape[0]], [shape.ndims])。

while_loop实现非严格的循环语义，从而可以并行的执行多个迭代。最大的并行数由parallel_iterations控制，我们可以通过它来进行一定程度的内存消耗和执行顺序控制。对于正确的代码，while_loop保证对于不同的parallel_iterations会返回相同的结果(但是中间过程并不完全按照顺序来执行)。

对于训练，TensorFlow会保存forward产生的变量，因为它们在反向计算时需要用到。这些临时变量是主要的内存消耗大户，在GPU这种内存有限的设备上很容易造成OOM错误。如果swap_memory是True，则会把GPU的Tensor换到CPU的内存里。这对于像RNN这样非常长或者非常大的batch时可能有用(但是就像用磁盘来模拟的虚拟内存，它对性能的影响是非常大的)。

下面是一个非常简单的例子：
```
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: (tf.add(i, 1), )
r = tf.while_loop(c, b, [i])
```
它实现的类似下面的代码：
```
i = 0
while(i < 10):
    i += 1

return i;
```

下面的例子用tuple来传递多个参数：
```
import collections
Pair = collections.namedtuple('Pair', 'j, k')
ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
c = lambda i, p: i < 10
b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
ijk_final = tf.while_loop(c, b, ijk_0)
print(ijk_final)
```
它等价于：
```
i, (j, k) = 0, (1, 2)
while i < 10:
    i = i + 1
    temp1 = j + k
    temp2 = j - k
    j = temp1
    k = temp2
print(i, j, k)
```
注意：不能写成这样：
```
while i < 10:
    i = i + 1
    j = j + k
    k = j - k
```

下面是循环变量的shape会发生变化的例子：
```
i0 = tf.constant(0)
m0 = tf.ones([2, 2])
c = lambda i, m: i < 10
b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
res =tf.while_loop(
    c, b, loop_vars=[i0, m0],
    shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])
```

最初的m是[2, 2]，然后每次shape都加倍[4, 2], [8, 2], ...。最终变成[2048, 2]，所以需要增加shape_invariants为[None, 2]，表明第一个维度的shape是不定可变的。

回到fast_decode函数，我先来看循环条件：
```
def is_not_finished(i, hit_eos, *_):
  finished = i >= decode_length
  if not force_decode_length:
    finished |= tf.reduce_all(hit_eos)
  return tf.logical_not(finished)
```
退出的条件之一是解码器输出的长度超过了最大的长度(decode_length)；另外如果没有force_decode_length，那么碰到eos也可以结束解码。

这里会用到tf.reduce_all函数，它的作用是计算逻辑“与”。举个例子：
```
x = tf.constant([[True,  True], [False, False]])
tf.reduce_all(x)  # False
tf.reduce_all(x, 0)  # [False, False]
tf.reduce_all(x, 1)  # [True, False]
```
第一个reduce_all没有传入dimension的信息，因此对所有4个元素进行逻辑与，所以返回False。第二个按照第1个dimension(行)进行逻辑与，也就是[True, True]与[False, False]这两行逐个元素计算逻辑与，所以返回两个False。第三个按照列做运算，也就是[True, False]与[True, False]逐个位置计算，所以返回[True, False]。

循环的body函数为：
```
def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
  """One step of greedy decoding."""
  logits, cache = symbols_to_logits_fn(next_id, i, cache)
  log_probs = common_layers.log_prob_from_logits(logits)
  temperature = sampling_temperature
  if hparams.sampling_method == "random_per_example":
    next_id = common_layers.sample_temperature_per_example(
        logits, temperature, top_k)
  else:
    if hparams.sampling_method == "argmax":
      temperature = 0.0
    next_id = common_layers.sample_with_temperature(logits, temperature,
                                                    top_k)

  log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
                              axis=1)
  log_prob += tf.gather_nd(
      log_probs, log_prob_indices) * (1 - tf.to_float(hit_eos))
  # Note(thangluong): we purposely update hit_eos after aggregating log_prob
  # There is a subtle detail here that we want to include log_probs up to
  # (and inclusive of) the first eos generated, but not subsequent tokens.
  hit_eos |= tf.equal(next_id, eos_id)

  next_id = tf.expand_dims(next_id, axis=1)
  decoded_ids = tf.concat([decoded_ids, next_id], axis=1)

  return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob
```

tf.while_loop真正执行的是如下的代码：

``` 
  if not callable(cond):
    raise TypeError("cond must be callable.")
  if not callable(body):
    raise TypeError("body must be callable.")
  if parallel_iterations < 1:
    raise TypeError("parallel_iterations must be a positive integer.")

  # Always enable control flow v2 if building a function, regardless of toggle.
  executing_eagerly = context.executing_eagerly()
  if (util.EnableControlFlowV2(ops.get_default_graph()) and
      not executing_eagerly):
    return while_v2.while_loop(
        cond,
        body,
        loop_vars,
        shape_invariants=shape_invariants,
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
        name=name,
        return_same_structure=return_same_structure,
        back_prop=back_prop)

  with ops.name_scope(name, "while", loop_vars):
    if not loop_vars:
      raise ValueError("No loop variables provided")
    try_to_pack = (len(loop_vars) == 1 and not return_same_structure)
    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, name="maximum_iterations")
      if maximum_iterations.shape.ndims != 0:
        raise ValueError("maximum_iterations must be a scalar, saw shape: %s" %
                         maximum_iterations.shape)

      if executing_eagerly:
        counter = 0
        maximum_iterations = int(maximum_iterations.numpy())
      else:
        counter = constant_op.constant(
            0, dtype=maximum_iterations.dtype, name="iteration_counter")
      orig_cond = cond
      orig_body = body
      if try_to_pack:
        loop_vars = (counter, loop_vars[0])
        cond = lambda i, lv: (  # pylint: disable=g-long-lambda
            math_ops.logical_and(i < maximum_iterations, orig_cond(lv)))
        body = lambda i, lv: (i + 1, orig_body(lv))
      else:
        loop_vars = (counter, loop_vars)
        cond = lambda i, lv: (  # pylint: disable=g-long-lambda
            math_ops.logical_and(i < maximum_iterations, orig_cond(*lv)))
        body = lambda i, lv: (i + 1, orig_body(*lv))
      try_to_pack = False

    if executing_eagerly:
      packed = False  # whether the body result was packed into a 1-item tuple

      loop_var_structure = nest.map_structure(type_spec.type_spec_from_value,
                                              list(loop_vars))
      while cond(*loop_vars):
        loop_vars = body(*loop_vars)
        if try_to_pack and not isinstance(loop_vars, (list, _basetuple)):
          packed = True
          loop_vars = (loop_vars,)
        nest.assert_same_structure(loop_var_structure, list(loop_vars))

      def convert(x):
        if isinstance(x, tensor_array_ops.TensorArray):
          return x
        return ops.convert_to_tensor(x)

      loop_vars = nest.map_structure(convert, loop_vars, expand_composites=True)
      if maximum_iterations is not None:
        return loop_vars[1]
      else:
        return loop_vars[0] if packed else loop_vars

    if shape_invariants is not None:
      if maximum_iterations is not None:
        shape_invariants = (tensor_shape.TensorShape([]), shape_invariants)

      nest.assert_same_structure(
          loop_vars, shape_invariants, expand_composites=False)
      shape_invariants = nest.map_structure(
          _get_shape_invariant,
          loop_vars,
          shape_invariants,
          expand_composites=False)

    loop_context = WhileContext(
        maximum_iterations=maximum_iterations,
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)
    # Only add non-nested loops to the collection. Any nested control flow will
    # be encapsulated in the root context.
    if loop_context.outer_context is None:
      ops.add_to_collection(ops.GraphKeys.WHILE_CONTEXT, loop_context)
    result = loop_context.BuildLoop(cond, body, loop_vars, shape_invariants,
                                    return_same_structure)
    if maximum_iterations is not None:
      return result[1]
    else:
      return result

```
代码很长，这是tensorflow的内部代码，不感兴趣的读者可以跳过，我们重点只要关注：
```
while cond(*loop_vars):
  loop_vars = body(*loop_vars)
  if try_to_pack and not isinstance(loop_vars, (list, _basetuple)):
    packed = True
    loop_vars = (loop_vars,)
  nest.assert_same_structure(loop_var_structure, list(loop_vars))
```
这里的body就会调用前面的inner_loop函数，我们还是来跟踪inner_loop的执行过程：
```
logits, cache = symbols_to_logits_fn(next_id, i, cache)
```
传入symbols_to_logits_fn函数3个参数：第一个是模型预测的ids，是到当前时刻为止所有输出的id(第一个时刻是sos=0)，i表示时刻，cache就是前面介绍的k、v的cache。接着我们进入symbols_to_logits_fn函数：
```
def symbols_to_logits_fn(ids, i, cache):
  """Go from ids to logits for next symbol."""
  ids = ids[:, -1:]
  targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
  targets = preprocess_targets_method(targets, i)

  bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
  with tf.variable_scope("body"):
    body_outputs = dp(
        self.decode,
        targets,
        cache.get("encoder_output"),
        cache.get("encoder_decoder_attention_bias"),
        bias,
        hparams,
        cache,
        nonpadding=features_to_nonpadding(features, "targets"))

  update_decoder_attention_history(cache)

  modality_name = hparams.name.get(
      "targets",
      modalities.get_name(target_modality))(hparams, target_vocab_size)
  with tf.variable_scope(modality_name):
    top = hparams.top.get("targets", modalities.get_top(target_modality))
    logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

  ret = tf.squeeze(logits, axis=[1, 2, 3])
  if partial_targets is not None:
    # If the position is within the given partial targets, we alter the
    # logits to always return those values.
    # A faster approach would be to process the partial targets in one
    # iteration in order to fill the corresponding parts of the cache.
    # This would require broader changes, though.
    vocab_size = tf.shape(ret)[1]

    def forced_logits():
      return tf.one_hot(
          tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
          -1e9)

    ret = tf.cond(
        tf.less(i, partial_targets_length), forced_logits, lambda: ret)
  return ret, cache
```
这个函数首先拿到最后一个输出的id：
```
ids = ids[:, -1:]
```
这个时候是0。然后调用preprocess_targets_method，进行当前时刻的计算：
```
def preprocess_targets(targets, i):
  """Performs preprocessing steps on the targets to prepare for the decoder.

  This includes:
    - Embedding the ids.
    - Flattening to 3D tensor.
    - Optionally adding timing signals.

  Args:
    targets: inputs ids to the decoder. [batch_size, 1]
    i: scalar, Step number of the decoding loop.

  Returns:
    Processed targets [batch_size, 1, hidden_dim]
  """
  # _shard_features called to ensure that the variable names match
  targets = self._shard_features({"targets": targets})["targets"]
  modality_name = hparams.name.get(
      "targets",
      modalities.get_name(target_modality))(hparams, target_vocab_size)
  with tf.variable_scope(modality_name):
    bottom = hparams.bottom.get(
        "targets", modalities.get_targets_bottom(target_modality))
    targets = dp(bottom, targets, hparams, target_vocab_size)[0]
  targets = common_layers.flatten4d3d(targets)

  # GO embeddings are all zero, this is because transformer_prepare_decoder
  # Shifts the targets along by one for the input which pads with zeros.
  # If the modality already maps GO to the zero embeddings this is not
  # needed.
  if not self.get_decode_start_id():
    targets = tf.cond(
        tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

  if positional_encoding is not None:
    targets += positional_encoding[:, i:i + 1]
  return targets

。。。。。
省略了一些本次调用不会执行的代码。
```
这里主要就是调用dp(bottom, targets, hparams, target_vocab_size)[0]进行bottom的计算(把id变成Embedding)，同时加入位置编码。
接着回到symbols_to_logits_fn计算body：
```
bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
with tf.variable_scope("body"):
  body_outputs = dp(
      self.decode,
      targets,
      cache.get("encoder_output"),
      cache.get("encoder_decoder_attention_bias"),
      bias,
      hparams,
      cache,
      nonpadding=features_to_nonpadding(features, "targets"))
```
最终调用的还是decode函数：
```
def decode(self,
           decoder_input,
           encoder_output,
           encoder_decoder_attention_bias,
           decoder_self_attention_bias,
           hparams,
           cache=None,
           decode_loop_step=None,
           nonpadding=None,
           losses=None,
           **kwargs):
  """Decode Transformer outputs, see transformer_decode."""
  return transformer_decode(
      self._decoder_function, decoder_input, encoder_output,
      encoder_decoder_attention_bias, decoder_self_attention_bias,
      hparams, attention_weights=self.attention_weights, cache=cache,
      decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
      **kwargs)
```
这个函数最终调用我们前面介绍过的transformer_decoder函数，计算6层的self-attention和ffn等，得到body。
接着调用update_decoder_attention_history函数把decoder对encoder的attention保存下来，这可以用于可视化attention。
```
def update_decoder_attention_history(cache):
  """Save attention weights in cache, e.g., for vizualization."""
  for k in [x for x in self.attention_weights
            if "decoder" in x and "self" not in x and "logits" not in x]:
    idx = k.find("layer_")
    if idx < 0:
      continue
    # Get layer number from the string name.
    layer_nbr = k[idx + 6:]
    idx = 0
    while idx + 1 < len(layer_nbr) and layer_nbr[:idx + 1].isdigit():
      idx += 1
    layer_nbr = "layer_%d" % int(layer_nbr[:idx])
    if layer_nbr in cache["attention_history"]:
      cache["attention_history"][layer_nbr] = tf.concat(
          [cache["attention_history"][layer_nbr],
           self.attention_weights[k]],
          axis=2)
```

所有attention相关的变量都保存在self.attention_weights里，名字包含decoder并且不包含self或者logits的就是decoder对encoder的attention，比如'transformer/body/decoder/layer_0/encdec_attention/multihead_attention/dot_product_attention'，它就代表第一层decoder对encoder输出的attention。

<a name='img11'>![](/img/t2t-code/11.png)</a>
*图：attention_weights*

接着就是计算top和logits：
```
with tf.variable_scope(modality_name):
  top = hparams.top.get("targets", modalities.get_top(target_modality))
  logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]
```

计算完成logits后返回inner_loop：
```
logits, cache = symbols_to_logits_fn(next_id, i, cache)
log_probs = common_layers.log_prob_from_logits(logits)
```
common_layers.log_prob_from_logits函数根据logits计算概率。接着sample一个词：
```
temperature = sampling_temperature
if hparams.sampling_method == "random_per_example":
  next_id = common_layers.sample_temperature_per_example(
      logits, temperature, top_k)
else:
  if hparams.sampling_method == "argmax":
    temperature = 0.0
  next_id = common_layers.sample_with_temperature(logits, temperature,
                                                  top_k)

def sample_with_temperature(logits, temperature, sampling_keep_top_k=-1):

  if temperature == 0.0:
    # TF argmax doesn't handle >5 dimensions, so we reshape here.
    logits_shape = shape_list(logits)
    argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=1)
    return tf.reshape(argmax, logits_shape[:-1])
  。。。。省略了temperature > 0的代码。
```
我们这里的sampling_method是argmax，也就是选择概率最大的那个词。因此temperature设置成0。如果temperature>0，则概率不是最大的词也有可能被选中。最终得到了当前预测的next_id=107。对应的中文是“我”。接着计算当前的输出是否eos：
```
hit_eos |= tf.equal(next_id, eos_id)
```
最后把当前的id加到decoded_ids里：
```
next_id = tf.expand_dims(next_id, axis=1)
decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
```
至此一个时刻的处理完成，接着就是下一个时刻的循环，最终得到的就是tf.Tensor([[ 107  163   44 1867  460 2653 1266    4    1]], shape=(1, 9), dtype=int64)。
最后还需要把subword的id变成字符串：
```
def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["targets"].decode(np.squeeze(integers))
```
首先去掉1(eos)之后的内容，然后调用encoders["targets"].decode函数把id变成字符串：
```
def decode(self, ids, strip_extraneous=False):
  """Converts a sequence of subtoken ids to a native string.

  Args:
    ids: a list of integers in the range [0, vocab_size)
    strip_extraneous: bool, whether to strip off extraneous tokens
      (EOS and PAD).

  Returns:
    a native string
  """
  if strip_extraneous:
    ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
  return unicode_to_native(
      tokenizer.decode(self._subtoken_ids_to_tokens(ids)))
```
首先是_subtoken_ids_to_tokens，把subword的id合并成word的id。
```
def _subtoken_ids_to_tokens(self, subtokens):
  """Converts a list of subtoken ids to a list of tokens.

  Args:
    subtokens: a list of integers in the range [0, vocab_size)
  Returns:
    a list of strings.
  """
  concatenated = "".join(
      [self._subtoken_id_to_subtoken_string(s) for s in subtokens])
  split = concatenated.split("_")
  ret = []
  for t in split:
    if t:
      unescaped = _unescape_token(t + "_")
      if unescaped:
        ret.append(unescaped)
  return ret
```
首先把subtokens([ 107  163   44 1867  460 2653 1266    4])变成对应的字符串concatenated：'我认为他们永远不会回到美国_。_'。然后处理下划线。注意：这里还是有前面的问题，它会把'我认为他们永远不会回到美国_'当成一个词。因此最终得到两个词：'我认为他们永远不会回到美国'和'。'

恭喜！我们终于完成了tensor2tensor的代码阅读。
