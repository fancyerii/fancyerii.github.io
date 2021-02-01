---
layout:     post
title:      "Tensor2Tensor中英翻译代码阅读(四)" 
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

本系列文章是Tensor2Tensor的代码阅读，主要关注中英翻译的实现。本文是第四篇，继续介绍Transformer的训练代码。

<!--more-->

**目录**
* TOC
{:toc}

注意：本篇的代码量非常大，建议读者可以分成几次来阅读，记得上一次运行的断点。
 
## 训练代码

训练前首先设置超参数，
HPARAMS = 'transformer_base_single_gpu'
hparams = create_hparams(HPARAMS)
```
def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None,
                   hparams_path=None):
  """Create HParams with data_dir and problem hparams, if kwargs provided."""
  hparams = registry.hparams(hparams_set)
  if hparams_path and tf.gfile.Exists(hparams_path):
    hparams = create_hparams_from_json(hparams_path, hparams)
  if data_dir:
    hparams.add_hparam("data_dir", data_dir)
  if hparams_overrides_str:
    tf.logging.info("Overriding hparams in %s with %s", hparams_set,
                    hparams_overrides_str)
    hparams = hparams.parse(hparams_overrides_str)
  if problem_name:
    add_problem_hparams(hparams, problem_name)
  return hparams
```

transformer_base_single_gpu对应的超参数为：
```
[('activation_dtype', 'float32'), ('add_relative_to_values', False), ('area_key_mode', 'none'), ('area_value_mode', 'none'), ('attention_dropout', 0.1), ('attention_dropout_broadcast_dims', ''), ('attention_key_channels', 0), ('attention_value_channels', 0), ('attention_variables_3d', False), ('batch_shuffle_size', 512), ('batch_size', 1024), ('bottom', {}), ('causal_decoder_self_attention', True), ('clip_grad_norm', 0.0), ('compress_steps', 0), ('conv_first_kernel', 3), ('daisy_chain_variables', True), ('dropout', 0.2), ('eval_drop_long_sequences', False), ('eval_run_autoregressive', False), ('factored_logits', False), ('ffn_layer', 'dense_relu_dense'), ('filter_size', 2048), ('force_full_predict', False), ('gpu_automatic_mixed_precision', False), ('grad_noise_scale', 0.0), ('gumbel_noise_weight', 0.0), ('hard_attention_k', 0), ('heads_share_relative_embedding', False), ('hidden_size', 512), ('initializer', 'uniform_unit_scaling'), ('initializer_gain', 1.0), ('kernel_height', 3), ('kernel_width', 1), ('label_smoothing', 0.1), ('layer_postprocess_sequence', 'da'), ('layer_prepostprocess_dropout', 0.1), ('layer_prepostprocess_dropout_broadcast_dims', ''), ('layer_preprocess_sequence', 'n'), ('learning_rate', 0.2), ('learning_rate_constant', 0.1), ('learning_rate_cosine_cycle_steps', 250000), ('learning_rate_decay_rate', 1.0), ('learning_rate_decay_scheme', 'noam'), ('learning_rate_decay_staircase', False), ('learning_rate_decay_steps', 5000), ('learning_rate_minimum', None), ('learning_rate_schedule', 'constant*linear_warmup*rsqrt_decay'), ('learning_rate_warmup_steps', 16000), ('length_bucket_step', 1.1), ('loss', {}), ('max_area_height', 1), ('max_area_width', 1), ('max_input_seq_length', 0), ('max_length', 256), ('max_relative_position', 0), ('max_target_seq_length', 0), ('memory_height', 1), ('min_length', 0), ('min_length_bucket', 8), ('mixed_precision_optimizer_init_loss_scale', 32768), ('mixed_precision_optimizer_loss_scaler', 'exponential'), ('mlperf_mode', False), ('moe_hidden_sizes', '2048'), ('moe_k', 2), ('moe_loss_coef', 0.001), ('moe_num_experts', 16), ('moe_overhead_eval', 2.0), ('moe_overhead_train', 1.0), ('multiply_embedding_mode', 'sqrt_depth'), ('multiproblem_fixed_train_length', -1), ('multiproblem_label_weight', 0.5), ('multiproblem_max_input_length', -1), ('multiproblem_max_target_length', -1), ('multiproblem_mixing_schedule', 'constant'), ('multiproblem_per_task_threshold', ''), ('multiproblem_reweight_label_loss', False), ('multiproblem_schedule_max_examples', 10000000.0), ('multiproblem_schedule_threshold', 0.5), ('multiproblem_target_eval_only', False), ('multiproblem_vocab_size', -1), ('name', {}), ('nbr_decoder_problems', 1), ('no_data_parallelism', False), ('norm_epsilon', 1e-06), ('norm_type', 'layer'), ('num_area_layers', 0), ('num_decoder_layers', 0), ('num_encoder_layers', 0), ('num_heads', 8), ('num_hidden_layers', 6), ('optimizer', 'adam'), ('optimizer_adafactor_beta1', 0.0), ('optimizer_adafactor_beta2', 0.999), ('optimizer_adafactor_clipping_threshold', 1.0), ('optimizer_adafactor_decay_type', 'pow'), ('optimizer_adafactor_factored', True), ('optimizer_adafactor_memory_exponent', 0.8), ('optimizer_adafactor_multiply_by_parameter_scale', True), ('optimizer_adam_beta1', 0.9), ('optimizer_adam_beta2', 0.997), ('optimizer_adam_epsilon', 1e-09), ('optimizer_momentum_momentum', 0.9), ('optimizer_momentum_nesterov', False), ('optimizer_multistep_accumulate_steps', 0), ('optimizer_zero_grads', False), ('overload_eval_metric_name', ''), ('pack_dataset', False), ('pad_batch', False), ('parameter_attention_key_channels', 0), ('parameter_attention_value_channels', 0), ('pos', 'timing'), ('position_features', ''), ('prepend_mode', 'none'), ('pretrained_model_dir', ''), ('proximity_bias', False), ('relu_dropout', 0.1), ('relu_dropout_broadcast_dims', ''), ('sampling_keep_top_k', -1), ('sampling_method', 'argmax'), ('sampling_temp', 1.0), ('scheduled_sampling_gold_mixin_prob', 0.5), ('scheduled_sampling_method', 'parallel'), ('scheduled_sampling_num_passes', 1), ('scheduled_sampling_prob', 0.0), ('scheduled_sampling_warmup_schedule', 'exp'), ('scheduled_sampling_warmup_steps', 50000), ('self_attention_type', 'dot_product'), ('shared_embedding', False), ('shared_embedding_and_softmax_weights', True), ('split_targets_chunk_length', 0), ('split_targets_max_chunks', 100), ('split_targets_strided_training', False), ('split_to_length', 0), ('summarize_grads', False), ('summarize_vars', False), ('symbol_dropout', 0.0), ('symbol_modality_num_shards', 16), ('top', {}), ('tpu_enable_host_call', False), ('unidirectional_encoder', False), ('use_custom_ops', True), ('use_fixed_batch_size', False), ('use_pad_remover', True), ('use_target_space_embedding', True), ('video_num_input_frames', 1), ('video_num_target_frames', 1), ('vocab_divisor', 1), ('warm_start_from_second', ''), ('weight_decay', 0.0), ('weight_dtype', 'float32'), ('weight_noise', 0.0), ('weights_fn', {})]
```

接着创建run_config
```
RUN_CONFIG = create_run_config(
      model_dir=TRAIN_DIR,
      model_name=MODEL,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=5
)
```

```
def create_run_config(model_name,
                      master="",
                      model_dir=None,
                      iterations_per_loop=1000,
                      num_shards=8,
                      log_device_placement=False,
                      save_checkpoints_steps=1000,
                      save_checkpoints_secs=None,
                      keep_checkpoint_max=20,
                      keep_checkpoint_every_n_hours=10000,
                      num_gpus=1,
                      gpu_order="",
                      num_async_replicas=1,
                      enable_graph_rewriter=False,
                      gpu_mem_fraction=0.95,
                      no_data_parallelism=False,
                      optionally_use_dist_strat=False,
                      daisy_chain_variables=True,
                      schedule="continuous_train_and_eval",
                      worker_job="/job:localhost",
                      worker_id=0,
                      ps_replicas=0,
                      ps_job="/job:ps",
                      ps_gpu=0,
                      random_seed=None,
                      sync=False,
                      tpu_infeed_sleep_secs=None,
                      use_tpu=False,
                      use_tpu_estimator=False,
                      xla_jit_level=tf.OptimizerOptions.OFF,
                      inter_op_parallelism_threads=0,
                      log_step_count_steps=100,
                      intra_op_parallelism_threads=0,
                      tpu_config_extra_kwargs=None,
                      cloud_tpu_name="",
                      cloud_tpu_zone=None):
  """Create RunConfig, TPUConfig, and Parallelism object."""
  session_config = create_session_config(
      log_device_placement=log_device_placement,
      enable_graph_rewriter=enable_graph_rewriter,
      gpu_mem_fraction=gpu_mem_fraction,
      use_tpu=use_tpu,
      xla_jit_level=xla_jit_level,
      inter_op_parallelism_threads=inter_op_parallelism_threads,
      intra_op_parallelism_threads=intra_op_parallelism_threads)
  run_config_args = {
      "master": master,
      "evaluation_master": master,
      "model_dir": model_dir,
      "session_config": session_config,
      "save_summary_steps": 100,
      "save_checkpoints_steps": save_checkpoints_steps,
      "save_checkpoints_secs": save_checkpoints_secs,
      "keep_checkpoint_max": keep_checkpoint_max,
      "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
      "tf_random_seed": random_seed,
      "log_step_count_steps": log_step_count_steps,
  }
  if save_checkpoints_secs:
    del run_config_args["save_checkpoints_steps"]
  run_config_cls = contrib.learn().RunConfig

  if use_tpu or use_tpu_estimator:
    # If using TPUEstimator, use TPU RunConfig, add TPUConfig, and add
    # additional args.
    tpu_config_kwargs = {
        "iterations_per_loop": iterations_per_loop,
        "num_shards": num_shards,
        "per_host_input_for_training": True,
        "initial_infeed_sleep_secs": tpu_infeed_sleep_secs,
    }
    if tpu_config_extra_kwargs is not None:
      tpu_config_kwargs.update(tpu_config_extra_kwargs)
    run_config_cls = contrib.tpu().RunConfig
    tpu_config = contrib.tpu().TPUConfig(**tpu_config_kwargs)
    run_config_args["tpu_config"] = tpu_config
    if not master and "KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS" in os.environ:
      # If running on TPU but no master is set and the KUBE env var is present
      # then we're running on ML Engine. Set the master.
      run_config_args["master"] = os.environ[
          "KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS"]
      run_config_args["evaluation_master"] = run_config_args["master"]
    elif not master and cloud_tpu_name:
      # Update run_config to use cluster instead of master/evaluation_master
      # as we need the cluster spec to use Cloud Pods
      tpu_cluster_resolver = contrib.cluster_resolver().TPUClusterResolver(
          tpu=cloud_tpu_name, zone=cloud_tpu_zone)
      run_config_args["cluster"] = tpu_cluster_resolver
      del run_config_args["master"]
      del run_config_args["evaluation_master"]
  elif is_cloud_async_distributed():
    run_config_cls = tf.estimator.RunConfig
    del run_config_args["master"]
    del run_config_args["evaluation_master"]

  # tf.estimator RunConfig construction got totally broken in TF2.
  # we now have to specify master in a global environment variable
  if contrib.is_tf2:
    del run_config_args["evaluation_master"]
    del run_config_args["master"]

  config = run_config_cls(**run_config_args)

  # If not using TPU, add device info for data_parallelism
  config.use_tpu = use_tpu
  if not use_tpu:
    config.t2t_device_info = {
        "num_async_replicas": num_async_replicas,
    }
    use_distribution_strategy = (
        optionally_use_dist_strat and
        t2t_model.T2TModel.has_symmetric_shards(model_name) and
        not no_data_parallelism and ps_replicas == 0 and ps_gpu == 0 and
        num_async_replicas == 1)

    if use_distribution_strategy:
      tf.logging.info(
          "Configuring MirroredStrategy DistributionStrategy to replicate the "
          "model."
      )
      distribution = contrib.distribute().MirroredStrategy()
      config = config.replace(train_distribute=distribution)
      config.data_parallelism = None
    else:
      tf.logging.info("Configuring DataParallelism to replicate the model.")
      config.data_parallelism = devices.data_parallelism(
          daisy_chain_variables=daisy_chain_variables,
          ps_replicas=ps_replicas,
          ps_job=ps_job,
          ps_gpu=ps_gpu,
          schedule=schedule,
          sync=sync,
          worker_gpu=num_gpus,
          worker_replicas=num_async_replicas,
          worker_id=worker_id,
          gpu_order=gpu_order,
          worker_job=worker_job,
          no_data_parallelism=no_data_parallelism)

  return config

```

```
def create_session_config(log_device_placement=False,
                          enable_graph_rewriter=False,
                          gpu_mem_fraction=0.95,
                          use_tpu=False,
                          xla_jit_level=tf.OptimizerOptions.OFF,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0):
  """The TensorFlow Session config to use."""
  if use_tpu:
    graph_options = tf.GraphOptions()
  else:
    if enable_graph_rewriter:
      rewrite_options = rewriter_config_pb2.RewriterConfig()
      rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
      graph_options = tf.GraphOptions(rewrite_options=rewrite_options)
    else:
      graph_options = tf.GraphOptions(
          optimizer_options=tf.OptimizerOptions(
              opt_level=tf.OptimizerOptions.L1,
              do_function_inlining=False,
              global_jit_level=xla_jit_level))

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)

  config = tf.ConfigProto(
      allow_soft_placement=True,
      graph_options=graph_options,
      gpu_options=gpu_options,
      log_device_placement=log_device_placement,
      inter_op_parallelism_threads=inter_op_parallelism_threads,
      intra_op_parallelism_threads=intra_op_parallelism_threads,
      isolate_session_state=True)
  return config

```

```
def create_experiment(
    run_config,
    hparams,
    model_name,
    problem_name,
    data_dir,
    train_steps,
    eval_steps,
    min_eval_frequency=2000,
    eval_throttle_seconds=600,
    schedule="train_and_evaluate",
    export=False,
    decode_hparams=None,
    use_tfdbg=False,
    use_dbgprofile=False,
    eval_early_stopping_steps=None,
    eval_early_stopping_metric=None,
    eval_early_stopping_metric_delta=None,
    eval_early_stopping_metric_minimize=True,
    eval_timeout_mins=240,
    eval_use_test_set=False,
    use_tpu=False,
    use_tpu_estimator=False,
    use_xla=False,
    export_saved_model_api_version=1,
    use_guarantee_const_getter=False,
    additional_train_hooks=None,
    additional_eval_hooks=None,
    warm_start_from=None,
    decode_from_file="",
    decode_to_file="",
    decode_reference="",
    std_server_protocol=None):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("model_dir", run_config.model_dir)
  hparams.add_hparam("data_dir", data_dir)
  hparams.add_hparam("train_steps", train_steps)
  hparams.add_hparam("eval_steps", eval_steps)
  hparams.add_hparam("schedule", schedule)
  hparams.add_hparam("warm_start_from", warm_start_from)
  hparams.add_hparam("std_server_protocol", std_server_protocol)
  hparams.add_hparam("eval_freq_in_steps", min_eval_frequency)
  hparams.add_hparam("eval_timeout_mins", eval_timeout_mins)
  if decode_hparams is not None:
    decode_hparams.add_hparam("decode_from_file", decode_from_file)
    if decode_to_file and not decode_hparams.decode_to_file:
      decode_hparams.decode_to_file = decode_to_file
    if decode_reference and not decode_hparams.decode_reference:
      decode_hparams.decode_reference = decode_reference
  add_problem_hparams(hparams, problem_name)

  # Estimator
  estimator = create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu,
      use_tpu_estimator=use_tpu_estimator,
      use_xla=use_xla,
      export_saved_model_api_version=export_saved_model_api_version,
      use_guarantee_const_getter=use_guarantee_const_getter)

  # Input fns from Problem
  problem = hparams.problem
  train_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.TRAIN,
                                                   hparams)

  dataset_split = "test" if eval_use_test_set else None
  dataset_kwargs = {"dataset_split": dataset_split}
  eval_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.EVAL,
                                                  hparams,
                                                  dataset_kwargs=dataset_kwargs)

  # Export
  exporter = None
  if export:
    def compare_fn(best_eval_result, current_eval_result):
      metric = eval_early_stopping_metric or "loss"
      return current_eval_result[metric] < best_eval_result[metric]

    def serving_input_receiver_fn(hparams, decode_hparams, use_tpu):
      return problem.serving_input_fn(hparams, decode_hparams, use_tpu)

    exporter = tf.estimator.BestExporter(
        name="best",
        serving_input_receiver_fn=serving_input_receiver_fn,
        compare_fn=compare_fn,
        assets_extra=problem.export_assets)

  # Hooks
  validation_monitor_kwargs = dict(
      input_fn=eval_input_fn,
      eval_steps=eval_steps,
      every_n_steps=min_eval_frequency,
      early_stopping_rounds=eval_early_stopping_steps,
      early_stopping_metric=eval_early_stopping_metric,
      early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
  dbgprofile_kwargs = {"output_dir": run_config.model_dir}
  early_stopping_kwargs = dict(
      events_dir=os.path.join(run_config.model_dir, "eval_continuous"),
      tag=eval_early_stopping_metric,
      num_plateau_steps=eval_early_stopping_steps,
      plateau_decrease=eval_early_stopping_metric_minimize,
      plateau_delta=eval_early_stopping_metric_delta,
      every_n_steps=min_eval_frequency)

  # Eval on TPU Pods is not supported yet
  if use_tpu and run_config.tpu_config.num_shards > 8 and "eval" in schedule:
    raise ValueError("Eval is not currently supported on a TPU Pod")

  # In-process eval (and possible early stopping)
  if schedule == "continuous_train_and_eval" and min_eval_frequency:
    tf.logging.warn("ValidationMonitor only works with "
                    "--schedule=train_and_evaluate")
  use_validation_monitor = (
      schedule == "train_and_evaluate" and min_eval_frequency)
  # Distributed early stopping
  local_schedules = ["train_and_evaluate", "continuous_train_and_eval"]
  use_early_stopping = (
      schedule not in local_schedules and eval_early_stopping_steps)
  train_hooks, eval_hooks = create_hooks(
      use_tfdbg=use_tfdbg,
      use_dbgprofile=use_dbgprofile,
      dbgprofile_kwargs=dbgprofile_kwargs,
      use_validation_monitor=use_validation_monitor,
      validation_monitor_kwargs=validation_monitor_kwargs,
      use_early_stopping=use_early_stopping,
      early_stopping_kwargs=early_stopping_kwargs)

  hook_context = HookContext(
      estimator=estimator, problem=problem, hparams=hparams)

  train_hooks += t2t_model.T2TModel.get_train_hooks(model_name, hook_context)
  eval_hooks += t2t_model.T2TModel.get_eval_hooks(model_name, hook_context)
  if additional_train_hooks:
    train_hooks += additional_train_hooks
  if additional_eval_hooks:
    eval_hooks += additional_eval_hooks

  train_hooks = contrib.learn().monitors.replace_monitors_with_hooks(
      train_hooks, estimator)
  eval_hooks = contrib.learn().monitors.replace_monitors_with_hooks(
      eval_hooks, estimator)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=train_steps, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=eval_steps,
      hooks=eval_hooks,
      start_delay_secs=0 if hparams.schedule == "evaluate" else 120,
      throttle_secs=eval_throttle_seconds,
      exporters=exporter)

  return T2TExperiment(estimator, hparams, train_spec, eval_spec,
                       use_validation_monitor, decode_hparams)

```
这个函数首先把train_steps等参数放到hparams里。然后调用add_problem_hparams把这个问题('translate_enzh_wmt8k')相关的参数设置进去。
```
def add_problem_hparams(hparams, problem_name_or_instance):
  """Add problem hparams for the problems."""
  if isinstance(problem_name_or_instance, problem_lib.Problem):
    problem = problem_name_or_instance
  else:
    problem = registry.problem(problem_name_or_instance)
  p_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = p_hparams
```

problem.get_hparams会加载问题相关的超参数，其中比较重要的是加载分词模块(SubwordTextEncoder)：
```
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

重要的部分是构造estimator
```
estimator = create_estimator(
    model_name,
    hparams,
    run_config,
    schedule=schedule,
    decode_hparams=decode_hparams,
    use_tpu=use_tpu,
    use_tpu_estimator=use_tpu_estimator,
    use_xla=use_xla,
    export_saved_model_api_version=export_saved_model_api_version,
    use_guarantee_const_getter=use_guarantee_const_getter)
```

然后是输入：
```
train_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.TRAIN,
                                                 hparams)
```

训练：
```
def train_and_evaluate(self):
  if self._use_validation_monitor:
    tf.logging.warning("EvalSpec not provided. Estimator will not manage "
                       "model evaluation. Assuming ValidationMonitor present "
                       "in train_hooks.")
    self.train()

def train(self, max_steps=None):
  mlperf_log.transformer_print(key=mlperf_log.TRAIN_LOOP)
  mlperf_log.transformer_print(key=mlperf_log.TRAIN_EPOCH, value=0)
  self._estimator.train(
      self._train_spec.input_fn,
      hooks=self._train_spec.hooks,
      max_steps=max_steps or self._train_spec.max_steps)
```

训练首先调用problem.py里的input_fn：
```
def input_fn(self,
             mode,
             hparams,
             data_dir=None,
             params=None,
             config=None,
             force_repeat=False,
             prevent_repeat=False,
             dataset_kwargs=None):
  """Builds input pipeline for problem.

  Args:
    mode: tf.estimator.ModeKeys
    hparams: HParams, model hparams
    data_dir: str, data directory; if None, will use hparams.data_dir
    params: dict, may include "batch_size"
    config: RunConfig; should have the data_parallelism attribute if not using
      TPU
    force_repeat: bool, whether to repeat the data even if not training
    prevent_repeat: bool, whether to not repeat when in training mode.
      Overrides force_repeat.
    dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
      method when called

  Returns:
    (features_dict<str name, Tensor feature>, Tensor targets)
  """
  partition_id, num_partitions = self._dataset_partition(mode, config, params)
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  if config and config.use_tpu:
    num_threads = 64
  else:
    num_threads = data_reader.cpu_count() if is_training else 1
  data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)
  dataset_kwargs = dataset_kwargs or {}
  dataset_kwargs.update({
      "mode": mode,
      "data_dir": data_dir,
      "num_threads": num_threads,
      "hparams": hparams,
      "partition_id": partition_id,
      "num_partitions": num_partitions,
  })
  return data_reader.input_fn(
      self.dataset(**dataset_kwargs),
      self.filepattern(data_dir, mode),
      self.skip_random_fraction_when_training,
      self.batch_size_means_tokens,
      self.get_hparams().batch_size_multiplier,
      self.max_length(hparams),
      mode,
      hparams,
      data_dir=data_dir,
      params=params,
      config=config,
      force_repeat=force_repeat,
      prevent_repeat=prevent_repeat)

```


下面是batch的逻辑，主要是把相似长度的句子放到一起：
```
def batching_scheme(batch_size,
                    max_length,
                    min_length_bucket,
                    length_bucket_step,
                    drop_long_sequences=False,
                    shard_multiplier=1,
                    length_multiplier=1,
                    min_length=0):
  """A batching scheme based on model hyperparameters.

  Every batch contains a number of sequences divisible by `shard_multiplier`.

  Args:
    batch_size: int, total number of tokens in a batch.
    max_length: int, sequences longer than this will be skipped. Defaults to
      batch_size.
    min_length_bucket: int
    length_bucket_step: float greater than 1.0
    drop_long_sequences: bool, if True, then sequences longer than
      `max_length` are dropped.  This prevents generating batches with
      more than the usual number of tokens, which can cause out-of-memory
      errors.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.
    min_length: int, sequences shorter than this will be skipped.

  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example

  Raises:
    ValueError: If min_length > max_length
  """
  max_length = max_length or batch_size
  if max_length < min_length:
    raise ValueError("max_length must be greater or equal to min_length")

  boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                  length_bucket_step)
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  max_length *= length_multiplier

  batch_sizes = [
      max(1, batch_size // length) for length in boundaries + [max_length]
  ]
  max_batch_size = max(batch_sizes)
  # Since the Datasets API only allows a single constant for window_size,
  # and it needs divide all bucket_batch_sizes, we pick a highly-composite
  # window size and then round down all batch sizes to divisors of that window
  # size, so that a window can always be divided evenly into batches.
  # TODO(noam): remove this when Dataset API improves.
  highly_composite_numbers = [
      1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
      2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
      83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
      720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
      7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
      36756720, 43243200, 61261200, 73513440, 110270160
  ]
  window_size = max(
      [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
  divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
  batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
  window_size *= shard_multiplier
  batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
  # The Datasets API splits one window into multiple batches, which
  # produces runs of many consecutive batches of the same size.  This
  # is bad for training.  To solve this, we will shuffle the batches
  # using a queue which must be several times as large as the maximum
  # number of batches per window.
  max_batches_per_window = window_size // min(batch_sizes)
  shuffle_queue_size = max_batches_per_window * 3

  ret = {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "min_length": min_length,
      "max_length": (max_length if drop_long_sequences else 10**9),
      "shuffle_queue_size": shuffle_queue_size,
  }
  return ret

```

根据句子的长度来分桶(bucket)，<class 'list'>: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196, 215, 236]
```
def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
  """A default set of length-bucket boundaries."""
  assert length_bucket_step > 1.0
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x = max(x + 1, int(x * length_bucket_step))
  return boundaries
```

```
batch_sizes = [
    max(1, batch_size // length) for length in boundaries + [max_length]
]
```
batch_size是一个batch的token的数量；batch_sizes就是不同的桶的batch大学，桶的长度越长，batch越小。
后面的代码都是为了优化，让batch是一个合理的值。我们暂且不管。最终把dataset分成不同的桶：
```
dataset = dataset.apply(
    tf.data.experimental.bucket_by_sequence_length(
        example_length, cur_batching_scheme["boundaries"],
        cur_batching_scheme["batch_sizes"]))
```

接着把features都reshape成4维：
```
def standardize_shapes(features, batch_size=None):
  """Set the right shapes for the features."""
  for fname in ["inputs", "targets"]:
    if fname not in features:
      continue
    f = features[fname]
    while len(f.get_shape()) < 4:
      f = tf.expand_dims(f, axis=-1)
    features[fname] = f

  if batch_size:
    # Ensure batch size is set on all features
    for _, t in six.iteritems(features):
      shape = t.get_shape().as_list()
      shape[0] = batch_size
      t.set_shape(t.get_shape().merge_with(shape))
      # Assert shapes are fully known
      t.get_shape().assert_is_fully_defined()

  return features
```

接下来调用model_fn：
```
def wrapping_model_fn(features, labels, mode, params=None, config=None):
  return model_cls.estimator_model_fn(
      hparams,
      features,
      labels,
      mode,
      config=config,
      params=params,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu)
```

```
@classmethod
def estimator_model_fn(cls,
                       hparams,
                       features,
                       labels,
                       mode,
                       config=None,
                       params=None,
                       decode_hparams=None,
                       use_tpu=False):
  """Model fn for Estimator.

  Args:
    hparams: HParams, model hyperparameters
    features: dict<str name, Tensor feature>
    labels: Tensor
    mode: tf.estimator.ModeKeys
    config: RunConfig, possibly with data_parallelism attribute
    params: dict, may include batch_size, use_tpu
    decode_hparams: HParams, used when mode == PREDICT.
    use_tpu: A bool, whether to build the inference graph for TPU.

  Returns:
    TPUEstimatorSpec if use tpu else EstimatorSpec
  """
  if mode == tf.estimator.ModeKeys.TRAIN:
    create_dummy_vars()
  hparams = hparams_lib.copy_hparams(hparams)

  # Instantiate model
  data_parallelism = None
  if not use_tpu and config:
    data_parallelism = config.data_parallelism
  reuse = tf.get_variable_scope().reuse
  model = cls(
      hparams,
      mode,
      data_parallelism=data_parallelism,
      decode_hparams=decode_hparams,
      _reuse=reuse)

  # PREDICT mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    if use_tpu:
      inputs = features.get("inputs")
      if inputs is None:
        inputs = features.get("targets")
      if inputs is None:
        inputs = features["infer_targets"]
      shape = inputs.get_shape().as_list()
      if shape[0] is None:
        shape[0] = decode_hparams.batch_size or hparams.batch_size
      if shape[1] is None:
        shape[1] = hparams.max_input_seq_length or hparams.max_length
      inputs.set_shape(shape)
    return model.estimator_spec_predict(features, use_tpu=use_tpu)

  # TRAIN and EVAL modes
  if hparams.eval_run_autoregressive and mode == tf.estimator.ModeKeys.EVAL:
    logits, losses_dict = model.eval_autoregressive(features)
  else:
    logits, losses_dict = model(features)  # pylint: disable=not-callable

  # Support model-generated labels by overriding features["targets"] with
  # logits["self_generated_targets"].
  if isinstance(logits, dict) and "self_generated_targets" in logits:
    # Overwrite 'features["targets"]' and 'labels'
    # by logits["self_generated_targets"].
    tf.logging.info("Replacing targets with model-provided targets.")
    features["targets"] = labels = logits.pop("self_generated_targets")
    assert list(logits.keys()) == ["logits"], (
        # See "Returns" in the "top" method docstring for the expected
        # "logits" format when targets are generated at training time.
        "Expect only key 'logits' when there is 'self_generated_targets'. "
        "Found {}".format(logits.keys())
    )
    # Recover the original logits tensor from the logits dict.
    logits = logits["logits"]  # Can be a tf.Tensor or a dict.

  # Set known shapes
  if common_layers.is_xla_compiled():
    if isinstance(logits, dict):
      for k, v in sorted(six.iteritems(logits)):
        if "scalar/" in k:
          continue

        shape = v.get_shape().as_list()
        if shape[0] is None:
          shape[0] = params["batch_size"]
        if shape[1] is None:
          shape[1] = hparams.max_length
        v.set_shape(shape)
    else:
      shape = logits.get_shape().as_list()
      if shape[0] is None:
        shape[0] = params["batch_size"]
      if shape[1] is None:
        shape[1] = hparams.max_length
      logits.set_shape(shape)

  assert "training" in losses_dict

  # Attack mode
  if mode == "attack":
    return logits

  # Summarize losses
  model._summarize_losses(losses_dict)  # pylint: disable=protected-access

  # Accumulate losses
  loss = sum(losses_dict[key] for key in sorted(losses_dict.keys()))

  # EVAL mode
  if mode == tf.estimator.ModeKeys.EVAL:
    return model.estimator_spec_eval(features, logits, labels, loss,
                                     losses_dict)

  # TRAIN mode
  assert mode == tf.estimator.ModeKeys.TRAIN
  num_async_replicas = 1
  if config and not use_tpu:
    num_async_replicas = config.t2t_device_info["num_async_replicas"]
  return model.estimator_spec_train(
      loss, num_async_replicas=num_async_replicas, use_tpu=use_tpu)
```

这里的核心是model=cls()，进入Transformer
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

Transformer继承了t2t_model.T2TModel。这是它的说明：
```
class T2TModel(base.Layer):
  """Abstract base class for models.

  `T2TModel` has three typical usages:

  1. Estimator: The method `make_estimator_model_fn` builds a `model_fn` for
     the tf.Estimator workflow of training, evaluation, and prediction.
     It performs the method `call`, which performs the core computation,
     followed by `estimator_spec_train`, `estimator_spec_eval`, or
     `estimator_spec_predict` depending on the tf.Estimator mode.
  2. Layer: The method `call` enables `T2TModel` to be used a callable by
     itself. It calls the following methods:

     * `bottom`, which transforms features according to `problem_hparams`' input
       and target `Modality`s;
     * `body`, which takes features and performs the core model computation to
        return output and any auxiliary loss terms;
     * `top`, which takes features and the body output, and transforms them
       according to `problem_hparams`' input and target `Modality`s to return
       the final logits;
     * `loss`, which takes the logits, forms any missing training loss, and sums
       all loss terms.
  3. Inference: The method `infer` enables `T2TModel` to make sequence
     predictions by itself.

```


```
def model_fn(self, features):
  with tf.variable_scope(tf.get_variable_scope(), use_resource=True) as vs:
    self._add_variable_scope("model_fn", vs)
    transformed_features = self.bottom(features)

    if self.hparams.activation_dtype == "bfloat16":
      for k, v in sorted(six.iteritems(transformed_features)):
        if v.dtype == tf.float32:
          transformed_features[k] = tf.cast(v, tf.bfloat16)

    with tf.variable_scope("body") as body_vs:
      self._add_variable_scope("body", body_vs)
      log_info("Building model body")
      body_out = self.body(transformed_features)
    output, losses = self._normalize_body_output(body_out)

    if "training" in losses:
      log_info("Skipping T2TModel top and loss because training loss "
               "returned from body")
      logits = output
    else:
      logits = self.top(output, features)
      losses["training"] = 0.0
      if (self._hparams.mode != tf.estimator.ModeKeys.PREDICT and
          self._hparams.mode != "attack"):
        losses["training"] = self.loss(logits, features)

    return logits, losses

```

分为bottom、body、top和loss四个步骤，我们先看bottom：
```
def bottom(self, features):
  """Transforms features to feed into body.

  Args:
    features: dict of str to Tensor. Typically it is the preprocessed data
      batch after Problem's preprocess_example().

  Returns:
    transformed_features: dict of same key-value pairs as features. The value
      Tensors are newly transformed.
  """
  if not self._problem_hparams:
    log_warn("Without a Problem, T2TModel.bottom is a passthrough.")
    return features

  transformed_features = collections.OrderedDict()
  all_previous_modalities = []
  target_modality = _create_target_modality(self._problem_hparams.modality)

  # Transform features via its corresponding modality.
  for feature_name, modality in sorted(
      six.iteritems(self._problem_hparams.modality)):
    if feature_name not in features:
      tf.logging.warning("Missing feature %s - ignoring." % feature_name)
      continue
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    modality_name = self._hparams.name.get(
        feature_name,
        modalities.get_name(modality))(self._hparams, vocab_size)
    # Use if-else clauses to preserve behavior of previous changes: namely,
    # the variable scope name for the targets feature if there is only one
    # target modality; and to reuse variable scopes for only input modalities.
    if feature_name in target_modality:
      if len(target_modality) > 1:
        variable_scope_name = "%s/%s" % (modality_name, feature_name)
      else:
        variable_scope_name = modality_name
      bottom = self._hparams.bottom.get(
          feature_name,
          modalities.get_targets_bottom(modality))
      # TODO(aidangomez): share variables?
      with tf.variable_scope(variable_scope_name) as vs:
        self._add_variable_scope(variable_scope_name, vs)
        log_info("Transforming feature '%s' with %s.targets_bottom",
                 feature_name,
                 modality_name)
        transformed_features[feature_name] = bottom(features[feature_name],
                                                    self._hparams,
                                                    vocab_size)
    else:
      bottom = self._hparams.bottom.get(feature_name,
                                        modalities.get_bottom(modality))
      do_reuse = modality_name in all_previous_modalities
      with tf.variable_scope(modality_name, reuse=do_reuse) as vs:
        self._add_variable_scope(modality_name, vs)
        log_info("Transforming feature '%s' with %s.bottom",
                 feature_name,
                 modality_name)
        transformed_features[feature_name] = bottom(features[feature_name],
                                                    self._hparams,
                                                    vocab_size)
      all_previous_modalities.append(modality_name)

  for key in features:
    if key not in transformed_features:
      # For features without a modality, we pass them along as is
      transformed_features[key] = features[key]
    else:
      # Other features get passed along with the "raw" suffix
      transformed_features[key + "_raw"] = features[key]

  return transformed_features
```
核心代码是transformed_features[feature_name] = bottom(features[feature_name], self._hparams, vocab_size)，那么具体进入哪个bottom函数呢？
```
def get_bottom(modality_type, value=None):
  """Gets default bottom transformation; if none available, return value."""
  if modality_type == ModalityType.AUDIO:
    return audio_bottom
  elif modality_type == ModalityType.AUDIO_SPECTRAL:
    return audio_spectral_bottom
  elif modality_type in (ModalityType.CLASS_LABEL,
                         ModalityType.MULTI_LABEL,
                         ModalityType.ONE_HOT_CLASS_LABEL,
                         ModalityType.SIGMOID_CLASS_LABEL,
                         ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL,
                         ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL):
    return class_label_bottom
  elif modality_type in (ModalityType.CTC_SYMBOL,
                         ModalityType.SYMBOL,
                         ModalityType.SYMBOL_WEIGHTS_ALL):
    return symbol_bottom
  elif modality_type in (ModalityType.GENERIC_L2_LOSS,
                         ModalityType.IDENTITY,
                         ModalityType.IDENTITY_SYMBOL,
                         ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM):
    return identity_bottom
  elif modality_type == ModalityType.IMAGE:
    return image_bottom
  elif modality_type in (ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
                         ModalityType.IMAGE_CHANNEL_COMPRESS):
    return image_channel_compress_bottom
  elif modality_type in (ModalityType.REAL,
                         ModalityType.REAL_L2_LOSS,
                         ModalityType.REAL_LOG_POISSON_LOSS):
    return real_bottom
  elif modality_type == ModalityType.SPEECH_RECOGNITION:
    return speech_recognition_bottom
  elif modality_type == ModalityType.SYMBOL_ONE_HOT:
    return symbol_one_hot_bottom
  elif modality_type in (ModalityType.VIDEO,
                         ModalityType.VIDEO_L1,
                         ModalityType.VIDEO_L2):
    return video_bottom
  elif modality_type == ModalityType.VIDEO_BITWISE:
    return video_bitwise_bottom
  elif modality_type == ModalityType.VIDEO_IDENTITY:
    return video_identity_bottom
  elif modality_type in (ModalityType.VIDEO_L1_RAW,
                         ModalityType.VIDEO_L2_RAW):
    return video_raw_bottom
  elif modality_type == ModalityType.VIDEO_PIXEL_NOISE:
    return video_pixel_noise_bottom
  return value
```

对于我们的问题，inputs和targets分别对应如下的symbol_bottom和symbol_targets_bottom：
```
def symbol_bottom(x, model_hparams, vocab_size):
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    return _symbol_bottom_simple(
        x, model_hparams, vocab_size, "shared", reuse=None)
  return _symbol_bottom_simple(
      x, model_hparams, vocab_size, "input_emb", reuse=None)


def symbol_targets_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for target symbols."""
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    try:
      return _symbol_bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=True)
    except ValueError:
      # perhaps there were no inputs, and this is a new variable.
      return _symbol_bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=None)
  else:
    return _symbol_bottom_simple(
        x, model_hparams, vocab_size, "target_emb", reuse=None)
```
由于我们的例子没有share embedding，所以调用的是_symbol_bottom_simple，我们以symbol_targets_bottom为例来看看其代码：
```
def _symbol_bottom_simple(x, model_hparams, vocab_size, name, reuse):
  """Bottom transformation for symbols."""
  with tf.variable_scope(name, reuse=reuse):
    # Ensure the inputs are 3-D
    if len(x.get_shape()) == 4:
      x = tf.squeeze(x, axis=3)
    while len(x.get_shape()) < 3:
      x = tf.expand_dims(x, axis=-1)

    var = get_weights(model_hparams, vocab_size)
    x = common_layers.dropout_no_scaling(
        x, 1.0 - model_hparams.symbol_dropout)
    ret = common_layers.gather(var, x)
    if model_hparams.multiply_embedding_mode == "sqrt_depth":
      ret *= model_hparams.hidden_size**0.5
    ret *= tf.expand_dims(
        common_layers.cast_like(tf.not_equal(x, 0), ret), -1)
    return ret
```
首先把输入变成(batch, timestep, 1)，然后调用get_weights。
```
def get_weights(model_hparams, vocab_size, hidden_dim=None):
  """Create or get concatenated embedding or softmax variable.

  Args:
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.
    hidden_dim: dim of the variable. Defaults to _model_hparams' hidden_size

  Returns:
     a list of num_shards Tensors.
  """
  if hidden_dim is None:
    hidden_dim = model_hparams.hidden_size
  num_shards = model_hparams.symbol_modality_num_shards
  shards = []
  for i in range(num_shards):
    shard_size = (vocab_size // num_shards) + (
        1 if i < vocab_size % num_shards else 0)
    var_name = "weights_%d" % i
    shards.append(
        tf.get_variable(
            var_name, [shard_size, hidden_dim],
            initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5)))
  if num_shards == 1:
    ret = shards[0]
  else:
    ret = tf.concat(shards, 0)
  # Convert ret to tensor.
  if not tf.executing_eagerly():
    ret = common_layers.convert_gradient_to_tensor(ret)
  return ret

```
get_weights就是创建Embedding的矩阵，不过和一般的实现不太一样的是它会把vocab_size的词典分成num_shard个shard。


<a name='img9'>![](/img/t2t-code/9.png)</a>
*图：shard之后的Embedding矩阵*

```
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x
```

然后我们返回_symbol_bottom_simple，接着执行的是：
```
x = common_layers.dropout_no_scaling(
    x, 1.0 - model_hparams.symbol_dropout)

def dropout_no_scaling(x, keep_prob):
  """Like tf.nn.dropout, but does not scale up.  Works on integers also.
```
这里是做dropout，但是又不希望scale，所以使用了common_layers的dropout_no_scaling而不是tf.nn.dropout。
然后通过tf.gather得到下标对应的Embedding：
```
ret = common_layers.gather(var, x)
```
上面使用的是common_layers.gather，对于非tpu来说就是tf.gather，对于TPU的Embedding使用矩阵乘法更快。
接着乘以sqrt(hidden_size)，不明白其含义。
最后是把x中为0的Embedding设置为零：
```
ret *= tf.expand_dims(
    common_layers.cast_like(tf.not_equal(x, 0), ret), -1)
```


从bottom返回后回到model_fn，有一段bfloat16的转换，我们可以跳过，接下来就是body：
```
with tf.variable_scope("body") as body_vs:
  self._add_variable_scope("body", body_vs)
  log_info("Building model body")
  body_out = self.body(transformed_features)
output, losses = self._normalize_body_output(body_out)
```
基类(T2TModel)的body没有实现，这里会调用子类(Transformer)的body，这是最关键的代码：

```
def body(self, features):
  """Transformer main model_fn.

  Args:
    features: Map of features to the model. Should contain the following:
        "inputs": Transformer inputs. [batch_size, input_length, 1,
          hidden_dim].
        "targets": Target decoder outputs. [batch_size, decoder_length, 1,
          hidden_dim]
        "target_space_id": A scalar int from data_generators.problem.SpaceID.

  Returns:
    Final decoder representation. [batch_size, decoder_length, hidden_dim]
  """
  hparams = self._hparams

  losses = []

  if self.has_input:
    inputs = self._prepare_inputs_for_body(features)
    target_space = features["target_space_id"]
    encoder_output, encoder_decoder_attention_bias = self.encode(
        inputs, target_space, hparams, features=features, losses=losses)
  else:
    encoder_output, encoder_decoder_attention_bias = (None, None)

  targets = features["targets"]
  targets_shape = common_layers.shape_list(targets)
  targets = common_layers.flatten4d3d(targets)
  decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
      targets, hparams, features=features)

  # Not all subclasses of Transformer support keyword arguments related to
  # recurrent memory, so only pass these arguments if memory is enabled.
  decode_kwargs = {}
  if self.recurrent_memory_by_layer is not None:
    # TODO(kitaev): The chunk_number feature currently has the same shape as
    # "targets", but this is only for the purposes of sharing sharding code.
    # In fact every token within an example must have the same chunk number.
    chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
    chunk_number_each_example = chunk_number_each_token[:, 0]
    # Uncomment the code below to verify that tokens within a batch share the
    # same chunk number:
    # with tf.control_dependencies([
    #     tf.assert_equal(chunk_number_each_token,
    #                     chunk_number_each_example[:, None])
    # ]):
    #   chunk_number_each_example = tf.identity(chunk_number_each_example)
    decode_kwargs = dict(
        recurrent_memory_by_layer=self.recurrent_memory_by_layer,
        chunk_number=chunk_number_each_example,
        )
  decoder_output = self.decode(
      decoder_input,
      encoder_output,
      encoder_decoder_attention_bias,
      decoder_self_attention_bias,
      hparams,
      nonpadding=features_to_nonpadding(features, "targets"),
      losses=losses,
      **decode_kwargs
      )
  expected_attentions = features.get("expected_attentions")
  if expected_attentions is not None:
    attention_loss = common_attention.encoder_decoder_attention_loss(
        expected_attentions, self.attention_weights,
        hparams.expected_attention_loss_type,
        hparams.expected_attention_loss_multiplier)
    return decoder_output, {"attention_loss": attention_loss}

  ret = tf.reshape(decoder_output, targets_shape)
  if losses:
    return ret, {"extra_loss": tf.add_n(losses)}
  else:
    return ret
```


inputs = self._prepare_inputs_for_body(features)返回的就是features['inputs']，shape是[batch, timestep, 1, 512]。
target_space是shape为(1,)的int32。然后是encoder的代码：
```
encoder_output, encoder_decoder_attention_bias = self.encode(
    inputs, target_space, hparams, features=features, losses=losses)

def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)
```

最终调用的是transformer_encode函数。代码如下：
```
def transformer_encode(encoder_function, inputs, target_space, hparams,
                       attention_weights=None, features=None, losses=None,
                       prepare_encoder_fn=None, **kwargs):
  """Encode transformer inputs.

  Args:
    encoder_function: the encoder function
    inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
      will be flattened along the two spatial dimensions.
    target_space: scalar, target space ID.
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    losses: optional list onto which to append extra training losses
    prepare_encoder_fn: optional, alternative to transformer_prepare_encoder.
    **kwargs: additional arguments to pass to encoder_function

  Returns:
    Tuple of:
        encoder_output: Encoder representation.
            [batch_size, input_length, hidden_dim]
        encoder_decoder_attention_bias: Bias and mask weights for
            encoder-decoder attention. [batch_size, input_length]
  """
  inputs = common_layers.flatten4d3d(inputs)

  if not prepare_encoder_fn:
    prepare_encoder_fn = transformer_prepare_encoder
  encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
      prepare_encoder_fn(
          inputs, target_space, hparams, features=features))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  encoder_input = tf.nn.dropout(encoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  attn_bias_for_padding = None
  # Otherwise the encoder will just use encoder_self_attention_bias.
  if hparams.unidirectional_encoder:
    attn_bias_for_padding = encoder_decoder_attention_bias

  encoder_output = encoder_function(
      encoder_input,
      self_attention_bias,
      hparams,
      nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=attention_weights,
      make_image_summary=not common_layers.is_xla_compiled(),
      losses=losses,
      attn_bias_for_padding=attn_bias_for_padding,
      **kwargs)

  return encoder_output, encoder_decoder_attention_bias
```
这个函数首先把4d的向量reshape成3d。我们的输入开始是3d的，后来expand成4d，现在又回来成3d。为什么要这么麻烦呢？可能是t2t为了解决所有的seq2seq问题，比如视频的输入，所以定义了通用的4d向量。

```
def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result
```
然后进入transformer_prepare_encoder：
```
def transformer_prepare_encoder(inputs, target_space, hparams, features=None,
                                type_ids=None, num_types=None,
                                reuse_target_embedding=tf.AUTO_REUSE):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.
    type_ids: optional, an int64 Tensor of shape [batch, length] that allows
      for adding type embeddings, similar to positional embeddings.
    num_types: optional, an int that decides the number of types in type_ids.
    reuse_target_embedding: option to reuse variable name in the case that
      symbol modalities are reused between inputs/targets.

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  if features and "inputs_segmentation" in features:
    # Packed dataset.  Keep the examples from seeing each other.
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      encoder_self_attention_bias = (
          common_attention.attention_bias_same_segment(
              inputs_segmentation, inputs_segmentation))
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(targets_segmentation,
                                                     inputs_segmentation))
  else:
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      # Usual case - not a packed dataset.
      encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])
  if target_space is not None and hparams.get("use_target_space_embedding",
                                              True):
    # Append target_space_id embedding to inputs.
    emb_target_space = common_layers.embedding(
        target_space,
        32,
        ishape_static[-1],
        name="target_space_embedding",
        dtype=hparams.get("activation_dtype", "float32"),
        reuse=reuse_target_embedding)
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space
  if hparams.pos == "timing":
    if inputs_position is not None:
      encoder_input = common_attention.add_timing_signal_1d_given_position(
          encoder_input, inputs_position)
    else:
      encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  elif hparams.pos == "timing_from_features":
    encoder_input = common_attention.add_timing_signals_from_features(
        encoder_input, features, hparams.position_features)
  elif hparams.pos == "emb":
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "inputs_positional_embedding",
        inputs_position)

  # Add type embeddings
  if type_ids is not None:
    if not num_types:
      raise ValueError("Need to set num_types as well.")
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, num_types, "inputs_type_embedding", type_ids)

  encoder_self_attention_bias = common_layers.cast_like(
      encoder_self_attention_bias, encoder_input)
  encoder_decoder_attention_bias = common_layers.cast_like(
      encoder_decoder_attention_bias, encoder_input)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)

```
这个函数首先判断"inputs_segmentation" in features，我们的结果是False，所以走else分支：
首先执行：
```
encoder_padding = common_attention.embedding_to_padding(encoder_input)
```
这个函数是判断输入的某个位置是否是padding，因为之前会把padding的embedding设置为全零，所以这里用这种方法来判断。但是这是一种"tricky"，因为理论上非padding的embedding也是有可能(很小的概率)全零的。返回的tensor用1(True)表示padding，0表示非padding。

```
@expert_utils.add_name_scope()
def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.

  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].

  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = memory_padding * large_compatible_negative(memory_padding.dtype)
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)
```
这个函数给padding(memory_padding就是上的encoder_padding)的地方乘以一个很大的复数(-1e9)，这样的padding的地方是很大的负数，非padding的是0。这个tensor在后面计算attention会用到，它的作用是让padding的attention是0。它的shape是(batch, 1, 1, timestep)。

最后把这个值赋给encoder_self_attention_bias和encoder_decoder_attention_bias。前者用于transformer的self attention，后者用于decoder对encoder的attention。接着Append target_space_id embedding to inputs：
```
# Append target_space_id embedding to inputs.
emb_target_space = common_layers.embedding(
    target_space,
    32,
    ishape_static[-1],
    name="target_space_embedding",
    dtype=hparams.get("activation_dtype", "float32"),
    reuse=reuse_target_embedding)
```

emb_target_space的shape是(1, 512)。然后把它加到输入里：
```
emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
encoder_input += emb_target_space
```

接着加入position encoding：
```

if hparams.pos == "timing":
  if inputs_position is not None:
    encoder_input = common_attention.add_timing_signal_1d_given_position(
        encoder_input, inputs_position)
  else:
    encoder_input = common_attention.add_timing_signal_1d(encoder_input)
```
这里走的是else的分支，调用add_timing_signal_1d：
```
@expert_utils.add_name_scope()
def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor the same shape as x.
  """
  length = common_layers.shape_list(x)[1]
  channels = common_layers.shape_list(x)[2]
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale,
                                start_index)
  return x + common_layers.cast_like(signal, x)
```
也就是增加sin和cos的相对位置编码，具体实现为：
```
def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      tf.maximum(tf.to_float(num_timescales) - 1, 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal

```
具体实现和论文稍有出入，请参考[这个Issue](https://github.com/tensorflow/tensor2tensor/pull/177)。
对于翻译来说，没有type embedding，所以跳过了它。

最后返回：
```
return (encoder_input, encoder_self_attention_bias,
        encoder_decoder_attention_bias)
```

预处理完成后就是真正的encoder_function：
```
encoder_output = encoder_function(
    encoder_input,
    self_attention_bias,
    hparams,
    nonpadding=features_to_nonpadding(features, "inputs"),
    save_weights_to=attention_weights,
    make_image_summary=not common_layers.is_xla_compiled(),
    losses=losses,
    attn_bias_for_padding=attn_bias_for_padding,
    **kwargs)
```
在进入之前还有一个features_to_nonpadding：
```
def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None
```
在我们这里它返回的是None。然后进入：
```
def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        attn_bias_for_padding=None):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convolutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    attn_bias_for_padding: Padded attention bias in case a unidirectional
      encoder is being used where future attention is masked.

  Returns:
    y: a Tensors
  """
  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        attention_bias = attn_bias_for_padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          if layer < hparams.get("num_area_layers", 0):
            max_area_width = hparams.get("max_area_width", 1)
            max_area_height = hparams.get("max_area_height", 1)
            memory_height = hparams.get("memory_height", 1)
          else:
            max_area_width = 1
            max_area_height = 1
            memory_height = 1
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              hard_attention_k=hparams.get("hard_attention_k", 0),
              gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
              max_area_width=max_area_width,
              max_area_height=max_area_height,
              memory_height=memory_height,
              area_key_mode=hparams.get("area_key_mode", "none"),
              area_value_mode=hparams.get("area_value_mode", "none"),
              training=(hparams.get("mode", tf.estimator.ModeKeys.TRAIN)
                        == tf.estimator.ModeKeys.TRAIN))
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              pad_remover,
              conv_padding="SAME",
              nonpadding_mask=nonpadding,
              losses=losses)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)
```

我们先看一下nonpadding的说明：
```
nonpadding: optional Tensor with shape [batch_size, encoder_length]
  indicating what positions are not padding.  This must either be
  passed in, which we do for "packed" datasets, or inferred from
  encoder_self_attention_bias.  The knowledge about padding is used
  for pad_remover(efficiency) and to mask out padding in convolutional
  layers.
```
我们这里传入的是None，所以模型会通过encoder_self_attention_bias的tricky的方法来判断某个位置是否padding。

下面的代码是处理padding：
```
with tf.variable_scope(name):
  if nonpadding is not None:
    padding = 1.0 - nonpadding
  else:
    attention_bias = encoder_self_attention_bias
    if attn_bias_for_padding is not None:
      attention_bias = attn_bias_for_padding
    padding = common_attention.attention_bias_to_padding(attention_bias)
    nonpadding = 1.0 - padding
```
实际调用的是common_attention.attention_bias_to_padding(attention_bias)：
```
@expert_utils.add_name_scope()
def attention_bias_to_padding(attention_bias,
                              cast_fn=(lambda x: tf.cast(x, tf.float32))):
  """Inverse of attention_bias_ignore_padding().

  Args:
    attention_bias: a `Tensor` with shape [batch, 1, 1, memory_length], as
      returned by attention_bias_ignore_padding().
    cast_fn: function used to cast to output type.

  Returns:
    a Tensor with shape [batch, memory_length] with 1.0 in padding positions
    and 0.0 in non-padding positions. Type is determined by cast_fn.
  """
  # `attention_bias` is a large negative number in padding positions and 0.0
  # elsewhere.
  return tf.squeeze(cast_fn(tf.less(attention_bias, -1)), axis=[1, 2])

```
因为attention_bias要么是零(非padding)、要么是很大的负数，所以通过tf.less(attention_bias, -1)就能区分两者。

接下来是pad_remover = expert_utils.PadRemover(padding)，构造一个PadRemover对象：
```
class PadRemover(object):
  """Helper to remove padding from a tensor before sending to the experts.

  The padding is computed for one reference tensor containing the padding mask
  and then can be applied to any other tensor of shape [dim_origin,...].

  Ex:
      input = [
        [tok1, tok2],
        [tok3, tok4],
        [0, 0],
        [0, 0],
        [tok5, tok6],
        [0, 0],
      ]
      output = [
        [tok1, tok2],
        [tok3, tok4],
        [tok5, tok6],
      ]
  """
```
接下来就是逐层构造Transformer：
```
for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
    ....
```

这个循环核心就是构造self-attention和ffn，我们先看self-attention：
```
with tf.variable_scope("self_attention"):
  if layer < hparams.get("num_area_layers", 0):
    max_area_width = hparams.get("max_area_width", 1)
    max_area_height = hparams.get("max_area_height", 1)
    memory_height = hparams.get("memory_height", 1)
  else:
    max_area_width = 1
    max_area_height = 1
    memory_height = 1
  y = common_attention.multihead_attention(
      common_layers.layer_preprocess(x, hparams),
      None,
      encoder_self_attention_bias,
      hparams.attention_key_channels or hparams.hidden_size,
      hparams.attention_value_channels or hparams.hidden_size,
      hparams.hidden_size,
      hparams.num_heads,
      hparams.attention_dropout,
      attention_type=hparams.self_attention_type,
      max_relative_position=hparams.max_relative_position,
      heads_share_relative_embedding=(
          hparams.heads_share_relative_embedding),
      add_relative_to_values=hparams.add_relative_to_values,
      save_weights_to=save_weights_to,
      make_image_summary=make_image_summary,
      dropout_broadcast_dims=attention_dropout_broadcast_dims,
      max_length=hparams.get("max_length"),
      vars_3d=hparams.get("attention_variables_3d"),
      activation_dtype=hparams.get("activation_dtype", "float32"),
      weight_dtype=hparams.get("weight_dtype", "float32"),
      hard_attention_k=hparams.get("hard_attention_k", 0),
      gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
      max_area_width=max_area_width,
      max_area_height=max_area_height,
      memory_height=memory_height,
      area_key_mode=hparams.get("area_key_mode", "none"),
      area_value_mode=hparams.get("area_value_mode", "none"),
      training=(hparams.get("mode", tf.estimator.ModeKeys.TRAIN)
                == tf.estimator.ModeKeys.TRAIN))
  x = common_layers.layer_postprocess(x, y, hparams)
```
第一段逻辑没看懂，走的是else分支，然后很长的代码是调用common_attention.multihead_attention，参数很多，其实只有一个函数：
```
def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product",
                        max_relative_position=None,
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        image_shapes=None,
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,
                        layer_collection=None,
                        recurrent_memory=None,
                        chunk_number=None,
                        hard_attention_k=0,
                        gumbel_noise_weight=0.0,
                        max_area_width=1,
                        max_area_height=1,
                        memory_height=1,
                        area_key_mode="mean",
                        area_value_mode="sum",
                        training=True,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    heads_share_relative_embedding: boolean to share relative embeddings
    add_relative_to_values: a boolean for whether to add relative component to
                            values.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory: An optional transformer_memory.RecurrentMemory, which
      retains state across chunks. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    training: indicating if it is in the training mode.
    **kwargs (dict): Parameters for the attention function.

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  vars_3d_num_heads = num_heads if vars_3d else 0

  if layer_collection is not None:
    if cache is not None:
      raise ValueError("KFAC implementation only supports cache is None.")
    if vars_3d:
      raise ValueError("KFAC implementation does not support 3d vars.")

  if recurrent_memory is not None:
    if memory_antecedent is not None:
      raise ValueError("Recurrent memory requires memory_antecedent is None.")
    if cache is not None:
      raise ValueError("Cache is not supported when using recurrent memory.")
    if vars_3d:
      raise ValueError("3d vars are not supported when using recurrent memory.")
    if layer_collection is not None:
      raise ValueError("KFAC is not supported when using recurrent memory.")
    if chunk_number is None:
      raise ValueError("chunk_number is required when using recurrent memory.")

  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    if recurrent_memory is not None:
      (
          recurrent_memory_transaction,
          query_antecedent, memory_antecedent, bias,
      ) = recurrent_memory.pre_attention(
          chunk_number,
          query_antecedent, memory_antecedent, bias,
      )

    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads,
                            layer_collection=layer_collection)
    if cache is not None:
      if attention_type not in ["dot_product", "dot_product_relative"]:
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        q_filter_width, q_padding, "q",
                                        vars_3d_num_heads=vars_3d_num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

    q = split_heads(q, num_heads)
    if cache is None:
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      if max_area_width > 1 or max_area_height > 1:
        x = area_attention.dot_product_area_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            dropout_broadcast_dims=dropout_broadcast_dims,
            max_area_width=max_area_width,
            max_area_height=max_area_height,
            memory_height=memory_height,
            area_key_mode=area_key_mode,
            area_value_mode=area_value_mode,
            training=training)
      else:
        x = dot_product_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            make_image_summary=make_image_summary,
            dropout_broadcast_dims=dropout_broadcast_dims,
            activation_dtype=kwargs.get("activation_dtype"),
            hard_attention_k=hard_attention_k,
            gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_relative":
      x = dot_product_attention_relative(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          cache=cache is not None,
          allow_memory=recurrent_memory is not None,
          hard_attention_k=hard_attention_k,
          gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_unmasked_relative_v2":
      x = dot_product_unmasked_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "dot_product_relative_v2":
      x = dot_product_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "local_within_block_mask_right":
      x = masked_within_block_local_attention_1d(
          q, k, v, block_length=block_length)
    elif attention_type == "local_relative_mask_right":
      x = masked_relative_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary,
          dropout_rate=dropout_rate,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values,
          name="masked_relative_local_attention_1d")
    elif attention_type == "local_mask_right":
      x = masked_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary)
    elif attention_type == "local_unmasked":
      x = local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                           gap_size, num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = dilated_self_attention_1d(q, k, v, block_length, block_width,
                                    gap_size, num_memory_blocks)
    x = combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, name="output_transform",
          layer_collection=layer_collection)

    if recurrent_memory is not None:
      x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x

```

首先是检查参数，然后vars_3d_num_heads，这些都可以忽略。直接到下面的代码：
```
q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                      total_key_depth, total_value_depth, q_filter_width,
                      kv_filter_width, q_padding, kv_padding,
                      vars_3d_num_heads=vars_3d_num_heads,
                      layer_collection=layer_collection)
```
这是计算q、k和v的函数。

```
def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0,
                layer_collection=None):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    vars_3d_num_heads: an optional (if we want to use 3d variables)
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      q_padding,
      "q",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  k = compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  v = compute_attention_component(
      memory_antecedent,
      total_value_depth,
      kv_filter_width,
      kv_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  return q, k, v
```

它其实就是调用compute_attention_component函数分别计算q、k和v：
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

看起来很复杂，其实就是一个矩阵相乘：
```
return common_layers.dense(
    antecedent, total_depth, use_bias=False, name=name,
    layer_collection=layer_collection)

def dense(x, units, **kwargs):
  """Identical to layers.dense."""
  layer_collection = kwargs.pop("layer_collection", None)
  activations = layers().Dense(units, **kwargs)(x)
  if layer_collection:
    # We need to find the layer parameters using scope name for the layer, so
    # check that the layer is named. Otherwise parameters for different layers
    # may get mixed up.
    layer_name = tf.get_variable_scope().name
    if (not layer_name) or ("name" not in kwargs):
      raise ValueError(
          "Variable scope and layer name cannot be empty. Actual: "
          "variable_scope={}, layer name={}".format(
              layer_name, kwargs.get("name", None)))

    layer_name += "/" + kwargs["name"]
    layer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope=layer_name)
    assert layer_params
    if len(layer_params) == 1:
      layer_params = layer_params[0]

    tf.logging.info(
        "Registering dense layer to collection for tensor: {}".format(
            layer_params))

    x_shape = x.shape.as_list()
    if len(x_shape) == 3:
      # Handle [batch, time, depth] inputs by folding batch and time into
      # one dimension: reshaping inputs to [batchxtime, depth].
      x_2d = tf.reshape(x, [-1, x_shape[2]])
      activations_shape = activations.shape.as_list()
      activations_2d = tf.reshape(activations, [-1, activations_shape[2]])
      layer_collection.register_fully_connected_multi(
          layer_params, x_2d, activations_2d, num_uses=x_shape[1])
      activations = tf.reshape(activations_2d, activations_shape)
    else:
      layer_collection.register_fully_connected(layer_params, x, activations)
  return activations
```
在我们的例子里，输入x是(batch, timestep, 521)， units也是512。dense函数代码很多，但是我们的例子里真正执行的只有这么几行：
```
activations = layers().Dense(units, **kwargs)(x)
return activations
```

接着我们往上，回到multihead_attention。接着是
```
q = split_heads(q, num_heads)

@expert_utils.add_name_scope()
def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])
```
split_heads的作用是把输入的q从(batch, timestep, 512)变成(batch, num_heads=8, timestep, 512/8)。类似的k和v也会split，最终得到的q、k和v的shape都是(batch, 8, timestep, 64)，接着把q除以sqrt(d)：
```
q *= key_depth_per_head**-0.5
```

接着就是使用q、k和v计算self-attention，这里会有不同的计算方法，我们这里使用的是'dot_product'：
```
elif attention_type == "dot_product":
  if max_area_width > 1 or max_area_height > 1:
    x = area_attention.dot_product_area_attention(
        q, k, v, bias, dropout_rate, image_shapes,
        save_weights_to=save_weights_to,
        dropout_broadcast_dims=dropout_broadcast_dims,
        max_area_width=max_area_width,
        max_area_height=max_area_height,
        memory_height=memory_height,
        area_key_mode=area_key_mode,
        area_value_mode=area_value_mode,
        training=training)
else:
        x = dot_product_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            make_image_summary=make_image_summary,
            dropout_broadcast_dims=dropout_broadcast_dims,
            activation_dtype=kwargs.get("activation_dtype"),
            hard_attention_k=hard_attention_k,
            gumbel_noise_weight=gumbel_noise_weight)
```
我们走的是else分支，最终调用的是common_attention.py里的dot_product_attention函数：
```
def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None,
                          activation_dtype=None,
                          weight_dtype=None,
                          hard_attention_k=0,
                          gumbel_noise_weight=0.0):
  """Dot-product attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.
    activation_dtype: Used to define function activation dtype when using
      mixed precision.
    weight_dtype: The dtype weights are stored in when using mixed precision
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      logits += bias
    # If logits are fp16, upcast before softmax
    logits = maybe_upcast(logits, activation_dtype, weight_dtype)
    weights = tf.nn.softmax(logits, name="attention_weights")
    if hard_attention_k > 0:
      weights = harden_attention_weights(weights, hard_attention_k,
                                         gumbel_noise_weight)
    weights = common_layers.cast_like(weights, q)
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # Drop out attention links for each head.
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)
```
代码比较多，但是逻辑非常简单：首先通过q和k的矩阵成分得到logits，然后通过softmax得到weights，然后用weights加权(乘以)v得到最终的结果。我们简单的过一下。首先是
```
logits = tf.matmul(q, k, transpose_b=True)
```
q是(batch, head, time, hidden)，k的transpose是(batch, head, hidden, time)，相乘后是(batch, head, time, time)，也就是某个时刻t1 attend 另外一个时刻t2的score。这里还把bias也加到logits里。
接着用softmax得到weight：
```
weights = tf.nn.softmax(logits, name="attention_weights")
```
然后是对weight进行dropout：
```
weights = common_layers.dropout_with_broadcast_dims(
    weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
```
最后把weight乘以v返回：
```
return tf.matmul(weights, v)
```
得到的shape是v的shape(batch, head, time, hidden)。

接着把多个head的结果合并：
```
x = combine_heads(x)

@expert_utils.add_name_scope()
def combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))
```
最终得到是(batch, time, head*hidden)=(batch, time, 512)。最后再加一个全连接层：
```
x = common_layers.dense(
    x, output_depth, use_bias=False, name="output_transform",
    layer_collection=layer_collection)
```
shape保持不变。

然后我们回到transformer_encoder函数，继续执行：
```
y = common_attention.multihead_attention(...)
x = common_layers.layer_postprocess(x, y, hparams)

def layer_postprocess(layer_input, layer_output, hparams):
  """Apply layer postprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_postprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.

  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=hparams.layer_postprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_type=hparams.norm_type,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_postprocess")
```

最终调用的是layer_prepostprocess：
```
def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None,
                         layer_collection=None):
  """Apply a sequence of functions to the input or output of a layer.

  The sequence is specified as a string which may contain the following
  characters:
    a: add previous_value
    n: apply normalization
    d: apply dropout
    z: zero add

  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))

  Args:
    previous_value: A Tensor, to be added as a residual connection ('a')
    x: A Tensor to be transformed.
    sequence: a string.
    dropout_rate: a float
    norm_type: a string (see apply_norm())
    depth: an integer (size of last dimension of x).
    epsilon: a float (parameter for normalization)
    default_name: a string
    name: a string
    dropout_broadcast_dims:  an optional list of integers less than 3
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.

  Returns:
    a Tensor
  """
  with tf.variable_scope(name, default_name=default_name):
    if sequence == "none":
      return x
    for c in sequence:
      if c == "a":
        x += previous_value
      elif c == "z":
        x = zero_add(previous_value, x)
      elif c == "n":
        x = apply_norm(
            x, norm_type, depth, epsilon, layer_collection=layer_collection)
      else:
        assert c == "d", ("Unknown sequence step %s" % c)
        x = dropout_with_broadcast_dims(
            x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    return x
```
这个函数就是增加残差、layer_norm和dropout等。它是根据输入的sequence来处理的，比如输入是'dna'，则先执行dropout，然后layer_norm，最后是加入原始值(残差)：
```
For example, if sequence=="dna", then the output is
  previous_value + normalize(dropout(x))
```
这里输入是'da'，也就是dropout和add，layer_norm并没有在这里，而是在下面的ffn的preprocess里。

self-attention计算完成后就是一个ffn：
```
with tf.variable_scope("ffn"):
  y = transformer_ffn_layer(
      common_layers.layer_preprocess(x, hparams),
      hparams,
      pad_remover,
      conv_padding="SAME",
      nonpadding_mask=nonpadding,
      losses=losses)
  x = common_layers.layer_postprocess(x, y, hparams)
```

common_layers.layer_preprocess会调用apply_norm：
```
def apply_norm(x, norm_type, depth, epsilon, layer_collection=None):
  """Apply Normalization."""
  if layer_collection is not None:
    assert norm_type == "layer"
  if norm_type == "layer":
    return layer_norm(
        x, filters=depth, epsilon=epsilon, layer_collection=layer_collection)
  if norm_type == "group":
    return group_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "batch":
    return layers().BatchNormalization(epsilon=epsilon)(x)
  if norm_type == "noam":
    return noam_norm(x, epsilon)
  if norm_type == "l2":
    return l2_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "none":
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'lr', 'none'.")
```
然后就是调用transformer_ffn_layer函数：
```
def transformer_ffn_layer(x,
                          hparams,
                          pad_remover=None,
                          conv_padding="LEFT",
                          nonpadding_mask=None,
                          losses=None,
                          cache=None,
                          decode_loop_step=None,
                          readout_filter_size=0,
                          layer_collection=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparameters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
    conv_padding: a string - either "LEFT" or "SAME".
    nonpadding_mask: an optional Tensor with shape [batch_size, length].
      needed for convolutional layers with "SAME" padding.
      Contains 1.0 in positions corresponding to nonpadding.
    losses: optional list onto which to append extra training losses
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU.
    readout_filter_size: if it's greater than 0, then it will be used instead of
      filter_size
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.


  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]

  Raises:
    ValueError: If losses arg is None, but layer generates extra losses.
  """
  ffn_layer = hparams.ffn_layer
  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))
  if ffn_layer == "conv_hidden_relu":
    # Backwards compatibility
    ffn_layer = "dense_relu_dense"
  if ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims,
        layer_collection=layer_collection)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif ffn_layer == "conv_relu_conv":
    return common_layers.conv_relu_conv(
        x,
        readout_filter_size or hparams.filter_size,
        hparams.hidden_size,
        first_kernel_size=hparams.conv_first_kernel,
        second_kernel_size=1,
        padding=conv_padding,
        nonpadding_mask=nonpadding_mask,
        dropout=hparams.relu_dropout,
        cache=cache,
        decode_loop_step=decode_loop_step)
  elif ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, readout_filter_size or hparams.filter_size,
        hparams.num_heads,
        hparams.attention_dropout)
  elif ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        readout_filter_size or hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  elif ffn_layer == "sru":
    return common_layers.sru(x)
  elif ffn_layer == "local_moe_tpu":
    overhead = hparams.moe_overhead_eval
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      overhead = hparams.moe_overhead_train
    ret, loss = expert_utils.local_moe_tpu(
        x,
        hparams.filter_size // 2,
        hparams.hidden_size,
        hparams.moe_num_experts,
        overhead=overhead,
        loss_coef=hparams.moe_loss_coef)
  elif ffn_layer == "local_moe":
    overhead = hparams.moe_overhead_eval
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      overhead = hparams.moe_overhead_train
    ret, loss = expert_utils.local_moe(
        x,
        True,
        expert_utils.ffn_expert_fn(hparams.hidden_size, [hparams.filter_size],
                                   hparams.hidden_size),
        hparams.moe_num_experts,
        k=hparams.moe_k,
        hparams=hparams)
    losses.append(loss)
    return ret
  else:
    assert ffn_layer == "none"
    return x
```
我们这里的ffn_layer等于'dense_relu_dense'，所以走相应的if分支：
```
if pad_remover:
  original_shape = common_layers.shape_list(x)
  # Collapse `x` across examples, and remove padding positions.
  x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
  x = tf.expand_dims(pad_remover.remove(x), axis=0)
conv_output = common_layers.dense_relu_dense(
    x,
    hparams.filter_size,
    hparams.hidden_size,
    dropout=hparams.relu_dropout,
    dropout_broadcast_dims=relu_dropout_broadcast_dims,
    layer_collection=layer_collection)
if pad_remover:
  # Restore `conv_output` to the original shape of `x`, including padding.
  conv_output = tf.reshape(
      pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
return conv_output
```
首先是使用pad_remover去掉padding。然后是common_layers.dense_relu_dense，最后是pad_remover的restore后处理。
去掉padding的步骤为：首先把x从(batch, time, hidden) reshape成(batch*time, hidden)，然后调用pad_remover.remove(x)。
```
def remove(self, x):
  """Remove padding from the given tensor.

  Args:
    x (tf.Tensor): of shape [dim_origin,...]

  Returns:
    a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
  """
  with tf.name_scope("pad_reduce/remove"):
    x_shape = x.get_shape().as_list()
    x = tf.gather_nd(
        x,
        indices=self.nonpad_ids,
    )
    if not tf.executing_eagerly():
      # This is a hack but for some reason, gather_nd return a tensor of
      # undefined shape, so the shape is set up manually
      x.set_shape([None] + x_shape[1:])
  return x
```
这个函数的核心就是使用tf.gather_nd把输入x的非padding部分提取出来，会用到self.nonpad_ids，这个tensor是构造是传入，记录当前batch的非padding位置。如果记不清了，可以回去看看PadRemover类的构造函数。然后就是调用dense_relu_dense：
```
def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     layer_collection=None,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  layer_name = "%s_{}" % name if name else "{}"
  h = dense(
      inputs,
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      layer_collection=layer_collection,
      name=layer_name.format("conv1"))

  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      layer_collection=layer_collection,
      name=layer_name.format("conv2"))
  return o
```

也就是两个dense，第一个的激活函数是relu，第二个是None。它们之间有一个dropout。
这样的self-attention和ffn重复多次，上一层的输出作为下一层的输入，就得到了stacked的结果。

然后我们返回到body()的decoder部分，首先是_prepare_decoder_fn，是decoder前的准备工作：
```
if self.has_input:
  ...
  encoder_output, encoder_decoder_attention_bias = self.encode(...)
  ...

targets = features["targets"]
targets_shape = common_layers.shape_list(targets)
targets = common_layers.flatten4d3d(targets)
decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
    targets, hparams, features=features)
```

transformer_prepare_decoder的代码如下：
```
def transformer_prepare_decoder(targets, hparams, features=None, pad=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    pad: vector to use for padding when shifting targets right

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  if hparams.causal_decoder_self_attention:
    # Causal attention.
    if hparams.prepend_mode == "prepend_inputs_full_attention":
      decoder_self_attention_bias = (
          common_attention.attention_bias_prepend_inputs_full_attention(
              common_attention.embedding_to_padding(targets)))
    else:
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(targets)[1]))
  else:
    # Full attention.
    decoder_padding = common_attention.embedding_to_padding(targets)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(decoder_padding))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets, pad)
  if hparams.pos == "timing":
    if targets_position is not None:
      decoder_input = common_attention.add_timing_signal_1d_given_position(
          decoder_input, targets_position)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  elif hparams.pos == "timing_from_features":
    decoder_input = common_attention.add_timing_signals_from_features(
        decoder_input, features, hparams.position_features)
  elif hparams.pos == "emb":
    decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "targets_positional_embedding",
        targets_position)

  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)

```

这个函数首先构造decoder self-attention的mask：
```
decoder_self_attention_bias = (
    common_attention.attention_bias_lower_triangle(
        common_layers.shape_list(targets)[1]))

def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.

  Allows a query to attend to all positions up to and including its own.

  Args:
   length: a Scalar.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  return attention_bias_local(length, -1, 0)
```
attention_bias_lower_triangle会允许attend to所以当前位置之前以及当前位置的信息。它具体是使用attention_bias_local：
```
@expert_utils.add_name_scope()
def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.

  A position may attend to positions at most max_distance from it,
  forward and backwards.

  This does not actually save any computation.

  Args:
    length: int
    max_backward: int, maximum distance backward to attend. Negative values
      indicate unlimited.
    max_forward: int, maximum distance forward to attend. Negative values
      indicate unlimited.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = common_layers.ones_matrix_band_part(
      length,
      length,
      max_backward,
      max_forward,
      out_shape=[1, 1, length, length])
  return -1e9 * (1.0 - band)
```
最终是ones_matrix_band_part调用：
```
def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.

  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.

  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.linalg.band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band
```
最终调用但是tf.linalg.band_part函数。我们简单的解释一下这个函数。这个函数只处理一个tensor的最后两维，因此我们简单的假设输入是个矩阵：inputs[rows, cols]。则输出band[m, n] = in_band(m, n) * input[m, n]，这里的关键就是in_band(m,n)，它的定义是：
in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper < 0 || (n-m) <= num_upper)
看来有些吓人，我们举一个实际的例子。比如我们的decoder，传入的参数num_lower是-1，num_upper是0。则它得到的是一个下三角阵。为什么？
 in_band(m,n)=(-1<0||(m-n)<=-1) && (0<0 || (n-m)<=0)=(n-m)<=0。所以只要n<=m行，所以这是一个下三角阵。

接着是除了target的偏移，因为我们希望把target的第一个值作为第二个时刻的输入(第一个时刻的输入是encoder的eos)。
```
def shift_right_3d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets
```
具体是通过tf.pad来实现的。tf.pad的输入是一个(rank, 2)的向量，其中rank等于输入tensor的维度(rank)。我们举一个二维矩阵的例子：

```
t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2, 2]])
# 'constant_values' is 0.
# rank of 't' is 2.
tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                 #  [0, 0, 1, 2, 3, 0, 0],
                                 #  [0, 0, 4, 5, 6, 0, 0],
                                 #  [0, 0, 0, 0, 0, 0, 0]]

```
[1,1]表示在第一个维度(行)的前后各padding一行，[2,2]表示在第二个维度(列)的前后个插入一行。
我们这里的例子是tf.pad(x, [[0, 0], [1, 0], [0, 0]])，x的shape是(batch, time, hidden)，它只在time那个维度的左边padding一个0。然后[:, :-1, :]的意思是把time维度的最后一个去掉，因为target的最后一个时刻是不需要作为输入的。

接着加入position encoding，这个和encoder是一样的：
```
decoder_input = common_attention.add_timing_signal_1d(decoder_input)
```

到了这里，_prepare_decoder_fn就结束了，返回body并执行decoder：
```
decoder_output = self.decode(
    decoder_input,
    encoder_output,
    encoder_decoder_attention_bias,
    decoder_self_attention_bias,
    hparams,
    nonpadding=features_to_nonpadding(features, "targets"),
    losses=losses,
    **decode_kwargs
    )
```

它调用的是transformer_decode：
```
def transformer_decode(decoder_function,
                       decoder_input,
                       encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_self_attention_bias,
                       hparams,
                       attention_weights=None,
                       cache=None,
                       decode_loop_step=None,
                       nonpadding=None,
                       losses=None,
                       **kwargs):
  """Decode Transformer outputs from encoder representation.

  Args:
    decoder_function: the decoder function
    decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
      hidden_dim]
    encoder_output: Encoder representation. [batch_size, input_length,
      hidden_dim]
    encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
      attention. [batch_size, input_length]
    decoder_self_attention_bias: Bias and mask weights for decoder
      self-attention. [batch_size, decoder_length]
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    nonpadding: optional Tensor with shape [batch_size, decoder_length]
    losses: optional list onto which to append extra training losses
    **kwargs: additional arguments to pass to decoder_function

  Returns:
    Final decoder representation. [batch_size, decoder_length, hidden_dim]
  """
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)
  decoder_input = tf.nn.dropout(decoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  decoder_output = decoder_function(
      decoder_input,
      encoder_output,
      decoder_self_attention_bias,
      encoder_decoder_attention_bias,
      hparams,
      cache=cache,
      decode_loop_step=decode_loop_step,
      nonpadding=nonpadding,
      save_weights_to=attention_weights,
      losses=losses,
      **kwargs)

  if (common_layers.is_xla_compiled() and
      hparams.mode == tf.estimator.ModeKeys.TRAIN):
    # TPU does not react kindly to extra dimensions.
    # TODO(noam): remove this once TPU is more forgiving of extra dims.
    return decoder_output
  else:
    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)
```
它首先做一个dropout，然后就是调用decoder_function：
```
def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        layer_collection=None,
                        recurrent_memory_by_layer=None,
                        chunk_number=None):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.

  Returns:
    y: a Tensors
  """
  x = decoder_input

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      },
      hparams=hparams)

  with tf.variable_scope(name):
    for layer_idx in range(hparams.num_decoder_layers or
                           hparams.num_hidden_layers):
      x = transformer_decoder_layer(
          x,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          encoder_output=encoder_output,
          cache=cache,
          decode_loop_step=decode_loop_step,
          nonpadding=nonpadding,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          losses=losses,
          layer_collection=layer_collection,
          recurrent_memory_by_layer=recurrent_memory_by_layer,
          chunk_number=chunk_number
          )

    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(
        x, hparams, layer_collection=layer_collection)

```
这个函数和encoder很像，就是一个多层的transformer decoder：
```
def transformer_decoder_layer(decoder_input,
                              decoder_self_attention_bias,
                              layer_idx,
                              hparams,
                              encoder_output=None,
                              encoder_decoder_attention_bias=None,
                              cache=None,
                              decode_loop_step=None,
                              nonpadding=None,
                              save_weights_to=None,
                              make_image_summary=False,
                              losses=None,
                              layer_collection=None,
                              recurrent_memory_by_layer=None,
                              chunk_number=None):
  """A single transformer decoder layer."""
  x, layer_cache = transformer_self_attention_layer(
      decoder_input=decoder_input,
      decoder_self_attention_bias=decoder_self_attention_bias,
      layer_idx=layer_idx,
      hparams=hparams,
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      cache=cache,
      decode_loop_step=decode_loop_step,
      save_weights_to=save_weights_to,
      make_image_summary=make_image_summary,
      layer_collection=layer_collection,
      recurrent_memory_by_layer=recurrent_memory_by_layer,
      chunk_number=chunk_number)

  layer = layer_idx
  layer_name = "layer_%d" % layer
  with tf.variable_scope(layer_name):
    with tf.variable_scope("ffn"):
      y = transformer_ffn_layer(
          common_layers.layer_preprocess(
              x, hparams, layer_collection=layer_collection),
          hparams,
          conv_padding="LEFT",
          nonpadding_mask=nonpadding,
          losses=losses,
          cache=layer_cache,
          decode_loop_step=decode_loop_step,
          layer_collection=layer_collection)
      x = common_layers.layer_postprocess(x, y, hparams)
      return x

```

首先是transformer_self_attention_layer，然后是transformer_ffn_layer，我们分别来看：
```
def transformer_self_attention_layer(decoder_input,
                                     decoder_self_attention_bias,
                                     layer_idx,
                                     hparams,
                                     encoder_output=None,
                                     encoder_decoder_attention_bias=None,
                                     cache=None,
                                     decode_loop_step=None,
                                     save_weights_to=None,
                                     make_image_summary=False,
                                     layer_collection=None,
                                     recurrent_memory_by_layer=None,
                                     chunk_number=None):
  """A single transformer self-attention layer."""
  x = decoder_input
  layer = layer_idx
  layer_name = "layer_%d" % layer
  layer_cache = cache[layer_name] if cache is not None else None

  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  if recurrent_memory_by_layer is not None:
    recurrent_memory = recurrent_memory_by_layer[layer_name]
  else:
    recurrent_memory = None

  if layer < hparams.get("num_area_layers", 0):
    max_area_width = hparams.get("max_area_width", 1)
    max_area_height = hparams.get("max_area_height", 1)
    memory_height = hparams.get("max_area_height", 1)
  else:
    max_area_width = 1
    max_area_height = 1
    memory_height = 1
  with tf.variable_scope(layer_name):
    with tf.variable_scope("self_attention"):
      y = common_attention.multihead_attention(
          common_layers.layer_preprocess(
              x, hparams, layer_collection=layer_collection),
          None,
          decoder_self_attention_bias,
          hparams.attention_key_channels or hparams.hidden_size,
          hparams.attention_value_channels or hparams.hidden_size,
          hparams.hidden_size,
          hparams.num_heads,
          hparams.attention_dropout,
          attention_type=hparams.self_attention_type,
          max_relative_position=hparams.max_relative_position,
          heads_share_relative_embedding=(
              hparams.heads_share_relative_embedding),
          add_relative_to_values=hparams.add_relative_to_values,
          save_weights_to=save_weights_to,
          cache=layer_cache,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=attention_dropout_broadcast_dims,
          max_length=hparams.get("max_length"),
          decode_loop_step=decode_loop_step,
          vars_3d=hparams.get("attention_variables_3d"),
          activation_dtype=hparams.get("activation_dtype", "float32"),
          weight_dtype=hparams.get("weight_dtype", "float32"),
          layer_collection=layer_collection,
          recurrent_memory=recurrent_memory,
          chunk_number=chunk_number,
          hard_attention_k=hparams.get("hard_attention_k", 0),
          gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
          max_area_width=max_area_width,
          max_area_height=max_area_height,
          memory_height=memory_height,
          area_key_mode=hparams.get("area_key_mode", "none"),
          area_value_mode=hparams.get("area_value_mode", "none"),
          training=(hparams.get(
              "mode",
              tf.estimator.ModeKeys.TRAIN) == tf.estimator.ModeKeys.TRAIN))
      x = common_layers.layer_postprocess(x, y, hparams)
    if encoder_output is not None:
      if not isinstance(encoder_output, (list,)):
        encoder_output = [encoder_output]
      with tf.variable_scope("encdec_attention"):
        for enc_output in encoder_output:
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams, layer_collection=layer_collection),
              enc_output,
              encoder_decoder_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              layer_collection=layer_collection,
              hard_attention_k=hparams.get("hard_attention_k", 0),
              gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
              max_area_width=max_area_width,
              max_area_height=max_area_height,
              memory_height=memory_height,
              area_key_mode=hparams.get("area_key_mode", "none"),
              area_value_mode=hparams.get("area_value_mode", "none"),
              training=(hparams.get(
                  "mode",
                  tf.estimator.ModeKeys.TRAIN) == tf.estimator.ModeKeys.TRAIN))
          x = common_layers.layer_postprocess(x, y, hparams)
    return x, layer_cache
```
decoder和encoder类似，都是有一个self-attention，不同的是还有decoder到encoder的attention。attention的实现都是common_attention.multihead_attention，区别在于传入的参数：
```
with tf.variable_scope("self_attention"):
  y = common_attention.multihead_attention(
      common_layers.layer_preprocess(
          x, hparams, layer_collection=layer_collection),
      None,
      .....

  with tf.variable_scope("encdec_attention"):
        for enc_output in encoder_output:
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams, layer_collection=layer_collection),
              enc_output,
              ....
```
self-attention在调用multihead_attention的第二个参数memory_antecedent是None，而Decoder对encoder的attention传入的是enc_output，我们看一下decoder对encoder的调用的不同代码：
```
q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                      total_key_depth, total_value_depth, q_filter_width,
                      kv_filter_width, q_padding, kv_padding,
                      vars_3d_num_heads=vars_3d_num_heads,
                      layer_collection=layer_collection)

def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0,
                layer_collection=None):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    vars_3d_num_heads: an optional (if we want to use 3d variables)
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      q_padding,
      "q",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  k = compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  v = compute_attention_component(
      memory_antecedent,
      total_value_depth,
      kv_filter_width,
      kv_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  return q, k, v
```
唯一的区别是q、k和v的计算不同。在self-attention里，memory_antecedent是None，所以memory_antecedent被设置成query_antecedent；而在decoder-encoder的attention里，memory_antecedent是encoder最后一层的输出。计算q用的是query_antecedent，而kv的计算用的是memory_antecedent。

至此，body()函数就执行完毕。

接下来是top()函数，这是继承自基类的实现：
```
def top(self, body_output, features):
  """Computes logits given body output and features.

  Args:
    body_output: dict of str to Tensor, comprising one key-value pair for each
      target. Each value denotes the target's pre-logit activations.
      Alternatively, it may be a single Tensor denoting the pre-logits for
      that target.
    features: dict of str to Tensor. Typically it is the preprocessed data
      batch after Problem's preprocess_example().

  Returns:
    logits: dict of str to Tensor, denoting each logits for each target; or
      a single Tensor denoting the logits for that target.
      When targets are generated at training time:
        logits == {
          "self_generated_targets": <generated targets tensor>
          "logits": <original logits Tensor or dict>
        }
  """
  if isinstance(body_output, dict):
    logits = {}
    for k, v in six.iteritems(body_output):
      # TODO(aidangomez): share variables here?
      with tf.variable_scope(k) as top_vs:
        self._add_variable_scope("top_%s" % k, top_vs)
        logits[k] = self._top_single(v, k, features)
    return logits
  else:
    return self._top_single(body_output, "targets", features)
```
这个函数只是调用了_top_single：
```
def _top_single(self, body_output, feature_name, features):
  if not self._problem_hparams:
    log_warn("Without a Problem, T2TModel.top is a passthrough.")
    return body_output

  modality = self._problem_hparams.modality[feature_name]
  vocab_size = self._problem_hparams.vocab_size[feature_name]
  if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
    vocab_size += (-vocab_size) % self._hparams.vocab_divisor
  name = self._hparams.name.get(
      feature_name,
      modalities.get_name(modality))(self._hparams, vocab_size)
  with tf.variable_scope(name) as tm_vs:
    self._add_variable_scope(tm_vs.name, tm_vs)
    log_info("Transforming body output with %s.top", name)
    top = self._hparams.top.get(feature_name, modalities.get_top(modality))
    top_is_pointwise = getattr(top, "pointwise", False)
    last_only = (top_is_pointwise and
                 self.hparams.mode == tf.estimator.ModeKeys.PREDICT and
                 not self.hparams.force_full_predict)
    if not last_only:
      logits = top(body_output, features.get("targets"),
                   self._hparams, vocab_size)
    else:
      # Take body outputs for the last position only, and targets too.
      if "decode_loop_step" not in features:
        last_position_body_output = tf.expand_dims(
            body_output[:, -1, :, :], axis=[1])
        last_position_targets = tf.expand_dims(
            features["targets"][:, -1, :, :], axis=[1])
      else:
        body_output_shape = body_output.shape.as_list()
        last_position_body_output = tf.slice(
            body_output, [0, features["decode_loop_step"][0], 0, 0], [
                body_output_shape[0], 1, body_output_shape[2],
                body_output_shape[3]
            ])
        target_shape = features["targets"].shape.as_list()
        last_position_targets = tf.slice(
            features["targets"], [0, features["decode_loop_step"][0], 0, 0],
            [target_shape[0], 1, target_shape[2], target_shape[3]])
      logits = top(last_position_body_output, last_position_targets,
                   self._hparams, vocab_size)
  return logits
```
首先判断top的类型：
```
top = self._hparams.top.get(feature_name, modalities.get_top(modality))
def get_top(modality_type, value=None):
  """Gets default top transformation; if none available, return value."""
  if modality_type in (ModalityType.AUDIO,
                       ModalityType.AUDIO_SPECTRAL,
                       ModalityType.GENERIC_L2_LOSS,
                       ModalityType.IDENTITY,
                       ModalityType.IDENTITY_SYMBOL,
                       ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
                       ModalityType.SPEECH_RECOGNITION,
                       ModalityType.VIDEO_IDENTITY):
    return identity_top
  elif modality_type in (ModalityType.CLASS_LABEL,
                         ModalityType.MULTI_LABEL,
                         ModalityType.ONE_HOT_CLASS_LABEL,
                         ModalityType.SIGMOID_CLASS_LABEL):
    return class_label_top
  elif modality_type in (ModalityType.CTC_SYMBOL,
                         ModalityType.SYMBOL,
                         ModalityType.SYMBOL_WEIGHTS_ALL):
    return symbol_top
  elif modality_type == ModalityType.IMAGE:
    return image_top
  elif modality_type == ModalityType.IMAGE_CHANNEL_COMPRESS:
    return image_channel_compress_top
  elif modality_type == ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM:
    return image_channel_embeddings_top
  elif modality_type in (ModalityType.REAL,
                         ModalityType.REAL_L2_LOSS,
                         ModalityType.REAL_LOG_POISSON_LOSS):
    return real_top
  elif modality_type == ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL:
    return sigmoid_max_pooling_class_label_top
  elif modality_type == ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL:
    return softmax_average_pooling_class_label_top
  elif modality_type == ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL:
    return softmax_last_timestep_class_label_top
  elif modality_type == ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL:
    return softmax_max_pooling_class_label_top
  elif modality_type == ModalityType.SYMBOL_ONE_HOT:
    return symbol_one_hot_top
  elif modality_type in (ModalityType.VIDEO,
                         ModalityType.VIDEO_BITWISE,
                         ModalityType.VIDEO_PIXEL_NOISE):
    return video_top
  elif modality_type in (ModalityType.VIDEO_L1,
                         ModalityType.VIDEO_L2):
    return video_l1_top
  elif modality_type in (ModalityType.VIDEO_L1_RAW,
                         ModalityType.VIDEO_L2_RAW):
    return video_raw_top
  return value
```
我们这里返回的是symbol_top函数。不过在调用前还需要确定两个属性：
```
top_is_pointwise = getattr(top, "pointwise", False)
last_only = (top_is_pointwise and
             self.hparams.mode == tf.estimator.ModeKeys.PREDICT and
             not self.hparams.force_full_predict)
```
我们这里top_is_pointwise是True，last_only是False。top_is_pointwise的含义是不太清楚，但是会用于计算last_only；last_only表示计算loss是是否只需要最后一个。对于训练来说，我们需要每个时刻的输出。最终进入symbol_top函数：

```
def symbol_top(body_output, targets, model_hparams, vocab_size):
  """Generate logits.

  Args:
    body_output: A Tensor with shape
      [batch, p0, p1, model_hparams.hidden_size].
    targets: Unused.
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
  """
  del targets  # unused arg
  if model_hparams.shared_embedding_and_softmax_weights:
    scope_name = "shared"
    reuse = tf.AUTO_REUSE
  else:
    scope_name = "softmax"
    reuse = False
  with tf.variable_scope(scope_name, reuse=reuse):
    body_output_shape = common_layers.shape_list(body_output)
    var = get_weights(model_hparams, vocab_size, body_output_shape[-1])
    if (model_hparams.factored_logits and
        model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # insert channels dimension
      body_output = tf.expand_dims(body_output, 3)
      return common_layers.FactoredTensor(body_output, var)
    else:
      body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
      logits = tf.matmul(body_output, var, transpose_b=True)
      return tf.reshape(logits,
                        body_output_shape[:-1] + [1, vocab_size])
```
首先构造矩阵var：
```
var = get_weights(model_hparams, vocab_size, body_output_shape[-1])
```
var的shape是(vocab_size=8267, hidden_size=512)。
然后用输出(batch, time, hidden_size)乘以var的转置，最终变成(batch, time, vocab_size)的logits：
```
else:
  body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
  logits = tf.matmul(body_output, var, transpose_b=True)
  return tf.reshape(logits,
                    body_output_shape[:-1] + [1, vocab_size])
```
注：实际从top返回的logits是(batch, time, 1, 1, vocab_size)。
至此top函数完成。接下来回到model_fn计算loss：
```
logits = self.top(output, features)
losses["training"] = 0.0
if (self._hparams.mode != tf.estimator.ModeKeys.PREDICT and
    self._hparams.mode != "attack"):
  losses["training"] = self.loss(logits, features)
```
进入基类的loss方法：
```
def loss(self, logits, features):
  if isinstance(logits, dict):
    losses = {}
    for k, v in six.iteritems(logits):
      losses[k] = self._loss_single(
          v,
          k,
          features[k],
          weights=features.get(k + "_mask"))

      n, d = losses[k]
      if common_layers.should_generate_summaries():
        tf.summary.scalar(k + "_loss", n / d)
        tf.summary.scalar(k + "_loss_num", n)
        tf.summary.scalar(k + "_loss_den", d)
        if getattr(self.hparams, "visualize_logits_histogram", False):
          hist = tf.summary.histogram
          hist(k + "_predict", tf.argmax(tf.squeeze(v), axis=-1))
          hist(k + "_targets", features[k])

    return tf.add_n([n / d for n, d in losses.values()])
  else:
    return self._loss_single(
        logits,
        "targets",
        features["targets"],
        weights=features.get("targets_mask"))
```
这里走的是else分支，也就是调用_loss_single：
```
def _loss_single(self, logits, feature_name, feature, weights=None):
  # The current bfloat16 version still uses float32 for most parts of backward
  # propagation to keep model quality, so cast back before computing the loss
  # value.
  if not self._problem_hparams:
    log_warn(_no_problem_err("loss"))
    return (tf.constant(0., dtype=tf.float32),
            tf.constant(1., dtype=tf.float32))

  # Calculate loss contribution.
  modality = self._problem_hparams.modality[feature_name]
  vocab_size = self._problem_hparams.vocab_size[feature_name]
  if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
    vocab_size += (-vocab_size) % self._hparams.vocab_divisor
  loss = self._hparams.loss.get(feature_name, modalities.get_loss(modality))
  targets_weights_fn = self._hparams.weights_fn.get(
      "targets", modalities.get_weights_fn(modality))
  if weights is None:
    loss_num, loss_den = loss(logits, feature, self._hparams, vocab_size,
                              weights_fn=targets_weights_fn)
  else:

    def weights_fn(labels):
      """Per-token weights for loss."""
      # Use target_weights_fn() given by modality as well as explicitly given
      # weights.
      modality_weights = targets_weights_fn(labels)

      # Broadcast 'weights' along minor dimensions (TF's default is major).
      explicit_weights = weights
      if len(explicit_weights.shape) < len(modality_weights.shape):
        explicit_weights = common_layers.expand_squeeze_to_nd(
            weights, modality_weights.shape.ndims)

      return explicit_weights * modality_weights

    # Ensure that target.modality_loss() supports "weights_fn" keyword
    # argument. If it doesn't and "weights" is specified, raise an exception.
    argument_names = inspect.getargspec(loss).args
    if "weights_fn" not in argument_names:
      raise ValueError(
          "Explicit 'weights' given but default loss for modality doesn't "
          "support 'weights_fn' keyword argument: %s.loss(%s)." %
          (modality, ", ".join(argument_names)))

    loss_num, loss_den = loss(
        logits, feature, self._hparams, vocab_size, weights_fn=weights_fn)

  loss_num *= self._problem_hparams.loss_multiplier

  if hasattr(self.hparams, "problem") and hasattr(
      self.hparams.problem, "task_list"):
    if weights is not None:
      raise NotImplementedError("weights not yet implemented in "
                                "multitask setting.")
    loss_num, loss_den, summaries = multi_problem.aggregate_task_losses(
        self.hparams,
        self._problem_hparams,
        logits,
        feature_name,
        feature
    )

    for key, val in summaries:
      tf.summary.scalar(key, val)

  return loss_num, loss_den
```

和前面类似，也是通过modality_type找到对于的loss函数：
```
def get_loss(modality_type, value=None):
  """Gets default loss transformation; if none available, return value."""
  if modality_type in (ModalityType.AUDIO,
                       ModalityType.AUDIO_SPECTRAL,
                       ModalityType.CLASS_LABEL,
                       ModalityType.IDENTITY,
                       ModalityType.IDENTITY_SYMBOL,
                       ModalityType.IMAGE,
                       ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
                       ModalityType.IMAGE_CHANNEL_COMPRESS,
                       ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM,
                       ModalityType.REAL,
                       ModalityType.SPEECH_RECOGNITION,
                       ModalityType.SYMBOL,
                       ModalityType.SYMBOL_WEIGHTS_ALL):
    return generic_loss
  elif modality_type == ModalityType.CTC_SYMBOL:
    return ctc_symbol_loss
  elif modality_type == ModalityType.GENERIC_L2_LOSS:
    return generic_l2_loss
  elif modality_type == ModalityType.MULTI_LABEL:
    return multi_label_loss
  elif modality_type in (ModalityType.ONE_HOT_CLASS_LABEL,
                         ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL,
                         ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL):
    return one_hot_class_label_loss
  elif modality_type == ModalityType.REAL_L2_LOSS:
    return real_l2_loss
  elif modality_type == ModalityType.REAL_LOG_POISSON_LOSS:
    return real_log_poisson_loss
  elif modality_type == ModalityType.SIGMOID_CLASS_LABEL:
    return sigmoid_class_label_loss
  elif modality_type == ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL:
    return sigmoid_max_pooling_class_label_loss
  elif modality_type == ModalityType.SYMBOL_ONE_HOT:
    return symbol_one_hot_loss
  elif modality_type in (ModalityType.VIDEO,
                         ModalityType.VIDEO_BITWISE,
                         ModalityType.VIDEO_PIXEL_NOISE):
    return video_loss
  elif modality_type == ModalityType.VIDEO_IDENTITY:
    return video_identity_loss
  elif modality_type == ModalityType.VIDEO_L1:
    return video_l1_loss
  elif modality_type == ModalityType.VIDEO_L1_RAW:
    return video_l1_raw_loss
  elif modality_type == ModalityType.VIDEO_L2:
    return video_l2_loss
  elif modality_type == ModalityType.VIDEO_L2_RAW:
    return video_l2_raw_loss
  return value
```
这里返回的是generic_loss函数，不过在调用之前还需要weights_fn：
```
def get_weights_fn(modality_type, value=None):
  """Gets default weights function; if none available, return value."""
  if modality_type in (ModalityType.CTC_SYMBOL,
                       ModalityType.IDENTITY_SYMBOL,
                       ModalityType.MULTI_LABEL,
                       ModalityType.SYMBOL,
                       ModalityType.SYMBOL_ONE_HOT):
    return common_layers.weights_nonzero
  elif modality_type in ModalityType.get_choices():
    return common_layers.weights_all
  return value
```
对于modality_type等于'symbol'的输入来说，返回的是weights_nonzero，也就是非padding的位置loss的权重是1，padding的为0。
之后就进入generic_loss：
```
def generic_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = common_attention.maybe_upcast(logits, hparams=model_hparams)
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.0)
  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=cutoff,
      weights_fn=weights_fn)
```
最终调用的是common_layers.padded_cross_entropy：
```
def padded_cross_entropy(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True,
                         cutoff=0.0,
                         gaussian=False):
  """Compute cross-entropy assuming 0s are padding.

  Computes a loss numerator (the sum of losses), and loss denominator
  (the number of non-padding tokens).

  Args:
    logits: a `Tensor` with shape `[batch, timesteps, vocab_size]`.
      optionally a FactoredTensor.
    labels: an integer `Tensor` with shape `[batch, timesteps]`.
    label_smoothing: a floating point `Scalar`.
    weights_fn: A function from labels to weights.
    reduce_sum: a Boolean, whether to sum at the end or not.
    cutoff: a float, at which point to have no loss.
    gaussian: If true, use a Gaussian distribution for label smoothing

  Returns:
    loss_numerator: a `Scalar`.  Sum of losses.
    loss_denominator: a `Scalar.  The number of non-padding target tokens.

  Raises:
    ValueError: in case of unsupported argument types.
  """
  if isinstance(logits, FactoredTensor):
    if gaussian:
      raise ValueError("Factored padded cross entropy with Gaussian smoothing "
                       "is not implemented yet.")
    return padded_cross_entropy_factored(
        logits,
        labels,
        label_smoothing,
        weights_fn=weights_fn,
        reduce_sum=reduce_sum)
  confidence = 1.0 - label_smoothing
  logits_shape = shape_list(logits)
  vocab_size = logits_shape[-1]
  with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
    if len(logits_shape) == 2:
      # Deal with the case where we did not insert extra dimensions due to
      # TPU issues.  No pad-to-same-length happens in this case.
      # TODO(noam): remove this logic once TPU can handle extra dimensions.
      labels = tf.reshape(labels, [-1])
    else:
      logits, labels = pad_with_zeros(logits, labels)
    logits = tf.reshape(
        logits,
        shape_list(labels) + [vocab_size],
        name="padded_cross_entropy_size_check")
    logits = tf.cast(logits, tf.float32)
    xent = smoothing_cross_entropy(
        logits, labels, vocab_size, confidence, gaussian=gaussian)
    weights = weights_fn(labels)
    if cutoff > 0.0:
      xent = tf.nn.relu(xent - cutoff)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)
```
首先通过label_smoothing计算confidence，这里输入的label_smoothing是0.1。
因为labels不包含padding，所以先pad_with_zeros。细节就不赘述了，代码为：
```
logits, labels = pad_with_zeros(logits, labels)

def pad_with_zeros(logits, labels):
  """Pad labels on the length dimension to match logits length."""
  with tf.name_scope("pad_with_zeros", values=[logits, labels]):
    logits, labels = pad_to_same_length(logits, labels)
    if len(labels.shape) == 3:  # 2-d labels.
      logits, labels = pad_to_same_length(logits, labels, axis=2)
    return logits, labels

def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
  """Pad tensors x and y on axis 1 so that they have the same length."""
  if axis not in [1, 2]:
    raise ValueError("Only axis=1 and axis=2 supported for now.")
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = shape_list(x)[axis]
    y_length = shape_list(y)[axis]
    if (isinstance(x_length, int) and isinstance(y_length, int) and
        x_length == y_length and final_length_divisible_by == 1):
      return x, y
    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by
    length_diff1 = max_length - x_length
    length_diff2 = max_length - y_length

    def padding_list(length_diff, arg):
      if axis == 1:
        return [[[0, 0], [0, length_diff]],
                tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
      return [[[0, 0], [0, 0], [0, length_diff]],
              tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

    paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
    paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
    res_x = tf.pad(x, paddings1)
    res_y = tf.pad(y, paddings2)
    # Static shapes are the same except for axis=1.
    x_shape = x.shape.as_list()
    x_shape[axis] = None
    res_x.set_shape(x_shape)
    y_shape = y.shape.as_list()
    y_shape[axis] = None
    res_y.set_shape(y_shape)
    return res_x, res_y
```

然后回到padded_cross_entropy，调用xent = smoothing_cross_entropy计算交叉熵：
```
xent = smoothing_cross_entropy(
    logits, labels, vocab_size, confidence, gaussian=gaussian)
```
smoothing_cross_entropy代码为：
```
def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False):
  """Cross entropy with label smoothing to limit over-confidence.

  Args:
    logits: Tensor of shape [batch_size, ?, ?, ?, vocab_size].
    labels: Tensor of shape [batch_size, ?, ?, ?].
    vocab_size: Tensor representing the size of the vocabulary.
    confidence: Used to determine on and off values for label smoothing.
      If `gaussian` is true, `confidence` is the variance to the Gaussian
      distribution.
    gaussian: Uses a Gaussian distribution for label smoothing

  Returns:
    Tensor of shape [batch_size, ?, ?, ?].
  """
  with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
        confidence * tf.log(confidence) + to_float(vocab_size - 1) *
        low_confidence * tf.log(low_confidence + 1e-20))

    if gaussian and confidence > 0.0:
      labels = tf.cast(labels, tf.float32)

      normal_dist = tfp.distributions.Normal(loc=labels, scale=confidence)
      # Locations to evaluate the probability distributions.
      soft_targets = normal_dist.prob(
          tf.cast(tf.range(vocab_size), tf.float32)[:, None, None, None, None])
      # Reordering soft_targets from [vocab_size, batch_size, ?, ?, ?] to match
      # logits: [batch_size, ?, ?, ?, vocab_size]
      soft_targets = tf.transpose(soft_targets, perm=[1, 2, 3, 4, 0])
    else:
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing
```
这个函数和普通的交叉熵的区别是并不认为“真实”的分布是one-hot的，而是把很小的概率(0.1)平均分配给所有的非目标的词(共vocab_size-1个)，剩下的概率才给目标词。另外有一个小的不同就是减去掉"最优"的预测分布(也就是对真实词预测概率0.9，其余的词的概率为0.1/(vocab_size-1))对应的熵，这不影响梯度的计算。最终调用的是tf.nn.softmax_cross_entropy_with_logits_v2。

然后返回padded_cross_entropy，对这个loss应用weights_fn，也就是把padding的部分去掉。
```
weights = weights_fn(labels)
...
return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)
```
weights_fn代码很简单：
```
def weights_nonzero(labels):
  """Assign weight 1.0 to all labels except for padding (id=0)."""
  return to_float(tf.not_equal(labels, 0))
```

执行后返回到model_fn_sharded：
```
else:
  sharded_logits, sharded_losses = dp(self.model_fn, datashard_to_features)
  # 返回到这里！
  sharded_logits, sharded_losses = dp(
      self.maybe_scheduled_sampling,
      datashard_to_features, sharded_logits, sharded_losses)
  if isinstance(sharded_logits[0], dict):
    temp_dict = {k: [] for k, _ in six.iteritems(sharded_logits[0])}
    for k, _ in six.iteritems(sharded_logits[0]):
      for l in sharded_logits:
        temp_dict[k].append(l[k])
    sharded_logits = temp_dict
  losses = average_sharded_losses(sharded_losses)
```
接下来一个类似的dp，作用是随机采样。maybe_scheduled_sampling代码如下：
```
def maybe_scheduled_sampling(self, features, logits, losses):
  """Scheduled sampling.

  Performs forward inference again with "targets" feature replaced with values
  sampled from the model.

  This is the identity unless self.hparams.scheduled_sampling_prob > 0
  (default).

  **WARNING**: If hparams.scheduled_sampling_method == "parallel", this is
  not a faithful implementation of scheduled sampling. This implementation
  samples tokens for timestep t condtioned on gold tokens 1...t-1. A proper
  implementation must condition on a mix of gold and sampled tokens. Doing
  so is not efficient for models such like Transformer.

  Args:
    features: {str: Tensor}. Features sharded along batch dimension.
    logits: Tensor. Logits for each shard of data.
    losses: 0-D Tensor or (num: 0-D Tensor, denom: 0-D Tensor). Loss Tensor

  Returns:
    new_logits: Tensor.
    new_losses: {str: loss} where loss is one of (i) a 0-D Tensor or
      (ii) a (num: 0-D Tensor, denom: 0-D Tensor) pair to be used in a
      weighted average.
  """
  hparams = self.hparams
  problem_hparams = self._problem_hparams

  # Only do scheduled sampling if requested.
  if hparams.scheduled_sampling_prob == 0.0:
    return (logits, losses)

  # Only do scheduled sampling on language tasks.
  modality = problem_hparams.modality["targets"]
  if modality not in [
      modalities.ModalityType.SYMBOL,
      modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
      modalities.ModalityType.IMAGE
  ]:
    assert hparams.scheduled_sampling_prob == 0, (
        "Scheduled sampling only applies to ModalityType.(SYMBOL, "
        "SYMBOL_WEIGHTS_ALL, IMAGE). Found {modality}. Set "
        "hparams.scheduled_sampling_prob == 0.0.").format(modality=modality)
    return (logits, losses)

  # Only do scheduled sampling when training.
  is_training = (hparams.mode == tf.estimator.ModeKeys.TRAIN)
  if not is_training:
    tf.logging.info("Running in %s mode. Not using scheduled sampling.",
                    hparams.mode)
    return (logits, losses)

  # Pad vocabulary if vocab size must be evenly divisible by vocab_divisor.
  vocab_size = problem_hparams.vocab_size["targets"]
  assert vocab_size is not None
  assert hparams.vocab_divisor == 1

  # TODO(duckworthd): Move to scheduled_sampling.py.
  def sample(x):
    """Multinomial sampling from a n-dimensional tensor."""
    samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]), 1)
    reshaped_samples = tf.reshape(samples, common_layers.shape_list(x)[:-1])
    return tf.to_int32(reshaped_samples)

  # TODO(duckworthd): Move to scheduled_sampling.py.
  def mix_gold_sampled(gold_targets,
                       sampled_targets,
                       mixin_prob,
                       i,
                       prev_new_targets):
    """Interleave sampled and gold tokens randomly."""
    # Resample each location iid.
    should_use_sampled_targets = tf.less(
        tf.random_uniform(common_layers.shape_list(sampled_targets)),
        mixin_prob)
    mixed_targets = tf.where(
        should_use_sampled_targets,
        sampled_targets,
        gold_targets)

    # Reuse sample tokens for earlier timesteps.
    new_targets = tf.where(
        is_later_timestep(gold_targets, i),
        mixed_targets,
        prev_new_targets)
    return new_targets

  # TODO(duckworthd): Move to scheduled_sampling.py.
  def is_later_timestep(x, pass_idx):
    """Constructs mask based on timestep."""
    assert x.shape.ndims == 4, x.shape
    x_shape = tf.shape(x)
    num_timesteps = x_shape[1]
    timesteps = tf.range(num_timesteps)
    timesteps = tf.reshape(timesteps, [1, num_timesteps, 1, 1])
    # The following is a bit untrue. For images, "num_timesteps" actually
    # represents image height, not time. We ignore that fact here.
    timesteps = tf.broadcast_to(timesteps, x_shape)
    return tf.greater_equal(timesteps, pass_idx)

  # TODO(duckworthd): Move to scheduled_sampling.py.
  def parallel_scheduled_sampling_pass(
      i, prev_new_targets, features, logits, mixin_prob):
    """Generate scheduled sampling results."""
    sampled_targets = sample(logits)
    new_targets = mix_gold_sampled(features["targets"],
                                   sampled_targets,
                                   mixin_prob,
                                   i,
                                   prev_new_targets)
    new_targets = tf.stop_gradient(new_targets)  # Treat new_targets as given.
    new_features = copy.copy(features)
    new_features["targets"] = new_targets
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      # Compute bottom() for new_targets.
      #
      # TODO(duckworthd): Only apply bottom to 'new_targets'.
      new_transformed_features = self.bottom(new_features)

      # Compute body.
      with tf.variable_scope("body"):
        new_body_outputs, new_losses = self._normalize_body_output(
            self.body(new_transformed_features))
      assert "training" not in new_losses

      # Compute top.
      new_logits = self.top(new_body_outputs, new_features)

      # Compute loss. Use original features (== labels).
      if (hparams.mode != tf.estimator.ModeKeys.PREDICT and
          hparams.mode != "attack"):
        new_losses["training"] = self.loss(new_logits, features)
      else:
        new_losses["training"] = 0.0

    return new_targets, new_logits, new_losses

  tf.logging.info("Using scheduled sampling.")
  tf.logging.info("Warming scheduled sampling up with schedule: %s",
                  hparams.scheduled_sampling_warmup_schedule)
  assert hparams.scheduled_sampling_prob == 1.0, (
      "hparams.scheduled_sampling_prob must be 0 or 1.")

  if hparams.scheduled_sampling_method == "sequential":
    tf.logging.info("Using SEQUENTIAL scheduled sampling.")
    assert hparams.scheduled_sampling_num_passes == 1, (
        "hparams.scheduled_sampling_num_passes must equal 1 if "
        "doing sequential scheduled sampling.")
    return scheduled_sampling.sequential_scheduled_sampling_for_t2tmodel(
        self, features)
  elif hparams.scheduled_sampling_method == "parallel":
    tf.logging.info("Using PARALLEL scheduled sampling.")
    # TODO(duckworthd): Move this block to scheduled_sampling.py.

    # Gradually increase over a warmup period. Lower numbers mean more gold
    # tokens.
    mixin_prob = scheduled_sampling.inverse_decay_mix_prob(
        hparams.scheduled_sampling_warmup_schedule,
        hparams.scheduled_sampling_gold_mixin_prob,
        hparams.scheduled_sampling_warmup_steps)

    # Apply scheduled sampling over N passes. The logits from the (n-1)-th
    # pass will be mixed with gold tokens for conditioning in the n-th pass.
    assert hparams.scheduled_sampling_num_passes > 0, (
        "hparams.scheduled_sampling_num_passes must be > 0 if "
        "hparams.scheduled_sampling_prob > 0.0")
    new_logits = logits
    new_losses = losses
    prev_new_targets = features["targets"]
    for i in range(hparams.scheduled_sampling_num_passes):
      prev_new_targets, new_logits, new_losses = parallel_scheduled_sampling_pass(
          i, prev_new_targets, features, new_logits, mixin_prob)
    return new_logits, new_losses
  else:
    raise ValueError(
        "Unknown scheduled_sampling_method = %s" % (
            hparams.scheduled_sampling_method,))
```
我们的代码没有随机采样，所以就不分析其代码了。
```
# Only do scheduled sampling if requested.
if hparams.scheduled_sampling_prob == 0.0:
  return (logits, losses)
```

至此model的call完成，返回estimator_model_fn，后面的代码就是生成不同的EstimatorSpec，这里是训练的Spec。
```
if config and not use_tpu:
  num_async_replicas = config.t2t_device_info["num_async_replicas"]
return model.estimator_spec_train(
    loss, num_async_replicas=num_async_replicas, use_tpu=use_tpu)
```
具体为：
```
def estimator_spec_train(self, loss, num_async_replicas=1, use_tpu=False):
  """Constructs `tf.estimator.EstimatorSpec` for TRAIN (training) mode."""
  train_op = self.optimize(loss, num_async_replicas=num_async_replicas,
                           use_tpu=use_tpu)

  if use_tpu:
    if self._hparams.warm_start_from:
      def scaffold_fn():
        self.initialize_from_ckpt(self._hparams.warm_start_from)
        return tf.train.Scaffold()
    else:
      scaffold_fn = None

    # Note: important to call this before remove_summaries()
    if self.hparams.tpu_enable_host_call:
      host_call = self.create_train_host_call()
    else:
      host_call = None

    remove_summaries()

    return contrib.tpu().TPUEstimatorSpec(
        tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op,
        host_call=host_call,
        scaffold_fn=scaffold_fn)
  else:
    if self._hparams.warm_start_from:
      self.initialize_from_ckpt(self._hparams.warm_start_from)

    # When loading weights from a pre-trained model, you want to be able to
    # load separate weights into the encoder and decoder.
    if self._hparams.warm_start_from_second:
      self.initialize_from_ckpt(self._hparams.warm_start_from_second)

    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)
```
核心就这么几行：
```
train_op = self.optimize(loss, num_async_replicas=num_async_replicas,
                         use_tpu=use_tpu)
...
return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)
```

至此训练的代码分析完毕！
