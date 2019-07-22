---
layout:     post
title:      "XLNet代码分析(二)" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - 深度学习
    - XLNet
---

本文介绍XLNet的代码的第二部分，需要首先阅读[第一部分](/2019/06/30/xlnet-codes)，读者阅读前需要了解XLNet的原理，不熟悉的读者请先阅读[XLNet原理](/2019/06/30/xlnet-theory/)。

<!--more-->

**目录**
* TOC
{:toc}


书接上文，前面我们提到：如果忽略多设备(GPU)训练的细节，train的代码结构其实并不复杂，它大致可以分为3部分：

* 调用data_utils.get_input_fn得到train_input_fn

* 调用single_core_graph构造XLNet网络

* 使用session运行fetches进行训练

## train

首先我们来看一下train函数的主要代码，为了简单，我们这里只分析单个GPU的代码路径而忽略多GPU的情况：

```
def train(ps_device):
  ##### 得到input function和model function

  train_input_fn, record_info_dict = data_utils.get_input_fn(
      tfrecord_dir=FLAGS.record_info_dir,
      split="train",
      bsz_per_host=FLAGS.train_batch_size,
      seq_len=FLAGS.seq_len,
      reuse_len=FLAGS.reuse_len,
      bi_data=FLAGS.bi_data,
      num_hosts=1,
      num_core_per_host=1, # 不管多少GPU都设置为1
      perm_size=FLAGS.perm_size,
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      uncased=FLAGS.uncased,
      num_passes=FLAGS.num_passes,
      use_bfloat16=FLAGS.use_bfloat16,
      num_predict=FLAGS.num_predict)
 

  ##### 创建输入tensors(placeholder)
  # 训练的batch_size平均分配到多个设备(GPU)上
  bsz_per_core = FLAGS.train_batch_size // FLAGS.num_core_per_host

  params = {
      "batch_size": FLAGS.train_batch_size # the whole batch
  }
  # 调用前面的input函数得到Dataset
  train_set = train_input_fn(params)
  # 得到一个example
  example = train_set.make_one_shot_iterator().get_next()

  if FLAGS.num_core_per_host > 1:
    examples = [{} for _ in range(FLAGS.num_core_per_host)]
    for key in example.keys():
      vals = tf.split(example[key], FLAGS.num_core_per_host, 0)
      for device_id in range(FLAGS.num_core_per_host):
        examples[device_id][key] = vals[device_id]
  else:
    examples = [example]

  ##### 创建计算图的代码
  tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

  for i in range(FLAGS.num_core_per_host):
    reuse = True if i > 0 else None
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

      
      mems_i = {}
      if FLAGS.mem_len:
        # 创建mems，这是Transformer-XL里的思想，保留之前的96个Token的上下文
        mems_i["mems"] = create_mems_tf(bsz_per_core)
      # 这是创建计算图的核心代码，我们后面会详细分析
      # loss_i是loss；new_mems_i是个list，表示每一层的新的96个Token的隐状态
      # grads_and_vars_i是梯度和变量，因为需要多个GPU数据并行，
      # 所以需要自己来平均所有GPU的梯度然后用它更新参数(而不能用Optimizer)
      loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
          is_training=True,
          features=examples[i],
          mems=mems_i)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)
      tower_grads_and_vars.append(grads_and_vars_i)

  ## 如果有多个GPU，那么需要求出平均的梯度，我们这里不分析 
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
    grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
  else:
    loss = tower_losses[0]
    grads_and_vars = tower_grads_and_vars[0]

  ## 根据grads_and_vars得到训练的Opertaion，此外还会返回learning_rate和gnorm
  train_op, learning_rate, gnorm = model_utils.get_train_op(FLAGS, None,
      grads_and_vars=grads_and_vars)
  global_step = tf.train.get_global_step()

  ##### 训练的循环
  # 首先初始化mems
  tower_mems_np = []
  for i in range(FLAGS.num_core_per_host):
    mems_i_np = {}
    for key in tower_mems[i].keys():
      mems_i_np[key] = initialize_mems_np(bsz_per_core)
    tower_mems_np.append(mems_i_np)
  # 保存checkpoint的Saver
  saver = tf.train.Saver()
  # GPUOptions，允许它动态的使用更多GPU内存
  gpu_options = tf.GPUOptions(allow_growth=True)
  # 从之前的checkpoint恢复参数
  model_utils.init_from_checkpoint(FLAGS, global_vars=True)
  
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
      gpu_options=gpu_options)) as sess:
    # 允许初始化所有变量的操作
    sess.run(tf.global_variables_initializer())
    # 需要运行的ops：包括loss，新的mem，global_step，gnorm，learning_rate和train_op
    fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]

    total_loss, prev_step = 0., -1
    while True:
      feed_dict = {}
      # 用上一个数据的96个mem作为当前的mem
      for i in range(FLAGS.num_core_per_host):
        for key in tower_mems_np[i].keys():
          for m, m_np in zip(tower_mems[i][key], tower_mems_np[i][key]):
            feed_dict[m] = m_np
      # 进行一次随机梯度下降
      fetched = sess.run(fetches, feed_dict=feed_dict)
      # 拿到loss；新的mems和当前的steps
      loss_np, tower_mems_np, curr_step = fetched[:3]
      total_loss += loss_np

      if curr_step > 0 and curr_step % FLAGS.iterations == 0:
        curr_loss = total_loss / (curr_step - prev_step)
        tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
            "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
            curr_step, fetched[-3], fetched[-2],
            curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
        total_loss, prev_step = 0., curr_step

      if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
        save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        saver.save(sess, save_path)
        tf.logging.info("Model saved in path: {}".format(save_path))

      if curr_step >= FLAGS.train_steps:
        break
```

我们在计算当前句子时会用到之前的96个Token作为上下文，因此需要创建mems：
```
def create_mems_tf(bsz_per_core):
  mems = [tf.placeholder(dtype=tf.float32,
                         shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model])
          for layer in range(FLAGS.n_layer)]

  return mems
```

在我的例子里它是创建了长度为6的list(因为我测试用的XLNet只有6层)，每一个元素的shape是[96, 8, 1024]，96是mem的长度，8是batch_size，1024是隐单元个数。

上面的train函数的训练过程非常的简单，接下面我们详细的来介绍train函数处理Dataset输入以及构建计算图的代码。

## data_utils.get_input_fn



### get_input_fn函数主要代码
这个函数的返回值是Estimator的一个函数和一个record_info对象，这个record_info对象记录读取的TFRecord的num_batch和file_names等数据；而第一个函数执行后的返回值是一个tf.data.Dataset，这是Tensorflow标准的输入方式，不了解的读者可以参考[Datasets for Estimators](https://www.tensorflow.org/guide/datasets_for_estimators)或者[深度学习理论与实战：基础篇](https://fancyerii.github.io/2019/07/05/book/)里的相关内容。

返回值知道了，我们再来看这个函数的参数：

* tfrecord_dir 训练数据目录，如果有多个用逗号","分开，这里是'traindata/tfrecords'，参考第一部分。
* split 这里是"train"表示训练数据。
* bsz_per_host 每个host(机器)的batch大小，这里是8。
* seq_len 句子长度，这里是128。
* reuse_len 句子中前面不变的部分，这里是64，请参考第一部分。
* bi_data 是否双向的模型，这里是True。
* num_hosts=1 有多少台机器，这里是1。
* num_core_per_host=1 每天机器有多少个设备(GPU)，这里是1。
* perm_size=None, 排列打散的长度，这里是32，后面会讲到。
* mask_alpha=None, 这里是6
* mask_beta=None, 这里是1
* uncased=False, 是否不区分大小，这里是True，表示不区分大小写
* num_passes=None, 训练的趟数，这里是1表示这是第一趟训练
* use_bfloat16=False, 是否使用16位表示的浮点数，这里是False
* num_predict=None: 用于预测的Token的个数，这里是21


下面我们来看这个函数的代码，请仔细阅读其中的注释。

```

  # 把所有的record info合并成一个
  # 首先读取record_info*.json文件，这个文件是前面介绍的代码生成的，比如
  # 'record_info-train-*.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.json'
  # 从这个文件名可以看成它是"train"的文件，bsz-8表示batch是8，seqlen-128表示句子长度128，
  # reuse-64表示resue是64，alpha-6表示alpha是6，beta-1表示beta是1，fnp-21表示预测的个数是21
  # 因此从这里看成，train_gpu.py的参数一定要和生成pretraining数据的data_utils.py脚本的参数一致。
  # 注意上面不是文件名，而是glob，也就是带通配符的路径，实际的文件可能为：
  # train-0-0.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.tfrecords
  # 我这里只有一个文件，实际上data_utils.py可能会生成很多个这样的文件，比如：
  # train-0-1.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.tfrecords
  # train-0-2.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.tfrecords
  record_glob_base = format_filename(
      prefix="record_info-{}-*".format(split),
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="json",
      mask_alpha=mask_alpha,
      mask_beta=mask_beta,
      reuse_len=reuse_len,
      uncased=uncased,
      fixed_num_predict=num_predict)

  record_info = {"num_batch": 0, "filenames": []}

  tfrecord_dirs = tfrecord_dir.split(",")
  tf.logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)
  
  # 变量目录，这些目录是用逗号分开的 
  for idx, record_dir in enumerate(tfrecord_dirs):
    record_glob = os.path.join(record_dir, record_glob_base)
    tf.logging.info("[%d] Record glob: %s", idx, record_glob)

    record_paths = sorted(tf.gfile.Glob(record_glob))
    tf.logging.info("[%d] Num of record info path: %d",
                    idx, len(record_paths))

    cur_record_info = {"num_batch": 0, "filenames": []}
    # 遍历record_info-train-*.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.json
    # 能匹配的所有json文件，这里只匹配了：
    # record_info-train-0-0.bsz-8.seqlen-128.reuse-64.uncased.bi.alpha-6.beta-1.fnp-21.json
    for record_info_path in record_paths:
      if num_passes is not None:
        # record_info_path为record_info-train-0-0.bsz-8.seqlen-128.....
        # record_info-train-0-0代表这是pretraining的训练数据，最后一个0代表这是它的pass_id
        # 如果pass_id >= num_passes，那么就会跳过，比如某个数据是record_info-train-0-3，
        # 则如果当前训练的趟数num_passes是1,2,3，都会跳过，只有当num_passes大于3时才会训练这个文件
        record_info_name = os.path.basename(record_info_path)
        fields = record_info_name.split(".")[0].split("-")
        pass_id = int(fields[-1])
        if len(fields) == 5 and pass_id >= num_passes:
          tf.logging.info("Skip pass %d: %s", pass_id, record_info_name)
          continue

      with tf.gfile.Open(record_info_path, "r") as fp:
        # 打开这个json文件
        info = json.load(fp)
        if num_passes is not None:
          eff_num_passes = min(num_passes, len(info["filenames"]))
          ratio = eff_num_passes / len(info["filenames"])
          cur_record_info["num_batch"] += int(info["num_batch"] * ratio)
          cur_record_info["filenames"] += info["filenames"][:eff_num_passes]
        else:
          cur_record_info["num_batch"] += info["num_batch"]
          cur_record_info["filenames"] += info["filenames"]

    # 根据json文件找到对应的tfrecord文件，放到`cur_record_info`
    new_filenames = []
    for filename in cur_record_info["filenames"]:
      basename = os.path.basename(filename)
      new_filename = os.path.join(record_dir, basename)
      new_filenames.append(new_filename)
    cur_record_info["filenames"] = new_filenames

    tf.logging.info("[Dir %d] Number of chosen batches: %s",
                    idx, cur_record_info["num_batch"])
    tf.logging.info("[Dir %d] Number of chosen files: %s",
                    idx, len(cur_record_info["filenames"]))
    tf.logging.info(cur_record_info["filenames"])

    # 把`cur_record_info`放到全局的`record_info`里
    record_info["num_batch"] += cur_record_info["num_batch"]
    record_info["filenames"] += cur_record_info["filenames"]

  tf.logging.info("Total number of batches: %d",
                  record_info["num_batch"])
  tf.logging.info("Total number of files: %d",
                  len(record_info["filenames"]))
  tf.logging.info(record_info["filenames"])

  # 返回的input function，调用这个函数后就会得到Dataset对象。
  def input_fn(params):
    # 一个host(机器)的batch大小 = 每个设备(GPU)的batch大小 * 设备的个数
    assert params["batch_size"] * num_core_per_host == bsz_per_host
    # 使用get_datset函数构造Dataset对象，后面我们会详细介绍这个函数
    dataset = get_dataset(
        params=params,
        num_hosts=num_hosts,
        num_core_per_host=num_core_per_host,
        split=split,
        file_names=record_info["filenames"],
        num_batch=record_info["num_batch"],
        seq_len=seq_len,
        reuse_len=reuse_len,
        perm_size=perm_size,
        mask_alpha=mask_alpha,
        mask_beta=mask_beta,
        use_bfloat16=use_bfloat16,
        num_predict=num_predict)

    return dataset

  return input_fn, record_info
```

这个函数主要的代码就是遍历目录下的json文件，然后找到对应的tfrecord文件，放到record_info里，同时返回input_fn函数，如果执行input_fn函数，则它会调用get_dataset函数返回Dataset，下面我们来看get_dataset函数。

### get_dataset函数

代码如下：

```
def get_dataset(params, num_hosts, num_core_per_host, split, file_names,
                num_batch, seq_len, reuse_len, perm_size, mask_alpha,
                mask_beta, use_bfloat16=False, num_predict=None):

  bsz_per_core = params["batch_size"]
  # 我们这里值考虑一台服务器(host_id=0)的情况
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  #### parse tfrecord的函数
  def parser(record):
    # 这个可以参考第一部分生成数据的内容
    # input是输入的句子的id，长度为seq_len=128
    # target是预测的id序列，长度也是seq_len
    # seg_id 64个reused的Token，A和A后的SEP是0，B和B后的SEP是1，最后的CLS是2
    # label 如果两个句子是连续的，那么是1，否则是0
    # is_masked表示某个Token是否是masked(用于预测)
    record_spec = {
        "input": tf.FixedLenFeature([seq_len], tf.int64),
        "target": tf.FixedLenFeature([seq_len], tf.int64),
        "seg_id": tf.FixedLenFeature([seq_len], tf.int64),
        "label": tf.FixedLenFeature([1], tf.int64),
        "is_masked": tf.FixedLenFeature([seq_len], tf.int64),
    }

    # parse
    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)

    inputs = example.pop("input")
    target = example.pop("target")
    is_masked = tf.cast(example.pop("is_masked"), tf.bool)

    non_reuse_len = seq_len - reuse_len
    assert perm_size <= reuse_len and perm_size <= non_reuse_len
    
    # 先处理前64(reuse)，后面会详细介绍这个函数
    # 现在我们知道它的输入是inputs的前64个，target的前64个，is_masked的前64个
    # perm_size是32，reuse_len是64。
    # 它返回的是：
    # 1. perm_mask，64x64，表示经过重新排列后第i个token能否attend to 第j个token，1表示不能attend
    # 2. target，64，表示真实的目标值，之前生成的target是预测下一个词，但是XLNet是预测当前词
    # 3. target_mask，64，哪些地方是Mask的(需要预测的)
    # 4. input_k, 64，content stream的初始值
    # 5. input_q, 64, 哪些位置是需要计算loss的，如果不计算loss，也就不计算Query Stream。
    perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
        inputs[:reuse_len],
        target[:reuse_len],
        is_masked[:reuse_len],
        perm_size,
        reuse_len)

    perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
        inputs[reuse_len:],
        target[reuse_len:],
        is_masked[reuse_len:],
        perm_size,
        non_reuse_len)
    # tf.ones(reuse_len, non_reuse_len)表示前64个reuse的不能attend to 后面64个(1表示不能attend)
    # concat起来就变成(reuse_len=64, 128)，perm_mask_0(i,j)表示i能不能attend to j(128)
    perm_mask_0 = tf.concat([perm_mask_0, tf.ones([reuse_len, non_reuse_len])],
                            axis=1)
    # tf.zeros(non_reuse_len, reuse_len)表示后面的64个可以attend to 前面的64(reuse_len)个
    # concat变成(64,128)，perm_mask_0(i,j)表示i(后面的64个)能不能attent to j(128)
    perm_mask_1 = tf.concat([tf.zeros([non_reuse_len, reuse_len]), perm_mask_1],
                            axis=1)
    # 把perm_mask_0和perm_mask_1 concat起来变成(128,128)
    # perm_mask(i,j)表示i(128)能不能attend to j(128)
    perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=0)
    # target也concat变成(128,)
    target = tf.concat([target_0, target_1], axis=0)
    # target_mask也concat成(128,)
    target_mask = tf.concat([target_mask_0, target_mask_1], axis=0)
    # input_k也concat
    input_k = tf.concat([input_k_0, input_k_1], axis=0)
    # input_q也concat
    input_q = tf.concat([input_q_0, input_q_1], axis=0)

    if num_predict is not None:
      # indices是[0,1,...,127]
      indices = tf.range(seq_len, dtype=tf.int64)
      # target_mask中1表示MASK的值，这里把它变成boolean
      bool_target_mask = tf.cast(target_mask, tf.bool)
      # 找到Mask对应的下标，比如MASK的Token的下标是2和3，那么bool_target_mask=[F,F,T,T,...]
      # tf.boolean_mask函数返回indices里为True的值，因此返回[2,3]
      indices = tf.boolean_mask(indices, bool_target_mask)

      # 因为随机抽样的MASK可能是CLS/SEP，这些是不会被作为预测值的，因此
      # 我们之前生成的数据有num_predict(21)个需要预测的，但实际需要预测的只有actual_num_predict
      # 所以还需要padding num_predict - actual_num_predict个。
      actual_num_predict = tf.shape(indices)[0]
      pad_len = num_predict - actual_num_predict

      # target_mapping
      # 假设indices=[2,3]的话
      # 则target_mapping就变成 [[0,0,1,0,....],[0,0,0,1,....]]
      # 也就是把2和3用one-hot的方法来表示
      target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
      # padding的部分也表示成向量，但是它是"zero-hot"的表示。
      paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
      # concat成[num_redict, seq_len]的向量
      # target_mapping(i,j) = 1 表示第i个要预测的Token的目标(真实)值是j
      target_mapping = tf.concat([target_mapping, paddings], axis=0)
      # 其实不reshape也是可以的。因为除非代码有bug，否则target_mapping就是[num_redict, seq_len]
      # reshape的好处是显式的说明target_mapping的shape，调试方便一点。
      # 读者可能会问，pad_len = num_predict - actual_num_predict，然后我又
      # 把padding的和actual_num_predict的concat起来，为什么TF不知道它的shape呢？
      # 因为TF是一种静态图，pad_len只是一个Operation，还没有执行，TF并不知道它的值。
      example["target_mapping"] = tf.reshape(target_mapping,
                                             [num_predict, seq_len])

      ##### target
      # 拿到target的Token ID
      target = tf.boolean_mask(target, bool_target_mask)
      # 同样需要padding
      paddings = tf.zeros([pad_len], dtype=target.dtype)
      target = tf.concat([target, paddings], axis=0)
      example["target"] = tf.reshape(target, [num_predict])

      ##### target mask
      # 长度为21(num_predict)的向量，1表示是真正需要预测的Token；0表示是padding的，是不计算loss的
      target_mask = tf.concat(
          [tf.ones([actual_num_predict], dtype=tf.float32),
           tf.zeros([pad_len], dtype=tf.float32)],
          axis=0)
      example["target_mask"] = tf.reshape(target_mask, [num_predict])
    else:
      example["target"] = tf.reshape(target, [seq_len])
      example["target_mask"] = tf.reshape(target_mask, [seq_len])

    # reshape back to fixed shape
    example["perm_mask"] = tf.reshape(perm_mask, [seq_len, seq_len])
    example["input_k"] = tf.reshape(input_k, [seq_len])
    example["input_q"] = tf.reshape(input_q, [seq_len])

    _convert_example(example, use_bfloat16)

    for k, v in example.items():
      tf.logging.info("%s: %s", k, v)

    return example

  # Get dataset
  dataset = parse_files_to_dataset(
      parser=parser,
      file_names=file_names,
      split=split,
      num_batch=num_batch,
      num_hosts=num_hosts,
      host_id=host_id,
      num_core_per_host=num_core_per_host,
      bsz_per_core=bsz_per_core)

  return dataset

```

主要的代码都在parser函数里，它定义了怎么读取我们在第一部分代码里生成的tfrecord文件，然后进行排列打散(我们还没有具体介绍的_local_perm函数)，最后把处理的结果放到example里。接着使用parse_files_to_dataset来得到Dataset，我们来看这个函数：

```
def parse_files_to_dataset(parser, file_names, split, num_batch, num_hosts,
                           host_id, num_core_per_host, bsz_per_core):
  # list of file pathes
  num_files = len(file_names)
  num_files_per_host = num_files // num_hosts
  my_start_file_id = host_id * num_files_per_host
  my_end_file_id = (host_id + 1) * num_files_per_host
  if host_id == num_hosts - 1:
    my_end_file_id = num_files
  file_paths = file_names[my_start_file_id: my_end_file_id]
  tf.logging.info("Host %d handles %d files", host_id, len(file_paths))

  assert split == "train"
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # 文件级别的shuffle
  if len(file_paths) > 1:
    dataset = dataset.shuffle(len(file_paths))

  # 注意：这里我们不能对每一个sample进行打散，这样会破坏句子的Token的顺序。
  dataset = tf.data.TFRecordDataset(dataset)

  # (zihang): 因为我们是online的随机排列，因此每次session.run的时候排列的顺序都是不同的，
  # 所以cache是没有作用的，它会导致OOM。因此我们只cache parser之前的数据(cache函数在map(parser)之前)
  # map(parser)就是对每一个tfrecord使用前面介绍的parser函数来处理
  dataset = dataset.cache().map(parser).repeat()
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

  return dataset
```



### _local_perm函数

这个函数比较难懂，这里介绍一个小技巧。如果读者对比过PyTorch和Tensorflow就会感觉使用PyTorch调试会简单很多，原因就是PyTorch是动态的计算图，因此我们把两个Tensor一相加，结果马上就出来了；但是Tensoflow是静态图，我们定义了两个Tensor相加后得到的不是它们的计算结果，而是一个Operation，我们还要用session来run它才能得到结果。如果计算简单还好，一个复杂的的函数经过一系列变换后就完全不知道它的shape是什么了。因此调试PyTorch的代码就像调试普通的Python代码一样；而调试Tensorflow的代码就像"阅读"Pyton代码一样——你看不到执行的结果。不过还有Tensorflow引入了Eager Execution。但是很多代码(包括这里的XLNet)还是习惯用之前的静态构造方法。不过没有关系，如果我们遇到一个函数看不到，那么我们可以对这个函数进行Eager Execution来调试它。比如我们如果看不到_local_perm，则我们可以这样来调试它：


```
import tensorflow as tf
# 开启Eager Execution
tf.enable_eager_execution()
seq_len = 16
reuse_len = 8
perm_size = 8
inputs=tf.constant([10,13,15,20,21,22,4,16,33,34,35,36,37,38,4,3])
targets=tf.constant([13,15,20,21,22,4,16,33,34,35,36,37,38,10,3,3])
is_masked=tf.constant([False, False, False, False, True, True,False,
                   False, False,False, False, False,
                   True, True, False, False])
_local_perm(inputs,targets, is_masked, perm_size, seq_len)
```
其中seq_len表示输入的长度，perm_size表示排列打散的长度。inputs表示输入，targets表示输出，其中4和3是特殊的SEP和CLS。is_masked表示某个输入是否是MASK(用于预测的)，上面的例子里第5和6个位置、第13和14个位置都是Masked，也就是需要模型预测计算loss的位置。

接下来我们就可以增加断点单步执行从而来了解这个函数的作用了。

```
def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
    随机的采样一种排列方式，然后创建对应的attention mask 

    参数:
      inputs: int64 Tensor shape是[seq_len]，输入的id
      targets: int64 Tensor shape是[seq_len]，目标值的id
      is_masked: bool Tensor shape是[seq_len]，True代表用于预测
      perm_size: 最长排列的长度，具体含义参考下面的代码
      seq_len: int, 序列长度
```

#### 生成随机的排列

```
# 随机生成一个下标的排列
index = tf.range(seq_len, dtype=tf.int64)
index = tf.transpose(tf.reshape(index, [-1, perm_size]))
index = tf.random_shuffle(index)
index = tf.reshape(tf.transpose(index), [-1])
```

根据上面的输入，首先用tf.range生成[0, 15]的序列，然后第二行代码首先把它reshape成[2, 8]，然后transpose成[8, 2]，从而得到：
```
0 8
1 9
2 10
3 11
4 12
5 13
6 14
7 15
```

然后使用tf.random_shuffle对第一列进行随机打散，得到：
```
4 12
6 14
7 15
2 10
3 11
5 13
0  8
1  9
```

最后transpose然后reshape成长度为16的向量：
```
[4  6  7  2  3  5  0  1 12 14 15 10 11 13  8  9]
```

总结一下代码的效果：把长度为seq_len(=16)的向量分成seq_len/perm_size(=2)段，每段进行随机打散。

#### 得到特殊Token

1.首先是得到non_func_tokens，所谓的non_func_tokens是指SEP和CLS之外的"正常"的Token。
```
non_func_tokens = tf.logical_not(tf.logical_or(
      tf.equal(inputs, data_utils.SEP_ID),
      tf.equal(inputs, data_utils.CLS_ID)))
# 计算后的non_func_tokens为：

[True  True  True  True  True  True False  True  True  True  True  True True  True False False]
```

根据前面的输入inputs，第7个位置为SEP，15和16为SEP和CLS，因此这三个位置为False，其余都为True。

2.然后是non_mask_tokens，non_mask_tokens指的是"正常"的并且没有被Mask的Token。
```
non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
# non_mask_tokens的值为：
# [True  True  True  True False False False  True  True  True  True  True False False False False]
```

因此如果一个Token是non_mask_tokens，那么它首先是"正常"的Token(non_func_tokens为True)，然后还得没有被Mask(is_masked为False)。

3.masked_or_func_tokens，它和non_mask_tokens相反，包括Masked的Token和SEP与CLS。

```
masked_or_func_tokens = tf.logical_not(non_mask_tokens)
# [False False False False True True True False False False False False True True True True]
```


#### 计算rev_index
```
# 把非Mask(也不是CLS和SEP)的Token的排列下标设置为最小的-1，这样：
# (1) 它们可以被所有其它的位置看到 
# (2) 它们看不到Masked位置，从而不会有信息泄露
smallest_index = -tf.ones([seq_len], dtype=tf.int64)
rev_index = tf.where(non_mask_tokens, smallest_index, index)
# [-1 -1 -1 -1  3  5  0 -1 -1 -1 -1 -1 11 13  8  9]
```
tf.where函数的作用等价于：
```
for i in range(16):
    if non_mask_tokens[i]:
        rev_index[i]=smallest_index[i]
    else:
        smallest_index[i]=index[i]
```
这样得到的rev_index为：如果某个位置是非Mask的，则其值为-1；反正如果某个位置是Mask(5,6,13,14)或者为特殊的SEP/CLS(7,15,16)，则值为前面随机生成的下标。

#### target_mask

```
# 创建`target_mask`: 它是"普通"的并且被Masked的Token，它的值代表：
# 1: 值为1代表使用mask作为输入并且计算loss
# 0: 使用token(或者SEP/CLS)作为输入并且不计算loss
target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens)
target_mask = tf.cast(target_tokens, tf.float32)
# [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
```
我们看到，is_masked为True的下标对应的target_mask为1，其余为0。

### perm_mask

```
# `target_tokens` 不能看到自己
self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)
# [0  0  0  0  3  5  1  0  0  0  0  0 11 13  9 10]
```
因为target_tokens不能看到自己，因此它的值就是rev_index，而其余的加1(如果原来是-1，那么变成0，否则就是排列下标加一)。

```
# 1: 如果i <= j并且j不是非masked(masked或者特殊的SEP/CLS)则不能attend，因此值为1
# 0: 如果i > j或者j非masked，则为0
perm_mask = tf.logical_and(
	self_rev_index[:, None] <= rev_index[None, :],
	masked_or_func_tokens)
perm_mask = tf.cast(perm_mask, tf.float32)
# 值为：
[[0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]]
```

上面的代码"self_rev_index[:, None] <= rev_index[None, :]"我们来详细分析一下。

首先self_rev_index是一个长度为16的Tensor，我们使用self_rev_index[:, None]把它变成(16,1)的Tensor。

rev_index也是长度为的Tensor，rev_index[None, :]变成了(1,16)的Tensor。然后它们比较("<=")时会使用Broadcasting，我们可以了解为self_rev_index[:, None]变成了(16,16)的，每行都是相同的：

```
Broadcasting之前：
[[ 0] 
 [ 0]
 [ 0]
 [ 0]
 [ 3]
 [ 5]
 [ 1]
 [ 0]
 [ 0]
 [ 0]
 [ 0]
 [ 0]
 [11]
 [13]
 [ 9]
 [10]]
Broadcasting之后：
[[ 0, 0, 0, ....] 
 [ 0, 0, 0, ....]
 [ 0, 0, 0, ....]
 [ 0, 0, 0, ....]
 [ 3, 3, 3, ....]
 [ 5, 5, 5, ....]
 [ 1, 1, 1, ....]
 [ 0, 0, 0, ....] 
 [ 0, 0, 0, ....] 
 [ 0, 0, 0, ....] 
 [ 0, 0, 0, ....] 
 [ 0, 0, 0, ....] 
 [11, 11,11,....]
 [13,13, 13,....]
 [ 9, 9, 9, ....]
 [10, 10,10,....]]
```

类似的，rev_index[None, :]在Broadcasting之前为：
```
[-1 -1 -1 -1  3  5  0 -1 -1 -1 -1 -1 11 13  8  9]
```
而Broadcasting之后变成：
```
[[-1 -1 -1 -1  3  5  0 -1 -1 -1 -1 -1 11 13  8  9],
 [-1 -1 -1 -1  3  5  0 -1 -1 -1 -1 -1 11 13  8  9],
....

]
```

因此，最终得到的perm_mask(i,j)=1,则表示i不能attend to j。有两种情况i能attend to j：i的排列下标大于j(后面的可以attend to前面的)；j没有被Mask在i也可以attend to j。而i不能attend to j需要同时满足：i<=j并且j被Mask了我们来看一个例子：perm_mask(3,4)=1，因为第3个Token的排列下标是2，第4个的排列下标是3，所以满足"2<3"。请读者检查确认一下j确实是被Mask了。

#### new target
对于常规的语言模型来说，我们是预测下一个词，而XLNet是根据之前的状态和当前的位置预测被MASK的当前词。所以真正的new_targets要前移一个。
```
# new target: [next token] for LM and [curr token] (self) for PLM
new_targets = tf.concat([inputs[0: 1], targets[: -1]],
                  axis=0)
# [10 13 15 20 21 22  4 16 33 34 35 36 37 38 10 3]
inputs=tf.constant([10,13,15,20,21,22,4,16,33,34,35,36,37,38,4,3])
targets=tf.constant([13,15,20,21,22,4,16,33,34,35,36,37,38,10,3,3])
```

最终的返回值：
```
  # construct inputs_k
  inputs_k = inputs

  # construct inputs_q
  inputs_q = target_mask

  return perm_mask, new_targets, target_mask, inputs_k, inputs_q
```
#### _local_perm的实现和论文的对比

如果读者还没有完全明白，我们再来把代码和论文的描述对比一下。论文在Partial Prediction部分提到：为了提高效率，我们只预测排列最后几个词。比如假设总共4个词，并且假设某次随机排列的顺序为$3 \rightarrow 2 \rightarrow 4 \rightarrow 1$，我们假设预测(Mask)最后两个词，也就是预测第4和第1个词。那么1可以attend to [2,3,4]；4可以attend to [2,3]。而非Mask(预测)的第2和3个词使用普通的Self-Attention，也就是可以互相看到所有的非Mask词(包括自己)，因此2可以attend to [1,2]，1也可以attend to [1,2]。

上面按照论文我们随机排列，然后选择最后的几个词作为Mask；而前面的代码我们先已经选定了要Mask的值，然后再排列，这就可能出现非Mask值的排列下标有可能Mask的值，比如假设我们选定Mask第2个和第3个词，但是随机的排列为：
```
2 3 1 4
```

也就是说第2个词在排列里的顺序是1，第3个词是2。那么按照论文应该是预测第1个和第4个词。那怎么办呢？代码使用了一个tricky，把所有的非Mask的词(第1个和第4个都变成-1)，而Mask的不变(总是大于0的)，因此Mask的词就排在后面了。非Mask的词互相都可以attend to但是非Mask的词不能attend to Mask的词。Mask的词可以attend to 非Mask的词而且后面的Mask的词也能attend to 前面的Mask的词。比如上面的例子2和3都是Mask的词，因为3在2后面，所以3可以attend to 2，但2不能attend to 3。同时，Mask的词不能attend to 自己(否则就是用自己预测自己了)。

如果读者还是没有理解这个函数也没有太多关系，但是至少要知道这个函数的干什么的，也就是输入输出是什么，具体怎么实现的可以当成黑盒。

总结一下，_local_perm返回的值：

* perm_mask，64x64，表示经过重新排列后第i个token能否attend to 第j个token，1表示不能attend
* target，64，表示真实的目标值，之前生成的target是预测下一个词，但是XLNet是预测当前词
* target_mask，64，哪些地方是Mask的(需要预测的)
* input_k, 64，content stream的初始值
* input_q, 64, 哪些位置是需要计算loss的，如果不计算loss，也就不计算Query Stream。


## single_core_graph
这个函数的作用构建XLNet计算图的，是我们需要重点理解的部分。不过这个函数没几行代码：
```
def single_core_graph(is_training, features, mems):
  model_fn = get_model_fn()

  model_ret = model_fn(
      features=features,
      labels=None,
      mems=mems,
      is_training=is_training)

  return model_ret
```

它调用get_model_fn得到model function，然后调用这个函数返回total_loss, new_mems, grads_and_vars这3个Operation，下面我们来看get_model_fn函数。

### get_model_fn

```
def get_model_fn():
  def model_fn(features, labels, mems, is_training):
    #### 根据输入计算loss
    total_loss, new_mems, monitor_dict = function_builder.get_loss(
        FLAGS, features, labels, mems, is_training)

    #### 打印模型的参数(便于调试)
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    # GPU
    assert is_training
    # 得到所有可能训练的变量
    all_vars = tf.trainable_variables()
    # 计算梯度
    grads = tf.gradients(total_loss, all_vars)
    # 把梯度和变量配对，最终返回一个list，这个list的每一个元素都是(grad, var)的pair
    grads_and_vars = list(zip(grads, all_vars))

    return total_loss, new_mems, grads_and_vars

  return model_fn
```

这个函数主要是调用function_builder.get_loss得到total_loss, new_mems，然后计算梯度并且返回。

### function_builder.get_loss
```
def get_loss(FLAGS, features, labels, mems, is_training):
  """Pretraining loss with two-stream attention Transformer-XL."""
  if FLAGS.use_bfloat16:
    with tf.tpu.bfloat16_scope():
      return two_stream_loss(FLAGS, features, labels, mems, is_training)
  else:
    return two_stream_loss(FLAGS, features, labels, mems, is_training)
```

我们这里使用普通的float(float32)而不是压缩的bfloat16，关于bfloat16，有兴趣的读者可以参考[Using bfloat16 with TensorFlow models](https://cloud.google.com/tpu/docs/bfloat16)。最终它们都是调用two_stream_loss函数。

### two_stream_loss

```
def two_stream_loss(FLAGS, features, labels, mems, is_training):
  # two-stream attention Transformer-XL模型的Pretraining loss

  #### Unpack input
  mem_name = "mems" 
  # mems是长度为6的list，每个元素是[96, 8(batch), 1024(hidden state)]
  mems = mems.get(mem_name, None)
  
  # input_k是(8,128)变成(128,8)
  inp_k = tf.transpose(features["input_k"], [1, 0])
  # input_q也是(8,128)，变成(128,8) 
  inp_q = tf.transpose(features["input_q"], [1, 0])
  # seg_id也是(8,128)，变成(128,8)
  seg_id = tf.transpose(features["seg_id"], [1, 0])

  inp_mask = None
  # 从(8,128,128)变成(128,128,8)
  perm_mask = tf.transpose(features["perm_mask"], [1, 2, 0])

  if FLAGS.num_predict is not None:
    # 从(8, 21, 128)变成(21(num_predict), 128, 8)
    target_mapping = tf.transpose(features["target_mapping"], [1, 2, 0])
  else:
    target_mapping = None

  # 语言模型loss的target，从(8,21)变成(21,8)
  tgt = tf.transpose(features["target"], [1, 0])

  # 语言模型losss的target mask，从(8,21)变成(21,8)
  tgt_mask = tf.transpose(features["target_mask"], [1, 0])

  # 构造xlnet的config然后保存到model_dir目录下
  # XLNetConfig包含某个checkpoint特定的超参数
  # 也就是pretraining和finetuing都相同的超参数
  xlnet_config = xlnet.XLNetConfig(FLAGS=FLAGS)
  xlnet_config.to_json(os.path.join(FLAGS.model_dir, "config.json"))

  # 根据FLAGS构造run config，它是XLNet模型Pretraining的超参数。
  run_config = xlnet.create_run_config(is_training, False, FLAGS)
  
  xlnet_model = xlnet.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp_k,
      seg_ids=seg_id,
      input_mask=inp_mask,
      mems=mems,
      perm_mask=perm_mask,
      target_mapping=target_mapping,
      inp_q=inp_q)

  output = xlnet_model.get_sequence_output()
  new_mems = {mem_name: xlnet_model.get_new_memory()}
  lookup_table = xlnet_model.get_embedding_table()

  initializer = xlnet_model.get_initializer()

  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    # LM loss
    lm_loss = modeling.lm_loss(
        hidden=output,
        target=tgt,
        n_token=xlnet_config.n_token,
        d_model=xlnet_config.d_model,
        initializer=initializer,
        lookup_table=lookup_table,
        tie_weight=True,
        bi_data=run_config.bi_data,
        use_tpu=run_config.use_tpu)

  #### Quantity to monitor
  monitor_dict = {}

  if FLAGS.use_bfloat16:
    tgt_mask = tf.cast(tgt_mask, tf.float32)
    lm_loss = tf.cast(lm_loss, tf.float32)

  total_loss = tf.reduce_sum(lm_loss * tgt_mask) / tf.reduce_sum(tgt_mask)
  monitor_dict["total_loss"] = total_loss

  return total_loss, new_mems, monitor_dict
```

### XLNetConfig

XLNetConfig包含某个checkpoint特定的超参数，也就是pretraining和finetuing都相同的超参数。它包括：

* n_layer: int, XLNet的层数，这里是6。
* d_model: int, 隐单元个数，这里是1024。
* n_head: int, attention head的个数，这里是16。
* d_head: int, 每个attention head的大小，要求n_head\*d_head=d_model，这里是64。
* d_inner: int, 全连接网络的隐单元个数，这里是4096。 
* ff_activation: str, "relu"或者"gelu"。
* untie_r: bool, 是否不同层不共享bias，这里为True，也就是每层都有独立的bias。
* n_token: int, 词典大小，这里是32000。


### RunConfig
这是Pretraining的一些超参数：
```
bi_data = {bool} True 表示一个句子会变成两个：一个是正常的，一个是逆序的(反过来)。
clamp_len = {int} -1
dropatt = {float} 0.1 attention的dropout
dropout = {float} 0.1 
init = {str} 'normal' 参数初始化方法，这里是正态分布
init_range = {float} 0.1 对于normal无效果
init_std = {float} 0.02 正态分布的方差
is_training = {bool} True 是否Pretraining
mem_len = {int} 96 
reuse_len = {int} 64
same_length = {bool} False
use_bfloat16 = {bool} False
use_tpu = {bool} False
```

### XLNetModel

这是真正定义XLNet模型的类。它定义XLNet的代码都在构造函数里，其余的一些函数都是一些getter类函数，比如get_sequence_output可以拿到最后一层的隐状态。我们首先来看它的构造函数的参数。

#### 构造函数的参数


* xlnet_config: XLNetConfig，Pretraining和Fine-tuning都一样的超参数
* run_config: RunConfig，Pretraining的超参数
* input_ids: int32 Tensor，shape是[len, bsz], 输入token的ID
* seg_ids: int32 Tensor，shape是[len, bsz], 输入的segment ID
* input_mask: float32 Tensor，shape是[len, bsz], 输入的mask，0是真正的tokens而1是padding的
* mems: list，每个元素是float32 Tensors，shape是[mem_len, bsz, d_model], 上一个batch的memory
* perm_mask: float32 Tensor，shape是[len, len, bsz]。
    * 如果perm_mask[i, j, k] = 0，则batch k的第i个Token可以attend to j
    * 如果perm_mask[i, j, k] = 1, 则batch k的第i个Token不可以attend to j
    * 如果是None，则每个位置都可以attend to 所有其它位置(包括自己)
* target_mapping: float32 Tensor，shape是[num_predict, len, bsz]
    * 如果target_mapping[i, j, k] = 1，则batch k的第i个要预测的是第j个Token，这是一种one-hot表示
    * 只是在pretraining的partial prediction时使用，finetuning时设置为None
* inp_q: float32 Tensor，shape是[len, bsz]
    * 需要计算loss的(Mask的位置)为1，不需要的值为0，只在pretraining使用，finetuning时应为None


注意：这里的input_mask不是我们之前介绍的is_masked，is_masked是表示这个位置的Token是被预测的。而这里的input_mask其实指的是padding，我们这里是Pretraining，所有的Token都是真实的，因此传入的为None，后面的代码会把None当成没有padding处理。


#### 构造函数

```
  def __init__(self, xlnet_config, run_config, input_ids, seg_ids, input_mask,
               mems=None, perm_mask=None, target_mapping=None, inp_q=None,
               **kwargs):
    # 初始化类
    initializer = _get_initializer(run_config)
    # 所有超参数
    tfm_args = dict(
        n_token=xlnet_config.n_token,
        initializer=initializer,
        attn_type="bi",
        n_layer=xlnet_config.n_layer,
        d_model=xlnet_config.d_model,
        n_head=xlnet_config.n_head,
        d_head=xlnet_config.d_head,
        d_inner=xlnet_config.d_inner,
        ff_activation=xlnet_config.ff_activation,
        untie_r=xlnet_config.untie_r,

        is_training=run_config.is_training,
        use_bfloat16=run_config.use_bfloat16,
        use_tpu=run_config.use_tpu,
        dropout=run_config.dropout,
        dropatt=run_config.dropatt,

        mem_len=run_config.mem_len,
        reuse_len=run_config.reuse_len,
        bi_data=run_config.bi_data,
        clamp_len=run_config.clamp_len,
        same_length=run_config.same_length
    )
    # 所有输入
    input_args = dict(
        inp_k=input_ids,
        seg_id=seg_ids,
        input_mask=input_mask,
        mems=mems,
        perm_mask=perm_mask,
        target_mapping=target_mapping,
        inp_q=inp_q)
    tfm_args.update(input_args)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      (self.output, self.new_mems, self.lookup_table
          ) = modeling.transformer_xl(**tfm_args)

    self.input_mask = input_mask
    self.initializer = initializer
    self.xlnet_config = xlnet_config
    self.run_config = run_config
```


上面的构造函数核心的代码其实只有一行：
```
(self.output, self.new_mems, self.lookup_table) = modeling.transformer_xl(**tfm_args)
```

#### transformer_xl构造函数的参数

它的构造函数的参数是前面XLNetModel的构造函数传过来的，不过我们还是列举一下。

* inp_k: int32 Tensor，shape是[len, bsz], 输入token的ID
* seg_ids: int32 Tensor，shape是[len, bsz], 输入的segment ID
* input_mask: float32 Tensor，shape是[len, bsz], 输入的mask，0是真正的tokens而1是padding的
* mems: list，每个元素是float32 Tensors，shape是[mem_len, bsz, d_model], 上一个batch的memory
* perm_mask: float32 Tensor，shape是[len, len, bsz]。
    * 如果perm_mask[i, j, k] = 0，则batch k的第i个Token可以attend to j
    * 如果perm_mask[i, j, k] = 1, 则batch k的第i个Token不可以attend to j
    * 如果是None，则每个位置都可以attend to 所有其它位置(包括自己)
* target_mapping: float32 Tensor，shape是[num_predict, len, bsz]
    * 如果target_mapping[i, j, k] = 1，则batch k的第i个要预测的是第j个Token，这是一种one-hot表示
    * 只是在pretraining的partial prediction时使用，finetuning时设置为None
* inp_q: float32 Tensor，shape是[len, bsz]
    * 需要计算loss的(Mask的位置)为1，不需要的值为0，只在pretraining使用，finetuning时应为None

* n_layer: int, XLNet的层数，这里是6。
* d_model: int, 隐单元个数，这里是1024。
* n_head: int, attention head的个数，这里是16。
* d_head: int, 每个attention head的大小，要求n_head\*d_head=d_model，这里是64。
* d_inner: int, 全连接网络的隐单元个数，这里是4096。 
* ff_activation: str, "relu"或者"gelu"。
* untie_r: bool, 是否不同层不共享bias，这里为True，也就是每层都有独立的bias。
* n_token: int, 词典大小，这里是32000。

* is_training: bool, 是否是Training
* use_tpu: bool, 是否使用TPU
* use_bfloat16: bool, 是否用bfloat16替代float32
* dropout: float, dropout大小.
* dropatt: float, attention概率的dropout
* init: str, 初始化方法，值为"normal"或者"uniform"
* init_range: float, 均匀分布的范围，[-init_range, init_range]。只有init="uniform"时有效
* init_std: float, 正态分布的方程。只有init="normal"时有效
* mem_len: int, cache的token个数
* reuse_len: int, 当前batch里reuse的数量，参考第一部分。
* bi_data: bool, 是否双向处理输入，通常在pretraining设置为True，而finetuning为False
* clamp_len: int, clamp掉相对距离大于clamp_len的attention，-1表示不clamp
* same_length: bool, 是否对于每个Token使用相同的attention长度 
* summary_type: str, "last", "first", "mean"和"attn"。怎么把最后一层的多个向量整合成一个向量
* initializer: 初始化器
* scope: 计算图的scope

#### transformer_xl构造函数

下面我们来分段阅读这个最重要的函数。
##### 第一段
```
  with tf.variable_scope(scope):
    if untie_r:
      # 我们这里是不共享bias
      # r_w_bias和r_r_bias都是[6, 16, 64]
      r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                 dtype=tf_float, initializer=initializer)
      r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                 dtype=tf_float, initializer=initializer)
    else:
      r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                 dtype=tf_float, initializer=initializer)
      r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                 dtype=tf_float, initializer=initializer)

    bsz = tf.shape(inp_k)[1]    # 8
    qlen = tf.shape(inp_k)[0] # 128
    mlen = tf.shape(mems[0])[0] if mems is not None else 0 # 96
    klen = mlen + qlen # 224

```

上面的代码注意是定义r_w_bias和r_r_bias，以及读取一些超参数。
##### 第二段
```
    ##### Attention mask
    # 因果关系的(causal)attention mask
    if attn_type == 'uni':
      attn_mask = _create_mask(qlen, mlen, tf_float, same_length)
      attn_mask = attn_mask[:, :, None, None]
    elif attn_type == 'bi':
      # 我们走的这个分支
      attn_mask = None
    else:
      raise ValueError('Unsupported attention type: {}'.format(attn_type))

    # data mask: input mask & perm mask
    if input_mask is not None and perm_mask is not None:
      data_mask = input_mask[None] + perm_mask
    elif input_mask is not None and perm_mask is None:
      data_mask = input_mask[None]
    elif input_mask is None and perm_mask is not None:
      # data_mask=perm_mask [128,128,8]
      data_mask = perm_mask
    else:
      data_mask = None

    if data_mask is not None:
      # 所有的mems的可以attended to，因此构造[128,96,8]的zeros
      mems_mask = tf.zeros([tf.shape(data_mask)[0], mlen, bsz],
                           dtype=tf_float)
      # 然后拼接成[128,224,8]的data_mask。
      # data_mask[i,j,k]=0则表示batch k的第i(0-128)个Token可以attend to 
      # 第j(0-224，包括前面的mem)个Token。
      # 注意j的下标范围，0-95表示mem，96-223表示128个输入。
      data_mask = tf.concat([mems_mask, data_mask], 1)
      if attn_mask is None:
        # attn_mask为[128, 224, 8, 1]
        attn_mask = data_mask[:, :, :, None]
      else:
        attn_mask += data_mask[:, :, :, None]

    if attn_mask is not None:
      # 把attn_mask里大于0的变成1(前面相加有可能大于1)
      attn_mask = tf.cast(attn_mask > 0, dtype=tf_float)

    if attn_mask is not None:
      # 参考下面
      non_tgt_mask = -tf.eye(qlen, dtype=tf_float)
      non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=tf_float),
                                non_tgt_mask], axis=-1)
      non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                             dtype=tf_float)
    else:
      non_tgt_mask = None
```
下面来看一些non_tgt_mask，为了简单，我们假设qlen是4， mlen是3，前两行的结果为：
```
0 0 0   -1 0 0 0
0 0 0   0 -1 0 0
0 0 0   0 0 -1 0
0 0 0   0 0 0 -1
```

attn_mask是(qlen, qlen+mlen, batch, 1)，它和non_tgt_mask[:,:,None,None]相加。它的作用是让Token不能attend to 自己，除后面的对角线外，non_tgt_mask等于attn_mask，而对角线的位置由1变成了0。

##### 第三段

```
    ##### Word embedding
    # 输入inp_k是(128,8)，embedding后的word_emb_k是(128,8,1024)
    # lookup_table是(32000, 1024)
    word_emb_k, lookup_table = embedding_lookup(
        x=inp_k,
        n_token=n_token,
        d_embed=d_model,
        initializer=initializer,
        use_tpu=use_tpu,
        dtype=tf_float,
        scope='word_embedding')
    # inp_q是(128,8)表示某个位置是否要计算loss
    if inp_q is not None:
      with tf.variable_scope('mask_emb'):
        # mask_emb是[1, 1, 1024]
        mask_emb = tf.get_variable('mask_emb', [1, 1, d_model], dtype=tf_float)
        if target_mapping is not None:
          # tile(复制)成[21, 8, 1024]
          word_emb_q = tf.tile(mask_emb, [tf.shape(target_mapping)[0], bsz, 1])
        else:
          inp_q_ext = inp_q[:, :, None]
          word_emb_q = inp_q_ext * mask_emb + (1 - inp_q_ext) * word_emb_k
    # output_h是word_emb_k的dropout的结果，shape也是[128, 8, 1024]
    output_h = tf.layers.dropout(word_emb_k, dropout, training=is_training)
    if inp_q is not None:
      # output_g是word_emb_q的dropout，shape也是[21, 8, 1024]
      output_g = tf.layers.dropout(word_emb_q, dropout, training=is_training)
```

上面的output_h和output_g分别是two-stream的初始输入。

##### 第四段
```
    ##### Segment embedding
    if seg_id is not None:
      if untie_r:
        # [6, 16, 64]
        r_s_bias = tf.get_variable('r_s_bias', [n_layer, n_head, d_head],
                                   dtype=tf_float, initializer=initializer)
      else:
        # default case (tie)
        r_s_bias = tf.get_variable('r_s_bias', [n_head, d_head],
                                   dtype=tf_float, initializer=initializer)
      # [6, 2, 16, 64]
      seg_embed = tf.get_variable('seg_embed', [n_layer, 2, n_head, d_head],
                                  dtype=tf_float, initializer=initializer)

      # Convert `seg_id` to one-hot `seg_mat`
      mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)
      # 输入需要padding 96个0，最终变成(224, 8)
      cat_ids = tf.concat([mem_pad, seg_id], 0)

      # `1` indicates not in the same segment [qlen x klen x bsz]
      # 参考下面的说明。
      seg_mat = tf.cast(
          tf.logical_not(tf.equal(seg_id[:, None], cat_ids[None, :])),
          tf.int32)
      seg_mat = tf.one_hot(seg_mat, 2, dtype=tf_float)
    else:
      seg_mat = None
```

seg_embed是Segment embedding，XLNet是相对Segment编码：如果两个Token在同一个Segment则是0；否则是1。所以第二维是2。

在阅读seg_mat那段时我们首先需要熟悉Tensorflow增加维度的方法。

seg_id是[128, 8]，那么seg_id[:, None]的shape呢？有的读者可能会猜测是[128, 8, 1]\(我一开始也是这么以为)，但这是不对的。[:, None]的意思是在第二个维度增加一维，而原来的第二维(8)被往后推到第三维了，因此seg_id[:, None]的shape是[128, 1, 8]，它等价于seg_id[:, None, :]。而cat_ids[None, :]是在第一个维度增加一维，因此是[1, 224, 8]。

接下来tf.equal(seg_id[:, None], cat_ids[None, :])会首先进行broadcasting：

```
seg_id[:, None]: [128, 1, 8] -> [128, 224, 8] 
cat_ids[None, :]: [1, 224, 8] -> [128, 224, 8]
```

计算的结果是：如果(i,j)的seg_id相同，则为True，否则为False。注意i的取值范围是0-127；而j是0-223。因为我们计算Attention时是用128个去attend to 224(包括96个mem)。最后在用tf.logical_not反过来：1表示在不同Segment而0表示同一个Segment。

最后表示成one-hot的方式(在同一个Segment为[1,0]；不同的Segment为[0,1])，变成[128, 224, 8, 2]，第四位就是one-hot。



**本想一气把train函数写完，但今天太晚了就先到这里吧。**


请继续阅读[XLNet代码分析(三)](/2019/07/20/xlnet-codes3/)。

