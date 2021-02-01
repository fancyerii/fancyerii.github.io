---
layout:     post
title:      "Tensor2Tensor中英翻译代码阅读(三)" 
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

本系列文章是Tensor2Tensor的代码阅读，主要关注中英翻译的实现。本文是第三篇，继续介绍训练数据生成的代码。

<!--more-->

**目录**
* TOC
{:toc}

本文接着上文继续阅读训练数据生成的代码。

构造完词典source_vocab(SubwordTextEncoder)和target_vocab之后，就可以用它们把句子(字符串)变成id序列了。不过首先需要预处理训练数据，这就是translate.py里的compile_data函数。
```
def compile_data(tmp_dir, datasets, filename, datatypes_to_clean=None):
  """Concatenates all `datasets` and saves to `filename`."""
  datatypes_to_clean = datatypes_to_clean or []
  filename = os.path.join(tmp_dir, filename)
  lang1_fname = filename + ".lang1"
  lang2_fname = filename + ".lang2"
  if tf.gfile.Exists(lang1_fname) and tf.gfile.Exists(lang2_fname):
    tf.logging.info("Skipping compile data, found files:\n%s\n%s", lang1_fname,
                    lang2_fname)
    return filename
  with tf.gfile.GFile(lang1_fname, mode="w") as lang1_resfile:
    with tf.gfile.GFile(lang2_fname, mode="w") as lang2_resfile:
      for dataset in datasets:
        url = dataset[0]
        compressed_filename = os.path.basename(url)
        compressed_filepath = os.path.join(tmp_dir, compressed_filename)
        if url.startswith("http"):
          generator_utils.maybe_download(tmp_dir, compressed_filename, url)
        if compressed_filename.endswith(".zip"):
          zipfile.ZipFile(os.path.join(compressed_filepath),
                          "r").extractall(tmp_dir)

        if dataset[1][0] == "tmx":
          cleaning_requested = "tmx" in datatypes_to_clean
          tmx_filename = os.path.join(tmp_dir, dataset[1][1])
          if tmx_filename.endswith(".gz"):
            with gzip.open(tmx_filename, "rb") as tmx_file:
              _tmx_to_source_target(tmx_file, lang1_resfile, lang2_resfile,
                                    do_cleaning=cleaning_requested)
          else:
            with tf.gfile.Open(tmx_filename) as tmx_file:
              _tmx_to_source_target(tmx_file, lang1_resfile, lang2_resfile,
                                    do_cleaning=cleaning_requested)

        elif dataset[1][0] == "tsv":
          _, src_column, trg_column, glob_pattern = dataset[1]
          filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          if not filenames:
            # Capture *.tgz and *.tar.gz too.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
            filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          for tsv_filename in filenames:
            if tsv_filename.endswith(".gz"):
              new_filename = tsv_filename.strip(".gz")
              generator_utils.gunzip_file(tsv_filename, new_filename)
              tsv_filename = new_filename
            with tf.gfile.Open(tsv_filename) as tsv_file:
              for line in tsv_file:
                if line and "\t" in line:
                  parts = line.split("\t")
                  source, target = parts[src_column], parts[trg_column]
                  source, target = source.strip(), target.strip()
                  clean_pairs = [(source, target)]
                  if "tsv" in datatypes_to_clean:
                    clean_pairs = cleaner_en_xx.clean_en_xx_pairs(clean_pairs)
                  for source, target in clean_pairs:
                    if source and target:
                      lang1_resfile.write(source)
                      lang1_resfile.write("\n")
                      lang2_resfile.write(target)
                      lang2_resfile.write("\n")

        else:
          lang1_filename, lang2_filename = dataset[1]
          lang1_filepath = os.path.join(tmp_dir, lang1_filename)
          lang2_filepath = os.path.join(tmp_dir, lang2_filename)
          is_sgm = (
              lang1_filename.endswith("sgm") and lang2_filename.endswith("sgm"))

          if not (tf.gfile.Exists(lang1_filepath) and
                  tf.gfile.Exists(lang2_filepath)):
            # For .tar.gz and .tgz files, we read compressed.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
          if lang1_filepath.endswith(".gz"):
            new_filepath = lang1_filepath.strip(".gz")
            generator_utils.gunzip_file(lang1_filepath, new_filepath)
            lang1_filepath = new_filepath
          if lang2_filepath.endswith(".gz"):
            new_filepath = lang2_filepath.strip(".gz")
            generator_utils.gunzip_file(lang2_filepath, new_filepath)
            lang2_filepath = new_filepath

          for example in text_problems.text2text_txt_iterator(
              lang1_filepath, lang2_filepath):
            line1res = _preprocess_sgm(example["inputs"], is_sgm)
            line2res = _preprocess_sgm(example["targets"], is_sgm)
            clean_pairs = [(line1res, line2res)]
            if "txt" in datatypes_to_clean:
              clean_pairs = cleaner_en_xx.clean_en_xx_pairs(clean_pairs)
            for line1res, line2res in clean_pairs:
              if line1res and line2res:
                lang1_resfile.write(line1res)
                lang1_resfile.write("\n")
                lang2_resfile.write(line2res)
                lang2_resfile.write("\n")

  return filename

```
它的作用就是产生/home/lili/t2tcn2/tmp/wmt_enzh_8192k_tok_train.lang1和/home/lili/t2tcn2/tmp/wmt_enzh_8192k_tok_train.lang2。也就是训练的句对，它们分别是中文和英文的句子，按行一一对应。在这里，它几乎技术把training-parallel-nc-v13/news-commentary-v13.zh-en.en复制到wmt_enzh_8192k_tok_train.lang1，只是txt_line_iterator会把前后的空格strip掉。

然后我们回到generate_encoded_samples：
```
def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
  if dataset_split == problem.DatasetSplit.TRAIN:
    mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
  elif dataset_split == problem.DatasetSplit.EVAL:
    mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

  generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
  encoder = self.get_or_create_vocab(data_dir, tmp_dir)
  return text2text_generate_encoded(generator, encoder,
                                    has_inputs=self.has_inputs,
                                    inputs_prefix=self.inputs_prefix,
                                    targets_prefix=self.targets_prefix)
```
其中encoder就是我们前面的SubwordTextEncoder，generator会生成一个sample(句对)，后面我们会看到它们的用法。总之text2text_generate_encoded是一个generator，它会生成一个sample：
```
def text2text_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True,
                               inputs_prefix="",
                               targets_prefix=""):
  """Encode Text2Text samples from the generator with the vocab."""
  targets_vocab = targets_vocab or vocab
  for sample in sample_generator:
    if has_inputs:
      sample["inputs"] = vocab.encode(inputs_prefix + sample["inputs"])
      sample["inputs"].append(text_encoder.EOS_ID)
    sample["targets"] = targets_vocab.encode(targets_prefix + sample["targets"])
    sample["targets"].append(text_encoder.EOS_ID)
    yield sample
```
也就是它对generator生成的句对进行encode(分词和subword切分并变成id，添加EOS)

下面的步骤就是generate_files函数，它利用text2text_generate_encoded，它生成的id保存到TFRecord文件，用于训练：
```
def generate_files(generator, output_filenames,
                   max_cases=None, cycle_every_n=1):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filenames: List of output file paths.
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.
    cycle_every_n: how many cases from the generator to take before
      switching to the next shard; by default set to 1, switch every case.
  """
  if outputs_exist(output_filenames):
    tf.logging.info("Skipping generator because outputs files exists at {}"
                    .format(output_filenames))
    return
  tmp_filenames = [fname + ".incomplete" for fname in output_filenames]
  num_shards = len(output_filenames)
  # Check if is training or eval, ref: train_data_filenames().
  if num_shards > 0:
    if "-train" in output_filenames[0]:
      tag = "train"
    elif "-dev" in output_filenames[0]:
      tag = "eval"
    else:
      tag = "other"

  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filenames]
  counter, shard = 0, 0
  for case in generator:
    if case is None:
      continue
    if counter % 100000 == 0:
      tf.logging.info("Generating case %d." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    example = to_example(case)
    writers[shard].write(example.SerializeToString())
    if counter % cycle_every_n == 0:
      shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filenames, output_filenames):
    tf.gfile.Rename(tmp_name, final_name)

  if num_shards > 0:
    if tag == "train":
      mlperf_log.transformer_print(
          key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES, value=counter)
    elif tag == "eval":
      mlperf_log.transformer_print(
          key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES, value=counter)

  tf.logging.info("Generated %s Examples", counter)

```
它的输出会保存在文件里：


<a name='img8'>![](/img/t2t-code/8.png)</a>
*图：output_filenames*

核心代码很简单：
```
for case in generator:
  if case is None:
    continue
  if counter % 100000 == 0:
    tf.logging.info("Generating case %d." % counter)
  counter += 1
  if max_cases and counter > max_cases:
    break
  example = to_example(case)
  writers[shard].write(example.SerializeToString())
  if counter % cycle_every_n == 0:
    shard = (shard + 1) % num_shards
```
对于generator生成的每一个sample，把它用to_example变成tf.train.Example，然后存到文件。cycle_every_n是每隔几个sample切换shard。默认1，也就是把第一个sample写到shard1，第二个sample写到shard2，……。

现在我们回过头看看generator是怎么生成一个sample的。首先是text2text_txt_iterator生成inputs和targets：
```
def text2text_txt_iterator(source_txt_path, target_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  for inputs, targets in zip(
      txt_line_iterator(source_txt_path), txt_line_iterator(target_txt_path)):
    yield {"inputs": inputs, "targets": targets}
```
我们这里的第一个句对例子是{'inputs': '1929 or 1989?', 'targets': '1929年还是1989年?'}，encoder的代码：
```
def encode(self, s):
  """Converts a native string to a list of subtoken ids.

  Args:
    s: a native string.
  Returns:
    a list of integers in the range [0, vocab_size)
  """
  return self._tokens_to_subtoken_ids(
      tokenizer.encode(native_to_unicode(s)))

```
分词的结果为<class 'list'>: ['1929', 'or', '1989', '?']，接着把每个token切分成subword并且变成subword的id：
```
def _tokens_to_subtoken_ids(self, tokens):
  """Converts a list of tokens to a list of subtoken ids.

  Args:
    tokens: a list of strings.
  Returns:
    a list of integers in the range [0, vocab_size)
  """
  ret = []
  for token in tokens:
    ret.extend(self._token_to_subtoken_ids(token))
  return ret
```
遍历tokens的每一个token，然后用_token_to_subtoken_ids()把每个token切分成subword并且变成id：
```
def _token_to_subtoken_ids(self, token):
  """Converts token to a list of subtoken ids.

  Args:
    token: a string.
  Returns:
    a list of integers in the range [0, vocab_size)
  """
  cache_location = hash(token) % self._cache_size
  cache_key, cache_value = self._cache[cache_location]
  if cache_key == token:
    return cache_value
  ret = self._escaped_token_to_subtoken_ids(
      _escape_token(token, self._alphabet))
  self._cache[cache_location] = (token, ret)
  return ret
```
这个函数会有个cache机制，如果cache没有命中，则执行ret = self._escaped_token_to_subtoken_ids( _escape_token(token, self._alphabet))，也就是先escape，然后在切分subword并且变成id。
```
def _escape_token(token, alphabet):
  """Escape away underscores and OOV characters and append '_'.

  This allows the token to be expressed as the concatenation of a list
  of subtokens from the vocabulary. The underscore acts as a sentinel
  which allows us to invertibly concatenate multiple such lists.

  Args:
    token: A unicode string to be escaped.
    alphabet: A set of all characters in the vocabulary's alphabet.

  Returns:
    escaped_token: An escaped unicode string.

  Raises:
    ValueError: If the provided token is not unicode.
  """
  if not isinstance(token, six.text_type):
    raise ValueError("Expected string type for token, got %s" % type(token))

  token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
  ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
  return u"".join(ret) + "_"
```
escape函数的功能是：把反斜杠变成两个反斜杠；把下划线变成'\u'；把字母表之外的符号变成'\123;'——其中123代表这个字符的unicode的十进制数。最后再加上一个下划线。所以'1929'变成了'1929_'，接着在_escaped_token_to_subtoken_strings被切分成<class 'list'>: ['192', '9_']
```
def _escaped_token_to_subtoken_strings(self, escaped_token):
  """Converts an escaped token string to a list of subtoken strings.

  Args:
    escaped_token: An escaped token as a unicode string.
  Returns:
    A list of subtokens as unicode strings.
  """
  # NOTE: This algorithm is greedy; it won't necessarily produce the "best"
  # list of subtokens.
  ret = []
  start = 0
  token_len = len(escaped_token)
  while start < token_len:
    for end in range(
        min(token_len, start + self._max_subtoken_len), start, -1):
      subtoken = escaped_token[start:end]
      if subtoken in self._subtoken_string_to_id:
        ret.append(subtoken)
        start = end
        break

    else:  # Did not break
      # If there is no possible encoding of the escaped token then one of the
      # characters in the token is not in the alphabet. This should be
      # impossible and would be indicative of a bug.
      assert False, "Token substring not found in subtoken vocabulary."

  return ret
```

最后通过词典变成id：<class 'list'>: [4943, 528]。

最终sample['inputs']变成了<class 'list'>: [4943, 528, 38, 4731, 114, 1]，其中1是<EOS>。

中文被分词为<class 'list'>: ['1929年还是1989年', '?']，最后也变成subword的id。

最后来看一下to_example：
```
def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s" % str((k, v)))
    # Subtly in PY2 vs PY3, map is not scriptable in py3. As a result,
    # map objects will fail with TypeError, unless converted to a list.
    if six.PY3 and isinstance(v, map):
      v = list(v)
    if (isinstance(v[0], six.integer_types) or
        np.issubdtype(type(v[0]), np.integer)):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))

```
我们这里的k是'inputs'，v是整数list，所以执行的是features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))。最后就是用features构造tf.train.Example，这个Example里其实就是inputs和targets的两个整数list(Int64List)。

总用量 36572
drwxrwxr-x  2 lili lili    4096 1月  19 15:17 ./
drwxrwxr-x 11 lili lili    4096 1月  11 15:27 ../
-rw-rw-r--  1 lili lili 3734782 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00000-of-00010
-rw-rw-r--  1 lili lili 3734573 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00001-of-00010
-rw-rw-r--  1 lili lili 3717090 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00002-of-00010
-rw-rw-r--  1 lili lili 3731031 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00003-of-00010
-rw-rw-r--  1 lili lili 3724367 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00004-of-00010
-rw-rw-r--  1 lili lili 3721636 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00005-of-00010
-rw-rw-r--  1 lili lili 3718926 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00006-of-00010
-rw-rw-r--  1 lili lili 3732492 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00007-of-00010
-rw-rw-r--  1 lili lili 3722908 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00008-of-00010
-rw-rw-r--  1 lili lili 3709790 1月  19 15:17 translate_enzh_wmt8k-unshuffled-train-00009-of-00010
-rw-rw-r--  1 lili lili   72026 1月  18 14:01 vocab.translate_enzh_wmt8k.8192.subwords.en
-rw-rw-r--  1 lili lili   62094 1月  18 15:05 vocab.translate_enzh_wmt8k.8192.subwords.zh

最后一步是shuffle：
```
def shuffle_dataset(filenames, extra_fn=None):
  """Shuffles the dataset.

  Args:
    filenames: a list of strings
    extra_fn: an optional function from list of records to list of records
      to be called after shuffling a file.
  """
  if outputs_exist(filenames):
    tf.logging.info("Skipping shuffle because output files exist")
    return
  tf.logging.info("Shuffling data...")
  for filename in filenames:
    _shuffle_single(filename, extra_fn=extra_fn)
  tf.logging.info("Data shuffled.")
```
对于这11个文件，每个都调用_shuffle_single函数进行shuffle。
```
def _shuffle_single(fname, extra_fn=None):
  """Shuffle a single file of records.

  Args:
    fname: a string
    extra_fn: an optional function from list of TFRecords to list of TFRecords
      to be called after shuffling.
  """
  records = read_records(fname)
  random.shuffle(records)
  if extra_fn is not None:
    records = extra_fn(records)
  out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
  write_records(records, out_fname)
  tf.gfile.Remove(fname)
```
这里的代码主要是使用read_records把tfrecord从文件读到内存，然后shuffle，最后再写回去：
```
def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info("read: %d", len(records))
  return records
```

到了这里，生成训练数据的代码阅读完毕！我们已经深入了解了t2t是怎么把机器翻译的双语句对变成tfrecord的过程。


