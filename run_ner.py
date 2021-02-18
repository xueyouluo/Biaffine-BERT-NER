#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os, sys
import pickle
import json
import pdb

import tensorflow as tf
import numpy as np

import modeling
import optimization
import tokenization

import time

from tqdm import tqdm

# 这里为了避免打印重复的日志信息
tf.get_logger().propagate = False

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_bool("output_score", False, "Whether to output ner score.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer(
    "train_batch_size", 64,
    "Total batch size for training.")

flags.DEFINE_integer(
    "eval_batch_size", 16,
    "Total batch size for eval.")

flags.DEFINE_integer(
    "predict_batch_size", 16,
    "Total batch size for predict.")

flags.DEFINE_float(
    "learning_rate", 5e-6,
    "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 10.0,
    "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")


flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_bool("amp", False, "Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.")
flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, span_mask, gold_labels ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.span_mask = span_mask
        self.gold_labels = gold_labels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class CLUENERProcessor(DataProcessor):
    def get_train_examples(self, data_dir, file_name='train.json'):
      examples = []
      for i,line in enumerate(open(os.path.join(data_dir, file_name))):
        item = json.loads(line)
        guid = "%s-%s" % ('train',i)
        text = item['text']
        label = item['label']
        label = self.check(text,label)
        examples.append(InputExample(guid=guid,text=text,label=label))
      return examples

    def get_dev_examples(self, data_dir, file_name="dev.json"):
        examples = []
        for i,line in enumerate(open(os.path.join(data_dir,file_name))):
            item = json.loads(line)
            guid = '%s-%s' %('dev',i)
            examples.append(InputExample(guid=guid,text=item['text'],label=None))
        return examples
    
    def get_test_examples(self, data_dir, file_name="test.json"):
        examples = []
        for i,line in enumerate(open(os.path.join(data_dir,file_name))):
            item = json.loads(line)
            guid = '%s-%s' %('test',i)
            examples.append(InputExample(guid=guid,text=item['text'],label=None))
        return examples

    def get_labels(self):
        return ['O','address','book','company','game','government','movie','name','organization','position','scene']

    def check(self, text, label):
      new_labels = []
      for key in label:
        for name,positions in label[key].items():
          for s,e in positions:
            try:
              assert text[s:e+1] == name
            except:
              # 你不应该来到这里，来了说明数据出问题了
              pdb.set_trace()
            new_labels.append((s,e,key))
      return new_labels

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, is_training):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens = []
    text = example.text
    if tokenizer.basic_tokenizer.do_lower_case:
      text = text.lower()
    
    # 及其简化的tokenizer，把每个字符都拆开
    tokens = [t if t in tokenizer.vocab else tokenizer.wordpiece_tokenizer.unk_token for t in text]
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    span_mask = []

    ntokens.append("[CLS]")
    segment_ids.append(0)
    span_mask.append(0)
    for i, token in enumerate(tokens):
      ntokens.append(token)
      segment_ids.append(0)
      span_mask.append(1)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    span_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        span_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(span_mask) == max_seq_length

    gold_labels = []
    if is_training:
      ner = {(s,e):label_map[t] for s,e,t in example.label}
      for s in range(len(text)):
        for e in range(s,len(text)):
          gold_labels.append(ner.get((s,e),0))

      if ex_index < 5:
          tf.compat.v1.logging.info("*** Example ***")
          tf.compat.v1.logging.info("guid: %s" % (example.guid))
          tf.compat.v1.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in tokens]))
          tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
          tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
          tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
          tf.compat.v1.logging.info("span_mask: %s" % " ".join([str(x) for x in span_mask]))
          tf.compat.v1.logging.info("gold_labels: {}".format(gold_labels))
    else:
        gold_labels.append(0)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        span_mask=span_mask,
        gold_labels=gold_labels,
    )
    return feature

def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, is_training=True):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                                         is_training)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features['span_mask'] = create_int_feature(feature.span_mask)
        features['gold_labels'] = create_int_feature(feature.gold_labels)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, batch_size, seq_length, is_training, drop_remainder=False, hvd=None):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "span_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "gold_labels": tf.io.VarLenFeature(tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            if name == 'gold_labels':
              t = tf.sparse_tensor_to_dense(t)
            example[name] = t
        return example

    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.map(lambda record: _decode_record(record, name_to_features))

        d = d.padded_batch(
          batch_size,
          padded_shapes={
            "input_ids": (tf.TensorShape([seq_length])),
            "input_mask": tf.TensorShape([seq_length]),
            "segment_ids": tf.TensorShape([seq_length]),
            "span_mask": tf.TensorShape([seq_length]),
            "gold_labels": tf.TensorShape([None])
          },
          padding_values={
              'input_ids':0,
              "input_mask":0,
              "segment_ids":0,
              'span_mask':0,
              'gold_labels':-1
          },
          drop_remainder=drop_remainder
        )
        return d

    return input_fn

def biaffine_mapping(vector_set_1,
               vector_set_2,
               output_size,
               add_bias_1=True,
               add_bias_2=True,
              initializer= None):
  """Bilinear mapping: maps two vector spaces to a third vector space.
  The input vector spaces are two 3d matrices: batch size x bucket size x values
  A typical application of the function is to compute a square matrix
  representing a dependency tree. The output is for each bucket a square
  matrix of the form [bucket size, output size, bucket size]. If the output size
  is set to 1 then results is [bucket size, 1, bucket size] equivalent to
  a square matrix where the bucket for instance represent the tokens on
  the x-axis and y-axis. In this way represent the adjacency matrix of a
  dependency graph (see https://arxiv.org/abs/1611.01734).
  Args:
     vector_set_1: vectors of space one
     vector_set_2: vectors of space two
     output_size: number of output labels (e.g. edge labels)
     add_bias_1: Whether to add a bias for input one
     add_bias_2: Whether to add a bias for input two
     initializer: Initializer for the bilinear weight map
  Returns:
    Output vector space as 4d matrix:
    batch size x bucket size x output size x bucket size
    The output could represent an unlabeled dependency tree when
    the output size is 1 or a labeled tree otherwise.
  """
  with tf.variable_scope('Bilinear'):
    # Dynamic shape info
    batch_size = tf.shape(vector_set_1)[0]
    bucket_size = tf.shape(vector_set_1)[1]

    if add_bias_1:
      vector_set_1 = tf.concat(
          [vector_set_1, tf.ones([batch_size, bucket_size, 1])], axis=2)
    if add_bias_2:
      vector_set_2 = tf.concat(
          [vector_set_2, tf.ones([batch_size, bucket_size, 1])], axis=2)

    # Static shape info
    vector_set_1_size = vector_set_1.get_shape().as_list()[-1]
    vector_set_2_size = vector_set_2.get_shape().as_list()[-1]

    if not initializer:
      initializer = tf.orthogonal_initializer()

    # Mapping matrix
    bilinear_map = tf.get_variable(
        'bilinear_map', [vector_set_1_size, output_size, vector_set_2_size],
        initializer=initializer)

    # The matrix operations and reshapings for bilinear mapping.
    # b: batch size (batch of buckets)
    # v1, v2: values (size of vectors)
    # n: tokens (size of bucket)
    # r: labels (output size), e.g. 1 if unlabeled or number of edge labels.

    # [b, n, v1] -> [b*n, v1]
    vector_set_1 = tf.reshape(vector_set_1, [-1, vector_set_1_size])

    # [v1, r, v2] -> [v1, r*v2]
    bilinear_map = tf.reshape(bilinear_map, [vector_set_1_size, -1])

    # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
    bilinear_mapping = tf.matmul(vector_set_1, bilinear_map)

    # [b*n, r*v2] -> [b, n*r, v2]
    bilinear_mapping = tf.reshape(
        bilinear_mapping,
        [batch_size, bucket_size * output_size, vector_set_2_size])

    # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
    bilinear_mapping = tf.matmul(bilinear_mapping, vector_set_2, adjoint_b=True)

    # [b, n*r, n] -> [b, n, r, n]
    bilinear_mapping = tf.reshape(
        bilinear_mapping, [batch_size, bucket_size, output_size, bucket_size])
    return bilinear_mapping

def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, span_mask, num_labels):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )

    # cls_layer = model.get_pooled_output()

    output_layer = model.get_sequence_output()

    batch_size, seq_length, hidden_size = modeling.get_shape_list(output_layer,expected_rank=3)

    if is_training:
      # cls_layer = tf.nn.dropout(cls_layer, keep_prob=0.9)
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    # Magic Number
    size = 150
    starts = tf.layers.dense(output_layer,size,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    ends = tf.layers.dense(output_layer,size,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    if is_training:
      starts = tf.nn.dropout(starts,keep_prob=0.9)
      ends = tf.nn.dropout(ends,keep_prob=0.9)

    biaffine = biaffine_mapping(
      starts,
      ends,
      num_labels,
      add_bias_1=True,
      add_bias_2=True,
      initializer=tf.zeros_initializer())
    # B,L,L,N
    candidate_ner_scores = tf.transpose(biaffine,[0,1,3,2])

    # [B,1,L] [B,L,1] -> [B,L,L]
    span_mask = tf.cast(span_mask,dtype=tf.bool)
    candidate_scores_mask = tf.logical_and(tf.expand_dims(span_mask,axis=1),tf.expand_dims(span_mask,axis=2))
    # B,L,L
    sentence_ends_leq_starts = tf.tile(
      tf.expand_dims(
        tf.logical_not(tf.sequence_mask(tf.range(seq_length),seq_length)), 
          0),
      [batch_size,1,1]
    )
    # B,L,L
    candidate_scores_mask = tf.logical_and(candidate_scores_mask,sentence_ends_leq_starts)
    # B*L*L
    flattened_candidate_scores_mask = tf.reshape(candidate_scores_mask,[-1]) 

    candidate_ner_scores = tf.boolean_mask(tf.reshape(candidate_ner_scores,[-1,num_labels]),flattened_candidate_scores_mask)
    return candidate_ner_scores

def model_fn_builder(bert_config, num_labels, init_checkpoint=None, learning_rate=None,
                     num_train_steps=None, num_warmup_steps=None,
                     use_one_hot_embeddings=False, hvd=None, amp=False):
    def model_fn(features, labels, mode, params):
        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        span_mask = features["span_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        candidate_ner_scores = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, span_mask,num_labels)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            gold_labels = features['gold_labels']
            gold_labels = tf.boolean_mask(gold_labels,tf.not_equal(gold_labels,-1))

            # 真实实体
            true_labels = tf.boolean_mask(gold_labels, tf.not_equal(gold_labels,0))
            pred_labels = tf.boolean_mask(candidate_ner_scores,tf.not_equal(gold_labels,0))
            # 只统计真实实体的准确率，否则准确率虚高
            accuracy = tf.metrics.accuracy(true_labels,tf.arg_max(pred_labels,dimension=-1))

            negative_labels = tf.boolean_mask(gold_labels, tf.equal(gold_labels,0))
            negative_pred_labels = tf.boolean_mask(candidate_ner_scores,tf.equal(gold_labels,0))
            # 只统计真实实体的准确率，否则准确率虚高
            negative_accuracy = tf.metrics.accuracy(negative_labels,tf.arg_max(negative_pred_labels,dimension=-1))
            tensor_to_log = {
                "positive_accuracy": accuracy[1] * 100,
                "negative_accuracy": negative_accuracy[1] * 100
            }
            total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gold_labels,logits=candidate_ner_scores)
            batch_size = tf.shape(input_ids)[0]
            total_loss = tf.reduce_sum(total_loss) / tf.to_float(batch_size)
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, hvd, amp)
            output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              training_hooks=[tf.train.LoggingTensorHook(tensor_to_log, every_n_iter=50)])
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"score":tf.expand_dims(candidate_ner_scores,0)} # 因为用了boolen_mask，导致原来的batch信息丢失
            )
        else:
          raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))
        return output_spec

    return model_fn

def get_pred_ner(text, span_scores, is_flat_ner=True):
    candidates = []
    for s in range(len(text)):
        for e in range(s,len(text)):
            candidates.append((s,e))
    
    top_spans = []
    for i,tp in enumerate(np.argmax(span_scores,axis=1)):
        if tp > 0:
            s,e = candidates[i]
            top_spans.append((s,e,tp,span_scores[i]))

    top_spans = sorted(top_spans, key=lambda x:x[3][x[2]], reverse=True)
    
    if not top_spans:
        # 无论如何找一个span
        # 这里是因为cluener里面基本上每句话都有实体，因此这样使用
        # 如果是真实的场景，可以去掉这部分
        tmp_span_scores = span_scores[:,1:]
        for i,tp in enumerate(np.argmax(tmp_span_scores,axis=1)):
            s,e = candidates[i]
            top_spans.append((s,e,tp+1,span_scores[i]))
        top_spans = sorted(top_spans, key=lambda x:x[3][x[2]], reverse=True)[:1]

    sent_pred_mentions = []
    for ns,ne,t,score in top_spans:
        for ts,te,_,_ in sent_pred_mentions:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                #for both nested and flat ner no clash is allowed
                break
            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                #for flat ner nested mentions are not allowed
                break
        else:
            sent_pred_mentions.append((ns,ne,t,[float(x) for x in score.flat]))
    return sent_pred_mentions


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if FLAGS.horovod:
      # NER的场景了，基本上也不需要多卡了，一个卡差不多
      import horovod.tensorflow as hvd
      hvd.init()

    processors = {
        "cluener": CLUENERProcessor,
    }
    
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
       raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tf.io.gfile.makedirs(FLAGS.output_dir)

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    master_process = True
    training_hooks = []
    global_batch_size = FLAGS.train_batch_size
    hvd_rank = 0

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if FLAGS.horovod:
      global_batch_size = FLAGS.train_batch_size * hvd.size()
      master_process = (hvd.rank() == 0)
      hvd_rank = hvd.rank()
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      if hvd.size() > 1:
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        if FLAGS.amp:
            tf.enable_resource_variables()

    run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir if master_process else None,
      session_config=config,
      log_step_count_steps=50,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
      keep_checkpoint_max=1)

    if master_process:
      tf.compat.v1.logging.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          tf.compat.v1.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      tf.compat.v1.logging.info("**************************")

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / global_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        start_index = 0
        end_index = len(train_examples)
        tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]

        if FLAGS.horovod:
          tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record{}".format(i)) for i in range(hvd.size())]
          num_examples_per_rank = len(train_examples) // hvd.size()
          remainder = len(train_examples) % hvd.size()
          if hvd.rank() < remainder:
            start_index = hvd.rank() * (num_examples_per_rank+1)
            end_index = start_index + num_examples_per_rank + 1
          else:
            start_index = hvd.rank() * num_examples_per_rank + remainder
            end_index = start_index + (num_examples_per_rank)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False,
        hvd=None if not FLAGS.horovod else hvd,
        amp=FLAGS.amp)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

    if FLAGS.do_train:
        filed_based_convert_examples_to_features(
          train_examples[start_index:end_index], label_list, FLAGS.max_seq_length, tokenizer, tmp_filenames[hvd_rank],True)
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=tmp_filenames, #train_file,
            batch_size=FLAGS.train_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            hvd=None if not FLAGS.horovod else hvd)
        
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=training_hooks)

    if FLAGS.do_predict or FLAGS.do_eval:
        if FLAGS.do_eval:
            predict_examples = processor.get_dev_examples(FLAGS.data_dir)
        else:
            predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, '../predict.tf_record')
        filed_based_convert_examples_to_features(predict_examples,label_list,FLAGS.max_seq_length,tokenizer,predict_file,False)
        tf.compat.v1.logging.info("***** Running prediction*****")
        tf.compat.v1.logging.info("  Num examples = %d", len(predict_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            batch_size=FLAGS.predict_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False
        )

        final_results = []
        for i,prediction in enumerate(tqdm(estimator.predict(input_fn=predict_input_fn,yield_single_examples=True),total=len(predict_examples))):
            text = predict_examples[i].text
            results = get_pred_ner(text,prediction['score'])
            labels = {}
            for s,e,t,score in results:
                span = text[s:e+1]
                label = label_list[t]
                item = [s,e,score] if FLAGS.output_score else [s,e]
                if label not in labels:
                    labels[label] = {span:[item]}
                else:
                    if span in labels[label]:
                        labels[label][span].append(item)
                    else:
                        labels[label][span] = [item]
            final_results.append(labels)

        result_file = os.path.join(FLAGS.output_dir,'../predict.jsonl')
        with open(result_file,'w') as f:
            for item in final_results:
                f.write(json.dumps({"label":item},ensure_ascii=False) + '\n')


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
