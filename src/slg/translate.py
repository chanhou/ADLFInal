# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import json

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import seq2seq_model

try:
  import cPickle as pickle
except:
  import pickle

import dataset_walker

#0.3
tf.app.flags.DEFINE_float("learning_rate", 0.3, "Learning rate to start with.")
#0.99
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.5,
                          "Learning rate decays by this much.")

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")

tf.app.flags.DEFINE_integer("size", 300, "Size of each model layer.")
#2
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

tf.app.flags.DEFINE_integer("in_vocab_size", 600, "query input vocabulary size.")

tf.app.flags.DEFINE_integer("out_vocab_size", 20000, "query ouput vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "./src/slg/data", "Data directory")

#tf.app.flags.DEFINE_string("train_dir", "./SLG/model", "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")

tf.app.flags.DEFINE_integer("max_training_steps", 50000, "Max training steps.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 500,
                            "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_boolean("use_LSTM", True,
                            "Set to True for using LSTM, otherwise, use GRU.")

tf.app.flags.DEFINE_boolean("decode", True,
                            "Set to True for interactive decoding.")

# tf.app.flags.DEFINE_boolean("self_test", False,
#                             "Run a self-test if this is set to True.")

tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

tf.app.flags.DEFINE_string("dataroot", './src/slg', "the data directory")

tf.app.flags.DEFINE_string("output_file", 'track.json', "SLG output json file")


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.
  Args:
    source_path: path to the files with "token-ids" for the source language.
    target_path: path to the file with "token-ids" for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
          # print("  reading data line %d" % counter)
          sys.stdout.write('\r  reading data line {}'.format(counter))
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
      print('')
  return data_set


def create_model(session, forward_only, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

  model = seq2seq_model.Seq2SeqModel(
      FLAGS.in_vocab_size,
      FLAGS.out_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      use_lstm = FLAGS.use_LSTM,
      forward_only=forward_only,
      dtype=dtype)

  if not model_path:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      #session.run(tf.global_variables_initializer())
      session.run(tf.initialize_all_variables())
  else:
    print("Reading model parameters from %s" % model_path)
    model.saver.restore(session, model_path)

  return model


def train():
  """Train an query input -> query output translation model."""
  print("[*] Training...")
  print("[*] Preparing data in %s..." % FLAGS.data_dir)
  in_train, out_train, in_dev, out_dev, _, _ = data_utils.prepare_data(
      FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)

  print('[*] data_dir')
  print(FLAGS.data_dir)
  print('[*] train_dir = ')
  print(FLAGS.train_dir)
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config = config) as sess:
    # Create model.
    print("[*] Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, forward_only=False)
    #sess.run(model.learning_rate.assign(0.0015))

    # Read data into buckets and compute their sizes.
    print ("[*] Reading training and validation data... (limit: %d)."
           % FLAGS.max_train_data_size)

    # ======= read data from source file and store as pickle file ========
    train_set = read_data(in_train, out_train, FLAGS.max_train_data_size)
    dev_set = read_data(in_dev, out_dev)

    # pickle_filename = os.path.join(FLAGS.data_dir, 'train_set.pickle')
    # with open(pickle_filename, 'wb') as fileId:
    #   pickle.dump(train_set, fileId, pickle.HIGHEST_PROTOCOL)

    # pickle_filename = os.path.join(FLAGS.data_dir, 'dev_set.pickle')
    # with open(pickle_filename, 'wb') as fileId:
    #   pickle.dump(dev_set, fileId, pickle.HIGHEST_PROTOCOL)

    # ======= read data from pickle file =================================
    # pickle_filename = os.path.join(FLAGS.data_dir, 'train_set.pickle')
    # with open(pickle_filename, 'rb') as fileId:
    #   train_set = pickle.load(fileId)

    # pickle_filename = os.path.join(FLAGS.data_dir, 'dev_set.pickle')
    # with open(pickle_filename, 'rb') as fileId:
    #   dev_set = pickle.load(fileId)
    # ====================================================================

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = model.global_step.eval()
    previous_losses = []
    best_global_step = 0
    best_eval_loss = float("inf")
    stop_count = 0
    flag = True
    while current_step <= FLAGS.max_training_steps:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      sys.stdout.write('\rtraining step {} / {} learning rate : {} stop count : {}'.
                       format(current_step, FLAGS.max_training_steps, model.learning_rate.eval(), stop_count))
      sys.stdout.flush()

      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        train_set, bucket_id)

      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, forward_only=False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d, learning rate %.4f, step-time %.2f, perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          print('[*] Decay learning rate')
          sess.run(model.learning_rate_decay_op)

        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.

        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        average_train_loss = 0.0
        average_eval_loss  = 0.0
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              train_set, bucket_id)
          _, train_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, forward_only=True)

          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, forward_only=True)

          average_train_loss += float(train_loss)
          average_eval_loss  += float(eval_loss)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()

        average_train_loss = average_train_loss / len(_buckets)
        average_eval_loss = average_eval_loss / len(_buckets)

        if average_eval_loss <= best_eval_loss:
          best_eval_loss = average_eval_loss
          best_global_step = model.global_step.eval()
          stop_count = 0
        else:
          stop_count += 1

        # if average_eval_loss < 3.95 and flag:
        #     print('[*] Change learning rate')
        #     sess.run(model.learning_rate.assign(0.0015))
        #     flag = False

        # if stop_count == 4:
        #     print('[*] Decay learning rate')
        #     sess.run(model.learning_rate_decay_op)

        # early stop
        if stop_count >= 10:
          break

        model_name = "SLG_trainLoss_%.2f_evalLoss_%.2f.ckpt" % (average_train_loss,
                                                                      average_eval_loss)
        checkpoint_path = os.path.join(FLAGS.train_dir, model_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step.eval())
    print('Finished training')
    print('Best global step is %d' % best_global_step)

def generateSplitTag(log_utter):
  string = ""
  split = ""
  element = []
  for dictionary in log_utter['speech_act']:
      elementSplit = []
      for sub in dictionary[u'attributes']:
          element.append(sub)
          elementSplit.append(sub)
      element.append(dictionary[u'act'])
      elementSplit.append(dictionary[u'act'])
      intentSplit = "_".join(elementSplit)
      split += " "
      split += intentSplit
  intent = "_".join(element)
  string += intent
  string += " "
  split += " "
  tag = ""
  mentionList = []
  #print log_utter['semantic_tags']
  # tag
  for dictionary in log_utter['semantic_tags']:
      element = []
      element.append(dictionary[u'main'])
      mentionList.append(dictionary[u'mention'])
      subdict = dictionary[u'attributes']
      for key in subdict:
          element.append(subdict[key])
      tag += "_".join(element)
      tag += " "
  split += tag
  string += tag
  tagList = []
  if len(tag.strip()) > 0:
      tagList = tag.strip().split(" ")
  return split.strip(), tagList, mentionList

def replaceResult(string, tagList, mentionList):
  for i in range(len(tagList)):
    string = string.replace(tagList[i].encode("utf8"), mentionList[i].encode("utf8"))
  return string

def decodeOnTest(model_path):
  print('[*] Test on test_slg set')
  print('[*] output file = %s' % FLAGS.output_file)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config = config) as sess:
    # Create model and load parameters.
    model = create_model(sess, forward_only=True, model_path=model_path)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    in_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.in" % FLAGS.in_vocab_size)
    out_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.out" % FLAGS.out_vocab_size)
    in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)
    _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)

    print('[*] Start testing')
    output = {'sessions': []}

    print('Loading TOURIST testing instances ...')
    dataroot = os.path.join(FLAGS.dataroot, 'test_slg')
    testset = dataset_walker.dataset_walker('dstc5_test_slg_tourist', dataroot=dataroot,
                                            labels=False, translations=True, task='SLG',
                                            roletype='tourist')
    count = 0
    for call in testset:
      this_session = {"session_id": call.log["session_id"], "utterances": []}

      for (log_utter, _, _) in call:
        if log_utter['speaker'].lower() == 'tourist':
          count += 1
          sys.stdout.write('\r  reading instance {}'.format(count))
          sys.stdout.flush()

          #instance = {'semantic_tags': log_utter['semantic_tags'], 'speech_act': log_utter['speech_act']}
          splitTag, tagList, mentionList = generateSplitTag(log_utter)
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(splitTag), in_vocab)
          # Which bucket does it belong to?
          bucket_id = len(_buckets) - 1
          for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
              bucket_id = i
              break
          else:
            logging.warning("Sentence truncated: %s", sentence)

          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, forward_only=True)
          # This is a greedy decoder - outputs are just argmaxes of output_logits.
          outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
          # If there is an EOS symbol in outputs, cut them at that point.
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          outputs = [rev_out_vocab[idx] for idx in outputs]
          outputs = "".join(outputs)
          #print('\n'+'='*50)
          #print(outputs)
          outputs = replaceResult(outputs, tagList, mentionList)
          #print(outputs)

          slg_result = {'utter_index': log_utter['utter_index'], 'generated': outputs}
          this_session['utterances'].append(slg_result)
      output['sessions'].append(this_session)

    print('')
    print('[*] Loading GUIDE testing instances ...')
    dataroot = os.path.join(FLAGS.dataroot, 'test_slg')
    testset = dataset_walker.dataset_walker('dstc5_test_slg_guide', dataroot=dataroot,
                                            labels=False, translations=True, task='SLG',
                                            roletype='guide')
    count = 0
    for call in testset:
      this_session = {"session_id": call.log["session_id"], "utterances": []}

      for (log_utter, _, _) in call:
        if log_utter['speaker'].lower() == 'guide':
          count += 1
          sys.stdout.write('\r  reading instance {}'.format(count))
          sys.stdout.flush()

          #instance = {'semantic_tags': log_utter['semantic_tags'], 'speech_act': log_utter['speech_act']}
          splitTag, tagList, mentionList = generateSplitTag(log_utter)
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(splitTag), in_vocab)
          # Which bucket does it belong to?
          bucket_id = len(_buckets) - 1
          for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
              bucket_id = i
              break
          else:
            logging.warning("Sentence truncated: %s", sentence)

          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, forward_only=True)
          # This is a greedy decoder - outputs are just argmaxes of output_logits.
          outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
          # If there is an EOS symbol in outputs, cut them at that point.
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          outputs = [rev_out_vocab[idx] for idx in outputs]
          outputs = "".join(outputs)
          #print('\n'+'='*50)
          #print(outputs)
          outputs = replaceResult(outputs, tagList, mentionList)
          #print(outputs)

          slg_result = {'utter_index': log_utter['utter_index'], 'generated': outputs}
          this_session['utterances'].append(slg_result)
      output['sessions'].append(this_session)

    sorted_sessions = sorted(output['sessions'], key = lambda x : x["session_id"])
    output['sessions'] = sorted_sessions

    print('[*] Done')

    print('[*] Outputing json file...')
    with open(FLAGS.output_file, "wb") as of:
        json.dump(output, of, indent=4)

    print('[*] Finish')

def main(_):
  if FLAGS.decode:
      decodeOnTest(model_path="./src/slg/best_model.ckpt")
  else:
    train()

if __name__ == "__main__":
  tf.app.run()