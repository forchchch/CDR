from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import os
import shutil
import sys
import time
import datetime
import json
import multiprocessing
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from scipy import sparse
from copy import copy, deepcopy
from tensorflow.contrib.distributions import RelaxedOneHotCategorical
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
from utils import load_data_dfn, get_model, to_ctr_labels, to_dtr_labels
from constants import FEEDBACK_MODELS, CTR_MODELS
import unclick_model_CL_tune as unclick_model_CL
#import unclick_model as unclick_model_CL


cores = multiprocessing.cpu_count() // 2

ARG = argparse.ArgumentParser()
ARG.add_argument('--load_json', 
                 help='Load settings from file in json format. Command line options override values in file.')
ARG.add_argument('--data', type=str, required=True,
                 help='dataset name: ./data/xxx/prep')
ARG.add_argument('--model', type=str, default='DFN',
                 help='DFN, DFR or future work.')
ARG.add_argument('--mode', type=str, default='trn',
                 help='trn or tst.')
ARG.add_argument('--logdir', type=str, default='./runs/')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed. Ignored if < 0.')
ARG.add_argument('--epoch', type=int, default=2,
                 help='Number of training epochs.')
ARG.add_argument('--batch', type=int, default=256,
                 help='Training batch size. Default: 256 samples. Each sample includes (user, candidate, c/u/d histories, label).')
ARG.add_argument('--test_batch', type=int, default=64,
                 help='Validation/Testing batch size. Default: 64 samples are sent for prediction.')
ARG.add_argument('--verbose', type=int, default=1,
                 help='Per _____ epochs we run evaluation on valid set.')
ARG.add_argument('--batch_verbose', type=int, default=100,
                 help='Per _____ batches we print the training loss.')
parser = ARG
ARG = parser.parse_args()
if ARG.load_json:
    with open(ARG.load_json, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        ARG = parser.parse_args(namespace=t_args)
if ARG.seed < 0:
    ARG.seed = int(time.time())
batch_size = ARG.batch
batch_size_vad = batch_size_test = ARG.test_batch
if not os.path.exists(ARG.logdir):
    os.mkdir(ARG.logdir)

def update_log_dir():
    global LOG_DIR
    LOG_DIR = ARG.data.replace('/', '_')+"-"+str(int(time.time()))+'tester'
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_DIR = os.path.join(ARG.logdir, LOG_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)


def set_rng_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def get_item_features(path):
    a = 0
    items = []
    cates = []
    with open(path+"/itemfeats.txt",'r') as f1:
        for line in f1:
            if a>0:
                info = line.strip().split()
                items.append(int(info[0]))
                cates.append(int(info[1]))
            a = a + 1
    return items,cates
                
def get_feed_dict(samples, model, model_name, st_idx, end_idx, stats, is_train, to_float=False):

    def get_index(bs, indicies_batch):
        index = np.zeros((bs, 2), dtype='int32')
        index[:, 0] = np.arange(bs)
        index[:, 1] = indicies_batch
        return index

    assert model_name in FEEDBACK_MODELS+CTR_MODELS
    if model_name == 'DFN':
        users, candidates, weights, labels, click_histories, unclick_histories, dislike_histories, \
            click_positions, unclick_positions, dislike_positions, click_hist_lens, unclick_hist_lens, dislike_hist_lens = samples
    elif model_name == 'DFR':
        users, candidates, weights, labels, click_histories, unclick_histories, dislike_histories, \
            click_positions, unclick_positions, dislike_positions, click_hist_lens, unclick_hist_lens, dislike_hist_lens, \
            pos_indices, neg_indices = samples[:15]
        if len(samples) == 16:
            whether_sures = samples[15]
    elif model_name in CTR_MODELS:  # for CTR_MODELS ('DeepFM', 'AutoInt')
        feat_indices, labels = samples  

    samp_idx = None
    bs = -1
    if end_idx is not None:
        samp_idx = np.arange(st_idx, end_idx)
        bs = end_idx - st_idx
    else:
        samp_idx = st_idx
        bs = len(samp_idx)
    assert bs != -1

    user_info_cate, number_each_user_cate, item_info_cate, number_each_item_cate = stats
    feed_dict = {}
    if model_name in FEEDBACK_MODELS:
        for cate in range(user_info_cate):
            feed_dict[model.group_feature["user_feature_"+str(cate)]] = users[samp_idx, cate]
        for cate in range(item_info_cate):
            feed_dict[model.group_feature["candidate_item_feature_"+str(cate)]] = candidates[samp_idx, cate]
        for cate in range(item_info_cate):
            feed_dict[model.group_feature["clicked_item_feature_"+str(cate)]] = click_histories [ samp_idx, :, cate]
            feed_dict[model.group_feature["unclick_item_feature_"+str(cate)]] = unclick_histories[samp_idx, :, cate]
            feed_dict[model.group_feature["dislike_item_feature_"+str(cate)]] = dislike_histories[samp_idx, :, cate]
        feed_dict[model.group_feature["clicked_item_feature_position"]] = click_positions [ samp_idx, :]
        feed_dict[model.group_feature["unclick_item_feature_position"]] = unclick_positions[samp_idx, :]
        feed_dict[model.group_feature["dislike_item_feature_position"]] = dislike_positions[samp_idx, :]
        feed_dict[model.group_feature["clicked_histLen"]] = click_hist_lens [ samp_idx]
        feed_dict[model.group_feature["unclick_histLen"]] = unclick_hist_lens[samp_idx]
        feed_dict[model.group_feature["dislike_histLen"]] = dislike_hist_lens[samp_idx]
        feed_dict[model.weights] = weights[samp_idx]
        feed_dict[model.labels]  = labels [samp_idx]
        if hasattr(model, 'train_phase'):
            if hasattr(model, 'dropout_keep_fm'):  # DeepFM (DFN ver.)
                feed_dict[model.dropout_keep_fm] = model.dropout_fm
                feed_dict[model.dropout_keep_deep] = model.dropout_deep
            if hasattr(model, 'dropout_keep_prob'):  # AutoInt (DFN ver.)
                feed_dict[model.dropout_keep_prob] = model.drop_keep_prob
            feed_dict[model.train_phase] = is_train
            feed_dict[model.labels] = labels[st_idx:end_idx, np.newaxis]
        if model_name == 'DFR':
            feed_dict[model.group_feature["pos_index"]] = get_index(bs, pos_indices[samp_idx])
            feed_dict[model.group_feature["neg_index"]] = get_index(bs, neg_indices[samp_idx])
            feed_dict[model.group_feature["c_index"]] = get_index(bs, click_hist_lens[samp_idx] - 1)
            feed_dict[model.group_feature["d_index"]] = get_index(bs, dislike_hist_lens[samp_idx] - 1)
            if hasattr(model, 'dropout_rate'):
                if is_train:
                    feed_dict[model.dropout_rate] = ARG.dropout_rate
                else:
                    feed_dict[model.dropout_rate] = 1.
    elif model_name in CTR_MODELS:
        feed_dict[model.feat_index] = feat_indices[samp_idx]
        feed_dict[model.label] = labels[samp_idx]
        if model_name == 'DeepFM':
            feed_dict[model.dropout_keep_fm] = model.dropout_fm
            feed_dict[model.dropout_keep_deep] = model.dropout_deep
        elif model_name == 'AutoInt':
            feed_dict[model.dropout_keep_prob] = model.drop_keep_prob
        feed_dict[model.train_phase] = is_train

    if to_float:
        pass
        for key in feed_dict.keys():
            # feed_dict[key] = tf.cast(feed_dict[key], dtype=tf.float32)
            try:
                feed_dict[key] = feed_dict[key].astype(np.float32)
            except:
                pass

    return feed_dict 


def main_trn(n_items, train_samples, valid_samples, stats, config):
    '''
    [paras]
        train_samples: a list of 13 fields (users, cadidates, weights, ...) // see get_feed_dict()
                       each field is of shape (n_train_samples, ...).
        valid_samples: similar to train_samples
        stats: some statistics about user/item features, see normal_DFN_model.py for more details.
    '''
    global LOG_DIR
    set_rng_seed(ARG.seed)
    n_train_samples = len(train_samples[0])
    n_train_batches_per_epoch = n_train_samples // batch_size
    n_valid_samples = len(valid_samples[0])

    def get_valid_labels(valid_samples):
        vl = []
        if ARG.model in FEEDBACK_MODELS:
            vl = valid_samples[3]
        elif ARG.model in CTR_MODELS:
            vl = valid_samples[1]
        else:
            print("Oh no!")
        return vl

    valid_labels = get_valid_labels(valid_samples)
    valid_labels = to_ctr_labels(valid_labels)
    print("valid_labels[:20]:", valid_labels[:20])

    with tf.variable_scope("predict", reuse=None):
        model_predict = unclick_model_CL.Add_unclick(config)

    train_op_predict = tf.train.AdagradOptimizer(ARG.learning_rate, 1e-6).minimize(model_predict.loss)

    def evaluate_on_valid(best_auc):
        pred_vals = []
        for bnum, st_idx in enumerate(range(0, n_valid_samples, batch_size_vad)):
            end_idx = min(st_idx + batch_size_vad, n_valid_samples)
            pred_val = sess.run(model_predict.results, feed_dict=get_feed_dict(valid_samples, model_predict, ARG.model, st_idx, end_idx, stats, is_train=False))
            pred_vals.extend(list(pred_val))
        try:
            auc = roc_auc_score(valid_labels, pred_vals)
        except:
            tmp = pred_vals
            print("pred_val:")
            print(tmp)
            exit(233)
        if auc > best_auc:
            model_predict.saver.save(sess, '{}/chkpt'.format(LOG_DIR))
            best_auc = auc
        print("[%s][valid] auc: %.6f , best_auc: %.6f " % (epoch, auc, best_auc))
        return best_auc

    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth=True
    with tf.Session(config=config1) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        best_auc = -np.inf
        training_start_time = time.time()
        tao = 10.0
        #start_cl = int(0.25*int(ARG.epoch*n_train_samples/batch_size))
        start_cl = int(0.0*int(ARG.epoch*n_train_samples/batch_size))
        time_factor = 1.0
        all_time = int( ((n_train_samples*(ARG.epoch)/batch_size) - start_cl)/ARG.batch_verbose)
        for epoch in range(1, ARG.epoch+1):
            total_loss = 0.0
            for field in train_samples:
                np.random.seed(ARG.seed)  # we have to reset random seed to ensure the same shuffle across fields
                np.random.shuffle(field)
            if ARG.model in FEEDBACK_MODELS: 
                train_labels = train_samples[3]
            elif ARG.model in CTR_MODELS:
                train_labels = train_samples[1]
            train_samples[3] = to_ctr_labels(train_labels)
            train_labels = train_samples[3]

            for bnum, st_idx in enumerate(range(0, n_train_samples, batch_size)):
                end_idx = min(st_idx + batch_size, n_train_samples)
                if len(train_samples) == 16: 
                    whether_sures = train_samples[15][st_idx:end_idx]
                # #######################algorithm start
                # print("algorithm start")
                train_predict_dict = get_feed_dict(train_samples, model_predict, ARG.model, st_idx, end_idx, stats, is_train=False)
                if  (bnum + 1)>start_cl:
                    init_weights = train_predict_dict[model_predict.weights]

                    results_predict,loss_old = sess.run([model_predict.soft_results,model_predict.loss_not_average],feed_dict=train_predict_dict)
                    ##########mean-performance-based curriculum
                    center = 0.0
                    given_label = train_predict_dict[model_predict.labels]
                    diff = np.abs(given_label-results_predict)
                    weights = np.exp(-tao*(diff-center)*(diff-center))
                    train_predict_dict[model_predict.weights] = weights/sum(weights)
                    loss_to_print,_ = sess.run([model_predict.loss,train_op_predict],feed_dict=train_predict_dict)  
                    if (bnum + 1) % (ARG.batch_verbose*2) == 0:
                        time_factor = time_factor+1
                        tao = tao*0.4


                else:
                    init_weights = train_predict_dict[model_predict.weights]
                    train_predict_dict[model_predict.weights] = init_weights/sum(init_weights)
                    loss_to_print,_ = sess.run([model_predict.loss,train_op_predict],feed_dict=train_predict_dict)

                total_loss += loss_to_print
                if (bnum + 1) % ARG.batch_verbose == 0:
                    print("[epoch %d : loss : %.6f, time: %.3fs]" % (epoch, total_loss / (bnum + 1),
                                                                  time.time()-training_start_time))
                    print("finish",bnum*batch_size/n_train_samples,"in this epoch")

            if epoch % ARG.verbose == 0:
                best_auc = evaluate_on_valid(best_auc)

        print("training over")
        print("="*100)
        return best_auc


def main_tst(n_items, test_samples, stats):
    global LOG_DIR
    pool = multiprocessing.Pool(cores)
    set_rng_seed(ARG.seed)

    config = ARG
    with tf.variable_scope("predict", reuse=None):
        model = unclick_model_CL.Add_unclick(config)

    n_test_samples = len(test_samples[0])
    if ARG.model in FEEDBACK_MODELS:
        test_labels = test_samples[3]
    elif ARG.model in CTR_MODELS:
        test_labels = test_samples[1]
    test_labels = to_ctr_labels(test_labels)

    with tf.Session() as sess:
        model.saver.restore(sess, '{}/chkpt'.format(LOG_DIR))
        pred_vals = []
        for bnum, st_idx in enumerate(range(0, n_test_samples, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, n_test_samples)
            pred_val = sess.run(model.results, feed_dict=get_feed_dict(test_samples, model, ARG.model, st_idx, end_idx, stats, is_train=False))
            pred_vals.extend(list(pred_val))
        test_auc = roc_auc_score(test_labels, pred_vals) 
        print("Test AUC=%.5f" % (test_auc))

    pool.close()
    return test_auc


def analyze_distribution(samples, mode):
    print("="*20+" samples of "+mode+" "+"="*20)
    from collections import Counter
    user_c = Counter()
    item_c = Counter()
    user_label_c = Counter()
    item_label_c = Counter()
    click_user_c = Counter()
    click_item_c = Counter()
    users, candis, labels = samples[0], samples[1], samples[3]
    for i in range(len(users)):
        user_id = users[i][-1]
        candi_id = candis[i][-1]
        label = labels[i]
        user_c.update([user_id])
        item_c.update([candi_id])
        user_label_c.update([(user_id, label)])
        item_label_c.update([(candi_id, label)])
        if label == 1:
            click_user_c.update([user_id])
            click_item_c.update([candi_id])
    print("num samples:", len(samples[0]))
    print("num users:", len(user_c))
    print("most common:", user_c.most_common(10))
    print("num items:", len(item_c))
    print("most common:", item_c.most_common(10))
    print("num user_labels:", len(user_label_c))
    print("most common:", user_label_c.most_common(10))
    print("num item_labels:", len(item_label_c))
    print("most common:", item_label_c.most_common(10))
    print("num click_users:", len(click_user_c))
    print("most common:", click_user_c.most_common(10))
    print("num click_items:", len(click_item_c))
    print("most common:", click_item_c.most_common(10))

def main():
    update_log_dir()
    if hasattr(ARG, 'lambda_c'):
        weight_dict = {1: ARG.lambda_c, 0: ARG.lambda_u, -1: ARG.lambda_d}
    else:
        weight_dict = {1: 1., 0: 1., -1: 1.}
    if not hasattr(ARG, 'max_hist_len'):
        ARG.max_hist_len = None
    # return

    (user_info_cate, number_each_user_cate, item_info_cate, number_each_item_cate, 
     position_len, train_samples, valid_samples, test_samples, n_users, n_items, user_cate_dicts, item_cate_dicts) = load_data_dfn(ARG.data, weight_dict, ARG.model, ARG.max_hist_len, ARG.seed)
    if ANALYZE:
        analyze_distribution(train_samples, 'train')
        analyze_distribution(valid_samples, 'valid')
        analyze_distribution(test_samples, 'test')
        exit(25)


    best_vad_auc, test_auc = 0, 0
    stats = user_info_cate, number_each_user_cate, item_info_cate, number_each_item_cate 
    ARG.user_info_cate = user_info_cate
    ARG.number_each_user_cate = number_each_user_cate
    ARG.item_info_cate = item_info_cate
    ARG.number_each_item_cate = number_each_item_cate
    ARG.position_len = position_len
    print("ARG:", ARG)

    if ARG.mode in ('trn',):
        tf.reset_default_graph()
        best_vad_auc = main_trn(n_items, train_samples, valid_samples, stats, config=ARG)
    if ARG.mode in ('trn', 'tst'):
        tf.reset_default_graph()
        test_auc = main_tst(n_items, test_samples, stats)
        print('(%.5f, %.5f)' % (best_vad_auc, test_auc))
        print(datetime.datetime.now())
    return best_vad_auc, test_auc


if __name__ == '__main__':

    ANALYZE = False   # analyze the distribution of the train/valid/test samples
    main()
    exit(33)
