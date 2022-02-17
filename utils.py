# -*- coding: UTF-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
#import bottleneck as bn
import collections
import copy
import cPickle as pickle
import gc
import itertools
import json
import numpy as np
import os
import pandas as pd
import random as rd
import tensorflow as tf
import time
from scipy import sparse
DEBUG = False   # False: full dataset,   True: 10000 samples
INCLUDE_ID_EMB = True   # False: do not use user/item id embedding in the features
USE_ONLY_ID_EMB = False   # True: only use user/item id embedding in the features, and drop others (e.g., 'age')
WHETHER_SURE = False#True  # True: generate whether_sure data in DFR
from constants import FEEDBACK_MODELS, CTR_MODELS

print('DEBUG: ', DEBUG)
print('INCLUDE_ID_EMB: ', INCLUDE_ID_EMB)
print('USE_ONLY_ID_EMB: ', USE_ONLY_ID_EMB)
print('WHETHER_SURE:', WHETHER_SURE)
# print('DISLIKE_PREDICTION: ', DISLIKE_PREDICTION)

def get_model(user_info_cate, number_each_user_cate, item_info_cate, number_each_item_cate, position_len, ARG):
    assert ARG.model in FEEDBACK_MODELS+CTR_MODELS
    if ARG.model == 'DFN':
        model = DFN(user_info_cate, number_each_user_cate, item_info_cate, number_each_item_cate, position_len, 
                    batch_size=ARG.batch, embed_dim=ARG.embed_dim, hist_size=ARG.max_hist_len, args=ARG)
    elif ARG.model == 'DFR':
        # model = DFR(user_info_cate, number_each_user_cate, item_info_cate, number_each_item_cate, position_len,
        #             batch_size=ARG.batch, embed_dim=ARG.embed_dim, hist_size=ARG.max_hist_len, layers=ARG.layers, K=ARG.K,
        #             alpha=ARG.alpha, beta=ARG.beta, gamma=ARG.gamma, zeta=ARG.zeta)
        model = DFR(ARG)

    elif ARG.model in CTR_MODELS:
        field_size = user_info_cate + item_info_cate
        feature_size = np.sum(list(number_each_user_cate)) + user_info_cate  # + user_info_cate: leave sizes for the 0 indices of each field
        feature_size += np.sum(list(number_each_item_cate)) + item_info_cate        
        if ARG.model == 'DeepFM':
            model = DeepFM(feature_size, field_size, embedding_size=ARG.embed_dim, random_seed=ARG.seed)
        elif ARG.model == 'AutoInt':
            model = AutoInt(ARG, feature_size, field_size)
    return model

def to_ctr_labels(labels):
    ret = []
    for l in labels:
        if l > 0:
            ret.append(1)
        else:
            ret.append(0)
    return np.array(ret)

def to_dtr_labels(labels):
    ret = []
    for l in labels:
        if l < 0:
            ret.append(1)
        else:
            ret.append(0)
    return np.array(ret)


def read_samples(data_file, final_file, model, userfeat_dict, itemfeat_dict, max_hist_len=None, weight_dict=None, phase='train', experiment='CTR'):
    def raw_label_to_weight_and_label(raw_label, weight_dict=None):
        '''
        [paras]
            raw_label: can be click_num (>0, for click samples), 0 (for unclick samples), or -1 (for dislike samples).
            weights: a dict of sample loss weights. {1: lambda_c, 0: lambda_u, -1: lambda_d}
        [return]
            label: 0 (for unclick/dislike), or 1 (for click)
            weight: sample weight
        '''
        label = int(raw_label)
        if label > 0:
            label = 1
        if weight_dict is not None:
            weight = weight_dict[label]
        else:
            weight = 1.
        '''
        if (phase != 'train') and (experiment == 'DTR'):  # For valid and test set of dislike prediction, label is different: click=0, unclick=0, dislike=1
            if label > 0:
                label = 0
            if label < 0:
                label = 1
        else:
            if label < 0:
                label = 0
        '''
        return weight, label

    def raw_label_to_whether_sure(raw_label):
        label = int(raw_label)
        whether_sure = 0.0
        if label < 0:
            whether_sure = 1.0
        if label > 0:
            whether_sure = 0.5
        # if label > 0 or label < 0:
        #     whether_sure = 1
        return whether_sure

    def hist_str_to_list(hist_str, max_hist_len=None):
        if len(hist_str) > 0:
            hist_real = list(map(int, hist_str.split(',')))
            if max_hist_len is None:
                return hist_real
            hist = hist_real + [0] * (max_hist_len - len(hist_real))
        else:
            if max_hist_len is None:
                return []
            hist = [0] * max_hist_len
        return hist

    # if os.path.exists(final_file):
    #     print("Loading data from %s ..." % final_file)
    #     start_time = time.time()
    #     with open(final_file, 'rb') as f:
    #         gc.disable()
    #         samples = pickle.load(f)
    #         gc.enable()
    #         print("Loaded samples, time: %.3f min" % (time.time() - start_time) / 60)
    #         return samples
    # else:
    assert model in FEEDBACK_MODELS+CTR_MODELS
    print("Reading data from %s ..." % data_file)

    if model in FEEDBACK_MODELS:
        users, candidates, weights, labels = [], [], [], []
        click_histories, unclick_histories, dislike_histories = [], [], []
        click_positions, unclick_positions, dislike_positions = [], [], []
        click_hist_lens, unclick_hist_lens, dislike_hist_lens = [], [], []  
        pos_indices, neg_indices,  = [], []   # for DFR
        whether_sures = []
    elif model in CTR_MODELS:
        feat_indices, labels = [], []


    cnt = 0
    start_time = time.time()
    with open(data_file, 'r') as f:
        cur_user = -1

        for line in f:
            cnt += 1
            if DEBUG:
                if cnt < 10000:
                    continue

            user_id, cand_id, clk_hist, unc_hist, dis_hist, clk_posi, unc_posi, dis_posi, raw_label = line.strip().split('\t')
            user_id = int(user_id)
            cand_id = int(cand_id)  # item candidate
            weight, label = raw_label_to_weight_and_label(raw_label, weight_dict)

            if model in FEEDBACK_MODELS:
                
                clk_hist_len = len(clk_posi.split(','))
                unc_hist_len = len(unc_posi.split(','))
                dis_hist_len = len(dis_posi.split(','))
                
                clk_hist = hist_str_to_list(clk_hist, max_hist_len)  # click history, a list of item_ids
                unc_hist = hist_str_to_list(unc_hist, max_hist_len)  # unclick history
                dis_hist = hist_str_to_list(dis_hist, max_hist_len)  # dislike history
                clk_posi = hist_str_to_list(clk_posi, max_hist_len)  # click position history, a list of "delta-time bin" positions for Transformers
                unc_posi = hist_str_to_list(unc_posi, max_hist_len)  # unclick position history
                dis_posi = hist_str_to_list(dis_posi, max_hist_len)  # dislike position history

                # clk_hist_len = len(clk_posi)
                # unc_hist_len = len(unc_posi)
                # dis_hist_len = len(dis_posi)

                users.append(userfeat_dict[user_id].values())
                candidates.append(itemfeat_dict[cand_id].values())
                weights.append(weight)
                labels.append(label)
                click_histories.append(  list(map(lambda item_id: itemfeat_dict[item_id].values(), clk_hist)))
                unclick_histories.append(list(map(lambda item_id: itemfeat_dict[item_id].values(), unc_hist)))
                dislike_histories.append(list(map(lambda item_id: itemfeat_dict[item_id].values(), dis_hist)))
                click_positions.append(  clk_posi)
                unclick_positions.append(unc_posi)
                dislike_positions.append(dis_posi)
                click_hist_lens.append(  clk_hist_len)
                unclick_hist_lens.append(unc_hist_len)
                dislike_hist_lens.append(dis_hist_len)
                if model == 'DFR':
                    if clk_hist_len > 0:
                        pos_indices.append(np.random.choice(clk_hist_len))
                    else:
                        pos_indices.append(0)
                    if dis_hist_len > 0:
                        neg_indices.append(np.random.choice(dis_hist_len))
                    else:
                        neg_indices.append(0)
                    if WHETHER_SURE:
                        whether_sure = raw_label_to_whether_sure(raw_label)
                        whether_sures.append(whether_sure)

            elif model in CTR_MODELS:
                if cur_user != user_id:  # collect the samples in the first frame of each user's interaction history
                    clk_hist = hist_str_to_list(clk_hist)
                    unc_hist = hist_str_to_list(unc_hist)
                    dis_hist = hist_str_to_list(dis_hist)
                    for pos_item in clk_hist:
                        feat_indices.append(userfeat_dict[user_id].values() + itemfeat_dict[pos_item].values())
                        labels.append(1)
                    for neg_item in (unc_hist + dis_hist):
                        feat_indices.append(userfeat_dict[user_id].values() + itemfeat_dict[neg_item].values())
                        labels.append(0)
                feat_indices.append(userfeat_dict[user_id].values() + itemfeat_dict[cand_id].values())
                labels.append(label)
                cur_user = user_id

                # print("origin data")
                # print(line)
                # print("feat_indices")
                # for i, fi in enumerate(feat_indices):
                #     print(fi)
                #     print("label=%d" % labels[i])
                # exit(2)

            if cnt % 100000 == 0:
                print("%s samples, time: %.3f min" % (cnt, (time.time() - start_time) / 60))
            if DEBUG:
                if cnt > 12000:
                    break    # TODO: this is only for debugging.

    if model in FEEDBACK_MODELS:
        samples = [np.asarray(users), np.asarray(candidates), np.asarray(weights), np.asarray(labels), \
                  np.asarray(click_histories), np.asarray(unclick_histories), np.asarray(dislike_histories), \
                  np.asarray(click_positions), np.asarray(unclick_positions), np.asarray(dislike_positions), \
                  np.asarray(click_hist_lens), np.asarray(unclick_hist_lens), np.asarray(dislike_hist_lens)]
        # print("click_histories.shape:", np.asarray(click_histories).shape)
        if model == 'DFR':
            samples.append(np.asarray(pos_indices, dtype='int32'))
            samples.append(np.asarray(neg_indices, dtype='int32'))
            if WHETHER_SURE:
                samples.append(np.asarray(whether_sures, dtype='float32'))
    elif model in CTR_MODELS:
        labels = np.asarray(labels)
        samples = [np.asarray(feat_indices), labels[:, np.newaxis]]
    # with open(final_file, 'wb') as f:
    #     pickle.dump(samples, f)

    return samples


def read_feature_dict(data_file, index):
    '''
    data_file: userfeats.txt or itemfeats.txt
    index: 'user' or 'item'
    return: e.g. of user:  {user_id: {'device': xx, 'age': xxx, ...} }
    '''
    df = pd.read_csv(data_file, sep='\t')
    if index == 'item':
        df.drop(['doc_tags', 'doc_title'], axis=1, inplace=True, errors='ignore')  # TODO: support the hash ids for text features in prep.py
    if INCLUDE_ID_EMB:
        df[str('zzz_'+index+'_id')] = df[index]   # Add user_id or item_id in the last column of features
    if USE_ONLY_ID_EMB:
        if index == 'item':
            df.drop(['category', 'biz_type', 'subcat'], axis=1, inplace=True, errors='ignore')
        elif index == 'user':
            df.drop(['age', 'gender', 'device', 'elite_level', 'user_identity'], axis=1, inplace=True)
    # if INCLUDE_ID_EMB:
    #     df[str('zzz_'+index+'_id')] = df[index]   # Add user_id or item_id in the last column of features
    # id_values = df[index]
    # df.insert(loc=df.shape[1], column=str(index+'_id'), value=id_values)
    feat_dict = df.set_index(index).to_dict('index')
    feat_dict[0] = copy.deepcopy(feat_dict[1])
    for feat in feat_dict[1].keys():
        feat_dict[0][feat] = 0
    for i in feat_dict.keys():
        feat_dict[i] = collections.OrderedDict(sorted(feat_dict[i].items()))
    return feat_dict


def load_data_dfn(data_dir, weight_dict, model, max_hist_len=None, seed=98765, experiment='CTR'):
    '''
    load data for DFN.
    '''
    train_file = os.path.join(data_dir, 'train.txt')
    valid_file = os.path.join(data_dir, 'valid.txt')
    test_file  = os.path.join(data_dir, 'test.txt' )
    userfeat_file = os.path.join(data_dir, 'userfeats.txt')
    itemfeat_file = os.path.join(data_dir, 'itemfeats.txt')
    userfeat_stat_file = os.path.join(data_dir, 'userfeat_stat.json')
    itemfeat_stat_file = os.path.join(data_dir, 'itemfeat_stat.json')
    user_catedicts_file = os.path.join(data_dir, 'user_cate_dicts.json')
    item_catedicts_file = os.path.join(data_dir, 'item_cate_dicts.json')
    train_final_file = os.path.join(data_dir, 'train.pkl')  # deprecated. pickles are large and slow than texts.
    valid_final_file = os.path.join(data_dir, 'valid.pkl')
    test_final_file  = os.path.join(data_dir, 'test.pkl' )

    # read feature statistics dicts
    userfeat_stat_dict = json.load(open(userfeat_stat_file))
    itemfeat_stat_dict = json.load(open(itemfeat_stat_file))
    userfeat_stat_dict = collections.OrderedDict(sorted(userfeat_stat_dict.items()))
    itemfeat_stat_dict = collections.OrderedDict(sorted(itemfeat_stat_dict.items()))

    user_cate_dicts = json.load(open(user_catedicts_file))
    item_cate_dicts = json.load(open(item_catedicts_file))

    position_len = 12  # TODO: no hard coding

    # read feature dicts
    userfeat_dict = read_feature_dict(userfeat_file, 'user')
    itemfeat_dict = read_feature_dict(itemfeat_file, 'item')
    n_users = len(userfeat_dict) + 1
    n_items = len(itemfeat_dict) + 1
    print("n_users: %s, n_items: %s" % (n_users, n_items))
    # print(userfeat_dict[1001].values())
    # print(itemfeat_dict[1001].values())
    # exit(22)
    if USE_ONLY_ID_EMB:
        user_info_cate = 1
        number_each_user_cate = [n_users]
        item_info_cate = 1
        number_each_item_cate = [n_items]
    elif INCLUDE_ID_EMB:
        user_info_cate = len(userfeat_stat_dict) + 1
        number_each_user_cate = list(userfeat_stat_dict.values()) + [n_users]
        item_info_cate = len(itemfeat_stat_dict) + 1
        number_each_item_cate = list(itemfeat_stat_dict.values()) + [n_items]
    else:
        user_info_cate = len(userfeat_stat_dict)
        number_each_user_cate = list(userfeat_stat_dict.values())
        item_info_cate = len(itemfeat_stat_dict)
        number_each_item_cate = list(itemfeat_stat_dict.values())


    # read raw samples, convert to correct dtype
    # train_samples = []
    # valid_samples = []
    train_samples = read_samples(train_file, train_final_file, model, userfeat_dict, itemfeat_dict, max_hist_len, weight_dict, phase='train', experiment=experiment)
    valid_samples = read_samples(valid_file, valid_final_file, model, userfeat_dict, itemfeat_dict, max_hist_len, phase='valid', experiment=experiment)
    test_samples  = read_samples(test_file,  test_final_file,  model, userfeat_dict, itemfeat_dict, max_hist_len, phase='test', experiment=experiment)
    # valid_samples = []
    # test_samples = []

    return user_info_cate, number_each_user_cate, item_info_cate, number_each_item_cate, \
           position_len, train_samples, valid_samples, test_samples, n_users, n_items, user_cate_dicts, item_cate_dicts
    





