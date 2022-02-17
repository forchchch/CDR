import tensorflow as tf 
import numpy as np

class Add_unclick(object):

    # user_info_cate: int,how many features used in user_side_profile
    # number_each_user_cate: tuple. how many categories for each user feature
    # item_info_cate: int,how many features used for each item
    # number_each_item_cate:tuple, how many categories for each item feature
    def __init__(self, args):
        self.intention_weight = args.intention_weight
        self.embed_dim = args.embed_dim
        self.hist_size = args.hist_size
        self.batch_size = args.batch_size
        self.item_info_cate = args.item_info_cate
        self.user_info_cate = args.user_info_cate
        self.clicked_item_dim = args.item_info_cate * args.embed_dim
        self.unclick_item_dim = args.item_info_cate * args.embed_dim
        self.dislike_item_dim = args.item_info_cate * args.embed_dim
        self.pos_item_dim = args.embed_dim
        self.position_len = args.position_len
        self.user_dim = args.user_info_cate * self.embed_dim
        self.item_dim = self.clicked_item_dim
        self.group_feature = {}
        self.results = None
        self.train_op = None
        self.loss = None

        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
        ##################### add placeholder
        # user_side_infomation
        for cate in range(args.user_info_cate):
            self.group_feature["user_feature_" + str(cate)] = tf.placeholder(tf.int32, shape=[None],name=("user_feature_" + str(cate)))

        # candidate_item_information
        for cate in range(args.item_info_cate):
            self.group_feature["candidate_item_feature_" + str(cate)] = tf.placeholder(tf.int32, shape=[None], name=("candidate_item_feature_" + str(cate)))

        # history_item_information
        # clicked + unclick + dislike [batch_size,his]*item_info_cate
        for cate in range(args.item_info_cate):
            self.group_feature["clicked_item_feature_" + str(cate)] = tf.placeholder(tf.int32, shape=[None, args.hist_size],name=("clicked_item_feature_" + str(cate)))
            self.group_feature["unclick_item_feature_" + str(cate)] = tf.placeholder(tf.int32, shape=[None, args.hist_size],name=("unclick_item_feature_" + str(cate)))
            self.group_feature["dislike_item_feature_" + str(cate)] = tf.placeholder(tf.int32, shape=[None, args.hist_size],name=("dislike_item_feature_" + str(cate)))

        # history_item_position_information
        self.group_feature["clicked_item_feature_position"] = tf.placeholder(tf.int32, shape=[None, args.hist_size],name=("clicked_item_feature_position"))
        self.group_feature["unclick_item_feature_position"] = tf.placeholder(tf.int32, shape=[None, args.hist_size],name=("unclick_item_feature_position"))
        self.group_feature["dislike_item_feature_position"] = tf.placeholder(tf.int32, shape=[None, args.hist_size],name=("dislike_item_feature_position"))

        # the actual length of the history for each user
        self.group_feature["clicked_histLen"] = tf.placeholder(tf.int32, shape=[None], name=("clicked_histLen"))
        self.group_feature["unclick_histLen"] = tf.placeholder(tf.int32, shape=[None], name=("unclick_histLen"))
        self.group_feature["dislike_histLen"] = tf.placeholder(tf.int32, shape=[None], name=("dislike_histLen"))

        # self.group_feature["pos_list"] = tf.placeholder(tf.int32, shape=[None], name=("pos_list"))
        # self.group_feature["neg_list"] = tf.placeholder(tf.int32, shape=[None], name=("neg_list"))

        self.group_feature["c_index"] = tf.placeholder(tf.int32, shape=[None, 2], name=("c_dinex"))
        self.group_feature["d_index"] = tf.placeholder(tf.int32, shape=[None, 2], name=("d_index"))
        self.group_feature["pos_index"] = tf.placeholder(tf.int32, shape=[None, 2], name=("pos_dinex"))
        self.group_feature["neg_index"] = tf.placeholder(tf.int32, shape=[None, 2], name=("neg_index"))

        # penalization weights for each user-candidate pair
        self.weights = tf.placeholder(tf.float32, [None], name='weights')

        # labels for each user-candiate pair
        self.labels = tf.placeholder(tf.float32, [None], name='label')

        ###############################build model
        self.buildDFR(args)

        ###############################calculate loss
        self.loss_not_average = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.results, labels=self.labels)
        self.loss_predict = tf.reduce_sum(self.loss_not_average * self.weights)
        self.loss = self.loss_predict + args.gamma * self.loss_addition

        self.soft_results = tf.nn.sigmoid(self.results)
        self.log_loss_not_average = tf.log(self.soft_results)
        self.log_loss = tf.reduce_mean(self.log_loss_not_average*self.weights)
        #self.train_op = tf.train.AdagradOptimizer(args.learning_rate, 1e-6).minimize(self.loss)

    def buildDFR(self, args):
        self.zeta = args.zeta
        ###############################add embedding lookup table

        init_w = tf.truncated_normal_initializer(mean=0, stddev=0.01)
        # user_side_infomation_table
        user_table_list = {}
        for cate in range(args.user_info_cate):
            user_table_list[cate] = tf.get_variable('user_emb_' + str(cate),shape=[args.number_each_user_cate[cate] + 1, self.embed_dim], dtype=tf.float32,initializer=init_w)
            # item_side_information_table
        item_table_list = {}
        for cate in range(args.item_info_cate):
            item_table_list[cate] = tf.get_variable('item_emb_' + str(cate),shape=[args.number_each_item_cate[cate] + 1, self.embed_dim],dtype=tf.float32,initializer=init_w)
            # item_position_table
        item_position_table = tf.get_variable('item_position_emb', shape=[self.position_len + 1, self.embed_dim],dtype=tf.float32,initializer=init_w)

        # ###############################add wide embedding lookup table
        # user_wide_table_list = {}
        # for cate in range(args.user_info_cate):
        #     user_wide_table_list[cate] = tf.get_variable('user_wide_emb_' + str(cate),shape=[args.number_each_user_cate[cate] + 1, 1],dtype=tf.float32,initializer=tf.zeros_initializer())
        # item_wide_table_list = {}
        # for cate in range(args.item_info_cate):
        #     item_wide_table_list[cate] = tf.get_variable('item_wide_emb_' + str(cate),shape=[args.number_each_item_cate[cate] + 1, 1],dtype=tf.float32,initializer=tf.zeros_initializer())

        ###################################### get corresponding initialized embeddings
        user_side_embeddings = []
        for cate in range(args.user_info_cate):
            user_side_embedding = tf.nn.embedding_lookup(user_table_list[cate],self.group_feature["user_feature_" + str(cate)])
            user_side_embeddings.append(user_side_embedding)
        main_embedding = tf.concat(user_side_embeddings, axis=1)

        candidate_item_embeddings = []
        for cate in range(args.item_info_cate):
            candidate_item_embedding = tf.nn.embedding_lookup(item_table_list[cate],self.group_feature["candidate_item_feature_" + str(cate)])
            candidate_item_embeddings.append(candidate_item_embedding)
        candidate_embedding = tf.concat(candidate_item_embeddings, axis=1)
        pos_w_clicked = tf.get_variable('pos_w_clicked', shape=[self.clicked_item_dim + self.pos_item_dim, self.item_dim],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        pos_w_unclick = tf.get_variable('pos_w_unclick', shape=[self.unclick_item_dim + self.pos_item_dim, self.item_dim],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        pos_w_dislike = tf.get_variable('pos_w_dislike', shape=[self.dislike_item_dim + self.pos_item_dim, self.item_dim],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        clicked_embeddings = []
        unclick_embeddings = []
        dislike_embeddings = []
        for cate in range(args.item_info_cate):
            clicked_embedding = tf.nn.embedding_lookup(item_table_list[cate],self.group_feature["clicked_item_feature_" + str(cate)])
            unclick_embedding = tf.nn.embedding_lookup(item_table_list[cate],self.group_feature["unclick_item_feature_" + str(cate)])
            dislike_embedding = tf.nn.embedding_lookup(item_table_list[cate],self.group_feature["dislike_item_feature_" + str(cate)])
            clicked_embeddings.append(clicked_embedding)
            unclick_embeddings.append(unclick_embedding)
            dislike_embeddings.append(dislike_embedding)

        clicked_position_embedding = tf.nn.embedding_lookup(item_position_table,self.group_feature["clicked_item_feature_position"])
        unclick_position_embedding = tf.nn.embedding_lookup(item_position_table,self.group_feature["unclick_item_feature_position"])
        dislike_position_embedding = tf.nn.embedding_lookup(item_position_table,self.group_feature["dislike_item_feature_position"])

        clicked_embeddings.append(clicked_position_embedding)
        unclick_embeddings.append(unclick_position_embedding)
        dislike_embeddings.append(dislike_position_embedding)

        clicked_pos1 = tf.concat(clicked_embeddings, axis=2)
        unclick_pos1 = tf.concat(unclick_embeddings, axis=2)
        dislike_pos1 = tf.concat(dislike_embeddings, axis=2)

        clicked_z = tf.tensordot(clicked_pos1, pos_w_clicked, axes=1)
        unclick_z = tf.tensordot(unclick_pos1, pos_w_unclick, axes=1)
        dislike_z = tf.tensordot(dislike_pos1, pos_w_dislike, axes=1)

        trans_clicked = clicked_z
        trans_dislike = dislike_z
        trans_unclick = unclick_z

        #####################begin user intention modeling
        for layer in range(args.layers):
            trans_clicked = self.transformer_encoder(trans_clicked, self.hist_size, self.item_dim, self.group_feature["clicked_histLen"],prefix="clicked" + str(layer))
            trans_dislike = self.transformer_encoder(trans_dislike, self.hist_size, self.item_dim, self.group_feature["dislike_histLen"],prefix="dislike" + str(layer))
            trans_unclick = self.transformer_encoder(trans_unclick, self.hist_size, self.item_dim, self.group_feature["unclick_histLen"],prefix="unclick" + str(layer))

        unclick_score1 = tf.reshape( self.unclick_scoring(main_embedding,trans_unclick,prefix ="score_") , [-1,self.hist_size])
        neg_preference = tf.reduce_sum( trans_dislike, axis=1)
        unclick_score2 = self.unclick_scoring2(neg_preference, trans_unclick ,prefix ="score_2")
        unclick_weights = tf.nn.sigmoid( unclick_score1 + unclick_score2 )
        z_c = trans_clicked
        z_t1 = candidate_embedding
        self.user_intention_capture(z_c, trans_unclick, z_t1, unclick_weights, clicked_position_embedding, args.alpha, args.beta, args.K, neg_preference, self.group_feature["clicked_histLen"], self.group_feature["unclick_histLen"], prefix1="ccontent_")
        self.feature_results = self.feature_interaction(main_embedding,candidate_embedding,prefix="judge_")
        
        self.results = self.history_results+self.feature_results
        # print(self.results.shape.as_list())
        self.saver = tf.train.Saver()

    def attention(self, candidate_embedding, hist_embeddings, hist_embedding_dim, hisLens, prefix=""):
        attention_hidden_ = 32
        attW1 = tf.get_variable(prefix + "attention_hidden_w1", shape=[hist_embedding_dim * 4, attention_hidden_],
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        attB1 = tf.get_variable(prefix + "attention_hidden_b1", shape=[attention_hidden_], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

        attW2 = tf.get_variable(prefix + "attention_hidden_w2", shape=[attention_hidden_, 1], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        attB2 = tf.get_variable(prefix + "attention_hidden_b2", shape=[1], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
        hist_embedding_list = []
        for i in range(0, self.hist_size):
            z1 = tf.concat([candidate_embedding, hist_embeddings[i], candidate_embedding * hist_embeddings[i],
                            candidate_embedding - hist_embeddings[i]], axis=1)
            hist_embedding_list.append(z1)
        hist_z_all = tf.stack(hist_embeddings, axis=1)  # (batch, hist_size, hist_embedding_dim)
        z2 = tf.concat(hist_embedding_list, axis=1)  # (batch, hist_size * hist_embedding_dim * 4)
        z3 = tf.reshape(z2, [-1, self.hist_size, 4 * hist_embedding_dim])
        z4 = tf.tensordot(z3, attW1, axes=1) + attB1  # (batch , hist_size, attention_hidden_)
        z5 = tf.nn.relu(z4)
        z6 = tf.tensordot(z5, attW2, axes=1) + attB2  # (batch, hist_size, 1)
        att_w_all = tf.reshape(z6, [-1, self.hist_size])

        # mask
        hist_masks = tf.sequence_mask(hisLens, self.hist_size)  # (batch, hist_size)
        padding = tf.ones_like(att_w_all) * (-2 ** 32 + 1)
        att_w_all_rep = tf.where(hist_masks, att_w_all, padding)

        # scale
        att_w_all_scale = att_w_all_rep / (hist_embedding_dim ** 0.5)

        # norm
        att_w_all_norm = tf.nn.softmax(att_w_all_scale)

        att_w_all_mul = tf.reshape(att_w_all_norm, [-1, 1, self.hist_size])
        weighted_hist_all = tf.matmul(att_w_all_mul, hist_z_all)  # (batch, 1, hist_embedding_dim)
        return tf.reshape(weighted_hist_all, [-1, hist_embedding_dim])

    def transformer_encoder(self, hist_z_all, hist_size, hist_embedding_dim, hisLens, prefix=""):
        headnum = 4
        mutil_head_att = []
        hist_z_all = self.layer_normalize(hist_z_all,scope=prefix+"ln1",params_shape=hist_embedding_dim)
        # attention
        for i in range(0, headnum):
            attQ_w = tf.get_variable(prefix + "attQ_w" + str(i), shape=[hist_embedding_dim, hist_embedding_dim / headnum],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            attK_w = tf.get_variable(prefix + "attK_w" + str(i), shape=[hist_embedding_dim, hist_embedding_dim / headnum],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            attV_w = tf.get_variable(prefix + "attV_w" + str(i), shape=[hist_embedding_dim, hist_embedding_dim / headnum],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            attQ = tf.tensordot(hist_z_all, attQ_w, axes=1)  # (batch, hist_size, hist_embedding_dim/headnum)
            attK = tf.tensordot(hist_z_all, attK_w, axes=1)  # (batch, hist_size, hist_embedding_dim/headnum)
            attV = tf.tensordot(hist_z_all, attV_w, axes=1)  # (batch, hist_size, hist_embedding_dim/headnum)

            attQK = tf.matmul(attQ, attK, transpose_b=True)  # (batch, hist_size, hist_size)

            # scale
            attQK_scale = attQK / (hist_embedding_dim ** 0.5)
            padding = tf.ones_like(attQK_scale) * (-2 ** 32 + 1)  # (batch, hist_size, hist_size)

            # mask
            key_masks = tf.sequence_mask(hisLens, hist_size)  # (batch, hist_size)
            key_masks_new = tf.reshape(key_masks, [-1, 1, hist_size])
            key_masks_tile = tf.tile(key_masks_new, [1, hist_size, 1])  # (batch, hist_size, hist_size)
            key_masks_cast = tf.cast(key_masks_tile, dtype=tf.float32)
            outputs_QK = tf.where(key_masks_tile, attQK_scale, padding)  # (batch, hist_size, hist_size)

            # norm
            outputs_QK_norm = tf.nn.softmax(outputs_QK)

            # query mask
            outputs_QK_q = tf.multiply(outputs_QK_norm, key_masks_cast)  # (batch, hist_size, hist_size)
            # weighted sum
            outputs_QKV_head = tf.matmul(outputs_QK_q, attV)  # (batch, hist_embedding_dim/headnum)
            mutil_head_att.append(outputs_QKV_head)

        outputs_QKV = tf.concat(mutil_head_att, axis=2) + hist_z_all
        outputs_QKV = self.layer_normalize(outputs_QKV,scope=prefix+"ln2",params_shape=hist_embedding_dim)
        # FFN
        FFN_w0 = tf.get_variable(prefix + 'FFN_w0', shape=[hist_embedding_dim, hist_embedding_dim * 4], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        FFN_b0 = tf.get_variable(prefix + 'FFN_b0', shape=[hist_embedding_dim * 4], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

        FFN_w1 = tf.get_variable(prefix + 'FFN_w1', shape=[hist_embedding_dim * 4, hist_embedding_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        FFN_b1 = tf.get_variable(prefix + 'FFN_b1', shape=[hist_embedding_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

        TH0 = tf.tensordot(outputs_QKV, FFN_w0, axes=1) + FFN_b0  # (batch, hist_size, hist_embedding_dim * 4)
        TZ0 = tf.nn.dropout(tf.nn.relu(TH0), self.dropout_rate)
        TH1 = tf.nn.relu(tf.tensordot(TZ0, FFN_w1, axes=1) + FFN_b1)
        TH11 = tf.nn.dropout(TH1, self.dropout_rate)
        # return hist_z_all + tf.nn.dropout( TH1, self.dropout_rate) #(batch,his_len,hist_embedding_dim)
        return TH11 + outputs_QKV
        # return tf.reduce_sum(TH1, axis=1) #(batch, hist_embedding_dim)

    def user_intention_capture(self, z_c, z_u, z_t1, unclick_weights, click_position, alpha, beta, K, neg_preference, click_len, unclick_len ,prefix1):
        Center_embedding = tf.get_variable(prefix1 + "embedding_table", shape=[K, self.item_dim], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
        intention_bias = tf.get_variable(prefix1 + "intention_bias", shape=[K, self.item_dim], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
        cik_tmp = tf.tensordot(self.layer_normalize(z_c,scope="linear1",params_shape=self.item_dim), tf.transpose( self.layer_normalize(Center_embedding,scope="linear2",params_shape=self.item_dim)),
                            axes=1)/ (self.item_dim ** 0.5) # batch, his_len, K, cos similarity)
        cik = tf.exp(cik_tmp) / tf.reduce_sum(tf.exp(cik_tmp), axis=2,
                                            keep_dims=True)  # in other tf versions, 'keep_dims' should be written as 'keepdims'
        uik_tmp = tf.tensordot(self.layer_normalize(z_u,scope="un_linear1",params_shape=self.item_dim), tf.transpose( self.layer_normalize(Center_embedding,scope="un_linear2",params_shape=self.item_dim)),
                            axes=1)/ (self.item_dim ** 0.5) # batch, his_len, K, cos similarity)
        uik = tf.exp(uik_tmp) / tf.reduce_sum(tf.exp(uik_tmp), axis=2,keep_dims=True) 
        
        w1 = tf.get_variable(prefix1 + 'w1', shape=[self.embed_dim + self.item_dim, self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(prefix1 + 'b1', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(prefix1 + 'w2', shape=[self.item_dim, self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(prefix1 + 'b2', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable(prefix1 + 'w3', shape=[self.embed_dim + self.item_dim, self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(prefix1 + 'b3', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable(prefix1 + 'b4', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        
        uw2 = tf.get_variable(prefix1 + 'uw2', shape=[self.item_dim, self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        ub2 = tf.get_variable(prefix1 + 'ub2', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        ub4 = tf.get_variable(prefix1 + 'ub4', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        time_info = tf.concat([z_c,click_position],axis=2)  
        tmp_ec = tf.tensordot(time_info, w1, axes=1) + b1                               
        e_c = z_c + tf.nn.relu(tmp_ec)  ##batch,his_size,item_dim,all history with time information
        e_c1 = z_c + tf.nn.relu( tf.tensordot(z_c,w2,axes=1)+b2)  ####all clicked history for candidate weights
        e_u1 = z_u + tf.nn.relu( tf.tensordot(z_u,uw2,axes=1)+ub2)  ####all unclick history for candidate weights

        click_masks = tf.cast(tf.sequence_mask(click_len, self.hist_size), tf.float32)
        unclick_masks = tf.cast(tf.sequence_mask(unclick_len, self.hist_size), tf.float32)
        z_cm_tmp = tf.gather_nd(time_info, indices=self.group_feature["c_index"])
        z_c_last = tf.gather_nd(z_c, indices=self.group_feature["c_index"])
        z_cm_tmp1 = z_c_last + tf.nn.relu( tf.matmul(z_cm_tmp, w3)+b3 )

        z_cm = tf.reshape(z_cm_tmp1, [-1, 1, self.item_dim])
        z_cm_tile = tf.tile(z_cm, [1, self.hist_size, 1])
        z_t1_tile = tf.tile(tf.reshape(z_t1, [-1, 1, self.item_dim]), [1, self.hist_size, 1])

        ############the right one
        ci_tmp1 = tf.exp(tf.reduce_sum(tf.multiply(self.layer_normalize(e_c, scope="linear3",params_shape=self.item_dim), self.layer_normalize(z_cm_tile, scope="linear4",params_shape=self.item_dim)),
                                    axis=2)/(self.item_dim ** 0.5))  # batch,hist
        ci_tmp2 = tf.exp(tf.reduce_sum(tf.multiply(self.layer_normalize(e_c1, scope="linear5",params_shape=self.item_dim), self.layer_normalize(z_t1_tile + b4, scope="linear6",params_shape=self.item_dim)),
                                    axis=2)/(self.item_dim ** 0.5))  # batch,hist
        ui_tmp2 = tf.exp(tf.reduce_sum(tf.multiply(self.layer_normalize(e_u1, scope="un_linear5",params_shape=self.item_dim), self.layer_normalize(z_t1_tile + ub4, scope="un_linear6",params_shape=self.item_dim)),
                                    axis=2)/(self.item_dim ** 0.5))
        ci = alpha * tf.divide(ci_tmp1, tf.reduce_sum(tf.multiply(click_masks, ci_tmp1), axis=1, keep_dims=True)) + (1 - alpha) * tf.divide(ci_tmp2,tf.reduce_sum(tf.multiply(click_masks, ci_tmp2), axis=1, keep_dims=True))
        ui = tf.divide(ui_tmp2,tf.reduce_sum(tf.multiply(unclick_masks, ui_tmp2), axis=1, keep_dims=True))
       #############the right one
        ci = tf.multiply(click_masks, ci)
        ui = tf.multiply(unclick_masks, ui)
        di = self.get_pos_neg_weight(neg_preference,z_c,click_masks)

        intention = []
        for k in range(K):
            cik_partk = cik[:, :, k]
            uik_partk = uik[:, :, k]
            cof_pos = tf.reshape(di*ci*cik_partk, [-1, self.hist_size, 1])
            cof_unc = tf.reshape(unclick_weights*ui*uik_partk, [-1, self.hist_size, 1])
            z_c1 = self.purify(z_c,neg_preference,k,prefix="pos")
            z_u1 = self.purify(z_u,neg_preference,k,prefix="unclick")
            intention_k1 = tf.reduce_sum(tf.multiply(cof_pos, beta * z_c1 + intention_bias[k]), axis=1)  # batch,item_dim
            intention_k2 = tf.reduce_sum(tf.multiply(cof_unc, beta * z_u1 + intention_bias[k]), axis=1)
            intention_k = intention_k1 + self.intention_weight*intention_k2
            intention.append(intention_k)

        df1 = tf.matmul(tf.nn.l2_normalize(Center_embedding, dim=1),
                        tf.transpose(tf.nn.l2_normalize(Center_embedding, dim=1))) - tf.eye(K)
        reg1 = tf.reduce_sum(tf.reduce_sum(tf.multiply(df1, df1), axis=1), axis=0)

        predict_score_list = []
        for k in range(K):
            predict_k = tf.multiply(intention[k], z_t1)/(self.item_dim**0.5)
            predict_score_list.append(predict_k)

        self.loss_addition = reg1
        self.history_results = tf.reduce_sum(tf.concat(predict_score_list, 1), axis=1)
        # print("***************sixth_check***************")
        # print(self.results.shape.as_list())

    def batch_group_fm_quadratic2(self, fm_input):
        assert len(fm_input.shape) == 3
        sum1 = tf.reduce_sum(fm_input, axis=1)
        sum2 = tf.reduce_sum(fm_input * fm_input, axis=1)
        z = (sum1 * sum1 - sum2) * 0.5
        return z

    def feature_interaction(self, user_embedding, item_embedding, prefix):
        all_features = tf.concat([user_embedding, item_embedding], axis=1)
        feature_feed = tf.reshape(all_features, [-1, self.user_info_cate + self.item_info_cate, self.embed_dim])
        for round1 in range(3):
            feature_feed = self.transformer_encoder2(feature_feed, self.embed_dim,prefix + str(round1))
        feature_after_fusion = tf.reshape(feature_feed, [-1, (self.user_info_cate + self.item_info_cate) * self.embed_dim])
        w = tf.get_variable(prefix + 'pred_w2', shape=[(self.user_info_cate + self.item_info_cate) * self.embed_dim, 1],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(prefix + 'pred_b2', shape=[1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        feature_results = tf.reshape(tf.matmul(feature_after_fusion, w) + b, [-1])
        
        return feature_results
        # print("feature_results_shape:",self.feature_results.shape.as_list())

    def unclick_scoring(self, user_embedding, unclick_embedding, prefix):
        user1 = tf.reshape(user_embedding, [-1, 1, self.user_dim])
        user_tile = tf.tile(user1, [1, self.hist_size, 1]) 
        fusion_feature = tf.concat([user_tile,unclick_embedding],axis=2)
        w1 = tf.get_variable(prefix + 'w1', shape=[self.user_dim + self.item_dim, 32],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(prefix + 'b1', shape=[32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(prefix + 'w2', shape=[32, 1],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(prefix + 'b2', shape=[1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        z1 =  tf.nn.dropout( tf.nn.relu( tf.tensordot(fusion_feature,w1,axes=1) + b1 ), self.dropout_rate )
        z2 = tf.tensordot(z1,w2,axes=1) + b2
        return z2   #batch,hist,1

    def unclick_scoring2(self, neg_preference, unclick_emb, prefix):
        w3 = tf.get_variable(prefix+'neg_pos_w3', shape=[self.item_dim, self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(prefix+'neg_pos_b3', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable(prefix+'neg_pos_b4', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        neg_tile = tf.tile( tf.reshape(neg_preference,[-1,1,self.item_dim]), [1,self.hist_size,1])
        neg_query = neg_tile + b4
        pos_key  = unclick_emb + tf.nn.relu( tf.tensordot(unclick_emb,w3,axes=1)+ b3)
        soft_weights = - tf.reduce_sum(tf.multiply(self.layer_normalize(pos_key, scope=prefix+"pos_negly1",params_shape=self.item_dim), self.layer_normalize(neg_query, scope=prefix+"pos_negly2",params_shape=self.item_dim)),axis=2)/(self.item_dim ** 0.5)         
        
        return soft_weights

    def layer_normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None, params_shape=0):
        with tf.variable_scope(scope, reuse=reuse):
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable('beta', shape=[params_shape], dtype=tf.float32,initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', shape=[params_shape], dtype=tf.float32,initializer=tf.ones_initializer())
            #beta = tf.Variable(tf.zeros(params_shape))
            #gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def transformer_encoder2(self, features, hist_embedding_dim, prefix=""):
        #####hist_embedding_dim: dimensions of the feature in each field
        headnum = 4
        mutil_head_att = []
        # attention
        for i in range(0, headnum):
            attQ_w = tf.get_variable(prefix + "attQ_w" + str(i), shape=[hist_embedding_dim, hist_embedding_dim / headnum],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            attK_w = tf.get_variable(prefix + "attK_w" + str(i), shape=[hist_embedding_dim, hist_embedding_dim / headnum],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            attV_w = tf.get_variable(prefix + "attV_w" + str(i), shape=[hist_embedding_dim, hist_embedding_dim / headnum],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            attQ = tf.tensordot(features, attQ_w, axes=1)  # (batch, hist_size, hist_embedding_dim/headnum)
            attK = tf.tensordot(features, attK_w, axes=1)  # (batch, hist_size, hist_embedding_dim/headnum)
            attV = tf.tensordot(features, attV_w, axes=1)  # (batch, hist_size, hist_embedding_dim/headnum)

            attQK = tf.matmul(attQ, attK, transpose_b=True)  # (batch, hist_size, hist_size)
            # scale
            attQK_scale = attQK / (hist_embedding_dim ** 0.5)
            outputs_QK_norm = tf.nn.softmax(attQK_scale)
            outputs_QKV_head = tf.matmul(outputs_QK_norm, attV)  # (batch, hist_embedding_dim/headnum)
            mutil_head_att.append(outputs_QKV_head)

        outputs_QKV = tf.concat(mutil_head_att, axis=2)
        return tf.nn.relu(features + outputs_QKV)  # (batch,his_len,hist_embedding_dim)

    def get_pos_neg_weight(self, neg_preference, z_c, click_masks):
        w3 = tf.get_variable('neg_pos_w3', shape=[self.item_dim, self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('neg_pos_b3', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable('neg_pos_b4', shape=[self.item_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        neg_tile = tf.tile( tf.reshape(neg_preference,[-1,1,self.item_dim]), [1,self.hist_size,1])
        neg_query = neg_tile + b4
        pos_key  = z_c + tf.nn.relu( tf.tensordot(z_c,w3,axes=1)+ b3)
        soft_weights = tf.exp( - tf.reduce_sum(tf.multiply(self.layer_normalize(pos_key, scope="pos_negly1",params_shape=self.item_dim), self.layer_normalize(neg_query, scope="pos_negly2",params_shape=self.item_dim)),axis=2)/(self.item_dim ** 0.5) )        
        weights = tf.divide( soft_weights,tf.reduce_sum(tf.multiply(click_masks, soft_weights), axis=1, keep_dims=True))
        return weights
    
    def purify(self, pos_features, neg_representation,k,prefix):
        neg_rep_tile = tf.tile( tf.reshape(neg_representation,[-1,1,self.item_dim]), [1, self.hist_size, 1 ])
        w1 = tf.get_variable(prefix+"pure_w1_"+str(k), shape=[self.item_dim*4, self.item_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(prefix+"pure_b1_"+str(k), shape=[self.item_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(prefix+"pure_w2_"+str(k), shape=[self.item_dim, self.item_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(prefix+"pure_b2_"+str(k), shape=[self.item_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        feature_feed = tf.concat( [pos_features, neg_rep_tile, pos_features*neg_rep_tile,pos_features-neg_rep_tile], axis=2 )
        feature_feed = tf.nn.dropout( tf.nn.relu( tf.tensordot(feature_feed,w1,axes=1)+b1 ), self.dropout_rate)
        feature_feed = tf.nn.dropout( tf.tensordot(feature_feed,w2,axes=1) + b2 , self.dropout_rate)
        return pos_features + feature_feed
    
    def stacked_fully_connect(self, x, dims, activation='relu', prefix='deep'):
        activation_dict = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
        }
        assert len(x.shape) == 2
        if dims[0] != x.shape[1]:
            dims = [x.shape[1]] + dims
        dim_size = len(dims) - 1
        hidden = x
        for i in range(0, dim_size):
            w = tf.get_variable(prefix + 'w' + str(i), shape=[dims[i], dims[i + 1]], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(prefix + 'b' + str(i), shape=[1, dims[i + 1]], dtype=tf.float32,
                                initializer=tf.zeros_initializer)
            hidden = tf.matmul(hidden, w) + b
            if prefix != 'output':
                hidden = activation_dict[activation](hidden)
        return hidden

