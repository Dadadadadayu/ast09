# 设置随机种子
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import random as rn
import os
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, Embedding, Concatenate, Input


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=24, inter_op_parallelism_threads=36)
    from tensorflow.keras import backend as K
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


class FM(tf.keras.Model):
    def __init__(self, rate, d_model, num_skills, kc_length):
        super(FM, self).__init__()
        self.nb_skill = num_skills
        self.kc_ = kc_length

        self.d_model = d_model

        self.pre_dropout1 = tf.keras.layers.Dropout(rate)
        self.pre_dropout2 = tf.keras.layers.Dropout(rate)

        self.second_dropout = tf.keras.layers.Dropout(rate)
        self.second_bn = tf.keras.layers.BatchNormalization()

        self.wins_dense = Dense(d_model)
        self.fails_dense = Dense(d_model)

        self.bn_v1 = tf.keras.layers.BatchNormalization()
        self.bn_v2 = tf.keras.layers.BatchNormalization()

        self.wins_w = Dense(1)
        self.fails_w = Dense(1)
        self.bn_w1 = tf.keras.layers.BatchNormalization()
        self.bn_w2 = tf.keras.layers.BatchNormalization()

        self.matrix_v = Dense(d_model)
        self.matrix_w = Dense(1)

        self.matrix_bn1 = tf.keras.layers.BatchNormalization()
        self.matrix_bn2 = tf.keras.layers.BatchNormalization()

        self.dense_2 = Dense(num_skills)  # pattern number
        self.sim_dense = Dense(num_skills)

        self.pre_dense1 = Dense(d_model)
        self.pre_dense2 = Dense(d_model//2)

        self.out_dense = Dense(1)

        self.pre_bn1 = tf.keras.layers.BatchNormalization()
        self.pre_bn2 = tf.keras.layers.BatchNormalization()

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)

        self.kq = tf.keras.layers.Dense(d_model)

        self.vk = tf.keras.layers.Dense(d_model)
        self.vp = tf.keras.layers.Dense(d_model)

    def call(self, inp2, inp3, cur_kc, cur_kc_mask, cur_p, cur_u, training):
        cur_kc_all_sim = tf.multiply(self.sim_dense(self.kq(cur_kc)), cur_kc_mask)

        q = self.wq(cur_p)  # item_query
        k = self.wk(cur_kc)  # concept_key

        q_k_mask = 1 - cur_kc_mask
        mask_attention_logits = (q_k_mask * -1e9)
        
        din = tf.multiply(tf.reduce_sum(tf.multiply(q, k), 2, keep_dims=True), cur_kc_mask)  # batch * kc_* 1
        din += mask_attention_logits
        din = tf.reshape(tf.nn.softmax(tf.reshape(din, [-1, self.kc_]), axis=-1), [-1, self.kc_, 1])

        self.wins_input = tf.reshape(inp3[0], [-1, 1, self.nb_skill])  # batch * kc_* num_skills
        self.fails_input = tf.reshape(inp3[1], [-1, 1, self.nb_skill])  # batch * kc_* num_skills

        # self.wins_embed_2 = tf.reduce_sum(tf.multiply(self.bn_v1(self.wins_dense(self.wins_input)), din), 1, keep_dims=True)
        # self.fails_embed_2 = tf.reduce_sum(tf.multiply(self.bn_v2(self.fails_dense(self.fails_input)), din), 1, keep_dims=True)
        
        self.wins_embed_2 = tf.reduce_sum(self.wins_dense(self.wins_input), 1, keep_dims=True)
        self.fails_embed_2 = tf.reduce_sum(self.fails_dense(self.fails_input), 1, keep_dims=True)


        study_embed = tf.concat([self.wins_embed_2, self.fails_embed_2], 2)
        study_user_embed = tf.concat([study_embed, cur_u], 2)

        self.master_kc = tf.nn.sigmoid(self.dense_2(study_user_embed))

        self.cur_master_kc = tf.multiply(cur_kc_all_sim, self.master_kc)

        # self.pattern_v = self.matrix_bn2(self.matrix_v(self.cur_master_kc))

        self.pattern_v = self.matrix_v(self.cur_master_kc)

        self.tog_master = tf.reduce_sum(tf.multiply(self.pattern_v, din), 1, keep_dims=True)
        print(tf.shape(self.tog_master), self.tog_master.shape)

        self.kc_q = tf.reduce_sum(tf.multiply(self.vk(cur_kc), din), 1, keep_dims=True)
        self.p_q = self.vp(cur_p)

        inp2 = tf.concat([inp2, self.p_q], 1)
        inp2 = tf.concat([inp2, self.kc_q], 1)
        inp2 = tf.concat([inp2, self.tog_master], 1)

        # inp1 = tf.concat([inp1, tf.reduce_sum(self.pattern_w, 1, keep_dims=True)], 1)
        
        self.summed_features_emb = tf.reduce_sum(inp2, 1)  # None * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        self.squared_features_emb = tf.square(inp2)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        self.second = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
        self.second = self.second_dropout(self.second)

        self.first = tf.reduce_sum(inp2, 1)  # None * 1

        self.pooling_vector = tf.concat([self.first, self.second], 1)

        '''
        conv_bias = tf.layers.conv1d(inp2, filters=self.d_model, kernel_size=3, strides=1, padding='VALID')
        conv_bias = tf.nn.relu(conv_bias)

        conv_bias = tf.layers.conv1d(conv_bias, filters=self.d_model, kernel_size=2, strides=1, padding='VALID')
        conv_bias = tf.reshape(tf.nn.relu(conv_bias), [-1, self.d_model])
        '''

        self.pre_out1 = self.pre_dropout1(tf.nn.relu(self.pre_bn1(self.pre_dense1(self.pooling_vector))))
        # self.pre_out2 = self.pre_dropout2(tf.nn.relu(self.pre_bn2(self.pre_dense2(self.pre_out1))))

        self.out = tf.nn.sigmoid(self.out_dense(self.pre_out1))  # None * 1

        return self.out
