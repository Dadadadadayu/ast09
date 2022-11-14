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
    def __init__(self, rate, d_model, num_skills):
        super(FM, self).__init__()
        self.nb_skill = num_skills

        self.dropout = tf.keras.layers.Dropout(rate)
        self.bn = tf.keras.layers.BatchNormalization()

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

        self.pre_dense1 = Dense(d_model//2)
        self.pre_dense2 = Dense(1)

        self.pre_bn = tf.keras.layers.BatchNormalization()

    def call(self, inp2, inp3, cur_kc, cur_kc_mask, training):
        cur_kc_all_sim = tf.multiply(self.sim_dense(cur_kc), cur_kc_mask)

        self.wins_input = tf.multiply(cur_kc_all_sim, tf.reshape(inp3[0], [-1, 1, self.nb_skill]))  # batch * kc_* num_skills
        self.fails_input = tf.multiply(cur_kc_all_sim, tf.reshape(inp3[1], [-1, 1, self.nb_skill]))  # batch * kc_* num_skills

        self.wins_embed_2 = tf.reduce_sum(self.bn_v1(self.wins_dense(self.wins_input)), 1, keep_dims=True)
        self.fails_embed_2 = tf.reduce_sum(self.bn_v2(self.fails_dense(self.fails_input)), 1, keep_dims=True)

        study_embed = tf.concat([self.wins_embed_2, self.fails_embed_2], axis=2)

        self.master_kc = tf.nn.sigmoid(self.dense_2(study_embed))

        self.cur_master_kc = tf.multiply(cur_kc_all_sim, self.master_kc)

        self.pattern_v = self.matrix_bn2(self.matrix_v(self.cur_master_kc))
        # self.pattern_w = self.matrix_bn1(self.matrix_w(self.cur_master_kc))

        inp2 = tf.concat([inp2, tf.reduce_sum(self.pattern_v, 1, keep_dims=True)], 1)

        # inp1 = tf.concat([inp1, tf.reduce_sum(self.pattern_w, 1, keep_dims=True)], 1)

        self.summed_features_emb = tf.reduce_sum(inp2, 1)  # None * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        self.squared_features_emb = tf.square(inp2)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        self.second = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K

        self.first = tf.reduce_sum(inp2, 1)  # None * 1

        self.pooling_vector = tf.concat([self.first, self.second], 1)

        self.pre_out = self.dropout(tf.nn.relu(self.pre_bn(self.pre_dense1(self.pooling_vector))))

        self.out = tf.nn.sigmoid(self.pre_dense2(self.pre_out))  # None * 1

        return self.out
