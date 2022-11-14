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
    def __init__(self, rate):
        super(FM, self).__init__()
        self.dropout = tf.keras.layers.Dropout(rate)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inp1, inp2, training):

        self.summed_features_emb = tf.reduce_sum(inp2, 1)  # None * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        self.squared_features_emb = tf.square(inp2)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # ________ FM __________
        self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
        self.FM = self.bn(self.FM)
        self.FM = self.dropout(self.FM, training=training)  # dropout at each Deep layer

        # _________out _________
        Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1

        self.Bilinear_bias = tf.reduce_sum(tf.reduce_sum(inp1, 1), 1, keep_dims=True)

        self.out = tf.nn.sigmoid(tf.add_n([Bilinear, self.Bilinear_bias]))  # None * 1

        return self.out
