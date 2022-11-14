'''
Tensorflow implementation of Neural Factorization Machines as described in:
Xiangnan He, Tat-Seng Chua. Neural Factorization Machines for Sparse Predictive Analytics. In Proc. of SIGIR 2017.

This is a deep version of factorization machine and is more expressive than FM.

@author:
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

@references:
'''
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import logging
from sklearn.metrics import mean_squared_error
import pandas as pd

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run RMS-KTM-kc-din.")
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.3,0.3]',
                        help='Keep probability (i.e., 1-dropout_ratio).')
    parser.add_argument('--lamda', type=float, default=0.1,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')

    return parser.parse_args()


class RMS_KTM(BaseEstimator, TransformerMixin):
    def __init__(self, keep_prob, feature_config, hidden_factor, loss_type, epoch,
                 batch_size, learning_rate, lamda_bilinear, optimizer_type, batch_norm, verbose, kc_length,
                 random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.loss_type = loss_type
        self.feature_config = feature_config

        self.kc_ = kc_length

        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed

        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose

        # performance of each epoch
        self.train_acc, self.test_acc, self.train_auc, self.test_auc = [], [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.item_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.skill_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.wins_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.fails_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.cur_kc_mask = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M

            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1

            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings_user = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user_features)  # batch * 1 * d
            nonzero_embeddings_item = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_features)
            nonzero_embeddings_skill = tf.reduce_sum(tf.multiply(
                tf.nn.embedding_lookup(self.weights['skill_embeddings'], self.skill_features),
                tf.expand_dims(self.cur_kc_mask, 2)), 1, keepdims=True)

            nonzero_embeddings_wins = tf.reduce_sum(tf.multiply(
                tf.nn.embedding_lookup(self.weights['wins_embeddings'], self.skill_features),
                tf.expand_dims(self.wins_nums, 2)), 1, keepdims=True)
            nonzero_embeddings_fails = tf.reduce_sum(tf.multiply(
                tf.nn.embedding_lookup(self.weights['fails_embeddings'], self.skill_features),
                tf.expand_dims(self.fails_nums, 2)), 1, keepdims=True)

            nonzero_embeddings = tf.concat([nonzero_embeddings_user, nonzero_embeddings_item], 1)
            nonzero_embeddings = tf.concat([nonzero_embeddings, nonzero_embeddings_skill], 1)
            nonzero_embeddings = tf.concat([nonzero_embeddings, nonzero_embeddings_wins], 1)
            nonzero_embeddings = tf.concat([nonzero_embeddings, nonzero_embeddings_fails], 1)

            # second-order interaction
            self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)
            self.summed_features_emb_square = tf.square(self.summed_features_emb)

            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # batch * k * d
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_fm')
            self.second = tf.nn.dropout(self.FM, self.dropout_keep[0])

            self.Feature_bias_user = tf.nn.embedding_lookup(self.weights['user_bias'], self.user_features)
            self.Feature_bias_item = tf.nn.embedding_lookup(self.weights['item_bias'], self.item_features)
            self.Feature_bias_skill = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['skill_bias'], self.skill_features),
                            tf.expand_dims(self.cur_kc_mask, 2)), 1, keepdims=True)
            self.Feature_bias_wins = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['wins_bias'], self.skill_features),
                            tf.expand_dims(self.wins_nums, 2)), 1, keepdims=True)
            self.Feature_bias_fails = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['fails_bias'], self.skill_features),
                            tf.expand_dims(self.fails_nums, 2)), 1, keepdims=True)

            self.bias = tf.concat([self.Feature_bias_user, self.Feature_bias_item], 1)
            self.bias = tf.concat([self.bias, self.Feature_bias_skill], 1)
            self.bias = tf.concat([self.bias, self.Feature_bias_wins], 1)
            self.bias = tf.concat([self.bias, self.Feature_bias_fails], 1)

            Bilinear = tf.reduce_sum(self.second, 1, keep_dims=True)  # None * 1

            self.Bilinear_bias = tf.reduce_sum(self.bias, 1)
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Bilinear, self.Bilinear_bias, Bias])  # None * 1

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(
                        tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out,
                                                                        labels=self.train_labels)
                else:
                    self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels)
            self.out = tf.nn.sigmoid(self.out)
            self.total_loss = tf.reduce_mean(self.loss)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.total_loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.total_loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.total_loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.total_loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()

        # parameters initialization
        all_weights['user_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_config['user_id'], self.hidden_factor], 0.0, 0.01),
            name='user_embeddings')  # features_M * K
        all_weights['item_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_config['item_id'], self.hidden_factor], 0.0, 0.01),
            name='item_embeddings')  # features_M * K
        all_weights['skill_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_config['skill_id'], self.hidden_factor], 0.0, 0.01),
            name='skill_embeddings')  # features_M * K

        all_weights['wins_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_config['skill_id'], self.hidden_factor], 0.0, 0.01),
            name='wins_embeddings')  # features_M * K
        all_weights['fails_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_config['skill_id'], self.hidden_factor], 0.0, 0.01),
            name='fails_embeddings')  # features_M * K

        all_weights['user_bias'] = tf.Variable(
            tf.random_normal([self.feature_config['user_id'], 1], 0.0, 0.01),
            name='user_bias')  # features_M * K
        all_weights['item_bias'] = tf.Variable(
            tf.random_normal([self.feature_config['item_id'], 1], 0.0, 0.01),
            name='item_bias')  # features_M * K
        all_weights['skill_bias'] = tf.Variable(
            tf.random_normal([self.feature_config['skill_id'], 1], 0.0, 0.01),
            name='skill_bias')  # features_M * K

        all_weights['wins_bias'] = tf.Variable(
            tf.random_normal([self.feature_config['skill_id'], 1], 0.0, 0.01),
            name='wins_bias')  # features_M * K
        all_weights['fails_bias'] = tf.Variable(
            tf.random_normal([self.feature_config['skill_id'], 1], 0.0, 0.01),
            name='fails_bias')  # features_M * K

        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.train_labels: data['Y'],
                     self.wins_nums: data['X_wins_nums'],
                     self.fails_nums: data['X_fails_nums'],
                     self.cur_kc_mask: data['X_skill_mask'],
                     self.item_features: data['X_item'],
                     self.skill_features: data['X_skill'],
                     self.train_phase: True,
                     self.dropout_keep: self.keep_prob}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X_user, X_item, X_skill, Y, Y_diff, X_skill_nums, X_wins, X_wins_nums, X_fails, X_fails_nums = [], [], [], [], [], [], [], [], [], []
        X_last, X_last_nums = [], []
        # forward get sample
        i = start_index
        while len(X_user) < batch_size and i < len(data['X_user']):
            if len(data['X_user'][i]) == len(data['X_user'][start_index]):
                Y.append(data['Y'][i])
                X_user.append(data['X_user'][i])
                X_item.append(data['X_item'][i])
                X_skill.append(data['X_skill'][i])
                X_skill_nums.append(data['X_skill_mask'][i])
                X_wins_nums.append(data['X_wins_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X_user) < batch_size and i >= 0:
            if len(data['X_user'][i]) == len(data['X_user'][start_index]):
                Y.append(data['Y'][i])
                X_user.append(data['X_user'][i])
                X_item.append(data['X_item'][i])
                X_skill.append(data['X_skill'][i])
                X_skill_nums.append(data['X_skill_mask'][i])
                X_wins_nums.append(data['X_wins_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                i = i - 1
            else:
                break
        return {'X_user': X_user, 'X_item': X_item, 'X_skill': X_skill, 'Y': Y, 'X_skill_mask': X_skill_nums,
                'X_wins_nums': X_wins_nums, 'X_fails_nums': X_fails_nums}

    def shuffle_in_unison_scary(self, a, b, c, d, e, f, g):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)
        np.random.set_state(rng_state)
        np.random.shuffle(f)
        np.random.set_state(rng_state)
        np.random.shuffle(g)


    def train(self, Train_data, Test_data):  # fit a dataset
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X_user'], Train_data['X_item'], Train_data['X_skill'],
                                         Train_data['Y'], Train_data['X_skill_mask'],
                                         Train_data['X_wins_nums'], Train_data['X_fails_nums'],)
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_acc, train_auc = self.evaluate(Train_data)
            test_acc, test_auc = self.evaluate(Test_data)

            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)
            self.train_auc.append(train_auc)
            self.test_auc.append(test_auc)

            if self.verbose > 0 and epoch % self.verbose == 0:
                logger.info("Epoch %d [%.1f s]\ttrain_acc=%.4f, test_acc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_acc, test_acc, time() - t2))
                logger.info("Epoch %d [%.1f s]\ttrain_auc=%.4f, test_auc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_auc, test_auc, time() - t2))

            # if self.early_stop > 0 and self.eva_termination(self.test_auc):
            #     # print "Early stop at %d based on validation result." %(epoch+1)
            #     break

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        else:
            if len(valid) > 10:
                if valid[-1] < valid[-2] < valid[-3] < valid[-4] < valid[-5] < valid[-6] < valid[-7] < valid[-8] < \
                        valid[-9] < valid[-10]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        feed_dict = {self.user_features: [item for item in data['X_user']], self.train_labels: [y for y in data['Y']],
                     self.item_features: [item for item in data['X_item']],
                     self.skill_features: [item for item in data['X_skill']],
                     self.cur_kc_mask: [item for item in data['X_skill_mask']],
                     self.wins_nums: [item for item in data['X_wins_nums']],
                     self.fails_nums: [item for item in data['X_fails_nums']],
                     self.train_phase: False,
                     self.dropout_keep: self.no_dropout}
        predictions, loss = self.sess.run((self.out, self.loss), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':

            auc = roc_auc_score(y_true, y_pred)

            acc = np.mean(y_true == np.round(y_pred))

            return acc, auc


if __name__ == '__main__':
    # Data loading
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('2021-9-19-rms-ktm-dims64-din-min10.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    data_path = 'assist2009.csv'
    df = pd.read_csv(data_path, encoding="latin1", index_col=False)

    split_symbol = '_'

    df['is_correct'] = df['correct'].astype(int)

    df = df[df['is_correct'].isin([0, 1])]  # Remove potential continuous outcomes

    # Filter out users that have less than min_interactions interactions
    df = df.groupby("user_id").filter(lambda x: len(x) >= 3)

    # Remove NaN skills
    df = df[~df["skill_id"].isnull()]

    print('Encoding features...')
    df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df.reset_index(inplace=True, drop=True)

    from data_loader_ktm import Data_loader

    data = Data_loader(df, split_symbol)

    X_wins_nums, X_fails_nums, num_users, num_items, num_skills, skill_pad, dic_kc = data.data_load()

    df['wins_nums'] = X_wins_nums
    df['fails_nums'] = X_fails_nums

    print('data loaded successfully ...')

    print(df.head())

    split_ratio = 0.8

    dic_user_count = {}
    cnt = []
    for user in df['user_id']:
        if user not in dic_user_count:
            dic_user_count[user] = 0
        dic_user_count[user] += 1
        cnt.append(dic_user_count[user])
    df['user_records'] = cnt

    tr_te = []
    for user, ct in zip(df['user_id'], df['user_records']):
        if ct <= int(dic_user_count[user] * split_ratio):
            tr_te.append(1)
        else:
            tr_te.append(0)
    df['train_symbol'] = tr_te

    train_data = df[df['train_symbol'] == 1]
    test_data = df[df['train_symbol'] == 0]
    print('Splitting train test successfully...')
    print('train set length is ', len(train_data), 'test set length is ', len(test_data))

    SPARSE_FEATURES = ['wins_nums', 'fails_nums', 'item_id', 'user_id']
    LIST_FEATURES = ['skill_id']

    train_seq = [[[item] for item in train_data['is_correct']]]
    test_seq = [[[item] for item in test_data['is_correct']]]

    train_list_seq = []
    test_list_seq = []
    for col in SPARSE_FEATURES:
        if col in ['wins_nums', 'fails_nums']:
            train_seq.append([item for item in train_data[col]])
            test_seq.append([item for item in test_data[col]])
        else:
            train_seq.append([[item] for item in train_data[col]])
            test_seq.append([[item] for item in test_data[col]])
    for col in LIST_FEATURES:
        train_skill = []
        train_skill_mask = []
        for item in train_data[col]:
            cur_train = []
            cur_train_mask = []
            for kc in str(item).split(split_symbol):
                cur_train.append(dic_kc[kc])
                cur_train_mask.append(1)
            cur_train += [num_skills] * (skill_pad - len(cur_train))
            cur_train_mask += [0] * (skill_pad - len(cur_train_mask))
            train_skill.append([cur_train])
            train_skill_mask.append([cur_train_mask])
        train_list_seq.append(np.reshape(train_skill, (-1, skill_pad)))
        train_list_seq.append(np.reshape(train_skill_mask, (-1, skill_pad)))

        test_skill = []
        test_skill_mask = []
        for item in test_data[col]:
            cur_test = []
            cur_test_mask = []
            for kc in str(item).split(split_symbol):
                cur_test.append(dic_kc[kc])
                cur_test_mask.append(1)
            cur_test += [num_skills] * (skill_pad - len(cur_test))
            cur_test_mask += [0] * (skill_pad - len(cur_test_mask))
            test_skill.append([cur_test])
            test_skill_mask.append([cur_test_mask])
        test_list_seq.append(np.reshape(test_skill, (-1, skill_pad)))
        test_list_seq.append(np.reshape(test_skill_mask, (-1, skill_pad)))

    print('Filling config...')
    feature_config = {}
    feature_config['user_id'] = num_users
    feature_config['item_id'] = num_items
    feature_config['skill_id'] = num_skills + 1
    print(feature_config)
    print('Done.')
    
    logger.info("user_nums=%d, item_nums=%d, skill_nums=%d" % (feature_config['user_id'], feature_config['item_id'], 
                                                               feature_config['skill_id']))
    logger.info("Data loaded successfully-----")
    
    
    Train_data = {'X_user': train_seq[4], 'X_item': train_seq[3], 'X_skill': train_list_seq[0], 'Y': train_seq[0],
                  'X_skill_mask': train_list_seq[1], 'X_wins_nums': train_seq[1], 'X_fails_nums': train_seq[2]}
    Test_data = {'X_user': test_seq[4], 'X_item': test_seq[3], 'X_skill': test_list_seq[0], 'Y': test_seq[0],
                  'X_skill_mask': test_list_seq[1], 'X_wins_nums': test_seq[1], 'X_fails_nums': test_seq[2]}
    print("Data is splitted successfully-----")
    args = parse_args()
    if args.verbose > 0:
        logger.info("hidden_factor=%d, lr=%.4f, lambda=%.4f, optimizer=%s, dropout_keep=%s" %
                    (args.hidden_factor, args.lr, args.lamda, args.optimizer, args.keep_prob))

    # Training
    t1 = time()
    model = RMS_KTM(eval(args.keep_prob), feature_config, args.hidden_factor, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
                  args.optimizer, args.batch_norm, args.verbose, skill_pad)
    model.train(Train_data, Test_data)

    best_auc_score = max(model.test_auc)
    best_acc_score = max(model.test_acc)

    best_epoch = model.test_acc.index(best_acc_score)
    best_epoch_auc = model.test_auc.index(best_auc_score)

    logger.info("Best Iter(test_acc)= %d\t train(acc) = %.4f, test(acc) = %.4f [%.1f s]"
                % (best_epoch + 1, model.train_acc[best_epoch], model.test_acc[best_epoch], time() - t1))
    logger.info("Best Iter(test_auc)= %d\t train(auc) = %.4f, test(auc) = %.4f [%.1f s]"
                % (
                    best_epoch_auc + 1, model.train_auc[best_epoch_auc], model.test_auc[best_epoch_auc],
                    time() - t1))
