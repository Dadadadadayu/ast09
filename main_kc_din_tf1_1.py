import numpy as np
import pandas as pd
from data_loader_din_tf1 import Data_loader
import random
import logging

# Data loading
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(
    '2021-9-20-rms-ktm-din.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)

data = Data_loader()
dic, num_users, num_items, num_skills, skill_pad = data.data_load()
logger.info("user_nums=%d, item_nums=%d, skill_max_nums=%d" % (num_users, num_items, num_skills))
logger.info("Data loaded successfully-----")
print("Data loaded successfully-----")

print('Filling config...')
feature_config = {}
feature_config['user_id'] = num_users
feature_config['item_id'] = num_items
feature_config['skill_id'] = num_skills + 1
print(feature_config)
print('Done.')

list_acc = []
list_auc = []
list_acc_best, list_auc_best = [], []
total_time = []
X_train_user, X_train_item, X_train_skill, X_train_skill_nums, X_train_wins, X_train_fails, X_train_wins_nums, X_train_fails_nums, \
Y_train, Y_train_diff, X_train_last, X_train_last_nums = [], [], [], [], [], [], [], [], [], [], [], []
X_test_user, X_test_item, X_test_skill, X_test_skill_nums, X_test_wins, X_test_fails, X_test_wins_nums, X_test_fails_nums, \
Y_test, Y_test_diff, X_test_last, X_test_last_nums = [], [], [], [], [], [], [], [], [], [], [], []
train_rate = 0.8
test_rate = 0.2
for user in dic:
    random.shuffle(dic[user])
    length = len(dic[user])
    # if length < 10:
    #     continue
    for index in range(int(train_rate * length)):
        X_train_user.append(dic[user][index][0])
        X_train_item.append(dic[user][index][1])
        X_train_skill.append(dic[user][index][2])
        X_train_skill_nums.append(dic[user][index][3])
        X_train_wins_nums.append(dic[user][index][4])
        X_train_fails_nums.append(dic[user][index][5])
        Y_train.append(dic[user][index][6])
    for index in range(length-int(test_rate * length), length):
        X_test_user.append(dic[user][index][0])
        X_test_item.append(dic[user][index][1])
        X_test_skill.append(dic[user][index][2])
        X_test_skill_nums.append(dic[user][index][3])
        X_test_wins_nums.append(dic[user][index][4])
        X_test_fails_nums.append(dic[user][index][5])
        Y_test.append(dic[user][index][6])
Train_data = {'X_user': X_train_user, 'X_item': X_train_item, 'X_skill': X_train_skill, 'Y': Y_train,
              'X_skill_mask': X_train_skill_nums, 'X_wins_nums': X_train_wins_nums,
              'X_fails_nums': X_train_fails_nums}
Test_data = {'X_user': X_test_user, 'X_item': X_test_item, 'X_skill': X_test_skill, 'Y': Y_test,
             'X_skill_mask': X_test_skill_nums, 'X_wins_nums': X_test_wins_nums, 'X_fails_nums': X_test_fails_nums}

print('Filling config...')
feature_config = {}
feature_config['user_id'] = num_users
feature_config['item_id'] = num_items
feature_config['skill_id'] = num_skills+1
print(feature_config)
print('Done.')

from rms_ktm_kc_din1 import tf, FM, set_seed
import numpy as np
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score

SPARSE_FEATURES = ['wins_nums', 'fails_nums', 'item_id', 'user_id']
LIST_FEATURES = ['skill_id']

# set_seed(2021)


def build(d_model, rate, num_skills, kc_length):
    # embeddings
    embeddings_1 = []
    embeddings_2 = []
    for col in SPARSE_FEATURES[2:]:
        # embeddings_1.append(Embedding(feature_config[col], 1))
        embeddings_2.append(Embedding(feature_config[col], d_model))

    list_embeddings_1 = []
    list_embeddings_2 = []
    for col in LIST_FEATURES:
        # list_embeddings_1.append(Embedding(feature_config[col], 1))
        list_embeddings_2.append(Embedding(feature_config[col], d_model))

    inputs = [Input(batch_shape=(None, None), name=f'input_of_wins_nums'),
              Input(batch_shape=(None, None), name=f'input_of_fails_nums')]

    inputs_embed_1 = []
    inputs_embed_2 = []
    cur_skill = []
    cur_skill_mask = []
    cur_item = []
    # ------ sparse ------
    for i, col in enumerate(SPARSE_FEATURES[2:]):
        single_input = Input(batch_shape=(None, None), name=f'input_of_{col}')
        inputs.append(single_input)

        # single_input_embed_1 = embeddings_1[i](single_input)
        single_input_embed_2 = embeddings_2[i](single_input)
        # inputs_embed_1.append(single_input_embed_1)
        if col == 'item_id':
            cur_item.append(single_input_embed_2)
            # inputs_embed_2.append(tf.keras.layers.Dense(d_model)(single_input_embed_2))
        elif col == 'user_id':
            inputs_embed_2.append(single_input_embed_2)
    list_inputs = []
    # ------ list ------
    for i, col in enumerate(LIST_FEATURES):
        single_input = Input(batch_shape=(None, None), name=f'input_of_{col}')
        list_inputs.append(single_input)

        # single_input_embed_1 = list_embeddings_1[i](single_input)  # batch_size*1*kc_*1
        single_input_embed_2 = list_embeddings_2[i](single_input)  # batch_size*kc_*d_model

        mask_input = Input(batch_shape=(None, None), name=f'input_of_skill_mask')
        list_inputs.append(mask_input)
        cur_skill_mask.append(tf.reshape(mask_input, [-1, kc_length, 1]))

        cur_skill.append(single_input_embed_2)
        # inputs_embed_2.append(tf.keras.layers.Dense(d_model)(single_input_embed_2))
    print(len(inputs_embed_2))
    # inputs_embed_1 = tf.concat(inputs_embed_1, axis=1)
    inputs_embed_2 = tf.concat(inputs_embed_2, axis=1)
    print(inputs_embed_2.shape)

    cur_kc = tf.concat(cur_skill, axis=1)
    cur_kc_mask = tf.concat(cur_skill_mask, axis=1)

    cur_p = tf.concat(cur_item, axis=1)

    fm = FM(rate, d_model, num_skills, kc_length)

    output = fm(inputs_embed_2, inputs[:2], cur_kc, cur_kc_mask, cur_p, True)
    predict_output = fm(inputs_embed_2, inputs[:2], cur_kc, cur_kc_mask, cur_p, False)

    model = Model(inputs=inputs+list_inputs, outputs=output)
    model.compile(
        optimizer='adam', loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    pred_model = Model(inputs=inputs+list_inputs, outputs=predict_output)

    return model, pred_model

D_MODEL = 128  # embedding size
LEARNING_RATE = 0.001  # 学习率
DROP = 0.8

model, pred_model = build(D_MODEL, DROP, num_skills, skill_pad)
tf.keras.backend.set_value(model.optimizer.learning_rate, LEARNING_RATE)

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def calc_gauc(df, label_col='correct_rate', pred_col='prediction', group_col='stu_id'):
    dfg = df.groupby(group_col).agg(list)
    auc_sum = 0.0
    cnt = 0
    for idx in dfg.index:
        y_true = dfg[label_col][idx]
        if np.mean(y_true) > 0.0 and np.mean(y_true) < 1.0:
            y_pred = dfg[pred_col][idx]
            auc = roc_auc_score(y_true, y_pred)
            auc_sum += auc * len(y_true)
            cnt += len(y_true)
    gauc = auc_sum / cnt
    return gauc

train = []
train.append(X_train_wins_nums)
train.append(X_train_fails_nums)
train.append(X_train_item)
train.append(X_train_user)
train.append(X_train_skill)
train.append(X_train_skill_nums)

test = []
test.append(X_test_wins_nums)
test.append(X_test_fails_nums)
test.append(X_test_item)
test.append(X_test_user)
test.append(X_test_skill)
test.append(X_test_skill_nums)

list_gauc, list_auc, list_acc = [], [], []
for epoch in range(500):
    model.fit(train, [Y_train], batch_size=4096, epochs=1, verbose=1, shuffle=True)
    test_pred = pred_model.predict(test)
    num_example = len(Y_test)
    y_pred = np.reshape(test_pred, (num_example,))
    y_true = np.reshape(Y_test, (num_example,))
    # stu = np.reshape(test_seq[-1], (num_example,))

    print('epoch: ', epoch)
    acc = accuracy_score(y_true, (y_pred>0.5))
    print('acc %.4f' % acc)
    auc = roc_auc_score(y_true, y_pred)
    print('auc %.4f' % auc)
    # # group metric by stu
    # cur = pd.DataFrame()
    # cur['label'] = y_true
    # cur['pred'] = y_pred
    # cur['stu'] = stu
    # gauc = calc_gauc(cur, label_col='label', pred_col='pred', group_col='stu')
    # print('gauc %.4f' % gauc)

    list_acc.append(acc)
    list_auc.append(auc)
    # list_gauc.append(gauc)
best_auc = max(list_auc)
best_auc_index = list_auc.index(best_auc)
# print('best epoch %d, auc %.4f, gauc %.4f' % (best_gauc_index, list_auc[best_gauc_index], list_gauc[best_gauc_index]))
print('best epoch %d, acc %.4f, auc %.4f' % (best_auc_index, list_acc[best_auc_index], list_auc[best_auc_index]))
