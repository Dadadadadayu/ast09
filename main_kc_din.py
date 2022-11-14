import numpy as np
import pandas as pd

data_path = 'assist2009.csv'
df = pd.read_csv(data_path, encoding="latin1", index_col=False)

split_symbol = '_'

# df['skill_id'] = df['new_sort_skill_id']

df['is_correct'] = df['correct'].astype(int)

df = df[df['is_correct'].isin([0, 1])]  # Remove potential continuous outcomes

# Filter out users that have less than min_interactions interactions
df = df.groupby("user_id").filter(lambda x: len(x) >= 10)


# Remove NaN skills
df = df[~df["skill_id"].isnull()]

print('Encoding features...')
df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
df.reset_index(inplace=True, drop=True)

from data_loader import Data_loader

data = Data_loader(df, split_symbol)

X_wins_nums, X_fails_nums, num_users, num_items, num_skills, skill_pad, dic_kc = data.data_load()

df['wins_nums'] = X_wins_nums
df['fails_nums'] = X_fails_nums

print('data loaded successfully ...')

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
for user,ct in zip(df['user_id'], df['user_records']):
    if ct <= int(dic_user_count[user] * split_ratio):
        tr_te.append(1)
    else:
        tr_te.append(0)
df['train_symbol'] = tr_te

train_data = df[df['train_symbol']==1]
test_data = df[df['train_symbol']==0]
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
    train_list_seq.append(np.reshape(train_skill, (-1, 1, skill_pad)))
    train_list_seq.append(np.reshape(train_skill_mask, (-1, 1, skill_pad)))

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
    test_list_seq.append(test_skill)
    test_list_seq.append(test_skill_mask)

print('Filling config...')
feature_config = {}
feature_config['user_id'] = num_users
feature_config['item_id'] = num_items
feature_config['skill_id'] = num_skills+1
print(feature_config)
print('Done.')

from rms_ktm_kc_din import tf, FM, set_seed
import numpy as np
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score

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
        inputs_embed_2.append(single_input_embed_2)
        if col == 'item_id':
            cur_item.append(single_input_embed_2)
    list_inputs = []
    # ------ list ------
    for i, col in enumerate(LIST_FEATURES):
        single_input = Input(batch_shape=(None, None, kc_length), name=f'input_of_{col}')
        list_inputs.append(single_input)

        # single_input_embed_1 = list_embeddings_1[i](single_input)  # batch_size*1*kc_*1
        single_input_embed_2 = list_embeddings_2[i](single_input)  # batch_size*1*kc_*d_model

        mask_input = Input(batch_shape=(None, None, kc_length), name=f'input_of_skill_mask')
        list_inputs.append(mask_input)
        cur_skill_mask.append(tf.reshape(mask_input, [-1, kc_length, 1]))

        cur_skill.append(tf.reduce_sum(single_input_embed_2, 1))

    # inputs_embed_1 = tf.concat(inputs_embed_1, axis=1)
    inputs_embed_2 = tf.concat(inputs_embed_2, axis=1)

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

D_MODEL = 64  # embedding size
LEARNING_RATE = 0.001  # 学习率
DROP = 0.7

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


list_gauc, list_auc, list_acc = [], [], []
for epoch in range(500):
    model.fit(train_seq[1:]+train_list_seq, train_seq[0:1], batch_size=4096, epochs=1, verbose=1, shuffle=True)
    test_pred = pred_model.predict(test_seq[1:]+test_list_seq)
    num_example = len(test_seq[0])
    y_pred = np.reshape(test_pred, (num_example,))
    y_true = np.reshape(test_seq[0], (num_example,))
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
