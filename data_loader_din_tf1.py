import numpy as np
import pandas as pd
from scipy import sparse


class Data_loader(object):
    def __init__(self):
        self.assist09 = pd.read_csv("assist09_akt.csv", encoding="latin1", index_col=False)

    def data_load(self):
        assist09 = self.assist09
        # Filter out users that have less than min_interactions interactions
        assist09 = assist09.groupby("user_id").filter(lambda x: len(x) >= 10)

        # Remove NaN skills
        assist09 = assist09[~assist09["skill_id"].isnull()]

        assist09["item_id"] = np.unique(assist09["problem_id"], return_inverse=True)[1]
        assist09["user_id"] = np.unique(assist09["user_id"], return_inverse=True)[1]
        assist09.reset_index(inplace=True, drop=True)

        # Build q-matrix
        listOfKC = []
        length = 0
        for kc_raw in assist09["skill_id"].unique():
            m = str(kc_raw).split('_')
            length = max(length, len(m))
            for elt in m:
                listOfKC.append(str(int(float(elt))))
        listOfKC = np.unique(listOfKC)  # 将所有技能ID排序，删去重复的ID
        print("The max length of skill is: ", length)
        dict1_kc = {}
        for k, v in enumerate(listOfKC):  # 0 skill1, 1 skill2, ...
            dict1_kc[v] = k  # dict1_kc[skill1] = 0
        # print(dict1_kc)

        assist09 = assist09[assist09.correct.isin([0, 1])]  # Remove potential continuous outcomes  # 移除答题不是1/0的结果
        assist09['correct'] = assist09['correct'].astype(np.int32)  # Cast outcome as int32

        num_users = 1 + assist09['user_id'].max()  # user/item/skill IDs are distinct
        num_items = int(1 + assist09['item_id'].max())
        num_skills = len(listOfKC)

        # Build Q-matrix
        Q_wins = np.zeros((len(assist09["user_id"].unique()), len(listOfKC)))  # 构建Q矩阵，题目个数*知识点个数
        Q_fails = np.zeros((len(assist09["user_id"].unique()), len(listOfKC)))  # 构建Q矩阵，题目个数*知识点个数


        # list_user, list_item, list_skill, list_skill_nums, list_wins_nums, list_fails_nums, target, difficulty = [], [], [], [], [], [], [], []
        list_user, list_item, list_skill, list_skill_nums, list_wins, list_fails, list_wins_nums, list_fails_nums, target, difficulty = [], [], [], [], [], [], [], [], [], []
        list_last_attempt, list_last_nums = [], []
        dic = {}
        for user, item_id, correct, skill_ids in zip(assist09['user_id'], assist09['item_id'],
                                                           assist09['correct'],
                                                           assist09['skill_id']):
            sub_user = [user]
            sub_item = [item_id]
            target.append(float(correct))
            sub_wins_nums, sub_fails_nums = Q_wins[user].copy(), Q_fails[user].copy()
            sub_skill, sub_skill_nums = [], []
            for skill in str(skill_ids).split('_'):
                sub_skill.append(dict1_kc[str(int(float(skill)))])
                sub_skill_nums.append(1)
                # sub_wins_nums.append(Q_wins[user, dict1_kc[str(int(float(skill)))]])
                # sub_fails_nums.append(Q_fails[user, dict1_kc[str(int(float(skill)))]])
                if correct == 1:
                    Q_wins[user, dict1_kc[str(int(float(skill)))]] += 1
                else:
                    Q_fails[user, dict1_kc[str(int(float(skill)))]] += 1

            len_ = length - len(sub_skill)
            # list_user.append(sub_user)
            # list_item.append(sub_item)
            # list_skill.append(sub_skill + [num_skills] * len_)
            # list_skill_nums.append(sub_skill_nums + [0] * len_)
            # list_wins_nums.append(sub_wins_nums + [0] * len_)
            # list_fails_nums.append(sub_fails_nums + [0] * len_)
            if user not in dic:
                dic[user] = []
            dic[user].append([sub_user, sub_item, sub_skill + [num_skills] * len_, sub_skill_nums + [0] * len_,
            sub_wins_nums, sub_fails_nums, [float(correct)]])
        return dic, num_users, num_items, num_skills, length
