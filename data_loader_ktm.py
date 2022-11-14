import numpy as np
import pandas as pd
from scipy import sparse
import pickle


class Data_loader(object):
    def __init__(self, df, sp):
        self.df = df
        self.sp = sp

    def data_load(self):
        df = self.df

        num_users = 1 + df['user_id'].max()  # user/item/skill IDs are distinct
        num_items = int(1 + df['item_id'].max())

        # Build q-matrix
        listOfKC = []
        length = 0
        for kc_raw in df["skill_id"].unique():
            m = str(kc_raw).split(self.sp)
            length = max(length, len(m))
            for elt in m:
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)  # 将所有技能ID排序，删去重复的ID
        num_skills = len(listOfKC)
        print("The max length of skill is: ", length)
        dic_kc = {}
        for k, v in enumerate(listOfKC):  # 0 skill1, 1 skill2, ...
            dic_kc[v] = k  # dict1_kc[skill1] = 0

        # Build Q-matrix
        Q_wins = np.zeros((num_users, num_skills))  # 构建Q矩阵，题目个数*知识点个数
        Q_fails = np.zeros((num_users, num_skills))  # 构建Q矩阵，题目个数*知识点个数

        list_wins_nums, list_fails_nums = [], []

        for user, skills, correct in zip(df['user_id'], df['skill_id'], df['is_correct']):
            cur_wins, cur_fails = [], []
            for skill in str(skills).split(self.sp):
                cur_wins.append(Q_wins[user, dic_kc[skill]])
                cur_fails.append(Q_fails[user, dic_kc[skill]])
                if correct:
                    Q_wins[user, dic_kc[skill]] += 1
                else:
                    Q_fails[user, dic_kc[skill]] += 1
            cur_wins += [0] * (length-len(cur_wins))
            cur_fails += [0] * (length - len(cur_fails))
            list_wins_nums.append(cur_wins)
            list_fails_nums.append(cur_fails)

        return list_wins_nums, list_fails_nums, num_users, num_items, num_skills, length, dic_kc
