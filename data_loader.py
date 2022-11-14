import numpy as np
import pandas as pd
from scipy import sparse
import pickle


class Data_loader(object):
    def __init__(self, df, split_symbol):
        self.df = df
        self.split_symbol = split_symbol
    def data_load(self):
        df = self.df

        num_users = 1 + df['user_id'].max()  # user/item/skill IDs are distinct
        num_items = int(1 + df['item_id'].max())

        # Build q-matrix
        listOfKC = []
        length = 0
        for kc_raw in df["skill_id"].unique():
            m = str(kc_raw).split(self.split_symbol)
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

        list_user, list_item, list_skill, list_skill_nums, list_wins_nums, list_fails_nums, target = [], [], [], [], [], [], []
        list_user_test, list_item_test, list_skill_test, list_skill_nums_test, list_wins_nums_test, \
        list_fails_nums_test, target_test = [], [], [], [], [], [], []

        for user, skills, correct in zip(df['user_id'], df['skill_id'], df['is_correct']):
            list_wins_nums.append(Q_wins[user])
            list_fails_nums.append(Q_fails[user])
            for skill in str(skills).split(self.split_symbol):
                if correct:
                    Q_wins[user, dic_kc[skill]] += 1
                else:
                    Q_fails[user, dic_kc[skill]] += 1

        return list_wins_nums, list_fails_nums, num_users, num_items, num_skills, length, dic_kc
