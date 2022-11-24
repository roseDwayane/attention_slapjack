import csv
import os
import random
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset
import pickle

epsilon = np.finfo(float).eps

class myDataset(Dataset):
    def __init__(self, mode, data_class):
        self.sample_rate = 250
        self.lenth = 23*40*len(data_class)
        self.lenthtest = 3*40*len(data_class)
        self.lenthval = 3*40*len(data_class)
        self.mode = mode
        self.data_class = data_class

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        #data_class = ["singleplayer", "singlebystander", "coop", "comp"]

        #print("dataset_idx: ", idx)

        if self.mode == 2:
            mode_num = idx // (self.lenthval//len(self.data_class))
            temp_idx = idx % (self.lenthval//len(self.data_class))
            subj_num = temp_idx // 40
            trialnum = temp_idx % 40

            playerA = './slapjack_csv/' + self.data_class[mode_num] + '/' + str(subj_num+27) + '_' + str(trialnum+1) + '_A.csv'
            playerB = './slapjack_csv/' + self.data_class[mode_num] + '/' + str(subj_num+27) + '_' + str(trialnum+1) + '_B.csv'
            dataA = self.read_train_data(playerA)
            dataB = self.read_train_data(playerB)

        elif self.mode == 1:
            mode_num = idx // (self.lenthtest//len(self.data_class))
            temp_idx = idx % (self.lenthtest//len(self.data_class))
            subj_num = temp_idx // 40
            trialnum = temp_idx % 40

            playerA = './slapjack_csv/' + self.data_class[mode_num] + '/' + str(subj_num + 24) + '_' + str(
                trialnum + 1) + '_A.csv'
            playerB = './slapjack_csv/' + self.data_class[mode_num] + '/' + str(subj_num + 24) + '_' + str(
                trialnum + 1) + '_B.csv'
            dataA = self.read_train_data(playerA)
            dataB = self.read_train_data(playerB)

        else:
            mode_num = idx // (self.lenth//len(self.data_class))
            temp_idx = idx % (self.lenth//len(self.data_class))
            subj_num = temp_idx // 40
            trialnum = temp_idx % 40

            playerA = './slapjack_csv/' + self.data_class[mode_num] + '/' + str(subj_num + 1) + '_' + str(
                trialnum + 1) + '_A.csv'
            playerB = './slapjack_csv/' + self.data_class[mode_num] + '/' + str(subj_num + 1) + '_' + str(
                trialnum + 1) + '_B.csv'
            dataA = self.read_train_data(playerA)
            dataB = self.read_train_data(playerB)


        dataA = np.array(dataA).astype(np.float)
        dataB = np.array(dataB).astype(np.float)
        data = np.concatenate((dataA, dataB), axis=0)
        #label = np.zeros(len(self.data_class))
        #label[mode_num] = 1
        #label = np.array(label).astype(np.float)
        label = np.array(mode_num).astype(np.float)

        data = data.copy()
        data = torch.tensor(data, dtype=torch.float32)

        label = label.copy()
        #label = torch.tensor(label, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        #print("dataset_file: ", playerA)
        #print("dataset_label: ", label)

        return data, label


    def read_train_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        new_data = np.array(data).astype(np.float)

        return new_data