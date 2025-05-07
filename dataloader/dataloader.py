import argparse
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings
import torch
warnings.filterwarnings("ignore")


class UnivariateMethaneHourly(Dataset):
    def __init__(self, args, flag="train"):
        self.data_dir = args.data_dir
        self.data_file = args.data_file

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.timesfm = args.timesfm
        if self.timesfm:
            if args.freq_type not in [0, 1, 2]:
                raise ValueError("freq_type must be 0, 1, or 2")
            self.freq_type = args.freq_type

        self.scaler = MinMaxScaler()
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type_flag = type_map[flag]

        self._read_data()
        self._slicing_data()

    def _read_data(self):
        df_raw = pd.read_csv(str(os.path.join(self.data_dir, self.data_file)),
                             parse_dates=True, skipinitialspace=True, index_col=0)
        self.data = self.scaler.fit_transform(df_raw['target'].values.reshape(-1, 1))
        border1s = [0, int(0.7 * len(self.data)), int(0.85 * len(self.data))]
        border2s = [int(0.7 * len(self.data)), int(0.85 * len(self.data)), int(len(self.data))]

        border1 = border1s[self.type_flag]
        border2 = border2s[self.type_flag]

        self.data = self.data[border1:border2]
        self.indices = df_raw.index

    def _slicing_data(self):
        self.features = []
        self.targets = []
        self.target_datestamp = []

        for i in tqdm(range(0, len(self.data) - self.seq_len - self.pred_len + 1)):
            end_idx = i + self.seq_len
            self.features.append(self.data[i:end_idx])
            self.targets.append(self.data[end_idx:end_idx + self.pred_len])

            date_strings = self.indices[end_idx:end_idx + self.pred_len]
            self.target_datestamp.append(date_strings)

        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        self.target_datestamp = np.array(self.target_datestamp)

    def __len__(self):
        return len(self.features)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def __getitem__(self, index):
        # input_padding = torch.zeros_like(x_context)
        # freq = torch.tensor([self.freq_type], dtype=torch.long)
        if self.timesfm:
            x_context = torch.tensor(self.features[index], dtype=torch.float32)
            x_future = torch.tensor(self.targets[index], dtype=torch.float32)

            input_padding = torch.zeros_like(x_context)
            freq = torch.tensor([self.freq_type], dtype=torch.long)

            return x_context, input_padding, freq, x_future
        else:
            return self.features[index], self.targets[index]
