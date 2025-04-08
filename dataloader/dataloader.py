import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


class UnivariateMethaneHourly(Dataset):
    def __init__(self, args, flag="train"):
        self.data_path = str(args.root_path)
        self.data_file = args.file

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.freq = args.freq
        self.scaler = MinMaxScaler()

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type_flag = type_map[flag]

        self._read_data()

    def _read_data(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.data_file),
                             parse_dates=True, skipinitialspace=True)
        dates = df_raw.pop('date').values
        date_feature_stamp = time_features(pd.to_datetime(dates), freq=self.freq).transpose(1, 0)

        feature_values = np.round(df_raw["ch4"].to_numpy().squeeze(), 3)
        feature_values = self.scaler.fit_transform(feature_values.reshape(-1, 1))

        border1s = [0, int(0.7 * len(df_raw)), int(0.85 * len(df_raw))]
        border2s = [int(0.7 * len(df_raw)), int(0.85 * len(df_raw)), int(len(df_raw))]

        border1 = border1s[self.type_flag]
        border2 = border2s[self.type_flag]

        self.features, self.target, self.date_stamp, self.orig_dates = (
                self.rolling_window(feature_values[border1:border2],
                                    dates[border1:border2],
                                    date_feature_stamp[border1:border2]))

    def __len__(self):
        return len(self.features)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def __getitem__(self, index):
        return self.features[index], self.target[index], self.date_stamp[index], self.orig_dates[index]

    def rolling_window(self, features_scaled, dates, data_stamp):
        x = []
        y = []
        date_index = []
        orig_dates = []
        for i in tqdm(range(0, len(features_scaled) - self.seq_len - self.pred_len, self.pred_len)):
            x.append(features_scaled[i:i + self.seq_len])
            y.append(features_scaled[i + self.seq_len:i + self.seq_len + self.pred_len])

            date_index.append(data_stamp[i:i + self.seq_len])
            date_string = dates[i + self.seq_len:i + self.seq_len + self.pred_len][0]
            orig_dates.append(date_string)

        time.sleep(0.5)
        return np.array(x), np.array(y), np.array(date_index), np.array(orig_dates)


class MultivariateDaily(Dataset):
    def __init__(self, args, flag="train"):
        self.data_path = str(args.root_path)
        self.data_file = args.file
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.target_col = args.target_col
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.station = args.file[:-4]
        self.freq = args.freq
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type_flag = type_map[flag]

        self._read_data()

    def _read_data(self):
        df_raw = pd.read_csv(str(os.path.join(self.data_path, self.data_file)), skipinitialspace=True)

        date_index = pd.to_datetime(dict(year=df_raw.iloc[:, 0], month=df_raw.iloc[:, 1], day=df_raw.iloc[:, 2]))

        df_raw = df_raw[[col for col in df_raw.columns if 'Unnamed' not in col]]
        df_raw.index = date_index

        self.features_cols = [col.split(':')[0] for col in df_raw.columns]
        self.units_cols = [col.split(':')[1] for col in df_raw.columns]
        df_raw.columns = self.features_cols

        try:
            assert self.target_col in self.features_cols
        except AssertionError:
            raise AssertionError("{} does not have the target emission {}".format(self.station, self.target_col))

        self.features_cols = [col for col in self.features_cols if col != self.target_col]

        border1s = [0, int(0.7 * len(df_raw)), int(0.85 * len(df_raw))]
        border2s = [int(0.7 * len(df_raw)), int(0.85 * len(df_raw)), len(df_raw)]

        border1 = border1s[self.type_flag]
        border2 = border2s[self.type_flag]

        df = df_raw[border1:border2]
        df = df.fillna(0)
        targets = df[self.target_col].values.reshape(-1, 1)
        df = df[self.features_cols]
        # Check target value
        try:
            assert not all(np.isnan(targets))
        except AssertionError:
            raise AssertionError("{}'s target {} dataset are all NaN".format(self.station, self.target_col))

        targets_temp = self.target_scaler.fit_transform(targets)
        features_temp = self.feature_scaler.fit_transform(df.values)

        self.features, self.targets, self.feature_stamp, self.target_stamp, self.orig_date \
            = self.rolling_window(features_temp, targets_temp, df.index)

    def rolling_window(self, features_scaled, targets_scaled, dates):
        date_feature_stamps = time_features(dates, freq=self.freq).transpose(1, 0)

        x = []
        y = []
        x_stamps = []
        y_stamps = []
        orig_dates = []
        for i in tqdm(range(0, len(features_scaled) - self.seq_len - self.pred_len, self.pred_len)):
            x.append(features_scaled[i:i + self.seq_len])
            y.append(targets_scaled[i + self.seq_len:i + self.seq_len + self.pred_len])

            x_stamps.append(date_feature_stamps[i:i + self.seq_len])
            y_stamps.append(date_feature_stamps[i + self.seq_len:i + self.seq_len + self.pred_len])

            date_string = dates[i + self.seq_len:i + self.seq_len + self.pred_len]
            orig_dates.append(date_string)

        time.sleep(0.5)
        return np.array(x), np.array(y), np.array(x_stamps), np.array(y_stamps), np.array(orig_dates)

    def __getitem__(self, index):
        return self.features[index], self.targets[index], self.feature_stamp[index], self.target_stamp[index]

    def __len__(self):
        return len(self.features)

    def features_count(self):
        return len(self.features_cols)

    def target_inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)





