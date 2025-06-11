from datetime import timedelta

import pandas as pd
import numpy as np
import os

data_dir = './Methane'

for file in os.listdir(data_dir):
    if not file.endswith('.csv'):
        continue

    df = pd.read_csv(os.path.join(data_dir, file), parse_dates=True, index_col=0)
    start_date = df.index.min()
    end_date = df.index.max()

    hourly_resampled = df.resample('h').mean()
    indices = hourly_resampled.index
    grouped = hourly_resampled.groupby([indices.year, indices.month, indices.day]).sum()

    print(grouped[grouped['target']==np.nan])
    # daily_resampled = hourly_resampled.resample('D').mean().round(3)
    # filtered = daily_resampled[daily_resampled['target'] == np.nan]
    # print(daily_df.head())
    print(daily_resampled.head())