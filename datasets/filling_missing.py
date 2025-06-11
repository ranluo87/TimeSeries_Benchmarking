import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

data_dir = './Methane'

for file in os.listdir(data_dir):
    if not file.endswith(".csv"):
        continue

    methane_df = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True)
    methane_df['time_diff'] = methane_df.index.diff()
    gap_end = methane_df[methane_df['time_diff'] == methane_df['time_diff'].max()]
    gap_end_date = gap_end.index[0]
    date_idx = methane_df.index.get_loc(gap_end_date)
    gap_start = methane_df.iloc[[date_idx - 1]]
    gap_start_date = gap_start.index[0]

    print("Gaping between {} and {} with {}".format(gap_start_date, gap_end_date, gap_end['time_diff']))

    date_min = methane_df.index.min()
    date_max = methane_df.index.max()

    filled_df = methane_df.reindex(pd.date_range(date_min, date_max, freq='h'), fill_value=np.nan)
    filled_df['target'] = filled_df['target'].interpolate(method='linear')

    if int(len(filled_df[gap_start_date:gap_end_date])) > 24 * 100:

        first = filled_df[date_min:gap_start_date]
        middle = filled_df[gap_start_date:gap_end_date]
        last = filled_df[gap_end_date:date_max]

        if len(first) > len(last):
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=first.index, y=first['target']))
        fig.add_trace(go.Scatter(x=middle.index, y=middle['target']))
        fig.add_trace(go.Scatter(x=last.index, y=last['target']))
        fig.update_layout(title='{}'.format(file.split('.')[0]))
        fig.show()

        filled_df = filled_df[gap_end_date:date_max]

    filled_df.drop(columns=['time_diff'], inplace=True)
    filled_df.to_csv('./CH4_H/{}'.format(file))


    # first = filled_df[date_min:gap_start_date]
    # middle = filled_df[gap_start_date:gap_end_date]
    # last = filled_df[gap_end_date:date_max]
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=first.index, y=first['target']))
    # fig.add_trace(go.Scatter(x=middle.index, y=middle['target']))
    # fig.add_trace(go.Scatter(x=last.index, y=last['target']))
    # fig.update_layout(title='{}'.format(file.split('.')[0]))
    # fig.show()

