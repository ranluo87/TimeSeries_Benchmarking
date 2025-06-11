import math

import pandas as pd
from exp.exp_main import Exp_Main

import argparse
import optuna
import numpy as np
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time

parser = argparse.ArgumentParser()

# Optimization
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')

# Model Parameters
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--model', type=str, default='iTransformer',
                    help='model name, options: [RNN, NBEATS, iTransformer]')
parser.add_argument('--rnn_model', type=str, default='LSTM',
                    help='RNN model names, options=[LSTM, GRU]')

# Model Define for iTransformer
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# Model Define for TimesNet
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=1, help='output size')

# TimesFM parameters
parser.add_argument('--timesfm', type=bool, default=False, help='TimesFM specific datasets')
parser.add_argument('--freq_type', type=int, default=0)

# Prediction Task
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--pred_method', type=str, default='static',
                    help='prediction method, options: [static, rolling]')
parser.add_argument('--pred_len', type=int, default=24)
args = parser.parse_args()

args.data_dir = "./datasets/select"


def update_args_(params):
    dargs = vars(args)
    dargs.update(params)


def objective(trial):
    # params = {
    #     'hidden_size': trial.suggest_categorical('hidden_size', [128, 256]),
    #     'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
    # }
    params = {
        # 'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'seq_len': trial.suggest_int('seq_len', 256, 1024, 128),
        'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64])
    }

    update_args_(params)

    trial_exp = Exp_Main(args)
    _, t_loss, v_loss = trial_exp.train()

    return t_loss[-1], v_loss[-1]


if __name__ == '__main__':
    evaluations = {"Station": [], "MAE": [], "RMSE": [], "Best Params": [], "Time Elapsed": []}
    # evaluations = {"Station": [], "MAE": [], "RMSE": [], "Time Elapsed": []}
    # models = ['LSTM', 'GRU', 'NBEATS']
    models = ['iTransformer', 'NBEATS', 'LSTM', 'GRU']
    for model in models:
        if model == 'RNN':
            args.model = 'RNN'
            args.rnn_model = model
            output_dir = "./results/{}".format(args.rnn_model)
        else:
            args.model = model
            output_dir = "./results/{}".format(args.model)

        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(args.data_dir):
            if not file.endswith(".csv"):
                continue

            args.data_file = file
            station_name = file.split(".")[0]
            if args.model == 'RNN':
                print(f"Modeling {station_name} with {args.rnn_model}")
            else:
                print(f"Modeling {station_name} with {args.model}")

            study = optuna.create_study(study_name=station_name, directions=['minimize', 'minimize'])
            study.optimize(objective, n_trials=10, gc_after_trial=True)

            best_trial = study.trials[0]

            for trail in study.trials:
                train_loss = trail.values[0]
                val_loss = trail.values[1]

                if train_loss < val_loss < best_trial.values[1]:
                    best_trial = trail

            print(best_trial.params)
            update_args_(best_trial.params)

            exp = Exp_Main(args)
            tic = time.time()
            _, train_loss, val_loss = exp.train()
            toc = time.time()

            time_elapsed = (toc - tic) / 60

            loss_df = pd.DataFrame({
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
            })

            # loss_df.to_csv(loss_path, index=False)
            loss_path = os.path.join(output_dir, "{}_Loss.html".format(station_name))
            loss_fig = go.Figure()
            loss_fig.add_trace(
                go.Scatter(x=loss_df.index, y=loss_df['Train Loss'], mode='lines', name='Train Loss'),
            )
            loss_fig.add_trace(
                go.Scatter(x=loss_df.index, y=loss_df['Validation Loss'], mode='lines', name='Validation Loss'),
            )
            loss_fig.write_html(loss_path)

            test_df, mae, rmse = exp.test()

            test_path = os.path.join(output_dir, "{} Test Fitting.csv".format(station_name))
            test_df.to_csv(test_path, index=False)

            fitting_fig = go.Figure()

            fitting_fig.add_trace(
                go.Scatter(x=test_df['date'], y=test_df['true'], mode='lines', name='True'),
            )

            fitting_fig.add_trace(
                go.Scatter(x=test_df['date'], y=test_df['pred'], mode='lines', name='Pred'),
            )

            fitting_path = os.path.join(output_dir, "{}_Fitting.html".format(station_name))
            fitting_fig.write_html(fitting_path)

            evaluations["Station"].append(station_name)
            evaluations["MAE"].append(mae)
            evaluations["RMSE"].append(rmse)
            evaluations['Best Params'].append(best_trial.params)
            evaluations['Time Elapsed'].append(time_elapsed)

        eval_df = pd.DataFrame(evaluations)
        eval_path = os.path.join(output_dir, "Evaluations.csv")
        eval_df.to_csv(eval_path, index=False)
