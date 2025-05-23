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
parser.add_argument('--model', type=str, default='RNN',
                    help='model name, options: [RNN, NBEATS]')
parser.add_argument('--rnn_model', type=str, default='LSTM',
                    help='RNN model names, options=[LSTM, GRU]')

# TimesFM parameters
parser.add_argument('--timesfm', type=bool, default=False, help='TimesFM specific datasets')
parser.add_argument('--freq_type', type=int, default=0)

# Prediction Task
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=24)
args = parser.parse_args()

args.data_dir = "./datasets"


def update_args_(params):
    dargs = vars(args)
    dargs.update(params)


def objective(trial):
    # params = {
    #     'hidden_size': trial.suggest_categorical('hidden_size', [128, 256]),
    #     'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
    # }
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'seq_len': trial.suggest_int('seq_len', 512, 1024, 128),
        'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64])
    }

    update_args_(params)

    exp = Exp_Main(args)
    _, train_loss, val_loss = exp.train()

    return val_loss[-1]


if __name__ == '__main__':
    evaluations = {"Station": [], "MAE": [], "RMSE": [], "Best Params": [], "Time Elapsed": []}
    # evaluations = {"Station": [], "MAE": [], "RMSE": [], "Time Elapsed": []}
    models = ['LSTM', 'GRU', 'NBEATS']

    for model in models:
        if model == 'NBEATS':
            args.model = model
            output_dir = "./results/{}".format(args.model)
        else:
            args.model = 'RNN'
            args.rnn_model = model
            output_dir = "./results/{}".format(args.rnn_model)

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

            study = optuna.create_study(study_name=station_name, direction='minimize')
            study.optimize(objective, n_trials=10)

            print(study.best_params)
            update_args_(study.best_params)

            exp = Exp_Main(args)
            tic = time.time()
            _, t_loss, v_loss = exp.train()
            toc = time.time()

            time_elapsed = (toc - tic) / 60

            loss_df = pd.DataFrame({
                "Train Loss": t_loss,
                "Validation Loss": v_loss,
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
            evaluations['Best Params'].append(study.best_params)
            evaluations['Time Elapsed'].append(time_elapsed)

        eval_df = pd.DataFrame(evaluations)
        eval_path = os.path.join(output_dir, "Evaluations.csv")
        eval_df.to_csv(eval_path, index=False)
