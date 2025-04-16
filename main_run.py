import math

import pandas as pd

from exp.exp_main import Exp_Main

import argparse
import optuna
import numpy as np
import os

parser = argparse.ArgumentParser()

# Optimization
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--hidden_size', type=int, default=256, help='prediction length')

# Model Parameters
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--model', type=str, default='RNN',
                    help='model name, options: [TimeMixer, TimesNet, iTransformer, RNN, MultiVar_RNN, T2V_Seq2Seq]')
parser.add_argument('--rnn_model', type=str, default='LSTM',
                    help='RNN model names, options=[LSTM, GRU]')
parser.add_argument('--timesfm', type=bool, default=False, help='TimesFM specific datasets')
# Prediction Task
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--freq_type', type=int, default=0)

args = parser.parse_args()

args.data_dir = "./datasets/select"


def update_args_(params):
    dargs = vars(args)
    dargs.update(params)


def objective(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256]),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'seq_len': trial.suggest_categorical('seq_len', [96, 168]),
    }

    update_args_(params)

    exp = Exp_Main(args)
    _, train_loss, val_loss = exp.train()

    return val_loss[-1]


if __name__ == '__main__':
    evaluations = {"Station": [], "MAE": [], "RMSE": []}

    output_dir = "/home/ran/Desktop/PycharmProjects/TimeSeries_Benchmarking/results/{}".format(args.rnn_model)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(args.data_dir):
        if not file.endswith(".csv"):
            continue

        args.data_file = file
        station_name = file.split(".")[0]

        study = optuna.create_study(study_name=station_name, direction='minimize')
        study.optimize(objective, n_trials=10)

        print(study.best_params)
        update_args_(study.best_params)

        exp = Exp_Main(args)
        _, t_loss, v_loss = exp.train()

        loss_df = pd.DataFrame({
            "Train Loss": t_loss,
            "Validation Loss": v_loss,
        })

        loss_path = os.path.join(output_dir, "{} Train Loss.csv".format(station_name))
        loss_df.to_csv(loss_path, index=False)

        test_df, mae, rmse = exp.test()
        test_path = os.path.join(output_dir, "{} Test Fitting.csv".format(station_name))
        test_df.to_csv(test_path, index=False)

        evaluations["Station"].append(station_name)
        evaluations["MAE"].append(mae)
        evaluations["RMSE"].append(rmse)

    eval_df = pd.DataFrame(evaluations)
    eval_path = os.path.join(output_dir, "{} Evaluations.csv".format(args.rnn_model))
    eval_df.to_csv(eval_path, index=False)