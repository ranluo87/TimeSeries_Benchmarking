import math

from exp.exp_main import Exp_Main

import argparse
import optuna
import numpy as np
import os

parser = argparse.ArgumentParser()

# Optimization
parser.add_argument('--epochs', type=int, default=10, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--hidden_size', type=int, default=256, help='prediction length')

# Model Parameters
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--model', type=str, default='RNN',
                    help='model name, options: [TimeMixer, TimesNet, iTransformer, RNN, MultiVar_RNN, T2V_Seq2Seq]')
parser.add_argument('--rnn_model', type=str, default='LSTM',
                    help='RNN model names, options=[LSTM, GRU]')

# Prediction Task
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--freq_type', type=int, default=0)

args = parser.parse_args()

args.data_dir = "C:\Python Projects\TimeSeries_Benchmarking\datasets\Select"
args.data_file = "Anzac.csv"

def update_args_(params):
    dargs = vars(args)
    dargs.update(params)

def objective(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [256]),
        'batch_size': trial.suggest_categorical('batch_size', [32]),
    }

    update_args_(params)

    exp = Exp_Main(args)
    _, train_loss, _ = exp.train()

    return np.average(train_loss)


if __name__ == '__main__':
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=1)
    #
    # print(study.best_params)
    #
    # update_args_(study.best_params)

    exp = Exp_Main(args)
    exp.train()
    test_df, mse, mae = exp.test()
    rmse = np.sqrt(mse)

    output_dir = "/results"

    station_name = args.data_file.split('.')[0]

    output_path = os.path.join(output_dir, "{} Test Fitting.csv".format(station_name))
    test_df.to_csv(output_path, index=False)
    print("{} best model achieving MAE: {} and RMSE: {}".format(station_name, mae, rmse))

    print("test")

