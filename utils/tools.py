import math
import numpy as np


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch + 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == 'cosine':
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.epochs * math.pi))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updated learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print('Validation loss decreased {:.3f} --> {:.3f}'.format(self.val_loss_min, val_loss))
                self.val_loss_min = val_loss
            self.counter = 0


    # # optimization
    # parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    # parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    # # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    # # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    #
    # # forecasting task
    # parser.add_argument('--seq_len', type=int, default=30, help='input sequence length')
    # parser.add_argument('--pred_len', type=int, default=7, help='prediction sequence length')
    # # parser.add_argument('--freq', type=str, default='d', help="Frequency of time series data")
    # # parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # # parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    # # for multivariate model training and prediciton
    # # parser.add_argument('--target_col', type=str, default='Methane', help="Column name for the target emission")
    #
    # # GPU
    # parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu')
    #
    # # model define
    # parser.add_argument('--model', type=str, default='T2V_Seq2Seq',
    #                     help='model name, options: [TimeMixer, TimesNet, iTransformer, RNN, MultiVar_RNN, T2V_Seq2Seq]')
    # parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    # parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    # parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    # parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    # parser.add_argument('--c_out', type=int, default=1, help='output size')
    # parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    # parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    # parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    # parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # parser.add_argument('--features', type=str, default='S',
    #                     help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate '
    #                          'predict univariate, MS:multivariate predict univariate')
    #
    # # iTransformer Special
    # parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    #
    # # TimeMixer Special
    # parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window')
    # parser.add_argument('--down_sampling_layers', type=int, default=1, help='down sampling layers')
    # parser.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method')
    # parser.add_argument('--decomp_method', type=str, default='moving_avg',
    #                     help='method of series decompsition, only support moving_avg or dft_decomp')
    # parser.add_argument('--moving_avg', type=int, default=21, help='window size of moving average')
    # # RNN Special
    # parser.add_argument('--n_layers', type=int, default=2, help='num of recurrent layers')
    # parser.add_argument('--rnn_model', type=str, default='GRU',
    #                     help='RNN model names, options=[LSTM, GRU]')