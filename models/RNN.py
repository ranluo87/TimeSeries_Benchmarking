import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        # self.seq_len = args.seq_len
        self.pred_len = args.pred_len if args.pred_method == 'static' else 1
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size

        # self.rnns = nn.ModuleList()
        #
        # for i in range(self.num_layers):
        #     input_size = self.seq_len if i == 0 else self.hidden_size
        #     if self.args.rnn_model == "LSTM":
        #         self.rnns.append(nn.LSTM(input_size, self.hidden_size, num_layers=1, batch_first=False))
        #     elif self.args.rnn_model == "GRU":
        #         self.rnns.append(nn.GRU(input_size, self.hidden_size, num_layers=1, batch_first=False))

        if self.args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )
        elif self.args.rnn_model == 'GRU':
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )

        self.fc = nn.Linear(self.hidden_size, self.pred_len)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in range(self.num_layers):
            for weight in self.rnn._all_weights[layer]:
                if 'weight_hh' in weight:
                    nn.init.orthogonal_(getattr(self.rnn, weight))
                elif 'weight_ih' in weight:
                    nn.init.xavier_uniform_(getattr(self.rnn, weight))
                elif 'bias' in weight:
                    nn.init.zeros_(getattr(self.rnn, weight))

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, x_mark=None):
        # x = x.squeeze(dim=2)  # x as in shape of [Batch_size, Seq_Len, num_Features] in this case num_features=1

        # h_0 = Variable(torch.zeros(self.num_layers, self.args.batch_size).to(x.device))
        # c_0 = Variable(torch.zeros(self.num_layers, self.args.batch_size).to(x.device))

        x, _ = self.rnn(x)
        return self.fc(x[:, -1, :])
