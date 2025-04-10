import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size

        if self.args.rnn_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif self.args.rnn_model == "GRU":
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
            )

        self.fc = nn.Linear(self.hidden_size, self.pred_len)

    def forward(self, x, x_mark=None):
        # x = x.squeeze(dim=2)  # x as in shape of [Batch_size, Seq_Len, num_Features] in this case num_features=1

        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return x