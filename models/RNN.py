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

        self.rnns = nn.ModuleList()

        for i in range(self.num_layers):
            input_size = 1 if i == 0 else self.hidden_size
            if self.args.rnn_model == "LSTM":
                self.rnns.append(nn.LSTM(input_size, self.hidden_size, num_layers=1, batch_first=True))
            elif self.args.rnn_model == "GRU":
                self.rnns.append(nn.GRU(input_size, self.hidden_size, num_layers=1, batch_first=True))

        self.fc = nn.Linear(self.hidden_size, self.pred_len)

    def forward(self, x, x_mark=None):
        # x = x.squeeze(dim=2)  # x as in shape of [Batch_size, Seq_Len, num_Features] in this case num_features=1
        output = []
        hidden = []

        for i in range(self.num_layers):
            if i == 0:
                output, hidden = self.rnns[i](x)
            else:
                output, _ = self.rnns[i](output, hidden)

        output = output[:, -1, :]
        output = self.fc(output)

        return output