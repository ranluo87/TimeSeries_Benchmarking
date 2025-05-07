import torch
from torch import nn
from torch.nn import functional as F
from nbeats_pytorch.model import NBeatsNet


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.hidden_size = args.hidden_size

        self.nbeats = NBeatsNet(
            stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
            forecast_length=self.pred_len,
            backcast_length=self.seq_len,
            hidden_layer_units=self.hidden_size,
        )

    def forward(self, x, x_mark=None):
       _, forecast = self.nbeats(x)
       return forecast