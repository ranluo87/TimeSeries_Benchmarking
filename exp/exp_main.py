from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from dataloader.dataloader import UnivariateMethaneHourly


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.model = None
        self.args = args

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _get_data(self, flag):
        dataset = UnivariateMethaneHourly(self.args, flag)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        return dataset, dataloader

    def train(self):
        train_dataset, train_dataloader = self._get_data('train')
        valid_dataset, valid_dataloader = self._get_data('valid')

        optimizer = self._select_optimizer()

        train_losses = []
        validation_losses = []

        pbar = tqdm(range(self.args.epochs))

        for epoch in pbar:
            self.model.train()
            epoch_loss = []

            for i, (batch_x, batch_y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)


