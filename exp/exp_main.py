import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from exp.exp_basic import Exp_Basic
from dataloader.dataloader import UnivariateMethaneHourly
from models import RNN, NBEATS, iTransformer, TimesNet

import warnings

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args = args

    def _build_model(self):
        model_dict = {
            'RNN': RNN,
            'NBEATS': NBEATS,
            'iTransformer': iTransformer,
            'TimesNet': TimesNet
        }

        model = model_dict[self.args.model].Model(self.args).float()
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _get_data(self, flag):
        dataset = UnivariateMethaneHourly(self.args, flag)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        return dataset, dataloader

    def train(self):
        train_dataset, train_dataloader = self._get_data('train')
        valid_dataset, valid_dataloader = self._get_data('val')

        optimizer = self._select_optimizer()

        train_loss = []
        validation_loss = []

        pbar = tqdm(range(self.args.epochs))

        # from torchinfo import summary
        # summary(self.model, input_size=(self.args.batch_size, self.args.seq_len, 1))

        for epoch in pbar:
            self.model.train(True)
            epoch_loss = []

            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().squeeze().to(self.device)

                if self.args.model != 'RNN' or self.args.model != 'NBEATS':
                    batch_x_mark = batch_x_mark.float().to(self.device)
                else:
                    batch_x_mark = None

                output_y = self.model(batch_x, batch_x_mark)
                step_loss = self.criterion(output_y, batch_y)

                epoch_loss.append(step_loss.item())
                step_loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = []

            with torch.no_grad():
                for i, (batch_x_val, batch_y_val, batch_x_val_mark, _) in enumerate(valid_dataloader):
                    batch_x_val = batch_x_val.float().to(self.device)
                    batch_y_val = batch_y_val.float().squeeze().to(self.device)

                    if self.args.model != 'RNN' or self.args.model != 'NBEATS':
                        batch_x_val_mark = batch_x_val_mark.float().to(self.device)
                    else:
                        batch_x_val_mark = None

                    output_y_val = self.model(batch_x_val, batch_x_val_mark)

                    val_step_loss = self.criterion(output_y_val, batch_y_val)
                    epoch_val_loss.append(val_step_loss.item())

            pbar.set_postfix({
                "Epoch": epoch + 1,
                "Training Loss": np.average(epoch_loss),
                "Validation Loss": np.average(epoch_val_loss)
            })

            train_loss.append(np.average(epoch_loss))
            validation_loss.append(np.average(epoch_val_loss))

        train_loss = np.array(train_loss)
        validation_loss = np.array(validation_loss)

        return self.model, train_loss, validation_loss

    def test(self):
        test_dataset, test_dataloader = self._get_data('test')

        preds = []
        trues = []

        with torch.no_grad():
            self.model.eval()
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(test_dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().squeeze().to(self.device)

                if self.args.model != 'RNN' or self.args.model != 'NBEATS':
                    batch_x_mark = batch_x_mark.float().to(self.device)
                else:
                    batch_x_mark = None

                outputs = self.model(batch_x, batch_x_mark)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # [batch_size, pred_len]
                pred = test_dataset.inverse_transform(outputs)
                true = test_dataset.inverse_transform(batch_y)

                preds.append(pred[:, -1])
                trues.append(true[:, -1])

        preds = np.hstack(preds)
        trues = np.hstack(trues)

        test_df = pd.DataFrame({
            'date': np.array(test_dataset.target_datestamp[:, -1]),
            'pred': preds,
            'true': trues,
        })

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)

        rmse = np.sqrt(mse)

        return test_df, mae, rmse
