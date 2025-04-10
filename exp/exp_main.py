import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from exp.exp_basic import Exp_Basic
from dataloader.dataloader import UnivariateMethaneHourly
from models import RNN

import warnings
warnings.filterwarnings("ignore")

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args = args

    def _build_model(self):
        model_dict = {
            "RNN": RNN,
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

        for epoch in pbar:
            self.model.train()
            epoch_loss = []

            for i, (batch_x, batch_y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().squeeze().to(self.device)

                output_y = self.model(batch_x)
                step_loss = self.criterion(output_y, batch_y)

                epoch_loss.append(step_loss.item())
                step_loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = []

            with torch.no_grad():
                for i, (batch_x_val, batch_y_val) in enumerate(valid_dataloader):
                    batch_x_val = batch_x_val.float().to(self.device)
                    batch_y_val = batch_y_val.float().squeeze().to(self.device)

                    output_y_val = self.model(batch_x_val)
                    val_step_loss = self.criterion(output_y_val, batch_y_val)
                    epoch_val_loss.append(val_step_loss.item())

            pbar.set_postfix({
                "Epoch": epoch+1,
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
        indices = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().squeeze().to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = test_dataset.inverse_transform(outputs.squeeze())
                true = test_dataset.inverse_transform(batch_y)

                preds.append(pred.flatten())
                trues.append(true.flatten())
                indices.append(test_dataset.target_datestamp[i])

        preds = np.hstack(preds)
        trues = np.hstack(trues)
        indices = np.hstack(indices)

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)

        test_df = pd.DataFrame({
            'datestamp': indices,
            'pred': preds,
            'true': trues,
        })

        return test_df, mse, mae,

