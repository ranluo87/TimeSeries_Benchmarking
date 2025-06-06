import os
import torch
import torch.nn as nn


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.args.device = self.device = self._acquire_device()

        self.model = self._build_model()
        # if self.use_gpu and torch.cuda.device_count() > 1:
        #     print('Model was parallelled on ', torch.cuda.device_count(), ' GPUs')
        #     self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        self.use_gpu = True if torch.cuda.is_available() else False

        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
            device = torch.device('cuda:{}'.format(1))
            print('Use GPU: {}'.format(device))
        else:
            device = torch.device('cpu')
            print('Use CPU')

        return device

    def _get_data(self, flag):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass
