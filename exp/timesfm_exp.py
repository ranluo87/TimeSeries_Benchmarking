from typing import Optional, Tuple
from os import path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from dataloader.dataloader import UnivariateMethaneHourly

from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
import argparse

data_dir = '/home/ran/Desktop/PycharmProjects/TimeSeries_Benchmarking/datasets/select'
data_file = 'Anzac.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--data_file', type=str, default=data_file)
# TimesFM configurations
parser.add_argument('--timesfm', type=bool, default=True)
parser.add_argument('--freq_type', type=int, default=0)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=24)
# Optimization Hyperparams
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-4)

args = parser.parse_args()

train_dataset = UnivariateMethaneHourly(args, flag='train')
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

val_dataset = UnivariateMethaneHourly(args, flag='val')
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

hparams = TimesFmHparams(
    backend='gpu',
    per_core_batch_size=32,
    num_layers=50,
    horizon_len=24,
    context_len=512,
    use_positional_embedding=False,
    output_patch_len=128
)

repo_id = "google/timesfm-2.0-500m-pytorch"
tfm = TimesFm(
    hparams=hparams,
    checkpoint=TimesFmCheckpoint(
        huggingface_repo_id=repo_id
    )
)

model = PatchedTimeSeriesDecoder(tfm._model_config)

checkpoint_path = path.join(snapshot_download(repo_id), 'torch_model.ckpt')
loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(loaded_checkpoint)

config = FinetuningConfig(
    batch_size=args.batch_size,
    num_epochs=args.epochs,
    learning_rate=args.learning_rate,
    use_wandb=False,
    freq_type=args.freq_type,
    log_every_n_steps=10,
    val_check_interval=0.5,
    use_quantile_loss=True
)

finetuner = TimesFMFinetuner(model, config)
print("\nStarting fine-tuning...")
results = finetuner.finetune(train_dataset=train_dataset, val_dataset=val_dataset)
print("\nFinished fine-tuning.")

print("test")