from typing import Optional

from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

import torch
from torch.utils.data import Dataset

from dataloader.dataloader import UnivariateMethaneHourly
from os import path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--pred_len', type=int, default=128)
parser.add_argument('--freq_type', type=int, default=0)
args = parser.parse_args()

args.data_dir = "C:\Python Projects\TimeSeries_Benchmarking\datasets\Select"
args.data_file = "Anzac.csv"


def get_model(load_weights: bool = False):
    repo_id = "google/timesfm-2.0-500m-pytorch"

    hparams = TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=24,
        num_layers=50,
        use_positional_embedding=False,
        context_len=96,
    )

    tfm = TimesFm(
        hparams=hparams,
        checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id)
    )

    model = PatchedTimeSeriesDecoder(tfm._model_config)

    if load_weights:
        checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(loaded_checkpoint)

    return model, hparams, tfm._model_config

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    model, hparams, tfm = get_model(load_weights=False)
    config = FinetuningConfig(
        batch_size=128,
        num_epochs=100,
        learning_rate=1e-4,
        use_wandb=False,
        freq_type=0,
        log_every_n_steps=10,
        val_check_interval=0.5,
        use_quantile_loss=False
    )

    train_dataset = UnivariateMethaneHourly(args, "train")
    val_dataset = UnivariateMethaneHourly(args, "val")

    finetuner = TimesFMFinetuner(model, config)

    print("\nStarting finetuning...")
    results = finetuner.finetune(train_dataset=train_dataset,
                                 val_dataset=val_dataset)

    print("\nFinetuning completed!")
    print(f"Training history: {len(results['history']['train_loss'])} epochs")

