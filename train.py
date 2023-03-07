"""
GPT2
Copyright (c) 2023-present KB Kookmin Bank Corp.
MIT License
"""
import pytorch_lightning as pl
from module import GPTModelModule, GPTDataModule
from utils import GPTDataset
from torch.utils.data import random_split
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pathlib import Path
import torch


def train():
    seed = 42
    pl.utilities.seed.seed_everything(seed, workers=True)
    
    #!config
    vocab_path = '/home/ubuntu/WORKSPACE/GPT2-lightning/vocab-ko-302M.txt'
    corpus_path = '/home/ubuntu/WORKSPACE/GPT2-lightning/data-toy.txt'
    seq_len = 64
    batch_size = 1
    result_path = 'result'
    exp_name = 'Test'
    exp_version = '230307'
    resume_from_checkpoint_path = None
    num_nodes = 1
    max_steps = 100
    val_check_interval = 16
    gradient_clip_val = 5
    layers = 1
    #!

    # tokenizer = tokenizer #TODO
    dataset = GPTDataset(vocab_path=vocab_path, 
                         corpus_path=corpus_path,
                         seq_len=seq_len)
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.2)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    model_module = GPTModelModule(layers=layers, 
                                  pad_idx=dataset.pad_idx,
                                  vocab_len=dataset.vocab_len,
                                  result_path=result_path,
                                  exp_name=exp_name,
                                  exp_version=exp_version)
    data_module = GPTDataModule(train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                batch_size=batch_size)
    
    logger = TensorBoardLogger(
        save_dir=result_path,
        name=exp_name,
        version=exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(result_path) / exp_name / exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    trainer = pl.Trainer(
        resume_from_checkpoint= resume_from_checkpoint_path,
        num_nodes=num_nodes,
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        strategy='dp',
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
        precision=16,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)

if __name__ == "__main__":
    train()