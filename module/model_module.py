import pytorch_lightning as pl
import torch
import torch.nn as nn
from model import Transformer
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path

class GPTModelModule(pl.LightningModule):
    def __init__(self, layers, pad_idx, vocab_len, result_path, exp_name, exp_version):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')

        self.model = Transformer(layers=layers, pad_idx=pad_idx,
                           words=vocab_len, seq_len=64,
                           heads=4, dims=512, rate=4,
                           dropout=0.01, bidirectional=False)
        
        # TODO: 각종 설정 넣기
        self.use_grad_ckpt = True
        self.total_steps = 10
        self.lr = 0.001
        self.result_path = result_path
        self.exp_name = exp_name
        self.exp_version = exp_version
        
    def training_step(self, batch, batch_idx):
        logits = self.model(batch['input'], use_grad_ckpt=self.use_grad_ckpt) 
        loss = self.criterion(logits.transpose(1, 2), batch['output'])
        self.log_dict({"train_loss":loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idex=0):
        # batch['input'], batch['output'] = batch['input'].to(self.device), batch['output'].to(self.device)
        logits, _ = self.model(batch['input'], past=None) 
        loss = self.criterion(logits.transpose(1, 2), batch['output'])
        self.log_dict({"val_metric":loss}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / self.total_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]
    
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.result_path) / self.exp_name / self.exp_version
        self.model.save_pretrained(save_path)
        # self.model.decoder.tokenizer.save_pretrained(save_path)
