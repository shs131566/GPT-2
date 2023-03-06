import pytorch_lightning as pl
import torch
import torch.nn as nn
from model import Transformer

class GPTModelModule(pl.LightningModule):
    def __init__(self, layers, pad_idx, vocab_len):
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
        
    def training_step(self, batch, batch_idx):
        print('#############',batch['input'].shape)
        # batch['input'], batch['output'] = batch['input'].to(self.device), batch['output'].to(self.device)
        logits = self.model(batch['input'], use_grad_ckpt=self.use_grad_ckpt) 
        loss = self.criterion(logits.transpose(1, 2), batch['output'])
        return loss

    def validation_step(self, batch, batch_idx, dataset_idex=0):
        # batch['input'], batch['output'] = batch['input'].to(self.device), batch['output'].to(self.device)
        logits, _ = self.model(batch['input'], past=None) 
        loss = self.criterion(logits.transpose(1, 2), batch['output'])
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / self.total_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]


from typing import Union


class Vocab(object):
    def __init__(self,
                 vocab_path: str,
                 unk_token: str = '<unk>',
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_token: str = '<pad>'):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        with open(vocab_path, 'r', encoding='utf-8') as fp:
            self.additional_tokens = [bos_token, eos_token, pad_token]

            # The additional tokens would be inserted before the words.
            self.words = self.additional_tokens + fp.read().split()
            self.vocab = {word: i for i, word in enumerate(self.words)}

    def __getitem__(self, idx_or_token: Union[int, str]) -> Union[str, int]:
        if isinstance(idx_or_token, str):
            return self.vocab[idx_or_token]
        else:
            return self.words[idx_or_token]

    def __contains__(self, token: str) -> bool:
        return token in self.words

    def __len__(self) -> int:
        # Note that vocabulary size must be a multiple of 8 although the actual
        # number of words is less than it.
        return (len(self.words) + 7) // 8 * 8

    @property
    def unk_idx(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def bos_idx(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def pad_idx(self) -> int:
        return self.vocab[self.pad_token]
