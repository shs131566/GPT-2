import pytorch_lightning as pl
import torch
import torch.nn as nn
"""
        self.vocab = Vocab(vocab_path=self.vocab_path)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx,
                                             reduction='mean')
"""


class ModelModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO: model 정의 
        # TODO: vocab 정의
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx, reduction='mean') # TODO : 모델 안에 넣기
    
    def training_step(self, batch, batch_idx):
        loss = 0
        return loss

    def validation_step(self, batch, batch_idx, dataset_idex=0):
        # TODO: model에서 infer 후 pred 와 answer 비교
        scores = 0
        return scores

    def configure_optimizers(self):

        max_iter = None
        max_iter = min(self.max_steps, max_iter) if max_iter is not None else self.max_steps 

        assert max_iter is not None
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / self.total_steps), # TODO : maxiter?
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
