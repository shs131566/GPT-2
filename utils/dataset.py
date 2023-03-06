from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional, Union
import torch
import regex as re

_CHINESE_CHAR_RANGE = ('\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df'
                       '\U0002a700-\U0002b73f\U0002b740-\U0002b81f'
                       '\U0002b820-\U0002ceaf\uf900-\ufaff'
                       '\U0002f800-\U0002fa1f')
_PUNCTUATION_RANGE = '\\p{P}\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e'


class GPTDataset(Dataset):
    def __init__(self, vocab_path: str,
                 corpus_path: str,
                 max_word_len: int = 100,
                 seq_len: int = 64,
                 repeat: bool = True,
                 unk_token: str = '<unk>',
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_token: str = '<pad>'):
        super().__init__()

        self.corpus_fp = open(corpus_path, 'r', encoding='utf-8')
        self.seq_len = seq_len
        self.repeat = repeat

        ### vocabulary
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.lines = self.corpus_fp.readlines()
        self.lines = [line for line in self.lines if len(line) + 2 > self.seq_len]
    
        with open(vocab_path, 'r', encoding='utf-8') as fp:
            self.additional_tokens = [bos_token, eos_token, pad_token]

            # The additional tokens would be inserted before the words.
            self.words = self.additional_tokens + fp.read().split()
            self.vocab = {word: i for i, word in enumerate(self.words)}

        self.vocab_len = (len(self.words) + 7) // 8 * 8 #! Transformer embedding dims
        
        ### tokenizer
        self.exclude_tokens = [self.unk_token] + self.additional_tokens
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        return len(self.lines)
    
    def __contains__(self, token: str) -> bool:
        return token in self.words

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        line = self.lines[idx]
        # Use token indices rather than the token names directly.
        # indices = [self._vocab(t) for t in line.split()]
        indices = [self._vocab(t) for t in self.encode(line)]


        # Decorate the sequence with additional tokens.
        indices = [self.bos_idx] + indices + [self.eos_idx]
        indices += [self.pad_idx] * (self.seq_len - len(indices) + 1)

        indices = indices[:self.seq_len+1]
        
        data = {'input': indices[:-1], 'output': indices[1:]}

        return {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}
        # return {'input': torch.Tensor(indices[:-1]).long(), 'output': torch.Tensor(indices[1:]).long()}
        # return {'input': torch.Tensor(indices[:-1]), 'output': torch.Tensor(indices[1:])}

    def _vocab(self, idx_or_token:Union[str, int]):
        if isinstance(idx_or_token, str):
            return self.vocab[idx_or_token]
        else:
            return self.words[idx_or_token]

    def encode(self, text: str) -> List[str]:
        return [token
                for normalized in self._normalize(text)
                for token in self._tokenize(normalized)]

    def decode(self, tokens: List[str]) -> str:
        return (' '.join(tokens).replace(' ##', '')
                                .replace(' .', '.')
                                .replace(' ?', '?')
                                .replace(' !', '!')
                                .replace(' ,', ',')
                                .replace(' \' ', '\'')
                                .replace(' \" ', '\"')
                                .replace('\'\'', '\' \'')
                                .replace('\"\"', "\" \""))

    def _normalize(self, text: str) -> List[str]:
        # Normalize whitespace characters and remove control characters.
        #text = ' '.join(re.sub('[\x00\uFFFD\\p{C}]', '', t)
                        #for t in text.split())
        text = re.sub('[\x00\uFFFD\\p{C}]', '', text)
                       
        # Insert whitespaces between chinese characters.
        text = re.sub(f'([{_CHINESE_CHAR_RANGE}])', r' \1 ', text)
        normalized = []
        for t in text.split():
            if t in self.exclude_tokens:
                normalized.append(t)
            else:
                # Prevent from treating tokens with punctuations.
                normalized += re.split(f'([{_PUNCTUATION_RANGE}])', t.lower())
        return ' '.join(normalized).split()

    def _tokenize(self, text: str) -> List[str]:
        subwords = []
        for token in text.split():
            if len(token) > self.max_word_len:
                subwords.append(self.unk_token)
                continue

            children = []
            while token and token != '##':
                current, token = token, ''
                while current and current != '##':
                    # If subword is in vocabulary, add to list and re-calibrate
                    # the target token.
                    if current in self.vocab:
                        children.append(current)
                        token = '##' + token
                        break

                    # If subword is not in vocabulary, reduce the search range
                    # and test it again.
                    current, token = current[:-1], current[-1] + token

                # Process current token as `unknown` since there is no any
                # proper tokenization (in greedy).
                if not current:
                    children, token = None, None
            subwords += children or [self.unk_token]

        return subwords
    
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
