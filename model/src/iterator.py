import os
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, BatchEncoding

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class KlueStsInputData:
    """ A single training/test example for klue semantic textual similarity. """

    guid: str = field()
    source: str = field()
    sentence1: str = field()
    sentence2: str = field()
    label: torch.Tensor = field()

    @staticmethod
    def _tokenize(tokenizer: PreTrainedTokenizer, sentence: list[str], max_length: int = None) -> BatchEncoding:
        """ Tokenizes the input sentences using the provided tokenizer. """
        return tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length if max_length is None else max_length,
            return_tensors='pt'
        )

    @staticmethod
    def _parse(data, i):
        return {k: data[k][i] for k in data.keys()}

    @classmethod
    def from_dict(cls, data: list[dict]) -> Iterable["KlueStsInputData"]:
        """ Converts a dictionary to a KlueStsInputData instance. """
        for i, d in enumerate(data):
            yield cls(
                guid=d['guid'],
                source=d['source'],
                sentence1=d['sentence1'],
                sentence2=d['sentence2'],
                label=d['labels']['real-label'],
            )


class KlueStsDataSet(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, train: bool = False, max_seq_length: int = None):
        self.tokenizer = tokenizer
        self.train = train
        self.max_seq_length = max_seq_length
        self.data: list[KlueStsInputData] = []
        self.mask_id = self.tokenizer.mask_token_id

    def fetch(self, data: list[dict]):
        """ Fetches the data from the dataset. """
        self.data = list(KlueStsInputData.from_dict(data))
        return self

    def _collect_fn(self, data):
        """ Collects the data into a batch. """
        sentence = [d[0] for d in data]
        label = [[d[1]] for d in data]
        sentence = self.tokenizer(sentence, padding='longest', return_tensors='pt').to(DEVICE)
        label = torch.tensor(label, dtype=torch.float32).to(DEVICE)
        if self.train:
            sentence = self._dynamic_masking(sentence)
        return sentence, label

    def _dynamic_masking(self, tokens):
        rand = torch.rand_like(tokens['input_ids'], dtype=torch.float16)
        tokens['input_ids'] = torch.where(rand > 0.9, self.mask_id, tokens['input_ids']) * tokens['attention_mask']
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """ Returns a single data point from the dataset.
        randomly switch the order of sentence1 and sentence2
        Args:
            index (int): Index of the data point to retrieve.
        """
        if self.train and (torch.rand(1) > 0.5):
            sentence = self.data[index].sentence2 + '[SEP]' + self.data[index].sentence1
        else:
            sentence = self.data[index].sentence1 + '[SEP]' + self.data[index].sentence2
        return sentence, self.data[index].label

    def get_dataloader(self, batch_size: int, shuffle: bool, **kwargs):
        # Create a DataLoader from the dataset
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collect_fn, **kwargs)
