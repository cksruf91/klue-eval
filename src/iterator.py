from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, BatchEncoding

from utils import read_parquet, read_json

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class KlueStsInputData:
    """ A single training/test example for klue semantic textual similarity. """

    guid: str = field()
    source: str = field()
    sentence1: dict = field()
    sentence2: dict = field()
    label: torch.Tensor = field()

    _input_order = ['input_ids', 'attention_mask', 'token_type_ids']

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
    def from_dict(cls, data: list[dict], tokenizer: PreTrainedTokenizer, max_length: int = None) -> Iterable[
        "KlueStsInputData"]:
        """ Converts a dictionary to a KlueStsInputData instance. """
        sentence1_tokens = cls._tokenize(tokenizer, [d['sentence1'] for d in data], max_length).to(DEVICE)
        sentence2_tokens = cls._tokenize(tokenizer, [d['sentence2'] for d in data], max_length).to(DEVICE)
        for i, d in enumerate(data):
            label = 1 if d['labels']['binary-label'] != 0 else -1
            yield cls(
                guid=d['guid'],
                source=d['source'],
                sentence1=cls._parse(sentence1_tokens, i),
                sentence2=cls._parse(sentence2_tokens, i),
                label=torch.tensor(label).to(DEVICE),
            )


class KlueStsDataSet(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data: list[KlueStsInputData] = []

    def fetch(self, file_path: str):
        """ Fetches the data from the dataset. """
        data = read_json(file_path) if file_path.endswith(".json") else read_parquet(file_path)
        self.data = list(KlueStsInputData.from_dict(data, self.tokenizer, self.max_seq_length))
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return (
            self.data[index].sentence1,
            self.data[index].sentence2,
            self.data[index].label,
        )

    def get_dataloader(self, batch_size: int, shuffle: bool, **kwargs):
        # Create a DataLoader from the dataset
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
