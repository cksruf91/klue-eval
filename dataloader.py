from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from dataset import KlueStsDataset
from utils import read_json, read_parquet


class KlueStsDataLoaderFetcher(object):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.sep = self.tokenizer.special_tokens_map["sep_token"]
        self.max_length = max_length if max_length else self.tokenizer.model_max_length

    def collate_fn(self, input_examples):
        """KlueStsFeature padded all input up to max_seq_length"""
        pass

    def get_dataloader(self, file_path: str, batch_size: int, **kwargs):
        data = read_json(file_path) if file_path.endswith(".json") else read_parquet(file_path)
        dataset = KlueStsDataset(data, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
